"""Rank skipping under CUDA graphs for the Stage 1 vLLM plugin (one Matryoshka adapter).

Why. The Stage 1 op (ops.py) runs RF1: every token block pads to r_limit, and under CUDA-graph
capture r_limit is forced to r_max, because a graph is replayed for later steps whatever their
ranks. So with graphs on (every decode step up to the largest captured size), RankFuse computes
rank r_max for every token, exactly the work of stock vLLM padded to max_lora_rank.

What this mode runs instead. Tony's RANK_SKIP split-K kernels (rankfuse_fused.py, section 8,
`_splitk_shrink_kernel` / `_splitk_expand_kernel` with RANK_SKIP): each BM-token block READS ITS
TOKENS' RANKS FROM DEVICE MEMORY at run time and branches to a part compiled for the smallest
bucket (16 / 32 / 64 / 128 / 256 rank rows) that covers the block's maximum rank; a block whose
ranks are all 0 does nothing. The rank width therefore follows the data at every graph replay;
r_limit only sets the compiled R_PAD (the workspace row width) and the A/B row mask.

    rf2_index    (default of this mode) the blocks are formed over a rank-SORTED token order
                 (INDEXED): positions [0, n) of a static int32 order buffer are the step's tokens
                 sorted by rank, positions [n, max) stay the identity, so CUDA-graph padding
                 tokens (rank 0) fall in their own blocks and are skipped. The order is computed
                 on the host with numpy from the per-token ranks the plugin already has there
                 (RankFusePunicaWrapper._update_rank_metadata), and copied with the ranks: no
                 GPU sync, nothing allocated per step on the device.
    rf2_segment  the same kernels over the batch order (no index). In decode, a block holds 16
                 different requests, so it runs at the largest of their ranks; useful as the
                 control that shows what the sort buys.

Graph safety: the kernels read token_ranks and the order from fixed-address static buffers that
update_metadata refills in place (also during capture, through the same path); the workspace is
the plugin's shared static one (reserve_workspace); launch settings depend only on (T, R_PAD),
which are fixed per captured graph. Not graph-safe and not used: rf2_ranges / rf2_bucketed
(host-side planning per call, shapes depend on the rank mix) and rf1_ma (its per-tile r_limit
only masks loads, the tl.dot stays R_PAD wide, and with one adapter there is one group at
r_max; during capture it always collapses to one group at r_max).

Activation (no change to the existing plugin files):

    RANKFUSE_VLLM=1 RANKFUSE_VLLM_KERNEL=rf2_index \
        vllm serve ... --worker-extension-cls rankfuse.integration.rank_skip.RankSkipWorkerExtension

vLLM imports the worker extension class in the worker process right after
load_general_plugins() and before the model (and so the Punica wrapper) is built, so importing
this module patches RankFusePunicaWrapper in time. RANKFUSE_VLLM_KERNEL unset or "rf1" leaves
the Stage 1 behaviour untouched. The mode also joins LoRAConfig.compute_hash, so compiled graphs
of the two modes never mix (they call different custom ops).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import torch

from rankfuse.kernels import rankfuse_fused as rf

MODES = ("rf1", "rf2_index", "rf2_segment")


def kernel_mode() -> str:
    mode = os.environ.get("RANKFUSE_VLLM_KERNEL", "rf1").strip() or "rf1"
    if mode not in MODES:
        raise ValueError(f"RANKFUSE_VLLM_KERNEL must be one of {MODES}, got {mode!r}")
    return mode


class RankSkipLauncher(rf.RF1Launcher):
    """RF1Launcher with RANK_SKIP (and INDEXED for rf2_index): same checks, plans, workspace and
    direct launches; only the plan's rank_skip / indexed flags and the index pointer differ."""

    def __init__(self, A_t, B, slice_sizes=None, *, max_tokens: int, scale_table=None, indexed: bool = True):
        super().__init__(A_t, B, slice_sizes, max_tokens=max_tokens, scale_table=scale_table)
        self.indexed = indexed

    def __call__(self, x, token_ranks, order, out, *, r_limit: int, add: bool = True) -> torch.Tensor:
        num_tokens = x.shape[0]
        if num_tokens == 0:
            return out
        if r_limit <= 0:
            if not add:
                out.zero_()
            return out
        out_stride = out.stride(0)
        r_pad = rf._r_pad(r_limit)
        pkey = (num_tokens, r_pad, add, out_stride, rf._settings_key())
        plan = self._plans.get(pkey)
        if plan is None:
            if num_tokens > self.max_tokens or r_limit > self.r_max:
                raise ValueError(f"T={num_tokens} / r_limit={r_limit} above the launcher's max_tokens / r_max")
            plan = self._plans[pkey] = rf._splitk_plan(
                num_tokens, self.h_in, self.n_slices, self.r_max, self.n_max, self.sizes, r_pad, self.elem_size,
                add, self.scale_table is not None, out_stride, self.device_index, None,
                rank_skip=True, rank_probe=0, indexed=self.indexed)
            if plan.ws_numel > self.ws.numel():
                raise RuntimeError("split-K workspace too small (reserve_workspace was not sized for this call)")
        scales = out if self.scale_table is None else self.scale_table
        index = order if self.indexed else token_ranks
        rf._launch(plan, x, self.A_t, self.B, token_ranks, index, out, self.ws, r_limit, 0, scales, self.offsets,
                   self.widths)
        return out


# ------------------------------------------------------------------------------- custom op

_LIB = torch.library.Library("rankfuse", "FRAGMENT")
_LAUNCHERS: dict[tuple, RankSkipLauncher] = {}


def launcher_for(A_t, B, slice_sizes, max_tokens, scale_table, indexed) -> RankSkipLauncher:
    key = (A_t.data_ptr(), B.data_ptr(), tuple(slice_sizes), max_tokens,
           None if scale_table is None else scale_table.data_ptr(), indexed)
    launcher = _LAUNCHERS.get(key)
    if launcher is None:
        launcher = _LAUNCHERS[key] = RankSkipLauncher(A_t, B, slice_sizes, max_tokens=max_tokens,
                                                      scale_table=scale_table, indexed=indexed)
    return launcher


def _lora_linear_rs(y, x, A_t, B, token_ranks, order, r_limit_cpu, slice_sizes, scale_table, add, max_tokens,
                    indexed) -> None:
    if x.size(0) == 0:
        return
    r_limit = int(r_limit_cpu[0])
    if r_limit > 0 and torch.cuda.is_current_stream_capturing():
        r_limit = A_t.shape[1]   # compiled width covers every rank; the blocks pick theirs at replay
    if not x.is_contiguous():
        x = x.contiguous()
    launcher_for(A_t, B, slice_sizes, max_tokens, scale_table, indexed)(x, token_ranks, order, y, r_limit=r_limit,
                                                                        add=add)


def _lora_linear_rs_fake(y, x, A_t, B, token_ranks, order, r_limit_cpu, slice_sizes, scale_table, add, max_tokens,
                         indexed) -> None:
    return None


_LIB.define(
    "lora_linear_rs(Tensor(a!) y, Tensor x, Tensor A_t, Tensor B, Tensor token_ranks, Tensor order, "
    "Tensor r_limit_cpu, int[] slice_sizes, Tensor? scale_table, bool add, int max_tokens, bool indexed) -> ()"
)
_LIB.impl("lora_linear_rs", _lora_linear_rs, "CUDA")
_LIB._register_fake("lora_linear_rs", _lora_linear_rs_fake)
lora_linear_rs = torch.ops.rankfuse.lora_linear_rs


# ----------------------------------------------------------------------- host-side ordering

def fill_order(order_host: np.ndarray, ranks: np.ndarray, n: int, prev_n: int) -> int:
    """Write the rank-sorted order of tokens [0, n) into order_host[:n] and restore the identity
    on [n, prev_n). Returns how many entries must be copied to the device (max(n, prev_n))."""
    if n:
        order_host[:n] = np.argsort(ranks[:n], kind="stable")
    if prev_n > n:
        order_host[n:prev_n] = np.arange(n, prev_n, dtype=order_host.dtype)
    return max(n, prev_n)


# ------------------------------------------------------------------------- the patches

_PATCHED = False


def install() -> bool:
    """Patch RankFusePunicaWrapper for the selected mode (idempotent). False for rf1 / plugin off."""
    global _PATCHED
    if os.environ.get("RANKFUSE_VLLM", "0") != "1":
        return False
    mode = kernel_mode()
    if mode == "rf1":
        return False
    if _PATCHED:
        return True
    _patch(mode)
    _PATCHED = True
    return True


def _patch(mode: str) -> None:
    from vllm.config.lora import LoRAConfig
    from vllm.logger import init_logger

    from rankfuse.integration import punica as P

    logger = init_logger("vllm.rankfuse_rank_skip")
    indexed = mode == "rf2_index"
    W = P.RankFusePunicaWrapper

    orig_hash = LoRAConfig.compute_hash

    def compute_hash(self) -> str:
        return hashlib.sha256(f"{orig_hash(self)}|rankfuse-kernel={mode}".encode()).hexdigest()

    LoRAConfig.compute_hash = compute_hash

    orig_init = W.__init__

    def __init__(self, max_num_batched_tokens, max_batches, device, **kwargs):
        orig_init(self, max_num_batched_tokens, max_batches, device, **kwargs)
        self._rs_indexed = indexed
        self._order_host = np.arange(max_num_batched_tokens, dtype=np.int32)
        self._token_order = torch.arange(max_num_batched_tokens, dtype=torch.int32, device=device)
        self._order_filled = 0

    def _update_rank_metadata(self, mapping, lora_index_to_id) -> None:
        # The Stage 1 body (punica.py), plus the rank-sorted order for rf2_index.
        ids = np.asarray(mapping.index_mapping, dtype=np.int64)
        n = ids.shape[0]
        ranks = getattr(mapping, "index_rank_mapping", ())
        if len(ranks) == n:
            ranks = np.asarray(ranks, dtype=np.int64)
        else:
            ranks = np.where(ids > 0, P.FULL_RANK, 0)
        in_slot = [i for i in lora_index_to_id if i is not None]
        ranks = np.where(np.isin(ids, in_slot) if in_slot else False, np.minimum(ranks, self.max_lora_rank), 0)
        r_limit = int(ranks.max()) if n else 0
        hi = max(n, self._ranks_filled)
        buf = np.zeros(hi, dtype=np.int32)
        buf[:n] = ranks
        if hi:
            self._token_lora_ranks[:hi].copy_(torch.from_numpy(buf).pin_memory(), non_blocking=True)
        self._ranks_filled = n
        self._r_limit_cpu[0] = r_limit
        if self._rs_indexed:
            k = fill_order(self._order_host, ranks, n, self._order_filled)
            if k:
                self._token_order[:k].copy_(torch.from_numpy(self._order_host[:k].copy()).pin_memory(),
                                            non_blocking=True)
            self._order_filled = n

    def _rankfuse_apply(self, y, x, rankfuse_weights, output_slices, add: bool) -> None:
        A_t, B = rankfuse_weights
        lora_linear_rs(y, x.view(-1, x.shape[-1]), A_t, B, self._token_lora_ranks, self._token_order,
                       self._r_limit_cpu, list(output_slices), self.scale_table, add, self.max_num_batched_tokens,
                       self._rs_indexed)

    W.__init__ = __init__
    W._update_rank_metadata = _update_rank_metadata
    W._rankfuse_apply = _rankfuse_apply
    logger.info("RankFuse rank skipping under CUDA graphs: kernel %s (RANKFUSE_VLLM_KERNEL)", mode)


class RankSkipWorkerExtension:
    """Empty worker extension: importing this module (which vLLM does in the worker process for
    --worker-extension-cls) installs the patches. It adds no attributes to the worker."""


install()
