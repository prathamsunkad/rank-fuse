"""B4: a Matryoshka adapter served with S-LoRA's mixed-rank decode kernel (bgmv), two ways.

S-LoRA (Sheng et al., 2023) batches requests whose adapters have different ranks with one CUDA
kernel pair: `dispatch_bgmv` computes, per token, a matrix-vector product with that token's
adapter rows, which it finds through an indirection (`start`, `rank`, `loc` into a unified
memory pool). The kernel is vendored from S-LoRA commit c1ddf48 in `slora_ext/` (Apache-2.0,
changes listed in each file's header: a per-thread bounds guard on the shrink's loads, launches on
PyTorch's current stream, and the narrow/wide sizes RankFuse uses). Its contract, as S-LoRA's
`lora_unordered_batch_infer.py` calls it:

    shrink  v[T, R] = 0;  v[b, j] = A_e[j] . x_b  for j < rank_e   (e = entry of token b)
    expand  y[T, h_out] += scale_e * sum_{k < rank_e} v[b, k] * B_e[k]

with R (the delta width, S-LoRA's max_lora_dim) one of 8/16/32/64, and h_in, h_out among the
compiled hidden sizes (WIDE_SIZES). Each entry's A rows are pool rows of length h_in; its B is
stored as B^T [h_out, rank] flattened into `rank` pool rows of length h_out, read in 16-byte
vectors of 8 rank elements, so the expand needs ranks that are multiples of 8 and its layout
depends on the rank. The kernel does one CUDA-core dot product per (token, rank row), with no
reuse of a weight row across tokens and no tensor cores.

Two ways to serve a Matryoshka adapter with it:

    b4_slora_copies  How S-LoRA would serve it today: every rank tier is its own adapter at its
                     true rank, each with its own A rows and B^T rows in the pool. Ranks below 8
                     are stored padded to 8 (zero A rows and zero B columns), since the expand
                     needs multiples of 8: tiers 1..64 take 8+8+8+8+16+32+64 = 144 pool rows per
                     matrix, against RankFuse's 64.
    b4_slora_shared  The reviewer's hypothetical: "MBGMV with every request pointing at the same
                     weights". A_t and B^T [h_out, r_max] are stored once (64 rows each). One
                     entry per rank 0..r_max, all at the same location; the shrink runs at each
                     token's true rank (any rank works there, 1/2/4 included), and the expand is
                     called with a second rank array that is r_max for every entry, because B^T's
                     layout is fixed by r_max. The expand then also reads v[b, k] for k >= the
                     token's rank, which is zero (the shrink did not write it), so the result is
                     exact; like rf1_fused, the expand does no rank skipping.

Rank 0 (no LoRA) maps to an entry of rank 0: the shrink skips every row, the expand adds 0, so
the output is untouched. Per-rank scales go in the entry's lora_scale (S-LoRA's a_scaling), in
the input dtype. Weights, entries and per-step metadata are built in `prepare`, outside timing,
so run_sweep's CUDA-graph timing captures only the two kernel launches and the buffer zeroing.

S-LoRA itself uses bgmv only for decode; for prompts (>= 200 tokens) it switches to a separate
Triton kernel. Running bgmv on prefill-sized batches is therefore not how S-LoRA serves prompts,
and results for it are labelled as such.

Not registered on `import rankfuse`: the kernel supports only fp16/bf16, the WIDE_SIZES hidden
sizes and r_max <= 64 (a multiple of 8), so the shape-generic tests in tests/test_correctness.py
cannot run it. Call `enable()` (bench/run_sweep.py and bench/ncu_driver.py do when a b4_* name is
requested; tests/test_slora_bgmv.py does for its own tests) to add both names to the registry.

The extension is JIT-built with torch.utils.cpp_extension on first use (about a minute on an H200 node; cached in
$TORCH_EXTENSIONS_DIR). It needs an nvcc whose major version matches torch.version.cuda; if none
is on CUDA_HOME, the toolkit pip installs with PyTorch (site-packages/nvidia/cu<major>) is used.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from rankfuse.interface import REGISTRY, MatryoshkaLoRA, register
from rankfuse.workloads import default_tiers

NARROW_SIZES = (8, 16, 32, 64)  # delta widths (ranks) compiled into slora_ext
WIDE_SIZES = (128, 512, 896, 1024, 3584, 4096, 4864, 14336, 18944)  # hidden sizes compiled in
NAMES = ("b4_slora_copies", "b4_slora_shared")
MAX_TOKENS = 65535  # the kernels put tokens on grid dimension y

_SRC = Path(__file__).resolve().parent / "slora_ext"
_ext_module = None


def _pip_cuda_home() -> Path | None:
    """The CUDA toolkit from pip (nvidia-cuda-nvcc), matching torch's CUDA major version."""
    major = (torch.version.cuda or "").split(".")[0]
    for p in sys.path:
        home = Path(p) / "nvidia" / f"cu{major}"
        if (home / "bin" / "nvcc").exists():
            return home
    return None


def _nvcc_major(home: str | Path | None) -> str | None:
    if home is None or not (Path(home) / "bin" / "nvcc").exists():
        return None
    import subprocess

    out = subprocess.run([str(Path(home) / "bin" / "nvcc"), "--version"], capture_output=True, text=True).stdout
    for tok in out.replace(",", " ").split():
        if tok[:1].isdigit() and "." in tok:
            return tok.split(".")[0]
    return None


def _ext():
    """Build (first call) or load the vendored S-LoRA extension."""
    global _ext_module
    if _ext_module is not None:
        return _ext_module
    from torch.utils import cpp_extension

    build_root = Path(os.environ.get("TORCH_EXTENSIONS_DIR") or cpp_extension._get_build_directory("", False))
    extra_ldflags = []
    torch_major = (torch.version.cuda or "").split(".")[0]
    if _nvcc_major(cpp_extension.CUDA_HOME) != torch_major:
        home = _pip_cuda_home()
        if home is None:
            raise RuntimeError(f"no nvcc for CUDA {torch.version.cuda} (CUDA_HOME={cpp_extension.CUDA_HOME})")
        cpp_extension.CUDA_HOME = os.environ["CUDA_HOME"] = str(home)
    home = Path(cpp_extension.CUDA_HOME)
    if not any((home / d / "libcudart.so").exists() for d in ("lib64", "lib")):
        # The pip toolkit ships only libcudart.so.<major>; give the linker's -lcudart a name to find.
        link_dir = build_root / "_cudart_link"
        link_dir.mkdir(parents=True, exist_ok=True)
        target = next((home / "lib").glob("libcudart.so.*"))
        link = link_dir / "libcudart.so"
        if not link.exists():
            link.symlink_to(target)
        extra_ldflags.append(f"-L{link_dir}")

    build_dir = build_root / "rankfuse_slora_bgmv"
    build_dir.mkdir(parents=True, exist_ok=True)
    _ext_module = cpp_extension.load(
        name="rankfuse_slora_bgmv",
        sources=[str(_SRC / "lora_ops.cc"), str(_SRC / "bgmv" / "bgmv_all.cu")],
        extra_cflags=["-O3"],
        # S-LoRA's setup.py removes these PyTorch defaults: the kernel relies on __half/__nv_bfloat16
        # conversions and operators. Undefined here only for this extension.
        extra_cuda_cflags=["-O3", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                           "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__"],
        extra_ldflags=extra_ldflags,
        build_directory=str(build_dir),
        verbose=bool(os.environ.get("RF_VERBOSE_BUILD")),
    )
    return _ext_module


def _check_supported(lora: MatryoshkaLoRA) -> int:
    """Delta width R for this adapter; raises if the compiled kernel cannot run it."""
    if lora.A_t.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"S-LoRA bgmv supports fp16/bf16 only, got {lora.A_t.dtype}")
    for side, h in (("h_in", lora.h_in), ("h_out", lora.h_out)):
        if h not in WIDE_SIZES:
            raise ValueError(f"{side}={h} is not compiled into slora_ext (WIDE_SIZES={WIDE_SIZES})")
    if lora.r_max > NARROW_SIZES[-1] or lora.r_max % 8:
        raise ValueError(f"r_max={lora.r_max}: needs a multiple of 8 up to {NARROW_SIZES[-1]}")
    return next(n for n in NARROW_SIZES if n >= lora.r_max)


def _scale(lora: MatryoshkaLoRA, r: int) -> float:
    return (lora.scale_per_rank or {}).get(r, 1.0) if r > 0 else 0.0


class Pool:
    """S-LoRA's adapter metadata for one adapter's entries, plus the A and B pools.

    A [rows, 1, h_in], B [rows, 1, h_out] (3-D, as S-LoRA's key/value buffers); entry e owns pool
    rows loc[start[e] : start[e] + rank_e]. A and B share start/loc, as in S-LoRA.
    """

    def __init__(self, A, B, start, shrink_ranks, expand_ranks, scales, entry_of_rank, width, dtype, dev):
        i64 = dict(dtype=torch.int64, device=dev)
        self.A, self.B, self.width = A, B, width
        self.start = torch.tensor(start, **i64)
        # S-LoRA stores 4 * rank (one rank per q, k, v, o); we use module slot qkvo = 0.
        self.shrink_ranks = torch.tensor([4 * r for r in shrink_ranks], **i64)
        self.expand_ranks = torch.tensor([4 * r for r in expand_ranks], **i64)
        # Identity page table, padded: the expand reads loc[start + bin] before checking the rank,
        # and bin can be one past the entry's last row.
        self.loc = torch.arange(A.shape[0] + 8, **i64)
        self.loc[A.shape[0]:] = 0
        self.scales = torch.tensor(scales, dtype=dtype, device=dev)
        self.entry_of_rank = torch.tensor(entry_of_rank, **i64)

    @staticmethod
    def copies(lora: MatryoshkaLoRA, ranks: list[int]) -> "Pool":
        """b4_slora_copies: one adapter per rank, at its rank rounded up to a multiple of 8."""
        width = _check_supported(lora)
        dev, dt = lora.A_t.device, lora.A_t.dtype
        ranks = sorted(set(ranks))
        stored = [-(-r // 8) * 8 for r in ranks]
        A = torch.zeros(sum(stored), 1, lora.h_in, device=dev, dtype=dt)
        B = torch.zeros(sum(stored), 1, lora.h_out, device=dev, dtype=dt)
        start, row = [], 0
        for r, s in zip(ranks, stored):
            start.append(row)
            A[row:row + r, 0] = lora.A_t[:r]
            bt = torch.zeros(lora.h_out, s, device=dev, dtype=dt)
            bt[:, :r] = lora.B[:r].T
            B[row:row + s, 0] = bt.reshape(s, lora.h_out)  # B^T [h_out, s] flattened into s rows
            row += s
        entry_of_rank = [len(ranks)] * (lora.r_max + 1)  # rank 0 and unknown ranks: the rank-0 entry
        for e, r in enumerate(ranks):
            entry_of_rank[r] = e
        return Pool(A, B, start + [0], stored + [0], stored + [0],
                    [_scale(lora, r) for r in ranks] + [0.0], entry_of_rank, width, dt, dev)

    @staticmethod
    def shared(lora: MatryoshkaLoRA) -> "Pool":
        """b4_slora_shared: A_t and B^T stored once; entry r = rank r, all at pool row 0."""
        width = _check_supported(lora)
        dev, dt = lora.A_t.device, lora.A_t.dtype
        R = lora.r_max
        A = lora.A_t.reshape(R, 1, lora.h_in).contiguous()
        B = lora.B.T.contiguous().reshape(R, 1, lora.h_out)  # B^T [h_out, r_max] flattened
        ranks = list(range(R + 1))
        return Pool(A, B, [0] * (R + 1), ranks, [0] + [R] * R,
                    [_scale(lora, r) for r in ranks], ranks, width, dt, dev)


def _pool_for(lora: MatryoshkaLoRA, mode: str, present: list[int]) -> Pool:
    """The adapter's pool, built once per (weights, scales) and, for copies, set of ranks."""
    key = (mode, lora.A_t.data_ptr(), lora.B.data_ptr(), tuple(sorted((lora.scale_per_rank or {}).items())))
    cache = lora.__dict__.setdefault("_b4_pools", {})
    cached = cache.get(mode)
    if mode == "shared":
        if cached is None or cached[0] != key:
            cached = cache[mode] = (key, Pool.shared(lora), None)
        return cached[1]
    if cached is None or cached[0] != key or not set(present) <= cached[2]:
        ranks = set(default_tiers(lora.r_max)) | set(present)
        if cached is not None and cached[0] == key:
            ranks |= cached[2]
        cached = cache[mode] = (key, Pool.copies(lora, sorted(ranks)), ranks)
    return cached[1]


def _prepare(mode: str):
    def prepare(x: torch.Tensor, lora: MatryoshkaLoRA, token_ranks: torch.Tensor) -> dict:
        """Pool and per-step metadata (token -> entry, zeroed delta buffer), built outside timing."""
        present = [r for r in torch.unique(token_ranks).tolist() if r > 0]
        pool = _pool_for(lora, mode, present)
        _ext()  # build or load the extension here, not inside a timed or captured call
        T = x.shape[0]
        return dict(
            pool=pool,
            entries=pool.entry_of_rank[token_ranks.long()],
            v=torch.zeros(T, pool.width, dtype=x.dtype, device=x.device),
        )

    return prepare


def apply(x: torch.Tensor, pool: Pool, entries: torch.Tensor, v: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """out += delta with S-LoRA's shrink and expand, as S-LoRA's LoRA layer calls them."""
    ext = _ext()
    v.zero_()
    ext.dispatch_bgmv(v, x, pool.A, pool.start, pool.shrink_ranks, pool.loc, entries, 0, pool.scales)
    ext.dispatch_bgmv(out, v, pool.B, pool.start, pool.expand_ranks, pool.loc, entries, 0, pool.scales)
    return out


def _impl(mode: str):
    prepare = _prepare(mode)

    def b4(
        x: torch.Tensor,
        lora: MatryoshkaLoRA,
        token_ranks: torch.Tensor,
        pool: Pool | None = None,
        entries: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        add: bool = False,
    ) -> torch.Tensor:
        """B4. Without pool/entries/v, builds them here (one host sync for the ranks present)."""
        if x.shape[0] > MAX_TOKENS:
            raise ValueError(f"{x.shape[0]} tokens: the bgmv grid holds at most {MAX_TOKENS}")
        if pool is None:
            meta = prepare(x, lora, token_ranks)
            pool, entries, v = meta["pool"], meta["entries"], meta["v"]
        if out is None:
            if add:
                raise ValueError("add=True needs an existing out to add into")
            out = torch.zeros((x.shape[0], lora.h_out), dtype=x.dtype, device=x.device)
        elif not add:
            out.zero_()
        if x.shape[0] == 0:
            return out
        return apply(x.contiguous(), pool, entries, v, out)

    return b4, prepare


def enable() -> None:
    """Register b4_slora_copies and b4_slora_shared (idempotent)."""
    for name in NAMES:
        if name not in REGISTRY:
            fn, prepare = _impl(name.removeprefix("b4_slora_"))
            register(name, requires_cuda=True, dtypes=(torch.float16, torch.bfloat16), prepare=prepare)(fn)


def disable() -> None:
    """Remove both names from the registry again."""
    for name in NAMES:
        REGISTRY.pop(name, None)


def enable_if_requested(names) -> None:
    """Call enable() if any of `names` (a list of implementation names, or "all") is a b4_* name."""
    if not isinstance(names, str) and any(str(n).startswith("b4_") for n in names):
        enable()
