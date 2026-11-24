"""Stage 2 contract: many Matryoshka adapters, each token carrying (adapter, rank).

Kept apart from `rankfuse.interface.REGISTRY` (one adapter, `impl(x, lora, token_ranks)`), whose
every entry tests/test_correctness.py runs: multi-adapter implementations take a different call
and register into MULTI_REGISTRY here.

Weights: a page pool plus a page table
--------------------------------------
Adapter a's rank rows are stored in pages of PAGE = 16 rows: page k holds rank rows 16k..16k+15 of
every matrix of the adapter, and page_table[a, k] is the frame that holds it (-1: not resident).
Frames need not be contiguous or in order. For merged projections (q/k/v, gate/up: S output slices
sharing x) one frame holds all S matrices' rows at fixed per-slice offsets, so the whole-model
layout (every layer's matrices in one frame, at an offset per matrix) is the same description:

    A row j of rank row 16k + j, slice s:  A_pool.flat[page_table[a, k] * frame_stride_a + a_off[s] + j * h_in]
    B row j of rank row 16k + j, slice s:  B_pool.flat[page_table[a, k] * frame_stride_b + b_off[s] + j * b_ld[s]]

followed by h_in (A) or slice_sizes[s] (B) contiguous elements. Two layouts are built here:

    PagedWeights.dense(A, B)   stacked A [N, S, r_max, h_in], B [N, S, r_max, n_max] used in place:
                               frame = a * S * r_max/16 + k (an identity page table, stride 16 rows)
    PagedWeights.paged(A, B)   per-layer pools A [F, S, 16, h_in], B [F, S, 16, n_max] with the
                               pages copied into frames in a contiguous or a shuffled order

Every rank row read goes through the page table, so swapping the dense layout for a whole-model
pool later changes only (pool pointer, frame stride, offsets).

Per-step metadata (built outside the timed call)
------------------------------------------------
`build_meta` turns the step's per-token adapter and rank arrays (host data the scheduler already
has) into device tensors: the token order (sorted by adapter, as vLLM's LoRAKernelMeta sorts),
group starts / lengths / adapter ids / rank limits, and a tile table (adapter, first position,
length, rank limit) of at most BM tokens per tile, every tile inside one group. Tokens of rank 0
(or adapter < 0) go to a final "no LoRA" group of limit 0. With `capacity` the tile table is padded
with empty tiles to a fixed size, so the call's grid stays fixed across steps and a captured CUDA
graph can be replayed after `MultiMeta.load_` copies a new step's metadata into the same tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from rankfuse.workloads import DISTRIBUTIONS, DTYPES, default_tiers

PAGE = 16  # rank rows per page


# ------------------------------------------------------------------------------------------------
# Weights
# ------------------------------------------------------------------------------------------------


@dataclass
class PagedWeights:
    A_pool: torch.Tensor  # any shape, contiguous; addressed flat
    B_pool: torch.Tensor
    page_table: torch.Tensor  # [num_adapters, r_max // PAGE] int32, frame index or -1
    frame_stride_a: int  # elements between frames
    frame_stride_b: int
    a_off: tuple[int, ...]  # per slice, elements from the frame start
    b_off: tuple[int, ...]
    b_ld: tuple[int, ...]  # per slice, elements between B rows
    h_in: int
    slice_sizes: tuple[int, ...]  # output columns per slice
    r_max: int
    # [num_adapters, r_max + 1] fp32 (scale of adapter a at rank r) or None (scale 1)
    scale_table: torch.Tensor | None = None
    # Dense copies [N, S, r_max, h_in] / [N, S, r_max, n_max] for baselines and references (the
    # same storage as the pools for the dense layout; separate tensors for a paged layout).
    A_dense: torch.Tensor | None = None
    B_dense: torch.Tensor | None = None
    _cache: dict = field(default_factory=dict, repr=False)  # per-weights tables built by kernels

    @property
    def num_adapters(self) -> int:
        return self.page_table.shape[0]

    @property
    def n_slices(self) -> int:
        return len(self.slice_sizes)

    @property
    def h_out(self) -> int:
        return sum(self.slice_sizes)

    @property
    def dtype(self) -> torch.dtype:
        return self.A_pool.dtype

    @property
    def device(self) -> torch.device:
        return self.A_pool.device

    def slice_table(self) -> torch.Tensor:
        """[S, 5] int64 on the device: out_col0, n_cols, a_off, b_off, b_ld per slice (cached)."""
        if "slice_table" not in self._cache:
            rows = [
                [sum(self.slice_sizes[:s]), self.slice_sizes[s], self.a_off[s], self.b_off[s], self.b_ld[s]]
                for s in range(self.n_slices)
            ]
            self._cache["slice_table"] = torch.tensor(rows, dtype=torch.int64, device=self.device)
        return self._cache["slice_table"]

    def aligned(self) -> bool:
        """Every row and column offset a kernel computes is a multiple of 16 elements."""
        vals = [self.h_in, self.frame_stride_a, self.frame_stride_b, *self.a_off, *self.b_off, *self.b_ld]
        vals += [sum(self.slice_sizes[:s]) for s in range(self.n_slices)] + list(self.slice_sizes)
        return all(v % 16 == 0 for v in vals)

    # -------------------------------------------------------------------------------------------
    @staticmethod
    def _stack(A: torch.Tensor, B: torch.Tensor, slice_sizes):
        if A.dim() == 3:
            A = A.unsqueeze(1)
        if B.dim() == 3:
            B = B.unsqueeze(1)
        N, S, r_max, h_in = A.shape
        if B.shape[:3] != (N, S, r_max):
            raise ValueError(f"B {tuple(B.shape)} does not match A {tuple(A.shape)}")
        if r_max % PAGE:
            raise ValueError(f"r_max must be a multiple of {PAGE}, got {r_max}")
        n_max = B.shape[3]
        sizes = tuple(int(s) for s in slice_sizes) if slice_sizes is not None else (n_max,) * S
        if len(sizes) != S or not all(0 < s <= n_max for s in sizes):
            raise ValueError(f"slice_sizes must give {S} widths in [1, {n_max}], got {sizes}")
        return A.contiguous(), B.contiguous(), sizes

    @staticmethod
    def _scales(scale_per_rank, N: int, r_max: int, device) -> torch.Tensor | None:
        """dict rank -> scale (shared by all adapters), a list of such dicts, or a tensor."""
        if scale_per_rank is None:
            return None
        if isinstance(scale_per_rank, torch.Tensor):
            t = scale_per_rank.float()
            return (t.expand(N, r_max + 1) if t.dim() == 1 else t).contiguous().to(device)
        dicts = scale_per_rank if isinstance(scale_per_rank, (list, tuple)) else [scale_per_rank] * N
        t = torch.zeros(N, r_max + 1, dtype=torch.float32)
        for a, d in enumerate(dicts):
            for r, s in d.items():
                t[a, r] = s
        return t.to(device)

    @classmethod
    def dense(cls, A, B, slice_sizes=None, scale_per_rank=None) -> "PagedWeights":
        """Stacked A [N, (S,) r_max, h_in] and B [N, (S,) r_max, n_max] behind an identity page table."""
        A, B, sizes = cls._stack(A, B, slice_sizes)
        N, S, r_max, h_in = A.shape
        n_max = B.shape[3]
        npages = r_max // PAGE
        pt = torch.arange(N * S * npages, dtype=torch.int32, device=A.device).view(N, S * npages)[:, :npages]
        return cls(
            A_pool=A, B_pool=B, page_table=pt.contiguous(),
            frame_stride_a=PAGE * h_in, frame_stride_b=PAGE * n_max,
            a_off=tuple(s * r_max * h_in for s in range(S)), b_off=tuple(s * r_max * n_max for s in range(S)),
            b_ld=(n_max,) * S, h_in=h_in, slice_sizes=sizes, r_max=r_max,
            scale_table=cls._scales(scale_per_rank, N, r_max, A.device), A_dense=A, B_dense=B,
        )

    @classmethod
    def paged(cls, A, B, slice_sizes=None, scale_per_rank=None, order: str = "shuffled", seed: int = 0,
              resident: torch.Tensor | None = None, extra_frames: int = 0) -> "PagedWeights":
        """Per-layer pools A [F, S, 16, h_in] and B [F, S, 16, n_max]; page (a, k) is copied into a
        frame chosen by `order`: "contiguous" (adapter-major) or "shuffled" (a random permutation).
        resident [N, r_max/16] bool marks the pages to store (others get page_table -1); a page an
        adapter does not store must never be read (every token rank <= 16 * resident pages)."""
        A, B, sizes = cls._stack(A, B, slice_sizes)
        N, S, r_max, h_in = A.shape
        n_max = B.shape[3]
        npages = r_max // PAGE
        dev = A.device
        res = torch.ones(N, npages, dtype=torch.bool) if resident is None else resident.cpu().bool()
        pages = torch.nonzero(res)  # [P, 2] (adapter, page), adapter-major
        F = pages.shape[0] + extra_frames
        if order == "shuffled":
            frames = torch.randperm(F, generator=torch.Generator().manual_seed(seed))[: pages.shape[0]]
        elif order == "contiguous":
            frames = torch.arange(pages.shape[0])
        else:
            raise ValueError(f"order must be 'contiguous' or 'shuffled', got {order!r}")
        pt = torch.full((N, npages), -1, dtype=torch.int32)
        pt[pages[:, 0], pages[:, 1]] = frames.to(torch.int32)
        A_pool = torch.zeros(F, S, PAGE, h_in, dtype=A.dtype, device=dev)
        B_pool = torch.zeros(F, S, PAGE, n_max, dtype=B.dtype, device=dev)
        fr = frames.to(dev)
        a_idx, k_idx = pages[:, 0].to(dev), pages[:, 1].to(dev)
        A_pages = A.view(N, S, npages, PAGE, h_in)
        B_pages = B.view(N, S, npages, PAGE, n_max)
        A_pool[fr] = A_pages[a_idx, :, k_idx]
        B_pool[fr] = B_pages[a_idx, :, k_idx]
        return cls(
            A_pool=A_pool, B_pool=B_pool, page_table=pt.to(dev),
            frame_stride_a=S * PAGE * h_in, frame_stride_b=S * PAGE * n_max,
            a_off=tuple(s * PAGE * h_in for s in range(S)), b_off=tuple(s * PAGE * n_max for s in range(S)),
            b_ld=(n_max,) * S, h_in=h_in, slice_sizes=sizes, r_max=r_max,
            scale_table=cls._scales(scale_per_rank, N, r_max, dev), A_dense=A, B_dense=B,
        )

    def gather_dense(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Rebuild [N, S, r_max, h_in] / [N, S, r_max, n_max] from the pools through the page table
        (non-resident pages read as zero). Used by tests to check the page table itself."""
        N, npages, S, h_in = self.num_adapters, self.r_max // PAGE, self.n_slices, self.h_in
        n_max = max(self.b_ld)
        A = torch.zeros(N, S, self.r_max, h_in, dtype=self.dtype, device=self.device)
        B = torch.zeros(N, S, self.r_max, n_max, dtype=self.dtype, device=self.device)
        pt = self.page_table.cpu()
        Af, Bf = self.A_pool.reshape(-1), self.B_pool.reshape(-1)
        for a in range(N):
            for k in range(npages):
                f = int(pt[a, k])
                if f < 0:
                    continue
                for s in range(S):
                    for j in range(PAGE):
                        ao = f * self.frame_stride_a + self.a_off[s] + j * h_in
                        bo = f * self.frame_stride_b + self.b_off[s] + j * self.b_ld[s]
                        A[a, s, k * PAGE + j] = Af[ao:ao + h_in]
                        B[a, s, k * PAGE + j, : self.slice_sizes[s]] = Bf[bo:bo + self.slice_sizes[s]]
        return A, B

    def scale(self, a: int, r: int) -> float:
        return 1.0 if self.scale_table is None else float(self.scale_table[a, r])


# ------------------------------------------------------------------------------------------------
# Per-step metadata
# ------------------------------------------------------------------------------------------------

MODES = ("group", "rank_sorted", "segment", "adapter_bucket")
WIDTH_BUCKETS = 5  # rank-width buckets 16 / 32 / 64 / 128 / 256


@dataclass
class MultiMeta:
    """Device tensors for one step plus the host-side launch parameters derived from them.

    order [T] int32         position -> token index (tokens of a group are consecutive positions)
    group_* [G]             start position, length, adapter id and rank limit of each group
    tiles [cap, 5] int32    (adapter, first position, length, rank limit, first token): the last
                            column is the tile's first token when its tokens are consecutive in
                            the batch (the kernels then skip the gather through `order`), else -1;
                            rows past num_tiles are empty (length 0) and their programs exit at once
    """

    mode: str
    T: int
    bm: int
    r_pad: int  # constexpr rank width of the kernels (>= every tile's limit, power of two >= 16)
    order: torch.Tensor
    group_start: torch.Tensor
    group_len: torch.Tensor
    group_adapter: torch.Tensor
    group_rank_limit: torch.Tensor
    tiles: torch.Tensor
    num_tiles: int
    num_groups: int
    zero_idx: torch.Tensor | None = None  # tokens with no LoRA (bucketed variant)
    extra: dict = field(default_factory=dict)

    @property
    def capacity(self) -> int:
        return self.tiles.shape[0]

    def load_(self, other: "MultiMeta") -> "MultiMeta":
        """Copy another step's metadata into these tensors (same T, capacity, bm and r_pad), so a
        CUDA graph captured with this object replays the new step."""
        if (other.T, other.capacity, other.bm, other.r_pad, other.mode) != (self.T, self.capacity, self.bm, self.r_pad, self.mode):
            raise ValueError("load_ needs metadata built with the same T, capacity, bm, r_pad and mode")
        self.order.copy_(other.order)
        self.tiles.copy_(other.tiles)
        for name in ("group_start", "group_len", "group_adapter", "group_rank_limit"):
            dst, src = getattr(self, name), getattr(other, name)
            dst.zero_()
            dst[: src.numel()].copy_(src[: dst.numel()])
        self.num_tiles, self.num_groups = other.num_tiles, other.num_groups
        return self


def mark_consecutive(tiles: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """Set tiles[:, 4] (CPU) to the first token of each tile whose tokens are order[start] + 0, 1, ...
    and to -1 otherwise (empty tiles: 0, never read)."""
    tiles = tiles.clone()
    tiles[:, 4] = 0
    order = order.to("cpu", torch.int64)
    T = order.numel()
    live = tiles[:, 2] > 0
    if T == 0 or not live.any():
        return tiles
    # step[i] = 1 where order[i + 1] == order[i] + 1; a tile is consecutive if every step inside it is.
    step = torch.zeros(T, dtype=torch.int64)
    step[:-1] = (order[1:] == order[:-1] + 1).long()
    csum = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(step, 0)])
    st, ln = tiles[live, 1].long(), tiles[live, 2].long()
    inside = csum[st + ln - 1] - csum[st]  # steps from position st to st + ln - 1
    first = torch.where(inside == ln - 1, order[st], torch.full_like(st, -1))
    tiles[live, 4] = first.to(torch.int32)
    return tiles


def _r_pad(r: int) -> int:
    p = 16
    while p < r:
        p *= 2
    return p


def default_block_m(T: int, num_groups: int, r_pad: int) -> int:
    """Tokens per tile: 16 for decode-sized batches or small groups; up to 64 for long groups
    (and <= 32 KiB of fp32 v per program for r_pad > 64, as rankfuse_fused._block_m)."""
    if T <= 128:
        return 16
    avg = T / max(num_groups, 1)
    bm = 16 if avg <= 16 else 32 if avg <= 32 else 64
    if r_pad > 64:
        bm = min(bm, 32 if r_pad <= 128 else 16)
    return bm


def tile_capacity(T: int, bm: int, num_adapters: int, mode: str = "group") -> int:
    """Most tiles any step of T tokens can need: every group adds at most one partial tile (in
    "segment" mode a group is a run of one adapter, so there can be T of them)."""
    if mode == "segment":
        return T + 1
    if mode == "adapter_bucket":  # a group per (adapter, width bucket)
        return -(-T // bm) + min(num_adapters * WIDTH_BUCKETS, T) + 1
    return -(-T // bm) + min(num_adapters, T) + 1


def build_meta(
    token_adapter,
    token_ranks,
    *,
    num_adapters: int,
    mode: str = "group",
    device="cuda",
    bm: int | None = None,
    r_pad: int | None = None,
    capacity: int | None = None,
) -> MultiMeta:
    """Per-step metadata for the RankFuse multi-adapter kernels (host work, done once per step).

    mode "group"        sort by adapter (stable); every tile's limit is its group's maximum rank (RF1-MA)
         "rank_sorted"  sort by (adapter, rank); every tile's limit is its own maximum rank (rf2_splitk)
         "segment"      keep the batch order; groups are runs of consecutive tokens on one adapter and
                        every tile's limit is its own maximum rank (rf2_segment)
         "adapter_bucket"  sort by (adapter, width bucket 16/32/64/128/256, rank); groups are
                        (adapter, bucket) pairs, so no tile spans two widths; every tile's limit is its
                        own maximum rank (RF2-MA with RANK_SKIP per adapter-width group)
    bm, r_pad and capacity default to what this step needs; pass fixed values (e.g. r_pad from
    r_max and capacity from tile_capacity) to reuse one captured CUDA graph across steps.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    ad = torch.as_tensor(token_adapter).to("cpu", torch.int64).flatten()
    rk = torch.as_tensor(token_ranks).to("cpu", torch.int64).flatten()
    T = rk.numel()
    if ad.numel() != T:
        raise ValueError("token_adapter and token_ranks must have the same length")
    if T and (int(ad.max()) >= num_adapters or int(rk.min()) < 0):
        raise ValueError("adapter ids must be < num_adapters and ranks >= 0")
    # Rank 0 or no adapter: the no-LoRA group, keyed past every adapter so it sorts last.
    none = (rk == 0) | (ad < 0)
    key_ad = torch.where(none, torch.full_like(ad, num_adapters), ad)
    rk = torch.where(none, torch.zeros_like(rk), rk)

    if mode == "segment":
        order = torch.arange(T)
        k_sorted = key_ad
        new_group = torch.ones(T, dtype=torch.bool)
        new_group[1:] = k_sorted[1:] != k_sorted[:-1]
    elif mode == "adapter_bucket":
        # width bucket index 1..5 (0 for rank 0): ceil(log2(max(r, 16) / 16)) + 1
        bidx = torch.where(rk > 0, torch.ceil(torch.log2(rk.clamp(min=16).double() / 16)).long() + 1, 0)
        gkey = key_ad * (WIDTH_BUCKETS + 1) + bidx
        order = torch.argsort(gkey * 1024 + rk, stable=True)
        k_sorted = key_ad[order]
        g_sorted = gkey[order]
        new_group = torch.ones(T, dtype=torch.bool)
        new_group[1:] = g_sorted[1:] != g_sorted[:-1]
    else:
        key = key_ad * 1024 + rk if mode == "rank_sorted" else key_ad
        order = torch.argsort(key, stable=True)
        k_sorted = key_ad[order]
        new_group = torch.ones(T, dtype=torch.bool)
        new_group[1:] = k_sorted[1:] != k_sorted[:-1]
    r_sorted = rk[order]
    g_start = torch.nonzero(new_group).flatten() if T else torch.zeros(0, dtype=torch.int64)
    G = g_start.numel()
    g_len = torch.diff(torch.cat([g_start, torch.tensor([T])])) if G else g_start
    g_ad = k_sorted[g_start] if G else g_start
    gid = torch.cumsum(new_group.long(), 0) - 1  # group of each position
    g_lim = torch.zeros(G, dtype=torch.int64).scatter_reduce_(0, gid, r_sorted, "amax") if G else g_start

    hi = int(rk.max()) if T else 0
    r_pad = r_pad or _r_pad(max(hi, 1))
    if hi > r_pad:
        raise ValueError(f"a rank of {hi} does not fit r_pad={r_pad}")
    real_groups = int((g_ad < num_adapters).sum()) if G else 0
    bm = bm or default_block_m(T, real_groups, r_pad)

    # Tiles: group g is cut into cdiv(len_g, bm) tiles.
    n_t = (g_len + bm - 1) // bm if G else g_start
    num_tiles = int(n_t.sum()) if G else 0
    t_group = torch.repeat_interleave(torch.arange(G), n_t) if G else g_start
    t_first = torch.cumsum(n_t, 0) - n_t if G else g_start
    t_idx = torch.arange(num_tiles) - t_first[t_group] if G else g_start  # tile index inside its group
    t_start = g_start[t_group] + t_idx * bm if G else g_start
    t_len = torch.clamp(g_len[t_group] - t_idx * bm, max=bm) if G else g_start
    t_ad = g_ad[t_group] if G else g_start
    if mode == "group":
        t_lim = g_lim[t_group] if G else g_start
    else:
        pos_tile = t_first[gid] + (torch.arange(T) - g_start[gid]) // bm if T else gid
        t_lim = torch.zeros(num_tiles, dtype=torch.int64).scatter_reduce_(0, pos_tile, r_sorted, "amax") if T else gid
    # The no-LoRA group keeps adapter -1 and limit 0 (its tiles only zero their rows).
    t_ad = torch.where(t_ad >= num_adapters, torch.full_like(t_ad, -1), t_ad)
    cap = capacity or max(num_tiles, 1)
    if num_tiles > cap:
        raise ValueError(f"{num_tiles} tiles exceed capacity {cap}")
    tiles = torch.zeros(cap, 5, dtype=torch.int32)
    if num_tiles:
        tiles[:num_tiles, :4] = torch.stack([t_ad, t_start, t_len, t_lim], 1).to(torch.int32)
    tiles = mark_consecutive(tiles, order)
    g_ad_out = torch.where(g_ad >= num_adapters, torch.full_like(g_ad, -1), g_ad) if G else g_start
    dev = torch.device(device)
    i32 = lambda t: t.to(torch.int32).to(dev)  # noqa: E731
    # Groups are sized to the largest possible count for in-place reloads (load_).
    most = T if mode == "segment" else min(num_adapters * WIDTH_BUCKETS, T) if mode == "adapter_bucket" else min(num_adapters, T)
    g_cap = max(G, most + 1) if capacity else max(G, 1)

    def padded(t):
        out = torch.zeros(g_cap, dtype=torch.int64)
        out[: t.numel()] = t
        return i32(out)

    return MultiMeta(
        mode=mode, T=T, bm=bm, r_pad=r_pad, order=i32(order), group_start=padded(g_start),
        group_len=padded(g_len), group_adapter=padded(g_ad_out), group_rank_limit=padded(g_lim),
        tiles=tiles.to(dev), num_tiles=num_tiles, num_groups=G,
        zero_idx=torch.nonzero(none).flatten().to(dev),
    )


# ------------------------------------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------------------------------------


@dataclass
class MultiImpl:
    """impl(x, W, token_adapter, token_ranks, out=None, add=False, **prepared) -> delta [T, h_out]

    prepare(x, W, token_adapter, token_ranks) -> dict of per-step metadata, built outside a timed
    or captured call (the call then makes no host syncs). out / add: add the delta into an
    existing output (add=True) or overwrite out (add=False; rows of rank-0 tokens become 0).
    """

    name: str
    fn: Callable
    prepare: Callable | None = None
    requires_cuda: bool = True
    dtypes: tuple[torch.dtype, ...] | None = None
    supports_slices: bool = True
    check: Callable | None = None  # check(W) -> None or a reason the impl cannot run W

    def __call__(self, x, W, token_adapter, token_ranks, **kw):
        return self.fn(x, W, token_adapter, token_ranks, **kw)

    def unsupported(self, W: PagedWeights) -> str | None:
        if self.dtypes is not None and W.dtype not in self.dtypes:
            return f"{self.name} does not support {W.dtype}"
        if W.n_slices > 1 and not self.supports_slices:
            return f"{self.name} does not support merged slices"
        return self.check(W) if self.check else None


MULTI_REGISTRY: dict[str, MultiImpl] = {}


def register_multi(name: str, **kw) -> Callable:
    def wrap(fn: Callable) -> MultiImpl:
        if name in MULTI_REGISTRY:
            raise ValueError(f"duplicate multi-adapter implementation name: {name}")
        impl = MultiImpl(name=name, fn=fn, **kw)
        MULTI_REGISTRY[name] = impl
        return impl

    return wrap


def load_all(include_slora: bool = True) -> dict[str, MultiImpl]:
    """Import every multi-adapter implementation module (registers them) and return the registry.
    B4 (S-LoRA) is registered only with include_slora (it JIT-builds a CUDA extension on first use)."""
    from importlib.util import find_spec

    from rankfuse.baselines import padded_multi, reference_multi  # noqa: F401

    if find_spec("triton") is not None:
        from rankfuse.kernels import rankfuse_multi  # noqa: F401
    if find_spec("vllm") is not None:
        from rankfuse.baselines import vllm_ops_multi  # noqa: F401
    if include_slora:
        from rankfuse.baselines import slora_bgmv_multi

        slora_bgmv_multi.enable()
    return MULTI_REGISTRY


def output_buffer(x: torch.Tensor, W: PagedWeights, out: torch.Tensor | None, add: bool) -> torch.Tensor:
    """The tensor a call writes into: `out` (checked) or a new one."""
    T = x.shape[0]
    if out is None:
        if add:
            raise ValueError("add=True needs an existing out to add into")
        return torch.empty((T, W.h_out), dtype=x.dtype, device=x.device)
    if tuple(out.shape) != (T, W.h_out) or out.dtype != x.dtype or out.stride(1) != 1:
        raise ValueError(f"out must be [{T}, {W.h_out}] {x.dtype} with unit column stride")
    return out


# ------------------------------------------------------------------------------------------------
# Workloads
# ------------------------------------------------------------------------------------------------

POPULARITY = ("uniform", "zipf")


@dataclass(frozen=True)
class MultiWorkload:
    num_requests: int
    tokens_per_request: int  # 1 = decode
    num_adapters: int
    popularity: str = "uniform"  # adapter of each request: uniform or Zipf(alpha) over adapter ids
    zipf_alpha: float = 1.0
    dist: str = "uniform"  # rank of each request over the tiers (rankfuse.workloads.DISTRIBUTIONS)
    h_in: int = 4096
    h_out: int = 4096
    r_max: int = 64
    dtype: str = "fp16"
    seed: int = 0
    pages: str = "dense"  # dense | contiguous | shuffled
    zero_frac: float = 0.0  # fraction of requests with no LoRA (rank 0)

    @property
    def num_tokens(self) -> int:
        return self.num_requests * self.tokens_per_request

    def as_row(self) -> dict:
        from dataclasses import asdict

        row = asdict(self)
        row["num_tokens"] = self.num_tokens
        return row


def sample_requests(w: MultiWorkload) -> tuple[list[int], list[int]]:
    """(adapter, rank) per request, seeded."""
    gen = torch.Generator().manual_seed(w.seed + 7)
    if w.popularity == "uniform":
        probs = torch.ones(w.num_adapters, dtype=torch.float64)
    elif w.popularity == "zipf":
        probs = 1.0 / torch.arange(1, w.num_adapters + 1, dtype=torch.float64) ** w.zipf_alpha
    else:
        raise ValueError(f"popularity must be one of {POPULARITY}")
    adapters = torch.multinomial(probs, w.num_requests, replacement=True, generator=gen).tolist()
    tiers = default_tiers(w.r_max)
    rp = torch.tensor(DISTRIBUTIONS[w.dist](len(tiers)), dtype=torch.float64)
    ranks = [tiers[i] for i in torch.multinomial(rp, w.num_requests, replacement=True, generator=gen).tolist()]
    if w.zero_frac > 0:
        zero = torch.rand(w.num_requests, generator=gen) < w.zero_frac
        ranks = [0 if z else r for r, z in zip(ranks, zero.tolist())]
    return adapters, ranks


def make_weights(w: MultiWorkload, device="cuda", slice_sizes=None, scale_per_rank=None) -> PagedWeights:
    """num_adapters seeded adapters at r_max, in the page layout w.pages."""
    gen = torch.Generator().manual_seed(w.seed)
    S = len(slice_sizes) if slice_sizes else 1
    n_max = max(slice_sizes) if slice_sizes else w.h_out
    dt = DTYPES[w.dtype]
    A = (torch.randn(w.num_adapters, S, w.r_max, w.h_in, generator=gen) / w.h_in**0.5).to(device, dt)
    B = (torch.randn(w.num_adapters, S, w.r_max, n_max, generator=gen) / w.r_max**0.5).to(device, dt)
    if w.pages == "dense":
        return PagedWeights.dense(A, B, slice_sizes, scale_per_rank)
    return PagedWeights.paged(A, B, slice_sizes, scale_per_rank, order=w.pages, seed=w.seed)


def make_batch(w: MultiWorkload, device="cuda", h_in: int | None = None):
    """(x [T, h_in], token_adapter [T] int32, token_ranks [T] int32) on the device; request order
    is arrival order (not sorted by adapter); a prompt's tokens are consecutive."""
    adapters, ranks = sample_requests(w)
    gen = torch.Generator().manual_seed(w.seed + 1)
    x = torch.randn(w.num_tokens, h_in or w.h_in, generator=gen).to(device, DTYPES[w.dtype])
    rep = lambda v: torch.tensor(v, dtype=torch.int32).repeat_interleave(w.tokens_per_request).to(device)  # noqa: E731
    return x, rep(adapters), rep(ranks)
