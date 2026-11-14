"""Workload generation: tier presets and the controlled low_frac_<pct> rank mixes (CPU only)."""

import pytest
import torch

from rankfuse.workloads import DISTRIBUTIONS, Workload, is_known_dist, make_batch, sample_request_ranks


@pytest.mark.parametrize("r_max, expected", [(64, [8, 16, 32, 64]), (128, [16, 32, 64, 128]), (4, [1, 2, 4])])
def test_fractions_preset(r_max, expected):
    assert Workload(1, 1, r_max=r_max, tiers="fractions").rank_tiers == expected


def test_default_tiers_unchanged():
    assert Workload(1, 1, r_max=64).rank_tiers == [1, 2, 4, 8, 16, 32, 64]


@pytest.mark.parametrize("requests", [1, 2, 3, 16, 100])
@pytest.mark.parametrize("pct", [0, 25, 50, 75, 90, 99, 100])
def test_low_frac_counts(requests, pct):
    w = Workload(requests, 1, r_max=64, tiers="fractions", dist=f"low_frac_{pct}")
    ranks = sample_request_ranks(w, torch.Generator().manual_seed(0))
    n_low = min(round(pct / 100 * requests), requests - 1)
    assert sorted(ranks) == [8] * n_low + [64] * (requests - n_low)
    assert max(ranks) == 64  # batch maximum rank is always r_max


@pytest.mark.parametrize("dist", ["zipf_1", "zipf_2"])
@pytest.mark.parametrize("n", [1, 4, 9])
def test_zipf_weights(dist, n):
    w = DISTRIBUTIONS[dist](n)
    assert sum(w) == pytest.approx(1.0)
    assert all(a > b for a, b in zip(w, w[1:]))  # every higher rank is rarer


def test_zipf_r256_bucket_shares():
    """The shares quoted for r_max 256's nine pow2 tiers: ranks 1-16 (the 16-row bucket) and r_max."""
    for dist, low, top in [("zipf_1", 0.807, 0.039), ("zipf_2", 0.951, 0.008)]:
        w = DISTRIBUTIONS[dist](len(Workload(1, 1, r_max=256).rank_tiers))
        assert sum(w[:5]) == pytest.approx(low, abs=1e-3) and w[-1] == pytest.approx(top, abs=1e-3)


def test_low_frac_seed_shuffles_order_only():
    a, b = (make_batch(Workload(32, 4, h_in=16, h_out=16, dist="low_frac_75", seed=s)) for s in (0, 1))
    assert sorted(a.token_ranks.tolist()) == sorted(b.token_ranks.tolist())
    assert a.token_ranks.tolist() != b.token_ranks.tolist()


def test_known_dists():
    assert all(is_known_dist(d) for d in ["uniform", "skew_low", "low_frac_0", "low_frac_100"])
    assert not any(is_known_dist(d) for d in ["low_frac_", "low_frac_101", "low_frac_x", "nope"])
    with pytest.raises(ValueError):
        make_batch(Workload(2, 1, h_in=8, h_out=8, dist="low_frac_x"))


def test_defaults_unchanged_by_order_and_lengths_fields():
    """Earlier sweeps' batches: same ranks and inputs when the new fields keep their defaults."""
    w = Workload(16, 4, h_in=16, h_out=16, dist="uniform", seed=5)
    assert w.num_tokens == 64 and w.request_lengths == [4] * 16
    a, b = make_batch(w), make_batch(Workload(16, 4, h_in=16, h_out=16, dist="uniform", seed=5, request_order="random"))
    assert torch.equal(a.x, b.x) and torch.equal(a.token_ranks, b.token_ranks)


def test_sorted_order_moves_whole_requests():
    w = Workload(1, 1, h_in=16, h_out=16, dist="uniform", seed=2)
    rand = make_batch(Workload(32, 1, h_in=16, h_out=16, dist="uniform", seed=2, lengths="serving_4"))
    done = make_batch(Workload(32, 1, h_in=16, h_out=16, dist="uniform", seed=2, lengths="serving_4", request_order="sorted"))
    assert w.num_tokens == 1
    assert done.token_ranks.tolist() == sorted(rand.token_ranks.tolist())
    starts = done.seg_starts.tolist()
    for lo, hi in zip(starts, starts[1:]):  # every request's tokens stay together at one rank
        assert len(set(done.token_ranks[lo:hi].tolist())) == 1
    assert sorted(b - a for a, b in zip(starts, starts[1:])) == sorted(
        b - a for a, b in zip(rand.seg_starts.tolist(), rand.seg_starts.tolist()[1:])
    )


@pytest.mark.parametrize("decode", [8, 32, 128])
@pytest.mark.parametrize("prompts", [0, 1, 4])
@pytest.mark.parametrize("seed", range(6))
def test_serving_step_shape(decode, prompts, seed):
    w = Workload(decode, 1, h_in=8, h_out=8, seed=seed, lengths=f"serving_{prompts}")
    lens = w.request_lengths
    assert lens == w.request_lengths  # reproducible from the seed
    assert len(lens) == decode + prompts and lens.count(1) >= decode
    assert sum(lens) == w.num_tokens <= 8192 and max(lens) <= 4096
    assert make_batch(w).num_tokens == w.num_tokens


def test_unknown_order_and_lengths():
    with pytest.raises(ValueError):
        make_batch(Workload(2, 1, h_in=8, h_out=8, request_order="nope"))
    with pytest.raises(ValueError):
        Workload(2, 1, lengths="nope").request_lengths
    with pytest.raises(ValueError):
        Workload(2, 4, lengths="serving_1").request_lengths


@pytest.mark.parametrize("decode, budget, chunk, expected", [
    (32, 2048, 512, [512, 512, 512, 480]),
    (256, 2048, 512, [512, 512, 512, 256]),
    (0, 1024, 512, [512, 512]),
    (64, 64, 512, []),
])
def test_chunked_step_shape(decode, budget, chunk, expected):
    w = Workload(decode, 1, h_in=8, h_out=8, seed=3, lengths=f"chunked_{budget}_{chunk}")
    lens, prefill = w.request_lengths, w.request_is_prefill
    assert w.num_tokens == budget and lens == w.request_lengths
    assert sorted(n for n, p in zip(lens, prefill) if p) == sorted(expected)
    assert [n for n, p in zip(lens, prefill) if not p] == [1] * decode
    assert make_batch(w).num_tokens == budget


@pytest.mark.parametrize("bad", ["chunked_2048", "chunked_2048_0", "chunked_16_x", "chunked_2048_512_1"])
def test_chunked_bad_presets(bad):
    with pytest.raises(ValueError):
        Workload(32, 1, lengths=bad).request_lengths


def test_prefill_dist_ranks_by_request_type():
    w = Workload(64, 1, h_in=8, h_out=8, r_max=128, dist="all_min", prefill_dist="all_max", lengths="chunked_2048_256", seed=1)
    ranks = sample_request_ranks(w, torch.Generator().manual_seed(0))
    assert all(r == (128 if p else 1) for r, p in zip(ranks, w.request_is_prefill))
    b = make_batch(w)
    assert b.token_ranks.tolist().count(1) == 64 and b.num_tokens == 2048


def test_prefill_dist_none_keeps_serving_batches():
    """serving_<k> steps keep their shapes and ranks with the new field at its default."""
    for seed in range(4):
        a = make_batch(Workload(32, 1, h_in=8, h_out=8, dist="skew_low", seed=seed, lengths="serving_4"))
        b = make_batch(Workload(32, 1, h_in=8, h_out=8, dist="skew_low", seed=seed, lengths="serving_4", prefill_dist=None))
        assert torch.equal(a.token_ranks, b.token_ranks) and torch.equal(a.x, b.x)


def test_prefill_dist_needs_a_step():
    with pytest.raises(ValueError):
        make_batch(Workload(4, 8, h_in=8, h_out=8, prefill_dist="all_max"))
