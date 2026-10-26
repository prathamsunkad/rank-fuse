"""Workload generation: tier presets and the controlled low_frac_<pct> rank mixes (CPU only)."""

import pytest
import torch

from rankfuse.workloads import Workload, is_known_dist, make_batch, sample_request_ranks


@pytest.mark.parametrize("r_max, expected", [(64, [8, 16, 32, 64]), (128, [16, 32, 64, 128]), (4, [1, 2, 4])])
def test_fractions_preset(r_max, expected):
    assert Workload(1, 1, r_max=r_max, tiers="fractions").rank_tiers == expected


def test_default_tiers_unchanged():
    assert Workload(1, 1, r_max=64).rank_tiers == [1, 2, 4, 8, 16, 32, 64]


@pytest.mark.parametrize("requests", [1, 2, 3, 16, 100])
@pytest.mark.parametrize("pct", [0, 25, 50, 75, 90, 100])
def test_low_frac_counts(requests, pct):
    w = Workload(requests, 1, r_max=64, tiers="fractions", dist=f"low_frac_{pct}")
    ranks = sample_request_ranks(w, torch.Generator().manual_seed(0))
    n_low = min(round(pct / 100 * requests), requests - 1)
    assert sorted(ranks) == [8] * n_low + [64] * (requests - n_low)
    assert max(ranks) == 64  # batch maximum rank is always r_max


def test_low_frac_seed_shuffles_order_only():
    a, b = (make_batch(Workload(32, 4, h_in=16, h_out=16, dist="low_frac_75", seed=s)) for s in (0, 1))
    assert sorted(a.token_ranks.tolist()) == sorted(b.token_ranks.tolist())
    assert a.token_ranks.tolist() != b.token_ranks.tolist()


def test_known_dists():
    assert all(is_known_dist(d) for d in ["uniform", "skew_low", "low_frac_0", "low_frac_100"])
    assert not any(is_known_dist(d) for d in ["low_frac_", "low_frac_101", "low_frac_x", "nope"])
    with pytest.raises(ValueError):
        make_batch(Workload(2, 1, h_in=8, h_out=8, dist="low_frac_x"))
