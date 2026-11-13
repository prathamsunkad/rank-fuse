"""Roofline numbers must match the worked example in docs/PLAN.md."""

import pytest

from rankfuse.roofline import cost

H = 4096
RANKS = [4, 8, 16, 32]  # one decode request per tier
MIB = 2**20


@pytest.mark.parametrize(
    "strategy,expected_mib",
    [("fused_ideal", 0.5), ("padded", 0.5), ("separate_true", 60 / 32 * 0.5), ("separate_padded", 2.0)],
)
def test_weight_bytes_worked_example(strategy, expected_mib):
    c = cost(strategy, RANKS, H, H, r_max=32, dtype="fp16")
    assert c.weight_bytes / MIB == pytest.approx(expected_mib)


def test_token_traffic_is_24kb_per_decode_token():
    c = cost("fused_ideal", [32], H, H, r_max=32, dtype="fp16")
    assert c.activation_bytes == 3 * H * 2  # read x, read + write y


def test_fused_never_moves_more_than_padded():
    for ranks in ([4] * 64, [4, 32] * 32, [32] * 64):
        assert cost("fused_ideal", ranks, H, H, 32).total_bytes <= cost("padded", ranks, H, H, 32).total_bytes


def test_weight_advantage_shrinks_with_batch():
    """Matryoshka's weight saving matters at small batch and fades at large batch."""

    def ratio(repeat):
        ranks = RANKS * repeat
        return cost("separate_padded", ranks, H, H, 32).total_bytes / cost("fused_ideal", ranks, H, H, 32).total_bytes

    assert ratio(1) > 2 * ratio(128)
