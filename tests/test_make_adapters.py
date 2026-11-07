"""scripts/make_adapters.py writes PEFT adapters whose rank-r copies are exact prefixes."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_adapters.py"
spec = importlib.util.spec_from_file_location("make_adapters", SCRIPT)
make_adapters = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_adapters)

TINY = {
    "hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 2,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
}


def _load(directory: Path) -> dict[str, torch.Tensor]:
    st = directory / "adapter_model.safetensors"
    if st.exists():
        from safetensors.torch import load_file

        return load_file(str(st))
    return torch.load(directory / "adapter_model.bin")


@pytest.fixture
def written(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(TINY))
    out = tmp_path / "adapters"
    make_adapters.main([
        "--config", str(config), "--base-model", "tiny/test", "--r-max", "16",
        "--num-adapters", "2", "--scale", "2.0", "--out", str(out),
    ])
    return out


def test_layout_and_shapes(written):
    full = _load(written / "matryoshka-0")
    shapes = make_adapters.projection_shapes(make_adapters.dims_from_config(TINY))
    assert len(full) == 2 * len(shapes) * TINY["num_hidden_layers"]
    for name, (parent, fan_in, fan_out) in shapes.items():
        prefix = f"base_model.model.model.layers.1.{parent}.{name}"
        assert full[f"{prefix}.lora_A.weight"].shape == (16, fan_in)
        assert full[f"{prefix}.lora_B.weight"].shape == (fan_out, 16)
        assert full[f"{prefix}.lora_A.weight"].dtype == torch.float16


def test_tier_copies_are_prefixes(written):
    full = _load(written / "matryoshka-0")
    meta = json.loads((written / "matryoshka-0" / "matryoshka.json").read_text())
    assert meta["tiers"] == [1, 2, 4, 8, 16]
    for r in meta["tiers"]:
        sliced = _load(written / f"matryoshka-0-r{r}")
        assert sliced.keys() == full.keys()
        for key, t in sliced.items():
            expected = full[key][:r] if key.endswith("lora_A.weight") else full[key][:, :r]
            assert torch.equal(t, expected), key


def test_configs_keep_the_same_effective_scale(written):
    for d in ["matryoshka-1", "matryoshka-1-r2", "matryoshka-1-r8"]:
        cfg = json.loads((written / d / "adapter_config.json").read_text())
        assert cfg["peft_type"] == "LORA" and cfg["base_model_name_or_path"] == "tiny/test"
        assert cfg["lora_alpha"] / cfg["r"] == pytest.approx(2.0)
        assert cfg["target_modules"] == make_adapters.ALL_TARGETS


def test_adapters_differ_by_seed(written):
    a, b = _load(written / "matryoshka-0"), _load(written / "matryoshka-1")
    key = next(iter(a))
    assert not torch.equal(a[key], b[key])
