"""RankFuse vLLM 0.32 plugin (rankfuse/integration): flag off = stock, patches, wrapper, op.

Tests that load the plugin run in a subprocess, so the monkeypatches never leak into the
rest of the test session. The end-to-end correctness gate against stock vLLM is
bench/vllm_correctness.py (several engines, minutes each); the small end-to-end test at the
bottom runs only with RANKFUSE_VLLM_E2E=1.
"""

import importlib.metadata
import os
import subprocess
import sys
import textwrap

import pytest
import torch

vllm = pytest.importorskip("vllm")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (vLLM's CUDA platform)")


def run_py(code: str, flag: bool, timeout=600) -> str:
    env = dict(os.environ)
    env.pop("RANKFUSE_VLLM", None)
    if flag:
        env["RANKFUSE_VLLM"] = "1"
    p = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env, capture_output=True, text=True,
                       timeout=timeout)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-5000:]
    return p.stdout


PATCH_TARGETS = """
from vllm.config.lora import LoRAConfig
from vllm.lora import model_manager as mm
from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
from vllm.platforms import current_platform
from vllm.v1 import engine as v1_engine
from vllm.v1.core import kv_cache_utils
from vllm.v1.worker import gpu_input_batch, lora_model_runner_mixin
from vllm.v1.worker.gpu import lora_utils

def snapshot():
    return dict(
        hash_keys=kv_cache_utils._gen_lora_extra_hash_keys,
        state_init=lora_utils.LoraState.__init__,
        add_request=lora_utils.LoraState.add_request,
        make_inputs=lora_utils.LoraState.make_lora_inputs,
        v1_inputs=gpu_input_batch.InputBatch.make_lora_inputs,
        set_active=lora_model_runner_mixin.LoRAModelRunnerMixin._set_active_loras,
        create=mm.LoRAModelManager._create_lora_modules,
        activate=mm.LoRAModelManager.activate_adapter,
        apply=BaseLinearLayerWithLoRA._apply_lora_to_output,
        punica=current_platform.get_punica_wrapper(),
        annotation=v1_engine.EngineCoreRequest.__annotations__["lora_request"],
        lora_hash=LoRAConfig(max_lora_rank=64, max_loras=1).compute_hash(),
    )
"""


def test_entry_point_registered():
    eps = importlib.metadata.entry_points(group="vllm.general_plugins")
    assert any(ep.name == "rankfuse" and ep.value == "rankfuse.integration.vllm_plugin:register" for ep in eps)


@cuda
def test_flag_off_is_stock_vllm():
    out = run_py(PATCH_TARGETS + """
before = snapshot()
from vllm.plugins import load_general_plugins
load_general_plugins()                       # what every vLLM process does
after = snapshot()
changed = [k for k in before if before[k] is not after[k] and before[k] != after[k]]
assert not changed, changed
assert after["punica"] == "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"
import rankfuse.integration.vllm_plugin as p
assert not p._DONE
print("STOCK OK")
""", flag=False)
    assert "STOCK OK" in out


@cuda
def test_flag_on_patches_and_request_rank():
    out = run_py(PATCH_TARGETS + """
before = snapshot()
from vllm.plugins import load_general_plugins
load_general_plugins()
after = snapshot()
assert all(before[k] != after[k] for k in before), [k for k in before if before[k] == after[k]]  # incl. the compile-cache hash
assert after["punica"] == "rankfuse.integration.punica.RankFusePunicaWrapper"
import rankfuse.integration.vllm_plugin as p
p.register()                                 # idempotent
assert p._DONE

# The rank survives the front end -> engine core wire format (msgpack, array_like structs).
from vllm import SamplingParams
from vllm.lora.request import LoRARequest
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from rankfuse.integration import MatryoshkaLoRARequest
enc, dec = MsgpackEncoder(), MsgpackDecoder(EngineCoreRequest)
for lr, want in [(MatryoshkaLoRARequest("m", 1, "/a", matryoshka_rank=8), 8),
                 (MatryoshkaLoRARequest("m", 1, "/a"), None), (LoRARequest("m", 1, "/a"), None)]:
    r = EngineCoreRequest("id", [1, 2, 3], None, SamplingParams(max_tokens=4), None, 0.0, lr, None, None)
    back = dec.decode(enc.encode(r)).lora_request
    assert back.lora_name == "m" and back.lora_int_id == 1 and getattr(back, "matryoshka_rank", None) == want, back

# Identity stays on the name: every rank of one adapter is one adapter (one slot).
a, b = MatryoshkaLoRARequest("m", 1, "/a", matryoshka_rank=4), MatryoshkaLoRARequest("m", 1, "/a", matryoshka_rank=64)
assert a == b and len({a, b, LoRARequest("m", 1, "/a")}) == 1
try:
    MatryoshkaLoRARequest("m", 1, "/a", matryoshka_rank=0)
    raise AssertionError("rank 0 accepted")
except ValueError:
    pass

# Prefix-cache keys: ranks of one adapter never share blocks; no rank = the stock key.
class R:  # the only attribute _gen_lora_extra_hash_keys reads
    def __init__(self, lr): self.lora_request = lr
keys = kv_cache_utils._gen_lora_extra_hash_keys
assert keys(R(a)) != keys(R(b))
assert keys(R(MatryoshkaLoRARequest("m", 1, "/a"))) == [("lora", "m", "/a")] == keys(R(LoRARequest("m", 1, "/a")))
assert keys(R(None)) == []

# Per-request ranks -> per-token rank tuple (model runner V2).
import numpy as np
st = lora_utils.LoraState(8)
st.add_request("r0", 0, MatryoshkaLoRARequest("m", 1, "/a", matryoshka_rank=4))
st.add_request("r1", 1, None)
st.add_request("r2", 2, MatryoshkaLoRARequest("m", 1, "/a"))
prompt, token, active = st.make_lora_inputs(["r0", "r1", "r2"], np.array([2, 0, 1]), np.array([1, 3, 2]))
assert tuple(token) == (1, 1, 1, 1, 0, 0) and token.ranks == (p_full := 1 << 30, 4, 4, 4, 0, 0), token.ranks
assert tuple(prompt) == (1, 1, 0) and len(active) == 1

# Mappings that differ only in ranks are different (the manager's skip-if-unchanged check).
M = p.RankedLoRAMapping
assert M((1, 1), (1,), True, index_rank_mapping=(4, 4)) != M((1, 1), (1,), True, index_rank_mapping=(4, 8))
assert M((1, 1), (1,), True, index_rank_mapping=(4, 4)) == M((1, 1), (1,), True, index_rank_mapping=(4, 4))
print("PATCHES OK")
""", flag=True)
    assert "PATCHES OK" in out


# ---------------------------------------------------------------------------------------------
# The op and the wrapper (in process: they patch nothing)
# ---------------------------------------------------------------------------------------------

def _weights(sizes, h_in, r_max=64, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = tuple((torch.randn(1, 1, r_max, h_in, generator=g, device="cuda") * 0.05).to(dtype) for _ in sizes)
    b = tuple((torch.randn(1, 1, w, r_max, generator=g, device="cuda") * 0.05).to(dtype) for w in sizes)
    A = torch.stack([t[0, 0] for t in a])
    B = torch.zeros(len(sizes), r_max, max(sizes), device="cuda", dtype=dtype)
    for s, t in enumerate(b):
        B[s, :, : t.shape[2]] = t[0, 0].T
    return a, b, A, B


def _reference(x, a, b, ranks):
    outs = []
    for As, Bs in zip(a, b):
        v = x.float() @ As[0, 0].float().T
        v = v * (torch.arange(v.shape[1], device="cuda")[None, :] < ranks[:, None])
        outs.append(v @ Bs[0, 0].float().T)
    return torch.cat(outs, 1)


@cuda
@pytest.mark.parametrize("sizes,h_in", [((896, 128, 128), 896), ((4864, 4864), 896), ((896,), 4864)])
def test_wrapper_matches_reference_and_stock(sizes, h_in):
    from vllm.config.lora import LoRAConfig
    from vllm.lora.layers import LoRAMapping
    from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU

    from rankfuse.integration.punica import FULL_RANK, RankFusePunicaWrapper

    cfg = LoRAConfig(max_lora_rank=64, max_loras=1)
    rfw = RankFusePunicaWrapper(512, 64, "cuda", lora_config=cfg)
    stock = PunicaWrapperGPU(512, 64, "cuda", lora_config=cfg)
    a, b, A, B = _weights(sizes, h_in)
    T = 37
    x = torch.randn(T, h_in, device="cuda", dtype=torch.bfloat16)
    y0 = torch.randn(T, sum(sizes), device="cuda", dtype=torch.bfloat16)
    ids = [0 if i % 9 == 4 else 1 for i in range(T)]  # some tokens without an adapter
    req_ranks = [[1, 2, 4, 8, 16, 32, 64, FULL_RANK][i % 8] for i in range(T)]

    mapping = LoRAMapping(tuple(ids), tuple(ids), is_prefill=True)
    mapping.index_rank_mapping = tuple(req_ranks)
    rfw.update_metadata(mapping, [1], 2, 32000)
    eff = torch.tensor([0 if i == 0 else min(r, 64) for i, r in zip(ids, req_ranks)], device="cuda")
    assert torch.equal(rfw._token_lora_ranks[:T].long(), eff) and int(rfw._r_limit_cpu[0]) == 64

    for add in (True, False):
        y = y0.clone()
        rfw.add_lora_linear(y, x, a, b, 1.0, sizes, rankfuse_weights=(A, B), add_inputs=add)
        ref = _reference(x, a, b, eff) + (y0.float() if add else 0)
        assert (y.float() - ref).abs().max().item() < 0.05, add

    # Every token at full rank: the same as stock vLLM's shrink/expand (to bf16 rounding).
    full = LoRAMapping(tuple([1] * T), tuple([1] * T), is_prefill=True)
    stock.update_metadata(full, [1], 2, 32000)
    rfw.update_metadata(full, [1], 2, 32000)  # no ranks on the mapping: full rank
    ys, yr = y0.clone(), y0.clone()
    stock.add_lora_linear(ys, x, a, b, 1.0, sizes)
    rfw.add_lora_linear(yr, x, a, b, 1.0, sizes, rankfuse_weights=(A, B))
    assert (ys.float() - yr.float()).abs().max().item() < 0.05

    # Tokens past this step's count (CUDA-graph padding) read rank 0.
    small = LoRAMapping((1, 1, 1), (1, 1, 1), is_prefill=True)
    small.index_rank_mapping = (8, 8, 8)
    rfw.update_metadata(small, [1], 2, 32000)
    assert rfw._token_lora_ranks[:3].tolist() == [8, 8, 8] and rfw._token_lora_ranks[3:T].abs().sum() == 0
    assert int(rfw._r_limit_cpu[0]) == 8
    # A step without adapter tokens: rank 0 everywhere, r_limit 0, the op adds nothing.
    none = LoRAMapping((0, 0, 0), (0, 0, 0), is_prefill=True)
    none.index_rank_mapping = (0, 0, 0)
    rfw.update_metadata(none, [1], 2, 32000)
    assert rfw._token_lora_ranks[:3].abs().sum() == 0 and int(rfw._r_limit_cpu[0]) == 0
    y = y0.clone()
    rfw.add_lora_linear(y[:3], x[:3], a, b, 1.0, sizes, rankfuse_weights=(A, B))
    assert torch.equal(y, y0)


@cuda
def test_op_graph_capture_uses_r_max():
    # Captured with a low step r_limit, the graph must still be exact for later steps at higher ranks.
    from rankfuse.integration import ops

    sizes, h_in, T = (896, 128, 128), 896, 8
    a, b, A, B = _weights(sizes, h_in, dtype=torch.float16)
    x = torch.randn(T, h_in, device="cuda", dtype=torch.float16)
    ranks = torch.zeros(64, dtype=torch.int32, device="cuda")
    ranks[:T] = 2
    r_cpu = torch.tensor([2], dtype=torch.int32)
    y0 = torch.randn(T, sum(sizes), device="cuda", dtype=torch.float16)
    y = y0.clone()
    ops.lora_linear(y, x, A, B, ranks, r_cpu, list(sizes), None, True, 64)  # eager warm-up
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s), torch.cuda.graph(g):
        ops.lora_linear(y, x, A, B, ranks, r_cpu, list(sizes), None, True, 64)
    ranks[:T] = torch.tensor([64, 1, 32, 2, 64, 16, 8, 4], dtype=torch.int32)
    y.copy_(y0)
    g.replay()
    torch.cuda.synchronize()
    ref = _reference(x, a, b, ranks[:T].long()) + y0.float()
    assert (y.float() - ref).abs().max().item() < 0.02


@cuda
def test_op_under_torch_compile():
    from rankfuse.integration import ops

    sizes, h_in, T = (896,), 896, 16
    a, b, A, B = _weights(sizes, h_in, dtype=torch.float16)
    ranks = torch.randint(0, 65, (128,), dtype=torch.int32, device="cuda")
    r_cpu = torch.tensor([64], dtype=torch.int32)

    def f(x, y):
        ops.lora_linear(y, x, A, B, ranks, r_cpu, list(sizes), None, True, 128)
        return y * 1

    x = torch.randn(T, h_in, device="cuda", dtype=torch.float16)
    y0 = torch.randn(T, 896, device="cuda", dtype=torch.float16)
    want = f(x, y0.clone())
    got = torch.compile(f, fullgraph=True)(x, y0.clone())
    assert torch.equal(got, want)


@pytest.mark.skipif(os.environ.get("RANKFUSE_VLLM_E2E") != "1", reason="end-to-end engine run; set RANKFUSE_VLLM_E2E=1")
def test_end_to_end_ranks_reach_the_kernel():
    out = run_py("""
import os
from vllm import LLM, SamplingParams
from rankfuse.integration import MatryoshkaLoRARequest
path = os.path.join(os.environ["RF_ADAPTERS"], "qwen2.5-0.5b", "matryoshka-0")
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", enable_lora=True, max_lora_rank=64, max_loras=1,
          max_model_len=1024, gpu_memory_utilization=0.3, enforce_eager=True)
sp = SamplingParams(temperature=0, max_tokens=24)
ps = ["Explain photosynthesis to a child.", "List three uses of copper.", "Why is the sky blue?", "What is TCP?"]
def gen(r):
    req = None if r == 0 else MatryoshkaLoRARequest("matryoshka-0", 1, path, matryoshka_rank=r)
    return [o.outputs[0].token_ids for o in llm.generate(ps, sp, lora_request=req, use_tqdm=False)]
base, r1, r64 = gen(0), gen(1), gen(64)
full = [o.outputs[0].token_ids for o in llm.generate(ps, sp, lora_request=MatryoshkaLoRARequest("matryoshka-0", 1, path), use_tqdm=False)]
assert full == r64 and r1 != r64 and r64 != base
print("E2E OK")
""", flag=True, timeout=1200)
    assert "E2E OK" in out
