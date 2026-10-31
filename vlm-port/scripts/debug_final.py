"""Final diagnosis: H2D/D2H integrity + host-staged cross-GPU + single-GPU compute.

A. H2D/D2H roundtrip per GPU (weights load path) — must be bit-exact
B. host-staged cross-GPU copy (cuda:0 → cpu → cuda:1) — the P2P workaround
C. weight integrity for layers 0-14 on cuda:0 (vs safetensors)
D. GPU vs CPU layer-by-layer forward (layers 0-14, h never leaves cuda:0)

Run: ../venv-cu128/bin/python scripts/debug_final.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_PROJECT_SRC))

import torch
from safetensors import safe_open
from transformers import LlamaForCausalLM, AutoConfig, AutoTokenizer

torch.backends.cudnn.enabled = False
NANBEIGE_DIR = "/home/acproject/usb/transplant/Nanbeige4.3-33B"

print("=== A. H2D/D2H roundtrip ===")
for g in range(3):
    a = torch.randn(2048, 7168, dtype=torch.bfloat16)
    ok = torch.equal(a, a.to(f"cuda:{g}").cpu())
    print(f"  cuda:{g}: roundtrip bit-exact = {ok}")

print("\n=== B. host-staged cross-GPU ===")
a = torch.randn(2048, 7168, device="cuda:0", dtype=torch.bfloat16)
b = a.cpu().to("cuda:1")
print(f"  cuda:0 → cpu → cuda:1: bit-exact = {torch.equal(a.cpu(), b.cpu())}")

print("\n=== C+D. model compare ===")
tok = AutoTokenizer.from_pretrained(NANBEIGE_DIR)
cfg = AutoConfig.from_pretrained(NANBEIGE_DIR)
ids = tok("The capital of France is", return_tensors="pt").input_ids

llm = LlamaForCausalLM.from_pretrained(
    NANBEIGE_DIR, config=cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    device_map="auto", max_memory={0: "22GiB", 1: "30GiB", 2: "30GiB"},
)
llm.eval()
model = llm.model

idx = json.load(open(Path(NANBEIGE_DIR) / "model.safetensors.index.json"))
wmap = idx["weight_map"]


def ref_weight(name):
    with safe_open(str(Path(NANBEIGE_DIR) / wmap[name]), framework="pt", device="cpu") as sf:
        return sf.get_tensor(name)


print("\n[C] weight integrity layers 0-14 (cuda:0):")
bad = []
for i in [0, 3, 7, 10, 14]:
    for part in ["self_attn.q_proj", "mlp.down_proj"]:
        name = f"model.layers.{i}.{part}.weight"
        gpu = dict(llm.named_parameters())[name].float().cpu()
        ok = torch.equal(gpu, ref_weight(name).float())
        if not ok:
            bad.append(name)
print(f"  {'ALL IDENTICAL ✅' if not bad else 'CORRUPTED: ' + ', '.join(bad)}")

print("\n[D] GPU vs CPU layer-by-layer (layers 0-14):")
with torch.no_grad():
    dev0 = model.embed_tokens.weight.device
    h_gpu = model.embed_tokens(ids.to(dev0))
    pos_ids = torch.arange(ids.shape[1], device=dev0).unsqueeze(0)
    cos_g, sin_g = model.rotary_emb(h_gpu, pos_ids)

    for i, layer in enumerate(model.layers[:15]):
        d = next(layer.parameters()).device
        assert d == torch.device("cuda:0"), f"layer {i} unexpectedly on {d}"
        out = layer(h_gpu.to(d), attention_mask=None, position_ids=pos_ids.to(d),
                    position_embeddings=(cos_g.to(d), sin_g.to(d)), use_cache=False)
        h_gpu = out[0] if isinstance(out, tuple) else out

        # CPU reference from safetensors
        sd = {}
        for part in ["input_layernorm", "post_attention_layernorm"]:
            sd[f"{part}.weight"] = ref_weight(f"model.layers.{i}.{part}.weight")
        for part in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
            sd[f"self_attn.{part}.weight" if "proj" in part and part in ("q_proj", "k_proj", "v_proj", "o_proj")
               else f"mlp.{part}.weight"] = ref_weight(f"model.layers.{i}.{part}.weight")
        import transformers.models.llama.modeling_llama as ml
        lc = ml.LlamaDecoderLayer(cfg, layer_idx=i).to(torch.bfloat16)
        lc.load_state_dict(sd, strict=False)
        lc.eval()
        h_cpu = h_in_cpu if i > 0 else model.embed_tokens(ids).to(torch.bfloat16)
        pos_c = torch.arange(ids.shape[1]).unsqueeze(0)
        cos_c, sin_c = model.rotary_emb(h_cpu.float(), pos_c)
        oc = lc(h_cpu, attention_mask=None, position_ids=pos_c,
                position_embeddings=(cos_c.to(torch.bfloat16), sin_c.to(torch.bfloat16)), use_cache=False)
        h_in_cpu = oc[0] if isinstance(oc, tuple) else oc

        diff = (h_gpu.float().cpu() - h_in_cpu.float()).abs().max().item()
        print(f"  layer {i:2d}: GPU max|h|={h_gpu.abs().max().item():10.4f} | "
              f"CPU max|h|={h_in_cpu.abs().max().item():10.4f} | max|Δ|={diff:.4f}")
        if i >= 14:
            break
