"""Compare GPU-loaded weights vs safetensors originals + bf16 per-layer forward.

1. Loads LLM bf16 device_map=auto (exact training path)
2. Bit-compares sampled weights against safetensors source values
3. Manual layer-by-layer bf16 forward, printing EVERY layer
4. At the first layer that breaks: re-run that layer on CPU, compare

Run: ../venv-cu128/bin/python scripts/debug_weight_integrity.py
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

tok = AutoTokenizer.from_pretrained(NANBEIGE_DIR)
cfg = AutoConfig.from_pretrained(NANBEIGE_DIR)
ids = tok("The capital of France is", return_tensors="pt").input_ids

llm = LlamaForCausalLM.from_pretrained(
    NANBEIGE_DIR, config=cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    device_map="auto", max_memory={0: "22GiB", 1: "30GiB", 2: "30GiB"},
)
llm.eval()

# ── 1. weight integrity: GPU copy vs safetensors file ──
idx = json.load(open(Path(NANBEIGE_DIR) / "model.safetensors.index.json"))
wmap = idx["weight_map"]


def ref_weight(name):
    f = str(Path(NANBEIGE_DIR) / wmap[name])
    with safe_open(f, framework="pt", device="cpu") as sf:
        return sf.get_tensor(name)


print("\n=== weight integrity (GPU vs file) ===")
sd = llm.state_dict()
checks = (["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]
          + [f"model.layers.{i}.self_attn.q_proj.weight" for i in (0, 1, 2, 3, 5, 7, 14, 20, 30, 40, 51)]
          + [f"model.layers.{i}.mlp.down_proj.weight" for i in (0, 1, 2, 3, 5, 7, 14, 20, 30, 40, 51)])
n_bad = 0
for name in checks:
    gpu = sd[name].float().cpu()
    ref = ref_weight(name).float()
    diff = (gpu - ref).abs().max().item()
    ok = diff == 0.0
    n_bad += (not ok)
    if not ok:
        print(f"  ✗ {name}: max|Δ|={diff:.6f} gpu_max={gpu.abs().max().item():.4f} ref_max={ref.abs().max().item():.4f}")
print(f"{'ALL WEIGHTS IDENTICAL ✅' if n_bad == 0 else f'{n_bad}/{len(checks)} WEIGHTS CORRUPTED ❌'}")

# ── 2. bf16 manual per-layer forward ──
print("\n=== bf16 layer-by-layer forward ===")
model = llm.model
with torch.no_grad():
    dev0 = model.embed_tokens.weight.device
    h = model.embed_tokens(ids.to(dev0))
    pos_ids = torch.arange(ids.shape[1], device=h.device).unsqueeze(0)
    cos, sin = model.rotary_emb(h, pos_ids)
    print(f"embed: max|h|={h.abs().max().item():.4f}")

    first_bad = None
    for i, layer in enumerate(model.layers):
        d = next(layer.parameters()).device
        h = h.to(d)
        out = layer(h, attention_mask=None, position_ids=pos_ids.to(d),
                    position_embeddings=(cos.to(d), sin.to(d)), use_cache=False)
        h = out[0] if isinstance(out, tuple) else out
        m = h.abs().max().item()
        bad = (m == 0) or (m != m) or not torch.isfinite(h).all().item()
        if bad and first_bad is None:
            first_bad = (i, m, d, h.clone(), layer)
        if i < 20 or bad or i == 51:
            print(f"  layer {i:2d} [{d}]: max|h|={m:12.4f}{'  ⚠️' if bad else ''}")
        if first_bad is not None and i > first_bad[0] + 2:
            break

    if first_bad is None:
        h = model.norm(h.to(model.norm.weight.device))
        logits = llm.lm_head(h.to(llm.lm_head.weight.device))
        print(f"logits: max={logits.abs().max().item():.2f} next={tok.decode([logits[0,-1].argmax().item()])!r}")
    else:
        # ── 3. re-run first bad layer on CPU with same input/weights ──
        i, m, d, h_in, layer = first_bad
        print(f"\n=== first bad layer {i} on {d}: re-run on CPU ===")
        layer_cpu = layer.to("cpu").float()
        h_cpu = h_in.to("cpu").float()
        pos = pos_ids.to("cpu")
        pe = (cos.to("cpu").float(), sin.to("cpu").float())
        out_cpu = layer_cpu(h_cpu, attention_mask=None, position_ids=pos,
                            position_embeddings=pe, use_cache=False)
        hc = out_cpu[0] if isinstance(out_cpu, tuple) else out_cpu
        print(f"CPU float32 same layer: max|h|={hc.abs().max().item():.4f} finite={torch.isfinite(hc).all().item()}")
