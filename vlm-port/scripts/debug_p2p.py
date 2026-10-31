"""Definitive: is it P2P cross-GPU copy corruption or per-GPU compute?

1. torch.cuda.can_device_access_peer for all pairs
2. Direct copy tests cuda:0→1→2 (bf16, small + large tensors) — compare bit-exact
3. bf16 manual forward printing input&output max for layers 10-22 and 30-40
4. Layer 15 fed RANDOM input computed directly on cuda:1 (no cross-GPU copy)

Run: ../venv-cu128/bin/python scripts/debug_p2p.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_PROJECT_SRC))

import torch
from transformers import LlamaForCausalLM, AutoConfig, AutoTokenizer

torch.backends.cudnn.enabled = False
NANBEIGE_DIR = "/home/acproject/usb/transplant/Nanbeige4.3-33B"

print("=== peer access ===")
for i in range(3):
    for j in range(3):
        if i != j:
            print(f"  can_device_access_peer({i},{j}) = {torch.cuda.can_device_access_peer(i, j)}")

print("\n=== copy tests (bf16) ===")
for src, dst in [(0, 1), (1, 2), (0, 2), (1, 0), (2, 1), (2, 0)]:
    for shape, tag in [((64, 7168), "small"), ((2048, 7168), "200MB")]:
        a = torch.randn(*shape, device=f"cuda:{src}", dtype=torch.bfloat16)
        b = a.to(f"cuda:{dst}")
        same = torch.equal(a.cpu(), b.cpu())
        if not same or tag == "small":
            print(f"  cuda:{src}→cuda:{dst} [{tag}]: bit-exact={same} "
                  f"src_max={a.abs().max().item():.4f} dst_max={b.abs().max().item():.4f}")

tok = AutoTokenizer.from_pretrained(NANBEIGE_DIR)
cfg = AutoConfig.from_pretrained(NANBEIGE_DIR)
ids = tok("The capital of France is", return_tensors="pt").input_ids
llm = LlamaForCausalLM.from_pretrained(
    NANBEIGE_DIR, config=cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    device_map="auto", max_memory={0: "22GiB", 1: "30GiB", 2: "30GiB"},
)
llm.eval()
model = llm.model

print("\n=== per-layer input/output (bf16) ===")
with torch.no_grad():
    dev0 = model.embed_tokens.weight.device
    h = model.embed_tokens(ids.to(dev0))
    pos_ids = torch.arange(ids.shape[1], device=h.device).unsqueeze(0)
    cos, sin = model.rotary_emb(h, pos_ids)
    for i, layer in enumerate(model.layers):
        d = next(layer.parameters()).device
        h_in = h.to(d)
        out = layer(h_in, attention_mask=None, position_ids=pos_ids.to(d),
                    position_embeddings=(cos.to(d), sin.to(d)), use_cache=False)
        h = out[0] if isinstance(out, tuple) else out
        if 10 <= i <= 22 or 30 <= i <= 40 or i in (0, 51):
            print(f"  layer {i:2d} [{d}]: in_max={h_in.abs().max().item():10.4f} → out_max={h.abs().max().item():10.4f}")

    # layer 15 with random input computed directly on its device
    layer15 = model.layers[15]
    d15 = next(layer15.parameters()).device
    r = torch.randn(1, 6, 7168, device=d15, dtype=torch.bfloat16)
    pe = (cos.to(d15), sin.to(d15))
    out = layer15(r, attention_mask=None, position_ids=pos_ids.to(d15),
                  position_embeddings=pe, use_cache=False)
    o = out[0] if isinstance(out, tuple) else out
    print(f"\nlayer15 random input ON {d15} (no copy): in_max={r.abs().max().item():.4f} "
          f"out_max={o.abs().max().item():.4f} finite={torch.isfinite(o).all().item()}")

    # same layer, same random input, but computed on cuda:0 (weights moved)
    layer15_gpu0 = layer15.to("cuda:0")
    r0 = r.to("cuda:0")
    pe0 = (cos.to("cuda:0"), sin.to("cuda:0"))
    out0 = layer15_gpu0(r0, attention_mask=None, position_ids=pos_ids.to("cuda:0"),
                        position_embeddings=pe0, use_cache=False)
    o0 = out0[0] if isinstance(out0, tuple) else out0
    print(f"layer15 same weights on cuda:0:        out_max={o0.abs().max().item():.4f} "
          f"finite={torch.isfinite(o0).all().item()}")
    diff = (o.float().cpu() - o0.float().cpu()).abs().max().item()
    print(f"max |cuda:1_out - cuda:0_out| = {diff:.6f}")
