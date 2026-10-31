"""Manual layer-by-layer forward bypassing accelerate hooks.

Prints max|h| after every layer + device transitions to find where hidden
states die (expect ~correct logits at the end if hooks are the culprit).

Run: ../venv-cu128/bin/python scripts/debug_manual_pipeline.py
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

tok = AutoTokenizer.from_pretrained(NANBEIGE_DIR)
cfg = AutoConfig.from_pretrained(NANBEIGE_DIR)
ids = tok("The capital of France is", return_tensors="pt").input_ids
print(f"ids: {ids.tolist()}")

llm = LlamaForCausalLM.from_pretrained(
    NANBEIGE_DIR, config=cfg, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    device_map="auto", max_memory={0: "22GiB", 1: "30GiB", 2: "30GiB"},
)
llm.eval()
model = llm.model

with torch.no_grad():
    dev0 = model.embed_tokens.weight.device
    h = model.embed_tokens(ids.to(dev0))
    print(f"\nembed: max|h|={h.abs().max().item():.4f} dev={h.device}")

    pos_ids = torch.arange(ids.shape[1], device=h.device).unsqueeze(0)
    cos, sin = model.rotary_emb(h, pos_ids)
    print(f"rotary: cos dev={cos.device} max={cos.abs().max().item():.4f}")

    prev_dev = h.device
    for i, layer in enumerate(model.layers):
        d = next(layer.parameters()).device
        if d != prev_dev:
            print(f"  ── layer {i}: device {prev_dev} → {d}")
        h = h.to(d)
        pos, attention_mask = pos_ids.to(d), None
        pe = (cos.to(d), sin.to(d))
        out = layer(h, attention_mask=attention_mask, position_ids=pos,
                    position_embeddings=pe, use_cache=False)
        h = out[0] if isinstance(out, tuple) else out
        prev_dev = d
        if i % 4 == 0 or d != prev_dev or i == len(model.layers) - 1:
            m = h.abs().max().item()
            flag = "  ⚠️ ZERO" if m == 0 else ""
            print(f"  layer {i:2d}: max|h|={m:10.4f} dev={d}{flag}")

    h = h.to(model.norm.weight.device)
    h = model.norm(h)
    print(f"final norm: max|h|={h.abs().max().item():.4f}")
    logits = llm.lm_head(h.to(llm.lm_head.weight.device))
    print(f"MANUAL PIPELINE logits: max|logits|={logits.abs().max().item():.4f}")
    nxt = logits[0, -1].argmax().item()
    print(f"next token: {tok.decode([nxt])!r}  (expect ~' Paris')")
