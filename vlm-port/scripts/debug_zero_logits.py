"""Isolate zero-logits bug: input_ids path vs inputs_embeds path vs weight stats.

Run: ../venv-cu128/bin/python scripts/debug_zero_logits.py
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
llm = LlamaForCausalLM.from_pretrained(
    NANBEIGE_DIR, config=cfg, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    device_map="auto", max_memory={0: "22GiB", 1: "30GiB", 2: "30GiB"},
)
llm.eval()

print("\n=== weight stats ===")
print(f"tie_word_embeddings(cfg): {getattr(cfg, 'tie_word_embeddings', None)}")
print(f"lm_head is embed_tokens  : {llm.lm_head.weight is llm.model.embed_tokens.weight}")
for name, w in [
    ("embed_tokens", llm.model.embed_tokens.weight),
    ("lm_head", llm.lm_head.weight),
    ("final_norm", llm.model.norm.weight),
    ("layer0.q_proj", llm.model.layers[0].self_attn.q_proj.weight),
    ("layer25.mlp.up", llm.model.layers[25].mlp.up_proj.weight),
    ("layer51.o_proj", llm.model.layers[51].self_attn.o_proj.weight),
]:
    print(f"{name:16s} shape={tuple(w.shape)} max|w|={w.abs().max().item():.4f} dev={w.device}")

print(f"\nlm_head dev={llm.lm_head.weight.device} | embed dev={llm.model.embed_tokens.weight.device}")
print(f"layer0 dev={llm.model.layers[0].self_attn.q_proj.weight.device} | "
      f"layer51 dev={llm.model.layers[51].self_attn.q_proj.weight.device}")

ids = tok("The capital of France is", return_tensors="pt").input_ids
print(f"\ninput ids: {ids.tolist()}")

with torch.no_grad():
    # 1) text-only via input_ids
    out1 = llm(input_ids=ids.to(llm.model.embed_tokens.weight.device)).logits
    print(f"\n[1] input_ids path : max|logits|={out1.abs().max().item():.4f} "
          f"finite={torch.isfinite(out1).all().item()}")
    nxt = out1[0, -1].argmax().item()
    print(f"    next token: {tok.decode([nxt])!r}")

    # 2) same via inputs_embeds (embed first, then pass)
    te = llm.model.embed_tokens(ids.to(llm.model.embed_tokens.weight.device))
    out2 = llm(inputs_embeds=te).logits
    print(f"[2] inputs_embeds  : max|logits|={out2.abs().max().item():.4f} "
          f"finite={torch.isfinite(out2).all().item()}")
    if out2.abs().max().item() > 0:
        nxt2 = out2[0, -1].argmax().item()
        print(f"    next token: {tok.decode([nxt2])!r}")

    # 3) inputs_embeds cast to fp16 + moved to cuda:0 (exactly like training)
    te3 = te.to(torch.float16).to("cuda:0")
    out3 = llm(inputs_embeds=te3).logits
    print(f"[3] fp16+cuda:0    : max|logits|={out3.abs().max().item():.4f} "
          f"finite={torch.isfinite(out3).all().item()}")

    # 4) hidden states via model.model, then manual lm_head
    hs = llm.model(inputs_embeds=te3, use_cache=False).last_hidden_state
    print(f"[4] final hidden   : max|h|={hs.abs().max().item():.4f} dev={hs.device}")
    man = hs @ llm.lm_head.weight.T.to(hs.device).to(hs.dtype)
    print(f"    manual lm_head : max|logits|={man.abs().max().item():.4f}")
    diff = (out3.to(man.device) - man).abs().max().item()
    print(f"    |out3 - manual|={diff:.6f}")
