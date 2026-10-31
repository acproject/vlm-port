"""Pin down broken forward combos: attention_mask / position_ids / use_cache on
GPU-fp16-sharded vs CPU-bf16 reference.

Run: ../venv-cu128/bin/python scripts/debug_mask_matrix.py
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
print(f"ids: {ids.tolist()}\n")


def run(tag, model, dev, use_mask, use_pos, use_cache):
    kw = {}
    if use_mask:
        kw["attention_mask"] = torch.ones_like(ids).to(dev)
    if use_pos:
        kw["position_ids"] = torch.arange(ids.shape[1], device=dev).unsqueeze(0)
    kw["use_cache"] = use_cache
    with torch.no_grad():
        out = model(input_ids=ids.to(dev), **kw).logits
    nxt = out[0, -1].argmax().item()
    print(f"[{tag:24s}] max|logits|={out.abs().max().item():9.4f}  next={tok.decode([nxt])!r}")


print("=== GPU fp16 sharded (device_map=auto) ===")
llm = LlamaForCausalLM.from_pretrained(
    NANBEIGE_DIR, config=cfg, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    device_map="auto", max_memory={0: "22GiB", 1: "30GiB", 2: "30GiB"},
)
llm.eval()
dev = llm.model.embed_tokens.weight.device
print(f"attn_implementation: {llm.config._attn_implementation}\n")
run("no mask, no pos, cache=T", llm, dev, False, False, True)
run("no mask, no pos, cache=F", llm, dev, False, False, False)
run("mask=1,   no pos, cache=F", llm, dev, True, False, False)
run("no mask, pos,     cache=F", llm, dev, False, True, False)
run("mask=1,   pos,     cache=F", llm, dev, True, True, False)
run("mask=1,   pos,     cache=T", llm, dev, True, True, True)
del llm
torch.cuda.empty_cache()

print("\n=== CPU bf16 reference (known-good from PoC) ===")
llm_cpu = LlamaForCausalLM.from_pretrained(
    NANBEIGE_DIR, config=cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
)
llm_cpu.eval()
run("no mask, no pos, cache=T", llm_cpu, "cpu", False, False, True)
run("mask=1,   pos,     cache=F", llm_cpu, "cpu", True, True, False)
