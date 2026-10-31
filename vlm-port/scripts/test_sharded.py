"""Validate sharded_llm_forward: correctness ('Paris'), autograd, timing.

Run: ../venv-cu128/bin/python scripts/test_sharded.py [bfloat16|float16]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_PROJECT_SRC))

import torch
from transformers import LlamaForCausalLM, AutoConfig, AutoTokenizer

from modeling.sharded_llm import remove_accelerate_hooks, sharded_llm_forward

torch.backends.cudnn.enabled = False
NANBEIGE_DIR = "/home/acproject/usb/transplant/Nanbeige4.3-33B"

dtype_name = sys.argv[1] if len(sys.argv) > 1 else "bfloat16"
dtype = getattr(torch, dtype_name)

tok = AutoTokenizer.from_pretrained(NANBEIGE_DIR)
cfg = AutoConfig.from_pretrained(NANBEIGE_DIR)
llm = LlamaForCausalLM.from_pretrained(
    NANBEIGE_DIR, config=cfg, torch_dtype=dtype, low_cpu_mem_usage=True,
    device_map="auto", max_memory={0: "20GiB", 1: "27GiB", 2: "25GiB"},
)
llm.eval()
for p in llm.parameters():
    p.requires_grad = False   # freeze — else backward computes 33B weight grads → OOM
remove_accelerate_hooks(llm)

import gc
from collections import Counter
split = Counter(str(v) for v in llm.hf_device_map.values())
print(f"device_map split: {dict(split)}")


def gpu_report(tag):
    llm_mem = {i: torch.cuda.memory_allocated(i) / 2**30 for i in range(3)}
    print(f"  [{tag}] GPU alloc: " + " ".join(f"cuda:{i}={m:.1f}GiB" for i, m in llm_mem.items()))


gpu_report("after load")

ids = tok("The capital of France is", return_tensors="pt").input_ids
dev = llm.model.embed_tokens.weight.device

with torch.no_grad():
    te = llm.model.embed_tokens(ids.to(dev))
    logits = sharded_llm_forward(llm, te)
    nxt = logits[0, -1].argmax().item()
    print(f"\n[{dtype_name}] max|logits|={logits.abs().max().item():.2f} "
          f"next={tok.decode([nxt])!r}  (expect ' Paris')")

# autograd test: grad flows through the whole sharded graph to inputs_embeds
te = llm.model.embed_tokens(ids.to(dev)).detach().requires_grad_(True)
logits = sharded_llm_forward(llm, te)
loss = logits.float().pow(2).mean()
loss.backward()
g = te.grad
print(f"[{dtype_name}] backward: grad max={g.abs().max().item():.6f} "
      f"finite={torch.isfinite(g).all().item()}")

# timing: S=100 forward+backward
gc.collect(); torch.cuda.empty_cache()
gpu_report("before timing")
S = 100
te = torch.randn(1, S, cfg.hidden_size, device=dev, dtype=dtype, requires_grad=True)
for _ in range(2):  # warmup
    logits = sharded_llm_forward(llm, te)
    logits.float().pow(2).mean().backward()
torch.cuda.synchronize()
t0 = time.time()
N = 5
for _ in range(N):
    logits = sharded_llm_forward(llm, te)
    logits.float().pow(2).mean().backward()
torch.cuda.synchronize()
dt = (time.time() - t0) / N
print(f"[{dtype_name}] S={S} fwd+bwd: {dt:.2f}s/iter")
