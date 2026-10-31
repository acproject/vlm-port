"""Validate fa_v100 attention integration on the real sharded 33B (fp16).

Compares the Volta-tuned FlashAttention kernels (registered as attention
implementation "fa_v100") against the SDPA baseline that test_sharded.py
already validated (max|logits|≈40.97, next=' Paris', 0.45s/iter).

Checks: next-token agreement, relative logits diff, autograd through the
flash path, fwd+bwd timing.

Run: ../venv-cu128/bin/python scripts/test_fa_v100.py
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

_PROJECT_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_PROJECT_SRC))

import torch
from transformers import LlamaForCausalLM, AutoConfig, AutoTokenizer

from modeling.sharded_llm import remove_accelerate_hooks, sharded_llm_forward
from modeling.fa_v100 import enable_fa_v100, ATTN_IMPL_NAME

torch.backends.cudnn.enabled = False
NANBEIGE_DIR = "/home/acproject/usb/transplant/Nanbeige4.3-33B"

tok = AutoTokenizer.from_pretrained(NANBEIGE_DIR)
cfg = AutoConfig.from_pretrained(NANBEIGE_DIR)
llm = LlamaForCausalLM.from_pretrained(
    NANBEIGE_DIR, config=cfg, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    device_map="auto", max_memory={0: "20GiB", 1: "27GiB", 2: "25GiB"},
)
llm.eval()
for p in llm.parameters():
    p.requires_grad = False
remove_accelerate_hooks(llm)
print(f"default _attn_implementation = {llm.config._attn_implementation}")

dev = llm.model.embed_tokens.weight.device
ids = tok("The capital of France is", return_tensors="pt").input_ids


def run_forward():
    te = llm.model.embed_tokens(ids.to(dev)).detach().requires_grad_(True)
    logits = sharded_llm_forward(llm, te)
    loss = logits.float().pow(2).mean()
    loss.backward()
    return logits, te.grad


# ── 1. SDPA baseline ──
with torch.no_grad():
    te0 = llm.model.embed_tokens(ids.to(dev))
    logits_sdpa = sharded_llm_forward(llm, te0)
nxt_sdpa = tok.decode([logits_sdpa[0, -1].argmax().item()])
print(f"\n[sdpa]  max|logits|={logits_sdpa.abs().max().item():.2f} next={nxt_sdpa!r} (expect ' Paris')")

# ── 2. Switch to fa_v100 and compare ──
assert enable_fa_v100(llm), "fa_v100 enable failed"
assert llm.model.layers[0].self_attn.config._attn_implementation == ATTN_IMPL_NAME

with torch.no_grad():
    logits_fa = sharded_llm_forward(llm, te0)
nxt_fa = tok.decode([logits_fa[0, -1].argmax().item()])
diff = (logits_fa.float() - logits_sdpa.float()).abs()
rel = diff.max().item() / logits_sdpa.abs().max().item()
cos = torch.nn.functional.cosine_similarity(
    logits_fa[0, -1].float(), logits_sdpa[0, -1].float(), dim=0).item()
print(f"[fa]    max|logits|={logits_fa.abs().max().item():.2f} next={nxt_fa!r}")
print(f"[cmp]   max|diff|={diff.max().item():.3f} rel={rel*100:.2f}% cos(last)={cos:.6f}")

ok_next = nxt_fa == nxt_sdpa and "Paris" in nxt_sdpa  # decode strips leading space
ok_rel = rel < 0.05
print(f"  next-token match: {ok_next}, rel diff <5%: {ok_rel}")

# ── 3. Autograd through the flash path ──
logits, g = run_forward()
print(f"[fa]    backward grad max={g.abs().max().item():.6f} finite={torch.isfinite(g).all().item()}")
ok_grad = torch.isfinite(g).all().item() and g.abs().max().item() > 0

# ── 4. Timing S=100 fwd+bwd: fa vs sdpa ──
def timing():
    gc.collect(); torch.cuda.empty_cache()
    S = 100
    te = torch.randn(1, S, cfg.hidden_size, device=dev, dtype=torch.float16, requires_grad=True)
    for _ in range(2):
        sharded_llm_forward(llm, te).float().pow(2).mean().backward()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        sharded_llm_forward(llm, te).float().pow(2).mean().backward()
    torch.cuda.synchronize()
    return (time.time() - t0) / 5

t_fa = timing()
llm.config._attn_implementation = "sdpa"
for m in llm.modules():
    cfg_ = getattr(m, "config", None)
    if cfg_ is not None and hasattr(cfg_, "_attn_implementation"):
        cfg_._attn_implementation = "sdpa"
t_sdpa = timing()
print(f"\n[time]  S=100 fwd+bwd: fa_v100={t_fa:.2f}s/iter  sdpa={t_sdpa:.2f}s/iter  "
      f"speedup={t_sdpa/t_fa:.2f}x")

all_ok = ok_next and ok_rel and ok_grad
print(f"\n{'FA_V100_VALIDATION_PASS' if all_ok else 'FA_V100_VALIDATION_FAIL'}")
raise SystemExit(0 if all_ok else 1)
