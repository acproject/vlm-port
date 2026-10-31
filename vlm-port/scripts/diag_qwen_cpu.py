"""CPU control: is native Qwen3.8-27B itself sane (no P2P, no bf16-Volta)?

Prefill-only first-token check on CPU fp32:
  "法国的首都是" → expect 巴 / 巴黎
  "The capital of France is" → expect Paris

If CPU is correct → GPU garbage is P2P transfer or bf16 numeric issue.
If CPU is also garbage → weight/implementation problem (transformers Qwen3.5).

Run: ../venv-cu128/bin/python scripts/diag_qwen_cpu.py   (~2-5 min/forward)
"""
from __future__ import annotations

import time

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

QWEN_DIR = "/home/acproject/usb/transplant/Qwen3.8-27B"


@torch.no_grad()
def first_token(model, tok, prompt: str):
    ids = tok(prompt, return_tensors="pt").input_ids
    t0 = time.time()
    out = model(input_ids=ids)
    logits = out.logits[0, -1]
    top = torch.topk(logits, 5)
    toks = [tok.decode([i]) for i in top.indices.tolist()]
    print(f"  '{prompt}' -> top5={list(zip(toks, [round(v,2) for v in top.values.tolist()]))} "
          f"({time.time()-t0:.0f}s)", flush=True)


def main():
    tok = AutoTokenizer.from_pretrained(QWEN_DIR)
    t0 = time.time()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        QWEN_DIR, torch_dtype=torch.float32, low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"loaded fp32 CPU ({time.time()-t0:.0f}s)", flush=True)

    first_token(model, tok, "法国的首都是")
    first_token(model, tok, "The capital of France is")


if __name__ == "__main__":
    main()
