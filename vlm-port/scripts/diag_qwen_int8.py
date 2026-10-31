"""Single-GPU int8 control for native Qwen3.8-27B.

3-GPU device_map=auto bf16 generated garbage (text-only too). CPU is impossible
(fla Triton kernels are GPU-only). Single-GPU int8 removes ALL cross-GPU
transfers — the only variable left vs the 3-GPU run:

  int8 single-GPU clean  → garbage was cross-GPU transfer (broken P2P via
                           accelerate hooks) → need host-relay sharded forward
  int8 single-GPU garbage → numeric/implementation issue (fla/bf16 on V100)

Run: ../venv-cu128/bin/python scripts/diag_qwen_int8.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image
import torch
from transformers import AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

QWEN_DIR = "/home/acproject/usb/transplant/Qwen3.8-27B"


@torch.no_grad()
def run(model, tok, prompt: str):
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda:0")
    logits = model(input_ids=ids).logits[0, -1]
    top = torch.topk(logits.float(), 5)
    toks = [tok.decode([i]) for i in top.indices.tolist()]
    print(f"  prefill '{prompt}' -> top5={list(zip(toks, [round(v,2) for v in top.values.tolist()]))}",
          flush=True)
    gen = model.generate(input_ids=ids, max_new_tokens=12, do_sample=False)
    print(f"  gen: '{tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)}'", flush=True)


def main():
    torch.backends.cudnn.enabled = False
    tok = AutoTokenizer.from_pretrained(QWEN_DIR)
    t0 = time.time()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        QWEN_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map={"": 0},
    )
    model.eval()
    print(f"loaded int8 single-GPU ({time.time()-t0:.0f}s)", flush=True)
    print(f"mem: {torch.cuda.memory_allocated(0)/2**30:.1f} GiB", flush=True)

    run(model, tok, "法国的首都是")
    run(model, tok, "The capital of France is")

    # one val image with the Qwen processor (full native pipeline, single GPU)
    processor = AutoProcessor.from_pretrained(QWEN_DIR)
    samples = [json.loads(l) for l in open("data/val_manifest.jsonl") if l.strip()]
    s = samples[7]  # '灰色' color question
    img = Image.open(Path("data") / s["image"]).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": s["question"]}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[img], text=text, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=12, do_sample=False)
    gen = processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                 skip_special_tokens=True)[0].strip()
    print(f"  [img] a='{s['answer']}' gen='{gen[:40]}'", flush=True)


if __name__ == "__main__":
    main()
