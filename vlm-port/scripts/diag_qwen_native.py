"""Isolate WHY native Qwen3.8-27B generated garbage on our val set.

Controls on ONE loaded model (device_map=auto, bf16):
  1. plain-text generation (no image): "法国的首都是哪里？" → expect 巴黎
     garbage → cross-GPU layer transfer (known broken P2P) or bf16 numeric issue
     clean   → problem is in the vision path (image token injection)
  2. same text with image attached (a real photo-like val image)

Run: ../venv-cu128/bin/python scripts/diag_qwen_native.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image
import torch
from transformers import AutoProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

QWEN_DIR = "/home/acproject/usb/transplant/Qwen3.8-27B"


def gen(model, processor, text: str, img: Image.Image | None, tag: str):
    msgs = [{"role": "user", "content": ([{"type": "image"}] if img else []) + [
        {"type": "text", "text": text}]}]
    prompt = processor.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
    kw = {"images": [img]} if img else {}
    inputs = processor(text=prompt, return_tensors="pt", **kw).to("cuda:0")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    gen_text = processor.batch_decode(
        out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    n_in = inputs.input_ids.shape[1]
    print(f"[{tag}] S={n_in} gen='{gen_text[:60]}'", flush=True)
    return gen_text


def main():
    torch.backends.cudnn.enabled = False
    processor = AutoProcessor.from_pretrained(QWEN_DIR)
    t0 = time.time()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        QWEN_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map="auto", max_memory={0: "26GiB", 1: "30GiB", 2: "30GiB"},
    )
    model.eval()
    print(f"loaded ({time.time()-t0:.0f}s)", flush=True)

    # shard layout
    from collections import Counter
    print("device_map sample:", dict(Counter(str(v) for v in model.hf_device_map.values())))

    # 1. plain text (no image)
    gen(model, processor, "法国的首都是哪里？直接回答城市名。", None, "text-1")
    gen(model, processor, "1加1等于几？直接回答数字。", None, "text-2")

    # 2. with image (val samples: purple blobs / WORLD text)
    samples = [json.loads(l) for l in open("data/val_manifest.jsonl") if l.strip()]
    for s in [samples[1], samples[7]]:
        img = Image.open(Path("data") / s["image"]).convert("RGB")
        gen(model, processor, s["question"], img, f"img a='{s['answer']}'")


if __name__ == "__main__":
    main()
