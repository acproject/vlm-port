"""Native Qwen3.8-27B VLM on OUR val set — SINGLE GPU int8.

3-GPU device_map=auto bf16 produced garbage on this box (broken P2P — verified
2026-10-31: single-GPU int8 is fluent, 3-GPU accelerate path is garbage, our
own host-relay sharded forward is fine). Native eval therefore runs
single-GPU int8 (~27 GiB, zero cross-GPU transfer).

Question: does the ORIGINAL Qwen3.8-27B (its own vision tower + its own LLM,
the exact features our pipeline consumes as pooler_output) answer our
synthetic color/shape/count/OCR questions correctly?

  native ACC high → vision features carry the signal; our stage-1 plateau is
                    a projector-training problem (data-limited shortcut learning)
  native ACC low  → the synthetic images/questions are the problem; expanding
                    THIS kind of data won't help

Run: ../venv-cu128/bin/python scripts/eval_qwen_native.py [limit]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from PIL import Image
import torch
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

QWEN_DIR = "/home/acproject/usb/transplant/Qwen3.8-27B"


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    torch.backends.cudnn.enabled = False

    samples = [json.loads(l) for l in open("data/val_manifest.jsonl") if l.strip()][:limit]

    processor = AutoProcessor.from_pretrained(QWEN_DIR)
    t0 = time.time()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        QWEN_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map={"": 0},
    )
    model.eval()
    print(f"model loaded ({time.time()-t0:.0f}s), running {len(samples)} samples", flush=True)

    exact = contains = 0
    t0 = time.time()
    for i, s in enumerate(samples):
        img = Image.open(Path("data") / s["image"]).convert("RGB")
        msgs = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": s["question"]},
        ]}]
        try:
            text = processor.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True,
                                                 enable_thinking=False)
        except TypeError:  # template without thinking support
            text = processor.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True)
        inputs = processor(images=[img], text=text, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        gen = processor.batch_decode(
            out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        ans = s["answer"].strip()
        ok_e = gen == ans
        ok_c = ans in gen
        exact += ok_e
        contains += ok_c
        print(f"[{i+1:2d}/{len(samples)}] {'OK ' if ok_c else 'X  '} "
              f"q='{s['question'][:18]}' a='{ans}' gen='{gen[:40]}'", flush=True)

    print("=" * 60)
    print(f"NATIVE QWEN (int8, 1-GPU): exact={exact}/{len(samples)} "
          f"({exact/len(samples)*100:.0f}%)  contains={contains}/{len(samples)} "
          f"({contains/len(samples)*100:.0f}%)  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
