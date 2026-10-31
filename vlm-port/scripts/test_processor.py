#!/usr/bin/env python
"""Processor-only test: image_grid_thw, visual tokens, special tokens.

No model weights loaded — just processor + tokenizer shapes.
Run: python test_processor.py
"""
import os, json
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from PIL import Image
from transformers import AutoProcessor

MODEL_DIR = "/home/acproject/usb/transplant/Qwen3.8-27B"
processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
tok = processor.tokenizer

print(f"processor: {type(processor).__name__}")
print(f"tokenizer: {type(tok).__name__}  vocab={tok.vocab_size}")

# 1. image sizes → grid_thw → visual_token count
print(f"\n{'='*70}\n1. IMAGE → image_grid_thw → visual_token_count\n{'='*70}")
for sz in [336, 512, 672, 768, 1024, 1280, 1568, 2048]:
    img = Image.new("RGB", (sz, sz), color=(100, 150, 200))
    inputs = processor(images=img, return_tensors="pt")
    grid = inputs.get("image_grid_thw")
    if grid is not None:
        t, h, w = grid[0].tolist()
        n_vis = h * w
        print(f"  {sz}x{sz} → grid_thw=[{t},{h},{w}] → visual_tokens={n_vis}")

# 2. Multi-image
print(f"\n{'='*70}\n2. MULTI-IMAGE → multiple grids\n{'='*70}")
imgs = [Image.new("RGB", (sz, sz), color=(i*50, 100, 200)) for i, sz in enumerate([672, 1024, 768])]
inputs = processor(images=imgs, return_tensors="pt")
print(f"  image_grid_thw shape: {inputs['image_grid_thw'].shape}")
print(f"  values: {inputs['image_grid_thw'].tolist()}")

# 3. Chat template rendering
print(f"\n{'='*70}\n3. CHAT TEMPLATE with image → token count\n{'='*70}")
from transformers import AutoTokenizer
nan_tok = AutoTokenizer.from_pretrained("/home/acproject/usb/transplant/Nanbeige4.3-33B")
print(f"  Nanbeige tokenizer: vocab={nan_tok.vocab_size}")

# Render Qwen-style chat with image  
messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "Describe this image."}
]}]
qwen_rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"  Qwen chat template output:\n{qwen_rendered}")

# Tokenize and count
qwen_ids = processor.tokenizer(qwen_rendered, return_tensors="pt").input_ids
print(f"  Qwen tokenized: {qwen_ids.shape[1]} tokens, ids[:20]={qwen_ids[0,:20].tolist()}")

# 4. Verify Nanbeige free vocab slots
print(f"\n{'='*70}\n4. NANBEIGE VOCAB FREE SLOTS\n{'='*70}")
ncfg = json.load(open("/home/acproject/usb/transplant/Nanbeige4.3-33B/config.json"))
print(f"  Nanbeige vocab_size={ncfg['vocab_size']}")
print(f"  Available ids for new vision tokens: 166107 - 166911 (805 slots)")

# Check what Nanbeige chat template would render for a multimodal message
# (it currently says "can't handle multimodal")
try:
    msg2 = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]}]
    nan_rendered = nan_tok.apply_chat_template(msg2, tokenize=False, add_generation_prompt=True)
    print(f"  Nanbeige chat template with image:\n{nan_rendered}")
except Exception as e:
    print(f"  Nanbeige chat template with image: {type(e).__name__}: {str(e)[:200]}")

# 5. Tokenize the vision special tokens
print(f"\n{'='*70}\n5. Qwen VISION SPECIAL TOKEN IDS\n{'='*70}")
for tok_str in ["<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>"]:
    ids = processor.tokenizer.encode(tok_str, add_special_tokens=False)
    print(f"  {tok_str} → ids={ids}")

# 6. Count processor output keys for single image
print(f"\n{'='*70}\n6. FULL PROCESSOR OUTPUT (single 768x768 image)\n{'='*70}")
img = Image.new("RGB", (768, 768))
inputs = processor(images=img, text="What is this?", return_tensors="pt")
for k, v in inputs.items():
    if hasattr(v, 'shape'):
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")
    else:
        print(f"  {k}: {v}")
# The actual token sequence
print(f"  input_ids first 40: {inputs.input_ids[0, :40].tolist()}")
print(f"  input_ids last 20: {inputs.input_ids[0, -20:].tolist()}")
# What are image_pad positions?
ipd = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
vs = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
ve = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
print(f"  <|vision_start|>= {vs}  <|vision_end|>= {ve}  <|image_pad|>= {ipd}")
positions = (inputs.input_ids[0] == ipd).nonzero().squeeze()
print(f"  image_pad token positions: count={len(positions)} first={positions[:5].tolist() if len(positions)>0 else 'none'}")

print("\nDONE.")
