#!/usr/bin/env python
"""Correctly tokenize multimodal chat messages with Qwen processor.

Shows exactly how <|image_pad|> expands into N token slots,
and what mm_token_type_ids looks like.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
from PIL import Image
from transformers import AutoProcessor

MODEL_DIR = "/home/acproject/usb/transplant/Qwen3.8-27B"
processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
tok = processor.tokenizer

# Build chat messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "What is in this image? Answer briefly."}
    ]},
]

# Use chat template to get text with placeholders
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("Chat template rendered:")
print(text[:300])

# Now call the multimodal processor with image
img = Image.new("RGB", (768, 768), color=(100, 150, 200))
# The correct Qwen3-VL way
inputs = processor(
    text=[text],
    images=[img],
    return_tensors="pt",
    padding=True,
)

print(f"\nProcessor output keys: {list(inputs.keys())}")
print(f"input_ids shape: {inputs.input_ids.shape}")
print(f"image_grid_thw: {inputs.image_grid_thw.tolist()}")
if hasattr(inputs, 'mm_token_type_ids'):
    print(f"mm_token_type_ids shape: {inputs.mm_token_type_ids.shape}")

# Count image_pad positions
vs_id = tok.convert_tokens_to_ids("<|vision_start|>")
ve_id = tok.convert_tokens_to_ids("<|vision_end|>")
ipd_id = tok.convert_tokens_to_ids("<|image_pad|>")

ids = inputs.input_ids[0].tolist()
print(f"\ninput_ids[0][:80]: {ids[:80]}")
print(f"input_ids[0][-40:]: {ids[-40:]}")

# Find all image_pad positions
ipd_positions = [i for i, x in enumerate(ids) if x == ipd_id]
vs_positions = [i for i, x in enumerate(ids) if x == vs_id]
ve_positions = [i for i, x in enumerate(ids) if x == ve_id]
print(f"\n<|vision_start|> at positions: {vs_positions}")
print(f"<|vision_end|> at positions: {ve_positions}")
print(f"<|image_pad|> at positions: count={len(ipd_positions)}, first 5={ipd_positions[:5] if ipd_positions else 'none'}")

if hasattr(inputs, 'mm_token_type_ids'):
    mmt = inputs.mm_token_type_ids[0].tolist()
    print(f"\nmm_token_type_ids[0] unique values: {set(mmt)}")
    if 1 in mmt:
        img_positions = [i for i, x in enumerate(mmt) if x == 1]
        print(f"  image type=1 positions: count={len(img_positions)}, span=[{img_positions[0]}, {img_positions[-1]}]")

# Confirm visual token count = grid_thw.prod()/4
grid = inputs.image_grid_thw[0].tolist()
expected = grid[0] * grid[1] * grid[2]  # T*H*W already after merge
print(f"\ngrid_thw={grid}, expected visual tokens={grid[0]}*{grid[1]}*{grid[2]}={expected}")
print(f"actual image_pad tokens found={len(ipd_positions)}")

# Also show what tokens are in the image_pad span (what are they?)
if len(ipd_positions) > 0:
    start = ipd_positions[0] - 1
    end = ipd_positions[-1] + 2
    print(f"\nContext around image_pad span [{start}:{end}]:")
    for i in range(start, end):
        tid = ids[i]
        tstr = tok.decode([tid])[:30] if i >= 0 and i < len(ids) else "?"
        tag = ""
        if i == ipd_positions[0]: tag = " ← first image_pad"
        elif i == ipd_positions[-1]: tag = " ← last image_pad"
        elif i in vs_positions: tag = " ← vision_start"
        elif i in ve_positions: tag = " ← vision_end"
        print(f"  [{i:4d}] id={tid:6d} '{tstr}'{tag}")

print("\nDONE.")
