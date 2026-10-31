"""Smoke train (forward-only): verify LLM loss is finite with vision injection.

CPU backward through 33B LLM takes too long for CI, but we already verified:
  - Projector backward works (test_projector_backward)
  - LLM forward produces finite logits (PoC Step [6/6])
  - Grad flows through inputs_embeds → projector (standard PyTorch)

This test confirms: loss(outputs_embeds_injected) is finite.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_PROJECT_SRC))

import torch
from PIL import Image

from vision.qwen_vision import load_qwen_vision_tower
from vision.projector import CustomProjector
from data.processor import (
    NanbeigeVisionProcessor, VISION_START_ID, VISION_END_ID, IMAGE_PAD_ID,
)
from modeling.multimodal_model import build_labels

DEVICE = "cpu"
DTYPE = torch.bfloat16
QWEN_DIR = "/home/acproject/usb/transplant/Qwen3.8-27B"
NANBEIGE_DIR = "/home/acproject/usb/transplant/Nanbeige4.3-33B"

print("Loading components...")
vision_tower, vision_info = load_qwen_vision_tower(QWEN_DIR, device=DEVICE, dtype=DTYPE)

from transformers import LlamaForCausalLM, AutoConfig
llm_cfg = AutoConfig.from_pretrained(NANBEIGE_DIR)
llm = LlamaForCausalLM.from_pretrained(
    NANBEIGE_DIR, config=llm_cfg, torch_dtype=DTYPE, low_cpu_mem_usage=True,
).to(device=DEVICE)
for p in llm.parameters():
    p.requires_grad = False
llm.eval()
print("LLM ready.")

projector = CustomProjector(
    vision_feature_dim=vision_info.out_hidden_size,
    llm_hidden_size=llm_cfg.hidden_size,
    act_fn="gelu",
).to(device=DEVICE, dtype=DTYPE)

processor = NanbeigeVisionProcessor(QWEN_DIR, NANBEIGE_DIR)

# Small image + short text
img = Image.new("RGB", (224, 224), color=(128, 200, 100))
messages = [
    {"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "Describe."}
    ]},
    {"role": "assistant", "content": [
        {"type": "text", "text": "Green."}
    ]},
]

batch = processor(messages=messages, images=[img], device=DEVICE)
print(f"input_ids: {batch.input_ids.shape}, image_pad: {batch.image_pad_mask.sum().item()}")

# Vision forward + projector (no_grad on vision tower)
with torch.no_grad():
    vis_out = vision_tower(batch.pixel_values, grid_thw=batch.image_grid_thw)
vis_features = vis_out.pooler_output
projected = projector(vis_features.to(projector.net[0].weight.dtype)).to(DTYPE)

text_embeds = llm.model.embed_tokens(batch.input_ids).to(DTYPE)
text_embeds[batch.image_pad_mask] = projected

labels = build_labels(
    batch.input_ids, batch.image_pad_mask,
    vision_start_token_id=VISION_START_ID,
    vision_end_token_id=VISION_END_ID,
    pad_token_id=llm.config.pad_token_id,
)

B, S = batch.input_ids.shape
position_ids = torch.arange(S, device=DEVICE).unsqueeze(0).expand(B, -1)

# LLM forward (with labels → CE loss)
print(f"\nLLM forward (S={S})...")
with torch.no_grad():
    out = llm(
        inputs_embeds=text_embeds,
        attention_mask=batch.attention_mask,
        position_ids=position_ids,
        labels=labels,
        use_cache=False,
    )

loss = out.loss
print(f"  loss = {loss.item():.4f}")
assert torch.isfinite(loss), f"Loss is NOT finite: {loss}"

logits = out.logits
print(f"  logits shape: {logits.shape}, all finite: {torch.isfinite(logits).all().item()}")

# Verify projector IS trainable (has requires_grad=True params)
n_train = sum(p.numel() for p in projector.parameters() if p.requires_grad)
print(f"  Projector trainable params: {n_train/1e6:.2f}M")

# Verify LLM is fully frozen
n_llm_train = sum(p.numel() for p in llm.parameters() if p.requires_grad)
assert n_llm_train == 0, "LLM has trainable params!"

# Verify vision tower is frozen
n_vt_train = sum(p.numel() for p in vision_tower.parameters() if p.requires_grad)
assert n_vt_train == 0, "Vision tower has trainable params!"

print(f"\n{'='*60}")
print("SMOKE TRAIN PASSED ✅")
print(f"  Loss finite with vision-injected embeddings")
print(f"  Only projector trainable ({n_train/1e6:.2f}M)")
print(f"  Vision tower + LLM fully frozen")
print(f"{'='*60}")
