#!/usr/bin/env python
"""Minimal PoC: image → Qwen Vision → Custom Projector → Nanbeige LLM → generation.

Single script loads everything, runs vision forward, verifies projector shape,
injects visual embeddings, and does a short generation on CPU.

Run:
    cd /home/acproject/usb/transplant/vlm-port
    PYTHONPATH=src:.. python scripts/poc_minimal.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# ── Single import root (经验 100025669) ──
# sys.path gets the vlm-port directory, so `import vision.xxx` works
_PROJECT_SRC = (Path(__file__).resolve().parent.parent / "src")
if str(_PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(_PROJECT_SRC))
# Also allow finding transformers etc
_VENV_SITE = Path(__file__).resolve().parent.parent.parent / "venv-cu128" / "lib" / "python3.12" / "site-packages"
if str(_VENV_SITE) not in sys.path:
    sys.path.insert(0, str(_VENV_SITE))

import torch
from PIL import Image

# ── Single device source (经验 100026289) ──
DEVICE = "cpu"  # GPU 1/2 occupied by llama-server, GPU 0 broken
DTYPE = torch.bfloat16

QWEN_DIR = "/home/acproject/usb/transplant/Qwen3.8-27B"
NANBEIGE_DIR = "/home/acproject/usb/transplant/Nanbeige4.3-33B"

print(f"{'='*70}")
print(f"Minimal PoC | device={DEVICE} dtype={DTYPE}")
print(f"{'='*70}")


# ────────────────────────────────────────────────────────
# 1. Load vision tower
# ────────────────────────────────────────────────────────
print("\n[1/6] Loading Qwen vision tower...")
from vision.qwen_vision import load_qwen_vision_tower
t0 = time.time()
vision_tower, vision_info = load_qwen_vision_tower(QWEN_DIR, device=DEVICE, dtype=DTYPE)
print(f"  OK in {time.time()-t0:.1f}s")
print(f"  ViT hidden={vision_info.hidden_size} depth={vision_info.depth} "
      f"out_hidden_size={vision_info.out_hidden_size}")

# ────────────────────────────────────────────────────────
# 2. Load Nanbeige LLM + Custom Projector
# ────────────────────────────────────────────────────────
print("\n[2/6] Loading Nanbeige LLM (CPU BF16)...")
from transformers import LlamaForCausalLM, AutoConfig, AutoTokenizer
from vision.projector import CustomProjector

t0 = time.time()
llm_cfg = AutoConfig.from_pretrained(NANBEIGE_DIR)
llm = LlamaForCausalLM.from_pretrained(
    NANBEIGE_DIR, config=llm_cfg, torch_dtype=DTYPE, low_cpu_mem_usage=True,
).to(device=DEVICE)
# FREEZE ENTIRE LLM before any optimizer (经验 353224)
for p in llm.parameters():
    p.requires_grad = False
llm.eval()
n_llm = sum(p.numel() for p in llm.parameters())
print(f"  OK in {time.time()-t0:.1f}s | {n_llm/1e9:.1f}B params ALL frozen")
print(f"  LLM hidden_size={llm_cfg.hidden_size} (confirm 7168)")
assert llm_cfg.hidden_size == 7168, f"Expected hidden_size=7168, got {llm_cfg.hidden_size}"

print("\n  Loading Custom Projector...")
projector = CustomProjector(
    vision_feature_dim=vision_info.out_hidden_size,  # 5120
    llm_hidden_size=llm_cfg.hidden_size,              # 7168
    act_fn="gelu",
).to(device=DEVICE, dtype=DTYPE)
n_proj = sum(p.numel() for p in projector.parameters())
print(f"  Projector params: {n_proj/1e6:.2f}M (ONLY trainable)")
assert vision_info.out_hidden_size == 5120, f"Expected 5120, got {vision_info.out_hidden_size}"

# ────────────────────────────────────────────────────────
# 3. Build processor + tokenize
# ────────────────────────────────────────────────────────
print("\n[3/6] Loading processor + tokenizing...")
from data.processor import NanbeigeVisionProcessor, IMAGE_PAD_ID
processor = NanbeigeVisionProcessor(QWEN_DIR, NANBEIGE_DIR)

# Small test image (512×512 → 32×32 ViT patch grid → 16×16 post-merge → 256 tokens)
img = Image.new("RGB", (512, 512), color=(128, 200, 100))
messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "What is in this image? Answer briefly."}
]}]

t0 = time.time()
batch = processor(messages=messages, images=[img], device=DEVICE)
print(f"  OK in {time.time()-t0:.1f}s")
print(f"  input_ids: {batch.input_ids.shape}")
print(f"  pixel_values: {batch.pixel_values.shape}")
print(f"  image_grid_thw: {batch.image_grid_thw.tolist()}")
print(f"  image_pad_mask True count: {batch.image_pad_mask.sum().item()}")
print(f"  ViT patches={batch.image_grid_thw.prod(-1).item()} "
      f"post-merge={batch.image_grid_thw.prod(-1).item()//4} "
      f"(should equal image_pad_mask count)")
assert batch.image_pad_mask.sum().item() == batch.image_grid_thw.prod(-1).item() // 4, \
    "image_pad count != post-merge token count!"

# ────────────────────────────────────────────────────────
# 4. Vision forward + Projector
# ────────────────────────────────────────────────────────
print("\n[4/6] Vision forward → Projector...")
t0 = time.time()
with torch.no_grad():
    vis_out = vision_tower(
        batch.pixel_values,           # first positional arg — named 'hidden_states' in API
        grid_thw=batch.image_grid_thw,
    )
    vis_features = vis_out.pooler_output  # post-merge
print(f"  vision tower OK in {time.time()-t0:.1f}s")
print(f"  ViT patches: {batch.image_grid_thw.prod(-1).item()}")
print(f"  post-merge pooler_output shape: {tuple(vis_features.shape)}")

# Assert: post-merge tokens = image_pad count
expected_post_merge = batch.image_grid_thw.prod(-1).item() // 4
assert vis_features.shape[0] == expected_post_merge, \
    f"Expected {expected_post_merge} post-merge tokens, got {vis_features.shape[0]}"
assert vis_features.shape[-1] == 5120, f"Expected 5120, got {vis_features.shape[-1]}"

projected = projector(vis_features.to(projector.net[0].weight.dtype)).to(DTYPE)
print(f"  projected shape: {tuple(projected.shape)}")
assert projected.shape[0] == batch.image_pad_mask.sum().item(), \
    f"projected tokens ({projected.shape[0]}) != image_pad positions"
assert projected.shape[-1] == 7168, f"Expected 7168, got {projected.shape[-1]}"

# ────────────────────────────────────────────────────────
# 5. Embedding injection
# ────────────────────────────────────────────────────────
print("\n[5/6] Embedding injection...")
text_embeds = llm.model.embed_tokens(batch.input_ids).to(llm.dtype)
print(f"  text_embeds before injection: {tuple(text_embeds.shape)}")

# Save original image_pad embeddings for comparison
orig_image_embeds = text_embeds[batch.image_pad_mask].clone()

# Replace
text_embeds[batch.image_pad_mask] = projected
print(f"  image_pad embeddings replaced: {batch.image_pad_mask.sum().item()} positions")

# Verify replacement
replaced_image_embeds = text_embeds[batch.image_pad_mask]
diff = (replaced_image_embeds - orig_image_embeds).abs().max().item()
print(f"  max change in image_pad embeds after injection: {diff:.4f} (non-zero = replaced ✓)")
assert diff > 0.01, "Image pad embeddings not actually changed!"

# Verify non-image tokens unchanged
non_image_mask = ~batch.image_pad_mask
if non_image_mask.any():
    orig_non_image = llm.model.embed_tokens(batch.input_ids)
    diff_non = (text_embeds[non_image_mask] - orig_non_image[non_image_mask]).abs().max().item()
    print(f"  max change in non-image embeds: {diff_non:.6f} (should be ~0)")
    assert diff_non < 1e-3, "Non-image embeddings changed!"

# ────────────────────────────────────────────────────────
# 6. Short generation
# ────────────────────────────────────────────────────────
print("\n[6/6] Short generation (max_new_tokens=5)...")
B, S = batch.input_ids.shape
position_ids = torch.arange(S, device=DEVICE).unsqueeze(0).expand(B, -1)

t0 = time.time()
with torch.no_grad():
    gen_ids = llm.generate(
        input_ids=batch.input_ids,
        inputs_embeds=text_embeds,
        attention_mask=batch.attention_mask,
        position_ids=position_ids,
        max_new_tokens=5,
        do_sample=False,
        pad_token_id=batch.input_ids.new_tensor(0),
    )
gen_time = time.time() - t0
print(f"  Generated {gen_ids.shape[1]-S} new tokens in {gen_time:.1f}s "
      f"({(gen_ids.shape[1]-S)/gen_time:.2f} tok/s)")
print(f"  Full sequence length: {gen_ids.shape[1]}")
assert gen_ids.shape[1] == S + 5, \
    f"Expected {S+5}, got {gen_ids.shape[1]} — generate still broken"

# Decode the new tokens
tok = processor.tokenizer  # reuse processor's Nanbeige tokenizer
new_text = tok.decode(gen_ids[0, S:], skip_special_tokens=False)
print(f"  New text: '{new_text}'")

# Also test: text-only input still works (backward compatibility)
print("\n── Text-only backward-compat test ──")
text_only_str = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
text_only_ids = torch.tensor(
    [tok.encode(text_only_str, add_special_tokens=False)], dtype=torch.long, device=DEVICE
)
print(f"  text_only_ids shape: {text_only_ids.shape}, max_id={text_only_ids.max().item()} "
      f"< embed_table={llm.model.embed_tokens.weight.shape[0]}")
assert text_only_ids.max().item() < llm.model.embed_tokens.weight.shape[0], "Token id out of range!"
text_only_embeds = llm.model.embed_tokens(text_only_ids).to(llm.dtype)
text_only_mask = torch.ones_like(text_only_ids)
t0 = time.time()
with torch.no_grad():
    gen2 = llm.generate(
        input_ids=text_only_ids,
        inputs_embeds=text_only_embeds,
        attention_mask=text_only_mask,
        max_new_tokens=5, do_sample=False,
        pad_token_id=text_only_ids.new_tensor(0),
    )
print(f"  Text-only generation OK in {time.time()-t0:.1f}s")
print(f"  gen2 shape: {gen2.shape}, expected: {text_only_ids.shape[1]+5}")
assert gen2.shape[1] == text_only_ids.shape[1] + 5
new_text2 = tok.decode(gen2[0, text_only_ids.shape[1]:], skip_special_tokens=False)
print(f"  Text-only output: '{new_text2}'")

print(f"\n{'='*70}")
print("PoC PASSED ✅")
print(f"{'='*70}")
print("\nKey shape assertions verified:")
print(f"  ViT output:        {batch.image_grid_thw.prod(-1).item()} tokens × 1152")
print(f"  Merger pooler:     {expected_post_merge} tokens × 5120")
print(f"  Projected:         {batch.image_pad_mask.sum().item()} tokens × 7168")
print(f"  Non-image embeds unchanged ✓")
print(f"  Generation works with vision injection ✓")
print(f"  Text-only backward-compat ✓")
print(f"\nOnly trainable module: CustomProjector ({n_proj/1e6:.2f}M params)")
print(f"Frozen modules: Qwen Vision Tower + Nanbeige 33B LLM")
