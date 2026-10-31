#!/usr/bin/env python
"""Final deep checks: Nanbeige token ID collision + multi-image + position strategy."""
import os, json
os.environ["HF_HUB_OFFLINE"] = "1"
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

# ── 1. Nanbeige token ID collision safety ────────────
print("="*70)
print("1. NANBEIGE VISION TOKEN IDS: collision safety")
print("="*70)
tok = AutoTokenizer.from_pretrained("/home/acproject/usb/transplant/Nanbeige4.3-33B")

# Proposed vision special tokens (zero vocab expand, use free slots 166107-166911)
proposed = {
    "<|vision_start|>": 166107,
    "<|vision_end|>":   166108,
    "<|image_pad|>":    166109,
}

# Check if these IDs are already used for meaningful tokens (not padding)
with open("/home/acproject/usb/transplant/Nanbeige4.3-33B/tokenizer.json") as f:
    tj = json.load(f)
base_vocab = tj["model"]["vocab"]
inv_base = {v: k for k, v in base_vocab.items()}
added = tj.get("added_tokens", [])
inv_added = {t["id"]: t["content"] for t in added}

for tok_str, tid in proposed.items():
    in_base = tid in inv_base
    in_added = tid in inv_added
    print(f"  {tok_str} = {tid}")
    print(f"    in base vocab? {'YES ('+inv_base[tid]+')' if in_base else 'no'}")
    print(f"    in added_tokens? {'YES ('+inv_added[tid]+')' if in_added else 'no'}")
    if in_base:
        print(f"    ⚠️  COLLISION! base vocab has '{inv_base[tid]}' at id {tid}")
    if not in_base and not in_added:
        print(f"    ✅  CLEAN — free slot, safe to claim")

# What does 166107 actually decode to currently?
test_tok = tok.decode([166107])
print(f"\n  Current decode of id 166107: '{test_tok}'")

# ── 2. Check if 166107-166911 have any real text tokens ──
print(f"\n  Base vocab max id: {max(inv_base.keys())}")
print(f"  Used added_tokens ids: {sorted(inv_added.keys())}")
# Are there any base vocab tokens above 166106?
high_base = [i for i in inv_base if i >= 166107]
print(f"  Base vocab tokens at id >= 166107: {len(high_base)}")
# What about below 166100?
low_base = [i for i in inv_base if i < 166100]
print(f"  Base vocab tokens at id < 166100: {len(low_base)}")
# Check contiguous below 166100
gaps_low = [i for i in range(166100) if i not in inv_base]
print(f"  Gaps below 166100: {len(gaps_low)}")

# ── 3. Multi-image Qwen processor test ────────────────
print(f"\n{'='*70}\n2. MULTI-IMAGE QWEN PROCESSOR\n{'='*70}")
processor = AutoProcessor.from_pretrained("/home/acproject/usb/transplant/Qwen3.8-27B", trust_remote_code=True)
messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "Compare these images."},
    {"type": "image"},
]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"Chat template:\n{text}")

imgs = [Image.new("RGB", (768, 768), (100,150,200)), Image.new("RGB", (1024, 1024), (200,100,150))]
inputs = processor(text=[text], images=imgs, return_tensors="pt")
print(f"\nProcessor output:")
for k, v in inputs.items():
    if hasattr(v, 'shape'):
        print(f"  {k}: {v.shape}")

ids = inputs.input_ids[0].tolist()
vs_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
ve_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
ipd_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

vs_pos = [i for i, x in enumerate(ids) if x == vs_id]
ve_pos = [i for i, x in enumerate(ids) if x == ve_id]
ipd_count = sum(1 for x in ids if x == ipd_id)
print(f"\n  image_grid_thw: {inputs.image_grid_thw.tolist()}")
print(f"  <|vision_start|> positions: {vs_pos}")
print(f"  <|vision_end|> positions: {ve_pos}")
print(f"  total <|image_pad|>: {ipd_count}")

# Verify: each image_pad count = grid_thw.prod()/4
for g in inputs.image_grid_thw.tolist():
    expected = g[0] * g[1] * g[2] // 4
    print(f"  grid {g} → expected post-merge tokens = {expected}")

# ── 4. Position strategy for Nanbeige ────────────────
print(f"\n{'='*70}\n3. POSITION STRATEGY FOR NANBEIGE (Llama 1D RoPE)\n{'='*70}")
print("  Qwen: 3D position_ids (3, B, S) = [T_grid, H_grid_post_merge, W_grid_post_merge]")
print("  Nanbeige: 1D position_ids (B, S) = sequential 0,1,2,...")
print()
print("  Our strategy (LLaVA-style):")
print("    text tokens → sequential pos: 0..N_text_before")
print("    vision tokens → sequential pos: N_text_before+1 .. N_text_before+N_vis")
print("    more text → continues: N_text_before+N_vis+1 ...")
print()
print("  Why this works:")
print("    - Spatial info is already in ViT pos_embed + merger (5120-dim features)")
print("    - Nanbeige's 1D RoPE encodes sequence order, not spatial layout")
print("    - We don't need MRoPE because we don't do video or high-res split")
print("    - The projector's job is to align visual features to Nanbeige's sequential space")
print()
print("  Caveat for future: if we want video / high-res split, need 3D→1D mapping")
print("  (e.g., flatten T×H×W into 1D, or implement partial 3D→offset mapping)")

print("\nDONE.")
