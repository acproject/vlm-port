#!/usr/bin/env python
"""Run minimal Qwen vision forward + processor test.

Loads Qwen3.8-27B vision tower only (not the 27B text model),
feeds a dummy image, and prints every intermediate tensor shape.
Then tests the processor with the same image to get image_grid_thw.
"""
import os, sys, json
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from PIL import Image
from pathlib import Path

MODEL_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/acproject/usb/transplant/Qwen3.8-27B")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")

# ── 1. Load processor ──────────────────────────────
from transformers import AutoProcessor, AutoModel
print(f"\n{'='*70}\n1. LOADING PROCESSOR\n{'='*70}")
processor = AutoProcessor.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
print(f"processor: {type(processor).__name__}")
print(f"image_processor: {type(processor.image_processor).__name__}")
ip = processor.image_processor
print(f"  patch_size={ip.patch_size} merge_size={ip.merge_size}")
print(f"  size={ip.size}")
attrs = [a for a in dir(ip) if not a.startswith('_') and any(k in a for k in ['pixel','patch','merge','size','pad','resolution'])]
for a in attrs:
    if a != 'size':
        print(f"  {a}={getattr(ip, a)}")

# ── 2. Load vision-only model (partial load) ────────
print(f"\n{'='*70}\n2. LOADING VISION TOWER ONLY\n{'='*70}")

# Strategy: load the full model config, then construct a minimal vision-only module
# by directly loading model.visual.* weights from the safetensors
import json as _json
cfg = _json.load(open(MODEL_DIR / "config.json"))
vc = cfg["vision_config"]

# Use transformers Qwen3VLVisionModel if available
try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3VisionModel, Qwen3Config
    print("Found Qwen3VisionModel in transformers")
except ImportError:
    from transformers import AutoModel
    # load the whole conditional gen model then extract .visual
    print("Loading full model then extracting .visual...")

# Try AutoModel.from_pretrained with trust_remote_code=True
# but only load vision weights
from safetensors import safe_open
index = _json.load(open(MODEL_DIR / "model.safetensors.index.json"))
wm = index["weight_map"]

# Build a state dict for vision tower only
vision_sd = {}
for k, fname in wm.items():
    if k.startswith("model.visual."):
        if fname not in vision_sd:
            vision_sd[fname] = []
        vision_sd[fname].append(k)

# Load vision tensors
for fname, keys in vision_sd.items():
    with safe_open(MODEL_DIR / fname, framework="pt", device=device) as f:
        for k in keys:
            new_k = k.replace("model.visual.", "")
            vision_sd[fname + ":" + k] = f.get_tensor(k)

# Flatten: rebuild a single dict
vision_state = {}
for fname, keys in vision_sd.items():
    if ":" not in fname:  # skip list entries
        continue
    k = fname.split(":", 1)[1]
    vision_state[k] = vision_sd[fname]

print(f"Extracted {len(vision_state)} vision tensors")

# Now load the Qwen3_5 vision module
# We need to build it from config
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3Config, Qwen3VisionModel, Qwen3VisionConfig
)
vconfig = Qwen3VisionConfig(**vc)
print(f"vision config: hidden_size={vconfig.hidden_size}, depth={vconfig.depth}, out_hidden_size={vconfig.out_hidden_size}")

vision_model = Qwen3VisionModel(vconfig).to(device).to(torch.bfloat16)
# Strip "model.visual." prefix from keys to match the module
stripped = {k.replace("visual.", ""): v for k, v in vision_state.items()}
# Check if keys match
model_keys = set(vision_model.state_dict().keys())
state_keys = set(stripped.keys())
missing = model_keys - state_keys
extra = state_keys - model_keys
print(f"model state_dict keys: {len(model_keys)}")
print(f"state keys after strip: {len(state_keys)}")
print(f"missing: {missing}")
print(f"extra: {extra}")

vision_model.load_state_dict(stripped, strict=True)
vision_model.eval()
print("Vision tower loaded OK!")

# ── 3. Forward pass with dummy image ─────────────────
print(f"\n{'='*70}\n3. VISION FORWARD (dummy 1024×1024 image)\n{'='*70}")

# Create dummy image
img = Image.new("RGB", (1024, 1024), color=(128, 128, 128))
inputs = processor(images=img, return_tensors="pt").to(device=device, dtype=torch.bfloat16)
print(f"processor output keys: {list(inputs.keys())}")
print(f"  pixel_values shape: {inputs.pixel_values.shape}")
print(f"  image_grid_thw shape: {inputs.get('image_grid_thw', None)}")
print(f"  image_grid_thw value: {inputs.get('image_grid_thw', None)}")

# Feed into vision model
with torch.no_grad():
    # Qwen3VisionModel.forward returns BaseModelOutputWithPast or similar
    out = vision_model(
        pixel_values=inputs.pixel_values,
        grid_thw=inputs.image_grid_thw,
    )

# Print output shape
print(f"\nvision_model output type: {type(out)}")
if hasattr(out, 'last_hidden_state'):
    hs = out.last_hidden_state
    print(f"last_hidden_state shape: {tuple(hs.shape)}  dtype={hs.dtype}")
elif hasattr(out, 'hidden_states'):
    print(f"hidden_states len: {len(out.hidden_states)}")
    hs = out.hidden_states[-1]
    print(f"last hidden_state shape: {tuple(hs.shape)}")
else:
    # Maybe it returns the visual features directly
    print(f"output: {out}")

# Also check what merger outputs
# The merger is the tail of Qwen3VisionModel
# Let's check model structure
print(f"\nvision_model children:")
for name, child in vision_model.named_children():
    print(f"  {name}: {type(child).__name__}")

# Directly run merger if accessible
merger = None
for name, module in vision_model.named_modules():
    if "merger" in name.lower():
        print(f"  merger module: {name} -> {type(module).__name__}")
        merger = module

if merger is not None and hasattr(out, 'last_hidden_state'):
    # merger input = ViT features
    features = out.last_hidden_state
    print(f"\nRunning merger directly on features shape={tuple(features.shape)}")
    print(f"  merger.norm.weight: {merger.norm.weight.shape}")
    print(f"  merger.linear_fc1.weight: {merger.linear_fc1.weight.shape}")
    print(f"  merger.linear_fc2.weight: {merger.linear_fc2.weight.shape}")

# ── 4. Test with various image sizes ─────────────────
print(f"\n{'='*70}\n4. PROCESSOR OUTPUT for various image sizes\n{'='*70}")
for sz in [512, 672, 768, 1024, 1568, 2048]:
    img = Image.new("RGB", (sz, sz), color=(100, 150, 200))
    inputs = processor(images=img, return_tensors="pt")
    grid = inputs.get("image_grid_thw", None)
    if grid is not None:
        t, h, w = grid[0].tolist()
        n_vis_tokens = h * w  # T=1 for single image
        print(f"  {sz}×{sz} → grid_thw=({t},{h},{w}) → visual_tokens={n_vis_tokens}")
        # also compute directly from formula
        n = processor.image_processor
        print(f"    patch={n.patch_size} merge={n.merge_size} stride={n.patch_size*n.merge_size} → formula tokens={-(-sz // n.patch_size // n.merge_size)**2}")
    else:
        print(f"  {sz}×{sz} → no image_grid_thw in output, keys={list(inputs.keys())}")

print("\nDONE.")
