#!/usr/bin/env python
"""Deep dive: MRoPE structure, position_ids, merger reshape, image_grid_thw.

Reads transformers source + configs + minimal tensor headers to answer:
  1. What exactly is Qwen's MRoPE [11,11,10] partial_rotary_factor 0.25?
  2. How does Nanbeige generate position_ids?
  3. What is Qwen merger's spatial reshape path?
  4. What does image_grid_thw actually look like for real images?
  5. Token ID collision safety for 3 new vision tokens in Nanbeige vocab.
"""
import json, sys, importlib.util
from pathlib import Path

MODEL_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/acproject/usb/transplant")
QWEN_DIR = MODEL_DIR / "Qwen3.8-27B"
NAN_DIR = MODEL_DIR / "Nanbeige4.3-33B"

# ────────────────────────────────────────────────────────
# 1. Load both configs
# ────────────────────────────────────────────────────────
qcfg = json.load(open(QWEN_DIR / "config.json"))
ncfg = json.load(open(NAN_DIR / "config.json"))
qrope = qcfg["text_config"]["rope_parameters"]
nrope = {k: ncfg[k] for k in ["rope_theta", "rotary_base", "rope_scaling", "head_dim"] if k in ncfg}
print("="*70)
print("MRoPE vs Standard RoPE")
print("="*70)
print(f"  Qwen MRoPE sections: {qrope['mrope_section']}  interleaved={qrope['mrope_interleaved']}")
print(f"  Qwen partial_rotary_factor: {qrope['partial_rotary_factor']}")
print(f"  Qwen head_dim: {qcfg['text_config']['head_dim']}  => rotary_dim = {int(qcfg['text_config']['head_dim'] * qrope['partial_rotary_factor'])}")
print(f"  Qwen rope_theta: {qrope['rope_theta']}")
print(f"  Qwen max_pos: {qcfg['text_config']['max_position_embeddings']}")
print()
print(f"  Nanbeige rope_theta: {ncfg['rope_theta']}  rotary_base: {ncfg.get('rotary_base')}")
print(f"  Nanbeige rope_scaling: {ncfg['rope_scaling']}")
print(f"  Nanbeige head_dim: {ncfg['head_dim']}  => rotary_dim = {ncfg['head_dim']} (full rotary)")
print(f"  Nanbeige max_pos: {ncfg['max_position_embeddings']}")

# What mrope_section [11,11,10] + partial_rotary_factor 0.25 means:
# 11+11+10 = 32, which is exactly half of rotary_dim=64 (256*0.25)
# So full head_dim=256 is split as:
#   - 64 rotary dim = 11 + 11 + 10 (x,y,z 3D interleaved)
#   - 192 non-rotary dim
# The 32 values (half of 64) → 3 pairs (x,y,z) each with 11,11,10 positions
print()
print("  ── MRoPE section math ──")
hd = qcfg['text_config']['head_dim']
prf = qrope['partial_rotary_factor']
rotary_dim = int(hd * prf)   # 64
sections = qrope['mrope_section']  # [11,11,10]
total = sum(sections)  # 32 → half of rotary_dim for cos/sin
print(f"  head_dim={hd}, partial={prf} → rotary_dim={rotary_dim}")
print(f"  sum(sections)={total} = rotary_dim/2 → cos/sin each cover {total} dims")
print(f"  3 sections map to [x, y, z] spatial axes")
print(f"  → Qwen expects 3D position_ids (B, 3, S) where dim 1 = [x, y, z] coordinates")
print(f"  → Nanbeige uses 1D position_ids (B, S)")

# ────────────────────────────────────────────────────────
# 2. Check transformers source for mrope_position
# ────────────────────────────────────────────────────────
print()
print("="*70)
print("transformers MRoPE source inspection")
print("="*70)
import transformers, os
tr_dir = os.path.dirname(transformers.__file__)
print(f"transformers dir: {tr_dir}")
# find mrope-related files
for root, dirs, files in os.walk(tr_dir):
    for f in files:
        if 'qwen3_5' in f or 'qwen3vl' in f.lower():
            print(f"  {root}/{f}")

# Search for mrope_position usage
import subprocess
result = subprocess.run(['grep', '-rn', 'mrope_position\|mrope_section\|mrope_interleaved', tr_dir],
                       capture_output=True, text=True, timeout=10)
lines = result.stdout.strip().split('\n')
print(f"\n  MRoPE references ({len(lines)}):")
for l in lines[:15]:
    print(f"    {l}")
