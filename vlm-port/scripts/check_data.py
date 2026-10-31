"""Fast data-pipeline check (no LLM load): dataset → collator → label masking.

Verifies for EVERY train/val sample:
  - tokenization boundary consistency (prefix == full[:n])
  - image_pad count == grid_thw.prod() // merge²
  - labels: only answer + <|im_end|> unmasked, decode round-trips
  - collated batch shapes

Run: ../venv-cu128/bin/python scripts/check_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_PROJECT_SRC))

from data.processor import NanbeigeVisionProcessor
from data.dataset import ImageTextDataset
from data.collator import MultimodalDataCollator

QWEN_DIR = "/home/acproject/usb/transplant/Qwen3.8-27B"
NANBEIGE_DIR = "/home/acproject/usb/transplant/Nanbeige4.3-33B"

proc = NanbeigeVisionProcessor(QWEN_DIR, NANBEIGE_DIR)
collator = MultimodalDataCollator(proc)
tok = proc.tokenizer

for split in ("train", "val"):
    ds = ImageTextDataset(f"data/{split}_manifest.jsonl", proc, image_dir="data")
    n_img = 0
    for i in range(len(ds)):
        s = ds[i]
        ids, labels = s["input_ids"], s["labels"]
        unmasked = [t for t in labels if t != -100]
        assert len(unmasked) >= 2, f"[{split}#{i}] too few unmasked label tokens: {unmasked}"
        decoded = tok.decode(unmasked)
        expect = s["answer_text"] + "<|im_end|>"
        assert decoded == expect, f"[{split}#{i}] label decode '{decoded}' != '{expect}'"
        # image token count consistency
        n_pad = ids.count(proc.image_pad_id)
        if s["num_images"]:
            n_img += 1
            assert s["image_grid_thw"].prod().item() // (proc.spatial_merge_size ** 2) == n_pad, \
                f"[{split}#{i}] image_pad {n_pad} != grid post-merge"
        # labels before first unmasked position must be -100
        first = labels.index(next(t for t in labels if t != -100))
        assert all(t == -100 for t in labels[:first]), f"[{split}#{i}] prompt tokens unmasked"
        assert all(t == -100 for t in labels[first + len(unmasked):]), f"[{split}#{i}] trailing unmasked"
    print(f"{split}: {len(ds)} samples OK ({n_img} with images)")

# collator batch
ds = ImageTextDataset("data/train_manifest.jsonl", proc, image_dir="data")
samples = [ds[i] for i in (0, 1, 2, 3)]
b = collator(samples)
S = b.input_ids.shape[1]
assert b.input_ids.shape == b.attention_mask.shape == b.image_pad_mask.shape == b.labels.shape
assert (b.labels != -100).sum().item() > 0
assert (b.labels[:, -1] == -100).all(), "padded tail must be masked"
print(f"collated: input_ids {tuple(b.input_ids.shape)}, pixel_values {tuple(b.pixel_values.shape)}, "
      f"grid_thw {tuple(b.image_grid_thw.shape)}, unmasked labels {(b.labels != -1).sum().item()}")
print("\nDATA PIPELINE OK ✅")
