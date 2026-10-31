"""MultimodalDataCollator: batches variable-length multimodal samples.

Handles:
- Variable image counts per sample (some have images, some don't)
- Variable text lengths → pad to max in batch
- Concatenated pixel_values across all images in the batch
- Correct image_grid_thw batching
- image_pad_mask construction
"""
from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Any

from .processor import NanbeigeVisionProcessor


@dataclass
class CollatedBatch:
    """Output of MultimodalDataCollator."""
    input_ids: torch.Tensor           # (B, max_S)
    attention_mask: torch.Tensor      # (B, max_S)
    pixel_values: torch.Tensor        # (total_viT_patches_in_batch, 1536)
    image_grid_thw: torch.Tensor      # (total_images_in_batch, 3)
    image_pad_mask: torch.Tensor      # (B, max_S) bool
    labels: torch.Tensor              # (B, max_S) long, -100 = masked
    num_images_per_sample: list[int]

    def to(self, device: torch.device) -> "CollatedBatch":
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.pixel_values = self.pixel_values.to(device)
        self.image_grid_thw = self.image_grid_thw.to(device)
        self.image_pad_mask = self.image_pad_mask.to(device)
        self.labels = self.labels.to(device)
        return self


class MultimodalDataCollator:
    """Collate multimodal samples into a batch.

    Each sample is expected to have pre-tokenized fields OR the processor is
    called internally. For training, the training loop will call the processor
    per-sample (since each sample may have a different number of images), and
    the collator just merges.
    """

    def __init__(self, processor: NanbeigeVisionProcessor):
        self.processor = processor
        self.spatial_merge_size = processor.spatial_merge_size

    def __call__(self, samples: list[dict[str, Any]]) -> CollatedBatch:
        """Collate a list of samples.

        Each sample dict should have keys:
          - input_ids: list[int] — tokenized text WITH image_pad expanded
          - attention_mask: list[int]
          - pixel_values: tensor or None — (n_patches, 1536)
          - image_grid_thw: tensor or None — (num_images, 3)
          - num_images: int

        Returns CollatedBatch.
        """
        batch_size = len(samples)
        B = batch_size

        # ── 1. Pad input_ids + attention_mask + labels ──
        max_len = max(len(s["input_ids"]) for s in samples)
        pad_id = self.processor.tokenizer.pad_token_id or 0

        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        image_pad_mask = torch.zeros((B, max_len), dtype=torch.bool)
        labels = torch.full((B, max_len), -100, dtype=torch.long)

        for bi, s in enumerate(samples):
            L = len(s["input_ids"])
            input_ids[bi, :L] = torch.tensor(s["input_ids"], dtype=torch.long)
            attention_mask[bi, :L] = torch.tensor(s["attention_mask"], dtype=torch.long)
            image_pad_mask[bi, :L] = input_ids[bi, :L] == self.processor.image_pad_id
            labels[bi, :L] = torch.tensor(s["labels"], dtype=torch.long)

        # ── 2. Concatenate pixel_values + image_grid_thw ──
        all_pixel_values: list[torch.Tensor] = []
        all_grid_thw: list[torch.Tensor] = []
        num_images_per_sample: list[int] = []

        for s in samples:
            n_imgs = s.get("num_images", 0)
            num_images_per_sample.append(n_imgs)
            if n_imgs > 0 and s.get("pixel_values") is not None:
                all_pixel_values.append(s["pixel_values"])
                all_grid_thw.append(s["image_grid_thw"])

        if all_pixel_values:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_grid_thw = torch.cat(all_grid_thw, dim=0)
        else:
            pixel_values = torch.zeros(0, 1536, dtype=torch.float32)
            image_grid_thw = torch.zeros(0, 3, dtype=torch.long)

        return CollatedBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_pad_mask=image_pad_mask,
            labels=labels,
            num_images_per_sample=num_images_per_sample,
        )
