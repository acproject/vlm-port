"""NanbeigeVisionProcessor: wraps Qwen2VLImageProcessor + Nanbeige tokenizer + image_pad expansion.

Responsibilities:
  1. Resize images to patch_size×merge_size=32 multiples (via Qwen processor)
  2. Block-major reshape + flatten pixel_values (via Qwen processor)
  3. Generate ViT image_grid_thw (T, H_viT_patches, W_viT_patches)
  4. Tokenize chat template text with Nanbeige tokenizer
  5. Expand <|image_pad|> placeholder into N_post_merge copies per image

Image placeholder format in chat template:
    <|vision_start|><|image_pad|><|vision_end|>
After expansion:
    <|vision_start|><|image_pad|>×N_post_merge<|vision_end|>

Usage:
    processor = NanbeigeVisionProcessor(
        qwen_processor=AutoProcessor.from_pretrained(QWEN_DIR, trust_remote_code=True),
        nanbeige_tokenizer=AutoTokenizer.from_pretrained(NANBEIGE_DIR),
    )
    batch = processor(
        chat_messages=[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "..."}]}],
        images=[PIL.Image.new("RGB", (768, 768))],
    )
    # batch.input_ids, batch.attention_mask, batch.pixel_values, batch.image_grid_thw
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer


# Nanbeige free vocab slot IDs for vision special tokens (zero vocab expand)
VISION_START_ID = 166107
VISION_END_ID = 166108
IMAGE_PAD_ID = 166109

VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"
IMAGE_PAD_TOKEN = "<|image_pad|>"


@dataclass
class MultimodalBatch:
    """Output of NanbeigeVisionProcessor.__call__."""
    input_ids: torch.Tensor          # (B, S)
    attention_mask: torch.Tensor     # (B, S)
    pixel_values: torch.Tensor       # (total_patches, 1536) — Qwen processor output
    image_grid_thw: torch.Tensor     # (num_images, 3) — ViT patch grid
    vision_start_positions: list[list[int]]   # per-batch: list of indices where vision_start_id appears
    image_pad_mask: torch.Tensor     # (B, S) bool — True for image_pad positions
    num_images_per_sample: list[int] # per-batch: how many images

    def to(self, device: torch.device) -> "MultimodalBatch":
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.pixel_values = self.pixel_values.to(device)
        self.image_grid_thw = self.image_grid_thw.to(device)
        self.image_pad_mask = self.image_pad_mask.to(device)
        return self


class NanbeigeVisionProcessor:
    """Multimodal processor for Nanbeige + Qwen vision tower.

    Combines Qwen2VLImageProcessor (for image preprocessing + grid_thw) and
    Nanbeige tokenizer (for text tokenization + special vocab).
    """

    def __init__(
        self,
        qwen_processor_dir: str | Path,
        nanbeige_tokenizer_dir: str | Path,
    ):
        qwen_processor_dir = Path(qwen_processor_dir)
        nanbeige_tokenizer_dir = Path(nanbeige_tokenizer_dir)

        # Qwen image processor — handles resize, block-major reshape, grid_thw
        self.qwen_processor = AutoProcessor.from_pretrained(
            str(qwen_processor_dir), trust_remote_code=True
        )
        self.qwen_image_processor = self.qwen_processor.image_processor

        # Nanbeige tokenizer — text tokenization + vision special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(str(nanbeige_tokenizer_dir))

        # Add vision special tokens to Nanbeige tokenizer (zero vocab expand)
        # These IDs are in the FREE slot 166107-166911
        self._ensure_vision_special_tokens()

        # Vision config scalars (from Qwen preprocessor_config.json)
        self.patch_size = self.qwen_image_processor.patch_size
        self.spatial_merge_size = self.qwen_image_processor.merge_size

    def _ensure_vision_special_tokens(self):
        """Register vision special tokens at pre-allocated IDs in Nanbeige tokenizer."""
        existing_ids = {VISION_START_ID, VISION_END_ID, IMAGE_PAD_ID}
        # Add each token if not already present
        for tok_str, tid in [
            (VISION_START_TOKEN, VISION_START_ID),
            (VISION_END_TOKEN, VISION_END_ID),
            (IMAGE_PAD_TOKEN, IMAGE_PAD_ID),
        ]:
            current_id = self.tokenizer.convert_tokens_to_ids(tok_str)
            if current_id != tid:
                # Manually set the token at the correct vocab position
                # add_special_tokens can't place at arbitrary IDs, so we patch
                # the tokenizer's vocabulary directly
                self.tokenizer.add_tokens([tok_str])
                new_id = self.tokenizer.convert_tokens_to_ids(tok_str)
                if new_id != tid:
                    # Need to move it — remove and re-add with a mapping
                    # Simplest: accept new_id and update constants
                    print(f"  [WARNING] Could not place {tok_str} at id={tid}, got id={new_id}. "
                          f"Updating constant.")
                    if tok_str == VISION_START_TOKEN:
                        globals()["VISION_START_ID"] = new_id
                    elif tok_str == VISION_END_TOKEN:
                        globals()["VISION_END_ID"] = new_id
                    else:
                        globals()["IMAGE_PAD_ID"] = new_id

        # Re-read after potential updates
        self.vision_start_id = globals()["VISION_START_ID"]
        self.vision_end_id = globals()["VISION_END_ID"]
        self.image_pad_id = globals()["IMAGE_PAD_ID"]
        print(f"[processor] Vision special tokens: "
              f"<|vision_start|>={self.vision_start_id} "
              f"<|vision_end|>={self.vision_end_id} "
              f"<|image_pad|>={self.image_pad_id}")

    def render_chat_template(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Render chat messages into text with vision placeholders.

        Reuses Nanbeige's chat_template but overrides image/video handling.
        """
        # Build a simple renderer that handles image/video types
        rendered_parts: list[str] = []
        rendered_parts.append("<|im_start|>system\n你是一个具备图像理解能力的多模态助手。<|im_end|>\n")
        for msg in messages:
            role = msg["role"]
            rendered_parts.append(f"<|im_start|>{role}\n")
            content = msg["content"]
            if isinstance(content, str):
                rendered_parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type", "")
                        if item_type == "image":
                            rendered_parts.append(f"{VISION_START_TOKEN}{IMAGE_PAD_TOKEN}{VISION_END_TOKEN}")
                        elif item_type == "text":
                            rendered_parts.append(item.get("text", ""))
                        else:
                            rendered_parts.append(str(item))
                    else:
                        rendered_parts.append(str(item))
            rendered_parts.append(f"<|im_end|>\n")

        if add_generation_prompt:
            rendered_parts.append("<|im_start|>assistant\n")

        return "".join(rendered_parts)

    def _expand_image_pads(
        self,
        input_ids: list[int],
        n_post_merge_per_image: list[int],
    ) -> list[int]:
        """Expand each <|image_pad|> placeholder into N_post_merge copies.

        Args:
            input_ids: tokenized text (contains one <|image_pad|> per image)
            n_post_merge_per_image: [N1, N2, ...] post-merge token counts per image

        Returns:
            expanded input_ids list
        """
        result: list[int] = []
        image_count = 0
        for tid in input_ids:
            if tid == self.image_pad_id:
                if image_count >= len(n_post_merge_per_image):
                    raise RuntimeError(
                        f"More image_pad tokens ({image_count + 1}) than images "
                        f"({len(n_post_merge_per_image)})"
                    )
                n = n_post_merge_per_image[image_count]
                result.extend([self.image_pad_id] * n)
                image_count += 1
            else:
                result.append(tid)
        return result

    def __call__(
        self,
        messages: list[dict[str, Any]] | list[list[dict[str, Any]]],
        images: Image.Image | list[Image.Image] | list[list[Image.Image]] | None = None,
        device: torch.device | str = "cpu",
        max_length: int | None = None,
    ) -> MultimodalBatch:
        """Process chat messages + images → MultimodalBatch.

        Args:
            messages: single conversation (list[dict]) or batch (list[list[dict]])
            images: single image, list of images, or list of image-lists (batch)
            device: target device for returned tensors
            max_length: optional truncation

        Returns:
            MultimodalBatch with all tensors on `device`
        """
        # Normalize to batch
        if not isinstance(messages[0], list):
            messages = [messages]
        batch_size = len(messages)

        # Normalize images
        if images is None:
            images_batch: list[list[Image.Image]] = [[] for _ in range(batch_size)]
        elif isinstance(images, Image.Image):
            images_batch = [[images]] + [[] for _ in range(batch_size - 1)]
        elif isinstance(images[0], Image.Image):
            images_batch = [list(images)]
            if batch_size > 1:
                images_batch += [[] for _ in range(batch_size - 1)]
        else:
            images_batch = [list(imglist) for imglist in images]

        num_images_per_sample = [len(ilist) for ilist in images_batch]

        # ── 1. Image preprocessing via Qwen processor ──
        all_images_flat: list[Image.Image] = []
        for ilist in images_batch:
            all_images_flat.extend(ilist)

        if all_images_flat:
            qwen_out = self.qwen_image_processor(all_images_flat, return_tensors="pt")
            pixel_values = qwen_out["pixel_values"]
            image_grid_thw = qwen_out["image_grid_thw"]  # (total_images, 3) ViT patch grid
        else:
            pixel_values = torch.zeros(0, 1536, dtype=torch.float32)
            image_grid_thw = torch.zeros(0, 3, dtype=torch.long)

        # Compute post-merge token count per image
        n_post_merge_per_image = (
            (image_grid_thw.prod(-1) // (self.spatial_merge_size ** 2)).tolist()
            if image_grid_thw.numel() > 0 else []
        )

        # ── 2. Text rendering + tokenization ──
        all_expanded_ids: list[list[int]] = []
        vision_start_positions: list[list[int]] = []

        for bi, msg in enumerate(messages):
            text = self.render_chat_template(msg, add_generation_prompt=True)
            base_ids = self.tokenizer.encode(text, add_special_tokens=False)

            # Expand image_pad tokens
            # Gather N_post_merge for THIS sample's images
            my_n_per_image: list[int] = []
            idx = 0
            for ni in range(bi):
                idx += num_images_per_sample[ni]
            my_n_per_image = n_post_merge_per_image[idx: idx + num_images_per_sample[bi]]

            expanded = self._expand_image_pads(base_ids, my_n_per_image)
            all_expanded_ids.append(expanded)

            # Record vision_start positions
            vsp = [i for i, tid in enumerate(expanded) if tid == self.vision_start_id]
            vision_start_positions.append(vsp)

        # ── 3. Pad to uniform length ──
        max_len = max(len(ids) for ids in all_expanded_ids)
        if max_length is not None and max_len > max_length:
            max_len = max_length
            all_expanded_ids = [ids[:max_len] for ids in all_expanded_ids]

        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        image_pad_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

        for bi, ids in enumerate(all_expanded_ids):
            L = len(ids)
            input_ids[bi, :L] = torch.tensor(ids, dtype=torch.long)
            attention_mask[bi, :L] = 1
            image_pad_mask[bi, :L] = (input_ids[bi, :L] == self.image_pad_id)

        batch = MultimodalBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            vision_start_positions=vision_start_positions,
            image_pad_mask=image_pad_mask,
            num_images_per_sample=num_images_per_sample,
        )
        return batch.to(device)
