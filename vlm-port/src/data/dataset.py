"""Image-caption / VQA / text-only dataset for Stage 1 alignment.

Each sample is rendered as:
    <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n<vision>...question<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>\n

Labels: -100 everywhere EXCEPT the answer tokens + final <|im_end|>
(LLaVA-style response masking — no loss on prompt/system/image tokens).

JSONL manifest format (one line per sample):
    {"image": "/path/to/img.jpg", "question": "What color?", "answer": "Red"}
    {"text": "Hello world"}   # text-only sample (no image)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from torch.utils.data import Dataset

from .processor import NanbeigeVisionProcessor

IM_END_TOKEN = "<|im_end|>"


class ImageTextDataset(Dataset):
    """Image-caption / VQA / text-only dataset for Stage 1 alignment."""

    def __init__(
        self,
        manifest: str | Path | Iterable[dict[str, Any]],
        processor: NanbeigeVisionProcessor,
        image_dir: str | Path | None = None,
        max_text_length: int = 4096,
    ):
        if isinstance(manifest, (str, Path)):
            self.samples = []
            with open(manifest) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.samples.append(json.loads(line))
        else:
            self.samples = list(manifest)

        self.processor = processor
        self.image_dir = Path(image_dir) if image_dir else None
        self.max_text_length = max_text_length
        self.im_end_id = processor.tokenizer.convert_tokens_to_ids(IM_END_TOKEN)

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image(self, path: str) -> Image.Image:
        p = Path(path)
        if self.image_dir and not p.is_absolute():
            p = self.image_dir / p
        return Image.open(p).convert("RGB")

    def _build_tokens(
        self,
        question: str,
        answer: str,
        img: Image.Image | None,
    ) -> dict[str, Any]:
        """Tokenize prompt+answer, expand image pads, build response-masked labels."""
        processor = self.processor
        tok = processor.tokenizer

        user_msgs = [{"role": "user", "content": (
            [{"type": "image"}, {"type": "text", "text": question}]
            if img is not None else question
        )}]
        prefix_text = processor.render_chat_template(user_msgs, add_generation_prompt=True)
        full_text = prefix_text + answer + IM_END_TOKEN + "\n"

        prefix_ids = tok.encode(prefix_text, add_special_tokens=False)
        full_ids = tok.encode(full_text, add_special_tokens=False)
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise RuntimeError(
                f"Tokenizer boundary inconsistency:\nprefix_ids={prefix_ids}\n"
                f"full_ids[:n]={full_ids[:len(prefix_ids)]}"
            )

        pixel_values = None
        grid_thw = None
        n_images = 0

        if img is not None:
            qwen_out = processor.qwen_image_processor([img], return_tensors="pt")
            grid_thw = qwen_out["image_grid_thw"]
            n_per_image = [grid_thw.prod(-1).item() // (processor.spatial_merge_size ** 2)]
            prefix_ids = processor._expand_image_pads(prefix_ids, n_per_image)
            full_ids = processor._expand_image_pads(full_ids, n_per_image)
            pixel_values = qwen_out["pixel_values"]
            n_images = 1

        # Response span: after assistant prompt, through final <|im_end|>
        response_start = len(prefix_ids)
        if self.im_end_id not in full_ids:
            raise RuntimeError(f"<|im_end|> (id={self.im_end_id}) not found in tokenized sample")
        response_end = len(full_ids) - 1 - full_ids[::-1].index(self.im_end_id) + 1

        labels = [-100] * len(full_ids)
        for i in range(response_start, response_end):
            labels[i] = full_ids[i]

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw,
            "num_images": n_images,
            "answer_text": answer,
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        if "image" in sample:
            img = self._resolve_image(sample["image"])
            out = self._build_tokens(
                sample.get("question", "Describe this image."),
                sample.get("answer", ""),
                img,
            )
        elif "text" in sample:
            out = self._build_tokens(sample["text"], sample.get("answer", ""), None)
        else:
            raise ValueError(f"Sample {idx} has no 'image' or 'text' field: {sample}")

        # Truncate (rare; keeps response intact in practice for short samples)
        L = len(out["input_ids"])
        if L > self.max_text_length:
            for k in ("input_ids", "attention_mask", "labels"):
                out[k] = out[k][: self.max_text_length]
        return out
