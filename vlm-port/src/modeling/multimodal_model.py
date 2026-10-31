"""Multimodal wrapper: Qwen Vision Tower → Custom Projector → Nanbeige LLM embedding injection.

The wrapper:
  1. Gets text embeddings from Nanbeige's embed_tokens
  2. Runs frozen Qwen vision tower on pixel_values → post-merge features (N, 5120)
  3. Projects features to Nanbeige hidden_size via CustomProjector
  4. Replaces image_pad token embeddings with projected features
  5. Feeds to frozen Nanbeige LLM decoder with standard 1D sequential position_ids
  6. Computes labels: image_pad + vision_start/vision_end positions → -100

Key design choices (confirmed by deep analysis):
  - position_ids: pure 1D sequential (LLaVA-style), NO MRoPE
  - Vision tower: FULLY frozen (ViT + Merger), output = pooler_output = post-merge (N, 5120)
  - Nanbeige LLM: FULLY frozen, no architecture modification
  - Projector: ONLY trainable module
  - Special tokens: zero vocab expand (Nanbeige ids 166107/166108/166109)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from vision.qwen_vision import load_qwen_vision_tower, VisionTowerInfo
from vision.projector import CustomProjector


@dataclass
class MultimodalModelConfig:
    vision_model_dir: str
    llm_model_dir: str
    projector_act_fn: str = "gelu"
    projector_use_input_norm: bool = False
    device: str = "cpu"
    dtype: torch.dtype = torch.bfloat16
    torch_dtype_llm: torch.dtype = torch.bfloat16


class NanbeigeVLM(nn.Module):
    """Vision-Language Model: Qwen3.8 Vision + Custom Projector + Nanbeige4.3-33B LLM."""

    def __init__(self, config: MultimodalModelConfig):
        super().__init__()
        self.config = config
        device = config.device
        dtype = config.dtype

        # ── 1. Load frozen Qwen vision tower ──
        print(f"[VLM] Loading Qwen vision tower from {config.vision_model_dir}")
        self.vision_tower, self.vision_info = load_qwen_vision_tower(
            config.vision_model_dir, device=device, dtype=dtype,
        )

        # ── 2. Load Nanbeige LLM ──
        print(f"[VLM] Loading Nanbeige LLM from {config.llm_model_dir}")
        from transformers import LlamaForCausalLM, AutoConfig
        llm_cfg = AutoConfig.from_pretrained(config.llm_model_dir)
        self.llm = LlamaForCausalLM.from_pretrained(
            config.llm_model_dir,
            config=llm_cfg,
            torch_dtype=config.torch_dtype_llm,
            low_cpu_mem_usage=True,
        ).to(device=device)

        # Freeze LLM ENTIRELY — before any optimizer touches params
        for p in self.llm.parameters():
            p.requires_grad = False
        self.llm.eval()
        n_llm_params = sum(p.numel() for p in self.llm.parameters())
        print(f"[VLM] Nanbeige LLM loaded: {n_llm_params/1e9:.1f}B params, ALL frozen")

        self.llm_hidden_size = llm_cfg.hidden_size  # 7168
        self.vocab_size = llm_cfg.vocab_size

        # ── 3. Custom Projector (ONLY trainable module) ──
        self.projector = CustomProjector(
            vision_feature_dim=self.vision_info.vision_feature_dim,  # 5120
            llm_hidden_size=self.llm_hidden_size,                    # 7168
            act_fn=config.projector_act_fn,
            use_input_norm=config.projector_use_input_norm,
        )
        self.projector.to(device=device, dtype=dtype)

        # Verify frozen count
        n_total = sum(p.numel() for p in self.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[VLM] Total params: {n_total/1e9:.2f}B, trainable: {n_train/1e6:.2f}M "
              f"({n_train/n_total*100:.4f}%)")

    # ── Vision forward (frozen, no grad) ──
    @torch.no_grad()
    def _vision_forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Run frozen Qwen vision tower. Returns (sum N_post_merge, 5120)."""
        if pixel_values.numel() == 0:
            return torch.zeros(0, self.vision_info.out_hidden_size,
                               device=pixel_values.device, dtype=pixel_values.dtype)

        out = self.vision_tower(
            pixel_values,               # first positional arg — named 'hidden_states' in API
            grid_thw=image_grid_thw,
        )
        pooler = out.pooler_output  # post-merge features
        if isinstance(pooler, (list, tuple)):
            # get_image_features splits per-image, VisionModel returns flat tensor
            pooler = pooler[0] if len(pooler) == 1 else torch.cat(pooler, dim=0)
        assert pooler.shape[-1] == self.vision_info.out_hidden_size, (
            f"Vision pooler_output dim mismatch: got {pooler.shape[-1]}, "
            f"expected {self.vision_info.out_hidden_size}"
        )
        return pooler

    # ── Main forward ──
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        image_pad_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass with embedding injection.

        Args:
            input_ids: (B, S) token ids
            attention_mask: (B, S) bool/int mask
            pixel_values: (total_vision_patches, 1536) Qwen-processor output
            image_grid_thw: (num_images, 3) ViT patch grid
            image_pad_mask: (B, S) bool — True at image_pad positions
            labels: (B, S) — if provided, CE loss is computed; image/vision tokens should be -100

        Returns:
            loss if labels provided, else logits (B, S, vocab_size)
        """
        B, S = input_ids.shape
        device = input_ids.device
        dtype = self.llm.dtype

        # ── 1. Get text embeddings ──
        text_embeds = self.llm.model.embed_tokens(input_ids).to(dtype)
        assert text_embeds.shape == (B, S, self.llm_hidden_size), (
            f"text_embeds shape wrong: {text_embeds.shape}, "
            f"expected ({B}, {S}, {self.llm_hidden_size})"
        )

        # ── 2. Vision features + Projector ──
        has_images = (pixel_values is not None and pixel_values.numel() > 0
                      and image_pad_mask is not None and image_pad_mask.any().item())

        if has_images:
            vis_features = self._vision_forward(pixel_values, image_grid_thw)  # (N_post_merge_total, 5120)
            projected = self.projector(vis_features.to(self.projector.net[0].weight.dtype))
            projected = projected.to(dtype)
            assert projected.shape[-1] == self.llm_hidden_size, (
                f"Projected dim wrong: {projected.shape[-1]} != {self.llm_hidden_size}"
            )
            n_pads = image_pad_mask.sum().item()
            assert projected.shape[0] == n_pads, (
                f"Projected tokens ({projected.shape[0]}) != image_pad positions ({n_pads}). "
                f"Check image_pad expansion vs grid_thw consistency."
            )

            # ── 3. Replace image_pad embeddings ──
            text_embeds[image_pad_mask] = projected

        # ── 4. 1D sequential position_ids (LLaVA-style) ──
        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

        # ── 5. LLM forward ──
        outputs = self.llm(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            use_cache=False,  # no KV cache in training
        )
        return outputs

    # ── Generation wrapper ──
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        image_pad_mask: torch.Tensor | None = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        **gen_kwargs: Any,
    ) -> torch.Tensor:
        """Generate with vision prefill + auto-regressive decode.

        Vision tokens are only used in prefill (replacement).
        Decode phase is standard autoregressive — no more vision input.
        """
        B, S = input_ids.shape
        device = input_ids.device
        dtype = self.llm.dtype

        # ── Prefill: embed injection once ──
        text_embeds = self.llm.model.embed_tokens(input_ids).to(dtype)

        has_images = (pixel_values is not None and pixel_values.numel() > 0
                      and image_pad_mask is not None and image_pad_mask.any().item())
        if has_images:
            vis_features = self._vision_forward(pixel_values, image_grid_thw)
            projected = self.projector(vis_features.to(self.projector.net[0].weight.dtype)).to(dtype)
            text_embeds[image_pad_mask] = projected

        # Standard 1D sequential position_ids
        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

        # ── Generate ──
        # Must pass BOTH input_ids (for autoregressive decode steps) AND
        # inputs_embeds (for vision-injected prefill step).
        # See: LlamaForCausalLM.prepare_inputs_for_generation
        outputs = self.llm.generate(
            input_ids=input_ids,
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **gen_kwargs,
        )
        return outputs

    # ── Frozen parameter accounting ──
    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def frozen_params(self):
        return [p for p in self.parameters() if not p.requires_grad]


def build_labels(
    input_ids: torch.Tensor,
    image_pad_mask: torch.Tensor,
    vision_start_token_id: int | None = None,
    vision_end_token_id: int | None = None,
    pad_token_id: int | None = None,
) -> torch.Tensor:
    """Build labels for CE loss.

    image_pad positions → -100 (no loss on visual tokens)
    vision_start/vision_end positions → -100 (structure markers)
    pad positions → -100
    All other positions → input_ids shifted left by 1 (standard next-token pred)
    """
    B, S = input_ids.shape
    labels = input_ids.clone()

    # Mask image_pad
    labels[image_pad_mask] = -100

    # Mask vision start/end (optional)
    if vision_start_token_id is not None:
        labels[input_ids == vision_start_token_id] = -100
    if vision_end_token_id is not None:
        labels[input_ids == vision_end_token_id] = -100
    if pad_token_id is not None:
        labels[input_ids == pad_token_id] = -100

    # Shift for next-token prediction: we let the model handle it via LLM's shift_labels
    # Actually, LlamaForCausalLM.forward handles shift_labels internally when labels is passed
    # So we just return the masked labels, and LLM will shift internally
    return labels
