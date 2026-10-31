"""Load Qwen3.8 Vision Tower only (ViT 27 layers + Merger), freeze all params.

The tower produces post-merge visual embeddings (N_post_merge, out_hidden_size)
where out_hidden_size=5120 for our Qwen3.8-27B checkpoint.

Usage:
    from vlm_port.vision.qwen_vision import load_qwen_vision_tower
    tower, vision_info = load_qwen_vision_tower(model_dir, device=device, dtype=torch.bfloat16)
    vis_out = tower(pixel_values, grid_thw=grid_thw)
    vis_features = vis_out.pooler_output   # (sum post_merge_tokens, out_hidden_size=5120)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


@dataclass
class VisionTowerInfo:
    """Structured metadata about the loaded vision tower."""
    hidden_size: int          # ViT width (1152 for Qwen3.8)
    out_hidden_size: int      # Merger output dim (=5120 = Qwen LLM hidden_size)
    depth: int                # ViT layers (27)
    num_heads: int
    patch_size: int
    spatial_merge_size: int   # 2
    temporal_patch_size: int  # 2
    num_position_embeddings: int
    vision_feature_dim: int   # = out_hidden_size, what projector consumes


def load_qwen_vision_tower(
    model_dir: str | Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.nn.Module, VisionTowerInfo]:
    """Load Qwen3.8 vision tower (ViT + Merger) only.

    Loads only `model.visual.*` weights from the safetensors shards,
    constructs a Qwen3VisionModel from transformers, strips the
    `model.visual.` prefix, and loads the state dict. All params
    are frozen (requires_grad=False).

    Returns (vision_tower, info).
    """
    model_dir = Path(model_dir)
    cfg = json.load(open(model_dir / "config.json"))
    vc = cfg["vision_config"]

    info = VisionTowerInfo(
        hidden_size=vc["hidden_size"],
        out_hidden_size=vc["out_hidden_size"],
        depth=vc["depth"],
        num_heads=vc["num_heads"],
        patch_size=vc["patch_size"],
        spatial_merge_size=vc["spatial_merge_size"],
        temporal_patch_size=vc["temporal_patch_size"],
        num_position_embeddings=vc["num_position_embeddings"],
        vision_feature_dim=vc["out_hidden_size"],  # Merger output = our projector input
    )

    # ── 1. Build Qwen3VisionModel from config ──
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5VisionModel, Qwen3_5VisionConfig,
    )
    vconfig = Qwen3_5VisionConfig(**vc)
    tower = Qwen3_5VisionModel(vconfig)

    # ── 2. Extract model.visual.* weights from safetensors ──
    index = json.load(open(model_dir / "model.safetensors.index.json"))
    wm = index["weight_map"]
    vision_keys = {k: fname for k, fname in wm.items() if k.startswith("model.visual.")}

    # Group keys by shard
    by_shard: dict[str, list[str]] = {}
    for k, fname in vision_keys.items():
        by_shard.setdefault(fname, []).append(k)

    state: dict[str, torch.Tensor] = {}
    for fname, keys in by_shard.items():
        with safe_open(model_dir / fname, framework="pt", device="cpu") as f:
            for k in keys:
                new_k = k.replace("model.visual.", "")
                state[new_k] = f.get_tensor(k)

    # ── 3. Load state dict (strict) ──
    missing, unexpected = tower.load_state_dict(state, strict=True)
    assert not missing, f"Missing vision keys: {missing}"
    assert not unexpected, f"Unexpected vision keys: {unexpected}"

    # ── 4. Move to target device/dtype, FREEZE ALL ──
    tower = tower.to(device=device, dtype=dtype).eval()
    for p in tower.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in tower.parameters())
    n_frozen = sum(p.numel() for p in tower.parameters() if not p.requires_grad)
    print(f"[qwen_vision] Loaded vision tower: {n_params/1e6:.1f}M params "
          f"(frozen={n_frozen/1e6:.1f}M) → pooler_output dim={info.out_hidden_size}")
    return tower, info


def compute_post_merge_tokens(grid_thw: torch.Tensor, spatial_merge_size: int) -> torch.Tensor:
    """Compute post-merge token count per image from ViT grid.

    grid_thw: (num_images, 3) — ViT patch grid [T, H, W]
    Returns:  (num_images,) — post-merge tokens per image
    """
    return grid_thw.prod(-1) // (spatial_merge_size ** 2)
