"""Custom Projector: maps Qwen merger output (5120) → Nanbeige LLM hidden (7168).

Default architecture: Linear(5120→7168) + GELU + Linear(7168→7168)
Configurable: add optional input RMSNorm, swap GELU for SiLU.

All dimensions auto-read from vision_config + target LLM hidden_size.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CustomProjector(nn.Module):
    """2-layer MLP projector from vision_feature_dim → llm_hidden_size.

    Args:
        vision_feature_dim: input dimension (= Qwen merger out_hidden_size = 5120)
        llm_hidden_size: output dimension (= Nanbeige hidden_size = 7168)
        hidden_dim: inner hidden dim (default = llm_hidden_size)
        act_fn: 'gelu' or 'silu'
        use_input_norm: if True, apply RMSNorm before the first Linear
    """

    def __init__(
        self,
        vision_feature_dim: int,
        llm_hidden_size: int,
        hidden_dim: int | None = None,
        act_fn: str = "gelu",
        use_input_norm: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else llm_hidden_size

        layers: list[nn.Module] = []
        if use_input_norm:
            layers.append(nn.RMSNorm(vision_feature_dim, eps=eps))

        layers.append(nn.Linear(vision_feature_dim, hidden_dim))
        if act_fn == "gelu":
            layers.append(nn.GELU())
        elif act_fn == "silu":
            layers.append(nn.SiLU())
        else:
            raise ValueError(f"Unknown act_fn: {act_fn}")
        layers.append(nn.Linear(hidden_dim, llm_hidden_size))

        self.net = nn.Sequential(*layers)
        self.vision_feature_dim = vision_feature_dim
        self.llm_hidden_size = llm_hidden_size

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[projector] CustomProjector({vision_feature_dim}→{hidden_dim}→{llm_hidden_size}) "
              f"act={act_fn} input_norm={use_input_norm} params={n_params/1e6:.2f}M")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.vision_feature_dim, (
            f"Projector input dim mismatch: got {x.shape[-1]}, expected {self.vision_feature_dim}"
        )
        out = self.net(x)
        assert out.shape[-1] == self.llm_hidden_size, (
            f"Projector output dim mismatch: got {out.shape[-1]}, expected {self.llm_hidden_size}"
        )
        return out
