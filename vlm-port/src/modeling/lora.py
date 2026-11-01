"""Minimal LoRA for Llama layers on a broken-P2P sharded setup.

Why hand-rolled: peft not installed + transformers 5.14.1 compat risk.
~120 lines, full control over dtype / device placement / ckpt format.

Design (matched to this project's diagnostics):
  - LoRA params are FP32 and live on the SAME device as the base Linear's
    weight (they scatter with the 15/19/22-layer shard split; AdamW must
    run with foreach=False across devices).
  - lora_B zero-initialized → initial delta = 0, model behavior unchanged
    (safe joint-training start: random projector + intact LLM).
  - Forward computes the delta in FP32 and casts back to the base output
    dtype (bf16 LLM activations) — same numerics philosophy as the
    projector staying fp32.
  - Base Linear stays frozen (requires_grad=False set by the training
    script BEFORE apply_lora).

Checkpoint format: {"<param_name>": tensor} over all params containing
"lora_" (e.g. "model.layers.0.self_attn.q_proj.lora_A") — restore by
re-applying apply_lora() then load_state_dict(..., strict=False) or
direct param copy.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank FP32 residual.

    out = base(x) + ( dropout(x).float() @ A^T @ B^T ) * (alpha / r)
    """

    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0,
                 dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear), type(base)
        self.base = base
        self.r = r
        self.scaling = alpha / r
        dev, dt = base.weight.device, torch.float32
        # A: (r, in) kaiming — matches nn.Linear default init family
        # B: (out, r) zeros → delta = 0 at step 0
        self.lora_A = nn.Parameter(
            torch.empty(r, base.in_features, dtype=dt, device=dev))
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, r, dtype=dt, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)  # frozen bf16 matmul
        # delta always computed: at train time B is nonzero from step 1 on,
        # at step 0 delta=0 (B zeros) → identical to the frozen model.
        xf = self.lora_dropout(x).float()
        delta = (xf @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return out + delta.to(out.dtype)

    def extra_repr(self) -> str:
        return f"r={self.r}, scaling={self.scaling:.3f}, base={self.base.in_features}→{self.base.out_features}"


DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj")


def apply_lora(llm, r: int = 16, alpha: float = 32.0, dropout: float = 0.0,
               targets=DEFAULT_TARGETS) -> list[nn.Module]:
    """Replace target Linears inside every decoder layer with LoRALinear.

    Call AFTER `for p in llm.parameters(): p.requires_grad = False` (base
    weights frozen) — new lora params default to requires_grad=True.

    Returns the list of LoRALinear modules (for param collection / logging).
    """
    modules: list[nn.Module] = []
    targets = set(targets)
    for layer in llm.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name, child in list(parent.named_children()):
                if name in targets and isinstance(child, nn.Linear):
                    setattr(parent, name, LoRALinear(child, r=r, alpha=alpha,
                                                     dropout=dropout))
                    modules.append(getattr(parent, name))
    if not modules:
        raise RuntimeError(
            "apply_lora replaced 0 modules — unexpected Llama structure? "
            f"targets={sorted(targets)}")
    return modules


def lora_state_dict(llm) -> dict[str, torch.Tensor]:
    """All LoRA params by full name (base weights excluded)."""
    return {n: p.detach().cpu() for n, p in llm.named_parameters()
            if "lora_" in n}


def load_lora_state(llm, state: dict[str, torch.Tensor]) -> int:
    """Copy a saved LoRA state back into an already-applied LoRA model.

    Returns number of params restored; raises if keys are missing."""
    mine = {n: p for n, p in llm.named_parameters() if "lora_" in n}
    missing = set(mine) - set(state)
    if missing:
        raise KeyError(f"LoRA state missing {len(missing)} keys, e.g. {sorted(missing)[:3]}")
    with torch.no_grad():
        for n, p in mine.items():
            p.copy_(state[n].to(p.device, p.dtype))
    return len(mine)
