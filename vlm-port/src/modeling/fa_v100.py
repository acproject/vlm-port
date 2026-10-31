"""Integrate flash_attn_v100 (Volta-tuned FlashAttention kernels) into HF transformers.

The flash_attn_v100 project ships prebuilt kernels (cp312 .so matches our venv)
with GQA prefill forward+backward for sm_70 — the hardware that official
flash-attn dropped. We register a custom attention implementation into
transformers' ALL_ATTENTION_FUNCTIONS so LlamaAttention layers dispatch to
the V100 kernel instead of SDPA.

Kernel semantics (verified against SDPA by the project's own autograd tests):
  - layout: q [B, H_Q, N, D], k/v [B, H_KV, N, D] — identical to what
    transformers 5.x passes attention functions
  - internal 1/sqrt(D) scaling (matches SDPA default and Llama self.scaling
    for head_dim=128)
  - fp16 equal-length (prefill) -> CUDA GQA kernel with backward;
    fp32 / bf16 / decode-style unequal lengths -> differentiable SDPA fallback

Only requirement: import torch BEFORE the .so (it links libc10.so).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

FA_V100_DIR = Path("/home/acproject/workspace/python_projects/flash_attn_v100")
ATTN_IMPL_NAME = "fa_v100"


def _fa_v100_attention_forward(
    module,
    query_states: torch.Tensor,   # [B, H_Q, S, D]
    key_states: torch.Tensor,     # [B, H_KV, S, D]
    value_states: torch.Tensor,  # [B, H_KV, S, D]
    attention_mask=None,
    dropout=0.0,
    scaling=None,
    **kwargs,
):
    """Drop-in attention for HF's ALL_ATTENTION_FUNCTIONS (5.x calling convention).

    Our sharded forward always calls layers with attention_mask=None
    (batch=1, unpadded, pure causal), so we hardcode causal=True — exactly
    Llama decoder semantics. Non-None masks are rejected loudly rather than
    silently mis-masked.
    """
    if attention_mask is not None:
        # 4D masks from full HF forward could contain padding; we only support
        # the pure-causal batch=1 path used by sharded_llm_forward.
        if attention_mask.dim() == 2 and bool((attention_mask == 1).all()):
            pass  # trivially all-ones 2D mask == pure causal
        else:
            raise ValueError(
                "fa_v100 attention wrapper only supports attention_mask=None "
                "or an all-ones 2D mask (batch=1 unpadded causal)."
            )

    from flash_attn_autograd import flash_attn_gqa

    # Kernel applies 1/sqrt(D) internally; rescale q if the model's scaling
    # differs (Llama head_dim=128 -> scaling == D**-0.5 -> no-op).
    D = query_states.size(-1)
    if scaling is not None and abs(scaling - D ** -0.5) > 1e-9:
        query_states = query_states * (scaling * math.sqrt(D))

    out = flash_attn_gqa(query_states, key_states, value_states, causal=True)
    # HF attention functions must return [B, S, H, D] (sdpa_attention_forward
    # does the same transpose) — the caller reshape(B, S, -1) assumes it.
    return out.transpose(1, 2).contiguous(), None


def enable_fa_v100(llm) -> bool:
    """Register the implementation and switch all attention layers to it.

    Call AFTER loading + freezing the model, BEFORE any forward.
    Returns False (and leaves SDPA in place) if the kernels are unavailable.
    """
    if str(FA_V100_DIR) not in sys.path:
        sys.path.insert(0, str(FA_V100_DIR))
    try:
        import torch as _t  # noqa: F401  — must precede .so import (libc10.so)
        from flash_attn_autograd import flash_attn_gqa  # noqa: F401  — import check
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        if ATTN_IMPL_NAME not in ALL_ATTENTION_FUNCTIONS:
            ALL_ATTENTION_FUNCTIONS.register(ATTN_IMPL_NAME, _fa_v100_attention_forward)
    except Exception as e:
        print(f"[fa_v100] unavailable ({type(e).__name__}: {e}); staying on SDPA")
        return False

    # get_interface() raises for unregistered names, so registration must
    # precede this. Every sub-module carrying a config gets the new impl
    # (they normally share one config object; the loop is just defensive).
    n = 0
    for m in llm.modules():
        cfg = getattr(m, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = ATTN_IMPL_NAME
            n += 1
    print(f"[fa_v100] enabled on {n} module configs — fp16 GQA prefill kernel, "
          f"autograd-capable (decode/fp32 falls back to SDPA)")
    return True
