"""Manual sharded LLM forward for broken-P2P multi-GPU environments.

This machine's V100s have corrupted GPU→GPU P2P copies (peer access reports
True but transfers deliver zeros/stale data), while H2D/D2H and host-staged
transfers are bit-exact. accelerate's dispatch hooks use direct P2P .to(),
so we remove them and route every cross-GPU tensor through host staging.

Also computes RoPE cos/sin on CPU (rotary_emb.inv_freq lives on a different
GPU → its .to() would be a corrupting P2P copy).

Usage:
    llm = LlamaForCausalLM.from_pretrained(..., device_map="auto", ...)
    remove_accelerate_hooks(llm)
    logits = sharded_llm_forward(llm, inputs_embeds)   # autograd-safe
"""
from __future__ import annotations

import torch


def remove_accelerate_hooks(llm) -> None:
    """Strip accelerate dispatch hooks — we do all device routing manually."""
    from accelerate.hooks import remove_hook_from_module
    for module in llm.modules():
        remove_hook_from_module(module)


def _stage(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move tensor to device via host staging if cross-GPU (P2P is broken)."""
    if t.device == device:
        return t
    if t.device.type == "cpu" or device.type == "cpu":
        return t.to(device)          # H2D / D2H — verified safe
    return t.cpu().to(device)        # GPU→GPU — must go through host


@torch.no_grad()
def _rope_cos_sin(model, seq_len: int, batch: int, device_hint: torch.device):
    """Compute RoPE cos/sin on CPU (inv_freq may live on another GPU)."""
    pos = torch.arange(seq_len, device="cpu").unsqueeze(0).expand(batch, -1)
    dummy = torch.zeros(1, 1, 1, dtype=torch.float32, device="cpu")
    # inv_freq is moved D2H internally (safe); rotary only reads pos + inv_freq
    inv_freq = model.rotary_emb.inv_freq.float().cpu()
    scaling = getattr(model.rotary_emb, "attention_scaling", 1.0)
    freqs = pos[:, :, None].float() * inv_freq[None, None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos() * scaling
    sin = emb.sin() * scaling
    return pos, cos.to(device_hint.dtype), sin.to(device_hint.dtype)


def sharded_llm_forward(llm, inputs_embeds: torch.Tensor) -> torch.Tensor:
    """Full LlamaForCausalLM forward with manual host-staged GPU routing.

    Assumes: batch=1 unpadded sequences (pure causal, no attention_mask needed).
    Autograd flows through (for projector training with frozen LLM).
    """
    model = llm.model
    B, S, _ = inputs_embeds.shape

    # RoPE tables on CPU (dtype-matched), staged per layer
    pos_cpu, cos_cpu, sin_cpu = _rope_cos_sin(model, S, B, inputs_embeds)

    h = inputs_embeds
    for layer in model.layers:
        d = next(layer.parameters()).device
        h = _stage(h, d)
        pos = torch.arange(S, device=d).unsqueeze(0).expand(B, -1)
        out = layer(
            h,
            attention_mask=None,
            position_ids=pos,
            position_embeddings=(_stage(cos_cpu, d), _stage(sin_cpu, d)),
            use_cache=False,
        )
        h = out[0] if isinstance(out, tuple) else out

    h = _stage(h, model.norm.weight.device)
    h = model.norm(h)
    h = _stage(h, llm.lm_head.weight.device)
    return llm.lm_head(h)
