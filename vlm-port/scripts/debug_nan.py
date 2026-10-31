"""Diagnose NaN in val loss: isolate LLM-fp16 vs projector output vs specific samples.

Checks per sample: vision feats / projector out / logits finiteness + magnitudes,
plus zero-projected and text-only (no injection) controls.

Run: ../venv-cu128/bin/python scripts/debug_nan.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_PROJECT_SRC))

import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM, AutoConfig

from vision.qwen_vision import load_qwen_vision_tower
from vision.projector import CustomProjector
from data.processor import NanbeigeVisionProcessor
from data.dataset import ImageTextDataset
from data.collator import MultimodalDataCollator


def manual_ce_loss(logits, labels):
    shift_logits, shift_labels = logits[:, :-1, :], labels[:, 1:].to(logits.device)
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1), ignore_index=-100)


def inject(te, mask, proj):
    B, S, H = te.shape
    flat = te.reshape(-1, H)
    idx = mask.reshape(-1).nonzero(as_tuple=True)[0]
    return torch.index_put(flat, (idx,), proj.to(flat.dtype)).reshape(B, S, H)

torch.backends.cudnn.enabled = False
QWEN_DIR = "/home/acproject/usb/transplant/Qwen3.8-27B"
NANBEIGE_DIR = "/home/acproject/usb/transplant/Nanbeige4.3-33B"

dev = torch.device("cuda:0")
vision_tower, vinfo = load_qwen_vision_tower(QWEN_DIR, device="cuda:0", dtype=torch.float32)
cfg = AutoConfig.from_pretrained(NANBEIGE_DIR)
llm = LlamaForCausalLM.from_pretrained(
    NANBEIGE_DIR, config=cfg, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    device_map="auto", max_memory={0: "22GiB", 1: "30GiB", 2: "30GiB"},
)
for p in llm.parameters():
    p.requires_grad = False
llm.eval()
projector = CustomProjector(5120, 7168, act_fn="gelu", use_input_norm=True).to(dev).float()

processor = NanbeigeVisionProcessor(QWEN_DIR, NANBEIGE_DIR)
collator = MultimodalDataCollator(processor)
val_ds = ImageTextDataset("data/val_manifest.jsonl", processor, image_dir="data")
train_ds = ImageTextDataset("data/train_manifest.jsonl", processor, image_dir="data")


@torch.no_grad()
def diagnose(batch, tag, mode="normal"):
    batch = batch.to(dev)
    embed_dev = llm.model.embed_tokens.weight.device
    te = llm.model.embed_tokens(batch.input_ids.to(embed_dev)).to(torch.float16).to(dev)
    stats = {}
    if batch.pixel_values.numel() > 0:
        vp = next(vision_tower.parameters())
        vis = vision_tower(batch.pixel_values.to(vp.dtype),
                           grid_thw=batch.image_grid_thw.to(vp.device)).pooler_output.to(dev)
        stats["vis_max"] = vis.abs().max().item()
        proj = projector(vis)
        stats["proj_max"] = proj.abs().max().item()
        if mode == "zero":
            proj = torch.zeros_like(proj)
            stats["proj_max"] = 0.0
        proj = proj.to(torch.float16)
        if mode == "textonly":
            pass
        else:
            te = inject(te, batch.image_pad_mask, proj)
        stats["embed_max"] = te.abs().max().item()
    B, S = batch.input_ids.shape
    pos = torch.arange(S, device=te.device).unsqueeze(0).expand(B, -1)
    logits = llm(inputs_embeds=te, attention_mask=batch.attention_mask.to(te.device),
                 position_ids=pos, use_cache=False).logits
    loss = manual_ce_loss(logits, batch.labels)
    stats.update(logits_max=logits.abs().max().item(),
                 logits_finite=bool(torch.isfinite(logits).all()),
                 loss=loss.item())
    print(f"[{tag}] mode={mode}: " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                             for k, v in stats.items()), flush=True)
    return stats


print("=== TRAIN samples (known finite in training) ===")
for i in range(2):
    diagnose(collator([train_ds[i]]), f"train#{i}")

print("\n=== VAL samples ===")
nan_found = None
for i in range(len(val_ds)):
    s = diagnose(collator([val_ds[i]]), f"val#{i}")
    if not s["logits_finite"] or s["loss"] != s["loss"]:
        nan_found = i
        break

if nan_found is not None:
    print(f"\n=== NaN at val#{nan_found}: controls ===")
    b = collator([val_ds[nan_found]])
    diagnose(b, f"val#{nan_found}", mode="zero")       # zero projector output
    diagnose(b, f"val#{nan_found}", mode="textonly")   # no injection at all
else:
    print("\nNo NaN in val set at init!")

print("\nanswer decode of val#0:", processor.tokenizer.decode(
    [t for t in collator([val_ds[0]]).labels[0].tolist() if t != -100]))
