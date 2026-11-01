"""Norm-mismatch hypothesis test: WHY does the frozen Nanbeige LLM ignore
visual tokens? (follow-up to diag_v2_shortcut findings: projector output
norm 95 vs LLM embed norm 1.64 = 58x mismatch; cross-image cos 0.9984)

5 variants on the first 48 v2 val samples:
  1. as-is    : v2 projector output (norm ~95, collapsed directions)
  2. scaled   : same output, per-token norm rescaled to LLM embed mean (1.64)
  3. random   : random directions at norm 1.64 (information-free, matched scale)
  4. zeros    : zero vectors at visual positions (minimal disturbance)
  5. text-only: no visual tokens at all (question rendered as pure text)

Interpretation guide:
  2 << 1      → norm mismatch is the blocker → fix = output norm layer
  2 ≈ 1       → per-token norm is cancelled by layer-1 RMSNorm anyway; the
                problem is directional collapse / optimization
  3 > 1       → the collapsed output is a carefully-learned "least disturbing"
                direction (better than random noise)
  4 ≈ 5       → LLM treats visual positions as if absent (pure ignoring)
  5 vs 1      → total contribution of the (collapsed) visual tokens

Run: cd vlm-port && ../venv-cu128/bin/python scripts/diag_norm_test.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_PROJECT_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_PROJECT_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM, AutoConfig, AutoTokenizer

from vision.qwen_vision import load_qwen_vision_tower
from vision.projector import CustomProjector
from data.processor import NanbeigeVisionProcessor
from data.dataset import ImageTextDataset
from data.collator import MultimodalDataCollator
from modeling.sharded_llm import remove_accelerate_hooks, sharded_llm_forward
from train_stage1 import build_multimodal_embeds, manual_ce_loss

torch.backends.cudnn.enabled = False

CFG_PATH = Path(__file__).resolve().parent.parent / "configs/train_stage1_v2.yaml"
cfg = yaml.safe_load(open(CFG_PATH))
qwen_dir, nanbeige_dir = cfg["qwen_model_dir"], cfg["llm_model_dir"]
N_SAMPLES = 48
LLM_EMB_NORM = 1.6356  # measured in diag_v2_shortcut [B]


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(nanbeige_dir)
    processor = NanbeigeVisionProcessor(qwen_dir, nanbeige_dir)

    vision_tower, vinfo = load_qwen_vision_tower(qwen_dir, device="cuda:0",
                                                 dtype=torch.float32)
    llm_cfg = AutoConfig.from_pretrained(nanbeige_dir)
    llm = LlamaForCausalLM.from_pretrained(
        nanbeige_dir, config=llm_cfg, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map="auto",
        max_memory={0: "20GiB", 1: "27GiB", 2: "25GiB"})
    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False
    remove_accelerate_hooks(llm)

    projector = CustomProjector(
        vision_feature_dim=vinfo.out_hidden_size, llm_hidden_size=llm_cfg.hidden_size,
        use_input_norm=True).to(device="cuda:0", dtype=torch.float32)
    ck = torch.load(cfg["save_dir"] + "/projector_final.pt",
                    map_location="cuda:0", weights_only=True)
    projector.load_state_dict(ck["projector_state_dict"])
    projector.eval()

    ds = ImageTextDataset(cfg["val_manifest"], processor,
                          image_dir=cfg.get("image_dir"),
                          max_text_length=cfg.get("max_text_length", 4096))
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=MultimodalDataCollator(processor), num_workers=0)
    samples = [json.loads(l) for l in open(cfg["val_manifest"]) if l.strip()][:N_SAMPLES]
    dev = torch.device("cuda:0")
    print(f"models loaded ({time.time()-t0:.0f}s), N={len(samples)}\n", flush=True)

    orig_forward = projector.forward

    def set_transform(fn):
        if fn is None:
            projector.forward = orig_forward
        else:
            def fwd(x):
                return fn(orig_forward(x))
            projector.forward = fwd

    def run(name, transform=None, text_only=False):
        set_transform(transform)
        losses = []
        with torch.no_grad():
            for batch, rec in zip(loader, samples):
                if text_only:
                    item = ds._build_tokens(rec["question"], rec["answer"], None)
                    input_ids = torch.tensor([item["input_ids"]], dtype=torch.long)
                    labels = torch.tensor([item["labels"]], dtype=torch.long)
                    embed_dev = llm.model.embed_tokens.weight.device
                    te = (llm.model.embed_tokens(input_ids.to(embed_dev))
                          .to(torch.bfloat16).to(dev))
                    logits = sharded_llm_forward(llm, te)
                    losses.append(manual_ce_loss(logits, labels).item())
                else:
                    batch = batch.to(dev)
                    te = build_multimodal_embeds(
                        llm, vision_tower, projector, batch, torch.bfloat16, dev)
                    logits = sharded_llm_forward(llm, te)
                    losses.append(manual_ce_loss(logits, batch.labels).item())
        t = torch.tensor(losses)
        print(f"[{name:9s}] mean={t.mean().item():.4f} std={t.std().item():.4f} "
              f"min={t.min().item():.4f} max={t.max().item():.4f}", flush=True)
        return t.mean().item()

    def to_llm_norm(o):
        return o / (o.norm(dim=-1, keepdim=True) / LLM_EMB_NORM)

    def random_at_llm_norm(o):
        r = torch.randn_like(o)
        return r / (r.norm(dim=-1, keepdim=True) / LLM_EMB_NORM)

    base = run("as-is")
    scaled = run("scaled", to_llm_norm)
    run("random", random_at_llm_norm)
    run("zeros", lambda o: torch.zeros_like(o))
    text = run("text-only", text_only=True)

    print(f"\nΔ(scaled - as-is) = {scaled - base:+.4f}  "
          f"(strongly negative → norm mismatch is the blocker)")
    print(f"Δ(text-only - as-is) = {text - base:+.4f}  "
          f"(visual tokens' total contribution)")
    print(f"\ndone ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
