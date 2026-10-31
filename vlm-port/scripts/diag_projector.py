"""Diagnose projector output collapse.

Evidence so far: per-sample mm val loss is 9.11±0.05 across ALL 48 samples
(different questions AND different images) → LLM's answer distribution is
input-independent → suspicion: projected vision embeddings are (nearly)
identical for every image.

Checks:
  A. pooler_output (vision tower) diversity across different images
  B. projector output diversity across the same images
  C. pairwise cosine similarity of per-image mean embeddings
  D. output norm vs LLM embed_tokens row norms (scale mismatch check)

Run: ../venv-cu128/bin/python scripts/diag_projector.py
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT / "src"))
sys.path.insert(0, str(_PROJECT / "scripts"))

import torch
from PIL import Image

from vision.qwen_vision import load_qwen_vision_tower
from vision.projector import CustomProjector
from data.processor import NanbeigeVisionProcessor
from modeling.sharded_llm import remove_accelerate_hooks

NANBEIGE_DIR = "/home/acproject/usb/transplant/Nanbeige4.3-33B"
QWEN_DIR = "/home/acproject/usb/transplant/Qwen3.8-27B"


def pairwise_cos(mat: torch.Tensor) -> list[float]:
    """mat: (N, D) → list of pairwise cosine similarities."""
    normed = torch.nn.functional.normalize(mat, dim=-1)
    out = []
    for i, j in combinations(range(mat.shape[0]), 2):
        out.append(float((normed[i] * normed[j]).sum()))
    return out


def stats(t: torch.Tensor) -> str:
    return f"mean={t.mean().item():+.4f} std={t.std().item():.4f} " \
           f"norm_mean={t.norm(dim=-1).mean().item():.2f}"


def main():
    torch.backends.cudnn.enabled = False
    dev = torch.device("cuda:0")

    processor = NanbeigeVisionProcessor(QWEN_DIR, NANBEIGE_DIR)
    samples = [json.loads(l) for l in open("data/val_manifest.jsonl") if l.strip()]

    # pick visually diverse samples: every 6th → different colors/shapes/text
    picks = samples[::6][:8]
    imgs = [Image.open(Path("data") / s["image"]).convert("RGB") for s in picks]
    print(f"picked {len(picks)} images:")
    for s in picks:
        print(f"  {s['image']}  q='{s['question'][:20]}' a='{s['answer']}'")

    # ── vision tower + projector ──
    tower, info = load_qwen_vision_tower(QWEN_DIR, device="cuda:0", dtype=torch.float32)
    projector = CustomProjector(
        vision_feature_dim=info.out_hidden_size, llm_hidden_size=7168,
        act_fn="gelu", use_input_norm=True,
    ).to(device=dev, dtype=torch.float32)
    sd = torch.load("checkpoints/stage1/projector_final.pt",
                    map_location=dev, weights_only=True)
    projector.load_state_dict(sd["projector_state_dict"])
    projector.eval()

    with torch.no_grad():
        pool_means, proj_means, proj_all = [], [], []
        for i, img in enumerate(imgs):
            out = processor.qwen_image_processor([img], return_tensors="pt")
            pv, grid = out["pixel_values"], out["image_grid_thw"]
            pool = tower(pv.to(dev), grid_thw=grid.to(dev)).pooler_output  # (N,5120)
            proj = projector(pool)                                         # (N,7168)
            pool_means.append(pool.mean(0))
            proj_means.append(proj.mean(0))
            proj_all.append(proj)
            if i == 0:
                print(f"\n[A] pooler_output (image 0): {stats(pool)}")
                print(f"[B] projector output (image 0): {stats(proj)}")
                within = pairwise_cos(proj)  # token-to-token within image 0
                print(f"    within-image token cos: min={min(within):.4f} "
                      f"mean={sum(within)/len(within):.4f} max={max(within):.4f}")

        pool_means = torch.stack(pool_means)
        proj_means = torch.stack(proj_means)

        print(f"\n[A] pooler per-image mean: {stats(pool_means)}")
        pc = pairwise_cos(pool_means)
        print(f"    cross-image cos: min={min(pc):.4f} mean={sum(pc)/len(pc):.4f} max={max(pc):.4f}")

        print(f"\n[B] projector per-image mean: {stats(proj_means)}")
        mc = pairwise_cos(proj_means)
        print(f"    cross-image cos: min={min(mc):.4f} mean={sum(mc)/len(mc):.4f} max={max(mc):.4f}")

        # per-token cross-image: image0 token k vs image1 token k (position aligned)
        a, b = proj_all[0], proj_all[1]
        pos_cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        print(f"    position-aligned token cos (img0 vs img1): mean={pos_cos.mean().item():.4f}")

        # ── scale check vs LLM embeddings ──
        from transformers import LlamaForCausalLM, AutoConfig
        llm = LlamaForCausalLM.from_pretrained(
            NANBEIGE_DIR, config=AutoConfig.from_pretrained(NANBEIGE_DIR),
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            device_map="auto", max_memory={0: "20GiB", 1: "27GiB", 2: "25GiB"})
        remove_accelerate_hooks(llm)
        w = llm.model.embed_tokens.weight  # (V, 7168)
        emb_norms = w.norm(dim=-1)
        # common text tokens only (first 100k of trained vocab, skip added/unused)
        common = emb_norms[:100000]
        print(f"\n[D] LLM embed row norms (common tokens): mean={common.mean().item():.2f} "
              f"p5={common.quantile(0.05).item():.2f} p95={common.quantile(0.95).item():.2f}")
        p0 = proj_all[0]
        print(f"    projector output norms: mean={p0.norm(dim=-1).mean().item():.2f}")
        print(f"    (vision special rows: "
              f"{[f'{emb_norms[t].item():.2f}' for t in (166107, 166108, 166109)]})")


if __name__ == "__main__":
    main()
