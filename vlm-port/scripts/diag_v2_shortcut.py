"""Post-v2 shortcut diagnosis: WHY does the projector still collapse?

v2 (anti-shortcut data, 3000 samples, 594 steps) still generates garbage and
train loss plateaus at 8.85 — the model never fit even the TRAINING set.

This script measures, in one pass:
  [A] per-sample mm loss on val → within-template variance.
      Same question template, different images: identical loss = shortcut;
      spread = visual info flowing.
  [B] projector output norms vs LLM embedding norms (the check that failed
      with a bf16 quantile error in v1 diagnostics). Scale mismatch → LLM
      learns to ignore visual tokens (mechanistic shortcut cause).
  [C] projector output cross-image cos (v1 measured 0.993 after training).

Run: cd vlm-port && ../venv-cu128/bin/python scripts/diag_v2_shortcut.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
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

_Q_COUNT = {"图中有几个圆形？", "数一数图里有几个圆？"}


def task_kind(q: str) -> str:
    if "还是" in q:
        return "either"
    if any(s in q for s in ("描述", "有什么", "看到了什么")):
        return "caption"
    if any(s in q for s in ("分别", "什么颜色的什么形状", "它们是什么颜色")) or \
            ("几个" in q and q not in _Q_COUNT):
        return "joint"
    return "single"


def stats(t: torch.Tensor) -> str:
    t = t.float()
    return f"mean={t.mean().item():.4f} std={t.std().item():.4f} " \
           f"min={t.min().item():.4f} max={t.max().item():.4f}"


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
    print(f"models loaded ({time.time()-t0:.0f}s), projector step "
          f"{ck.get('global_step')}", flush=True)

    ds = ImageTextDataset(cfg["val_manifest"], processor,
                          image_dir=cfg.get("image_dir"),
                          max_text_length=cfg.get("max_text_length", 4096))
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=MultimodalDataCollator(processor), num_workers=0)
    samples = [json.loads(l) for l in open(cfg["val_manifest"]) if l.strip()]

    dev = torch.device("cuda:0")
    per_loss = []
    proj_pooled = []  # mean-pooled projector output per image, for cos

    print("\n[A] per-sample mm loss over val...", flush=True)
    with torch.no_grad():
        for i, (batch, rec) in enumerate(zip(loader, samples)):
            batch = batch.to(dev)
            text_embeds = build_multimodal_embeds(
                llm, vision_tower, projector, batch, torch.bfloat16, dev)
            loss = manual_ce_loss(sharded_llm_forward(llm, text_embeds),
                                  batch.labels).item()
            per_loss.append((task_kind(rec["question"]), rec["question"],
                             rec["answer"], loss))

            # [C] collect projector outputs (recompute here = same as in embeds)
            vp = next(vision_tower.parameters())
            vis = vision_tower(batch.pixel_values.to(vp.dtype),
                               grid_thw=batch.image_grid_thw.to(vp.device)).pooler_output
            out = projector(vis.to(dev).to(projector.net[0].weight.dtype))
            proj_pooled.append(out.float().mean(0))
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(samples)} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n[A] overall loss: {stats(torch.tensor([x[3] for x in per_loss]))}")
    by_task = defaultdict(list)
    for k, _, _, l in per_loss:
        by_task[k].append(l)
    for k, ls in sorted(by_task.items()):
        print(f"  {k:8s} n={len(ls):3d} {stats(torch.tensor(ls))}")

    print("\n[A2] within-template spread (same question, different images):")
    by_q = defaultdict(list)
    for _, q, _, l in per_loss:
        by_q[q].append(l)
    big = sorted(by_q.items(), key=lambda kv: -len(kv[1]))[:6]
    for q, ls in big:
        print(f"  n={len(ls)} {stats(torch.tensor(ls))}  '{q[:30]}'")

    print("\n[B] norm scale check (fp32 casts):")
    emb = llm.model.embed_tokens.weight.detach().float()  # (V, H)
    emb_norms = emb.norm(dim=-1)
    print(f"  LLM embed row norms : {stats(emb_norms)}")
    tok_norms = []
    with torch.no_grad():
        for i, (batch, _) in enumerate(zip(loader, samples)):
            if i >= 20:
                break
            batch = batch.to(dev)
            vp = next(vision_tower.parameters())
            vis = vision_tower(batch.pixel_values.to(vp.dtype),
                               grid_thw=batch.image_grid_thw.to(vp.device)).pooler_output
            out = projector(vis.to(dev).to(projector.net[0].weight.dtype))
            tok_norms.append(out.float().norm(dim=-1))
    tok_norms = torch.cat(tok_norms)
    print(f"  projector out norms : {stats(tok_norms)}  (n={tok_norms.numel()} tokens)")
    ratio = tok_norms.mean() / emb_norms.mean()
    print(f"  RATIO proj/llm = {ratio:.2f}x  (≈1x expected)")

    print("\n[C] projector output cross-image cos (mean-pooled):")
    P = torch.stack(proj_pooled)  # (N, H)
    Pn = P / P.norm(dim=-1, keepdim=True)
    cos = Pn @ Pn.T
    n = cos.shape[0]
    off = cos[~torch.eye(n, dtype=torch.bool, device=cos.device)]
    print(f"  pairwise cos: {stats(off)}  (v1 trained: 0.993, vision pooler: 0.95)")
    print(f"\ndone ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
