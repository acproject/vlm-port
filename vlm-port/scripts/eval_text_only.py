"""Text-only upper-bound test: can the frozen LLM predict val answers from
question text ALONE (no vision tokens at all)?

Same val set, same response-masked CE, three conditions:
  1. uniform baseline  ln(vocab) = 12.03
  2. --mode text       question rendered as plain chat text (no image, no vision tokens)
  3. --mode mm         full multimodal pipeline (vision tower + stage1 projector)

Interpretation guide:
  text ≈ 12        answers unpredictable from text (expected: color/OCR/shape
                   questions NEED the image) → any mm loss < 12 is real visual signal
  text ≈ mm ≈ 9.1  vision tokens add nothing over text prior → structural plateau
                   (LLM can't read projector output, or translation destroys signal)
  text < mm        projector output actively hurts vs just ignoring the image

Run (from vlm-port/):
  ../venv-cu128/bin/python scripts/eval_text_only.py --mode text
  ../venv-cu128/bin/python scripts/eval_text_only.py --mode mm
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT / "src"))
sys.path.insert(0, str(_PROJECT / "scripts"))

import torch

from vision.qwen_vision import load_qwen_vision_tower
from vision.projector import CustomProjector
from data.processor import NanbeigeVisionProcessor
from data.dataset import ImageTextDataset
from modeling.sharded_llm import remove_accelerate_hooks, sharded_llm_forward
from train_stage1 import manual_ce_loss, build_multimodal_embeds

NANBEIGE_DIR = "/home/acproject/usb/transplant/Nanbeige4.3-33B"
QWEN_DIR = "/home/acproject/usb/transplant/Qwen3.8-27B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["text", "mm"], default="text",
                    help="text = question only (no vision tokens); mm = full pipeline")
    ap.add_argument("--val-manifest", default="data/val_manifest.jsonl")
    ap.add_argument("--image-dir", default="data")
    ap.add_argument("--projector-ckpt", default="checkpoints/stage1/projector_final.pt")
    args = ap.parse_args()

    torch.backends.cudnn.enabled = False  # same Volta conv workaround as training
    t_start = time.time()

    processor = NanbeigeVisionProcessor(QWEN_DIR, NANBEIGE_DIR)
    samples = [json.loads(l) for l in open(args.val_manifest) if l.strip()]
    print(f"val samples: {len(samples)} | mode={args.mode}")

    # ── LLM: bf16, sharded, identical to training config ──
    from transformers import LlamaForCausalLM, AutoConfig
    llm_cfg = AutoConfig.from_pretrained(NANBEIGE_DIR)
    llm = LlamaForCausalLM.from_pretrained(
        NANBEIGE_DIR, config=llm_cfg, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map="auto",
        max_memory={0: "20GiB", 1: "27GiB", 2: "25GiB"},
    )
    llm.eval()
    remove_accelerate_hooks(llm)
    print(f"LLM loaded ({time.time()-t_start:.1f}s)")

    # ── vision + projector only for mm mode ──
    vision_tower = projector = None
    if args.mode == "mm":
        vision_tower, vision_info = load_qwen_vision_tower(
            QWEN_DIR, device="cuda:0", dtype=torch.float32)
        projector = CustomProjector(
            vision_feature_dim=vision_info.out_hidden_size,
            llm_hidden_size=llm_cfg.hidden_size,
            act_fn="gelu", use_input_norm=True,
        ).to(device=torch.device("cuda:0"), dtype=torch.float32)
        sd = torch.load(args.projector_ckpt, map_location="cuda:0", weights_only=True)
        projector.load_state_dict(sd["projector_state_dict"])
        projector.eval()
        print(f"projector loaded from {args.projector_ckpt} (step {sd.get('global_step')})")

    # ── eval loop ──
    ds = ImageTextDataset(args.val_manifest, processor,
                          image_dir=args.image_dir, max_text_length=4096)
    compute_device = torch.device("cuda:0")
    losses = []
    with torch.no_grad():
        for i, sample in enumerate(samples):
            if args.mode == "text":
                # text-only: no image, question as plain string content
                item = ds._build_tokens(
                    sample.get("question", "Describe this image."),
                    sample.get("answer", ""), None)
                input_ids = torch.tensor([item["input_ids"]], dtype=torch.long)
                labels = torch.tensor([item["labels"]], dtype=torch.long)
                embed_dev = llm.model.embed_tokens.weight.device
                text_embeds = (
                    llm.model.embed_tokens(input_ids.to(embed_dev))
                    .to(torch.bfloat16).to(compute_device))
                logits = sharded_llm_forward(llm, text_embeds)
                loss = manual_ce_loss(logits, labels).item()
                seq_len = input_ids.shape[1]
            else:
                # multimodal: full pipeline via dataset __getitem__ (loads image)
                item = ds[i]
                from data.collator import MultimodalDataCollator
                batch = MultimodalDataCollator(processor)([item]).to(compute_device)
                text_embeds = build_multimodal_embeds(
                    llm, vision_tower, projector, batch,
                    torch.bfloat16, compute_device)
                logits = sharded_llm_forward(llm, text_embeds)
                loss = manual_ce_loss(logits, batch.labels).item()
                seq_len = batch.input_ids.shape[1]

            losses.append(loss)
            q = sample.get("question", "")[:30]
            a = sample.get("answer", "")[:20]
            print(f"  [{i+1:2d}/{len(samples)}] S={seq_len:3d} loss={loss:7.4f} "
                  f"q='{q}' a='{a}'", flush=True)

    import math
    mean = sum(losses) / len(losses)
    print("=" * 60)
    print(f"MODE={args.mode}  n={len(losses)}")
    print(f"  mean loss      = {mean:.4f}  (PPL {math.exp(mean):.1f})")
    print(f"  uniform approx = {math.log(166200):.4f}  (ln vocab)")
    print(f"  mm baseline    = 9.1132  (stage1 final, teacher-forced)")
    print(f"  elapsed {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
