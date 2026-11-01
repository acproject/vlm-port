"""Gradient sample-specificity test: does the gradient reaching the projector
carry sample-specific signal after backprop through 52 frozen bf16 LLM layers?

Evidence chain so far: v2 (3000 anti-shortcut samples) still plateaus at
train loss 8.85 — the model can't even fit the TRAINING set, regardless of
data recipe. Prime suspect: the backward pass through 52 frozen bf16 (Volta
emulated) layers destroys sample-specific signal, so the projector only
receives "average direction" gradients → output collapse (cos 0.998).

Test: pick 2 maximally different training samples (solid-color caption vs
OCR caption). Compute projector gradients for each. Compare.

  cos(gA, gB) ≈ 1.0  → no sample-specific signal (backprop degradation)
                        → projector can only learn the mean → collapse;
                          Stage-1-only (projector alone) is structurally
                          broken on this setup → need LoRA (Stage 2 early)
  cos(gA, gB) << 1    → signal exists → optimization/capacity issue

Also measures the same sample twice (determinism baseline) and compares
random-init vs v2-final projectors.

Run: cd vlm-port && ../venv-cu128/bin/python scripts/diag_grad_signal.py
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


def pick_pair(manifest: str):
    """One solid-color caption sample + one OCR caption sample."""
    solid = ocr = None
    for line in open(manifest):
        r = json.loads(line)
        if solid is None and "为主色调" in r["answer"]:
            solid = r
        if ocr is None and ("写着" in r["answer"] or "一行字" in r["answer"]):
            ocr = r
        if solid and ocr:
            break
    assert solid and ocr, "could not find sample pair"
    return solid, ocr


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

    dev = torch.device("cuda:0")

    def make_projector(final: bool):
        p = CustomProjector(
            vision_feature_dim=vinfo.out_hidden_size,
            llm_hidden_size=llm_cfg.hidden_size,
            use_input_norm=True).to(device="cuda:0", dtype=torch.float32)
        if final:
            ck = torch.load(cfg["save_dir"] + "/projector_final.pt",
                            map_location="cuda:0", weights_only=True)
            p.load_state_dict(ck["projector_state_dict"])
        return p

    a, b = pick_pair(cfg["train_manifest"])
    print(f"sample A: q='{a['question']}' a='{a['answer'][:30]}' img={a['image']}")
    print(f"sample B: q='{b['question']}' a='{b['answer'][:30]}' img={b['image']}\n")

    sub = Path("/tmp/grad_pair.jsonl")
    sub.write_text(json.dumps(a, ensure_ascii=False) + "\n" +
                   json.dumps(b, ensure_ascii=False) + "\n")
    ds = ImageTextDataset(str(sub), processor, image_dir=cfg.get("image_dir"),
                          max_text_length=cfg.get("max_text_length", 4096))
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=MultimodalDataCollator(processor), num_workers=0)
    batches = [batch.to(dev) for batch in loader]

    def grad_of(projector, batch):
        projector.zero_grad(set_to_none=True)
        te = build_multimodal_embeds(llm, vision_tower, projector, batch,
                                     torch.bfloat16, dev)
        loss = manual_ce_loss(sharded_llm_forward(llm, te), batch.labels)
        loss.backward()
        gs = [p.grad.detach().float().flatten()
              for p in projector.parameters() if p.grad is not None]
        return torch.cat(gs), loss.item()

    def cos(x, y):
        return (x @ y / (x.norm() * y.norm())).item()

    for name, final in (("random-init", False), ("v2-final", True)):
        projector = make_projector(final)
        gA, lA = grad_of(projector, batches[0])
        gB, lB = grad_of(projector, batches[1])
        gA2, lA2 = grad_of(projector, batches[0])  # determinism baseline
        print(f"[{name}] lossA={lA:.3f} lossB={lB:.3f}")
        print(f"  |gA|={gA.norm():.4f} |gB|={gB.norm():.4f}")
        print(f"  cos(gA,gA_repeat) = {cos(gA, gA2):.6f}   (determinism, expect 1.0)")
        print(f"  cos(gA,gB)        = {cos(gA, gB):.6f}   <<< sample specificity")
        rel = (gA - gB).norm() / ((gA + gB) / 2).norm()
        print(f"  ||gA-gB||/||mean|| = {rel:.4f}   (0=identical, 2=opposite)")
        print()

    print("done ({:.0f}s)".format(time.time() - t0))


if __name__ == "__main__":
    main()
