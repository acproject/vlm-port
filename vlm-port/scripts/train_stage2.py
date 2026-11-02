"""Stage 2 Joint Training: projector + LoRA on the LLM, vision tower frozen.

Stage 1 postmortem (v1 + v2 data, 4 diagnostic rounds): with a fully frozen
LLM and only the projector trainable, the projector collapses into a
"handshake" solution (cross-image cos 0.998, per-sample loss std 0.2) — a
wide basin that no lr / data / norm fix escapes. Every published VLM that
works lets the LLM adapt too. Stage 2 therefore adds LoRA (r=16) on all 52
decoder layers (q/k/v/o + gate/up/down) while keeping the 33B base frozen.

Starting point: RANDOM projector + zero-init LoRA-B → initial forward is
exactly the intact LLM (LLaVA-style joint start, no v2 collapse residue).

Deltas vs train_stage1.py:
  - apply_lora() after freeze + hook removal; LoRA params fp32, co-located
    with each layer's shard device
  - optimizer param groups: projector lr / LoRA lr (2e-4), foreach=False
    (params span cuda:0/1/2 — fused foreach ops would break)
  - clip_grad_norm_ over projector + LoRA params, foreach=False
  - checkpoints carry lora_state_dict

Run:
    cd /home/acproject/usb/transplant/vlm-port
    ../venv-cu128/bin/python scripts/train_stage2.py configs/train_stage2.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

_PROJECT_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(_PROJECT_SRC))

import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from vision.qwen_vision import load_qwen_vision_tower
from vision.projector import CustomProjector
from data.processor import NanbeigeVisionProcessor
from data.dataset import ImageTextDataset
from data.collator import MultimodalDataCollator
from modeling.sharded_llm import remove_accelerate_hooks, sharded_llm_forward
from modeling.fa_v100 import enable_fa_v100
from modeling.lora import apply_lora, lora_state_dict, load_lora_state


# ────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────

class TrainLogger:
    def __init__(self, log_file: str | None):
        self.fh = None
        if log_file:
            p = Path(log_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            self.fh = open(p, "a", buffering=1)  # line-buffered
            self.info(f"log file: {p.resolve()}")

    def info(self, msg: str, ts: bool = True):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}" if ts else msg
        print(line, flush=True)
        if self.fh:
            self.fh.write(line + "\n")

    def close(self):
        if self.fh:
            self.fh.close()


def gpu_mem_gb() -> str:
    return "/".join(f"{torch.cuda.memory_allocated(i)/2**30:.1f}" for i in range(torch.cuda.device_count()))


# ────────────────────────────────────────────────────────
# Functional blocks (autograd-safe) — identical to Stage 1
# ────────────────────────────────────────────────────────

def manual_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Shifted CE with -100 ignore. FP32 cast for stability.

    Cross-GPU label routing MUST go through host staging: this box's P2P
    copies corrupt data in every direction/dtype (verified: int64 label-
    shaped tensors 0/60 intact on 0→2). A direct labels.to(logits.device)
    silently trained on garbage in Stage 1 and hard-asserts with LoRA's
    different allocator layout.
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    if shift_labels.device != logits.device:
        shift_labels = shift_labels.cpu().to(logits.device)  # D2H+H2D, safe
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )


def inject_visual_embeddings(
    text_embeds: torch.Tensor,
    image_pad_mask: torch.Tensor,
    projected: torch.Tensor,
) -> torch.Tensor:
    """Replace image_pad embeddings with projected vision features.

    Out-of-place index_put: keeps autograd path to projector even though
    text_embeds itself doesn't require grad.
    """
    B, S, H = text_embeds.shape
    flat = text_embeds.reshape(-1, H)
    idx = image_pad_mask.reshape(-1).nonzero(as_tuple=True)[0]
    assert idx.numel() == projected.shape[0], (
        f"injection mismatch: {idx.numel()} pad positions vs {projected.shape[0]} projected tokens"
    )
    flat = torch.index_put(flat, (idx,), projected.to(flat.dtype))
    return flat.reshape(B, S, H)


def build_multimodal_embeds(
    llm, vision_tower, projector, batch, llm_dtype: torch.dtype,
    compute_device: torch.device,
) -> torch.Tensor:
    """embed_tokens → vision forward (no grad) → projector → injection."""
    embed_dev = llm.model.embed_tokens.weight.device
    text_embeds = (
        llm.model.embed_tokens(batch.input_ids.to(embed_dev))
        .to(llm_dtype)
        .to(compute_device)
    )

    if batch.pixel_values.numel() > 0 and batch.image_pad_mask.any().item():
        vp = next(vision_tower.parameters())
        with torch.no_grad():
            vis = vision_tower(
                batch.pixel_values.to(vp.dtype),
                grid_thw=batch.image_grid_thw.to(vp.device),
            ).pooler_output
        vis = vis.to(compute_device)
        projected = projector(vis.to(projector.net[0].weight.dtype)).to(llm_dtype)
        text_embeds = inject_visual_embeddings(text_embeds, batch.image_pad_mask, projected)
    return text_embeds


# ────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────

def evaluate(llm, vision_tower, projector, val_loader, llm_dtype, compute_device):
    """Forward-only avg loss on val set. LoRA acts inside sharded_llm_forward."""
    was_training = projector.training
    projector.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(compute_device)
            text_embeds = build_multimodal_embeds(
                llm, vision_tower, projector, batch, llm_dtype, compute_device)
            logits = sharded_llm_forward(llm, text_embeds)
            total += manual_ce_loss(logits, batch.labels).item()
            n += 1
    if was_training:
        projector.train()
    return total / max(n, 1)


def save_ckpt(save_dir: Path, name: str, projector, optimizer, global_step,
              epoch, lora_sd=None):
    ckpt = save_dir / name
    torch.save({
        "projector_state_dict": projector.state_dict(),
        "lora_state_dict": lora_sd,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "global_step": global_step,
        "epoch": epoch,
    }, ckpt)
    print(f"  ✓ saved {ckpt}", flush=True)


def train(config_path: str):
    cfg = yaml.safe_load(open(config_path))

    device = cfg.get("device", "cpu")
    use_cuda = device.startswith("cuda")
    llm_dtype = getattr(torch, cfg.get("dtype", "float16" if use_cuda else "bfloat16"))
    proj_dtype = getattr(torch, cfg.get("projector_dtype", "float32"))
    compute_device = torch.device("cuda:0" if use_cuda else "cpu")

    lr = cfg.get("learning_rate", 1e-4)
    lora_lr = cfg.get("lora_learning_rate", 2e-4)
    lora_r = cfg.get("lora_r", 16)
    lora_alpha = cfg.get("lora_alpha", 32)
    lora_dropout = cfg.get("lora_dropout", 0.0)
    epochs = cfg.get("epochs", 1)
    batch_size = cfg.get("batch_size", 1)
    grad_accum = cfg.get("gradient_accumulation_steps", 8)
    warmup_steps = cfg.get("warmup_steps", 50)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)
    save_dir = Path(cfg.get("save_dir", "checkpoints/stage2"))
    save_dir.mkdir(parents=True, exist_ok=True)
    log_interval = cfg.get("log_interval", 10)
    save_interval = cfg.get("save_interval", 1000)
    eval_interval = cfg.get("eval_interval", 0) or 10**9
    max_steps = cfg.get("max_steps")
    max_minutes = cfg.get("max_minutes")
    num_threads = cfg.get("num_threads", 0) or 8
    log_micro = cfg.get("log_micro", False)
    logger = TrainLogger(cfg.get("log_file"))

    logger.info("=" * 72)
    logger.info(f"Stage 2 Joint Training (projector + LoRA) — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  device={device} llm_dtype={llm_dtype} proj_dtype={proj_dtype}")
    logger.info(f"  lr(projector)={lr} lr(LoRA)={lora_lr} lora_r={lora_r} "
                f"lora_alpha={lora_alpha} dropout={lora_dropout}")
    logger.info(f"  epochs={epochs} batch_size={batch_size} grad_accum={grad_accum} "
                f"warmup={warmup_steps} max_grad_norm={max_grad_norm} seed={cfg.get('seed', 42)}")
    logger.info(f"  max_steps={max_steps} max_minutes={max_minutes} "
                f"log_micro={log_micro} use_fa_v100={cfg.get('use_fa_v100', False)}")
    logger.info(f"  config={config_path}")
    logger.info("=" * 72)

    torch.manual_seed(cfg.get("seed", 42))
    torch.set_num_threads(num_threads)
    if batch_size != 1:
        raise ValueError(
            f"batch_size={batch_size}: sharded_llm_forward only supports "
            "batch=1 unpadded causal sequences (no attention_mask); "
            "use gradient_accumulation_steps for effective batch size."
        )
    if use_cuda:
        # V100 + cuDNN 9 (bundled with torch cu128): Volta conv kernels removed
        # → "GET was unable to find an engine". Only conv in the whole model is
        # the ViT patch_embed (equivalent to one matmul); ATen fallback is fine.
        torch.backends.cudnn.enabled = False

    qwen_dir = cfg["qwen_model_dir"]
    nanbeige_dir = cfg["llm_model_dir"]

    # ── 1. Vision tower (frozen, fp32 on cuda:0) ──
    logger.info("[1/5] Loading Qwen vision tower (fp32, frozen)...", ts=False)
    t0 = time.time()
    vision_tower, vision_info = load_qwen_vision_tower(
        qwen_dir, device="cuda:0" if use_cuda else "cpu",
        dtype=torch.float32,
    )
    logger.info(f"  OK {time.time()-t0:.1f}s | vision_feature_dim={vision_info.out_hidden_size}")

    # ── 2. Nanbeige LLM (base frozen, fp16/bf16, sharded across GPUs) ──
    logger.info("[2/5] Loading Nanbeige LLM (base frozen, sharded)...", ts=False)
    t0 = time.time()
    from transformers import LlamaForCausalLM, AutoConfig
    llm_cfg = AutoConfig.from_pretrained(nanbeige_dir)
    device_map, max_memory = None, None
    if use_cuda and torch.cuda.device_count() >= 2:
        device_map = "auto"
        n_gpu = torch.cuda.device_count()
        _budget = {0: "20GiB", 1: "27GiB", 2: "25GiB"}
        max_memory = {i: _budget.get(i, "25GiB") for i in range(n_gpu)}
    llm = LlamaForCausalLM.from_pretrained(
        nanbeige_dir, config=llm_cfg, torch_dtype=llm_dtype,
        low_cpu_mem_usage=True, device_map=device_map, max_memory=max_memory,
    )
    if device_map is None:
        llm = llm.to(device)
    for p in llm.parameters():
        p.requires_grad = False  # MUST precede backward AND apply_lora
    llm.eval()
    remove_accelerate_hooks(llm)  # their cross-GPU .to() uses broken P2P
    if use_cuda and cfg.get("use_fa_v100", False):
        enable_fa_v100(llm)
    n_llm = sum(p.numel() for p in llm.parameters())  # pre-LoRA: base only
    logger.info(f"  OK {time.time()-t0:.1f}s | {n_llm/1e9:.2f}B base params frozen | "
                f"hidden={llm_cfg.hidden_size} layers={llm_cfg.num_hidden_layers} "
                f"heads={llm_cfg.num_attention_heads}:{llm_cfg.num_key_value_heads} "
                f"attn_impl={llm.config._attn_implementation}")
    if device_map:
        split = Counter(str(v) for v in llm.hf_device_map.values())
        logger.info(f"  shard split: {dict(split)} | mem {gpu_mem_gb()} GiB "
                    f"allocated after load")

    # ── 3. LoRA on every decoder layer ──
    logger.info(f"[3/5] Applying LoRA r={lora_r} alpha={lora_alpha} "
                f"(q/k/v/o + gate/up/down × {llm_cfg.num_hidden_layers} layers)...", ts=False)
    t0 = time.time()
    lora_modules = apply_lora(llm, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
    # NB: m.parameters() would include the frozen base weights — collect
    # exactly the two trainable tensors per module.
    lora_params = [p for m in lora_modules for p in (m.lora_A, m.lora_B)]
    for p in lora_params:
        assert p.requires_grad and p.dtype == torch.float32
    n_lora = sum(p.numel() for p in lora_params)
    dev_counter = Counter(str(p.device) for p in lora_params)
    logger.info(f"  OK {time.time()-t0:.1f}s | {len(lora_modules)} LoRALinear modules | "
                f"{n_lora/1e6:.2f}M params (fp32, B zero-init) | by device {dict(dev_counter)}")

    # ── 4. Projector (trainable, fp32, cuda:0) — random init by default ──
    logger.info("[4/5] Loading Custom Projector...", ts=False)
    projector = CustomProjector(
        vision_feature_dim=vision_info.out_hidden_size,   # 5120
        llm_hidden_size=llm_cfg.hidden_size,              # 7168
        act_fn=cfg.get("projector_act_fn", "gelu"),
        use_input_norm=cfg.get("projector_use_input_norm", True),
    ).to(device=compute_device, dtype=proj_dtype)
    if cfg.get("init_projector_from"):
        sd = torch.load(cfg["init_projector_from"], map_location=compute_device,
                        weights_only=True)
        projector.load_state_dict(sd["projector_state_dict"])
        logger.info(f"  init projector from {cfg['init_projector_from']} "
                    f"(step {sd.get('global_step')})")
    n_proj = sum(p.numel() for p in projector.parameters())
    logger.info(f"  Trainable: projector {n_proj/1e6:.2f}M + LoRA {n_lora/1e6:.2f}M "
                f"= {(n_proj+n_lora)/1e6:.2f}M")

    # ── 5. Data ──
    logger.info("[5/5] Loading data...", ts=False)
    processor = NanbeigeVisionProcessor(qwen_dir, nanbeige_dir)
    collator = MultimodalDataCollator(processor)
    train_ds = ImageTextDataset(
        cfg["train_manifest"], processor,
        image_dir=cfg.get("image_dir"), max_text_length=cfg.get("max_text_length", 4096))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collator, num_workers=0)
    logger.info(f"  Train samples: {len(train_ds)}")

    val_loader = None
    if cfg.get("val_manifest"):
        val_ds = ImageTextDataset(
            cfg["val_manifest"], processor,
            image_dir=cfg.get("image_dir"), max_text_length=cfg.get("max_text_length", 4096))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                collate_fn=collator, num_workers=0)
        logger.info(f"  Val samples: {len(val_ds)}")

    # ── Optimizer / scheduler / scaler ──
    # foreach=False everywhere: params span cuda:0/1/2 (LoRA co-located with
    # layer shards) — fused multi-tensor ops require a single device.
    optimizer = torch.optim.AdamW(
        [
            {"params": list(projector.parameters()), "lr": lr,
             "weight_decay": cfg.get("weight_decay", 0.01)},
            {"params": lora_params, "lr": lora_lr,
             "weight_decay": cfg.get("lora_weight_decay", 0.0)},
        ],
        lr=lr, foreach=False,
    )
    total_steps = len(train_loader) * epochs // grad_accum
    if max_steps:
        total_steps = min(total_steps, max_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    # GradScaler only for fp16 LLM. bf16 (emulated on Volta) needs no scaling.
    use_scaler = use_cuda and llm_dtype == torch.float16
    if use_cuda and llm_dtype == torch.float16:
        logger.info("WARNING: fp16 LLM — CE gradients may underflow through 52 "
                    "layers. Prefer bfloat16.")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    train_params = list(projector.parameters()) + lora_params

    logger.info("=" * 72)
    logger.info(f"Starting training: {total_steps} optimizer steps "
                f"({len(train_loader)} micro-batches/epoch × {epochs} epochs / {grad_accum})")
    logger.info("=" * 72)

    global_step = 0
    start_epoch = 0
    if cfg.get("resume_from"):
        ck = torch.load(cfg["resume_from"], map_location=compute_device,
                        weights_only=True)
        projector.load_state_dict(ck["projector_state_dict"])
        load_lora_state(llm, ck["lora_state_dict"])
        if ck.get("optimizer_state_dict"):
            optimizer.load_state_dict(ck["optimizer_state_dict"])
        global_step = ck.get("global_step", 0)
        start_epoch = ck.get("epoch", 0)
        for _ in range(global_step):  # fast-forward the cosine schedule
            scheduler.step()
        logger.info(f"resumed from {cfg['resume_from']}: step={global_step} "
                    f"epoch={start_epoch} (that epoch restarts from batch 0)")

    t_start = time.time()
    t_log = time.time()
    running_loss, running_n = 0.0, 0
    last_val = None
    best_val = float("inf")
    stop = False

    for epoch in range(start_epoch, epochs):
        if stop:
            break
        projector.train()
        optimizer.zero_grad()
        logger.info(f"── epoch {epoch+1}/{epochs} start ──")

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(compute_device)

            text_embeds = build_multimodal_embeds(
                llm, vision_tower, projector, batch, llm_dtype, compute_device)
            logits = sharded_llm_forward(llm, text_embeds)
            loss = manual_ce_loss(logits, batch.labels)

            if not torch.isfinite(loss):
                logger.info(f"[ABORT] non-finite loss at ep{epoch} b{batch_idx}: "
                            f"{loss.item()}")
                raise RuntimeError(
                    f"Non-finite loss at micro-batch {batch_idx} epoch {epoch}. "
                    f"loss={loss.item()}"
                )

            if log_micro:
                S = batch.input_ids.shape[1]
                vtok = int(batch.image_pad_mask.sum().item()) if batch.image_pad_mask is not None else 0
                n_resp = int((batch.labels != -100).sum().item())
                logger.info(f"[micro] ep{epoch} b{batch_idx:03d} S={S:<4d} vtok={vtok:<3d} "
                            f"resp={n_resp:<3d} loss={loss.item():.4f}")

            scaler.scale(loss / grad_accum).backward()
            running_loss += loss.item()
            running_n += 1

            if (batch_idx + 1) % grad_accum != 0:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                train_params, max_grad_norm, foreach=False)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % log_interval == 0:
                dt = time.time() - t_log
                logger.info(
                    f"[step {global_step:>4}/{total_steps}] "
                    f"loss={running_loss/max(running_n,1):.4f} | "
                    f"lr_p={scheduler.get_last_lr()[0]:.2e} "
                    f"lr_l={scheduler.get_last_lr()[1]:.2e} | "
                    f"gnorm={float(grad_norm):.3f} | "
                    f"{dt/log_interval:.1f}s/step | "
                    f"mem={gpu_mem_gb()}G | "
                    f"elapsed={(time.time()-t_start)/60:.1f}min")
                running_loss, running_n = 0.0, 0
                t_log = time.time()

            if save_interval and global_step % save_interval == 0:
                save_ckpt(save_dir, f"ckpt_step{global_step}.pt",
                          projector, optimizer, global_step, epoch,
                          lora_sd=lora_state_dict(llm))

            if eval_interval and val_loader and global_step % eval_interval == 0:
                vloss = evaluate(llm, vision_tower, projector, val_loader,
                                 llm_dtype, compute_device)
                delta = f" Δ={vloss-last_val:+.4f}" if last_val is not None else ""
                best = " ★best" if vloss < best_val else ""
                logger.info(f"[val] step {global_step} VAL loss={vloss:.4f}{delta}{best}")
                last_val, best_val = vloss, min(best_val, vloss)

            if max_steps and global_step >= max_steps:
                stop = True
                break
            if max_minutes and (time.time() - t_start) > max_minutes * 60:
                logger.info(f"max_minutes={max_minutes} reached, stopping")
                stop = True
                break

        if not stop:
            logger.info(f"── epoch {epoch+1}/{epochs} done | "
                        f"elapsed={(time.time()-t_start)/60:.1f}min ──")

    # ── Final: val + save ──
    if val_loader:
        vloss = evaluate(llm, vision_tower, projector, val_loader,
                         llm_dtype, compute_device)
        delta = f" Δ={vloss-last_val:+.4f}" if last_val is not None else ""
        logger.info(f"[val] FINAL VAL loss={vloss:.4f}{delta} "
                    f"(best seen: {best_val:.4f})")
    save_ckpt(save_dir, "ckpt_final.pt", projector, optimizer, global_step,
              epoch, lora_sd=lora_state_dict(llm))

    logger.info("=" * 72)
    logger.info(f"Training complete! {global_step} optimizer steps, "
                f"{(time.time()-t_start)/60:.1f} min")
    logger.info(f"Final checkpoint: {save_dir / 'ckpt_final.pt'}")
    logger.info(f"Checkpoints: {sorted(p.name for p in save_dir.glob('ckpt_*.pt'))}")
    logger.info("=" * 72)
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()
    train(args.config)
