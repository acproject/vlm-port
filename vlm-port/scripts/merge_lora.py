"""Merge Stage 2 LoRA into the Nanbeige4.3-33B base weights -> F32 safetensors.

Streams tensor-by-tensor (never holds the whole model in RAM):
  for each base shard: read tensors, add LoRA deltas where applicable,
  cast to F32, write output shards of ~5 GB + a fresh index json.

Delta: W += (lora_B @ lora_A) * (alpha / r)   [computed in fp32]

Usage:
    cd vlm-port && ../venv-cu128/bin/python scripts/merge_lora.py \
        --ckpt checkpoints/stage2/ckpt_final.pt --out ../Nanbeige4.3-33B-merged
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

TARGET_SHARD_BYTES = 5 * 2**30  # 5 GiB per output shard (F32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/home/acproject/usb/transplant/Nanbeige4.3-33B")
    ap.add_argument("--ckpt", default="checkpoints/stage2/ckpt_final.pt")
    ap.add_argument("--out", default="/home/acproject/usb/transplant/Nanbeige4.3-33B-merged")
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    base_dir = Path(args.base)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    lora_sd = ck["lora_state_dict"]
    assert lora_sd, "checkpoint has no lora_state_dict"
    scaling = args.lora_alpha / args.lora_r
    print(f"ckpt step={ck.get('global_step')} | LoRA tensors: {len(lora_sd)} | scaling={scaling}")

    # map base weight name -> (A, B)
    deltas: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for k, v in lora_sd.items():
        if k.endswith(".lora_A"):
            base_k = k[: -len(".lora_A")] + ".weight"
            b_key = k[: -len(".lora_A")] + ".lora_B"
            deltas[base_k] = (v, lora_sd[b_key])
    print(f"LoRA-merged base tensors: {len(deltas)} (expect 364)")

    index = json.load(open(base_dir / "model.safetensors.index.json"))
    weight_map: dict[str, str] = index["weight_map"]
    shards = sorted(set(weight_map.values()))

    out_dtype = getattr(torch, args.dtype)
    out_index: dict[str, str] = {}
    pending: dict[str, torch.Tensor] = {}
    pending_bytes = 0
    shard_no = 0
    total_shards = (len(shards) * 4 + 2)  # rough upper bound for naming width

    def flush():
        nonlocal pending, pending_bytes, shard_no
        if not pending:
            return
        shard_no += 1
        fname = f"model-{shard_no:05d}-of-?????.safetensors"
        # placeholder naming fixed after we know the total count
        save_file(pending, str(out_dir / fname), metadata={"format": "pt"})
        for k in pending:
            out_index[k] = fname
        print(f"  wrote {fname}: {len(pending)} tensors, {pending_bytes/2**30:.2f} GiB", flush=True)
        pending = {}
        pending_bytes = 0

    n_seen = n_merged = 0
    for shard in shards:
        print(f"processing {shard} ...", flush=True)
        with safe_open(base_dir / shard, framework="pt", device="cpu") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if k in deltas:
                    A, B = deltas[k]
                    assert A.shape[1] == t.shape[1] and B.shape[0] == t.shape[0], (
                        f"shape mismatch {k}: base {tuple(t.shape)} A {tuple(A.shape)} B {tuple(B.shape)}")
                    d = (B @ A) * scaling          # fp32 (LoRA states are fp32)
                    t = t.to(torch.float32) + d    # fp32 accumulate
                    n_merged += 1
                t = t.to(out_dtype).contiguous()
                pending[k] = t
                pending_bytes += t.numel() * t.element_size()
                n_seen += 1
                if pending_bytes >= TARGET_SHARD_BYTES:
                    flush()
    flush()

    # fix shard names to the standard HF pattern
    total = shard_no
    for i in range(1, total + 1):
        old = out_dir / f"model-{i:05d}-of-?????.safetensors"
        new = out_dir / f"model-{i:05d}-of-{total:05d}.safetensors"
        old.rename(new)
    for k, v in list(out_index.items()):
        out_index[k] = v.replace("?????", f"{total:05d}")

    (out_dir / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": int(sum(t.numel() * torch.tensor([], dtype=out_dtype).element_size()
                                             for t in [torch.empty(0)])) or 0},
         "weight_map": out_index}, indent=1))
    # total_size properly:
    total_size = 0
    for i in range(1, total + 1):
        total_size += (out_dir / f"model-{i:05d}-of-{total:05d}.safetensors").stat().st_size
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total_size}, "weight_map": out_index}, indent=1))

    print(f"tensors seen={n_seen} merged={n_merged} shards={total}")
    assert n_merged == len(deltas), "some LoRA tensors were not matched to base weights!"

    # copy config / tokenizer / generation config
    for fname in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer.model",
                  "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"):
        src = base_dir / fname
        if src.exists():
            shutil.copy(src, out_dir / fname)
    # reflect the storage dtype in config
    cfg = json.load(open(out_dir / "config.json"))
    cfg["torch_dtype"] = "float32" if out_dtype == torch.float32 else "bfloat16"
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
