"""Export mmproj GGUF for the VLM port (Qwen3.5 vision + custom projector).

Pipeline:
  1. Stage a synthetic HF dir (Qwen3_5ForConditionalGeneration) holding only
     model.visual.* tensors from Qwen3.8-27B — the tower INCLUDING its native
     merger (merger output = 5120).
  2. Run llama.cpp convert_hf_to_gguf.py --mmproj on it (official code path:
     tensor naming + all clip.* metadata) -> base mmproj (qwen3vl_merger).
  3. Rewrite the GGUF: clip.projector_type -> "qwen3vl_mlp" (our patched
     graph: ViT + merger + RMSNorm + Linear + GELU + Linear), appending the
     trained CustomProjector tensors as F16:
        mm.input_norm.weight  <- projector RMSNorm(5120)
        mm.4.weight/bias      <- Linear(5120->7168)
        mm.6.weight/bias      <- Linear(7168->7168)

Usage:
    cd vlm-port && ../venv-cu128/bin/python scripts/export_mmproj.py \
        --ckpt checkpoints/stage2/ckpt_final.pt \
        --out /home/acproject/usb/transplant/gguf/mmproj-NanbeigeVL-F16.gguf
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

LLAMA_CPP = Path("/home/acproject/workspace/cpp_projects/llama.cpp")
QWEN_DIR = Path("/home/acproject/usb/transplant/Qwen3.8-27B")
STAGE_DIR = Path("/tmp/qwen35_vis_hf")
BASE_MMPROJ = Path("/tmp/mmproj_base.gguf")

sys.path.insert(0, str(LLAMA_CPP / "gguf-py"))
import gguf  # noqa: E402


def stage_hf_dir():
    cfg = json.load(open(QWEN_DIR / "config.json"))
    vc = cfg["vision_config"]
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    (STAGE_DIR / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {"hidden_size": 7168, "rms_norm_eps": 1e-6},
        "vision_config": vc,
    }, indent=1))

    index = json.load(open(QWEN_DIR / "model.safetensors.index.json"))
    wm = index["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for k, f in wm.items():
        if k.startswith("model.visual."):
            by_shard.setdefault(f, []).append(k)
    state = {}
    for fname, keys in by_shard.items():
        with safe_open(QWEN_DIR / fname, framework="pt", device="cpu") as f:
            for k in keys:
                state[k] = f.get_tensor(k)
    print(f"staged {len(state)} vision tensors -> {STAGE_DIR}")
    save_file(state, str(STAGE_DIR / "model.safetensors"), metadata={"format": "pt"})
    # converter reads image_mean / image_std / patch size from here
    for fname in ("preprocessor_config.json",):
        src = QWEN_DIR / fname
        if src.exists():
            import shutil
            shutil.copy(src, STAGE_DIR / fname)


def run_converter():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(LLAMA_CPP / "gguf-py")
    cmd = [sys.executable, str(LLAMA_CPP / "convert_hf_to_gguf.py"), str(STAGE_DIR),
           "--mmproj", "--outfile", str(BASE_MMPROJ), "--outtype", "f16"]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def rewrite_with_projector(ckpt_path: Path, out_path: Path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    psd = ck["projector_state_dict"]
    keys = list(psd.keys())
    print("projector keys:", keys)
    # CustomProjector(use_input_norm=True): net.0 RMSNorm, net.1 Linear, net.3 Linear
    assert "net.0.weight" in psd and "net.1.weight" in psd and "net.3.weight" in psd, \
        f"unexpected projector layout: {keys}"
    rms_w = psd["net.0.weight"]                      # (5120,)
    fc1_w, fc1_b = psd["net.1.weight"], psd["net.1.bias"]   # (7168, 5120), (7168,)
    fc2_w, fc2_b = psd["net.3.weight"], psd["net.3.bias"]   # (7168, 7168), (7168,)
    assert rms_w.shape[0] == fc1_w.shape[1] and fc1_w.shape[0] == fc2_w.shape[1] == fc2_w.shape[0]
    print(f"projector: RMSNorm({rms_w.shape[0]}) + {tuple(fc1_w.shape)} + {tuple(fc2_w.shape)}")

    reader = gguf.GGUFReader(str(BASE_MMPROJ), "r")
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()

    writer = gguf.GGUFWriter(str(out_path), arch=arch, endianess=reader.endianess)

    # copy all metadata, overriding the projector type
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        if field.name == "clip.projector_type":
            writer.add_key_value(field.name, "qwen3vl_mlp", gguf.GGUFValueType.STRING)
            print("clip.projector_type -> qwen3vl_mlp")
            continue
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    # copy all existing tensors (info first, then data, like gguf_new_metadata.py)
    for tensor in reader.tensors:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                               tensor.data.nbytes, tensor.tensor_type)

    # append the custom projector tensors (F16)
    def f16(t: torch.Tensor) -> np.ndarray:
        return t.to(torch.float16).numpy()

    extra = [
        ("mm.input_norm.weight", f16(rms_w)),
        ("mm.4.weight", f16(fc1_w)), ("mm.4.bias", f16(fc1_b)),
        ("mm.6.weight", f16(fc2_w)), ("mm.6.bias", f16(fc2_b)),
    ]
    for name, arr in extra:
        writer.add_tensor_info(name, arr.shape, arr.dtype, arr.nbytes)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    for _, arr in extra:
        writer.write_tensor_data(arr)
    writer.close()
    print(f"wrote {out_path} ({out_path.stat().st_size/2**30:.2f} GiB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/stage2/ckpt_final.pt")
    ap.add_argument("--out", default="/home/acproject/usb/transplant/gguf/mmproj-NanbeigeVL-F16.gguf")
    ap.add_argument("--skip-converter", action="store_true",
                    help="reuse an existing /tmp/mmproj_base.gguf")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_converter:
        stage_hf_dir()
        run_converter()
    rewrite_with_projector(Path(args.ckpt), out_path)


if __name__ == "__main__":
    main()
