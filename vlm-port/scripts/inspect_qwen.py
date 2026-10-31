#!/usr/bin/env python
"""Inspect Qwen3.8-27B multimodal model.

Reads config.json + tokenizer_config.json + safetensors headers (no weight data)
and prints a full architecture report for the vision tower / merger / text model.

Usage:
    python inspect_qwen.py [MODEL_DIR]
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open

MODEL_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/acproject/usb/transplant/Qwen3.8-27B")


def load_json(name: str) -> dict:
    with open(MODEL_DIR / name) as f:
        return json.load(f)


def read_shapes(wanted: set[str]) -> dict[str, tuple]:
    """Read tensor shapes from safetensors headers only (memory-safe)."""
    index = load_json("model.safetensors.index.json")
    weight_map = index["weight_map"]
    # group wanted tensors by file
    by_file = defaultdict(list)
    for name, fname in weight_map.items():
        if name in wanted:
            by_file[fname].append(name)
    shapes = {}
    for fname, names in by_file.items():
        with safe_open(MODEL_DIR / fname, framework="pt") as f:
            for name in names:
                shapes[name] = tuple(f.get_slice(name).get_shape())
    return shapes


def main():
    config = load_json("config.json")
    tok_config = load_json("tokenizer_config.json")
    index = load_json("model.safetensors.index.json")
    weight_map = index["weight_map"]
    all_names = sorted(weight_map.keys())

    print("=" * 80)
    print(f"MODEL DIR: {MODEL_DIR}")
    print(f"architectures: {config['architectures']}  model_type: {config['model_type']}")
    print(f"total tensors: {len(all_names)}   total_size: {index['metadata']['total_size'] / 1e9:.2f} GB")

    # ---------------- vision config ----------------
    vc = config["vision_config"]
    print("\n" + "=" * 80)
    print("VISION CONFIG")
    for k in sorted(vc):
        print(f"  {k}: {vc[k]}")

    # ---------------- text config ----------------
    tc = config["text_config"]
    print("\n" + "=" * 80)
    print("TEXT CONFIG (key fields)")
    for k in [
        "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "head_dim", "intermediate_size", "vocab_size", "max_position_embeddings",
        "partial_rotary_factor", "rope_parameters", "full_attention_interval",
        "linear_conv_kernel_dim", "linear_key_head_dim", "linear_num_key_heads",
        "linear_num_value_heads", "linear_value_head_dim", "mtp_num_hidden_layers",
        "tie_word_embeddings", "hidden_act",
    ]:
        if k in tc:
            v = tc[k]
            if k == "layer_types" or k == "rope_parameters":
                v = "...(see config)" if k == "layer_types" else v
            print(f"  {k}: {v}")
    lt = tc["layer_types"]
    print(f"  layer_types: {lt.count('linear_attention')} linear_attention + {lt.count('full_attention')} full_attention, interval={tc['full_attention_interval']}")

    # ---------------- special tokens ----------------
    print("\n" + "=" * 80)
    print("SPECIAL TOKEN IDS (config.json)")
    for k in ["vision_start_token_id", "vision_end_token_id", "image_token_id", "video_token_id"]:
        print(f"  {k}: {config[k]}")
    print("TOKENIZER")
    print(f"  eos: {tok_config['eos_token']}  pad: {tok_config['pad_token']}")
    atd = tok_config["added_tokens_decoder"]
    for tid in sorted(atd, key=int):
        if int(tid) >= 248040:
            print(f"  {tid}: {atd[tid]['content']}")

    # ---------------- vision tower shapes ----------------
    print("\n" + "=" * 80)
    print("VISION TOWER TENSOR SHAPES")
    vis_names = [n for n in all_names if n.startswith("model.visual.")]
    shapes = read_shapes(set(vis_names))

    # print unique per-pattern shapes
    pattern_re = re.compile(r"model\.visual\.blocks\.(\d+)\.(.+)")
    block_patterns = defaultdict(set)
    for n, s in shapes.items():
        m = pattern_re.match(n)
        if m:
            block_patterns[m.group(2)].add((n, s))
    n_blocks = len({pattern_re.match(n).group(1) for n in shapes if pattern_re.match(n)})
    print(f"  num vision blocks: {n_blocks}  (depth in config: {vc['depth']})")
    print("  -- per-block unique tensor shapes (block 0 shown as representative):")
    for pat in sorted(block_patterns):
        for n, s in sorted(block_patterns[pat]):
            if ".blocks.0." in n or len(block_patterns[pat]) == 1:
                print(f"    {n}: {s}")

    print("  -- non-block vision tensors:")
    for n in sorted(shapes):
        if ".blocks." not in n:
            print(f"    {n}: {shapes[n]}")

    # ---------------- text model shapes ----------------
    print("\n" + "=" * 80)
    print("TEXT MODEL TENSOR SHAPES (key tensors)")
    text_names = [n for n in all_names if not n.startswith("model.visual.")]
    wanted_text = set()
    for n in text_names:
        if re.match(r"model\.language_model\.embed_tokens\.weight$", n) or re.match(r"lm_head\.weight$", n) or re.match(r"model\.language_model\.norm\.", n):
            wanted_text.add(n)
        m = re.match(r"model\.language_model\.layers\.(\d+)\.(.+)", n)
        if m and m.group(1) in ("0", "3", "63"):  # 0=linear_attn, 3=full_attention, 63=last
            wanted_text.add(n)
        if "mtp" in n.lower():
            wanted_text.add(n)
    tshapes = read_shapes(wanted_text)
    for n in sorted(tshapes):
        print(f"    {n}: {tshapes[n]}")

    # all unique text-layer tensor name suffixes
    suffixes = defaultdict(list)
    for n in text_names:
        m = re.match(r"model\.language_model\.layers\.(\d+)\.(.+)", n)
        if m:
            suffixes[m.group(2)].append(int(m.group(1)))
    print("  -- text layer tensor name suffixes (suffix: #layers having it):")
    for suf in sorted(suffixes):
        print(f"    {suf}: {len(suffixes[suf])}")

    # ---------------- 5120 verification ----------------
    print("\n" + "=" * 80)
    print("DIMENSION VERIFICATION: what is 5120?")
    print(f"  vision hidden_size (ViT width):        {vc['hidden_size']}")
    print(f"  vision out_hidden_size (merger out):   {vc['out_hidden_size']}")
    print(f"  text hidden_size (LLM width):          {tc['hidden_size']}")
    mfc1 = shapes.get("model.visual.merger.linear_fc1.weight")
    mfc2 = shapes.get("model.visual.merger.linear_fc2.weight")
    mn = shapes.get("model.visual.merger.norm.weight")
    print(f"  merger.linear_fc1.weight: {mfc1}")
    print(f"  merger.linear_fc2.weight: {mfc2}")
    print(f"  merger.norm.weight:       {mn}")
    if mfc2:
        print(f"  => merger OUTPUT dim (into Qwen LLM space): {mfc2[0]}")

    # ---------------- visual token count ----------------
    print("\n" + "=" * 80)
    print("VISUAL TOKEN COUNT (patch=16, temporal=2, merge=2 => 32x32 px per token)")
    patch, merge = vc["patch_size"], vc["spatial_merge_size"]
    stride = patch * merge
    for (h, w) in [(672, 672), (768, 768), (1024, 1024), (1280, 1280), (1568, 1568), (2048, 2048)]:
        gh = -(-h // stride)
        gw = -(-w // stride)
        print(f"  image {h}x{w} -> aligned {gh*stride}x{gw*stride} -> {gh}x{gw} = {gh*gw} visual tokens")
    pe = shapes.get("model.visual.pos_embed.weight")
    print(f"  pos_embed: {pe}  (num_position_embeddings config = {vc['num_position_embeddings']} = "
          f"{int(vc['num_position_embeddings']**0.5)}x{int(vc['num_position_embeddings']**0.5)} patch grid native)")

    print("\nDONE.")


if __name__ == "__main__":
    main()
