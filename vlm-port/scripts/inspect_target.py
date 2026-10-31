#!/usr/bin/env python
"""Inspect Nanbeige4.3-33B target text model.

Reads config.json + tokenizer_config.json + safetensors headers (no weight data)
and prints a full architecture report.

Usage:
    python inspect_target.py [MODEL_DIR]
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open

MODEL_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/acproject/usb/transplant/Nanbeige4.3-33B")


def load_json(name: str) -> dict:
    with open(MODEL_DIR / name) as f:
        return json.load(f)


def read_shapes(wanted: set[str]) -> dict[str, tuple]:
    index = load_json("model.safetensors.index.json")
    weight_map = index["weight_map"]
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

    print("\n" + "=" * 80)
    print("CONFIG (all keys)")
    for k in sorted(config):
        print(f"  {k}: {config[k]}")

    # ---------------- tensor shapes ----------------
    print("\n" + "=" * 80)
    print("KEY TENSOR SHAPES")
    wanted = {"model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"}
    m = re.match
    for n in all_names:
        if m(r"model\.layers\.0\.", n) or m(r"model\.layers\.51\.", n):
            wanted.add(n)
    shapes = read_shapes(wanted)
    for n in sorted(shapes):
        print(f"    {n}: {shapes[n]}")

    # verify all layers share the same shapes
    suffix_shapes = defaultdict(set)
    all_shapes = read_shapes(set(all_names))
    for n, s in all_shapes.items():
        mm = re.match(r"model\.layers\.(\d+)\.(.+)", n)
        if mm:
            suffix_shapes[mm.group(2)].add(s)
    print("  -- per-suffix unique shapes across all 52 layers:")
    for suf in sorted(suffix_shapes):
        sset = suffix_shapes[suf]
        flag = "OK (uniform)" if len(sset) == 1 else "!! NON-UNIFORM"
        print(f"    {suf}: {sorted(sset)}  {flag}")

    # ---------------- derived dims ----------------
    print("\n" + "=" * 80)
    print("DERIVED DIMENSIONS")
    eq = shapes["model.embed_tokens.weight"]
    lh = shapes["lm_head.weight"]
    q = shapes["model.layers.0.self_attn.q_proj.weight"]
    k = shapes["model.layers.0.self_attn.k_proj.weight"]
    v = shapes["model.layers.0.self_attn.v_proj.weight"]
    o = shapes["model.layers.0.self_attn.o_proj.weight"]
    print(f"  embed_tokens: {eq}  (vocab x hidden)")
    print(f"  lm_head:      {lh}  (vocab x hidden, tie_word_embeddings={config['tie_word_embeddings']})")
    print(f"  q_proj: {q} -> q_heads*head_dim = {q[0]} (expect {config['num_attention_heads']}*{config['head_dim']}={config['num_attention_heads']*config['head_dim']})")
    print(f"  k_proj: {k} -> kv_heads*head_dim = {k[0]} (expect {config['num_key_value_heads']}*{config['head_dim']}={config['num_key_value_heads']*config['head_dim']})")
    print(f"  v_proj: {v}")
    print(f"  o_proj: {o}")
    print(f"  GQA: {config['num_attention_heads']} query heads : {config['num_key_value_heads']} kv heads = {config['num_attention_heads']//config['num_key_value_heads']}:1")

    # rope / long context
    print("\nROPE / LONG CONTEXT")
    print(f"  rope_theta: {config['rope_theta']}  rotary_base: {config.get('rotary_base')}")
    print(f"  rope_scaling: {config['rope_scaling']}")
    print(f"  max_position_embeddings: {config['max_position_embeddings']}  seq_length: {config.get('seq_length')}")
    print(f"  => NATIVE 512K context, no YaRN/NTK scaling needed")

    # tokenizer
    print("\n" + "=" * 80)
    print("TOKENIZER")
    print(f"  tokenizer_class: {tok_config.get('tokenizer_class')}")
    print(f"  bos: {tok_config['bos_token']}  eos: {tok_config['eos_token']}  pad: {tok_config['pad_token']}")
    print(f"  model_max_length: {tok_config.get('model_max_length')}")
    atd = tok_config["added_tokens_decoder"]
    for tid in sorted(atd, key=int):
        print(f"  {tid}: {atd[tid]['content']} special={atd[tid]['special']}")
    # unused vocab slots
    used_high = max(int(t) for t in atd)
    total = config["vocab_size"]
    print(f"  vocab_size: {total}; highest used added-token id: {used_high}")
    print(f"  => unused id range above {used_high}: {total - used_high - 1} slots (potential room for vision special tokens, verify tokenizer.json first)")
    ct = tok_config.get("chat_template", "")
    print(f"  chat_template present: {bool(ct)}, len={len(ct)}")
    print("  chat_template handles multimodal content types: "
          f"{('image' in ct and 'video' in ct)}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
