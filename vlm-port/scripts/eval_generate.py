"""P4: Generative validation of Stage 1 — greedy decode answers for val images.

Loads the trained projector checkpoint and autoregressively generates answers
(no KV cache: full re-forward per token via sharded_llm_forward, ~2s/token).

Run: cd vlm-port && ../venv-cu128/bin/python scripts/eval_generate.py
"""
from __future__ import annotations

import json
import sys
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
from train_stage1 import build_multimodal_embeds

torch.backends.cudnn.enabled = False

CFG_PATH = Path(__file__).resolve().parent.parent / "configs/train_stage1.yaml"
cfg = yaml.safe_load(open(CFG_PATH))
qwen_dir, nanbeige_dir = cfg["qwen_model_dir"], cfg["llm_model_dir"]
PER_CAT = 3

# ── pick samples: 3 per task category ──
cats: dict[str, list] = {"颜色": [], "形状": [], "几个": [], "文字": []}
for line in open(cfg["val_manifest"]):
    r = json.loads(line)
    for k in cats:
        if k in r["question"] and len(cats[k]) < PER_CAT:
            cats[k].append(r)
            break
subset = [r for v in cats.values() for r in v]
sub_path = Path("/tmp/val_subset.jsonl")
sub_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in subset) + "\n")
print(f"eval subset: {len(subset)} samples "
      f"({', '.join(f'{k}={len(v)}' for k, v in cats.items())})", flush=True)

# ── models ──
tok = AutoTokenizer.from_pretrained(nanbeige_dir)
processor = NanbeigeVisionProcessor(qwen_dir, nanbeige_dir)
im_end_id = tok.convert_tokens_to_ids("<|im_end|>")

vision_tower, vinfo = load_qwen_vision_tower(qwen_dir, device="cuda:0", dtype=torch.float32)
llm_cfg = AutoConfig.from_pretrained(nanbeige_dir)
llm = LlamaForCausalLM.from_pretrained(
    nanbeige_dir, config=llm_cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    device_map="auto", max_memory={0: "20GiB", 1: "27GiB", 2: "25GiB"})
llm.eval()
for p in llm.parameters():
    p.requires_grad = False
remove_accelerate_hooks(llm)

projector = CustomProjector(
    vision_feature_dim=vinfo.out_hidden_size, llm_hidden_size=llm_cfg.hidden_size,
    use_input_norm=True).to(device="cuda:0", dtype=torch.float32)
ck = torch.load(cfg["save_dir"] + "/projector_final.pt", map_location="cuda:0",
                weights_only=True)
projector.load_state_dict(ck["projector_state_dict"])
projector.eval()
print(f"projector loaded from step {ck.get('global_step')}", flush=True)

ds = ImageTextDataset(str(sub_path), processor, image_dir=cfg.get("image_dir"),
                      max_text_length=cfg.get("max_text_length", 4096))
loader = DataLoader(ds, batch_size=1, shuffle=False,
                    collate_fn=MultimodalDataCollator(processor), num_workers=0)

embed_dev = llm.model.embed_tokens.weight.device
dev = torch.device("cuda:0")


@torch.no_grad()
def generate(text_embeds: torch.Tensor, max_new: int = 15) -> str:
    cur, ids = text_embeds, []
    for _ in range(max_new):
        logits = sharded_llm_forward(llm, cur)
        nxt = int(logits[0, -1].argmax().item())
        if nxt == im_end_id:
            break
        ids.append(nxt)
        e = llm.model.embed_tokens(torch.tensor([[nxt]], device=embed_dev))
        cur = torch.cat([cur, e.to(cur.dtype).to(dev)], dim=1)
    return tok.decode(ids, skip_special_tokens=True).strip()


correct, total = 0, 0
print(f"\n{'='*72}\nGENERATION EVAL (greedy, max {15} new tokens)\n{'='*72}")
for batch, rec in zip(loader, subset):
    batch = batch.to(dev)
    text_embeds = build_multimodal_embeds(
        llm, vision_tower, projector, batch, torch.bfloat16, dev)
    gen = generate(text_embeds)
    exp = rec["answer"]
    ok = gen == exp
    correct += ok
    total += 1
    print(f"[{'PASS' if ok else 'MISS'}] {rec['question']}\n"
          f"        gen={gen!r}  expect={exp!r}", flush=True)

print(f"\n{'='*72}\nEXACT MATCH: {correct}/{total} ({correct/total*100:.0f}%)")
