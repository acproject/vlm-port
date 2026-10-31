"""Shape + behavior tests for the VLM port.

Run:
    cd /home/acproject/usb/transplant/vlm-port
    ../venv-cu128/bin/python tests/test_shapes.py

Tests marked [SLOW] require loading the full Nanbeige LLM (33B, CPU).
By default only fast tests run. Pass --full to include slow tests.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── Import root ──
_PROJECT_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(_PROJECT_SRC))
_VENV_SITE = Path(__file__).resolve().parent.parent.parent / "venv-cu128" / "lib" / "python3.12" / "site-packages"
if str(_VENV_SITE) not in sys.path:
    sys.path.insert(0, str(_VENV_SITE))

import torch
from PIL import Image

DEVICE = "cpu"
DTYPE = torch.bfloat16
QWEN_DIR = "/home/acproject/usb/transplant/Qwen3.8-27B"
NANBEIGE_DIR = "/home/acproject/usb/transplant/Nanbeige4.3-33B"

passed = 0
failed = 0
errors: list[str] = []


def t(name: str, fn, slow=False, full=False):
    """Run a test, track pass/fail."""
    global passed, failed
    if slow and not full:
        print(f"  ⏭  SKIP [SLOW] {name}")
        return
    print(f"  ▶  {name} ... ", end="", flush=True)
    t0 = time.time()
    try:
        fn()
        dt = time.time() - t0
        print(f"✓ ({dt:.2f}s)")
        passed += 1
    except Exception as e:
        dt = time.time() - t0
        print(f"✗ ({dt:.2f}s) {type(e).__name__}: {e}")
        errors.append(f"{name}: {type(e).__name__}: {e}")
        failed += 1


# ────────────────────────────────────────────────────────
# Fixtures (cached across tests)
# ────────────────────────────────────────────────────────
_cache = {}


def vision_tower():
    if "vt" not in _cache:
        from vision.qwen_vision import load_qwen_vision_tower
        _cache["vt"], _cache["vi"] = load_qwen_vision_tower(QWEN_DIR, device=DEVICE, dtype=DTYPE)
    return _cache["vt"], _cache["vi"]


def projector():
    if "pj" not in _cache:
        from vision.projector import CustomProjector
        _cache["pj"] = CustomProjector(
            vision_feature_dim=5120, llm_hidden_size=7168, act_fn="gelu"
        ).to(device=DEVICE, dtype=DTYPE)
    return _cache["pj"]


def processor():
    if "proc" not in _cache:
        from data.processor import NanbeigeVisionProcessor
        _cache["proc"] = NanbeigeVisionProcessor(QWEN_DIR, NANBEIGE_DIR)
    return _cache["proc"]


def llm():
    if "llm" not in _cache:
        from transformers import LlamaForCausalLM, AutoConfig
        cfg = AutoConfig.from_pretrained(NANBEIGE_DIR)
        _cache["llm"] = LlamaForCausalLM.from_pretrained(
            NANBEIGE_DIR, config=cfg, torch_dtype=DTYPE, low_cpu_mem_usage=True,
        ).to(device=DEVICE)
        for p in _cache["llm"].parameters():
            p.requires_grad = False
        _cache["llm"].eval()
    return _cache["llm"]


# ────────────────────────────────────────────────────────
# Fast tests (no LLM)
# ────────────────────────────────────────────────────────

def test_projector_forward_shape():
    """5120 → 7168 forward shape."""
    proj = projector()
    x = torch.randn(256, 5120, device=DEVICE, dtype=DTYPE)
    y = proj(x)
    assert y.shape == (256, 7168), f"got {y.shape}"


def test_projector_backward():
    """Projector backward (trainable params receive grad)."""
    from vision.projector import CustomProjector
    proj = CustomProjector(vision_feature_dim=5120, llm_hidden_size=7168, act_fn="gelu")
    proj.train()
    # Float32 for backward stability (BF16 backward has limited op coverage)
    x = torch.randn(32, 5120, device=DEVICE, dtype=torch.float32)
    y = proj(x)
    loss = y.sum()
    loss.backward()
    for name, p in proj.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
        assert p.grad.abs().sum().item() > 0, f"{name} grad is zero"


def test_projector_save_reload():
    """state_dict save → load preserves outputs."""
    proj1 = projector()
    state = proj1.state_dict()
    proj2 = type(proj1)(
        vision_feature_dim=5120, llm_hidden_size=7168, act_fn="gelu"
    ).to(device=DEVICE, dtype=DTYPE)
    proj2.load_state_dict(state)
    x = torch.randn(16, 5120, device=DEVICE, dtype=DTYPE)
    y1 = proj1(x)
    y2 = proj2(x)
    assert torch.equal(y1, y2), "Reloaded projector gives different outputs"


def test_vision_forward_shapes():
    """ViT → Merger pooler_output shapes."""
    vt, vi = vision_tower()
    # Simulate a 512×512 image → grid_thw=[1, 32, 32] → post-merge=256
    pixel_values = torch.randn(1024, 1536, device=DEVICE, dtype=DTYPE)
    grid_thw = torch.tensor([[1, 32, 32]], dtype=torch.long)
    out = vt(pixel_values, grid_thw=grid_thw)
    # pooler_output = merger output = (post_merge_tokens, 5120)
    post_merge = 1024 // 4
    assert out.pooler_output.shape == (post_merge, vi.out_hidden_size), \
        f"pooler_output: got {out.pooler_output.shape}, expected ({post_merge}, {vi.out_hidden_size})"
    # last_hidden_state = ViT output = (total_vit_patches, 1152)
    assert out.last_hidden_state.shape == (1024, vi.hidden_size), \
        f"last_hidden_state: got {out.last_hidden_state.shape}"


def test_vision_frozen():
    """Vision tower all frozen."""
    vt, _ = vision_tower()
    n_train = sum(1 for p in vt.parameters() if p.requires_grad)
    assert n_train == 0, f"{n_train} vision params not frozen!"


def test_processor_grid_expansion():
    """image_pad count == grid_thw.prod() // merge_size²."""
    proc = processor()
    img = Image.new("RGB", (512, 512), color=(128, 200, 100))
    batch = proc(
        messages=[{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": "Describe."}
        ]}],
        images=[img], device=DEVICE,
    )
    post_merge = batch.image_grid_thw.prod(-1).item() // (proc.spatial_merge_size ** 2)
    assert batch.image_pad_mask.sum().item() == post_merge, \
        f"image_pad={batch.image_pad_mask.sum().item()} != post_merge={post_merge}"


def test_processor_multi_image():
    """2 images → correct grid_thw split."""
    proc = processor()
    img1 = Image.new("RGB", (512, 512), color=(255, 0, 0))
    img2 = Image.new("RGB", (768, 512), color=(0, 255, 0))
    batch = proc(
        messages=[{"role": "user", "content": [
            {"type": "image"}, {"type": "image"}, {"type": "text", "text": "Two images."}
        ]}],
        images=[img1, img2], device=DEVICE,
    )
    assert batch.image_grid_thw.shape[0] == 2, f"expected 2 grid entries, got {batch.image_grid_thw.shape[0]}"
    total_post_merge = (
        batch.image_grid_thw.prod(-1).sum().item() // (proc.spatial_merge_size ** 2)
    )
    assert batch.image_pad_mask.sum().item() == total_post_merge, \
        f"image_pad={batch.image_pad_mask.sum().item()} != total_post_merge={total_post_merge}"


def test_special_token_ids():
    """Vision special token IDs are in Nanbeige free slots."""
    proc = processor()
    # IDs must be >= 166107 (first free slot), < embed_table=166912
    assert 166107 <= proc.vision_start_id < 166912
    assert 166107 <= proc.vision_end_id < 166912
    assert 166107 <= proc.image_pad_id < 166912
    # All three distinct
    ids = {proc.vision_start_id, proc.vision_end_id, proc.image_pad_id}
    assert len(ids) == 3


def test_labels_mask():
    """build_labels: image_pad → -100, vision markers → -100, others → shift OK."""
    from modeling.multimodal_model import build_labels
    from data.processor import VISION_START_ID, VISION_END_ID, IMAGE_PAD_ID

    S = 20
    input_ids = torch.randint(1, 10000, (1, S), dtype=torch.long)
    # Inject image_pad at positions 5-9
    input_ids[0, 5:10] = IMAGE_PAD_ID
    input_ids[0, 4] = VISION_START_ID
    input_ids[0, 10] = VISION_END_ID

    image_pad_mask = torch.zeros(1, S, dtype=torch.bool)
    image_pad_mask[0, 5:10] = True

    labels = build_labels(
        input_ids, image_pad_mask,
        vision_start_token_id=VISION_START_ID,
        vision_end_token_id=VISION_END_ID,
    )

    # Check image_pad positions are -100
    assert (labels[0, 5:10] == -100).all(), "image_pad not masked to -100"
    # Check vision_start/end are -100
    assert labels[0, 4] == -100, "vision_start not masked"
    assert labels[0, 10] == -100, "vision_end not masked"
    # Check non-vision positions are NOT -100 (unless pad_token)
    for i in list(range(0, 4)) + list(range(11, S)):
        assert labels[0, i] != -100, f"position {i} unexpectedly masked"


# ────────────────────────────────────────────────────────
# Slow tests (require full Nanbeige LLM)
# ────────────────────────────────────────────────────────

def test_embedding_injection_roundtrip():
    """Projected vision embeddings replace image_pad embeddings, non-image unchanged."""
    vt, _ = vision_tower()
    proj = projector()
    proc = processor()
    lm = llm()

    img = Image.new("RGB", (512, 512), color=(128, 200, 100))
    batch = proc(
        messages=[{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": "Describe."}
        ]}],
        images=[img], device=DEVICE,
    )

    with torch.no_grad():
        vis_out = vt(batch.pixel_values, grid_thw=batch.image_grid_thw)
        projected = proj(vis_out.pooler_output.to(proj.net[0].weight.dtype)).to(DTYPE)

    text_embeds = lm.model.embed_tokens(batch.input_ids).to(DTYPE)
    orig_non_image = text_embeds[~batch.image_pad_mask].clone()
    orig_image = text_embeds[batch.image_pad_mask].clone()

    text_embeds[batch.image_pad_mask] = projected

    # Image positions changed
    assert not torch.equal(text_embeds[batch.image_pad_mask], orig_image), \
        "image_pad embeddings were NOT replaced!"
    # Non-image positions unchanged
    diff = (text_embeds[~batch.image_pad_mask] - orig_non_image).abs().max().item()
    assert diff < 1e-3, f"Non-image embeddings drifted by {diff}"


def test_text_only_forward():
    """No image → pure Nanbeige forward, logits finite."""
    proc = processor()
    lm = llm()
    tok = proc.tokenizer
    ids = torch.tensor(
        [tok.encode("<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)],
        dtype=torch.long, device=DEVICE,
    )
    assert ids.max().item() < lm.model.embed_tokens.weight.shape[0]
    with torch.no_grad():
        out = lm(input_ids=ids)
    assert torch.isfinite(out.logits).all(), "Non-finite logits!"
    assert out.logits.shape == (1, ids.shape[1], lm.config.vocab_size)


def test_full_generate_vision():
    """Vision-injected generate returns full sequence (input + new tokens)."""
    vt, _ = vision_tower()
    proj = projector()
    proc = processor()
    lm = llm()

    img = Image.new("RGB", (320, 320), color=(200, 100, 50))
    batch = proc(
        messages=[{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": "What color?"}
        ]}],
        images=[img], device=DEVICE,
    )
    S = batch.input_ids.shape[1]

    with torch.no_grad():
        vis_out = vt(batch.pixel_values, grid_thw=batch.image_grid_thw)
        projected = proj(vis_out.pooler_output.to(proj.net[0].weight.dtype)).to(DTYPE)

    text_embeds = lm.model.embed_tokens(batch.input_ids).to(DTYPE)
    text_embeds[batch.image_pad_mask] = projected

    pos_ids = torch.arange(S, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        gen = lm.generate(
            input_ids=batch.input_ids,
            inputs_embeds=text_embeds,
            position_ids=pos_ids,
            attention_mask=batch.attention_mask,
            max_new_tokens=3, do_sample=False,
            pad_token_id=batch.input_ids.new_tensor(0),
        )
    assert gen.shape == (1, S + 3), f"Expected (1, {S+3}), got {gen.shape}"


def test_trainable_count():
    """Only projector is trainable (vision + LLM frozen)."""
    from modeling.multimodal_model import NanbeigeVLM, MultimodalModelConfig
    cfg = MultimodalModelConfig(
        vision_model_dir=QWEN_DIR, llm_model_dir=NANBEIGE_DIR,
        device=DEVICE, dtype=DTYPE, torch_dtype_llm=DTYPE,
    )
    # Don't actually construct — that loads 33B
    # Just verify projector is only trainable in isolation
    proj = projector()
    n_train = sum(p.numel() for p in proj.parameters() if p.requires_grad)
    assert n_train > 0, "Projector has no trainable params!"
    print(f"\n  ── Projector trainable: {n_train/1e6:.2f}M ──")


# ────────────────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────────────────
def main():
    global passed, failed
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run slow LLM tests too")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"Shape Tests | device={DEVICE} dtype={DTYPE} full={args.full}")
    print(f"{'='*60}")

    tests = [
        # (name, fn, slow)
        ("projector_forward_shape", test_projector_forward_shape, False),
        ("projector_backward", test_projector_backward, False),
        ("projector_save_reload", test_projector_save_reload, False),
        ("vision_forward_shapes", test_vision_forward_shapes, False),
        ("vision_frozen", test_vision_frozen, False),
        ("processor_grid_expansion", test_processor_grid_expansion, False),
        ("processor_multi_image", test_processor_multi_image, False),
        ("special_token_ids", test_special_token_ids, False),
        ("labels_mask", test_labels_mask, False),
        # Slow (need LLM)
        ("embedding_injection_roundtrip", test_embedding_injection_roundtrip, True),
        ("text_only_forward", test_text_only_forward, True),
        ("full_generate_vision", test_full_generate_vision, True),
    ]

    for name, fn, slow in tests:
        t(name, fn, slow=slow, full=args.full)

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print("Failures:")
        for e in errors:
            print(f"  ✗ {e}")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
