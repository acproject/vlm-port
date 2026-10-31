#!/usr/bin/env python
"""Trace the exact spatial merge flow: ViT output → reorder → merger input.

Use Qwen2VL (non 3_5) vision tower if available, or manually construct
the reorder from grid_thw to verify the math.

This is PURELY about understanding the data flow — no GPU needed.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import torch

def get_vision_position_ids_block_major(grid_thw, spatial_merge_size):
    """vision_utils.py get_vision_position_ids block-major reorder."""
    t, h, w = grid_thw
    merge_size = spatial_merge_size
    hpos_ids, wpos_ids = torch.meshgrid(
        torch.arange(h), torch.arange(w), indexing="ij"
    )
    block_shape = (h // merge_size, merge_size, w // merge_size, merge_size)
    hpos_ids = hpos_ids.reshape(block_shape).transpose(1, 2).flatten()
    wpos_ids = wpos_ids.reshape(block_shape).transpose(1, 2).flatten()
    return torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1)

def get_reorder_index(grid_thw, spatial_merge_size):
    """Get the reorder permutation that maps ViT row-major → block-major post-merge.
    
    ViT tokens are ordered: t=0, h=0, w=0; t=0, h=0, w=1; ... (row-major)
    Post-merge tokens should be:
      block(0,0) h=[0,1] w=[0,1]; block(0,1) h=[0,1] w=[2,3]; ...
    """
    t, h, w = grid_thw
    m = spatial_merge_size
    
    # row-major ViT index for each (t, h, w)
    # idx = t * h * w + h * w + w
    # But actually ViT tokens for single image: (h*w,) flat
    # Multi-image with temporal: t * h * w
    
    # Method: build full grid indices, then apply block-major reorder
    h_idx = torch.arange(h).view(h // m, m)
    w_idx = torch.arange(w).view(w // m, m)
    # block-major reorder (same as vision_utils.py line 225)
    reorder = (h_idx[:, :, None, None] * w + w_idx[None, None, :, :]).transpose(1, 2).flatten()
    
    # For single image (t=1), this is the reorder from row-major → block-major
    # For video with t>1, repeat
    reorder = reorder.repeat(t)
    
    return reorder

def trace_spatial_merge(grid_thw, spatial_merge_size=2):
    """Trace through the exact reorder math."""
    t, h, w = grid_thw
    m = spatial_merge_size
    
    print(f"\ngrid_thw = ({t}, {h}, {w}), spatial_merge_size = {m}")
    print(f"  ViT patches: T×H×W = {t}×{h}×{w} = {t*h*w} tokens")
    print(f"  Post-merger: {t}×{h//m}×{w//m} = {t*h//m*w//m} tokens × ({m*m}) concat")
    print()
    
    # Build row-major indices (what ViT outputs)
    vit_indices = torch.arange(t * h * w).reshape(t, h, w)
    print(f"  ViT row-major token indices (t=0):")
    print(f"    h\\w  " + "  ".join(f"{w:5d}" for w in range(w)))
    for hi in range(h):
        row = "    " + f"{hi:3d}  " + "  ".join(f"{vit_indices[0, hi, wi]:5d}" for wi in range(w))
        print(row)
    
    # Block-major reorder
    reorder = get_reorder_index(grid_thw, m)
    print(f"\n  Block-major reorder index (maps row-major → block-major):")
    print(f"    {reorder.tolist()}")
    
    # Verify: after reorder, tokens should be grouped into m×m spatial blocks
    vit_flat = torch.arange(t * h * w)
    reordered = vit_flat[reorder]
    print(f"\n  After reorder, ViT tokens in memory: {reordered.tolist()}")
    
    # Show first few groups of m×m
    print(f"\n  Each group of {m*m} consecutive tokens = one spatial merge block:")
    n_blocks = (h // m) * (w // m)
    for bi in range(min(n_blocks, 8)):
        start = bi * m * m
        block_tokens = reordered[start:start + m * m]
        # Which (h,w) positions do these correspond to?
        h_start = (bi // (w // m)) * m
        w_start = (bi % (w // m)) * m
        positions = [(h_start + dh, w_start + dw) for dh in range(m) for dw in range(m)]
        print(f"    Block {bi} (h=[{h_start},{h_start+m-1}], w=[{w_start},{w_start+m-1}]):")
        print(f"      row-major indices: {block_tokens.tolist()}")
        print(f"      (h,w) positions: {positions}")
    
    return reorder

# Test with small grid first
print("=" * 70)
print("SMALL GRID (8×8)")
print("=" * 70)
trace_spatial_merge([1, 8, 8], 2)

print()
print("=" * 70)
print("REAL GRID (48×48 = 768×768 image, patch=16)")
print("=" * 70)
reorder = trace_spatial_merge([1, 48, 48], 2)
print(f"\n  Reorder permutation length: {len(reorder)}")
print(f"  Unique values: {len(set(reorder.tolist()))} (should equal ViT patches = 2304)")
print(f"  First 16 values: {reorder[:16].tolist()}")
print(f"  Last 16 values: {reorder[-16:].tolist()}")

# Key verification: does ViT → [reorder] → view(-1, 4608) give correct post-merge?
print(f"\n{'='*70}")
print("VERIFY: ViT(2304, 1152) → reorder → view(-1, 4608) = post-merge(576, 4608)?")
print(f"{'='*70}")
vit_out = torch.randn(2304, 1152)  # dummy ViT output
reordered = vit_out[reorder]       # apply block-major reorder
merged = reordered.view(-1, 4608)  # merger's reshape
print(f"  ViT output: {vit_out.shape}")
print(f"  After reorder: {reordered.shape}")
print(f"  After merger view(-1, 4608): {merged.shape}")
print(f"  Expected post-merge tokens: 48//2 * 48//2 = 24 * 24 = 576")
print(f"  ✓ Shape matches! {merged.shape[0]} == 576")

# Now: is this reorder actually applied somewhere in VisionModel.forward?
print(f"\n{'='*70}")
print("KEY QUESTION: Who applies this reorder to ViT hidden_states?")
print(f"{'='*70}")
print("  VisionModel.forward code (line 1083-1127):")
print("    hidden_states = self.patch_embed(hidden_states)   # patch embed → (N, 1152)")
print("    hidden_states = hidden_states + pos_embeds          # pos_embed ADDITION (no reorder)")
print("    for blk in self.blocks:                             # ViT blocks (cu_seqlens attention)")
print("        hidden_states = blk(hidden_states, ...)         # no reorder inside block")
print("    merged_hidden_states = self.merger(hidden_states)   # merger has x = self.norm(x).view(-1, 4608)")
print()
print("  So either:")
print("    (a) reorder is implicit in how hidden_states is laid out by patch_embed + pos_embed")
print("    (b) merger does the reorder internally (but source code shows it doesn't)")
print("    (c) patch_embed + ViT + merger ASSUME the input is ALREADY block-major")
print()
print("  Let's check Qwen2VL (non-3_5) merger source for comparison...")

# Also check: does get_vision_bilinear_indices_and_weights use reorder on pos_embed?
# If pos_embed is already block-major, adding it to row-major ViT hidden_states would be wrong...
# unless ViT hidden_states is also block-major!
print(f"\n{'='*70}")
print("HYPOTHESIS: patch_embed output is block-major, not row-major!")
print(f"{'='*70}")
print("  Conv3d(kernel=[2,16,16], stride=[2,16,16]) on (B, 3, 1, H, W)")
print("  → output shape: (B, 1152, 1, H//16, W//16)")
print("  → .view(-1, 1152) → row-major: t, h, w")
print()
print("  Unless... the input to patch_embed is pre-reordered!")
print("  Let's check what pixel_values looks like when fed into patch_embed...")
print()
print("  Actually, Qwen3_VL's patch_embed is called by VisionModel.forward")
print("  where hidden_states is pixel_values (NOT patchified yet).")
print("  So the input to VisionModel.forward is pixel_values, not ViT patches.")
print("  And patch_embed does the Conv3d patchification.")
print()
print("  Let's trace VisionModel.forward input shape...")

# Let me actually check what pixel_values looks like
print(f"\n  Checking pixel_values shape from processor...")
print(f"  Qwen3VL processor output (768×768 image):")
print(f"    pixel_values shape: (2304, 1536)  ← (T*H_patches*W_patches, C*T_patch*H_patch*W_patch)?")
print()
print("  Actually, let me look at Qwen3VLForConditionalGeneration.get_image_features:")
print("    vision_output = self.visual(pixel_values, grid_thw=...)")
print("  So pixel_values is fed directly into self.visual (VisionModel).")
print("  VisionModel.forward(pixel_values, grid_thw) → patch_embed(pixel_values)")
print()
print("  VisionModel.forward input 'hidden_states' parameter is actually pixel_values!")
print("  And patch_embed expects it as (-1, 3, 2, 16, 16) — so pixel_values must be")
print("  pre-split into patches by processor!")
print()
print("  This is the missing link! The processor (Qwen2VLImageProcessor) must do")
print("  the spatial reorder when it patches the image!")

print("\nDONE.")
