# Vision Merger 深度分析 — Spatial Reshape 完整数据流

> 源码路径：`transformers/models/qwen2_vl/image_processing_qwen2_vl.py` +
> `transformers/models/qwen3_5/modeling_qwen3_5.py` + `transformers/vision_utils.py`

---

## 1. 问题定义

Qwen3-VL 的 Vision Merger 接受 ViT 输出 (N_vis_patches, 1152)，输出 post-merge tokens (N_post_merge, 5120)。
N_vis_patches / N_post_merge = **4**，正好是 `spatial_merge_size² = 2² = 4`。

merger.forward 只有一行关键 reshape：

```python
x = self.norm(x).view(-1, self.hidden_size)  # hidden_size = 1152 × 4 = 4608
```

问题：**ViT 输出是 row-major 排列的 patch tokens，`.view(-1, 4608)` 怎么正确完成 spatial 2×2 concat？**

答案：**不是 merger 做的 reorder，是 processor 预处理阶段就做好了！**

---

## 2. 完整数据流（768×768 图像示例）

### 2.1 Processor 预处理（Qwen2VLImageProcessor._preprocess）

输入：PIL Image (768, 768, 3)

```python
# 1. Resize 到 patch_size × merge_size 的倍数
#    factor = 16 × 2 = 32
#    768 / 32 = 24 → 整除 ✓

# 2. Reshape 成 block-major patch grid
grid_h = 768 // patch_size = 48      # ViT patch grid H
grid_w = 768 // patch_size = 48      # ViT patch grid W

patches = image.reshape(
    1,                # batch
    3,                # channel
    48 // 2,          # H / merge_size = 24
    2,                # merge_size
    16,               # patch_size
    48 // 2,          # W / merge_size = 24
    2,                # merge_size
    16,               # patch_size
)
# shape: (1, 3, 24, 2, 16, 24, 2, 16)

# 3. Permute → block-major: [B, H/m, W/m, m, m, C, P, P]
patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
# shape: (1, 24, 24, 2, 2, 3, 16, 16)
# 每个 "merge block" 的 2×2 个 patch 在内存里连续！

# 4. Flatten
flatten_patches = patches.reshape(
    1,                          # batch
    48 * 48,                    # grid_h * grid_w = 2304 ViT patches
    3 * 2 * 16 * 16,            # C × temporal_patch_size × P × P = 1536
)
# shape: (1, 2304, 1536) → 去掉 batch dim → (2304, 1536) ✅ 实测值！
```

**关键**：这 2304 个 ViT patch 是按 **(H/m, W/m, m, m)** 排列的，不是 row-major 的 **(H, W)**！

```text
Memory layout (2304 ViT patches):
  [block(0,0) patch(h=0,w=0), block(0,0) patch(h=0,w=1),
   block(0,0) patch(h=1,w=0), block(0,0) patch(h=1,w=1),
   block(0,1) patch(h=0,w=2), block(0,1) patch(h=0,w=3),
   ...]
```

### 2.2 VisionModel.forward

```python
# Input: pixel_values = (2304, 1536)  ← 已经 block-major！
#        grid_thw = [[1, 48, 48]]

# Step 1: patch_embed
hidden_states = self.patch_embed(pixel_values)
# patch_embed.proj = Conv3d(3, 1152, [2, 16, 16], stride=[2, 16, 16])
# pixel_values view → (2304, 3, 2, 16, 16) → Conv3d → (2304, 1152)
# 2304 个 token 保持 block-major 排列

# Step 2: pos_embed（从 ViT 原生 48×48 → post-merge 24×24 做 bilinear 插值）
bilinear_indices, bilinear_weights = get_vision_bilinear_indices_and_weights(
    grid_thw, num_grid_per_side=48, spatial_merge_size=2  # grid_thw 是 ViT grid (1,48,48)
)
# bilinear_indices shape: (4, 2304) — 4 个 bilinear 角的 index
# bilinear_weights shape: (4, 2304)
# 重要：vision_utils.py 行 225 里 bilinear_indices 也用了 block-major reorder！
pos_embeds = (self.pos_embed(bilinear_indices) * bilinear_weights[:, :, None]).sum(0)
# pos_embeds shape: (2304, 1152) — 也是 block-major 排列

hidden_states = hidden_states + pos_embeds  # 同排列，直接加

# Step 3: 27 层 ViT blocks（Flash Attention，cu_seqlens 保持顺序）
cu_seqlens = get_vision_cu_seqlens(grid_thw)  # [0, 2304] for single image
for blk in self.blocks:
    hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, ...)
# hidden_states: (2304, 1152) — block-major 保持

# Step 4: Merger（真正的 spatial merge）
merged_hidden_states = self.merger(hidden_states)
# merger.norm: LayerNorm(1152) → 对每个 ViT token 做 LN
# .view(-1, 4608): 2304 个 token → 576 组 × 每组 4 个连续 token
#                  每组 4 × 1152 = 4608 = hidden_size
# linear_fc1: Linear(4608 → 4608)
# GELU
# linear_fc2: Linear(4608 → 5120)
# 输出: (576, 5120) ✅ post-merge tokens × out_hidden_size！
```

### 2.3 为什么 `.view(-1, 4608)` 是正确的 spatial merge？

因为 processor 已经把 pixel_values 排列成了：

```text
Memory: [block0_patch0, block0_patch1, block0_patch2, block0_patch3,
         block1_patch0, block1_patch1, block1_patch2, block1_patch3, ...]
```

`.view(-1, 4608)` 每 4 个连续 token 拼成一个 4608 维向量，正好是一个 2×2 spatial block 的 4 个 patch 特征 concat！

**merger 的 Linear(4608→4608→5120) 就是对每个 spatial merge block 做 MLP**。

### 2.4 关键 shape 汇总

| 阶段 | Shape | 含义 |
|---|---|---|
| 输入图像 | (768, 768, 3) | PIL Image |
| processor resize 后 | (768, 768, 3) | 768 % 32 = 0，无需 pad |
| pixel_values | **(2304, 1536)** | 2304 ViT patches (block-major)，每个 patch = 3×2×16×16 |
| grid_thw | **[1, 48, 48]** | T=1, H=48, W=48（ViT patch grid） |
| patch_embed 输出 | (2304, 1152) | ViT hidden，block-major |
| ViT blocks 输出 | (2304, 1152) | ViT hidden，block-major |
| merger 中间 | (576, 4608) | 2×2 concat 后的 merge block 特征 |
| merger 输出 | **(576, 5120)** | post-merge visual embeddings |
| split_sizes | **576** | `grid_thw.prod() // merge_size² = 1×48×48 // 4 = 576` |
| image_pad 展开 | **576** 个 | processor 把一个 `<|image_pad|>` 展开成 N_post_merge 个 |

---

## 3. image_grid_thw 的双重语义

**image_grid_thw 有两个阶段**：

### 3.1 阶段一：Processor 输出的 grid_thw

```python
grid_thw = [[1, 48, 48]]
# T=1, H=48, W=48  ← ViT 原始 patch 网格！
```

用途：
- `get_vision_bilinear_indices_and_weights(grid_thw, ...)` —— pos_embed 插值用 ViT grid
- `get_vision_cu_seqlens(grid_thw)` —— Flash Attention 用 ViT grid 计算 cu_seqlens
- `patch_embed` 内部 Conv3d stride 天然产生 grid_h×grid_w 个 patch

### 3.2 阶段二：Merger 后 split_sizes（post-merge grid）

```python
split_sizes = grid_thw.prod(-1) // spatial_merge_size²
# = 1×48×48 // 4 = 576 = 24×24
```

这是 **post-merge token 数**，用于：
- `torch.split(image_embeds, split_sizes)` —— 多图场景下分割每个图的 embedding
- Qwen LLM MRoPE position_ids 的计算（post-merge H/W）
- image_pad token 展开数

### 3.3 两个 grid 的转换公式

```text
ViT grid:          (T, H, W)
Post-merge grid:   (T, H//m, W//m)   其中 m = spatial_merge_size = 2

processor.image_grid_thw = ViT grid  ✅ 已验证
split_sizes = T × H × W // m² = post-merge tokens
```

---

## 4. ViT position_ids 的 block-major reorder

vision_utils.py 行 83-97（`get_vision_position_ids`）也用了 block-major：

```python
block_shape = (h // merge_size, merge_size, w // merge_size, merge_size)
hpos_ids = hpos_ids.reshape(block_shape).transpose(1, 2).flatten()
wpos_ids = wpos_ids.reshape(block_shape).transpose(1, 2).flatten()
```

返回的 (h, w) 坐标顺序和 ViT hidden_states 完全一致（block-major）。

同理，行 223-225（`get_vision_bilinear_indices_and_weights`）：

```python
h_idx = torch.arange(h).view(h // m, m)
w_idx = torch.arange(w).view(w // m, m)
reorder = (h_idx[:, :, None, None] * w + w_idx[None, None, :, :]).transpose(1, 2).flatten()
```

这个 `reorder` 就是 ViT patch index 的 block-major permutation！pos_embed bilinear 插值结果按这个 reorder 返回，和 ViT hidden_states 顺序对齐。

---

## 5. patch_embed Conv3d 的正确理解

```python
# patch_embed.proj.weight: (1152, 3, 2, 16, 16)
# Conv3d(in_channels=3, out_channels=1152, kernel_size=[2,16,16], stride=[2,16,16])
```

patch_embed 不做 spatial reorder！它只做 patchification：
- 输入 `pixel_values.view(-1, 3, 2, 16, 16)` —— processor 已经按 block-major 分好了 patch
- Conv3d 对每个 patch 内部做线性投影
- 输出保持 block-major 排列

**如果我们跳过 processor，直接把 raw image tensor 喂给 VisionModel，就会错！** 因为 raw image 是 row-major (B, 3, H, W)，但 VisionModel 期望的 pixel_values 是 block-major patchified (N, 1536)。

---

## 6. 对我们项目的影响

### 6.1 必须复用 Qwen2VLImageProcessor

我们的 NanbeigeVisionProcessor **必须复用 Qwen2VLImageProcessor** 来：
1. Resize 图像到 `patch_size × merge_size = 32` 的倍数
2. 做 block-major reshape + permute（行 196-208 的逻辑）
3. 生成正确的 `pixel_values` (N, 1536)
4. 生成正确的 `image_grid_thw` (T, H, W) —— **这是 ViT grid，不是 post-merge grid！**

### 6.2 直接用 Vision Tower 输出（pooler_output）

VisionModel 返回：
- `last_hidden_state` —— ViT 输出 (N_vis_patches, 1152)
- `pooler_output` —— **merger 输出 (N_post_merge, 5120)** ✅ 我们要的就是这个！

`get_image_features` 行 1428 直接取 `vision_output.pooler_output`，然后 `split_sizes = grid_thw.prod() // 4` 分割多图。

**我们的 Custom Projector 输入就是这个 pooler_output！** 它已经是正确的 post-merge 5120 维。

### 6.3 我们不需要自己做 spatial merge

不要在我们的代码里重新实现 merger 的 spatial merge 逻辑——vision tower 已经帮我们做好了。直接：

```python
vis_out = vision_tower(pixel_values=pixel_values, grid_thw=image_grid_thw)
vis_features = vis_out.pooler_output  # (sum post-merge tokens, 5120) ← 这就是我们要的！
projected = projector(vis_features)    # (sum, 7168)
```

### 6.4 image_pad token 展开数

```python
# 对每张图像：
n_post_merge = image_grid_thw[i].prod().item() // (spatial_merge_size ** 2)
# = 1×48×48 // 4 = 576 for 768×768 image
```

processor 把 chat template 里的一个 `<|image_pad|>` 展开成 N_post_merge 个 `<|image_pad|>` token。我们需要：
1. 在 NanbeigeVisionProcessor 里复用同样的展开逻辑
2. 确保 image_grid_thw 和展开后的 image_pad token 数对应

### 6.5 grid_thw 的"类型安全"

**image_grid_thw 永远是 ViT patch grid（T, H_patches, W_patches）**，不是 post-merge grid。所有用到它的函数：

| 函数 | 用 grid_thw 做什么 | 期望 ViT grid 还是 post-merge grid |
|---|---|---|
| `get_vision_bilinear_indices_and_weights` | pos_embed 插值 | ViT grid ✅ |
| `get_vision_position_ids` | MRoPE 计算 | ViT grid，内部除以 merge_size ✅ |
| `get_vision_cu_seqlens` | Flash Attention cu_seqlens | ViT grid ✅ |
| `split_sizes = grid_thw.prod() // m²` | 多图分割 | ViT grid，内部除以 m² ✅ |

**post-merge grid 永远是 grid_thw // merge_size**，不在任何 config 或 tensor 里直接出现，都是运行时计算。

---

## 7. 风险与注意事项

### 7.1 不要把 raw image 直接喂给 VisionModel

必须先用 Qwen2VLImageProcessor 处理，否则 block-major 排列错误，vision tower 输出的 token 顺序乱了，merger 的 view(-1, 4608) 拼出来的是错误的 4 个 patch 组合。

### 7.2 grid_thw 不要手动算

不要自己 `grid_h = height // patch_size` 然后传给 vision tower。Qwen2VLImageProcessor 会根据 `min_pixels/max_pixels` 动态 resize，grid_thw 必须从 processor 的输出取。

### 7.3 多图 grid_thw 格式

```python
image_grid_thw shape: (num_images, 3)  # T, H_viT_patches, W_viT_patches
# 例：两张图 768×768 + 1024×1024
# [[1, 48, 48], [1, 64, 64]]
```

### 7.4 Temporal 维度

单图 temporal_patch_size=2，但 Conv3d kernel=[2,16,16]，单图 T=1 时 processor 会在 temporal 维 expand 成 2（行 212-218 的 `.unsqueeze(6).expand(-1,-1,-1,-1,-1,-1, 2, -1, -1)`），所以 ViT grid 的 T=1 实际对应 patch_embed 处理时的 2 帧重复——对单图没影响，最终 ViT 输出还是 H×W 个 tokens（因为 Conv3d stride=[2,16,16] 在 T 方向只产生 1 个 output）。

### 7.5 我们跳过 merger 直接用 ViT 输出？不行

ViT 输出 (N_vis_patches, 1152)，576→4096 tokens 范围，比 merger 后多 4× 序列长度。虽然 merger 可以被冻结，但**merger 做的 spatial merge 是 token 压缩（4× 降）+ 特征重组（MLP）**，跳过 merger 会导致：
1. 序列长度膨胀 4×（影响 KV cache 和速度）
2. 丢失 merger 的 spatial feature mixing（2×2 patch concat + MLP）
3. 我们的 projector 要处理 1152 维输入而非 5120 维

**结论：完整保留 ViT + Merger，只在 merger 输出之后接 Custom Projector。**

---

## 附录：完整数据流一张图

```text
PIL Image (768, 768, 3)
  │
  ▼ Qwen2VLImageProcessor._preprocess
  ├─ smart_resize → 768 % 32 = 0，无需改
  ├─ reshape → (B, 3, 24, 2, 16, 24, 2, 16)     ← spatial_merge_size=2 拆分
  ├─ permute → (B, 24, 24, 2, 2, 3, 16, 16)     ← block-major 排列！
  ├─ expand temporal → (B, 24, 24, 2, 2, 3, 2, 16, 16)
  └─ flatten → pixel_values = (2304, 1536)      ← ViT patch 数 × (3×2×16×16)
                image_grid_thw = [[1, 48, 48]]  ← ViT patch grid
  │
  ▼ VisionModel.forward(pixel_values, grid_thw)
  ├─ patch_embed (Conv3d: 3→1152, kernel=[2,16,16], stride=[2,16,16])
  │   pixel_values.view(-1, 3, 2, 16, 16) → Conv3d → (2304, 1152)  block-major 保持
  │
  ├─ pos_embed (learnable, 48×48 grid, bilinear 插值到 ViT 实际 grid)
  │   bilinear_indices: block-major reorder (vision_utils.py line 225)
  │   → (2304, 1152)  block-major 保持
  │
  ├─ 27 layers ViT blocks (Flash Attention, cu_seqlens=[0, 2304])
  │   → (2304, 1152)  block-major 保持
  │
  └─ Qwen3_5VisionPatchMerger
      ├─ LN(1152)
      ├─ .view(-1, 4608)                         ← 每 4 个连续 token = 1 个 2×2 spatial block
      │   2304 / 4 = 576 blocks, 4×1152=4608 dim
      ├─ Linear(4608→4608) + GELU               ← spatial feature mixing
      └─ Linear(4608→5120)                        ← 投影到 LLM embedding dim
      → pooler_output = (576, 5120)              ← post-merge visual embeddings ✅
  │
  ▼ 我们的 Custom Projector (Linear(5120→7168) + GELU + Linear(7168→7168))
  → (576, 7168)                                   ← Nanbeige LLM 可接受的 embedding dim ✅
  │
  ▼ 替换 input_ids 中 image_pad 位置的 embedding
  → Nanbeige LLM forward ✅
```
