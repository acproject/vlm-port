# Deep Compatibility Analysis — Qwen3.8-27B Vision × Nanbeige4.3-33B LLM

> 基于源码（transformers 5.14.1）+ safetensors header（无权重数据）+ processor 实测。

---

## 1. MRoPE vs Standard RoPE：差异与我们的策略

### 1.1 Qwen 的 MRoPE 机制（源码）

Qwen3_5 的 position_ids 是 **3D**，shape = `(3, B, S)`：

```python
# modeling_qwen3_5.py line 1246-1296
def get_vision_position_ids(self, start_position, grid_thw, ..., spatial_merge_size):
    """grid_thw = (T, H, W) 是 ViT 原始 patch 网格"""
    llm_grid_t = grid_thw[0] // temp_merge_size        # 图像 → 1
    llm_grid_h = grid_thw[1] // spatial_merge_size     # 48/2 = 24
    llm_grid_w = grid_thw[2] // spatial_merge_size      # 48/2 = 24

    position_t = torch.arange(llm_grid_t) * time_interval
    position_h = torch.arange(llm_grid_h) + start_position
    position_w = torch.arange(llm_grid_w) + start_position

    T_grid, H_grid, W_grid = torch.meshgrid(position_t, position_h, position_w, indexing="ij")
    vision_position_ids = torch.stack([T_grid, H_grid, W_grid], dim=0).reshape(3, -1)
    return vision_position_ids   # shape (3, N_vis_post_merge)
```

**关键数学**：

```text
head_dim = 256
partial_rotary_factor = 0.25 → rotary_dim = 256 × 0.25 = 64
mrope_section = [11, 11, 10] → sum = 32 = rotary_dim / 2
```

RoPE 作用在 64 维上，分为 3 段分别编码 [T, H, W] 三个轴，interleaved 方式，剩余 192 维不做旋转。3D position_ids 的 **H 和 W 轴都加了 `start_position`**（文本长度的累积偏移），所以 vision token 的旋转角度是 `(seq_offset + h, seq_offset + w)` 形式。

文本 token 也被强制扩展为 3D：`position_ids[None, ...].expand(3, ...)`，值相同（T=0, H=text_pos, W=text_pos）。

### 1.2 Nanbeige 的 RoPE（标准 Llama）

```text
head_dim = 128，全旋转（partial_rotary_factor = 1.0）
position_ids: 1D (B, S) = [0, 1, 2, ..., S-1]
rope_theta = 1.2e8（原生 512K 长上下文）
```

Nanbeige **没有 MRoPE 代码路径**，`LlamaAttention.forward` 只接受 1D position_ids。

### 1.3 我们的策略：LLaVA 式 1D Sequential Position

**不实现 MRoPE**。理由：

| 因素 | 分析 |
|---|---|
| **空间信息已经由 ViT 携带** | Qwen ViT 有 learnable pos_embed (2304, 1152)，merger 输出的 5120 维特征本身包含了 (H,W) 空间结构。我们的 projector 要对齐的就是这个特征空间 |
| **Nanbeige 的 rotary_dim 全部编码** | 128 维全旋转，没有剩余维度放 3D 分解（不像 Qwen 192/256 是 non-rotary 部分） |
| **位置编码只决定序列顺序** | 对 LLM 来说，"视觉 token 在文本 token 之后" 的顺序关系由 1D RoPE 编码就够了。跨图片、跨模态的"距离"由语义（projector 学出来）和注意力机制处理 |
| **MVP 不做视频/高分辨率 split** | 不需要真正的 T 轴。单张图片 → 平面空间，H/W 网格信息在特征里 |

**实现细节**：

```python
# for each sample in batch:
#   1. tokenize chat template → input_ids
#   2. find image_pad positions → replace with projected visual embeddings
#   3. position_ids = arange(S)  # 纯 1D，不加任何空间偏移
#   4. attention_mask = 1 for all non-pad positions
```

这与 LLaVA-1.5、LLaVA-NeXT 的做法完全一致，是已被验证可行的路线。

### 1.4 未来扩展：如果要 MRoPE 怎么办？

若未来需要视频或 2D 空间定位能力，有两条路：

1. **Flatten 3D→1D**：将 T×H×W 展平为 1D 索引，对 RoPE 来说只是更长的序列（失去显式 3D 结构，但不引入新代码）
2. **自定义 rotary embedding**：改 Nanbeige 的 attention 层，让它接受 3D position_ids 并在 head_dim 上分配 section

MVP 阶段选方案 0（纯 1D），Stage 1 验证视觉对齐能工作。

---

## 2. Vision Merger 内部机制：Spatial Merge 是 MVP 的"免费降 token 器"

### 2.1 Merger 不是独立 Projector，它做了两件事

```python
# Qwen3VisionModel.forward (源码 grep 定位)
# merger 前：ViT output = (B, T*H*W, 1152)    # ViT patch grid
# merger 内部：
#   1. reshape → (B, T, H, W, 1152)
#   2. spatial_merge: 2×2 相邻 patch concat → (B, T, H/2, W/2, 1152*4=4608)
#   3. norm(4608) + Linear(4608→4608) + GELU + Linear(4608→5120)
# merger 后：(B, T*H/2*W/2, 5120)  # post-merge spatial grid
```

**关键数字**：`split_sizes = image_grid_thw.prod() // spatial_merge_size**2`

| 图像 | ViT patches (T×H×W) | Merger 后 tokens | 压缩比 |
|---|---|---|---|
| 512×512 | 1×32×32 = 1024 | 32×32÷4 = **256** | 4× |
| 768×768 | 1×48×48 = 2304 | 48×48÷4 = **576** | 4× |
| 1024×1024 | 1×64×64 = 4096 | 64×64÷4 = **1024** | 4× |
| 1568×1568 | 1×98×98 = 9604 | 98×98÷4 = **2401** | 4× |
| 2048×2048 | 1×128×128 = 16384 | 128×128÷4 = **4096** | 4× |

### 2.2 Merger 的 5120 输出到底是什么

Qwen merger 的 `out_hidden_size` 是 **5120 = Qwen LLM hidden_size**。在原 Qwen3-VL 模型里，merger 输出直接接 Qwen LLM 的 embedding layer（因为 embedding → 5120 → LLM 的输入是 5120）。

对我们来说：**merger 输出 = visual token embedding in Qwen LLM space**，5120 维。我们要做的是把这 5120 维**线性变换**到 Nanbeige LLM space 的 7168 维。

### 2.3 我们的 Custom Projector 截断点

```text
Qwen ViT (1152, 27层) → 完整保留，冻结
    ↓
Qwen Merger (LN+Linear+GELU+Linear, 输出 5120) → 完整保留，冻结
    ↓
Custom Projector: Linear(5120→7168) + GELU + Linear(7168→7168) → 新建，训练
    ↓
Nanbeige LLM (7168, 52层) → 完整保留，冻结
```

**不能在 ViT 输出 1152 维处截断**——merger 内部有 spatial_merge（2×2 concat → 4608 维中间态），这是**空间信息压缩和重组**的核心步骤。截断到 ViT 会丢失这个信息。

### 2.4 可选的额外 Token Compression（tl.md Mode B/C）

Merger 已经做了 4× spatial merge。如果 576 个 token 仍然太多（比如 2048×2048 → 4096 tokens），可以**在 projector 之前**加：

- **Mode B: Average Pooling**：(B, N, 5120) → reshape to (B, H/2, W/2, 5120) → 2×2 avg pool → (B, N/4, 5120)
- **Mode C: Perceiver Resampler**：learnable queries cross-attend over visual features → 256 fixed tokens

MVP 阶段**先不加**。先让 576 tokens 跑通，再根据显存/速度决定是否加 compressor。

---

## 3. Token ID 与 Vocab 改造：零扩表

### 3.1 三个 Vision 特殊 Token

| Token | ID | 来源 |
|---|---|---|
| `<\|vision_start\|>` | 166107 | 新建（空闲槽） |
| `<\|vision_end\|>` | 166108 | 新建（空闲槽） |
| `<\|image_pad\|>` | 166109 | 新建（空闲槽） |

**验证**（实测）：

```text
Nanbeige vocab_size = 166912
base sp vocab: 166100 tokens (id 0-166099, 无空洞)
added_tokens: 10 个 (id 0,1,2,166100-166106)
空闲槽: 166107 - 166911 = 805 个
→ embedding 矩阵 166912 行全保留
→ 新 3 个 token 在 166107-166911 内
→ base vocab 和 added_tokens 里都没有占用
→ 当前 decode 166107 = 空字符串
```

**零风险**：不需要 `resize_token_embeddings`，不需要重写 tokenizer.json，不需要改任何现有 token 的编码。

### 3.2 Chat Template 改造

Nanbeige 现有 chat template 已包含 `visible_text` 宏，当前对 image/video 内容输出：

```jinja
<reminder>You are unable to process this image because you don't have multi-modal input ability. Try different methods.</reminder>
```

改为：

```jinja
{%- elif item is mapping and item.type in ['image', 'image_url'] -%}
    {{- '<|vision_start|><|image_pad|><|vision_end|>' }}
```

**注意**：chat template 只生成**占位符**（一个 `<|image_pad|>`），**实际 token 展开**（576 个 `<|image_pad|>`）由我们自己的 processor 完成（模仿 Qwen processor 的 `expand_tokens` 逻辑）。

### 3.3 我们自己的 Multimodal Processor

Nanbeige 没有 vision processor，我们需要自建：

```python
class NanbeigeVisionProcessor:
    def __init__(self, qwen_processor, nanbeige_tokenizer):
        self.qwen_image_processor = qwen_processor.image_processor  # 复用 patch 16 merge 2 mean/std 0.5
        self.tokenizer = nanbeige_tokenizer
    
    def __call__(self, text, images=None, **kwargs):
        # 1. 用 Qwen image_processor 预处理图像 → pixel_values, image_grid_thw
        img_out = self.qwen_image_processor(images, return_tensors="pt")
        # 2. 用 Nanbeige tokenizer 文本 tokenization
        tok_out = self.tokenizer(text, **kwargs)
        # 3. 对每段 <|vision_start|><|image_pad|><|vision_end|> 展开：
        #    <|vision_start|> image_pad×N <|vision_end|>
        #    其中 N = grid_thw.prod() // 4
        # 4. 拼接 attention_mask
        return MultimodalBatch(
            input_ids, attention_mask, pixel_values, image_grid_thw,
        )
```

---

## 4. Multimodal Forward: Embedding Injection

### 4.1 数据流向

```python
def multimodal_forward(model, pixel_values, image_grid_thw, input_ids):
    # Step 1: 文本 embedding（包括 vision_start/end/image_pad 的行）
    text_embeds = model.model.embed_tokens(input_ids)   # (B, S, 7168)

    # Step 2: Vision forward（冻结）
    with torch.no_grad():
        vis_out = vision_tower(pixel_values=pixel_values, grid_thw=image_grid_thw)
        vis_features = vis_out.last_hidden_state          # (total_vis_tokens, 5120)

    # Step 3: Projector（可训）
    projected = projector(vis_features)                  # (total_vis_tokens, 7168)

    # Step 4: 替换 image_pad 位置
    mask = (input_ids == IMAGE_PAD_ID)                   # (B, S) bool
    text_embeds[mask] = projected.to(text_embeds.dtype)  # in-place replace

    # Step 5: Nanbeige LLM forward（冻结 decoder 权重）
    outputs = model.model(
        inputs_embeds=text_embeds,
        position_ids=torch.arange(S).unsqueeze(0),       # 纯 1D
        attention_mask=attention_mask,
    )
    return outputs
```

### 4.2 Labels 处理

```python
# labels: copy input_ids, image_pad 位置设为 -100（不参与 CE）
labels = input_ids.clone()
labels[mask] = -100
# vision_start 和 vision_end token 也设为 -100？
#   → 建议设为 -100（它们是结构标记，不应训练语言模型预测它们）
labels[(input_ids == VISION_START_ID) | (input_ids == VISION_END_ID)] = -100
```

### 4.3 位置编码

```python
# 纯 1D sequential，LLaVA 式
position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
# (1, S)，自动 broadcast 到 batch
```

**不做任何 spatial offset**。Nanbeige 的 1D RoPE 只关心"序列位置"，不关心"空间坐标"。

### 4.4 Generation 时的特殊处理

- **Vision encoding 只跑一次**（prefill 阶段），结果保存在 KV cache 里
- **Decode 阶段**：不再有 image_pad，正常 auto-regressive 生成
- **image_grid_thw 只在 prefill 时用**，decode 时不需要
- **注意**：Qwen3-VL 有 `mrope_position_deltas`（处理 vision token 偏移影响后续文本位置编码的计算），我们不需要这个——因为我们用纯 1D sequential

### 4.5 Shape 断言

```python
assert projected.shape[-1] == 7168, f"projector output wrong: {projected.shape}"
assert projected.shape[0] == mask.sum(), f"vis token count mismatch: {projected.shape[0]} vs {mask.sum()}"
assert text_embeds.shape[-1] == 7168
assert input_ids.shape == (B, S)
```

---

## 5. Generation / KV Cache 兼容

Nanbeige 是标准 Llama，generation 路径（`generate()`）完全可用。注意：

1. **use_cache**: Nanbeige config 里是 `False`，但 `generate()` 会自动启用 `StaticCache` 或 `DynamicCache`
2. **Past KV 处理**: 我们用纯 1D position_ids，past length 自动加到新生成的 token position 上，无特殊偏移
3. **不需要 Nanbeige 做任何 forward 改造**——我们在 LLM 之前替换 inputs_embeds，LLM 本身完全不动

---

## 6. GGUF 路线确认

| 组件 | GGUF 化策略 |
|---|---|
| **Nanbeige 33B 文本** | `convert_hf_to_gguf.py --outtype f16` → `llama` arch，直接支持 |
| **Vision Tower** | llama.cpp 已有 Qwen3-VL vision 支持，直接转换（无 projector 部分） |
| **Custom Projector** | 新增到 mmproj.gguf，tensor naming 约定参考现有 mmproj；可能需要加新 mmproj arch 或扩展现有 qwen3vl |

**关键**：llama.cpp 现有 `convert_hf_to_gguf.py` 支持 `--arch qwen3vl` 来转换 Qwen3-VL 的 vision + 文本一起。我们的情况是 vision tower + 纯 Llama 文本 + custom projector——**不伪装成 qwen3vl arch**，而是新增一个明确的 arch（如 `qwen3vl_llama`），因为：
- 文本 arch 是 Llama（不是 qwen3vl）
- vision projector 输出维度是 7168（不是 5120）
- 没有 MRoPE

或者，更简单的方案：**分别转换**——文本 gguf 用 `llama` arch，vision+projector mmproj 用自定义 arch，在 llama-server 里组合加载。

---

## 7. 兼容性矩阵（最终版）

| 项 | Qwen3.8-27B | Nanbeige4.3-33B | 我们的决策 |
|---|---|---|---|
| architecture | Qwen3_5ForConditionalGeneration (64-layer hybrid) | LlamaForCausalLM (52-layer full) | Qwen 文本丢弃，Nanbeige 保留 |
| hidden_size | 5120 | **7168** | Projector: 5120→7168 |
| vision hidden_size | **1152** | N/A | ViT 冻结 |
| merger output | **5120** | N/A | Merger 冻结，作为 projector 输入 |
| position_ids | 3D (3, B, S) MRoPE | 1D (B, S) 标准 | **LLaVA 式纯 1D** |
| rope_theta | 1e7 | **1.2e8** | 用 Nanbeige 的原生 1.2e8 |
| max_position | 262144 | **524288** | Nanbeige 原生 512K |
| vocab_size | 248320 | **166912** | 零扩表，用 166107-166109 |
| special tokens | 248053-248057 | 166100-166106 | 新增 166107-166109 |
| multimodal RoPE | 需要 mm_token_type_ids | 不需要 | **不实现 MRoPE** |
| DeepStack | **无** | N/A | 不需要 |
| MTP | 有 | 无 | Nanbeige 无 MTP，正常 |
| GGUF arch | qwen3vl | llama | 新增 arch 或分拆 |

---

## 8. 剩余风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| MRoPE→1D 导致视觉空间感知变差 | 高分辨率定位、多图区分能力下降 | Stage 1 先验证 Stage 2 LoRA 可部分弥补；极端场景考虑后续加 3D→1D flatten |
| vision_start/end embedding 初始化质量 | 冻结下 Stage 1 收敛慢 | 均值初始化（从现有 embedding 采样）；vision_start/end 可先不冻结只训 1 个 epoch |
| image_pad embedding 行不参与 forward（被替换） | 无直接风险，但该行初始值仍留在 embedding 矩阵 | 确认该行被设为零或忽略，训练时也不做梯度 |
| ViT 2304×2304 大图像 → 4096 tokens 显存压力 | KV cache 4096 额外条目 | Stage 1 先用 768×768→576 tokens；可选 Mode B avg pool |
| mm_token_type_ids Qwen 特有 | 我们不用 mm_token_type_ids（用简单 mask 找 image_pad 位置） | 无影响 |

---

## 附录：实测数据汇总

### image_grid_thw → visual token count（processor 实测）

| Image size | grid_thw (T,H,W) | ViT patches | Post-merger tokens |
|---|---|---|---|
| 512×512 | [1, 32, 32] | 1024 | 256 |
| 672×672 | [1, 42, 42] | 1764 | 441 |
| 768×768 | [1, 48, 48] | 2304 | 576 |
| 1024×1024 | [1, 64, 64] | 4096 | 1024 |
| 1280×1280 | [1, 80, 80] | 6400 | 1600 |
| 1568×1568 | [1, 98, 98] | 9604 | 2401 |
| 2048×2048 | [1, 128, 128] | 16384 | 4096 |

### Token ID 安全验证

```text
Nanbeige base vocab: 166100 tokens, id 0-166099, 无空洞
added_tokens: 10 tokens, id 0,1,2,166100-166106
空闲: 166107-166911 = 805 slots
proposed vision tokens: 166107, 166108, 166109 → 全部 CLEAN ✅
```

### 多图测试

```text
768×768 (grid [1,48,48]) → 576 image_pad tokens
1024×1024 (grid [1,64,64]) → 1024 image_pad tokens
Multi-image total: 1600 image_pad tokens ✅
```

---

## 下一步（Step 3：最小 PyTorch PoC）

1. `src/vision/qwen_vision.py`：从 safetensors 只加载 `model.visual.*`（27 层 ViT + merger），冻结
2. `src/vision/projector.py`：`Linear(5120→7168) + GELU + Linear(7168→7168)`，`vision_feature_dim` 从 `out_hidden_size` 自动读
3. `src/modeling/multimodal_model.py`：wrapper，embedding 注入 + labels mask + 纯 1D position_ids
4. `src/data/processor.py`：自建 NanbeigeVisionProcessor（复用 Qwen image_processor + Nanbeige tokenizer + 展开 image_pad）
5. 跑 vision forward → projector → embedding replace → 单图 generation
6. shape 断言全覆盖

**当前发现已足够进入 PoC 实现**，无需更多配置/shape 分析。
