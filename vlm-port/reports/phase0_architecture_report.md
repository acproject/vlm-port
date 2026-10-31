# Phase 0 — Architecture Compatibility Report

> 生成方式：直接读取两个模型目录的 config.json / tokenizer_config.json / preprocessor_config.json /
> model.safetensors.index.json，并通过 safetensors header 读取真实 tensor shape（未加载权重数据，未根据模型名猜测）。
>
> - 多模态模型：`/home/acproject/usb/transplant/Qwen3.8-27B`（实际为 27B，`Qwen3_5ForConditionalGeneration`）
> - 文本模型：`/home/acproject/usb/transplant/Nanbeige4.3-33B`（`LlamaForCausalLM`）

---

## 1. Qwen3.8-27B Vision Architecture Report

### 1.1 总览

| 项 | 值 |
|---|---|
| architectures | `Qwen3_5ForConditionalGeneration` |
| model_type | `qwen3_5`（transformers 5.14.1 原生支持） |
| 总 tensor 数 | 1199（visual 333 + text 866） |
| checkpoint 总大小 | 55.56 GB（BF16） |
| DeepStack | **无**（`deepstack_visual_indexes: []`） |
| 多层视觉 feature injection | **无** |
| 独立 multimodal projector | **无** —— merger 即 projector |

### 1.2 Vision Tower（`model.visual.*`）

| 项 | 值 |
|---|---|
| vision encoder class | SigLIP 风格 ViT（fused QKV，LayerNorm 带 bias，GELU-tanh MLP） |
| vision hidden size | **1152** |
| depth | 27 |
| num_heads | 16 |
| patch_size | 16 |
| spatial_merge_size | 2 |
| temporal_patch_size | 2 |
| 位置编码 | learnable `pos_embed` (2304, 1152)，原生 48×48 patch 网格，超分辨率时插值 |
| intermediate_size | 4304 |

关键 tensor shape（block 0 代表，27 层全部一致）：

```text
model.visual.patch_embed.proj.weight   (1152, 3, 2, 16, 16)   # Conv3d, temporal_patch=2
model.visual.pos_embed.weight          (2304, 1152)
model.visual.blocks.{0..26}.attn.qkv.weight      (3456, 1152)   # 3×1152 fused
model.visual.blocks.{i}.attn.proj.weight         (1152, 1152)
model.visual.blocks.{i}.mlp.linear_fc1.weight    (4304, 1152)
model.visual.blocks.{i}.mlp.linear_fc2.weight    (1152, 4304)
model.visual.blocks.{i}.norm1/norm2.weight       (1152,)
```

### 1.3 Vision Merger（`model.visual.merger.*`）—— 这就是"5120"的来源

```text
model.visual.merger.norm.weight         (1152,)      # pre-norm（LN，带 bias）
model.visual.merger.linear_fc1.weight   (4608, 4608) # in = 1152×(2×2 merge) = 4608
model.visual.merger.linear_fc2.weight   (5120, 4608) # out = out_hidden_size = 5120
```

结构：concat 2×2 相邻 patch → LN → Linear(4608→4608) → GELU → Linear(4608→5120)。
输出**直接进入 Qwen LLM 的 embedding 空间**（5120），没有额外的 projector。

### 1.4 5120 到底是什么（明确回答）

| 候选 | 实际值 |
|---|---|
| Vision Encoder hidden size | **1152**（不是 5120） |
| Merger output (`out_hidden_size`) | **5120** ✅ |
| 独立 Projector output | 不存在独立 projector |
| Qwen LLM hidden size | **5120**（与 merger 输出一致） |

**结论：观察到的 5120 = merger 输出 = Qwen LLM hidden size。我们需要的 projector 输入维度是 merger 的输出 5120，输出维度是 7168。**

### 1.5 Qwen 文本侧（仅供了解，移植时整体丢弃）

- 64 层混合架构：48 × linear_attention（gated deltanet：`linear_attn.in_proj_qkv (10240,5120)`、`conv1d (10240,1,4)`、`A_log/dt_bias (48,)`）+ 16 × full_attention（每 4 层 1 个，`self_attn.q_proj (12288,5120)`，attn_output_gate=true 所以是 2×6144）
- MRoPE：interleaved，section [11,11,10]，partial_rotary_factor 0.25，theta 1e7
- MTP 头 1 层；vocab 248320；max_position 262144
- tensor 前缀：`model.language_model.*`、`lm_head.*`、`mtp.*`

### 1.6 特殊 token / Processor / Template

| token | id |
|---|---|
| `<|vision_start|>` | 248053 |
| `<|vision_end|>` | 248054 |
| `<|vision_pad|>` | 248055 |
| `<|image_pad|>` | 248056 |
| `<|video_pad|>` | 248057 |
| eos `<|im_end|>` / pad `<|endoftext|>` | 248046 / 248044 |

- processor：`Qwen3VLProcessor`（image: `Qwen2VLImageProcessor`，mean/std=0.5，patch 16，merge 2）
- 图片在模板中的格式：`<|vision_start|><|image_pad|>...<|vision_end|>`（可选 `Picture N: ` 前缀）
- 像素预算（Qwen3-VL 面积语义）：65536 ~ 16777216 px²（实现时用实际 processor 验证）

### 1.7 视觉 token 数量公式

```text
每 token = (patch 16 × merge 2)² = 32×32 px
tokens(H,W) = ceil(H/32) × ceil(W/32)

768×768   → 576 tokens
1024×1024 → 1024 tokens
1280×1280 → 1600 tokens
2048×2048 → 4096 tokens
```

---

## 2. Target 33B (Nanbeige4.3-33B) Architecture Report

| 项 | 值 |
|---|---|
| architecture | `LlamaForCausalLM`（标准 Llama，llama.cpp arch=`llama`） |
| hidden_size | **7168** ✅（与用户观察一致） |
| num_hidden_layers | 52（全 full attention，无 hybrid） |
| num_attention_heads | 56 |
| num_key_value_heads | 8（GQA 7:1） |
| head_dim | 128 |
| intermediate_size | 22016 |
| vocab_size | 166912（embed/lm_head 均为 (166912, 7168)，untied） |
| rms_norm_eps | 1e-5 |
| RoPE | 标准 1D 全旋转（head_dim 128），theta = **1.2e8**，无 rope_scaling |
| max_position_embeddings | **524288（原生 512K）**，seq_length 524288 |
| BOS/EOS | `<|im_start|>`=166100 / `<|im_end|>`=166101 |
| checkpoint | 471 tensors，66.24 GB BF16，52 层 shape 全部 uniform |

关键 shape：

```text
model.embed_tokens.weight        (166912, 7168)
lm_head.weight                   (166912, 7168)
model.layers.{i}.self_attn.q_proj.weight  (7168, 7168)   # 56×128
model.layers.{i}.self_attn.k_proj.weight  (1024, 7168)   # 8×128
model.layers.{i}.self_attn.v_proj.weight  (1024, 7168)
model.layers.{i}.self_attn.o_proj.weight  (7168, 7168)
model.layers.{i}.mlp.gate_proj.weight     (22016, 7168)
```

### 词表空闲槽位（重要发现）

- tokenizer.json 基础词表 166100 条（id 0–166099，无空洞），added tokens 到 166106
- embedding 矩阵 166912 行 → **id 166107–166911 共 805 行完全空闲**（值近零，std≈1e-6，无 NaN）
- ⇒ 添加 vision 特殊 token **无需 resize_token_embeddings，原文本权重 0 改动**
- chat template 已含 `visible_text` 宏，能识别 image/video 内容类型（当前输出"无法处理多模态"提示，后续替换为 vision token 序列）

---

## 3. Compatibility Report

### 3.1 核心维度链路

```text
Image
  → Qwen ViT (1152, 27层, 冻结)
  → Qwen Merger (输出 5120, 冻结)          ← projector 截断点：merger 输出之后
  → Custom Projector Linear(5120→7168)+GELU+Linear(7168→7168)  ← 唯一新建训练模块
  → Nanbeige 33B LLM (7168, 冻结)          ← embedding 注入，decoder 不动
```

### 3.2 明确回答六个问题

| 问题 | 答案 |
|---|---|
| vision feature dim | merger 输出 **5120**（ViT 本身 1152） |
| Qwen multimodal output dim | **5120**（= Qwen LLM hidden size） |
| target hidden size | **7168** |
| projector replacement point | Qwen merger 输出之后（即完整保留 `model.visual.*`，替换掉"merger→Qwen LLM"这一恒等对接） |
| special tokens | Nanbeige 侧新增 `<|vision_start|>`=166107、`<|vision_end|>`=166108、`<|image_pad|>`=166109（零扩表） |
| position encoding 差异 | Qwen：interleaved MRoPE [11,11,10] + partial 0.25 + theta 1e7（262144）；Nanbeige：标准 1D 全旋转 RoPE + theta 1.2e8（原生 512K）。**移植后视觉 token 按普通 1D 位置注入，无需实现 MRoPE**（LLaVA 式做法；空间结构信息由 ViT pos_embed + merger 输出携带） |

### 3.3 保留 / 丢弃 / 冻结 / 新建 / 训练

| 模块 | 处置 |
|---|---|
| Qwen ViT 27 层 + patch_embed + pos_embed | **保留，冻结**（~0.4B 参数） |
| Qwen merger | **保留，冻结** |
| Qwen 文本模型 + lm_head + MTP（~52GB） | **丢弃，不加载** |
| Nanbeige 33B 全部权重 | **保留，冻结**（Stage 1） |
| Custom Projector (5120→7168) | **新建，训练** |
| vision_start/end/image_pad 三个 embedding 行 | **新初始化**（均值或 copy 初始化；image_pad 行实际总被 visual embedding 替换，不参与 forward） |
| 可选 token compressor（Mode A/B/C） | 新建（Mode A 先行） |

### 3.4 兼容性结论

1. **可行**：Qwen vision tower 与目标 LLM 完全解耦（vision 输出只是替换 sequence embedding），无任何结构冲突。
2. **文本性能零损失**：Nanbeige decoder 权重与结构 100% 不动；无图输入 = 原模型行为（新增 3 个特殊 token 不影响任何现有 token 的编码）。
3. **长上下文**：Nanbeige 原生 512K（theta 1.2e8 无 scaling），无需 YaRN；512K 主要瓶颈是 KV cache（52 层 × 8 kv heads × 128 dim），后续按 tl.md 十六章做量化组合 benchmark。
4. **GGUF 路线清晰**：文本侧 = 标准 llama arch（convert_hf_to_gguf.py 直接支持）；mmproj 侧 = llama.cpp 已有 Qwen3-VL vision 实现 + 我们的 custom projector（输出 7168），无需伪装成 Qwen3-VL LLM。
5. **风险**：① MRoPE→1D 会丢失显式 3D 位置分解（Qwen 用它做高分辨率/视频定位；MVP 可接受，Stage 2 LoRA 可弥补）；② vision_start/end embedding 冻结下的初始化质量会影响 Stage 1 效果（用均值初始化缓解）；③ Qwen3-VL processor 的面积缩放行为需在 PoC 中实测确认（65536–16777216 px² 预算）。

### 3.5 下一步（Step 2：最小改造方案）

1. `src/vision/qwen_vision.py`：从 Qwen checkpoint 只加载 `model.visual.*`（transformers 5.14.1 原生 `qwen3_5`），冻结。
2. `src/vision/projector.py`：`Linear(5120→7168) + GELU + Linear(7168→7168)`（可配 RMSNorm/SiLU 变体），`vision_feature_dim` 从 vision config 的 `out_hidden_size` 自动读取。
3. `src/modeling/multimodal_model.py`：wrapper，把 image_pad 位置 embedding 替换为 projected visual embeddings，labels 对 image 位置 mask。
4. tokenizer 增加 3 个特殊 token（id 166107–166109，零扩表）。
5. shape 断言：`merger_out.shape[-1]==5120`、`projected.shape[-1]==7168`。

---

## 附录：环境

- venv：`/home/acproject/usb/transplant/venv-cu128`（torch 2.8.0+cu128, transformers 5.14.1, safetensors 0.8.0, accelerate 1.14.0, qwen-vl-utils 0.0.14）
- 检查脚本：`vlm-port/scripts/inspect_qwen.py`、`vlm-port/scripts/inspect_target.py`（只读 header，可重复运行）
