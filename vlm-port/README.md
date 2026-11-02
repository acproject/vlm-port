# NanbeigeVL — Qwen3.8 Vision + Nanbeige4.3-33B 视觉语言模型移植

将 Qwen3.8-27B 的视觉能力移植到 Nanbeige4.3-33B 文本模型上，产出可推理的 VLM，
并完成 llama.cpp / GGUF 全套导出（BF16 / Q4_K_M / Q8_0 / mmproj）。

## 1. 架构

```
图像 ──► Qwen3.8-27B Vision Tower (冻结, 460.7M)
          │  27 层 ViT, hidden 1152, out_hidden 5120
          │  learned abs pos_embed (2304×1152, 双线性插值)
          │  M-RoPE theta=10000, 2×2 spatial merge
          ▼
        Qwen Merger (冻结)
          │  LayerNorm(1152) per-patch → fc1(4608→4608)+GELU → fc2(4608→5120)
          ▼
        CustomProjector (可训练, 88.10M)
          │  RMSNorm(5120, eps 1e-6) → Linear(5120→7168) → GELU(erf) → Linear(7168→7168)
          ▼
        Nanbeige4.3-33B (标准 Llama, LoRA r=16 α=32 作用于全部 364 个 Linear)
          │  hidden 7168, 52 层, 56 头/8 KV 头, vocab 166912
          │  原生上下文 512K (max_position_embeddings=524288, rope_theta=120M, 无需 YaRN)
          ▼
        文本输出
```

- **Stage 1 结论**（v1/v2 数据，4 轮诊断）：冻结 LLM + 只训 projector 会结构性塌缩
  （cross-image cos 0.998，loss 平台 8.89），LLM 必须参与适配。
- **Stage 2 方案**：随机初始化 projector + zero-init LoRA-B → step-0 前向即完整 base LLM，
  projector(lr 1e-4) 与 LoRA(lr 2e-4) 联合训练。

## 2. 训练结果（Stage 2）

| 指标 | 值 |
|---|---|
| 训练 | 750 optimizer steps, 2 epochs × 3000 样本, grad_accum 8 |
| 硬件/精度 | 3× V100-SXM2-32GB, bf16(模拟) + projector fp32 |
| VAL loss 轨迹 | 0.2483 (step 575) → 0.1846 (700) → **0.1796 (750, best)** |
| 生成 eval | 严格匹配 6/12；语义正确 **11/12**（MISS 多为句式差异，仅 1 个真实错误） |
| 训练时长 | 251 分钟（含一次 step 600 中断恢复） |

最终 checkpoint：`checkpoints/stage2/ckpt_final.pt`（projector + LoRA + optimizer 状态）。

## 3. 产出物

| 文件 | 大小 | 说明 |
|---|---|---|
| `../Nanbeige4.3-33B-merged/` | 124G | LoRA merge 后 F32 safetensors（23 分片，含 config/tokenizer） |
| `../gguf/Nanbeige4.3-33B-VLM-BF16.gguf` | 66.2G | 文本模型 BF16（标准 LLAMA arch） |
| `../gguf/Nanbeige4.3-33B-VLM-Q4_K_M.gguf` | 19.1G | 4.84 BPW |
| `../gguf/Nanbeige4.3-33B-VLM-Q8_0.gguf` | 33.5G | 8.50 BPW |
| `../gguf/mmproj-NanbeigeVL-F16.gguf` | 1.03G | 视觉塔 + merger + projector（`clip.projector_type=qwen3vl_mlp`） |
| `../gguf/Nanbeige4.3-33B-base-Q8_0.gguf` | 33G | 未训练 base（管线验证/文本基线用） |

## 4. llama.cpp 支持（本地补丁）

文本模型是标准 HF Llama 布局（`model_type: llama`），转换走 `convert_hf_to_gguf.py`
标准 LLAMA 路径，`conversion/nanbeige.py` 的 undo_permute **不适用**。

视觉侧因双层堆叠 MLP（Qwen merger + CustomProjector）无现成 projector 类型，
在 `tools/mtmd` 新增 `PROJECTOR_TYPE_QWEN3VL_MLP`（"qwen3vl_mlp"）：

- `clip-impl.h` — 枚举 + 名字表
- `clip.cpp` — 图分发 / hparams / 张量加载（mm.0/mm.2=merger，mm.input_norm+mm.4/mm.6=projector）/ positions / 输出维度
- `models/qwen3vl_mlp.cpp` — 复用 `clip_graph_qwen3vl` 图，末尾追加 RMSNorm + Linear + GELU(erf) + Linear
- `mtmd.cpp` — img_beg/img_end = `<|vision_start|>`/`<|vision_end|>`（与 Qwen3VL 同组）
- `CMakeLists.txt` — 编入 qwen3vl_mlp.cpp

结构已逐层对照 `qwen3vl.cpp` 与 HF `modeling_qwen3_5.py` 核实：
patch_embed 拆帧、learned pos 双线性插值、M-RoPE block-major 顺序、merger LN/fc1/fc2 映射均一致。
GELU 精度：ViT MLP 用 tanh 近似（FFN_GELU），projector 用精确 erf（FFN_GELU_ERF），与训练一致。

## 5. llama.cpp 运行

```bash
# 视觉问答（Q4_K_M）
llama-mtmd-cli -m gguf/Nanbeige4.3-33B-VLM-Q4_K_M.gguf \
  --mmproj gguf/mmproj-NanbeigeVL-F16.gguf \
  --image photo.jpg -p "描述一下这张图片。" -n 128 --temp 0 -ngl 99

# 512K 上下文（原生，无需 YaRN；KV 用 q8_0 控制内存 ≈52G，F16 需 ≈104G）
llama-cli -m gguf/Nanbeige4.3-33B-VLM-Q4_K_M.gguf \
  -c 524288 --cache-type-k q8_0 --cache-type-v q8_0 -ngl 99
```

训练用的对话格式（mtmd-cli 会套模型 chat template，图像占位符自动展开）：

```
<|im_start|>system
你是一个具备图像理解能力的多模态助手。<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|>×N<|vision_end|>问题文本<|im_end|>
<|im_start|>assistant
```

特殊 token：`<|vision_start|>`=166107，`<|vision_end|>`=166108，`<|image_pad|>`=166109；
bos=166100，eos=166101，`<|im_end|>` 为生成终止符。词表 166912 已与 GGUF 全量比对零差异。

## 6. 目录结构

```
vlm-port/
├── configs/            train_stage1.yaml / train_stage2.yaml（含 resume_from）
├── scripts/
│   ├── train_stage2.py     联合训练（param groups + clip 覆盖 LoRA + lora_state ckpt）
│   ├── merge_lora.py       流式 merge → F32 safetensors（W = W_base + B@A × 2.0）
│   ├── export_mmproj.py    三段式：合成 HF dir → convert --mmproj → 改写 projector_type+追加权重
│   └── eval_generate.py    贪心解码生成评估（single/either/joint/caption 四类）
├── src/
│   ├── vision/         qwen_vision.py（塔加载）/ projector.py（CustomProjector）
│   ├── data/           processor（chat template + image_pad 展开）/ dataset / collator
│   └── modeling/       lora.py（LoRALinear + apply_lora）/ sharded_llm.py（跨卡分片前向）
├── checkpoints/stage2/ ckpt_final.pt + step 快照
└── logs/               stage2_resume.log / gen_eval_stage2_final.log

（服务脚本在上级目录）
usb/transplant/
├── serve_nanbeige_hf.py    Python/transformers 直读 merged safetensors 的 OpenAI 兼容服务
└── serve_nanbeige.sh       llama.cpp llama-server 启动脚本（GGUF + mmproj 视觉）
```

## 6.1 OpenAI 兼容推理服务（Python / safetensors）

> 完整调用指南（curl / openai SDK / 参数表 / 行为限制 / 故障排查）见
> [`../README_SERVICE.md`](../README_SERVICE.md)。

`serve_nanbeige_hf.py` 直读 `Nanbeige4.3-33B-merged`（F32 safetensors → fp16 三卡分片）：

```bash
cd /home/acproject/usb/transplant
setsid nohup venv-cu128/bin/python serve_nanbeige_hf.py > /tmp/hf_server.log 2>&1 &
# 加载约 12-15 分钟（USB 盘读 124G）；就绪后监听 :8082
# 环境变量：PORT / DTYPE(float16|bfloat16|float32) / MAX_CTX

curl http://localhost:8082/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":64}'
# stream=true 走 SSE；另有 /v1/completions、/v1/models、/health
```

工程要点（均为本机实测踩坑）：

- **去掉 accelerate hooks + host-staged 层间路由**：本机 V100 跨卡 P2P 直传损坏数据，
  `device_map="auto"` 默认 dispatch 会输出乱码（见 `src/modeling/sharded_llm.py`）。
- **自实现 KV cache 分片生成**：每层 KV 留在本地卡，decode 只传 7168 维隐状态，
  ~10 tok/s（重算前向的 eval_generate 是 ~0.5 tok/s）。
- **生成跑在独立线程 + 队列桥接**：流式请求被客户端中途断开时，anyio 会放弃
  协程所在的生成器，若锁在生成器 `with` 内会被永久泄漏（实测复现）；独立线程
  的 `finally` 必然执行，锁总能释放。
- RoPE cos/sin 在 CPU 计算；fp16（V100 原生）；batch=1 串行。
- `kill -USR1 <pid>` 可随时 dump 全部线程栈到日志（faulthandler）。

## 7. 复现流程

```bash
cd vlm-port
# 训练（可从断点恢复，改 configs/train_stage2.yaml 的 resume_from）
../venv-cu128/bin/python -u scripts/train_stage2.py configs/train_stage2.yaml

# 最终生成评估
../venv-cu128/bin/python -u scripts/eval_generate.py configs/train_stage2.yaml

# LoRA merge → F32 safetensors
../venv-cu128/bin/python -u scripts/merge_lora.py

# 文本 GGUF + 量化
cd <llama.cpp> && PYTHONPATH=gguf-py python convert_hf_to_gguf.py <merged_dir> --outtype bf16 ...
./build/bin/llama-quantize <bf16.gguf> <q4.gguf> Q4_K_M   # 同理 Q8_0

# mmproj（用最新 ckpt_final.pt）
cd vlm-port && ../venv-cu128/bin/python -u scripts/export_mmproj.py
```

## 8. 已知问题

- V100 P2P 跨卡直传全零：labels 必须 host staging；`run_in_background` 平台故障须前台 + 长 timeout。
- 重负载任务（GGUF 写盘）与训练并发可能导致训练进程被杀：**先训练完再做转换**。
- llama-mtmd-cli 在本机 GPU 上加载 mmproj 时偶发 cudaMalloc OOM（GPU 空闲时亦出现），
  疑与沙箱环境 CUDA 上下文相关；torch 同尺寸分配正常，排查中。
