# Nanbeige4.3-33B-VLM

将 Qwen3.8-27B 的视觉能力移植到 Nanbeige4.3-33B 文本模型上，产出可推理的 VLM。

## llama.cpp 分支说明

本仓库内的 `llama.cpp/` 是上游 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) 的克隆，额外维护了一个分支：

| 分支 | 说明 |
|---|---|
| `nanbeige-vl` | **本项目使用**。新增 `qwen3vl_mlp` projector 类型：复用 Qwen3.8-27B 的 ViT + merger，并在其后追加自定义 projector（RMSNorm → Linear 5120→7168 → GELU(erf) → Linear 7168→7168）。缺少该分支的 `libmtmd` 时，加载 `mmproj-NanbeigeVL-*.gguf` 会失败或走错 projector。 |
| `master` | 上游原版，不支持 NanbeigeVL 的 mmproj。 |

### 切换分支

```bash
cd llama.cpp

# 查看当前分支与本地改动
git status -sb

# 切到本项目分支
git checkout nanbeige-vl

# 需要对照上游原版时切回（注意：切回前先确认没有未提交改动）
git checkout master
```

首次克隆或分支丢失时，可从 fork 直接拉取：

```bash
git clone -b nanbeige-vl https://github.com/acproject/llama.cpp.git
```

### 推送分支

`llama.cpp` 里有两个 remote：

| remote | 地址 | 权限 |
|---|---|---|
| `origin` | `https://github.com/ggml-org/llama.cpp.git` | 只读（上游，**不要 push**） |
| `fork` | `https://github.com/acproject/llama.cpp.git` | 可写（本项目 fork） |

推 `origin` 会失败（没有上游写权限），正确做法是推到 `fork`：

```bash
cd llama.cpp

# 首次：若还没有 fork remote 则添加
git remote add fork https://github.com/acproject/llama.cpp.git

# 推送分支（首次带 -u 建立跟踪）
git push -u fork nanbeige-vl
```

拉取上游最新代码用 `origin`，推送自己的改动用 `fork`：

```bash
git fetch origin && git rebase origin/master   # 同步上游
git push fork nanbeige-vl                       # 发布改动
```

## 编译

本机配置：Release、CUDA 开启、native 指令集、44 线程；编译器 `gcc-14` + `g++-patched`。

```bash
cd llama.cpp

# 配置（首次或修改选项时执行）
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON

# 编译所需目标
cmake --build build -j"$(nproc)" \
  --target llama-server llama-cli llama-quantize
```

产物位于 `llama.cpp/build/bin/`：`llama-server`、`llama-cli`、`llama-quantize`、`libmtmd.so` 等。

> 若只是改了 `tools/mtmd/` 下的代码，重新执行一次 `cmake --build build -j"$(nproc)"` 即可增量编译；
> 切换分支后建议先 `cmake --build build -j"$(nproc)" --clean-first` 避免旧对象文件混用。

### 验证编译结果

```bash
# 应能查到 qwen3vl_mlp 相关符号
nm -D llama.cpp/build/bin/libmtmd.so | grep qwen3vl_mlp
```

## 推理服务启动

```bash
cd llama.cpp
./build/bin/llama-server \
  -m ../gguf/v2/Nanbeige4.3-33B-VLM-v2-Q4_K_M.gguf \
  --mmproj ../gguf/v2/mmproj-NanbeigeVL-F16-v2.gguf \
  -c 32768 -ngl 0 -t 16 -np 1 \
  --api-key nbgvlm2026 \
  --chat-template '<|im_start|>system\n你是一个具备图像理解能力的多模态助手。<|im_end|>\n{%- for message in messages %}<|im_start|>{{ message.role }}\n{%- if message.content is string %}{{ message.content }}{%- else %}{%- for item in message.content %}{%- if item.type == "image" %}<__media__>{%- elif item.type == "text" %}{{ item.text }}{%- endif %}{%- endfor %}{%- endif %}<|im_end|>\n{%- endfor %}{%- if add_generation_prompt %}<|im_start|>assistant\n{%- endif %}'
```

关键点：

- `--chat-template` 必须覆盖 GGUF 内嵌的 Qwen3 模板（其默认注入 `<think>`），模型训练时用的是无 think 的纯 ChatML。
- `-np 1` 单 slot，避免多 slot 按相似度复用上下文污染结果。
- 注意：本机上游仓库为 `origin=https://github.com/ggml-org/llama.cpp.git`，**不要向 origin 推送**本项目分支。
