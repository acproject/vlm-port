#!/usr/bin/env bash
# NanbeigeVL 本地推理服务（OpenAI 兼容，基于 llama.cpp llama-server）
#
# 模型说明：Nanbeige4.3-33B-merged/ 是 LoRA merge 后的 F32 源权重（124G，无法直接进显存），
# 本服务加载由它导出的 GGUF 产物（BF16 / Q4_K_M / Q8_0），并挂载 mmproj 提供视觉能力。
#
# 用法:
#   ./serve_nanbeige.sh                       # 默认 Q4_K_M + mmproj, 端口 8081
#   MODEL=q8_0  ./serve_nanbeige.sh           # 换 Q8_0（更高精度）
#   MODEL=bf16  ./serve_nanbeige.sh           # 换 BF16（3 卡张量切分 ~22G/卡）
#   CTX=524288 KV=q8_0 ./serve_nanbeige.sh    # 512K 上下文（KV 量化省显存）
#   PORT=9000 NGL=0 ./serve_nanbeige.sh       # 换端口 / 纯 CPU
#
# 接口（OpenAI 兼容）:
#   POST /v1/chat/completions    支持 messages + image_url（base64 data URI，视觉）
#   POST /v1/completions
#   GET  /v1/models, /health
#
# 示例:
#   curl http://localhost:8081/v1/chat/completions -H 'Content-Type: application/json' \
#     -d '{"messages":[{"role":"user","content":"你好"}]}'
set -euo pipefail

# ── 可配置项（环境变量覆盖） ──
MODEL="${MODEL:-q4_k_m}"          # q4_k_m | q8_0 | bf16 | base_q8_0
PORT="${PORT:-8081}"
HOST="${HOST:-0.0.0.0}"
CTX="${CTX:-32768}"               # 上下文长度（模型原生支持至 524288）
NGL="${NGL:-99}"                  # GPU 层数，0 = 纯 CPU
KV="${KV:-f16}"                   # KV cache 精度: f16 | q8_0（512K 上下文建议 q8_0）
API_KEY="${API_KEY:-}"            # 留空 = 不鉴权
T="${T:-16}"                      # CPU 线程数

GGUF_DIR="/home/acproject/usb/transplant/gguf"
SERVER="/home/acproject/workspace/cpp_projects/llama.cpp/build/bin/llama-server"

case "$MODEL" in
  q4_k_m)    F="$GGUF_DIR/Nanbeige4.3-33B-VLM-Q4_K_M.gguf" ;;
  q8_0)      F="$GGUF_DIR/Nanbeige4.3-33B-VLM-Q8_0.gguf" ;;
  bf16)      F="$GGUF_DIR/Nanbeige4.3-33B-VLM-BF16.gguf" ;;
  base_q8_0) F="$GGUF_DIR/Nanbeige4.3-33B-base-Q8_0.gguf" ;;
  *)         F="$MODEL" ;;  # 允许直接传 gguf 绝对路径
esac
MMPROJ="$GGUF_DIR/mmproj-NanbeigeVL-F16.gguf"
[ -f "$MMPROJ" ] || MMPROJ=""    # base 模型无视觉时自动去掉 mmproj

# 与训练完全一致的 ChatML 模板（原生模板带 <think> 块，会把回答解析进 reasoning_content）
CHAT_TMPL='<|im_start|>system\n你是一个具备图像理解能力的多模态助手。<|im_end|>\n{%- for message in messages %}<|im_start|>{{ message.role }}\n{%- if message.content is string %}{{ message.content }}{%- else %}{%- for item in message.content %}{%- if item.type == "image" %}<__media__>{%- elif item.type == "text" %}{{ item.text }}{%- endif %}{%- endfor %}{%- endif %}<|im_end|>\n{%- endfor %}{%- if add_generation_prompt %}<|im_start|>assistant\n{%- endif %}'

ARGS=(
  -m "$F" --alias nanbeige-vlm
  --host "$HOST" --port "$PORT"
  -c "$CTX" -ngl "$NGL" -t "$T"
  --cache-type-k "$KV" --cache-type-v "$KV"
  --chat-template "$CHAT_TMPL"
  --metrics                    # /metrics Prometheus 指标
)
[ -n "$MMPROJ" ] && ARGS+=(--mmproj "$MMPROJ")
[ -n "$API_KEY" ] && ARGS+=(--api-key "$API_KEY")

echo ">>> llama-server: model=$F ctx=$CTX ngl=$NGL kv=$KV port=$PORT mmproj=${MMPROJ:-none}"
exec "$SERVER" "${ARGS[@]}"
