# Nanbeige4.3-33B OpenAI 兼容推理服务 — 调用指南

本地 OpenAI 兼容 HTTP 服务，直接加载 `Nanbeige4.3-33B-merged`（F32 safetensors，fp16 跨 3×V100 分片），无需 llama.cpp。

- 服务脚本：`serve_nanbeige_hf.py`
- 默认地址：`http://127.0.0.1:8082`
- 模型名：`nanbeige-4.3-33b`
- 生成速度：约 10 tok/s（fp16，KV cache 分片）
- 上下文上限：16384 tokens（超出 prompt 返回 400）

## 1. 启动 / 停止服务

```bash
# 启动（默认 :8082，fp16；从 USB 盘加载 124G 模型约需 15 分钟）
cd /home/acproject/usb/transplant
setsid nohup python serve_nanbeige_hf.py > /tmp/hf_server.log 2>&1 & disown

# 环境变量
#   MODEL_DIR  模型目录（默认 ./Nanbeige4.3-33B-merged）
#   PORT       端口（默认 8082）
#   HOST       监听地址（默认 0.0.0.0）
#   DTYPE      float16（推荐）/ bfloat16 / float32
#   MAX_CTX    上下文上限（默认 16384）

# 健康检查（加载完成前连接会被拒绝，看日志等 "model loaded"）
curl http://127.0.0.1:8082/health
# → {"status":"ok"}

# 停止
pkill -f serve_nanbeige_hf.py

# 卡住时 dump 全部线程栈到 /tmp/hf_server.log
kill -USR1 <pid>
```

## 2. 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 存活检查 |
| GET | `/v1/models` | 模型列表 |
| POST | `/v1/chat/completions` | 对话补全，支持 `stream=true` SSE |
| POST | `/v1/completions` | 文本补全（legacy，仅非流式） |

支持的请求参数（两个 POST 接口通用）：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `messages` / `prompt` | - | 必填 | 对话用 messages 数组，补全用字符串 |
| `temperature` | float | 0.7 | 0 = 贪心解码 |
| `top_p` | float | 1.0 | nucleus sampling，<1.0 启用 |
| `max_tokens` | int | 512 | 上限 4096 |
| `stream` | bool | false | chat 接口支持流式 |
| `model` | string | - | 兼容字段，忽略 |

> 仅支持这些字段；`tools` / `function_call` / `n` / `stop` 等未实现，传入会被忽略。

## 3. curl 调用

### 3.1 非流式对话

```bash
curl http://127.0.0.1:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nanbeige-4.3-33b",
    "messages": [
      {"role": "system", "content": "你是一个具备图像理解能力的多模态助手。"},
      {"role": "user", "content": "1+1等于几？"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

响应格式与 OpenAI 一致：

```json
{
  "id": "chatcmpl-xxxx",
  "object": "chat.completion",
  "created": 1762000000,
  "model": "nanbeige-4.3-33b",
  "choices": [
    {"index": 0, "message": {"role": "assistant", "content": "1+1等于2。"}, "finish_reason": "stop"}
  ],
  "usage": {"prompt_tokens": 42, "completion_tokens": 0, "total_tokens": 0}
}
```

### 3.2 流式对话（SSE）

```bash
curl -N http://127.0.0.1:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nanbeige-4.3-33b",
    "messages": [{"role": "user", "content": "介绍一下北京"}],
    "stream": true
  }'
```

每帧 `data: {...}`，`delta.content` 为增量文本，以 `data: [DONE]` 结束。

### 3.3 文本补全（legacy）

```bash
curl http://127.0.0.1:8082/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "中国的首都是", "max_tokens": 32, "temperature": 0}'
```

## 4. Python 调用

### 4.1 openai SDK（推荐）

```bash
pip install openai
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8082/v1", api_key="none")

# 非流式
resp = client.chat.completions.create(
    model="nanbeige-4.3-33b",
    messages=[{"role": "user", "content": "1+1等于几？"}],
    temperature=0.7,
)
print(resp.choices[0].message.content)

# 流式
stream = client.chat.completions.create(
    model="nanbeige-4.3-33b",
    messages=[{"role": "user", "content": "介绍一下北京"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

### 4.2 requests / httpx

```python
import json, requests

r = requests.post("http://127.0.0.1:8082/v1/chat/completions", json={
    "model": "nanbeige-4.3-33b",
    "messages": [
        {"role": "user", "content": "我叫小明，记住我的名字"},
        {"role": "assistant", "content": "好的，你叫小明。"},
        {"role": "user", "content": "我叫什么？"},
    ],
})
print(r.json()["choices"][0]["message"]["content"])
```

## 5. 行为说明与限制

- **多轮对话**：直接传完整 `messages` 历史即可，服务端按 ChatML 格式拼接（与训练格式一致，无 think 块），生成在 `<|im_end|>` 处停止。
- **system 消息**：只取第一条；未提供时自动插入默认 system（“你是一个具备图像理解能力的多模态助手。”）。
- **纯文本服务**：本服务不接收图像输入。视觉链路走 llama.cpp（`serve_nanbeige.sh` + `gguf/mmproj-NanbeigeVL-F16.gguf`）。
- **并发 = 1**：同一时刻只允许一个生成任务（内部全局锁串行化），并发请求会排队等待。排队与耗时见日志 `[gen] start / lock acquired / done/released`。
- **batch=1**：不支持 `n > 1` 批量采样。
- **流式断开安全**：客户端中途断开会排队照常生成完但结果被丢弃，不影响后续请求（锁不会泄漏）。
- **usage 统计**：`completion_tokens` 恒为 0，未做精确计数。

## 6. 故障排查

| 现象 | 处理 |
|---|---|
| 连接被拒绝 | 模型还在加载（约 15 分钟），`tail -f /tmp/hf_server.log` 等待 `model loaded` |
| 请求 400 | prompt 超过 `MAX_CTX - 512` tokens，缩短输入或调大 `MAX_CTX` 重启 |
| 请求挂住 | 有前序生成占锁，看日志 `[gen] lock acquired` 时间戳确认；必要时 `kill -USR1 <pid>` dump 栈 |
| 服务静默退出 | 偶发，重启即可；日志在 `/tmp/hf_server.log` |
