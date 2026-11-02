#!/usr/bin/env python
"""Nanbeige4.3-33B-merged (safetensors) 本地 OpenAI 兼容推理服务。

直接加载 /home/acproject/usb/transplant/Nanbeige4.3-33B-merged 的 F32 safetensors
（fp16 跨 3×V100 分片），提供：

  GET  /health                存活检查
  GET  /v1/models             模型列表
  POST /v1/chat/completions   对话（支持 stream=true SSE 流式）
  POST /v1/completions        补全（legacy）

关键设计（与训练栈 vlm-port/src/modeling/sharded_llm.py 一致）：
  本机 V100 跨卡 P2P 直传损坏数据，accelerate dispatch hooks 的层间激活传递
  正走此通道 → 必须去掉 hooks、用 host staging（cpu 中转）手动路由层间张量。
  RoPE cos/sin 在 CPU 计算（inv_freq 跨卡 .to() 同样会损坏）。
  每层 KV cache 留在该层所在 GPU，decode 时只传 7168 维隐状态跨卡。

用法：
  python serve_nanbeige_hf.py                        # 默认 :8082, fp16
  DTYPE=bfloat16 python serve_nanbeige_hf.py         # bf16（V100 上很慢，不推荐）
  PORT=9000 python serve_nanbeige_hf.py

对话格式与训练一致（ChatML，无 <think> 块）：
  <|im_start|>system\n你是一个具备图像理解能力的多模态助手。<|im_end|>
  <|im_start|>user\n...<|im_end|>
  <|im_start|>assistant\n
"""
from __future__ import annotations

import faulthandler
import os
import queue
import signal
import sys
import time
import uuid
from threading import Lock, Thread
from typing import Iterator

import torch

faulthandler.register(signal.SIGUSR1)   # kill -USR1 <pid> 可随时 dump 全部线程栈到日志
torch.set_num_threads(4)
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/acproject/usb/transplant/vlm-port/src")
from modeling.sharded_llm import remove_accelerate_hooks, _stage  # noqa: E402

MODEL_DIR = os.environ.get("MODEL_DIR", "/home/acproject/usb/transplant/Nanbeige4.3-33B-merged")
PORT = int(os.environ.get("PORT", "8082"))
HOST = os.environ.get("HOST", "0.0.0.0")
DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
    os.environ.get("DTYPE", "float16")
]
QUANT = os.environ.get("QUANT", "none").lower()        # none | 4bit (bnb NF4)
MAX_CTX = int(os.environ.get("MAX_CTX", "16384"))
DEFAULT_SYS = "你是一个具备图像理解能力的多模态助手。"


# ── 加载模型 ──────────────────────────────────────────────────────────
print(f"[serve] loading {MODEL_DIR} dtype={DTYPE} quant={QUANT} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
IM_END = tok.convert_tokens_to_ids("<|im_end|>")
if QUANT == "4bit":
    from transformers import BitsAndBytesConfig
    # 按各卡实时剩余显存（90%）显式传 max_memory，避免 device_map 把层放到 CPU
    max_memory = {}
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        max_memory[i] = int(free * 0.9)
    print(f"[serve] max_memory={max_memory}", flush=True)
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=DTYPE,
        ),
        device_map="auto", max_memory=max_memory, low_cpu_mem_usage=True,
    )
else:
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=DTYPE, device_map="auto", low_cpu_mem_usage=True,
    )
llm.eval()
remove_accelerate_hooks(llm)          # 关键：去掉走损坏 P2P 的 dispatch hooks
for p in llm.parameters():
    p.requires_grad_(False)
print(f"[serve] model loaded, <|im_end|> id={IM_END}", flush=True)

GEN_LOCK = Lock()


# ── KV-cache 手动分片生成器 ──────────────────────────────────────────
class ShardedGenerator:
    """逐层 host-staged 前向 + 每层本地 KV cache。仅支持 batch=1。"""

    def __init__(self, llm):
        from transformers.cache_utils import DynamicCache
        self._DynamicCache = DynamicCache
        self.llm = llm
        self.m = llm.model
        self.inv_freq = self.m.rotary_emb.inv_freq.float().cpu()
        self.devices = [next(l.parameters()).device for l in self.m.layers]
        self.embed_dev = self.m.embed_tokens.weight.device
        self.norm_dev = self.m.norm.weight.device
        self.head_dev = self.llm.lm_head.weight.device

    def _rope(self, start: int, end: int, dtype: torch.dtype):
        pos = torch.arange(start, end, device="cpu").unsqueeze(0)   # [1, L]
        freqs = pos[:, :, None].float() * self.inv_freq             # [1, L, H/2]
        emb = torch.cat([freqs, freqs], dim=-1)                     # [1, L, H]
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def _layers(self, h: torch.Tensor, caches, start: int, length: int) -> torch.Tensor:
        cos, sin = self._rope(start, start + length, h.dtype)
        for i, layer in enumerate(self.m.layers):
            d = self.devices[i]
            h = _stage(h, d)
            pos = torch.arange(start, start + length, device=d).unsqueeze(0)
            out = layer(
                h,
                attention_mask=None,          # batch=1 纯因果 + 单步解码均无需 mask
                position_ids=pos,
                past_key_values=caches[i],
                use_cache=True,
                cache_position=torch.arange(start, start + length, device=d),
                position_embeddings=(_stage(cos, d), _stage(sin, d)),
            )
            h = out[0] if isinstance(out, tuple) else out
        return h

    def _head(self, h_last: torch.Tensor) -> torch.Tensor:
        h = _stage(h_last, self.norm_dev)
        h = self.m.norm(h)
        h = _stage(h, self.head_dev)
        return self.llm.lm_head(h)[0, -1]     # [vocab]

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int,
                 temperature: float, top_p: float) -> Iterator[str]:
        """生成器：逐段 yield 解码文本，遇 <|im_end|> 停止。"""
        S = input_ids.shape[1]
        caches = [self._DynamicCache() for _ in range(len(self.m.layers))]

        h = self.m.embed_tokens(input_ids)                       # prefill
        h = self._layers(h, caches, 0, S)
        logits = self._head(h)

        ids: list[int] = []
        prev_text = ""
        for cur in range(S, S + max_new_tokens):
            if temperature and temperature > 0:
                lg = logits.float()
                probs = torch.softmax(lg / temperature, dim=-1)
                if top_p and top_p < 1.0:
                    sp, si = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sp, dim=-1)
                    sp[cum - sp > top_p] = 0.0
                    sp /= sp.sum()
                    nxt = int(si[torch.multinomial(sp, 1)])
                else:
                    nxt = int(torch.multinomial(probs, 1))
            else:
                nxt = int(logits.argmax().item())
            if nxt == IM_END:
                return
            ids.append(nxt)
            text = tok.decode(ids, skip_special_tokens=True)
            if len(text) > len(prev_text):
                yield text[len(prev_text):]
                prev_text = text

            e = self.m.embed_tokens(torch.tensor([[nxt]], device=self.embed_dev))
            e = self._layers(e, caches, cur, 1)                   # decode 1 token
            logits = self._head(e)

    @property
    def n_generated(self) -> int:
        return 0


GEN = ShardedGenerator(llm)


def build_prompt(messages: list[dict]) -> str:
    """ChatML，与训练格式一致（无 think 块）。"""
    parts = []
    for i, m in enumerate(messages):
        role = m.get("role", "user")
        content = m.get("content", "") if isinstance(m.get("content"), str) else \
            "".join(p.get("text", "") for p in m["content"] if p.get("type") == "text")
        if role == "system":
            if i == 0:
                parts.append(f"<|im_start|>system\n{content}<|im_end|>\n")
            continue
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if not any(m.get("role") == "system" for m in messages):
        parts.insert(0, f"<|im_start|>system\n{DEFAULT_SYS}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def tokenize(prompt: str) -> torch.Tensor:
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    if ids.shape[1] > MAX_CTX - 512:
        raise HTTPException(400, f"prompt {ids.shape[1]} tokens exceeds context {MAX_CTX}")
    return ids.to(GEN.embed_dev)


# ── API ──────────────────────────────────────────────────────────────
class ChatReq(BaseModel):
    messages: list[dict]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    model: str | None = None


class CompReq(BaseModel):
    prompt: str
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False


app = FastAPI(title="Nanbeige4.3-33B OpenAI-compatible server")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def models():
    return {"object": "list",
            "data": [{"id": "nanbeige-4.3-33b", "object": "model", "owned_by": "local"}]}


def _produce(prompt: str, body, q) -> None:
    """在独立线程中生成（与请求生命周期解耦）：客户端断开不会中断本线程，
    GEN_LOCK 的 finally 必然执行，杜绝锁泄漏。结果经 queue 送达。"""
    try:
        ids = tokenize(prompt)
        temp = body.temperature if body.temperature is not None else 0.7
        top_p = body.top_p if body.top_p is not None else 1.0
        max_new = min(body.max_tokens or 512, 4096)
        t0 = time.time()
        print(f"[gen] start prompt_tokens={ids.shape[1]} max_new={max_new}", flush=True)
        with GEN_LOCK:
            print(f"[gen] lock acquired (+{time.time()-t0:.1f}s)", flush=True)
            try:
                for piece in GEN.generate(ids, max_new, temp, top_p):
                    q.put(piece)
            finally:
                print(f"[gen] done/released (+{time.time()-t0:.1f}s)", flush=True)
    except Exception as e:                      # noqa: BLE001
        q.put(e)
    finally:
        q.put(None)                             # sentinel，永远会送达


def _start_produce(prompt: str, body):
    q: queue.Queue = queue.Queue()
    Thread(target=_produce, args=(prompt, body, q), daemon=True).start()
    return q


@app.post("/v1/chat/completions")
def chat(req: ChatReq):
    if not req.messages:
        raise HTTPException(400, "messages is empty")
    prompt = build_prompt(req.messages)
    rid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())

    if not req.stream:
        q = _start_produce(prompt, req)
        parts = []
        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise HTTPException(500, str(item))
            parts.append(item)
        text = "".join(parts)
        return {
            "id": rid, "object": "chat.completion", "created": created, "model": "nanbeige-4.3-33b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(tok(prompt).input_ids),
                      "completion_tokens": 0, "total_tokens": 0},
        }

    def sse():
        import json
        q = _start_produce(prompt, req)
        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                break
            d = {"id": rid, "object": "chat.completion.chunk", "created": created,
                 "model": "nanbeige-4.3-33b",
                 "choices": [{"index": 0, "delta": {"content": item}, "finish_reason": None}]}
            yield f"data: {json.dumps(d, ensure_ascii=False)}\n\n"
        d = {"id": rid, "object": "chat.completion.chunk", "created": created,
             "model": "nanbeige-4.3-33b",
             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(d, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.post("/v1/completions")
def completions(req: CompReq):
    q = _start_produce(req.prompt, req)
    parts = []
    while True:
        item = q.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise HTTPException(500, str(item))
        parts.append(item)
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:12]}", "object": "text_completion",
        "created": int(time.time()), "model": "nanbeige-4.3-33b",
        "choices": [{"index": 0, "text": "".join(parts), "finish_reason": "stop"}],
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
