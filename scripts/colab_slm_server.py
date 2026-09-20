"""
Colab Pro SLM Serving Script
============================
此腳本專為 Google Colab Pro (T4/L4/A100 GPU) 與 VS Code Colab Extension 設計。
啟動基於 vLLM 或 FastAPI + Transformers 的 OpenAI 相容 API 伺服器，
並透過 Cloudflare Tunnel 自動穿透，產生公開 HTTPS 端點供本地 Investment Assistant 串接。

執行方式 (在 Colab 終端機或 Notebook):
    !pip install -q transformers accelerate bitsandbytes fastapi uvicorn pydantic pycloudflared
    python colab_slm_server.py --model Qwen/Qwen2.5-1.5B-Instruct --port 8000
"""

import os
import sys
import time
import argparse
import subprocess
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, HTTPException
import torch

app = FastAPI(title="Colab Pro SLM Serving Endpoint")

model_pipeline = None
loaded_model_id = ""


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 256


@app.get("/health")
def health():
    return {"status": "ok", "device": "cuda" if torch.cuda.is_available() else "cpu", "model": loaded_model_id}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": loaded_model_id, "object": "model", "owned_by": "colab-pro"}
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    global model_pipeline
    if not model_pipeline:
        raise HTTPException(status_code=503, detail="Model is still loading...")

    # 將訊息轉換為 HuggingFace chat 格式
    conversation = [{"role": m.role, "content": m.content} for m in req.messages]

    try:
        outputs = model_pipeline(
            conversation,
            max_new_tokens=req.max_tokens or 256,
            temperature=req.temperature or 0.1,
            do_sample=req.temperature > 0.0,
            pad_token_id=model_pipeline.tokenizer.eos_token_id,
        )
        response_text = outputs[0]["generated_text"][-1]["content"]

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": loaded_model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text.strip(),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


def start_tunnel(port: int):
    """使用 pycloudflared 在 Colab 自動開啟公開安全 HTTPS 穿透端點"""
    try:
        from pycloudflared import try_cloudflare
        tunnel = try_cloudflare(port=port)
        tunnel_url = tunnel.tunnel.replace("http://", "https://")
        print("\n" + "=" * 65)
        print("🎉 [Colab Pro] Cloudflare Tunnel 成功啟動！")
        print(f"🔗 公開端點 URL: {tunnel_url}")
        print("=" * 65)
        print("請複製以下內容至您本地的 .env 檔案中：")
        print(f"SLM_ENABLED=true")
        print(f"SLM_ENDPOINT_URL={tunnel_url}/v1")
        print(f"SLM_MODEL={loaded_model_id}")
        print("=" * 65 + "\n")
        return tunnel
    except Exception as e:
        print(f"[Warning] Failed to start Cloudflare tunnel: {e}")
        print("若在純本地運行，可直接使用 http://localhost:8000/v1")
        return None


def main():
    global model_pipeline, loaded_model_id
    parser = argparse.ArgumentParser(description="Serve SLM on Colab Pro")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="HuggingFace model ID")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--load_in_4bit", action="store_true", default=False, help="Enable 4-bit quantization")
    args = parser.parse_args()

    loaded_model_id = args.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Colab Pro] Loading model {args.model} onto {device}...")

    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model_kwargs = {"torch_dtype": torch.float16 if device == "cuda" else torch.float32}
    if args.load_in_4bit and device == "cuda":
        model_kwargs["load_in_4bit"] = True

    model_pipeline = pipeline(
        "text-generation",
        model=args.model,
        tokenizer=tokenizer,
        device_map="auto" if device == "cuda" else None,
        model_kwargs=model_kwargs,
    )
    print(f"[Colab Pro] Model {args.model} loaded successfully!")

    # 啟動穿透隧道
    tunnel = start_tunnel(args.port)

    # 啟動 FastAPI 服務
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
