#!/usr/bin/env python3
"""
moondream_server.py
FastAPI 기반 MoonDream2 단일 추론 서버.
- 포트 1개당 1개 서버. 모델 인스턴스도 1개.
- N개 띄우면 N배 throughput (Jetson Thor 128GB VRAM 활용).
- 실행: python3 -m nero_ai.moondream_server --port 8000
- 또는 scripts/run_cluster.sh 로 N개 일괄 기동.
"""

import argparse
import io
import os
from typing import List

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from transformers import AutoModelForCausalLM


MODEL_REVISION = os.environ.get("MODEL_REVISION", "2025-06-21")


# ──────────────────────────────────────────────
# 요청/응답 스키마
# ──────────────────────────────────────────────
class DetectRequest(BaseModel):
    image_b64: str          # 베이스64 인코딩된 JPEG/PNG
    labels: List[str]       # 한 번에 여러 라벨 검출


class Detection(BaseModel):
    label: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class DetectResponse(BaseModel):
    detections: List[Detection]
    inference_ms: float


# ──────────────────────────────────────────────
# 서버 부팅
# ──────────────────────────────────────────────
def build_app(port: int) -> FastAPI:
    print(f"[server :{port}] MoonDream2 로딩 중...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print(f"[server :{port}] ⚠️ CPU 모드 (매우 느림)")

    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision=MODEL_REVISION,
        trust_remote_code=True,
        device_map={"": device},
    ).eval()
    if device == "cuda":
        model = model.to(torch.float16)
    print(f"[server :{port}] 모델 로드 완료")

    app = FastAPI(title=f"moondream-{port}")

    @app.get("/health")
    def health():
        return {"status": "ok", "port": port, "device": device}

    @app.post("/detect", response_model=DetectResponse)
    def detect(req: DetectRequest):
        import base64
        import time
        try:
            img_bytes = base64.b64decode(req.image_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(400, f"image decode failed: {e}")

        t0 = time.time()
        out: List[Detection] = []
        try:
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                embeds = model.encode_image(img)
                for label in req.labels:
                    try:
                        res = model.detect(embeds, label.strip())
                        for det in res.get("objects", []):
                            out.append(Detection(
                                label=label.strip(),
                                x_min=float(det.get("x_min", 0)),
                                y_min=float(det.get("y_min", 0)),
                                x_max=float(det.get("x_max", 0)),
                                y_max=float(det.get("y_max", 0)),
                            ))
                    except Exception as e:
                        print(f"[server :{port}] {label}: {e}")
        except Exception as e:
            raise HTTPException(500, f"inference failed: {e}")

        return DetectResponse(
            detections=out,
            inference_ms=(time.time() - t0) * 1000.0,
        )

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    app = build_app(args.port)
    # 단일 워커 (모델 1개만 메모리에). 멀티 프로세스는 외부에서 포트별로 띄움.
    uvicorn.run(app, host=args.host, port=args.port,
                log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
