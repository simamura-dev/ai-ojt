#!/usr/bin/env python3
"""
FastAPI バックエンドサーバー
===========================
スマホアプリから送られてきたフレーム画像を受け取り、
Claude Vision API で解析して結果を返す。

起動:
    export ANTHROPIC_API_KEY="sk-ant-xxx"
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

スマホからは同じWi-Fi内のPCのIPアドレスでアクセス:
    http://192.168.x.x:8000
"""

import base64
import io
import sys
import time
from pathlib import Path
from typing import Optional

import anthropic
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 既存のOCR処理を流用
sys.path.insert(0, str(Path(__file__).parent.parent))
from video_ocr_claude import (
    calc_sharpness,
    calc_similarity,
    preprocess_frame,
    frame_to_base64,
    BLUR_THRESHOLD,
    SIMILARITY_THRESHOLD,
    DEFAULT_MODEL,
)

# ──────────────────────────────────────────────
# FastAPI セットアップ
# ──────────────────────────────────────────────
app = FastAPI(title="Video OCR + Claude API")

# CORS（スマホアプリからのアクセスを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # ローカル利用なので全許可
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic()

# ──────────────────────────────────────────────
# セッション状態（簡易版 - メモリ上のみ）
# ──────────────────────────────────────────────
class SessionState:
    """ストリーミング中の状態を保持するクラス"""
    def __init__(self):
        self.prev_frame: Optional[np.ndarray] = None
        self.history: list[dict] = []        # OCR結果の履歴
        self.last_analysis_time: float = 0

    def should_process(self, frame: np.ndarray) -> tuple[bool, str]:
        """このフレームを処理すべきか判定する"""
        sharpness = calc_sharpness(frame)
        if sharpness < BLUR_THRESHOLD:
            return False, f"ブレ検出 (シャープネス: {sharpness:.1f})"

        if self.prev_frame is not None:
            sim = calc_similarity(frame, self.prev_frame)
            if sim > SIMILARITY_THRESHOLD:
                return False, f"前フレームと類似 ({sim:.2f})"

        return True, "OK"


# WebSocket 接続ごとの状態
sessions: dict[str, SessionState] = {}


# ──────────────────────────────────────────────
# スキーマ
# ──────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    image_base64: str
    prompt: Optional[str] = None


class AnalyzeResponse(BaseModel):
    ocr_text: str
    analysis: Optional[str] = None
    next_steps: Optional[str] = None      # ← Claudeが提案する「次にすべきこと」
    elapsed_ms: int
    skipped: bool = False
    reason: Optional[str] = None


class FollowupRequest(BaseModel):
    """解析結果に対する追加質問"""
    ocr_text: str
    previous_analysis: Optional[str] = None
    question: str


class FollowupResponse(BaseModel):
    answer: str
    elapsed_ms: int


# ──────────────────────────────────────────────
# Claude API 呼び出し
# ──────────────────────────────────────────────
def ocr_single_frame(frame_b64: str, model: str = DEFAULT_MODEL) -> str:
    """単一フレームからテキストを抽出"""
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": frame_b64,
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "この画像に写っている画面のテキストをすべて正確に読み取ってください。"
                        "UIの構造（見出し、ボタン、本文）を保ちつつ、コードがあればコードブロックで囲んでください。"
                        "テキストのみを出力し、説明は不要です。"
                    ),
                },
            ],
        }],
    )
    return response.content[0].text


def suggest_next_steps(
    ocr_text: str,
    analysis: Optional[str],
    model: str = DEFAULT_MODEL,
) -> str:
    """
    解析結果をもとに「次にすべきこと」をClaude に提案させる。
    具体的で実行可能な行動（アクション）を返す。
    """
    context = f"## 画面に表示されている内容\n{ocr_text}\n"
    if analysis:
        context += f"\n## これまでの分析\n{analysis}\n"

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=(
            "あなたはユーザーの開発作業を伴走するメンターです。"
            "画面の状況と分析結果を踏まえて、ユーザーが今すぐ取るべき具体的なアクションを提案してください。"
        ),
        messages=[{
            "role": "user",
            "content": (
                f"{context}\n"
                "## 依頼\n"
                "この状況を踏まえて、ユーザーが次にすべき具体的なアクションを3〜5個、"
                "優先度の高い順にリストアップしてください。\n\n"
                "各アクションは以下の形式で：\n"
                "1. **[アクション名]** - なぜこれをするのか（1文）\n"
                "   → 具体的な手順（1〜2文）\n\n"
                "短く、実行可能で、今すぐ着手できる内容にしてください。"
            ),
        }],
    )
    return response.content[0].text


def answer_followup(
    ocr_text: str,
    previous_analysis: Optional[str],
    question: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """ユーザーからの追加質問に回答する"""
    context = f"## 画面の内容\n{ocr_text}\n"
    if previous_analysis:
        context += f"\n## 直前の分析\n{previous_analysis}\n"

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=(
            "あなたはユーザーの作業を伴走するエキスパートです。"
            "画面の内容とこれまでの流れを踏まえて、追加の質問に具体的に回答してください。"
        ),
        messages=[{
            "role": "user",
            "content": f"{context}\n## 追加の質問\n{question}",
        }],
    )
    return response.content[0].text


def analyze_with_context(
    ocr_text: str,
    prompt: str,
    history: list[dict],
    model: str = DEFAULT_MODEL,
) -> str:
    """OCR結果 + 履歴 + プロンプトで Claude に分析依頼"""
    context = ""
    if history:
        context = "\n## これまでの画面履歴（直近5件）\n"
        for h in history[-5:]:
            context += f"\n[{h['timestamp']:.1f}]\n{h['text'][:300]}\n"

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=(
            "あなたはユーザーのPC画面を見ながら開発をサポートするエキスパートです。"
            "画面に表示されている内容を踏まえて、具体的で実用的なアドバイスをしてください。"
        ),
        messages=[{
            "role": "user",
            "content": (
                f"## 現在の画面\n{ocr_text}\n"
                f"{context}\n"
                f"## 質問・指示\n{prompt}"
            ),
        }],
    )
    return response.content[0].text


# ──────────────────────────────────────────────
# デコード
# ──────────────────────────────────────────────
def decode_base64_image(data: str) -> np.ndarray:
    """base64文字列をOpenCV画像に変換"""
    # data:image/png;base64,xxx 形式にも対応
    if "," in data:
        data = data.split(",", 1)[1]

    img_bytes = base64.b64decode(data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("画像のデコードに失敗しました")
    return frame


# ──────────────────────────────────────────────
# エンドポイント
# ──────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Video OCR + Claude API",
        "status": "running",
        "endpoints": {
            "POST /analyze": "単発画像を解析",
            "POST /analyze-upload": "画像ファイルアップロード解析",
            "WS /ws/stream/{session_id}": "リアルタイムストリーム解析",
            "GET /health": "ヘルスチェック",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "time": time.time()}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """
    base64エンコードされた画像を受け取って解析する。
    スマホアプリからはこのエンドポイントを使う。
    """
    start = time.time()

    try:
        frame = decode_base64_image(req.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"画像デコード失敗: {e}")

    # 前処理
    processed = preprocess_frame(frame)
    processed_b64 = frame_to_base64(processed)

    # OCR
    ocr_text = ocr_single_frame(processed_b64)

    # プロンプト指定があれば分析
    analysis = None
    if req.prompt:
        analysis = analyze_with_context(ocr_text, req.prompt, history=[])

    # 必ず「次にすべきこと」も生成する
    try:
        next_steps = suggest_next_steps(ocr_text, analysis)
    except Exception as e:
        next_steps = f"次のアクション提案の生成に失敗: {e}"

    elapsed_ms = int((time.time() - start) * 1000)
    return AnalyzeResponse(
        ocr_text=ocr_text,
        analysis=analysis,
        next_steps=next_steps,
        elapsed_ms=elapsed_ms,
    )


@app.post("/followup", response_model=FollowupResponse)
async def followup(req: FollowupRequest):
    """
    直前の解析結果に対する追加質問に答える。
    「このエラーの対処は？」「このコードをどう修正すれば？」などの自然な会話に対応。
    """
    start = time.time()
    try:
        answer = answer_followup(
            ocr_text=req.ocr_text,
            previous_analysis=req.previous_analysis,
            question=req.question,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"追加質問への回答失敗: {e}")

    return FollowupResponse(
        answer=answer,
        elapsed_ms=int((time.time() - start) * 1000),
    )


@app.post("/analyze-upload", response_model=AnalyzeResponse)
async def analyze_upload(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
):
    """画像ファイルを直接アップロードして解析する"""
    start = time.time()

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="画像のデコードに失敗しました")

    processed = preprocess_frame(frame)
    processed_b64 = frame_to_base64(processed)
    ocr_text = ocr_single_frame(processed_b64)

    analysis = None
    if prompt:
        analysis = analyze_with_context(ocr_text, prompt, history=[])

    return AnalyzeResponse(
        ocr_text=ocr_text,
        analysis=analysis,
        elapsed_ms=int((time.time() - start) * 1000),
    )


@app.websocket("/ws/stream/{session_id}")
async def websocket_stream(websocket: WebSocket, session_id: str):
    """
    WebSocketでリアルタイムストリーミング解析。

    クライアントから:
        {"type": "frame", "image": "<base64>", "prompt": "..."}
    サーバーから:
        {"type": "result", "ocr_text": "...", "analysis": "...", "elapsed_ms": 123}
        {"type": "skip", "reason": "ブレ検出"}
        {"type": "error", "message": "..."}
    """
    await websocket.accept()
    state = sessions.setdefault(session_id, SessionState())

    print(f"🔌 WebSocket接続: session={session_id}")

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") != "frame":
                continue

            try:
                frame = decode_base64_image(data["image"])
            except Exception as e:
                await websocket.send_json({"type": "error", "message": str(e)})
                continue

            # 重複・ブレチェック（API呼び出しを減らす重要な仕組み）
            should_process, reason = state.should_process(frame)
            if not should_process:
                await websocket.send_json({"type": "skip", "reason": reason})
                continue

            start = time.time()
            processed = preprocess_frame(frame)
            processed_b64 = frame_to_base64(processed)

            try:
                ocr_text = ocr_single_frame(processed_b64)
            except Exception as e:
                await websocket.send_json({"type": "error", "message": f"OCR失敗: {e}"})
                continue

            analysis = None
            prompt = data.get("prompt")
            if prompt:
                try:
                    analysis = analyze_with_context(ocr_text, prompt, state.history)
                except Exception as e:
                    analysis = f"分析失敗: {e}"

            # 状態更新
            state.prev_frame = frame
            state.history.append({
                "timestamp": time.time(),
                "text": ocr_text,
            })
            if len(state.history) > 20:
                state.history.pop(0)

            elapsed_ms = int((time.time() - start) * 1000)

            await websocket.send_json({
                "type": "result",
                "ocr_text": ocr_text,
                "analysis": analysis,
                "elapsed_ms": elapsed_ms,
            })

    except WebSocketDisconnect:
        print(f"🔌 WebSocket切断: session={session_id}")
        sessions.pop(session_id, None)
    except Exception as e:
        print(f"❌ WebSocketエラー: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
