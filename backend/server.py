#!/usr/bin/env python3
"""
ai-ojt バックエンドサーバー（HTML配信機能付き）
================================================
スマホからこのサーバーのURLにアクセスするだけで
フロントエンド（カメラUI）もAPIも全て使える。

起動:
    export ANTHROPIC_API_KEY="sk-ant-xxx"
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

スマホからアクセス:
    https://<PCのIP or ドメイン>:8000/
"""

import base64
import time
from pathlib import Path
from typing import Optional

import anthropic
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


# ──────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────
DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_IMAGE_SIZE = (1568, 1568)
BLUR_THRESHOLD = 50.0
SIMILARITY_THRESHOLD = 0.95

# HTMLファイルのパス（server.py と同階層の ../index.html を参照）
HTML_PATH = Path(__file__).parent.parent / "index.html"


# ──────────────────────────────────────────────
# FastAPI セットアップ
# ──────────────────────────────────────────────
app = FastAPI(title="ai-ojt API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic()


# ──────────────────────────────────────────────
# 画像処理
# ──────────────────────────────────────────────
def calc_sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def calc_similarity(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    gray_a = cv2.cvtColor(cv2.resize(frame_a, (320, 240)), cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(cv2.resize(frame_b, (320, 240)), cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(gray_a, gray_b, cv2.TM_CCOEFF_NORMED)
    return float(result[0][0])


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    max_w, max_h = MAX_IMAGE_SIZE
    if w > max_w or h > max_h:
        scale = min(max_w / w, max_h / h)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    frame = cv2.filter2D(frame, -1, kernel)
    return frame


def frame_to_base64(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", frame)
    return base64.standard_b64encode(buf).decode("utf-8")


def decode_base64_image(data: str) -> np.ndarray:
    if "," in data:
        data = data.split(",", 1)[1]
    img_bytes = base64.b64decode(data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("画像のデコードに失敗")
    return frame


# ──────────────────────────────────────────────
# Claude API 呼び出し
# ──────────────────────────────────────────────
def ocr_single_frame(frame_b64: str, model: str = DEFAULT_MODEL) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": frame_b64}},
                {"type": "text", "text": (
                    "画像のテキストをすべて正確に読み取れ。"
                    "コードはコードブロックで囲め。テキストのみ出力。説明不要。"
                )},
            ],
        }],
    )
    return response.content[0].text


def suggest_next_steps(ocr_text: str, analysis: Optional[str], model: str = DEFAULT_MODEL) -> str:
    context = f"画面内容:\n{ocr_text}\n"
    if analysis:
        context += f"\n分析:\n{analysis}\n"

    response = client.messages.create(
        model=model,
        max_tokens=800,
        system=(
            "あなたはOJTメンターだ。現場で手を止めずに読める短い回答を返せ。"
            "以下のフォーマットで回答しろ:\n\n"
            "【これは何か】1行で画面の状況を要約\n\n"
            "【次にやること】最大3個、各1行で\n"
            "1. アクション → 具体的な手順\n\n"
            "【注意・危険】あれば1〜2行。なければ省略\n\n"
            "【参考リンク】関連する公式ドキュメントやStack OverflowのURLがあれば記載。なければ省略\n\n"
            "長文禁止。箇条書き中心。合計200文字以内を目指せ。"
        ),
        messages=[{"role": "user", "content": context}],
    )
    return response.content[0].text


def analyze_with_context(ocr_text: str, prompt: str, history: list[dict], model: str = DEFAULT_MODEL) -> str:
    context = ""
    if history:
        context = "\n直近の画面:\n"
        for h in history[-3:]:
            context += f"{h['text'][:200]}\n---\n"

    response = client.messages.create(
        model=model,
        max_tokens=1000,
        system=(
            "あなたはOJTメンターだ。画面に映っている内容を特定し、質問に対して簡潔に答えろ。\n"
            "以下のフォーマットで回答しろ:\n\n"
            "【これは何か】1行で画面の状況を要約（何のツール/画面/作業をしているか）\n\n"
            "【回答】質問への回答を3〜5行以内で。結論ファースト。コード例は最小限（3行以内）\n\n"
            "【次にやること】最大3個、各1行で次のアクションを提示\n\n"
            "【注意・危険】あれば1〜2行。なければ省略\n\n"
            "【参考リンク】公式ドキュメントのURLがあれば記載。なければ省略\n\n"
            "ルール:\n"
            "- 「〜と思います」等の冗長表現禁止\n"
            "- 合計300文字以内を目指せ"
        ),
        messages=[{
            "role": "user",
            "content": f"画面:\n{ocr_text}\n{context}\n質問: {prompt}",
        }],
    )
    return response.content[0].text


def answer_followup(ocr_text: str, previous_analysis: Optional[str], question: str, model: str = DEFAULT_MODEL) -> str:
    context = f"画面: {ocr_text[:500]}\n"
    if previous_analysis:
        context += f"前回の回答: {previous_analysis[:300]}\n"

    response = client.messages.create(
        model=model,
        max_tokens=800,
        system=(
            "OJTメンターとして追加質問に簡潔に答えろ。"
            "3〜5行以内。結論ファースト。コード例は最小限。"
            "関連URLがあれば末尾に。"
        ),
        messages=[{
            "role": "user",
            "content": f"{context}\n追加質問: {question}",
        }],
    )
    return response.content[0].text


# ──────────────────────────────────────────────
# スキーマ
# ──────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    image_base64: str
    prompt: Optional[str] = None


class AnalyzeResponse(BaseModel):
    ocr_text: str
    analysis: Optional[str] = None
    next_steps: Optional[str] = None
    elapsed_ms: int


class FollowupRequest(BaseModel):
    ocr_text: str
    previous_analysis: Optional[str] = None
    question: str


class FollowupResponse(BaseModel):
    answer: str
    elapsed_ms: int


# ──────────────────────────────────────────────
# セッション状態（WebSocket用）
# ──────────────────────────────────────────────
class SessionState:
    def __init__(self):
        self.prev_frame: Optional[np.ndarray] = None
        self.history: list[dict] = []

    def should_process(self, frame: np.ndarray) -> tuple[bool, str]:
        sharpness = calc_sharpness(frame)
        if sharpness < BLUR_THRESHOLD:
            return False, f"ブレ検出 (シャープネス: {sharpness:.1f})"
        if self.prev_frame is not None:
            sim = calc_similarity(frame, self.prev_frame)
            if sim > SIMILARITY_THRESHOLD:
                return False, f"前フレームと類似 ({sim:.2f})"
        return True, "OK"


sessions: dict[str, SessionState] = {}


# ──────────────────────────────────────────────
# エンドポイント
# ──────────────────────────────────────────────

# ★ ルートにアクセスしたらフロントエンドHTMLを返す
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    if HTML_PATH.exists():
        return HTMLResponse(content=HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>index.html が見つかりません</h1><p>backend/ と同階層に index.html を配置してください</p>", status_code=404)


@app.get("/health")
def health():
    return {"status": "ok", "time": time.time()}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    start = time.time()

    try:
        frame = decode_base64_image(req.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"画像デコード失敗: {e}")

    processed = preprocess_frame(frame)
    processed_b64 = frame_to_base64(processed)

    ocr_text = ocr_single_frame(processed_b64)

    analysis = None
    if req.prompt:
        analysis = analyze_with_context(ocr_text, req.prompt, history=[])

    try:
        next_steps = suggest_next_steps(ocr_text, analysis)
    except Exception as e:
        next_steps = f"次のアクション提案の生成に失敗: {e}"

    return AnalyzeResponse(
        ocr_text=ocr_text,
        analysis=analysis,
        next_steps=next_steps,
        elapsed_ms=int((time.time() - start) * 1000),
    )


@app.post("/analyze-upload", response_model=AnalyzeResponse)
async def analyze_upload(file: UploadFile = File(...), prompt: Optional[str] = Form(None)):
    start = time.time()
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="画像デコード失敗")

    processed = preprocess_frame(frame)
    processed_b64 = frame_to_base64(processed)
    ocr_text = ocr_single_frame(processed_b64)

    analysis = None
    if prompt:
        analysis = analyze_with_context(ocr_text, prompt, history=[])

    try:
        next_steps = suggest_next_steps(ocr_text, analysis)
    except Exception as e:
        next_steps = f"次のアクション提案の生成に失敗: {e}"

    return AnalyzeResponse(
        ocr_text=ocr_text,
        analysis=analysis,
        next_steps=next_steps,
        elapsed_ms=int((time.time() - start) * 1000),
    )


@app.post("/followup", response_model=FollowupResponse)
async def followup(req: FollowupRequest):
    start = time.time()
    try:
        answer = answer_followup(req.ocr_text, req.previous_analysis, req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"回答失敗: {e}")

    return FollowupResponse(answer=answer, elapsed_ms=int((time.time() - start) * 1000))


@app.websocket("/ws/stream/{session_id}")
async def websocket_stream(websocket: WebSocket, session_id: str):
    await websocket.accept()
    state = sessions.setdefault(session_id, SessionState())

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

            state.prev_frame = frame
            state.history.append({"timestamp": time.time(), "text": ocr_text})
            if len(state.history) > 20:
                state.history.pop(0)

            await websocket.send_json({
                "type": "result",
                "ocr_text": ocr_text,
                "analysis": analysis,
                "elapsed_ms": int((time.time() - start) * 1000),
            })

    except WebSocketDisconnect:
        sessions.pop(session_id, None)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
