#!/usr/bin/env python3
"""
ai-ojt バックエンドサーバー（マルチAI対応）
================================================
Claude / ChatGPT / Gemini を切り替えて使える。

起動:
    export ANTHROPIC_API_KEY="sk-ant-xxx"
    export OPENAI_API_KEY="sk-xxx"          # ChatGPT用（任意）
    export GEMINI_API_KEY="AIzaSyXxx"       # Gemini用（任意）
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import base64
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel


# ──────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────
MAX_IMAGE_SIZE = (1568, 1568)
BLUR_THRESHOLD = 50.0
SIMILARITY_THRESHOLD = 0.95

HTML_PATH = Path(__file__).parent.parent / "index.html"

# AIプロバイダー設定
AI_PROVIDERS = {
    "claude": {
        "normal": "claude-sonnet-4-6",
        "fast": "claude-haiku-4-5-20251001",
        "label": "Claude (Anthropic)",
    },
    "chatgpt": {
        "normal": "gpt-4o",
        "fast": "gpt-4o-mini",
        "label": "ChatGPT (OpenAI)",
    },
    "gemini": {
        "normal": "gemini-2.5-flash",
        "fast": "gemini-2.0-flash-lite",
        "label": "Gemini (Google)",
    },
}

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

# ──────────────────────────────────────────────
# AIクライアント初期化（遅延ロード）
# ──────────────────────────────────────────────
_clients = {}


def get_claude_client():
    if "claude" not in _clients:
        import anthropic
        _clients["claude"] = anthropic.Anthropic()
    return _clients["claude"]


def get_openai_client():
    if "openai" not in _clients:
        from openai import OpenAI
        _clients["openai"] = OpenAI()
    return _clients["openai"]


def get_gemini_client():
    if "gemini" not in _clients:
        from google import genai
        _clients["gemini"] = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
    return _clients["gemini"]


# ──────────────────────────────────────────────
# 統一AIインターフェース
# ──────────────────────────────────────────────
def ai_vision(provider: str, model: str, image_b64: str, text_prompt: str, max_tokens: int = 1024) -> str:
    """画像+テキストプロンプトを送り、テキスト応答を返す統一関数"""
    if provider == "claude":
        client = get_claude_client()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
                    {"type": "text", "text": text_prompt},
                ],
            }],
        )
        return response.content[0].text

    elif provider == "chatgpt":
        client = get_openai_client()
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}", "detail": "high"}},
                    {"type": "text", "text": text_prompt},
                ],
            }],
        )
        return response.choices[0].message.content

    elif provider == "gemini":
        client = get_gemini_client()
        image_bytes = base64.b64decode(image_b64)
        response = client.models.generate_content(
            model=model,
            contents=[
                {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(image_bytes).decode()}},
                text_prompt,
            ],
        )
        return response.text

    raise ValueError(f"未対応プロバイダー: {provider}")


def ai_text(provider: str, model: str, system_prompt: str, user_message: str, max_tokens: int = 800) -> str:
    """テキストのみのプロンプトを送る統一関数"""
    if provider == "claude":
        client = get_claude_client()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text

    elif provider == "chatgpt":
        client = get_openai_client()
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content

    elif provider == "gemini":
        client = get_gemini_client()
        full_prompt = f"{system_prompt}\n\n---\n\n{user_message}"
        response = client.models.generate_content(
            model=model,
            contents=[full_prompt],
        )
        return response.text

    raise ValueError(f"未対応プロバイダー: {provider}")


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
# AI解析関数（プロバイダー対応版）
# ──────────────────────────────────────────────
def ocr_single_frame(frame_b64: str, provider: str = "claude", model: str = None) -> str:
    if model is None:
        model = AI_PROVIDERS[provider]["normal"]
    return ai_vision(
        provider, model, frame_b64,
        "画像のテキストをすべて正確に読み取れ。コードはコードブロックで囲め。テキストのみ出力。説明不要。",
        max_tokens=1024,
    )


def suggest_next_steps(ocr_text: str, analysis: Optional[str], provider: str = "claude", model: str = None) -> str:
    if model is None:
        model = AI_PROVIDERS[provider]["normal"]
    context = f"画面内容:\n{ocr_text}\n"
    if analysis:
        context += f"\n分析:\n{analysis}\n"
    return ai_text(
        provider, model,
        system_prompt=(
            "あなたはOJTメンターだ。現場で手を止めずに読める短い回答を返せ。\n"
            "以下のフォーマットで回答しろ:\n\n"
            "【これは何か】1行で画面の状況を要約\n\n"
            "【次にやること】最大3個、各1行で\n"
            "1. アクション → 具体的な手順\n\n"
            "【注意・危険】あれば1〜2行。なければ省略\n\n"
            "【参考リンク】関連する公式ドキュメントやStack OverflowのURLがあれば記載。なければ省略\n\n"
            "長文禁止。箇条書き中心。合計200文字以内を目指せ。"
        ),
        user_message=context,
    )


def analyze_with_context(ocr_text: str, prompt: str, history: list[dict], provider: str = "claude", model: str = None) -> str:
    if model is None:
        model = AI_PROVIDERS[provider]["normal"]
    context = ""
    if history:
        context = "\n直近の画面:\n"
        for h in history[-3:]:
            context += f"{h['text'][:200]}\n---\n"
    return ai_text(
        provider, model,
        system_prompt=(
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
        user_message=f"画面:\n{ocr_text}\n{context}\n質問: {prompt}",
    )


def analyze_image_fast(frame_b64: str, prompt: Optional[str] = None, provider: str = "claude", model: str = None) -> dict:
    """高速版: 2往復（① OCR ② 分析+アクション提案）"""
    if model is None:
        model = AI_PROVIDERS[provider]["fast"]

    ocr_text = ocr_single_frame(frame_b64, provider=provider, model=model)

    context = f"画面内容:\n{ocr_text}\n"
    if prompt:
        context += f"\n質問: {prompt}\n"

    combined_analysis = ai_text(
        provider, model,
        system_prompt=(
            "あなたはOJTメンターだ。画面に映っている内容を特定し、簡潔に答えろ。\n"
            "以下のフォーマットで回答しろ:\n\n"
            "【これは何か】1行で画面の状況を要約\n\n"
            + ("【回答】質問に3〜5行以内で回答。結論ファースト\n\n" if prompt else "")
            + "【次にやること】最大3個、各1行で\n\n"
            "【注意・危険】あれば1〜2行。なければ省略\n\n"
            "【参考リンク】関連URLがあれば記載。なければ省略\n\n"
            "ルール: 冗長表現禁止。合計300文字以内。"
        ),
        user_message=context,
    )
    return {"ocr_text": ocr_text, "combined_analysis": combined_analysis}


def deep_analyze(ocr_text: str, analysis: Optional[str], next_steps: Optional[str], provider: str = "claude", model: str = None) -> str:
    if model is None:
        model = AI_PROVIDERS[provider]["normal"]
    context = f"画面内容:\n{ocr_text[:500]}\n"
    if analysis:
        context += f"\n分析結果:\n{analysis[:300]}\n"
    if next_steps:
        context += f"\nアクション提案:\n{next_steps[:300]}\n"
    return ai_text(
        provider, model,
        system_prompt=(
            "OJTメンターとして、前の分析を踏まえて技術的な補足を行え。\n"
            "以下のフォーマットで回答:\n\n"
            "【技術的な補足】前の分析で触れていない重要なポイント（2〜3行）\n\n"
            "【よくあるミス】この作業で初心者がやりがちなミス（1〜2個）\n\n"
            "【ベストプラクティス】プロならこうする、という1行アドバイス\n\n"
            "既に述べたことの繰り返し禁止。新しい情報のみ。合計200文字以内。"
        ),
        user_message=context,
        max_tokens=600,
    )


def answer_followup(ocr_text: str, previous_analysis: Optional[str], question: str, provider: str = "claude", model: str = None) -> str:
    if model is None:
        model = AI_PROVIDERS[provider]["normal"]
    context = f"画面: {ocr_text[:500]}\n"
    if previous_analysis:
        context += f"前回の回答: {previous_analysis[:300]}\n"
    return ai_text(
        provider, model,
        system_prompt=(
            "OJTメンターとして追加質問に簡潔に答えろ。"
            "3〜5行以内。結論ファースト。コード例は最小限。"
            "関連URLがあれば末尾に。"
        ),
        user_message=f"{context}\n追加質問: {question}",
    )


# ──────────────────────────────────────────────
# スキーマ
# ──────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    image_base64: str
    prompt: Optional[str] = None
    provider: str = "claude"  # "claude" | "chatgpt" | "gemini"


class AnalyzeResponse(BaseModel):
    ocr_text: str
    analysis: Optional[str] = None
    next_steps: Optional[str] = None
    deep_analysis: Optional[str] = None
    provider: str = "claude"
    elapsed_ms: int


class AnalyzeFastRequest(BaseModel):
    image_base64: str
    prompt: Optional[str] = None
    speed: str = "fast"
    provider: str = "claude"


class AnalyzeFastResponse(BaseModel):
    ocr_text: str
    combined_analysis: str
    provider: str = "claude"
    elapsed_ms: int


class FollowupRequest(BaseModel):
    ocr_text: str
    previous_analysis: Optional[str] = None
    question: str
    provider: str = "claude"


class FollowupResponse(BaseModel):
    answer: str
    provider: str = "claude"
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

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    if HTML_PATH.exists():
        return HTMLResponse(content=HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>index.html が見つかりません</h1>", status_code=404)


@app.get("/health")
def health():
    return {"status": "ok", "time": time.time()}


@app.get("/providers")
def list_providers():
    """利用可能なAIプロバイダー一覧（APIキー設定済みかどうかも返す）"""
    result = {}
    for key, info in AI_PROVIDERS.items():
        if key == "claude":
            available = bool(os.environ.get("ANTHROPIC_API_KEY"))
        elif key == "chatgpt":
            available = bool(os.environ.get("OPENAI_API_KEY"))
        elif key == "gemini":
            available = bool(os.environ.get("GEMINI_API_KEY"))
        else:
            available = False
        result[key] = {"label": info["label"], "available": available, "models": {"normal": info["normal"], "fast": info["fast"]}}
    return JSONResponse(content=result)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    start = time.time()
    provider = req.provider if req.provider in AI_PROVIDERS else "claude"

    try:
        frame = decode_base64_image(req.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"画像デコード失敗: {e}")

    processed = preprocess_frame(frame)
    processed_b64 = frame_to_base64(processed)

    # 1往復目: OCR
    ocr_text = ocr_single_frame(processed_b64, provider=provider)

    # 2往復目: 分析（promptがある場合）
    analysis = None
    if req.prompt:
        analysis = analyze_with_context(ocr_text, req.prompt, history=[], provider=provider)

    # 3往復目: アクション提案
    try:
        next_steps = suggest_next_steps(ocr_text, analysis, provider=provider)
    except Exception as e:
        next_steps = f"次のアクション提案の生成に失敗: {e}"

    # 4往復目: 深掘り分析
    deep = None
    try:
        deep = deep_analyze(ocr_text, analysis, next_steps, provider=provider)
    except Exception as e:
        deep = f"深掘り分析の生成に失敗: {e}"

    return AnalyzeResponse(
        ocr_text=ocr_text, analysis=analysis, next_steps=next_steps,
        deep_analysis=deep, provider=provider,
        elapsed_ms=int((time.time() - start) * 1000),
    )


@app.post("/analyze-fast", response_model=AnalyzeFastResponse)
async def analyze_fast(req: AnalyzeFastRequest):
    start = time.time()
    provider = req.provider if req.provider in AI_PROVIDERS else "claude"

    try:
        frame = decode_base64_image(req.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"画像デコード失敗: {e}")

    processed = preprocess_frame(frame)
    processed_b64 = frame_to_base64(processed)

    fast_model = AI_PROVIDERS[provider]["fast"] if req.speed == "fast" else AI_PROVIDERS[provider]["normal"]
    try:
        result = analyze_image_fast(processed_b64, req.prompt, provider=provider, model=fast_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解析失敗: {e}")

    return AnalyzeFastResponse(
        ocr_text=result["ocr_text"], combined_analysis=result["combined_analysis"],
        provider=provider,
        elapsed_ms=int((time.time() - start) * 1000),
    )


@app.post("/analyze-upload", response_model=AnalyzeResponse)
async def analyze_upload(file: UploadFile = File(...), prompt: Optional[str] = Form(None), provider: Optional[str] = Form("claude")):
    start = time.time()
    prov = provider if provider in AI_PROVIDERS else "claude"
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="画像デコード失敗")

    processed = preprocess_frame(frame)
    processed_b64 = frame_to_base64(processed)
    ocr_text = ocr_single_frame(processed_b64, provider=prov)

    analysis = None
    if prompt:
        analysis = analyze_with_context(ocr_text, prompt, history=[], provider=prov)

    try:
        next_steps = suggest_next_steps(ocr_text, analysis, provider=prov)
    except Exception as e:
        next_steps = f"次のアクション提案の生成に失敗: {e}"

    return AnalyzeResponse(
        ocr_text=ocr_text, analysis=analysis, next_steps=next_steps,
        provider=prov, elapsed_ms=int((time.time() - start) * 1000),
    )


@app.post("/followup", response_model=FollowupResponse)
async def followup(req: FollowupRequest):
    start = time.time()
    provider = req.provider if req.provider in AI_PROVIDERS else "claude"
    try:
        answer = answer_followup(req.ocr_text, req.previous_analysis, req.question, provider=provider)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"回答失敗: {e}")
    return FollowupResponse(answer=answer, provider=provider, elapsed_ms=int((time.time() - start) * 1000))


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

            provider = data.get("provider", "claude")
            start = time.time()
            processed = preprocess_frame(frame)
            processed_b64 = frame_to_base64(processed)

            try:
                ocr_text = ocr_single_frame(processed_b64, provider=provider)
            except Exception as e:
                await websocket.send_json({"type": "error", "message": f"OCR失敗: {e}"})
                continue

            analysis = None
            prompt = data.get("prompt")
            if prompt:
                try:
                    analysis = analyze_with_context(ocr_text, prompt, state.history, provider=provider)
                except Exception as e:
                    analysis = f"分析失敗: {e}"

            state.prev_frame = frame
            state.history.append({"timestamp": time.time(), "text": ocr_text})
            if len(state.history) > 20:
                state.history.pop(0)

            await websocket.send_json({
                "type": "result", "ocr_text": ocr_text,
                "analysis": analysis, "provider": provider,
                "elapsed_ms": int((time.time() - start) * 1000),
            })

    except WebSocketDisconnect:
        sessions.pop(session_id, None)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
