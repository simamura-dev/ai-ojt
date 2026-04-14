#!/usr/bin/env python3
"""
video_ocr_claude.py
===================
画面録画動画からテキストを抽出し、Claude APIで分析するツール。

使い方:
    # 基本的なテキスト抽出
    python video_ocr_claude.py extract video.mp4

    # テキスト抽出 + Claude分析
    python video_ocr_claude.py analyze video.mp4 --prompt "コードのバグを見つけて"

    # リアルタイムカメラ入力
    python video_ocr_claude.py camera --interval 3
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import anthropic
import cv2
import numpy as np


# ──────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_FRAME_INTERVAL = 1.0      # 秒ごとにフレーム抽出
DEFAULT_MAX_FRAMES = 20           # 最大フレーム数（API コスト管理）
BLUR_THRESHOLD = 50.0             # ブレ検出しきい値（低い＝ブレている）
SIMILARITY_THRESHOLD = 0.95       # フレーム重複判定しきい値
MAX_IMAGE_SIZE = (1568, 1568)     # Claude API の推奨最大サイズ


# ──────────────────────────────────────────────
# フレーム前処理
# ──────────────────────────────────────────────
def calc_sharpness(frame: np.ndarray) -> float:
    """ラプラシアン分散でシャープネスを計測する。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def calc_similarity(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """2フレーム間の構造類似度(SSIM簡易版)を計算する。"""
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    # リサイズして高速比較
    size = (320, 240)
    gray_a = cv2.resize(gray_a, size)
    gray_b = cv2.resize(gray_b, size)

    # 正規化相互相関
    result = cv2.matchTemplate(gray_a, gray_b, cv2.TM_CCOEFF_NORMED)
    return float(result[0][0])


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """画面録画向けの前処理を行う。"""
    h, w = frame.shape[:2]

    # Claude API 推奨サイズにリサイズ
    max_w, max_h = MAX_IMAGE_SIZE
    if w > max_w or h > max_h:
        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # コントラスト強調（画面録画向け: CLAHEでテキスト可読性向上）
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge([l_channel, a_channel, b_channel])
    frame = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # 軽いシャープニング
    kernel = np.array([
        [0, -0.5, 0],
        [-0.5, 3, -0.5],
        [0, -0.5, 0]
    ])
    frame = cv2.filter2D(frame, -1, kernel)

    return frame


def frame_to_base64(frame: np.ndarray) -> str:
    """フレームをbase64エンコードしたPNG文字列に変換する。"""
    _, buffer = cv2.imencode(".png", frame)
    return base64.standard_b64encode(buffer).decode("utf-8")


# ──────────────────────────────────────────────
# 動画からフレーム抽出
# ──────────────────────────────────────────────
def extract_key_frames(
    video_path: str,
    interval: float = DEFAULT_FRAME_INTERVAL,
    max_frames: int = DEFAULT_MAX_FRAMES,
    blur_threshold: float = BLUR_THRESHOLD,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> list[dict]:
    """
    動画からキーフレームを抽出する。

    フィルタリング:
      1. 一定間隔（interval秒）でフレームを取得
      2. ブレの大きいフレームを除外
      3. 前フレームと類似度が高いフレームを除外（画面変化がないシーン）

    Returns:
        list[dict]: [{
            "frame": np.ndarray,
            "timestamp": float,
            "sharpness": float,
            "base64": str
        }, ...]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"動画を開けません: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    frame_step = max(1, int(fps * interval))

    print(f"📹 動画情報: {duration:.1f}秒, {fps:.1f}fps, {total_frames}フレーム")
    print(f"📊 {interval}秒間隔で抽出, 最大{max_frames}フレーム")

    key_frames = []
    prev_frame = None
    frame_idx = 0

    while cap.isOpened() and len(key_frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            sharpness = calc_sharpness(frame)

            # ブレチェック
            if sharpness < blur_threshold:
                frame_idx += 1
                continue

            # 類似度チェック（前フレームと比較）
            if prev_frame is not None:
                sim = calc_similarity(frame, prev_frame)
                if sim > similarity_threshold:
                    frame_idx += 1
                    continue

            # 前処理
            processed = preprocess_frame(frame)
            timestamp = frame_idx / fps

            key_frames.append({
                "frame": processed,
                "timestamp": timestamp,
                "sharpness": sharpness,
                "base64": frame_to_base64(processed),
            })

            prev_frame = frame
            print(f"  ✅ フレーム抽出: {timestamp:.1f}秒 (シャープネス: {sharpness:.1f})")

        frame_idx += 1

    cap.release()
    print(f"🎯 合計 {len(key_frames)} フレームを抽出しました")
    return key_frames


# ──────────────────────────────────────────────
# カメラからリアルタイム取得
# ──────────────────────────────────────────────
def capture_from_camera(
    interval: float = 3.0,
    max_frames: int = DEFAULT_MAX_FRAMES,
    camera_id: int = 0,
) -> list[dict]:
    """カメラからリアルタイムでフレームを取得する。"""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"カメラ(ID={camera_id})を開けません")

    print(f"📷 カメラ起動中... {interval}秒間隔で撮影 (qキーで終了)")

    key_frames = []
    prev_frame = None

    try:
        while len(key_frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Camera - Press 'q' to stop", frame)
            key = cv2.waitKey(int(interval * 1000)) & 0xFF

            if key == ord("q"):
                break

            sharpness = calc_sharpness(frame)
            if sharpness < BLUR_THRESHOLD:
                print(f"  ⚠️  ブレ検出 (シャープネス: {sharpness:.1f}) - スキップ")
                continue

            if prev_frame is not None:
                sim = calc_similarity(frame, prev_frame)
                if sim > SIMILARITY_THRESHOLD:
                    print(f"  ⏭️  前フレームと類似 ({sim:.2f}) - スキップ")
                    continue

            processed = preprocess_frame(frame)
            key_frames.append({
                "frame": processed,
                "timestamp": time.time(),
                "sharpness": sharpness,
                "base64": frame_to_base64(processed),
            })

            prev_frame = frame
            print(f"  ✅ フレーム取得 #{len(key_frames)} (シャープネス: {sharpness:.1f})")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"🎯 合計 {len(key_frames)} フレームを取得しました")
    return key_frames


# ──────────────────────────────────────────────
# Claude API 連携
# ──────────────────────────────────────────────
def ocr_with_claude(
    frames: list[dict],
    model: str = DEFAULT_MODEL,
) -> list[dict]:
    """
    Claude Vision API でフレームからテキストを抽出する。

    Returns:
        list[dict]: [{"timestamp": float, "text": str}, ...]
    """
    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY 環境変数を使用
    results = []

    print(f"\n🤖 Claude ({model}) でテキスト抽出中...")

    for i, frame_data in enumerate(frames):
        print(f"  [{i + 1}/{len(frames)}] {frame_data['timestamp']:.1f}秒のフレームを処理中...")

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": frame_data["base64"],
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "この画面録画のスクリーンショットに表示されているテキストを"
                                "すべて正確に読み取ってください。\n\n"
                                "ルール:\n"
                                "- 画面上のすべてのテキストを漏れなく抽出する\n"
                                "- UIの構造（メニュー、ボタン、本文など）がわかるように整理する\n"
                                "- コードが含まれている場合はコードブロックで囲む\n"
                                "- 読み取れない文字は [不明] と記載する\n"
                                "- テキストのみを出力し、説明は不要"
                            ),
                        },
                    ],
                }
            ],
        )

        text = response.content[0].text
        results.append({
            "timestamp": frame_data["timestamp"],
            "text": text,
        })
        print(f"    → {len(text)}文字を抽出")

    return results


def analyze_with_claude(
    ocr_results: list[dict],
    prompt: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    OCR結果をまとめてClaude に分析・アドバイスを依頼する。

    Args:
        ocr_results: ocr_with_claude() の出力
        prompt: ユーザーの質問・指示
        model: 使用するモデル

    Returns:
        Claude の分析結果テキスト
    """
    client = anthropic.Anthropic()

    # OCR結果を時系列でまとめる
    combined_text = ""
    for result in ocr_results:
        ts = result["timestamp"]
        combined_text += f"\n--- {ts:.1f}秒時点の画面 ---\n{result['text']}\n"

    print(f"\n🧠 Claude ({model}) で分析中...")
    print(f"   プロンプト: {prompt[:80]}...")

    response = client.messages.create(
        model=model,
        max_tokens=8192,
        system=(
            "あなたは画面録画の内容を分析するエキスパートです。"
            "以下のOCR結果は動画の各時点で画面に表示されていたテキストです。"
            "時系列の変化も考慮して、ユーザーの質問に的確に回答してください。"
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"## 画面録画のOCR結果\n{combined_text}\n\n"
                    f"## 質問・指示\n{prompt}"
                ),
            }
        ],
    )

    return response.content[0].text


def analyze_frames_with_claude(
    frames: list[dict],
    prompt: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    フレーム画像を直接Claude に送り、画像ベースで分析する。
    OCR を経由しないので、レイアウトや色も考慮した分析が可能。
    """
    client = anthropic.Anthropic()

    # 画像コンテンツを構築（最大5フレームを選択）
    selected = frames[:5] if len(frames) > 5 else frames
    content = []

    for i, frame_data in enumerate(selected):
        content.append({
            "type": "text",
            "text": f"[{frame_data['timestamp']:.1f}秒時点の画面]",
        })
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": frame_data["base64"],
            },
        })

    content.append({
        "type": "text",
        "text": f"\n## 質問・指示\n{prompt}",
    })

    print(f"\n🧠 Claude ({model}) で画像ベース分析中... ({len(selected)}フレーム)")

    response = client.messages.create(
        model=model,
        max_tokens=8192,
        system=(
            "あなたは画面録画の内容を分析するエキスパートです。"
            "送られた画像は動画から抽出したフレームです（時系列順）。"
            "画面の内容、レイアウト、変化を踏まえてユーザーの質問に回答してください。"
        ),
        messages=[{"role": "user", "content": content}],
    )

    return response.content[0].text


# ──────────────────────────────────────────────
# 結果の保存
# ──────────────────────────────────────────────
def save_results(
    ocr_results: list[dict],
    analysis: str | None,
    output_dir: str = "output",
) -> dict[str, str]:
    """結果をファイルに保存する。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved = {}

    # OCR結果をテキストファイルに保存
    ocr_path = out / "ocr_result.txt"
    with open(ocr_path, "w", encoding="utf-8") as f:
        for result in ocr_results:
            f.write(f"=== {result['timestamp']:.1f}秒 ===\n")
            f.write(result["text"])
            f.write("\n\n")
    saved["ocr"] = str(ocr_path)
    print(f"💾 OCR結果: {ocr_path}")

    # OCR結果をJSON形式でも保存
    json_path = out / "ocr_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ocr_results, f, ensure_ascii=False, indent=2)
    saved["json"] = str(json_path)

    # 分析結果を保存
    if analysis:
        analysis_path = out / "analysis.md"
        with open(analysis_path, "w", encoding="utf-8") as f:
            f.write("# Claude 分析結果\n\n")
            f.write(analysis)
        saved["analysis"] = str(analysis_path)
        print(f"💾 分析結果: {analysis_path}")

    return saved


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="画面録画からテキスト抽出 & Claude分析ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # テキスト抽出のみ
  python video_ocr_claude.py extract demo.mp4

  # テキスト抽出 + Claude分析
  python video_ocr_claude.py analyze demo.mp4 --prompt "このコードの問題点を教えて"

  # 画像ベース分析（OCRを経由せず画像をそのまま送る）
  python video_ocr_claude.py visual demo.mp4 --prompt "UIの改善点を提案して"

  # カメラからリアルタイム入力
  python video_ocr_claude.py camera --interval 5 --prompt "ホワイトボードの内容を要約して"
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- extract コマンド ---
    p_extract = subparsers.add_parser("extract", help="動画からテキスト抽出のみ")
    p_extract.add_argument("video", help="入力動画ファイルのパス")
    p_extract.add_argument("--interval", type=float, default=DEFAULT_FRAME_INTERVAL,
                           help=f"フレーム抽出間隔（秒） デフォルト: {DEFAULT_FRAME_INTERVAL}")
    p_extract.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
                           help=f"最大フレーム数 デフォルト: {DEFAULT_MAX_FRAMES}")
    p_extract.add_argument("--model", default=DEFAULT_MODEL, help="Claude モデル名")
    p_extract.add_argument("--output", default="output", help="出力ディレクトリ")

    # --- analyze コマンド ---
    p_analyze = subparsers.add_parser("analyze", help="テキスト抽出 + Claude分析")
    p_analyze.add_argument("video", help="入力動画ファイルのパス")
    p_analyze.add_argument("--prompt", required=True, help="Claude への質問・指示")
    p_analyze.add_argument("--interval", type=float, default=DEFAULT_FRAME_INTERVAL)
    p_analyze.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    p_analyze.add_argument("--model", default=DEFAULT_MODEL)
    p_analyze.add_argument("--output", default="output")

    # --- visual コマンド ---
    p_visual = subparsers.add_parser("visual", help="画像ベースで直接Claude分析")
    p_visual.add_argument("video", help="入力動画ファイルのパス")
    p_visual.add_argument("--prompt", required=True, help="Claude への質問・指示")
    p_visual.add_argument("--interval", type=float, default=DEFAULT_FRAME_INTERVAL)
    p_visual.add_argument("--max-frames", type=int, default=5)
    p_visual.add_argument("--model", default=DEFAULT_MODEL)
    p_visual.add_argument("--output", default="output")

    # --- camera コマンド ---
    p_camera = subparsers.add_parser("camera", help="カメラからリアルタイム取得")
    p_camera.add_argument("--interval", type=float, default=3.0,
                          help="撮影間隔（秒） デフォルト: 3")
    p_camera.add_argument("--max-frames", type=int, default=10)
    p_camera.add_argument("--prompt", default=None, help="Claude への質問（省略時はOCRのみ）")
    p_camera.add_argument("--model", default=DEFAULT_MODEL)
    p_camera.add_argument("--camera-id", type=int, default=0)
    p_camera.add_argument("--output", default="output")

    args = parser.parse_args()

    # ── フレーム取得 ──
    if args.command == "camera":
        frames = capture_from_camera(
            interval=args.interval,
            max_frames=args.max_frames,
            camera_id=args.camera_id,
        )
    else:
        frames = extract_key_frames(
            video_path=args.video,
            interval=args.interval,
            max_frames=args.max_frames,
        )

    if not frames:
        print("❌ フレームを抽出できませんでした")
        sys.exit(1)

    # ── 処理分岐 ──
    analysis = None

    if args.command == "visual":
        # 画像ベース分析（OCR不要）
        analysis = analyze_frames_with_claude(frames, args.prompt, args.model)
        # visual モードでは OCR 結果なし → ダミーの結果を作成
        ocr_results = [{"timestamp": f["timestamp"], "text": "(画像ベース分析)"} for f in frames]
    else:
        # OCR 実行
        ocr_results = ocr_with_claude(frames, args.model)

        # analyze / camera + prompt の場合は分析も実行
        prompt = getattr(args, "prompt", None)
        if prompt:
            analysis = analyze_with_claude(ocr_results, prompt, args.model)

    # ── 結果保存 ──
    saved = save_results(ocr_results, analysis, args.output)

    # ── 結果表示 ──
    print("\n" + "=" * 60)
    if analysis:
        print("📝 分析結果:")
        print("-" * 60)
        print(analysis)
    else:
        print("📝 OCR結果:")
        print("-" * 60)
        for r in ocr_results:
            print(f"\n[{r['timestamp']:.1f}秒]")
            print(r["text"][:500])
            if len(r["text"]) > 500:
                print("... (省略)")

    print("=" * 60)
    print(f"\n✅ 完了! 結果は {args.output}/ に保存されました")


if __name__ == "__main__":
    main()
