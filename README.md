# Video OCR + Claude 分析ツール

画面録画の動画からテキストを自動抽出し、Claude API で分析・アドバイスを得るツールです。

## セットアップ

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. APIキーの設定

```bash
export ANTHROPIC_API_KEY="sk-ant-xxxxxxxxxxxxxxxx"
```

Anthropic のAPIキーは https://console.anthropic.com で取得できます。

## 使い方

### テキスト抽出のみ

```bash
python video_ocr_claude.py extract 録画.mp4
```

動画から1秒間隔でフレームを抽出し、Claude Vision で画面上のテキストを読み取ります。結果は `output/ocr_result.txt` に保存されます。

### テキスト抽出 + 分析

```bash
python video_ocr_claude.py analyze 録画.mp4 --prompt "このコードのバグを見つけて改善案を教えて"
```

OCR でテキスト抽出した後、その内容を Claude に渡して分析します。コードレビュー、手順の要約、エラー解析などに使えます。

### 画像ベース分析（visual モード）

```bash
python video_ocr_claude.py visual 録画.mp4 --prompt "UIデザインの改善点を提案して"
```

OCR を経由せず、フレーム画像を直接 Claude に送って分析します。レイアウトや色、UIの配置など視覚的な分析に最適です。

### カメラ入力

```bash
python video_ocr_claude.py camera --interval 5 --prompt "画面の内容を要約して"
```

カメラからリアルタイムで撮影し、分析します。`q` キーで撮影を終了します。

## オプション

| オプション | 説明 | デフォルト |
|---|---|---|
| `--interval` | フレーム抽出間隔（秒） | 1.0 |
| `--max-frames` | 最大フレーム数 | 20 |
| `--model` | Claude モデル名 | claude-sonnet-4-6 |
| `--output` | 出力ディレクトリ | output |
| `--camera-id` | カメラデバイスID（camera時） | 0 |

## 出力ファイル

- `output/ocr_result.txt` - OCR結果（テキスト形式）
- `output/ocr_result.json` - OCR結果（JSON形式、プログラムから利用しやすい）
- `output/analysis.md` - Claude の分析結果（analyze/visual 使用時）

## 精度を高めるヒント

**フレーム抽出の調整:**
- `--interval 0.5` にすると細かい変化も拾える（API コストは増加）
- `--max-frames 30` で長い動画にも対応

**画面録画の品質:**
- 解像度は 1080p 以上を推奨
- フォントサイズが小さすぎると読み取り精度が落ちる
- 画面の動きが速い部分はブレとして自動スキップされる

**プロンプトの工夫:**
- 具体的な指示が精度向上に直結する
- 例: 「エラーメッセージを抜き出して原因を分析して」
- 例: 「このターミナル出力からデプロイ手順を再構成して」

## コスト目安

Claude Sonnet で 20 フレームを処理した場合、およそ $0.10〜$0.30 程度です（画像サイズとテキスト量による）。`--max-frames` で上限を制御できます。

## ライセンス

MIT
