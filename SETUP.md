# セットアップガイド（スマホアプリ + ローカルPCバックエンド）

スマホのカメラでPC画面を撮影して、Claude でリアルタイム解析する構成です。

## システム構成

```
[スマホ]                      [PC]                    [Anthropic]
  Expoアプリ ─── HTTP ──→  FastAPI (8000)  ──→  Claude API
  カメラ撮影                OCR + 解析
```

同じWi-Fiネットワーク内であれば動作します。

---

## 1. バックエンド（PC側）のセットアップ

### 1-1. 依存パッケージのインストール

```bash
cd backend
pip install -r requirements.txt
```

### 1-2. APIキーを設定

```bash
export ANTHROPIC_API_KEY="sk-ant-xxxxxxxxxxxxxx"
```

### 1-3. サーバー起動

```bash
cd backend
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

`--host 0.0.0.0` がポイント。これでスマホからアクセスできるようになります。

### 1-4. PC のIPアドレスを調べる

**macOS/Linux:**
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

**Windows:**
```bash
ipconfig
```

`192.168.x.x` のような値が表示されます。これを控えておきます。

### 1-5. 動作確認

ブラウザで `http://localhost:8000` にアクセスして JSON が返ればOK。
スマホと同じWi-Fiに繋いだ別デバイスから `http://<PCのIP>:8000` でも確認できます。

---

## 2. モバイルアプリ（スマホ側）のセットアップ

### 2-1. Node.js と Expo CLI のインストール

Node.js がない場合は https://nodejs.org からインストール。

```bash
npm install -g expo-cli
```

### 2-2. 依存パッケージのインストール

```bash
cd mobile
npm install
```

### 2-3. バックエンドURLの設定

`mobile/App.js` の冒頭を編集：

```javascript
const BACKEND_URL = 'http://192.168.1.10:8000';  // ← あなたのPCのIPに書き換える
```

### 2-4. Expo Go アプリを用意

スマホに **Expo Go** アプリをインストールします。
- iOS: App Store で "Expo Go" を検索
- Android: Google Play で "Expo Go" を検索

### 2-5. 開発サーバー起動

```bash
cd mobile
npx expo start
```

ターミナルに QR コードが表示されます。

### 2-6. スマホで QR コードを読み込む

- iOS → カメラアプリで QR を読む
- Android → Expo Go アプリ内の QR スキャナで読む

数秒でアプリが起動します。

---

## 3. 使い方

1. アプリ起動時にカメラ権限を許可
2. 上部のカメラプレビューにPCの画面が映るように構える
3. 下部のテキストボックスに Claude への指示を入力
   - 例: "エラーメッセージの原因を教えて"
   - 例: "このコードの改善点を提案して"
4. **📸 撮影＆解析** ボタンを押す
5. 数秒後、Claude の分析結果が表示される

**自動モード:** スイッチをONにすると5秒ごとに自動撮影されます。画面に変化がないフレームは自動スキップされます。

---

## トラブルシューティング

### スマホからPCに接続できない

- 同じWi-Fiに繋がっているか確認
- PCのファイアウォール設定で8000番ポートを許可
- macOS: 「システム設定」→「ネットワーク」→「ファイアウォール」
- Windows: Windows Defender ファイアウォールで許可

### 外出先からも使いたい

Ngrok を使うとインターネット越しにアクセスできます：

```bash
brew install ngrok   # または https://ngrok.com からダウンロード
ngrok http 8000
```

表示された `https://xxxx.ngrok-free.app` を `BACKEND_URL` に設定。

### OCR精度が低い

- カメラをPC画面に近づける（画面いっぱいに映す）
- 照明を明るくして画面の反射を減らす
- `backend/server.py` で `max_tokens` を増やす
- 角度をまっすぐにして歪みを減らす

### API コストが心配

- アプリの自動モードを使わず手動撮影にする
- `AUTO_CAPTURE_INTERVAL_MS` を長く設定（例: 10000ms = 10秒）
- バックエンドの類似度判定により、変化がない画面は自動スキップされるのでコスト削減されます

---

## 次の改善アイデア

- WebSocket ストリーミング（`/ws/stream/{session_id}`）を使ってレイテンシ削減
- 結果履歴をSQLiteに保存
- 音声で Claude の回答を読み上げ
- PC画面の特定領域のみクロップして送信（精度＆速度向上）
