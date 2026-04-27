FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依存関係のインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# サーバー実行に必要なコードのみをコピー
COPY server.py .
COPY video_ocr_claude.py .
COPY index.html .

# FastAPIを起動
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
