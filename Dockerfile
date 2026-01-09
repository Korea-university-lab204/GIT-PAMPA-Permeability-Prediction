FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=mysite.settings

WORKDIR /app

# ✅ Chrome(Plotly/Kaleido용) 실행에 필요한 시스템 라이브러리들
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpangocairo-1.0-0 \
    libpango-1.0-0 \
    libcairo2 \
    libdrm2 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxext6 \
    libxi6 \
    libglib2.0-0 \
    ca-certificates \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    # ✅ Kaleido가 쓸 Chrome 다운로드/설치 (이게 핵심)
    plotly_get_chrome

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "gunicorn mysite.wsgi:application --bind 0.0.0.0:${PORT:-8000} --workers 1 --timeout 180"]
