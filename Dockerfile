# 1) 기본 Python 이미지 사용
FROM python:3.10-slim

# 2) 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=mysite.settings

# 3) 작업 디렉토리
WORKDIR /app

# 4) Playwright(Chromium) 실행에 필요한 OS 라이브러리 설치
RUN apt-get update && apt-get install -y \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libgbm1 libpango-1.0-0 libpangocairo-1.0-0 libcairo2 libasound2 \
    libx11-6 libx11-xcb1 libxcb1 libxext6 libxi6 libxtst6 \
    fonts-liberation \
 && rm -rf /var/lib/apt/lists/*

# 5) 의존성 파일 복사 및 설치
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 6) Playwright Chromium 브라우저 설치 (⭐ 핵심)
RUN python -m playwright install chromium

# 7) 프로젝트 전체 복사
COPY . .

# 8) 컨테이너에서 열 포트
EXPOSE 8000

# 9) gunicorn으로 Django 실행
CMD ["gunicorn", "mysite.wsgi:application", "--bind", "0.0.0.0:8000"]
