# 1) 기본 Python 이미지 사용
FROM python:3.10-slim

# 2) 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=mysite.settings

# 3) 작업 디렉토리
WORKDIR /app

# 4) 의존성 파일 복사 및 설치
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 5) 프로젝트 전체 복사
COPY . .

# 6) 컨테이너에서 열 포트
EXPOSE 8000

# 7) gunicorn으로 Django 실행
CMD ["gunicorn", "mysite.wsgi:application", "--bind", "0.0.0.0:8000"]
