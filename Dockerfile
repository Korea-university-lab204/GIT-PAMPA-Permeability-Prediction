# RDKit이 이미 설치된 공식 이미지 사용
FROM rdkit/rdkit:latest

# 작업 디렉토리
WORKDIR /app

# 의존성 파일 복사
COPY requirements.txt .

# 파이썬 패키지 설치
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 프로젝트 전체 복사
COPY . .

# 환경 변수
ENV DJANGO_SETTINGS_MODULE=mysite.settings
ENV PYTHONUNBUFFERED=1

# 컨테이너 내에서 사용할 포트
EXPOSE 8000

# gunicorn으로 Django 실행
CMD ["gunicorn", "mysite.wsgi:application", "--bind", "0.0.0.0:8000"]
