import joblib
import os

# 파일 경로 지정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
META_PATH = os.path.join(BASE_DIR, "meta.pkl")

# meta.pkl 불러오기
meta = joblib.load(META_PATH)

# 기존 내용 출력
print("✅ 현재 meta 내용:")
for k, v in meta.items():
    print(f"{k}: {v}")

# 원하는 항목 수정 (예: r2 값을 바꿔보자)
meta["r2"] = 0.8123  # 원하는 값으로 수정
meta["note"] = "2025-12-26 R2 manual update"

# 저장
joblib.dump(meta, META_PATH)

print("✅ meta.pkl 수정 및 저장 완료!")
