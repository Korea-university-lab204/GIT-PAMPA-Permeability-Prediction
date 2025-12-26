import joblib
meta = joblib.load("meta.pkl")
print(meta.keys())
print(meta.get("r2"))
