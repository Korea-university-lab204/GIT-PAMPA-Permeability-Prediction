import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DESCRIPTOR_NAMES = (
    [f'PEOE_VSA{i}' for i in range(1, 8)] +
    [f'SlogP_VSA{i}' for i in range(1, 8)] +
    [f'SMR_VSA{i}' for i in range(1, 7)] +
    [f'AUTOCORR2D_{i}' for i in range(1, 7)] +
    ['BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW']
)

# 전역 변수: 한 번만 로딩
model = None
scaler = None
input_columns = None
model_r2 = None


def _load_artifacts():
    """train_offline.py에서 저장한 model/scaler/meta를 한 번만 로딩."""
    global model, scaler, input_columns, model_r2

    if model is not None:
        return  # 이미 로딩 됨

    model_path = os.path.join(BASE_DIR, "model.pkl")
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    meta_path = os.path.join(BASE_DIR, "meta.pkl")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    meta = joblib.load(meta_path)

    input_columns = meta.get("input_columns")
    model_r2 = meta.get("r2")

    if input_columns is None:
        raise ValueError("meta.pkl에 'input_columns'가 없습니다. train_offline.py를 최신 버전으로 다시 실행해주세요.")


def _calc_descriptors_for_smiles(smiles: str):
    """단일 SMILES에 대해 디스크립터 벡터 계산."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # SMILES 파싱 실패 시 0으로 채움
        return [0.0] * len(DESCRIPTOR_NAMES)

    values = []
    for name in DESCRIPTOR_NAMES:
        try:
            func = getattr(Descriptors, name)
            values.append(func(mol))
        except Exception:
            values.append(0.0)
    return values


def predict_permeability(smiles_list, lecithin_pct, pH, dmso_conc):
    """
    Django에서 호출하는 최종 함수.
    - smiles_list: SMILES 문자열 리스트
    - lecithin_pct, pH, dmso_conc: 모든 SMILES에 공통으로 적용할 실험 조건
    반환값: 각 SMILES에 대한 예측 logPe 리스트
    """
    _load_artifacts()

    rows = []

    for s in smiles_list:
        desc_values = _calc_descriptors_for_smiles(s)

        row_dict = {
            'Lecithin(%)':      lecithin_pct,
            'pH':               pH,
            'DMSO_Conc (w/v%)': dmso_conc,
        }

        for name, val in zip(DESCRIPTOR_NAMES, desc_values):
            row_dict[name] = val

        rows.append(row_dict)

    full_df = pd.DataFrame(rows)
    full_df = full_df.reindex(columns=input_columns)

    X_scaled = scaler.transform(full_df)
    preds = model.predict(X_scaled)

    return preds


def get_model_metrics():
    """뷰에서 쓸 수 있도록 R² 반환."""
    _load_artifacts()
    return {"r2": model_r2}
