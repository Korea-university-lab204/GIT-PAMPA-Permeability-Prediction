from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import torch
import torch.nn as nn

from rdkit import Chem
from rdkit.Chem import (
    Descriptors,
    Crippen,
    Lipinski,
    rdMolDescriptors,
    QED,
    AllChem,
)
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit import DataStructs


# =========================================================
# 경로 & 상수
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

# ✅ 최종 Stage 2 모델 파일명
MODEL_PATH = BASE_DIR / "MLP_XGBoost_Condition_model.pt"
PREPROCESSOR_PATH = BASE_DIR / "MLP_XGBoost_Condition_preprocessor.pkl"

# GIT-PAMPA condition range
LEC_MIN, LEC_MAX = 1.0, 20.0
PH_MIN, PH_MAX = 5.5, 7.4
DMSO_MIN, DMSO_MAX = 0.5, 10.0

COND_COLS = {
    "lec": "Lecithin(%)",
    "ph": "pH",
    "dmso": "DMSO_Conc (w/v%)",
}

# 웹/PDF 우측 성능 박스에 표시할 최종 Stage 2 성능
MODEL_META = {
    "model_name": "XGBoost-derived descriptors + MLP",
    "features": 2102,
    "r2": 0.9788,
    "rmse": 0.0815,
    "mae": 0.0519,
    "cv_rmse_mean": 0.0960,
    "cv_rmse_std": 0.0044,
    "cv_r2_mean": 0.9694,
    "cv_r2_std": 0.0027,
}


_model = None
_preprocessor = None
_feature_cols = None
_xgb_feature_cols = None
_condition_cols = None


# =========================================================
# MLP 모델 구조
# - 학습 시 저장된 state_dict 구조와 동일하게 구성
# =========================================================
class ConditionMLP(nn.Module):
    def __init__(self, input_dim, hidden=(512, 256, 128, 64), dropout=0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =========================================================
# 모델/전처리기 로딩
# =========================================================
def load_artifacts():
    """
    MLP_XGBoost_Condition_model.pt
    MLP_XGBoost_Condition_preprocessor.pkl
    두 파일을 predictor 폴더에서 로딩.
    """
    global _model, _preprocessor, _feature_cols, _xgb_feature_cols, _condition_cols

    if _model is not None:
        return _model, _preprocessor, _feature_cols, _xgb_feature_cols, _condition_cols

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    config = checkpoint["config"]

    model = ConditionMLP(
        input_dim=config["input_dim"],
        hidden=tuple(config["hidden"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # scikit-learn 버전 차이 대응
    # 학습 환경: sklearn 1.7.x, 실행 환경이 더 높으면 SimpleImputer 내부 속성명이 달라질 수 있음.
    try:
        imputer = preprocessor.named_steps.get("imputer")
        if imputer is not None and not hasattr(imputer, "_fill_dtype"):
            imputer._fill_dtype = getattr(imputer, "_fit_dtype", np.float64)
    except Exception:
        pass

    _model = model
    _preprocessor = preprocessor
    _feature_cols = list(checkpoint["feature_cols"])
    _xgb_feature_cols = list(checkpoint["xgb_feature_cols"])
    _condition_cols = list(checkpoint["condition_cols"])

    return _model, _preprocessor, _feature_cols, _xgb_feature_cols, _condition_cols


# =========================================================
# RDKit descriptor 계산
# =========================================================
def _mol_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol


def _safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return default
        return x
    except Exception:
        return default


def _calc_core_descriptors(mol):
    """
    checkpoint의 xgb_feature_cols 중 Morgan bit를 제외한
    RDKit descriptor들을 계산.
    """
    # partial charge 계산
    try:
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = []
        for atom in mol.GetAtoms():
            val = atom.GetProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else "0"
            charges.append(_safe_float(val))
        max_partial_charge = max(charges) if charges else 0.0
        min_partial_charge = min(charges) if charges else 0.0
    except Exception:
        max_partial_charge = 0.0
        min_partial_charge = 0.0

    # 원자 카운트
    atomic_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    num_n = sum(1 for z in atomic_nums if z == 7)
    num_s = sum(1 for z in atomic_nums if z == 16)
    num_hal = sum(1 for z in atomic_nums if z in [9, 17, 35, 53])

    # 고리 정보
    try:
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        num_aromatic_rings = 0
        num_aliphatic_rings = 0
        num_aromatic_hetero = 0

        for ring in atom_rings:
            atoms = [mol.GetAtomWithIdx(i) for i in ring]
            is_aromatic = all(a.GetIsAromatic() for a in atoms)
            if is_aromatic:
                num_aromatic_rings += 1
                if any(a.GetAtomicNum() not in [6, 1] for a in atoms):
                    num_aromatic_hetero += 1
            else:
                num_aliphatic_rings += 1
    except Exception:
        num_aromatic_rings = 0
        num_aliphatic_rings = 0
        num_aromatic_hetero = 0

    # amide bond count: C(=O)-N pattern
    try:
        amide_pattern = Chem.MolFromSmarts("C(=O)N")
        num_amide = len(mol.GetSubstructMatches(amide_pattern))
    except Exception:
        num_amide = 0

    desc = {
        "MolLogP": Crippen.MolLogP(mol),
        "MolMR": Crippen.MolMR(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "NumHDonors": Lipinski.NumHDonors(mol),
        "NumHAcceptors": Lipinski.NumHAcceptors(mol),
        "MolWt": Descriptors.MolWt(mol),
        "ExactMolWt": Descriptors.ExactMolWt(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        "NumRotatableBonds": Lipinski.NumRotatableBonds(mol),
        "NumRings": rdMolDescriptors.CalcNumRings(mol),
        "NumAromaticRings": num_aromatic_rings,
        "NumAliphaticRings": num_aliphatic_rings,
        "RingCount": Lipinski.RingCount(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "NumHeteroatoms": Lipinski.NumHeteroatoms(mol),
        "NumNitrogens": num_n,
        "NumSulfurs": num_s,
        "NumHalogens": num_hal,
        "FormalCharge": Chem.GetFormalCharge(mol),
        "MaxPartialCharge": max_partial_charge,
        "MinPartialCharge": min_partial_charge,
        "LabuteASA": rdMolDescriptors.CalcLabuteASA(mol),
        "Chi0v": Descriptors.Chi0v(mol),
        "Chi1v": Descriptors.Chi1v(mol),
        "Chi2v": Descriptors.Chi2v(mol),
        "Chi3v": Descriptors.Chi3v(mol),
        "Kappa1": Descriptors.Kappa1(mol),
        "Kappa2": Descriptors.Kappa2(mol),
        "Kappa3": Descriptors.Kappa3(mol),
        "HallKierAlpha": Descriptors.HallKierAlpha(mol),
        "BalabanJ": Descriptors.BalabanJ(mol),
        "MaxEStateIndex": Descriptors.MaxEStateIndex(mol),
        "MinEStateIndex": Descriptors.MinEStateIndex(mol),
        "QED": QED.qed(mol),
        "NumStereocenters": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        "NumAmideBonds": num_amide,
        "NumAromaticHetero": num_aromatic_hetero,
    }

    # VSA descriptors
    # 학습 feature에는 일부 VSA 번호만 들어있지만,
    # 없는 번호까지 계산해도 이후 reindex에서 필요한 컬럼만 사용됨.
    for prefix, max_i in [
        ("SlogP_VSA", 12),
        ("SMR_VSA", 10),
        ("PEOE_VSA", 14),
        ("EState_VSA", 11),
    ]:
        for i in range(1, max_i + 1):
            fname = f"{prefix}{i}"
            func = getattr(Descriptors, fname, None)
            if func is not None:
                desc[fname] = _safe_float(func(mol))
            else:
                desc[fname] = 0.0

    return desc


def _calc_morgan_bits(mol, n_bits=2048, radius=2):
    fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return {f"Morgan_{i}": int(arr[i]) for i in range(n_bits)}


def calc_features_for_smiles(smiles: str):
    """
    SMILES 하나에 대해 MLP 입력에 필요한 descriptor + Morgan FP 계산.
    """
    mol = _mol_from_smiles(smiles)

    desc = _calc_core_descriptors(mol)
    desc.update(_calc_morgan_bits(mol, n_bits=2048, radius=2))

    return desc


def _build_input_dataframe(smiles, lec, ph, dmso):
    """
    lec/ph/dmso는 scalar 또는 numpy array 가능.
    반환 DataFrame은 checkpoint["feature_cols"] 순서로 정렬.
    """
    _, _, feature_cols, xgb_feature_cols, _ = load_artifacts()

    lec_arr = np.atleast_1d(lec).astype(float)
    ph_arr = np.atleast_1d(ph).astype(float)
    dmso_arr = np.atleast_1d(dmso).astype(float)

    n = max(len(lec_arr), len(ph_arr), len(dmso_arr))

    if len(lec_arr) == 1:
        lec_arr = np.repeat(lec_arr, n)
    if len(ph_arr) == 1:
        ph_arr = np.repeat(ph_arr, n)
    if len(dmso_arr) == 1:
        dmso_arr = np.repeat(dmso_arr, n)

    feat = calc_features_for_smiles(smiles)
    base = {col: feat.get(col, 0.0) for col in xgb_feature_cols}

    rows = []
    for i in range(n):
        row = base.copy()
        row[COND_COLS["lec"]] = float(lec_arr[i])
        row[COND_COLS["ph"]] = float(ph_arr[i])
        row[COND_COLS["dmso"]] = float(dmso_arr[i])
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.reindex(columns=feature_cols, fill_value=0.0)
    return df


# =========================================================
# 모델 메타 정보
# =========================================================
def get_model_meta():
    return MODEL_META.copy()


# =========================================================
# 단일 예측 / 민감도
# =========================================================
def predict_single(smiles, lec, ph, dmso):
    """
    최종 Stage 2 MLP 예측.
    기존 웹 GUI는 그대로 두고 내부 모델만 교체.
    """
    model, preprocessor, _, _, _ = load_artifacts()

    full = _build_input_dataframe(smiles, lec, ph, dmso)
    X_scaled = preprocessor.transform(full)

    with torch.no_grad():
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        pred = model(x_tensor).cpu().numpy().reshape(-1)

    return float(pred[0])


def compute_local_sensitivity(smiles, lec, ph, dmso):
    """
    현재 조건에서
    - Lec +1
    - pH +0.1
    - DMSO +1
    변화 시 logPe 변화량 계산.
    """
    base = predict_single(smiles, lec, ph, dmso)

    lec2 = min(LEC_MAX, lec + 1)
    ph2 = min(PH_MAX, ph + 0.1)
    dmso2 = min(DMSO_MAX, dmso + 1)

    return {
        "lec": predict_single(smiles, lec2, ph, dmso) - base,
        "ph": predict_single(smiles, lec, ph2, dmso) - base,
        "dmso": predict_single(smiles, lec, ph, dmso2) - base,
    }


# =========================================================
# RDKit 기본 특성: 우측 하단 박스/PDF용
# =========================================================
def get_basic_rdkit_descriptors(smiles: str):
    mol = _mol_from_smiles(smiles)

    desc = {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotatableBonds": Lipinski.NumRotatableBonds(mol),
        "RingCount": Lipinski.RingCount(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
    }
    return desc


# =========================================================
# 3D surface 계산
# =========================================================
def get_surface_for_smiles(smiles, fixed_var="dmso", fixed_value=0.5, num_points=30):
    fixed_var = fixed_var.lower()

    lec_range = np.linspace(LEC_MIN, LEC_MAX, num_points)
    ph_range = np.linspace(PH_MIN, PH_MAX, num_points)
    dmso_range = np.linspace(DMSO_MIN, DMSO_MAX, num_points)

    if fixed_var == "dmso":
        X_vals, Y_vals = lec_range, ph_range
        x_label, y_label = "Lecithin (%)", "pH"
    elif fixed_var == "lec":
        X_vals, Y_vals = dmso_range, ph_range
        x_label, y_label = "DMSO (%)", "pH"
    else:
        X_vals, Y_vals = dmso_range, lec_range
        x_label, y_label = "DMSO (%)", "Lecithin (%)"

    X, Y = np.meshgrid(X_vals, Y_vals)
    N = X.size

    if fixed_var == "dmso":
        lec_flat = X.ravel()
        ph_flat = Y.ravel()
        dmso_flat = np.full(N, fixed_value)
    elif fixed_var == "lec":
        dmso_flat = X.ravel()
        ph_flat = Y.ravel()
        lec_flat = np.full(N, fixed_value)
    else:
        dmso_flat = X.ravel()
        lec_flat = Y.ravel()
        ph_flat = np.full(N, fixed_value)

    # batch prediction
    model, preprocessor, _, _, _ = load_artifacts()
    full = _build_input_dataframe(smiles, lec_flat, ph_flat, dmso_flat)
    X_scaled = preprocessor.transform(full)

    with torch.no_grad():
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        preds = model(x_tensor).cpu().numpy().reshape(-1)

    Z = preds.reshape(X.shape)
    return X, Y, Z, x_label, y_label


# =========================================================
# Plotly 그래프 + 슬라이더
# =========================================================
def make_plotly_surface_with_slider(smiles, fixed_var="dmso", num_points=25, n_steps=10):
    fixed_var = fixed_var.lower()

    if fixed_var == "dmso":
        slider_label = "DMSO"
        slider_values = np.linspace(DMSO_MIN, DMSO_MAX, n_steps)
    elif fixed_var == "lec":
        slider_label = "Lecithin"
        slider_values = np.linspace(LEC_MIN, LEC_MAX, n_steps)
    else:
        slider_label = "pH"
        slider_values = np.linspace(PH_MIN, PH_MAX, n_steps)

    slider_values = list(slider_values)
    first_val = slider_values[0]

    X, Y, Z0, x_label, y_label = get_surface_for_smiles(
        smiles=smiles,
        fixed_var=fixed_var,
        fixed_value=first_val,
        num_points=num_points,
    )

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=X,
        y=Y,
        z=Z0,
        colorscale="Inferno",
        colorbar=dict(title="logPe"),
    ))

    frames = []
    for v in slider_values:
        _, _, Z, _, _ = get_surface_for_smiles(
            smiles=smiles,
            fixed_var=fixed_var,
            fixed_value=v,
            num_points=num_points,
        )
        frames.append(go.Frame(
            data=[go.Surface(x=X, y=Y, z=Z, showscale=False, colorscale="Inferno")],
            name=f"{v:.3f}",
        ))

    fig.frames = frames

    steps = [{
        "label": f"{v:.3f}",
        "method": "animate",
        "args": [[f"{v:.3f}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
    } for v in slider_values]

    fig.update_layout(
        title=f"logPe Surface (fixed {slider_label})",
        width=900,
        height=800,
        margin=dict(l=0, r=0, t=80, b=160),
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title="Predicted logPe",
        ),
        sliders=[{
            "active": 0,
            "pad": {"t": 100, "b": 20},
            "currentvalue": {"prefix": f"{slider_label}: "},
            "steps": steps
        }],
    )

    return fig


def make_plotly_surface_static(smiles, fixed_var, fixed_value, num_points=35):
    """
    views.py에서 직접 사용하지 않더라도 기존 호환성 유지를 위해 보존.
    """
    fixed_var = fixed_var.lower()

    if fixed_var == "dmso":
        x_name, y_name = "lec", "ph"
        xs = np.linspace(LEC_MIN, LEC_MAX, num_points)
        ys = np.linspace(PH_MIN, PH_MAX, num_points)
        X, Y = np.meshgrid(xs, ys)
        Z = np.zeros_like(X, dtype=float)

        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                Z[i, j] = predict_single(smiles, float(X[i, j]), float(Y[i, j]), float(fixed_value))

    elif fixed_var == "lec":
        x_name, y_name = "ph", "dmso"
        xs = np.linspace(PH_MIN, PH_MAX, num_points)
        ys = np.linspace(DMSO_MIN, DMSO_MAX, num_points)
        X, Y = np.meshgrid(xs, ys)
        Z = np.zeros_like(X, dtype=float)

        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                Z[i, j] = predict_single(smiles, float(fixed_value), float(X[i, j]), float(Y[i, j]))

    else:
        x_name, y_name = "lec", "dmso"
        xs = np.linspace(LEC_MIN, LEC_MAX, num_points)
        ys = np.linspace(DMSO_MIN, DMSO_MAX, num_points)
        X, Y = np.meshgrid(xs, ys)
        Z = np.zeros_like(X, dtype=float)

        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                Z[i, j] = predict_single(smiles, float(X[i, j]), float(fixed_value), float(Y[i, j]))

    fig = go.Figure(data=[
        go.Surface(x=X, y=Y, z=Z, showscale=False, opacity=0.95)
    ])

    fig.update_layout(
        scene=dict(
            xaxis_title=x_name,
            yaxis_title=y_name,
            zaxis_title="logPe",
        ),
        margin=dict(l=0, r=0, t=20, b=0),
        title=f"3D Surface (fixed {fixed_var}={fixed_value})"
    )

    return fig
