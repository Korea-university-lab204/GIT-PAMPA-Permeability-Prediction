from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
import joblib
import plotly.graph_objects as go


# --------- 경로 & 상수 ---------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
META_PATH = BASE_DIR / "meta.pkl"

DESCRIPTOR_NAMES = (
    [f'PEOE_VSA{i}' for i in range(1, 8)] +
    [f'SlogP_VSA{i}' for i in range(1, 8)] +
    [f'SMR_VSA{i}' for i in range(1, 7)] +
    [f'AUTOCORR2D_{i}' for i in range(1, 7)] +
    ['BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW']
)

LEC_MIN, LEC_MAX = 1.0, 20.0
PH_MIN, PH_MAX = 5.5, 7.4
DMSO_MIN, DMSO_MAX = 0.5, 10.0

COND_COLS = {
    "lec": "Lecithin(%)",
    "ph": "pH",
    "dmso": "DMSO_Conc (w/v%)",
}

_model = None
_scaler = None
_input_columns = None

# meta.pkl에 r2_score가 없을 때 사용할 예비 값
# 실제 모델 성능(R^2)을 알고 있다면 여기를 수정해서 넣어도 됨.
MODEL_R2_FALLBACK = 0.0


# --------- 공통 유틸 ---------
def load_artifacts():
    global _model, _scaler, _input_columns
    if _model is not None:
        return _model, _scaler, _input_columns

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    meta = joblib.load(META_PATH)

    input_columns = meta["input_columns"]

    _model = model
    _scaler = scaler
    _input_columns = input_columns
    return _model, _scaler, _input_columns


def _mol_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol


def calc_descriptors_for_smiles(smiles: str):
    mol = _mol_from_smiles(smiles)
    values = []
    for name in DESCRIPTOR_NAMES:
        func = getattr(Descriptors, name)
        values.append(func(mol))
    return values


# --------- 모델 R^2 ---------
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

DATA_PATH = BASE_DIR / "GI_PAMPA.csv"
TARGET_COL = "logPe"   # ⚠️ 실제 타깃 컬럼명 확인해서 맞추기

def get_model_meta():
    try:
        meta = joblib.load(META_PATH)
        print(f"✅ meta 로딩 성공: {meta.keys()}")
        return meta
    except Exception as e:
        print(f"[get_model_meta ERROR] {e}")
        return {}


# --------- 단일 포인트 예측 / 감응도 ---------
def predict_single(smiles, lec, ph, dmso):
    model, scaler, input_columns = load_artifacts()
    desc = calc_descriptors_for_smiles(smiles)

    cond = {
        COND_COLS["lec"]: lec,
        COND_COLS["ph"]: ph,
        COND_COLS["dmso"]: dmso,
    }

    full = pd.DataFrame([cond | dict(zip(DESCRIPTOR_NAMES, desc))])
    full = full.reindex(columns=input_columns)

    X_scaled = scaler.transform(full)
    pred = model.predict(X_scaled)[0]
    return pred


def compute_local_sensitivity(smiles, lec, ph, dmso):
    """
    현재 조건(lec, ph, dmso)에서
    - Lec +1
    - pH +0.1
    - DMSO +1
    변동 시 logPe 변화량(ΔlogPe)을 계산
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


# --------- RDKit 기본 특성 (오른쪽 아래 박스용) ---------
def get_basic_rdkit_descriptors(smiles: str):
    """
    자주 쓰이는 RDKit 특성 몇 개를 dict 로 반환
    """
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


# --------- 3D surface 계산 ---------
def get_surface_for_smiles(smiles, fixed_var="dmso", fixed_value=0.5, num_points=30):
    fixed_var = fixed_var.lower()
    model, scaler, input_columns = load_artifacts()

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
        lec_flat, ph_flat, dmso_flat = X.ravel(), Y.ravel(), np.full(N, fixed_value)
    elif fixed_var == "lec":
        dmso_flat, ph_flat, lec_flat = X.ravel(), Y.ravel(), np.full(N, fixed_value)
    else:
        dmso_flat, lec_flat, ph_flat = X.ravel(), Y.ravel(), np.full(N, fixed_value)

    desc = calc_descriptors_for_smiles(smiles)
    desc_df = pd.DataFrame([desc] * N, columns=DESCRIPTOR_NAMES)

    cond_df = pd.DataFrame({
        COND_COLS["lec"]: lec_flat,
        COND_COLS["ph"]: ph_flat,
        COND_COLS["dmso"]: dmso_flat,
    })

    full = pd.concat([cond_df, desc_df], axis=1).reindex(columns=input_columns)
    X_scaled = scaler.transform(full)
    preds = model.predict(X_scaled)

    Z = preds.reshape(X.shape)
    return X, Y, Z, x_label, y_label


# --------- Plotly 그래프 + 슬라이더 ---------
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
        smiles=smiles, fixed_var=fixed_var, fixed_value=first_val, num_points=num_points
    )

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z0,
        colorscale="Inferno",
        colorbar=dict(title="logPe"),
    ))

    frames = []
    for v in slider_values:
        _, _, Z, _, _ = get_surface_for_smiles(
            smiles, fixed_var=fixed_var, fixed_value=v, num_points=num_points
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
    fixed_var 하나는 fixed_value로 고정하고,
    나머지 2개 축에 대해 surface(z=pred logPe)를 만든 뒤
    단일조건 점(마커)을 찍은 '정적' 3D fig 생성
    """
    # 축 범위는 기존 상수 사용
    if fixed_var == "dmso":
        x_name, y_name = "lec", "ph"
        x_min, x_max = LEC_MIN, LEC_MAX
        y_min, y_max = PH_MIN, PH_MAX
        dmso_fixed = fixed_value

        xs = np.linspace(x_min, x_max, num_points)
        ys = np.linspace(y_min, y_max, num_points)
        X, Y = np.meshgrid(xs, ys)

        Z = np.zeros_like(X, dtype=float)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                Z[i, j] = predict_single(smiles, float(X[i, j]), float(Y[i, j]), float(dmso_fixed))

    elif fixed_var == "lec":
        x_name, y_name = "ph", "dmso"
        x_min, x_max = PH_MIN, PH_MAX
        y_min, y_max = DMSO_MIN, DMSO_MAX
        lec_fixed = fixed_value

        xs = np.linspace(x_min, x_max, num_points)
        ys = np.linspace(y_min, y_max, num_points)
        X, Y = np.meshgrid(xs, ys)

        Z = np.zeros_like(X, dtype=float)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                Z[i, j] = predict_single(smiles, float(lec_fixed), float(X[i, j]), float(Y[i, j]))

    else:  # fixed_var == "ph"
        x_name, y_name = "lec", "dmso"
        x_min, x_max = LEC_MIN, LEC_MAX
        y_min, y_max = DMSO_MIN, DMSO_MAX
        ph_fixed = fixed_value

        xs = np.linspace(x_min, x_max, num_points)
        ys = np.linspace(y_min, y_max, num_points)
        X, Y = np.meshgrid(xs, ys)

        Z = np.zeros_like(X, dtype=float)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                Z[i, j] = predict_single(smiles, float(X[i, j]), float(ph_fixed), float(Y[i, j]))

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
