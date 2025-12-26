import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # ✅ plt 에러 방지

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========================
# 1. CSV 로드
# =========================
INPUT_CSV = r"C:\chemweb_project\mysite\predictor\GI_PAMPA_del.csv"
OUTPUT_CSV = "GI_PAMPA_PCA_acid_base_scored.csv"
OUTPUT_PNG = "GI_PAMPA_PC1_axis_plot.png"  # ✅ 그래프도 파일로 저장(선택)

df = pd.read_csv(INPUT_CSV)

# =========================
# 2. SMILES 클린 & 필터
# =========================
def clean_smiles_safe(s):
    """끝의 ? 제거 정도만 안전하게 수행하고, RDKit 파싱 실패면 None"""
    if not isinstance(s, str):
        return None
    s = s.strip()
    s = re.sub(r"\?+$", "", s)  # 끝에 붙은 ?/?? 제거
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

df["SMILES_clean"] = df["SMILES"].apply(clean_smiles_safe)
df_valid = df[df["SMILES_clean"].notna()].copy()

print(f"Total: {len(df)}  Valid: {len(df_valid)}")

# =========================
# 3. PhysChem descriptor
# =========================
FEATURES = ["MolWt", "LogP", "TPSA", "HBD", "HBA", "HeavyAtomCount", "RingCount", "RotB"]

def calc_physchem(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return pd.Series({
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        "RingCount": Lipinski.RingCount(mol),
        "RotB": Lipinski.NumRotatableBonds(mol),
    })

physchem_df = df_valid["SMILES_clean"].apply(calc_physchem)

# 혹시라도 None이 섞이면 제거(안전장치)
physchem_df = physchem_df.dropna()

# df_valid도 physchem_df 인덱스에 맞춰 정렬
df_valid = df_valid.loc[physchem_df.index].copy()

# =========================
# 4. PCA (scaler/pca 객체를 반드시 변수로 유지)
# =========================
X = physchem_df[FEATURES].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3, random_state=0)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)

# =========================
# 5. PC1 부호 고정 (acid + / basic -)
# =========================
loadings = pd.Series(pca.components_[0], index=FEATURES)

# 극성/산성 관련 지표(TPSA/HBD/HBA)가 +가 되도록 sign 고정
polar_keys = ["TPSA", "HBD", "HBA"]
sign = np.sign(loadings[polar_keys].sum())
if sign == 0:
    sign = 1

pc1_signed = X_pca[:, 0] * sign
score = pc1_signed / np.max(np.abs(pc1_signed))  # -1~+1 정규화

# =========================
# 6. 결과 저장
# =========================
def interpret_score(x):
    if x > 0.3:
        return "Acidic"
    elif x < -0.3:
        return "Basic"
    else:
        return "Neutral"

df_valid["PC1_acid_base_score"] = score
df_valid["AcidBaseClass"] = df_valid["PC1_acid_base_score"].apply(interpret_score)

df_valid.to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV}")

# =========================
# 7. PC1 loading 출력
# =========================
print("\nPC1 loadings (sorted by abs):")
print(loadings.reindex(loadings.abs().sort_values(ascending=False).index))

# =========================
# 8) 5개 기준 물질 SMILES
# =========================
ref_smiles = {
    "Chloramphenicol": "C1=CC(=CC=C1[C@H]([C@@H](CO)NC(=O)C(Cl)Cl)O)[N+](=O)[O-]",
    "Diclofenac": "C1=CC=C(C(=C1)CC(=O)O)NC2=C(C=CC=C2Cl)Cl",
    "Piroxicam": "CN1C(=C(C2=CC=CC=C2S1(=O)=O)O)C(=O)NC3=CC=CC=N3",
    "Verapamil HCl": "CC(C)C(CCCN(C)CCC1=CC(=C(C=C1)OC)OC)(C#N)C2=CC(=C(C=C2)OC)OC.Cl",
    "Lidocaine": "CCN(CC)CC(=O)NC1=C(C=CC=C1C)C",
}

def physchem_row_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        "RingCount": Lipinski.RingCount(mol),
        "RotB": Lipinski.NumRotatableBonds(mol),
    }

ref_labels = []
ref_X_rows = []

for name, smi in ref_smiles.items():
    row = physchem_row_from_smiles(smi)
    if row is None:
        print(f"[WARN] invalid reference SMILES: {name}")
        continue
    # ✅ 학습에 사용한 FEATURES 순서 그대로 맞추기
    ref_X_rows.append([row[f] for f in FEATURES])
    ref_labels.append(name)

ref_X = np.array(ref_X_rows)

# =========================
# 9) 기존 scaler/pca로 기준물질 투영
# =========================
# ref_X가 비어있을 수도 있으니 방어
ref_score = None
if len(ref_X) > 0:
    ref_X_scaled = scaler.transform(ref_X)
    ref_pca = pca.transform(ref_X_scaled)
    ref_pc1_signed = ref_pca[:, 0] * sign
    ref_score = ref_pc1_signed / np.max(np.abs(pc1_signed))  # 전체 분포 기준 정규화

# =========================
# 10) Plot: x=PC1 score, y=0에 일렬 배치
# =========================
plt.figure(figsize=(12, 3))
plt.scatter(df_valid["PC1_acid_base_score"], np.zeros(len(df_valid)), alpha=0.25)
plt.yticks([])
plt.xlabel("PC1 acid(+)/basic(-) score")
plt.title("Compounds placed along PC1 (acid/basic axis)")
plt.axvline(0, linestyle="--")

y_offsets = np.linspace(0.05, 0.25, len(ref_labels))
colors = ["red", "green", "purple", "orange", "black"]

for name, x, y, c in zip(ref_labels, ref_score, y_offsets, colors):
    plt.scatter([x], [y], s=180, color=c, zorder=5)
    plt.text(x, y + 0.015, f"{name}\n({x:+.2f})", ha="center", fontsize=9)


plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=200)  # ✅ 창이 안 떠도 파일로 남김
plt.show()

print(f"Saved plot: {OUTPUT_PNG}")
