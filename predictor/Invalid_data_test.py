import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

input_path = r"C:\chemweb_project\mysite\predictor\GI_PAMPA_del.csv"
df = pd.read_csv(input_path)

smiles_df = df[["SMILES"]].dropna().copy()

# 1) RDKit mol 생성 (실패하면 None)
smiles_df["mol"] = smiles_df["SMILES"].map(lambda s: Chem.MolFromSmiles(str(s)))

# 2) 실패/성공 분리
bad = smiles_df[smiles_df["mol"].isna()].copy()
good = smiles_df[smiles_df["mol"].notna()].copy()

print(f"Total: {len(smiles_df)}  Valid: {len(good)}  Invalid: {len(bad)}")

# (선택) 깨진 SMILES 저장
bad[["SMILES"]].to_csv("invalid_smiles.csv", index=False)

def calc_physchem_from_mol(mol):
    return pd.Series({
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
    })

# 3) 성공한 mol만 descriptor 계산 (여기서 절대 None 안 섞임)
physchem_df = good["mol"].apply(calc_physchem_from_mol)

# 4) PCA
X = physchem_df.values
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=3, random_state=0)
X_pca = pca.fit_transform(X_scaled)

# loadings
loadings = pd.DataFrame(
    pca.components_.T,
    index=physchem_df.columns,
    columns=["PC1", "PC2", "PC3"]
)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print(loadings.sort_values("PC1", ascending=False))

# 5) PC1 acid/base score (정규화)
score = X_pca[:, 0]
score_norm = score / abs(score).max()

result_df = good[["SMILES"]].copy()
result_df["PC1_acid_base_score"] = score_norm

def interpret_score(x):
    if x > 0.3:
        return "Acidic"
    elif x < -0.3:
        return "Basic"
    else:
        return "Neutral"

result_df["AcidBaseClass"] = result_df["PC1_acid_base_score"].apply(interpret_score)

result_df.to_csv("pca_acid_base_score.csv", index=False)
print(result_df.head(10))
