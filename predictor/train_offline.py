import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error

import joblib  # pip install joblib 필요


# ----- 기본 설정 -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "GI_PAMPA.csv")

DESCRIPTOR_NAMES = (
    [f'PEOE_VSA{i}' for i in range(1, 8)] +
    [f'SlogP_VSA{i}' for i in range(1, 8)] +
    [f'SMR_VSA{i}' for i in range(1, 7)] +
    [f'AUTOCORR2D_{i}' for i in range(1, 7)] +
    ['BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW']
)


def calc_descriptors_for_smiles(smiles_series):
    """SMILES 시리즈에 대해 RDKit 디스크립터 계산."""
    descriptions = []

    for smile in smiles_series:
        mol = Chem.MolFromSmiles(smile)

        desc_values = []
        for desc_name in DESCRIPTOR_NAMES:
            try:
                func = getattr(Descriptors, desc_name)
                value = func(mol)
            except Exception:
                value = 0.0  # 실패 시 0으로 처리
            desc_values.append(value)

        descriptions.append(desc_values)

    return pd.DataFrame(descriptions, columns=DESCRIPTOR_NAMES)


def main():
    print("CSV 불러오는 중...")
    experiment_data = pd.read_csv(CSV_PATH)

    experiment_conditions = experiment_data[['Lecithin(%)', 'pH', 'DMSO_Conc (w/v%)']]
    smiles = experiment_data['SMILES'].astype(str)

    print("디스크립터 계산 중...")
    molecular_descriptors = calc_descriptors_for_smiles(smiles)

    # Input & Output (네가 원래 쓰던 방식 그대로)
    Input = pd.concat([experiment_conditions, molecular_descriptors], axis=1)
    Output = experiment_data['logPe']
    Data = pd.concat([Input, Output], axis=1).dropna()

    Input = Data.iloc[:, 0:33]
    Output = Data.iloc[:, 33]

    print("입력 크기:", Input.shape)
    print("타깃 크기:", Output.shape)

    # 조건 평균 / 컬럼 이름 저장용 (웹 예측에서 재사용 가능)
    cond_means = Input[['Lecithin(%)', 'pH', 'DMSO_Conc (w/v%)']].mean()

    print("Train/Test 분할 및 스케일링 중...")

    scaler = StandardScaler()
    X_train, X_test, Y_train, Y_test = train_test_split(
        Input, Output, test_size=0.2, random_state=42
    )

    # Scale the features
    X_scaled_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_scaled_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # Create and train SVR model (네 파라미터 그대로 적용)
    SVR_Model_logPe = SVR(kernel='rbf', C=4, epsilon=0.1)
    SVR_Model_logPe.fit(X_scaled_train, Y_train)

    # Make predictions
    Y_pred = SVR_Model_logPe.predict(X_scaled_test)

    # Calculate metrics
    rmse = (mean_squared_error(Y_test, Y_pred)) ** 0.5
    r2 = r2_score(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred)
    mape = (np.abs((Y_test - Y_pred) / Y_test).mean()) * 100  # MAPE (%)

    # Print results
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")

    # 학습에 사용된 feature 순서 저장 (중요!)
    feature_order = X_scaled_train.columns.tolist()

    print("모델/스케일러/메타데이터 저장 중...")
    joblib.dump(SVR_Model_logPe, os.path.join(BASE_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))
    meta = {
        "input_columns": feature_order,          # 나중에 ml_model에서 사용
        "cond_means": cond_means.to_dict(),      # (옵션) 조건 평균
        "rmse": float(rmse),
        "r2": float(r2),
    }
    joblib.dump(meta, os.path.join(BASE_DIR, "meta.pkl"))

    print("=== 완료! ===")
    print("model.pkl, scaler.pkl, meta.pkl 이 predictor 폴더에 저장되었습니다.")


if __name__ == "__main__":
    main()
