import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 플롯 활성화용, 사용 안 해도 import 필요

from ml_model import predict_permeability  # 장고에서 쓰던 예측 함수 재사용


# 학습 데이터 기준 범위
LEC_MIN, LEC_MAX = 1.0, 20.0      # Lecithin (%)
PH_MIN, PH_MAX   = 5.5, 7.4       # pH
DMSO_MIN, DMSO_MAX = 0.5, 10.0    # DMSO (w/v%)


def plot_surface(smiles, fixed_var="dmso", fixed_value=0.5, num_points=50):
    """
    fixed_var: "dmso", "lec", "ph" 중 하나
    fixed_value: 고정값
    num_points: 그리드 분해수 (기본 50 → 50x50 = 2500포인트)
    """

    fixed_var = fixed_var.lower()

    # 1) 각 변수 범위 정의
    lec_range  = np.linspace(LEC_MIN, LEC_MAX, num_points)
    ph_range   = np.linspace(PH_MIN, PH_MAX, num_points)
    dmso_range = np.linspace(DMSO_MIN, DMSO_MAX, num_points)

    # 2) 어떤 변수를 고정하느냐에 따라 X,Y 축 설정
    if fixed_var == "dmso":
        # DMSO 고정 → (Lec, pH) 축
        X_vals = lec_range
        Y_vals = ph_range
        x_label = "Lecithin (%)"
        y_label = "pH"
        title = f"logPe surface (fixed DMSO={fixed_value:.2f}%)"

        # 그리드 생성
        X, Y = np.meshgrid(X_vals, Y_vals)

        # 예측값 계산
        Z = np.zeros_like(X)
        for i in range(num_points):
            for j in range(num_points):
                lec_val = X[i, j]
                ph_val = Y[i, j]

                pred = predict_permeability(
                    [smiles],
                    lecithin_pct=lec_val,
                    pH=ph_val,
                    dmso_conc=fixed_value,
                )[0]
                Z[i, j] = pred

    elif fixed_var == "lec":
        # Lecithin 고정 → (DMSO, pH) 축
        X_vals = dmso_range
        Y_vals = ph_range
        x_label = "DMSO_Conc (w/v%)"
        y_label = "pH"
        title = f"logPe surface (fixed Lecithin={fixed_value:.2f}%)"

        X, Y = np.meshgrid(X_vals, Y_vals)
        Z = np.zeros_like(X)
        for i in range(num_points):
            for j in range(num_points):
                dmso_val = X[i, j]
                ph_val = Y[i, j]

                pred = predict_permeability(
                    [smiles],
                    lecithin_pct=fixed_value,
                    pH=ph_val,
                    dmso_conc=dmso_val,
                )[0]
                Z[i, j] = pred

    elif fixed_var == "ph":
        # pH 고정 → (DMSO, Lec) 축
        X_vals = dmso_range
        Y_vals = lec_range
        x_label = "DMSO_Conc (w/v%)"
        y_label = "Lecithin (%)"
        title = f"logPe surface (fixed pH={fixed_value:.2f})"

        X, Y = np.meshgrid(X_vals, Y_vals)
        Z = np.zeros_like(X)
        for i in range(num_points):
            for j in range(num_points):
                dmso_val = X[i, j]
                lec_val = Y[i, j]

                pred = predict_permeability(
                    [smiles],
                    lecithin_pct=lec_val,
                    pH=fixed_value,
                    dmso_conc=dmso_val,
                )[0]
                Z[i, j] = pred

    else:
        raise ValueError("fixed_var는 'dmso', 'lec', 'ph' 중 하나여야 합니다.")

    # 3) 3D 플롯
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap="viridis")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel("Predicted logPe")
    ax.set_title(title)

    fig.colorbar(surf, shrink=0.5, aspect=10, label="logPe")

    plt.show()


if __name__ == "__main__":
    smiles = input("SMILES 문자열을 입력하세요: ").strip()

    print("고정할 변수를 선택하세요: (dmso / lec / ph)")
    fixed_var = input("fixed_var = ").strip().lower()

    # 고정값 기본 추천: dmso=0.5, lec=1, ph=7.4
    if fixed_var == "dmso":
        default_val = 0.5
        msg = f"고정할 DMSO 값({DMSO_MIN}~{DMSO_MAX}, 기본 {default_val}): "
    elif fixed_var == "lec":
        default_val = 1.0
        msg = f"고정할 Lecithin 값({LEC_MIN}~{LEC_MAX}, 기본 {default_val}): "
    elif fixed_var == "ph":
        default_val = 7.4
        msg = f"고정할 pH 값({PH_MIN}~{PH_MAX}, 기본 {default_val}): "
    else:
        print("잘못된 fixed_var 입니다. (dmso / lec / ph 중 하나)")
        exit(1)

    val_str = input(msg).strip()
    fixed_value = float(val_str) if val_str else default_val

    # grid 해상도는 50으로 시작
    plot_surface(smiles, fixed_var=fixed_var, fixed_value=fixed_value, num_points=50)
