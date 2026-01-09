from django.shortcuts import render
from django.http import HttpResponse
from django.template.loader import render_to_string
from .surface_utils import get_model_meta
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader


from .surface_utils import (
    make_plotly_surface_with_slider,
    compute_local_sensitivity,
    get_basic_rdkit_descriptors,
    predict_single,
    LEC_MIN, LEC_MAX, PH_MIN, PH_MAX, DMSO_MIN, DMSO_MAX,
)


def smiles_3d_view(request):
    plot_html = None
    error = None
    smiles_value = ""
    fixed_var = "dmso"

    # 오른쪽 패널용
    sensitivity = None
    rdkit_desc = None
    single_pred = None
    model_meta = get_model_meta()

    # 단일 예측 입력값 기본값
    lec_value = (LEC_MIN + LEC_MAX) / 2
    ph_value = (PH_MIN + PH_MAX) / 2
    dmso_value = (DMSO_MIN + DMSO_MAX) / 2

    if request.method == "POST":

        # 1) 그래프 생성 버튼
        if "create_graph" in request.POST:
            smiles_value = request.POST.get("smiles", "").strip()
            fixed_var = request.POST.get("fixed_var", "dmso").strip().lower()

            if not smiles_value:
                error = "SMILES를 입력해 주세요."
            else:
                try:
                    fig = make_plotly_surface_with_slider(
                        smiles=smiles_value,
                        fixed_var=fixed_var,
                        num_points=25,
                        n_steps=10,
                    )
                    plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

                    rdkit_desc = get_basic_rdkit_descriptors(smiles_value)

                    base_lec = (LEC_MIN + LEC_MAX) / 2
                    base_ph = (PH_MIN + PH_MAX) / 2
                    base_dmso = (DMSO_MIN + DMSO_MAX) / 2

                    sensitivity = compute_local_sensitivity(
                        smiles_value, base_lec, base_ph, base_dmso
                    )

                    lec_value = base_lec
                    ph_value = base_ph
                    dmso_value = base_dmso

                except Exception as e:
                    error = f"오류 발생: {str(e)}"

        # 2) 단일 예측 Confirm 버튼
        elif "single_predict" in request.POST:
            smiles_value = request.POST.get("smiles_hidden", "").strip()
            fixed_var = request.POST.get("fixed_var_hidden", "dmso").strip().lower()

            lec_str = request.POST.get("lec_value", "").strip()
            ph_str = request.POST.get("ph_value", "").strip()
            dmso_str = request.POST.get("dmso_value", "").strip()

            if not smiles_value:
                error = "먼저 SMILES와 고정 변수를 선택해서 그래프를 생성해 주세요."
            else:
                try:
                    lec_value = float(lec_str)
                    ph_value = float(ph_str)
                    dmso_value = float(dmso_str)
                except ValueError:
                    error = "단일 예측용 조건값을 숫자로 입력해 주세요."

                lec_value = max(LEC_MIN, min(LEC_MAX, lec_value))
                ph_value = max(PH_MIN, min(PH_MAX, ph_value))
                dmso_value = max(DMSO_MIN, min(DMSO_MAX, dmso_value))

                if not error:
                    try:
                        fig = make_plotly_surface_with_slider(
                            smiles=smiles_value,
                            fixed_var=fixed_var,
                            num_points=25,
                            n_steps=10,
                        )
                        plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

                        single_pred = predict_single(
                            smiles_value, lec_value, ph_value, dmso_value
                        )

                        sensitivity = compute_local_sensitivity(
                            smiles_value, lec_value, ph_value, dmso_value
                        )

                        rdkit_desc = get_basic_rdkit_descriptors(smiles_value)

                    except Exception as e:
                        error = f"오류 발생: {str(e)}"

    return render(request, "predictor/smiles_plot.html", {
        "plot_html": plot_html,
        "smiles_value": smiles_value,
        "fixed_var": fixed_var,

        "sensitivity": sensitivity,
        "rdkit_desc": rdkit_desc,
        "single_pred": single_pred,

        "lec_value": lec_value,
        "ph_value": ph_value,
        "dmso_value": dmso_value,

        "model_meta": model_meta,
        "error": error,
        "is_pdf": False,
    })

# =========================
# ✅ PDF 전용 로직
# =========================

def _clamp_float(val, vmin, vmax, default):
    try:
        x = float(val)
    except (TypeError, ValueError):
        x = default
    return max(vmin, min(vmax, x))


def _build_context_for_pdf(post_data):
    """
    PDF 버튼에서 넘어온 값(현재 상태)을 기반으로
    plot/rdkit/sensitivity/single_pred 등을 생성
    """
    plot_html = None
    error = None

    smiles_value = (post_data.get("smiles") or "").strip()
    fixed_var = (post_data.get("fixed_var") or "dmso").strip().lower()

    default_lec = (LEC_MIN + LEC_MAX) / 2
    default_ph = (PH_MIN + PH_MAX) / 2
    default_dmso = (DMSO_MIN + DMSO_MAX) / 2

    lec_value = _clamp_float(post_data.get("lec_value"), LEC_MIN, LEC_MAX, default_lec)
    ph_value = _clamp_float(post_data.get("ph_value"), PH_MIN, PH_MAX, default_ph)
    dmso_value = _clamp_float(post_data.get("dmso_value"), DMSO_MIN, DMSO_MAX, default_dmso)

    sensitivity = None
    rdkit_desc = None
    single_pred = None
    model_meta = get_model_meta()

    if not smiles_value:
        error = "SMILES를 입력해 주세요."
    else:
        try:
            fig = make_plotly_surface_with_slider(
                smiles=smiles_value,
                fixed_var=fixed_var,
                num_points=25,
                n_steps=10,
            )
            # PDF는 외부 CDN 로딩 이슈 방지 위해 inline
            plot_html = fig.to_html(full_html=False, include_plotlyjs="inline")

            rdkit_desc = get_basic_rdkit_descriptors(smiles_value)
            sensitivity = compute_local_sensitivity(smiles_value, lec_value, ph_value, dmso_value)
            single_pred = predict_single(smiles_value, lec_value, ph_value, dmso_value)

        except Exception as e:
            error = f"PDF 생성 중 오류 발생: {str(e)}"

    return {
        "plot_html": plot_html,
        "smiles_value": smiles_value,
        "fixed_var": fixed_var,
        "sensitivity": sensitivity,
        "rdkit_desc": rdkit_desc,
        "single_pred": single_pred,
        "lec_value": lec_value,
        "ph_value": ph_value,
        "dmso_value": dmso_value,
        "model_meta": model_meta,
        "error": error,
        "is_pdf": True,
    }


def permeability_pdf(request):
    if request.method != "POST":
        return HttpResponse("Method Not Allowed", status=405)

    context = _build_context_for_pdf(request.POST)
    if context.get("error"):
        return HttpResponse(context["error"], status=400)

    # ===== 값 꺼내기 =====
    smiles_value = context["smiles_value"]
    fixed_var = context["fixed_var"]
    lec_value = float(context["lec_value"])
    ph_value = float(context["ph_value"])
    dmso_value = float(context["dmso_value"])
    single_pred = float(context["single_pred"])

    meta = context["model_meta"]        # dict
    sens = context["sensitivity"]       # dict or obj
    rd = context["rdkit_desc"]          # dict or obj

    # 안전 getter (dict/obj 겸용)
    def _get(o, k, default=""):
        if o is None:
            return default
        return getattr(o, k, o.get(k, default))

    def _fmt4(x, default=""):
        try:
            return f"{float(x):.4f}"
        except Exception:
            return default

    # ===== 3D 그래프 PNG 생성 (저메모리) =====
    # num_points=12 권장 (무료 Render 512MB 보호)
    graph_png = _make_static_3d_png(
        smiles=smiles_value,
        fixed_var=fixed_var,
        lec_value=lec_value,
        ph_value=ph_value,
        dmso_value=dmso_value,
        single_pred=single_pred,
        num_points=12,
    )

    # ===== PDF(표) 생성 =====
    from io import BytesIO
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=28, rightMargin=28, topMargin=28, bottomMargin=28)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("GIT-PAMPA Permeability Report", styles["Title"]))
    story.append(Spacer(1, 10))

    # (A) Input / Single prediction 테이블
    t1_data = [
        ["Item", "Value"],
        ["SMILES", smiles_value],
        ["Fixed variable", fixed_var],
        ["Lec", f"{lec_value:.2f}"],
        ["pH", f"{ph_value:.2f}"],
        ["DMSO", f"{dmso_value:.2f}"],
        ["Pred logPe (single)", f"{single_pred:.3f}"],
    ]
    t1 = Table(t1_data, colWidths=[140, 360])
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    story.append(t1)
    story.append(Spacer(1, 10))

    # (B) 3D 그래프 이미지 삽입 (고정변수=단일조건 값)
    story.append(Paragraph("3D Surface (static, fixed at current condition)", styles["Heading3"]))
    img = Image(BytesIO(graph_png), width=520, height=330)  # A4 폭 고려
    story.append(img)
    story.append(Spacer(1, 10))

    # (C) Model performance (소수점 4자리)
    t2_data = [
        ["Metric", "Value"],
        ["R2", _fmt4(meta.get("r2"))],
        ["RMSE", _fmt4(meta.get("rmse"))],
        ["MAE", _fmt4(meta.get("mae"))],
        ["MAPE (%)", _fmt4(meta.get("mape"))],
    ]
    t2 = Table(t2_data, colWidths=[140, 140])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    story.append(Paragraph("Model Performance", styles["Heading3"]))
    story.append(t2)
    story.append(Spacer(1, 10))

    # (D) Local sensitivity 테이블
    lec_s = _get(sens, "lec", 0)
    ph_s = _get(sens, "ph", 0)
    dmso_s = _get(sens, "dmso", 0)

    t3_data = [
        ["Variable", "Delta rule", "ΔlogPe"],
        ["Lec", "+1", f"{float(lec_s):.3f}"],
        ["pH", "+0.1", f"{float(ph_s):.3f}"],
        ["DMSO", "+1", f"{float(dmso_s):.3f}"],
    ]
    t3 = Table(t3_data, colWidths=[140, 140, 140])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    story.append(Paragraph("Local Sensitivity (around this condition)", styles["Heading3"]))
    story.append(t3)
    story.append(Spacer(1, 10))

    # (E) RDKit descriptors 테이블
    t4_data = [
        ["Descriptor", "Value"],
        ["MolWt", f"{float(_get(rd,'MolWt',0)):.2f}"],
        ["LogP", f"{float(_get(rd,'LogP',0)):.2f}"],
        ["TPSA", f"{float(_get(rd,'TPSA',0)):.2f}"],
        ["HBD", str(_get(rd, "HBD", ""))],
        ["HBA", str(_get(rd, "HBA", ""))],
        ["RotatableBonds", str(_get(rd, "RotatableBonds", ""))],
        ["RingCount", str(_get(rd, "RingCount", ""))],
        ["HeavyAtomCount", str(_get(rd, "HeavyAtomCount", ""))],
    ]
    t4 = Table(t4_data, colWidths=[180, 180])
    t4.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    story.append(Paragraph("RDKit Descriptors", styles["Heading3"]))
    story.append(t4)

    doc.build(story)

    pdf_bytes = buf.getvalue()
    buf.close()

    resp = HttpResponse(pdf_bytes, content_type="application/pdf")
    resp["Content-Disposition"] = 'attachment; filename="permeability_report.pdf"'
    return resp

def _make_static_3d_png(smiles, fixed_var, lec_value, ph_value, dmso_value, single_pred, num_points=12):
    """
    ✅ Chrome/Playwright/Kaleido 없이 Matplotlib로 3D surface를 PNG로 만듦.
    - fixed_var를 단일조건 값으로 고정
    - 나머지 2개 변수 평면에서 z=logPe surface 생성
    - 단일조건 점(마커) 표시
    - num_points는 12~15 권장(무료 Render 메모리 보호)
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # 서버에서 GUI 없이 렌더링
    import matplotlib.pyplot as plt
    from io import BytesIO

    # grid 범위
    if fixed_var == "dmso":
        # x=lec, y=ph, z=logPe (dmso fixed)
        fixed_value = dmso_value
        xs = np.linspace(LEC_MIN, LEC_MAX, num_points)
        ys = np.linspace(PH_MIN, PH_MAX, num_points)
        X, Y = np.meshgrid(xs, ys)

        Z = np.zeros_like(X, dtype=float)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                Z[i, j] = predict_single(smiles, float(X[i, j]), float(Y[i, j]), float(fixed_value))

        point_x, point_y = lec_value, ph_value
        xlab, ylab = "Lec", "pH"
        title = f"3D Surface (DMSO fixed={fixed_value:.2f})"

    elif fixed_var == "lec":
        # x=ph, y=dmso (lec fixed)
        fixed_value = lec_value
        xs = np.linspace(PH_MIN, PH_MAX, num_points)
        ys = np.linspace(DMSO_MIN, DMSO_MAX, num_points)
        X, Y = np.meshgrid(xs, ys)

        Z = np.zeros_like(X, dtype=float)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                Z[i, j] = predict_single(smiles, float(fixed_value), float(X[i, j]), float(Y[i, j]))

        point_x, point_y = ph_value, dmso_value
        xlab, ylab = "pH", "DMSO"
        title = f"3D Surface (Lec fixed={fixed_value:.2f})"

    else:  # fixed_var == "ph"
        # x=lec, y=dmso (ph fixed)
        fixed_value = ph_value
        xs = np.linspace(LEC_MIN, LEC_MAX, num_points)
        ys = np.linspace(DMSO_MIN, DMSO_MAX, num_points)
        X, Y = np.meshgrid(xs, ys)

        Z = np.zeros_like(X, dtype=float)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                Z[i, j] = predict_single(smiles, float(X[i, j]), float(fixed_value), float(Y[i, j]))

        point_x, point_y = lec_value, dmso_value
        xlab, ylab = "Lec", "DMSO"
        title = f"3D Surface (pH fixed={fixed_value:.2f})"

    # Matplotlib 3D plot
    fig = plt.figure(figsize=(7.2, 4.6), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    # surface (저메모리: rstride/cstride 생략, alpha 낮게)
    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, alpha=0.85)

    # 단일조건 점 표시
    ax.scatter([point_x], [point_y], [single_pred], s=35)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlab, fontsize=9)
    ax.set_ylabel(ylab, fontsize=9)
    ax.set_zlabel("logPe", fontsize=9)

    # 출력 PNG
    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)  # ✅ 메모리 회수 필수
    return buf.getvalue()
