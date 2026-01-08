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

    smiles_value = context["smiles_value"]
    fixed_var = context["fixed_var"]
    lec_value = float(context["lec_value"])
    ph_value = float(context["ph_value"])
    dmso_value = float(context["dmso_value"])
    single_pred = context["single_pred"]

    # ✅ fixed_var에 해당하는 '고정값' 결정
    if fixed_var == "dmso":
        fixed_value = dmso_value
        point_x, point_y = lec_value, ph_value
    elif fixed_var == "lec":
        fixed_value = lec_value
        point_x, point_y = ph_value, dmso_value
    else:  # ph
        fixed_value = ph_value
        point_x, point_y = lec_value, dmso_value

    # ✅ 3D surface 생성 + 단일조건 점 찍기
    from .surface_utils import make_plotly_surface_static
    fig = make_plotly_surface_static(smiles_value, fixed_var, fixed_value, num_points=30)

    # 마커 추가
    import plotly.graph_objects as go
    fig.add_trace(go.Scatter3d(
        x=[point_x], y=[point_y], z=[single_pred],
        mode="markers",
        marker=dict(size=6),
        name="Single condition"
    ))

    # ✅ kaleido로 PNG export (Chromium 필요 없음)
    png_bytes = fig.to_image(format="png", width=1100, height=700, scale=2)

    # ✅ PDF 생성 (ReportLab)
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    y = h - 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "GIT-PAMPA Permeability Report")
    y -= 18

    c.setFont("Helvetica", 9)
    c.drawString(40, y, f"SMILES: {smiles_value}")
    y -= 14
    c.drawString(40, y, f"Fixed var: {fixed_var} | Lec={lec_value:.2f}, pH={ph_value:.2f}, DMSO={dmso_value:.2f}")
    y -= 14
    c.drawString(40, y, f"Pred logPe (single): {float(single_pred):.3f}")
    y -= 18

    # 3D 그래프 삽입
    img = ImageReader(BytesIO(png_bytes))
    img_w = w - 80
    img_h = img_w * (700/1100)  # 비율 유지
    c.drawImage(img, 40, y - img_h, width=img_w, height=img_h, preserveAspectRatio=True, anchor='c')
    y -= (img_h + 18)

    # 모델 성능 / 민감도 / RDKit 간단 출력 (원하면 표로도 가능)
    meta = context["model_meta"]
    sens = context["sensitivity"]
    rd = context["rdkit_desc"]

    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Model Performance")
    y -= 14
    c.setFont("Helvetica", 9)
    c.drawString(40, y, f"R2={meta.get('r2', '')}  RMSE={meta.get('rmse', '')}  MAE={meta.get('mae','')}  MAPE={meta.get('mape','')}%")
    y -= 18

    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Local Sensitivity (around this condition)")
    y -= 14
    c.setFont("Helvetica", 9)
    c.drawString(40, y, f"Lec(+1): {getattr(sens, 'lec', sens.get('lec')):.3f}   pH(+0.1): {getattr(sens, 'ph', sens.get('ph')):.3f}   DMSO(+1): {getattr(sens, 'dmso', sens.get('dmso')):.3f}")
    y -= 18

    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "RDKit Descriptors")
    y -= 14
    c.setFont("Helvetica", 9)
    # rd가 dict/obj 둘 다 대응
    def _get(o, k, default=""):
        return getattr(o, k, o.get(k, default)) if o is not None else default

    c.drawString(40, y, f"MolWt={float(_get(rd,'MolWt',0)):.2f}  LogP={float(_get(rd,'LogP',0)):.2f}  TPSA={float(_get(rd,'TPSA',0)):.2f}")
    y -= 12
    c.drawString(40, y, f"HBD={_get(rd,'HBD','')}  HBA={_get(rd,'HBA','')}  RotB={_get(rd,'RotatableBonds','')}  Rings={_get(rd,'RingCount','')}")
    y -= 12

    c.showPage()
    c.save()

    pdf_bytes = buf.getvalue()
    buf.close()

    resp = HttpResponse(pdf_bytes, content_type="application/pdf")
    resp["Content-Disposition"] = 'attachment; filename=\"permeability_report.pdf\"'
    return resp
