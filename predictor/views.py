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

    # ✅ PDF는 3D 그래프 제외: 수치 리포트만 생성 (무료 Render 안정)
    from io import BytesIO
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4

    smiles_value = context["smiles_value"]
    fixed_var = context["fixed_var"]
    lec_value = float(context["lec_value"])
    ph_value = float(context["ph_value"])
    dmso_value = float(context["dmso_value"])
    single_pred = context["single_pred"]
    meta = context["model_meta"]
    sens = context["sensitivity"]
    rd = context["rdkit_desc"]

    def _get(o, k, default=""):
        if o is None:
            return default
        return getattr(o, k, o.get(k, default))

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    y = h - 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "GIT-PAMPA Permeability Report (No 3D Graph)")
    y -= 18

    c.setFont("Helvetica", 9)
    c.drawString(40, y, f"SMILES: {smiles_value}")
    y -= 14
    c.drawString(40, y, f"Fixed var: {fixed_var} | Lec={lec_value:.2f}, pH={ph_value:.2f}, DMSO={dmso_value:.2f}")
    y -= 14
    c.drawString(40, y, f"Pred logPe (single): {float(single_pred):.3f}")
    y -= 20

    # --- Model Performance ---
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Model Performance")
    y -= 14
    c.setFont("Helvetica", 9)
    c.drawString(
        40, y,
        f"R2={meta.get('r2','')}   RMSE={meta.get('rmse','')}   MAE={meta.get('mae','')}   MAPE={meta.get('mape','')}%"
    )
    y -= 20

    # --- Local Sensitivity ---
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Local Sensitivity (around this condition)")
    y -= 14
    c.setFont("Helvetica", 9)
    # sens가 dict/obj 둘 다 대응
    lec_s = getattr(sens, "lec", sens.get("lec")) if sens else 0
    ph_s = getattr(sens, "ph", sens.get("ph")) if sens else 0
    dmso_s = getattr(sens, "dmso", sens.get("dmso")) if sens else 0
    c.drawString(40, y, f"Lec(+1): {float(lec_s):.3f}   pH(+0.1): {float(ph_s):.3f}   DMSO(+1): {float(dmso_s):.3f}")
    y -= 20

    # --- RDKit ---
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "RDKit Descriptors")
    y -= 14
    c.setFont("Helvetica", 9)
    c.drawString(40, y, f"MolWt={float(_get(rd,'MolWt',0)):.2f}   LogP={float(_get(rd,'LogP',0)):.2f}   TPSA={float(_get(rd,'TPSA',0)):.2f}")
    y -= 12
    c.drawString(40, y, f"HBD={_get(rd,'HBD','')}   HBA={_get(rd,'HBA','')}   RotB={_get(rd,'RotatableBonds','')}   Rings={_get(rd,'RingCount','')}")
    y -= 12
    c.drawString(40, y, f"HeavyAtoms={_get(rd,'HeavyAtomCount','')}")

    c.showPage()
    c.save()

    pdf_bytes = buf.getvalue()
    buf.close()

    resp = HttpResponse(pdf_bytes, content_type="application/pdf")
    resp["Content-Disposition"] = 'attachment; filename="permeability_report.pdf"'
    return resp
