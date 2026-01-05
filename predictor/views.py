from .surface_utils import get_model_meta
from django.shortcuts import render
from django.http import HttpResponse
from django.template.loader import render_to_string

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
        "is_pdf": False,  # ✅ 템플릿에서 PDF 여부 판단용
    })


# =========================
# ✅ PDF 전용 로직 추가
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
    smiles_3d_view와 동일하게 plot/rdkit/sensitivity/single_pred 등을 생성
    """
    plot_html = None
    error = None

    smiles_value = (post_data.get("smiles") or "").strip()
    fixed_var = (post_data.get("fixed_var") or "dmso").strip().lower()

    # 기본값(뷰와 동일)
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
            # ✅ PDF에서는 CDN 로딩 실패/느림 방지용으로 inline 권장
            # (그래프가 PDF에 빈칸으로 나오는 문제를 줄여줌)
            plot_html = fig.to_html(full_html=False, include_plotlyjs="inline")

            rdkit_desc = get_basic_rdkit_descriptors(smiles_value)

            # 민감도는 PDF 버튼이 "현재 상태"를 보내니 그 값 기준으로
            sensitivity = compute_local_sensitivity(smiles_value, lec_value, ph_value, dmso_value)

            # 단일예측 값도 현재 입력값 기준으로 계산해서 PDF에 포함
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
        "is_pdf": True,  # ✅ 템플릿에서 입력폼/버튼 숨김
    }


def permeability_pdf(request):
    """
    HTML의 'PDF 다운로드' 버튼이 POST로 보내는 값을 받아
    동일 화면을 렌더링한 뒤 PDF로 변환하여 다운로드.
    """
    if request.method != "POST":
        return HttpResponse("Method Not Allowed", status=405)

    context = _build_context_for_pdf(request.POST)

    # ✅ 네 템플릿 파일 그대로 사용
    html = render_to_string("predictor/smiles_plot.html", context, request=request)

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.set_content(html, wait_until="networkidle")
        page.emulate_media(media="print")

        pdf_bytes = page.pdf(
            format="A4",
            print_background=True,
            margin={"top": "10mm", "right": "10mm", "bottom": "10mm", "left": "10mm"},
        )
        browser.close()

    resp = HttpResponse(pdf_bytes, content_type="application/pdf")
    resp["Content-Disposition"] = 'attachment; filename="permeability_report.pdf"'
    return resp
