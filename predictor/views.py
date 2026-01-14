from django.shortcuts import render
from django.http import HttpResponse
from django.template.loader import render_to_string
from .surface_utils import get_model_meta
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
import base64
from io import BytesIO


from .surface_utils import (
    make_plotly_surface_with_slider,
    compute_local_sensitivity,
    get_basic_rdkit_descriptors,
    predict_single,
    LEC_MIN, LEC_MAX, PH_MIN, PH_MAX, DMSO_MIN, DMSO_MAX,
)


from pathlib import Path
from django.conf import settings
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle, Paragraph, Spacer, KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4

KU_CRIMSON = colors.HexColor("#8b0000")
INK = colors.HexColor("#111827")
MUTED = colors.HexColor("#6b7280")
BORDER = colors.HexColor("#e5e7eb")

def _static_file(*parts) -> str:
    # settings.BASE_DIR 기준: /app/static/img/...
    return str(Path(settings.BASE_DIR) / "static" / Path(*parts))

def _draw_header_footer(canvas, doc):
    """모든 페이지에 헤더/푸터 그리기"""
    canvas.saveState()
    page_w, page_h = A4

    # 헤더 영역
    header_h = 18 * mm
    canvas.setFillColor(colors.white)
    canvas.rect(0, page_h - header_h, page_w, header_h, stroke=0, fill=1)

    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.7)
    canvas.line(14*mm, page_h - header_h, page_w - 14*mm, page_h - header_h)

    # 로고 2개 (좌측)
    x = 14 * mm
    y = page_h - header_h + 3.2*mm

    ku_logo = _static_file("img", "ku_logo.png")
    pharm_logo = _static_file("img", "ku_pharm_logo.png")

    # ✅ 로고 크기: 여기서 조절
    ku_h = 12 * mm
    pharm_h = 12 * mm

    try:
        canvas.drawImage(ku_logo, x, y, height=ku_h, width=ku_h*2.2, preserveAspectRatio=True, mask="auto")
        x += 28*mm
        canvas.drawImage(pharm_logo, x, y, height=pharm_h, width=pharm_h*2.0, preserveAspectRatio=True, mask="auto")
    except Exception:
        # 로고 로딩 실패해도 PDF 생성은 계속
        pass

    # 우측 타이틀
    canvas.setFillColor(INK)
    canvas.setFont("Helvetica-Bold", 11)
    canvas.drawRightString(page_w - 14*mm, page_h - 7.2*mm, "Git-PAMPA Report")

    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8.5)
    canvas.drawRightString(page_w - 14*mm, page_h - 12.8*mm, "Korea University · College of Pharmacy · Permeability Prediction")

    # 푸터
    footer_y = 12 * mm
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.7)
    canvas.line(14*mm, footer_y + 6*mm, page_w - 14*mm, footer_y + 6*mm)

    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8.5)
    canvas.drawString(14*mm, footer_y, "Korea University College of Pharmacy · Git-PAMPA")
    canvas.drawRightString(page_w - 14*mm, footer_y, f"Page {doc.page}")

    canvas.restoreState()

def _make_card(title: str, inner_flowables, styles, width=170*mm):
    """섹션을 카드 박스처럼 감싸는 helper"""
    title_p = Paragraph(title, styles["CARD_H"])
    content = [title_p, Spacer(1, 6)] + inner_flowables

    t = Table([[content]], colWidths=[width])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.white),
        ("BOX", (0,0), (-1,-1), 0.8, BORDER),
        ("LEFTPADDING", (0,0), (-1,-1), 12),
        ("RIGHTPADDING", (0,0), (-1,-1), 12),
        ("TOPPADDING", (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
    ]))
    return KeepTogether([t, Spacer(1, 10)])


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

    mol_png_b64 = ""
    if smiles_value:
        mol_png_b64 = smiles_to_mol_png_b64(smiles_value, size=(520, 260))

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

        "mol_png_b64": mol_png_b64,
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
    def _get(o, k, default=None):
        if o is None:
            return default
        # dict면 get, obj면 getattr
        if hasattr(o, "get"):
            return o.get(k, default)
        return getattr(o, k, default)

    def _fmt4(x, default="-"):
        try:
            return f"{float(x):.4f}"
        except Exception:
            return default

    # ✅ PDF에서 plotly 생성은 안 쓰므로(무거움) _build_context_for_pdf도 추후 제거 권장
    # ===== 3D 그래프 PNG 생성 (저메모리) =====
    graph_png = _make_static_3d_png(
        smiles=smiles_value,
        fixed_var=fixed_var,
        lec_value=lec_value,
        ph_value=ph_value,
        dmso_value=dmso_value,
        single_pred=single_pred,
        num_points=9,   # ✅ 12 → 9로 낮춰서 안정화 추천
    )

    # ===== PDF 생성 =====
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=14*mm, rightMargin=14*mm,
        topMargin=26*mm, bottomMargin=18*mm
    )

    base_styles = getSampleStyleSheet()
    styles = {}
    styles["P"] = ParagraphStyle("P", parent=base_styles["Normal"], fontName="Helvetica", fontSize=10, leading=14, textColor=INK)
    styles["MUTED"] = ParagraphStyle("MUTED", parent=base_styles["Normal"], fontName="Helvetica", fontSize=9, leading=13, textColor=MUTED)
    styles["CARD_H"] = ParagraphStyle("CARD_H", parent=base_styles["Heading3"], fontName="Helvetica-Bold", fontSize=11, textColor=KU_CRIMSON, spaceAfter=0)

    story = []
    story.append(Spacer(1, 6))

    # -----------------------------
    # (A) Summary 카드
    # -----------------------------
    t1_data = [
        ["Item", "Value"],
        ["SMILES", smiles_value],
        ["Fixed variable", fixed_var],
        ["Lec", f"{lec_value:.2f}"],
        ["pH", f"{ph_value:.2f}"],
        ["DMSO", f"{dmso_value:.2f}"],
        ["Pred logPe (single)", f"{single_pred:.3f}"],
    ]
    t1 = Table(t1_data, colWidths=[45*mm, 120*mm])
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f5f6f8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), INK),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fbfbfc")]),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(_make_card("Summary", [t1], styles))

    # -----------------------------
    # (B) Molecule 2D 카드
    # -----------------------------
    mol_png = smiles_to_mol_png_bytes(smiles_value, size=(520, 220))
    mol_flows = []
    if mol_png:
        mol_flows.append(Image(BytesIO(mol_png), width=150*mm, height=55*mm))
        mol_flows.append(Spacer(1, 6))
        mol_flows.append(Paragraph("2D molecular structure (RDKit)", styles["MUTED"]))
    else:
        mol_flows.append(Paragraph("Molecule Structure (2D): invalid SMILES", styles["MUTED"]))
    story.append(_make_card("Molecule Structure (2D)", mol_flows, styles))

    # -----------------------------
    # (C) 3D Surface 카드
    # -----------------------------
    story.append(_make_card(
        "3D Surface (static, fixed at current condition)",
        [Image(BytesIO(graph_png), width=170*mm, height=108*mm)],
        styles
    ))

    story.append(PageBreak())

    # -----------------------------
    # (D) Model Performance 카드
    # -----------------------------
    t2_data = [
        ["Metric", "Value"],
        ["R²", _fmt4(_get(meta, "r2"))],
        ["RMSE", _fmt4(_get(meta, "rmse"))],
        ["MAE", _fmt4(_get(meta, "mae"))],
        ["MAPE (%)", _fmt4(_get(meta, "mape"))],
    ]
    t2 = Table(t2_data, colWidths=[55*mm, 55*mm])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f5f6f8")),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fbfbfc")]),
    ]))
    story.append(_make_card("Model Performance", [t2], styles))

    # -----------------------------
    # (E) Local Sensitivity 카드
    # -----------------------------
    lec_s = _get(sens, "lec")
    ph_s = _get(sens, "ph")
    dmso_s = _get(sens, "dmso")

    t3_data = [
        ["Variable", "Delta rule", "ΔlogPe"],
        ["Lec", "+1", f"{float(lec_s):.3f}" if lec_s is not None else "-"],
        ["pH", "+0.1", f"{float(ph_s):.3f}" if ph_s is not None else "-"],
        ["DMSO", "+1", f"{float(dmso_s):.3f}" if dmso_s is not None else "-"],
    ]
    t3 = Table(t3_data, colWidths=[55*mm, 55*mm, 55*mm])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f5f6f8")),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fbfbfc")]),
    ]))
    story.append(_make_card("Local Sensitivity (Around this condition)", [t3], styles))

    # -----------------------------
    # (F) RDKit Descriptors 카드
    # -----------------------------
    t4_data = [
        ["Descriptor", "Value"],
        ["MolWt", f"{float(_get(rd,'MolWt',0)):.2f}" if _get(rd,'MolWt') is not None else "-"],
        ["LogP", f"{float(_get(rd,'LogP',0)):.2f}" if _get(rd,'LogP') is not None else "-"],
        ["TPSA", f"{float(_get(rd,'TPSA',0)):.2f}" if _get(rd,'TPSA') is not None else "-"],
        ["HBD", str(_get(rd, "HBD", "-"))],
        ["HBA", str(_get(rd, "HBA", "-"))],
        ["RotatableBonds", str(_get(rd, "RotatableBonds", "-"))],
        ["RingCount", str(_get(rd, "RingCount", "-"))],
        ["HeavyAtomCount", str(_get(rd, "HeavyAtomCount", "-"))],
    ]
    t4 = Table(t4_data, colWidths=[70*mm, 95*mm])
    t4.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f5f6f8")),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fbfbfc")]),
    ]))
    story.append(_make_card("RDKit Descriptors", [t4], styles))

    # ✅ 헤더/푸터 적용해서 빌드
    doc.build(
        story,
        onFirstPage=lambda c, d: _draw_header_footer(c, d),
        onLaterPages=lambda c, d: _draw_header_footer(c, d),
    )

    pdf_bytes = buf.getvalue()
    buf.close()

    resp = HttpResponse(pdf_bytes, content_type="application/pdf")
    resp["Content-Disposition"] = 'attachment; filename="permeability_report.pdf"'
    return resp

def smiles_to_mol_png_b64(smiles: str, size=(520, 260)) -> str:
    """
    SMILES -> RDKit 2D 구조 PNG를 base64 문자열로 반환 (웹페이지 표시용)
    실패 시 "" 반환
    """
    try:
        import base64
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""

        w, h = size
        drawer = rdMolDraw2D.MolDraw2DCairo(int(w), int(h))
        opts = drawer.drawOptions()
        opts.centerMolecules = True
        opts.padding = 0.04  # 여백

        drawer.ClearDrawing()
        rdMolDraw2D.PrepareMolForDrawing(mol)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()

        png_bytes = drawer.GetDrawingText()
        return base64.b64encode(png_bytes).decode("utf-8")
    except Exception:
        return ""

def _make_static_3d_png(smiles, fixed_var, lec_value, ph_value, dmso_value, single_pred, num_points=9):
    """
    ✅ Matplotlib로 3D surface PNG(bytes) 생성 (Render에서도 동작)
    - fixed_var를 단일조건 값으로 고정
    - 나머지 2개 변수 평면에서 z=logPe surface 생성
    - 단일조건 점(마커) 표시
    - num_points는 8~12 권장 (무료 Render 안정화: 9 추천)
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # 서버에서 GUI 없이 렌더링
    import matplotlib.pyplot as plt
    from io import BytesIO

    # grid 범위
    if fixed_var == "dmso":
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

    fig = plt.figure(figsize=(7.2, 4.6), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, alpha=0.85)
    ax.scatter([point_x], [point_y], [single_pred], s=35)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlab, fontsize=9)
    ax.set_ylabel(ylab, fontsize=9)
    ax.set_zlabel("logPe", fontsize=9)

    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)  # ✅ 메모리 회수
    return buf.getvalue()
