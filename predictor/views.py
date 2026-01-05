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
        # ✅ 컨테이너/저메모리 환경 안정 옵션
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--single-process",
            ],
        )
        page = browser.new_page(viewport={"width": 1400, "height": 900})

        # ✅ networkidle은 오래 걸려서 worker timeout 유발 가능 → load로 변경
        page.set_content(html, wait_until="load")

        # ✅ 렌더링 안정화용 아주 짧은 대기
        page.wait_for_timeout(300)  # 0.3초

        page.emulate_media(media="print")

        pdf_bytes = page.pdf(
            format="A4",
            print_background=True,
            margin={"top": "10mm", "right": "10mm", "bottom": "10mm", "left": "10mm"},
        )

        # ✅ 메모리 회수
        page.close()
        browser.close()

    resp = HttpResponse(pdf_bytes, content_type="application/pdf")
    resp["Content-Disposition"] = 'attachment; filename="permeability_report.pdf"'
    return resp
