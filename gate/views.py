from django.conf import settings
from django.shortcuts import render, redirect

def gate_view(request):
    error = None

    if request.method == "POST":
        code = request.POST.get("code", "").strip()

        if code == settings.GATE_ACCESS_CODE:
            request.session["gate_ok"] = True
            return redirect("/")
        else:
            error = "접근 코드가 올바르지 않습니다."

    return render(request, "gate.html", {"error": error})
