from django.shortcuts import redirect
from django.urls import reverse

class GateMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        allowed_paths = [
            reverse("gate"),
            "/admin/",
            "/static/",
        ]

        if not request.session.get("gate_ok"):
            if not any(request.path.startswith(p) for p in allowed_paths):
                return redirect("gate")

        return self.get_response(request)
