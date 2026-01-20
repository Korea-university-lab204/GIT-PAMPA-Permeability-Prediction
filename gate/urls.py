from django.urls import path
from .views import gate_view

urlpatterns = [
    path("", gate_view, name="gate"),
]
