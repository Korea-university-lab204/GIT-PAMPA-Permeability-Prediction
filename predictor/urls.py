from django.urls import path
from . import views

urlpatterns = [
    path("smiles-3d/", views.smiles_3d_view, name="smiles_3d"),
]
