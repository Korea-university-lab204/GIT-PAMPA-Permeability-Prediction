from django.urls import path
from . import views

urlpatterns = [
    path("", views.smiles_3d_view, name="smiles_3d"),
    path("smiles-3d/", views.smiles_3d_view, name="smiles_3d"),
    path("permeability/pdf/", views.permeability_pdf, name="permeability_pdf"),
]
