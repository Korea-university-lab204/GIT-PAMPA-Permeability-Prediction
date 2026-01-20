from django.urls import path
from predictor.views import smiles_3d_view, permeability_pdf
from django.urls import path, include

urlpatterns = [
    path("gate/", include("gate.urls")),
    path("", smiles_3d_view, name="smiles_3d"),
    path("smiles-3d/", smiles_3d_view, name="smiles_3d"),
    path("permeability/pdf/", permeability_pdf, name="permeability_pdf"),
]

