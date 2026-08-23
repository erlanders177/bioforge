"""
bioforge.ai — extras OPCIONALES basados en modelos de lenguaje de proteínas.

Este subpaquete NO forma parte del núcleo NumPy: requiere ``pip install bioforge[ai]``
(torch + transformers). El motor y el predictor funcionan sin él; esto es la palanca
de arriba (el eje B: viabilidad/gramaticalidad de mutaciones con ESM-2).

Importar aquí no arrastra torch salvo que uses las funciones (carga perezosa).
"""

from .viability import grammaticality_profile, viability_scores

__all__ = ["viability_scores", "grammaticality_profile"]
