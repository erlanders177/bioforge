"""
Puente de compatibilidad: ``bioforge.ai`` se movió a ``bioforge.evolution.ai``.

En la v10.1 el paquete se reorganizó por funciones y el eje opcional ESM-2 pasó a
vivir junto al resto de la evolución. Esta ruta sigue funcionando, pero está
DESACONSEJADA: usa ``bioforge.evolution.ai``.
"""

import warnings as _warnings

from bioforge.evolution import ai as _target
from bioforge.evolution.ai import grammaticality_profile, viability_scores  # noqa: F401

_warnings.warn(
    "bioforge.ai se movió a bioforge.evolution.ai; esta ruta seguirá funcionando "
    "una versión más. Cambia el import cuando puedas.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["viability_scores", "grammaticality_profile"]


def __getattr__(name):
    return getattr(_target, name)
