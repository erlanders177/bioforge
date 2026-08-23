"""
Puente de compatibilidad: ``bioforge.msa`` se movió a ``bioforge.align.msa``.

En la v10.1 el paquete se reorganizó **por funciones**. Esta ruta antigua sigue
funcionando para no romper el código que ya la usaba, pero está DESACONSEJADA:
usa ``bioforge.align.msa`` (o el nombre desde ``bioforge`` directamente, que no ha cambiado).
"""

import warnings as _warnings

from bioforge.align.msa import *          # noqa: F401,F403  (nombres públicos)
from bioforge.align import msa as _target

_warnings.warn(
    "bioforge.msa se movió a bioforge.align.msa; esta ruta seguirá funcionando una versión "
    "más. Cambia el import cuando puedas.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name):       # también reenvía lo no público
    return getattr(_target, name)


def __dir__():
    return dir(_target)
