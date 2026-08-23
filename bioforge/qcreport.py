"""
Puente de compatibilidad: ``bioforge.qcreport`` se movió a ``bioforge.io.qcreport``.

En la v10.1 el paquete se reorganizó **por funciones**. Esta ruta antigua sigue
funcionando para no romper el código que ya la usaba, pero está DESACONSEJADA:
usa ``bioforge.io.qcreport`` (o el nombre desde ``bioforge`` directamente, que no ha cambiado).
"""

import warnings as _warnings

from bioforge.io.qcreport import *          # noqa: F401,F403  (nombres públicos)
from bioforge.io import qcreport as _target

_warnings.warn(
    "bioforge.qcreport se movió a bioforge.io.qcreport; esta ruta seguirá funcionando una versión "
    "más. Cambia el import cuando puedas.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name):       # también reenvía lo no público
    return getattr(_target, name)


def __dir__():
    return dir(_target)
