"""
Puente de compatibilidad: ``bioforge.analyze`` se movió a ``bioforge.cli.analyze``.

En la v10.1 el paquete se reorganizó **por funciones**. Esta ruta antigua sigue
funcionando para no romper el código que ya la usaba, pero está DESACONSEJADA:
usa ``bioforge.cli.analyze`` (o el nombre desde ``bioforge`` directamente, que no ha cambiado).
"""

import warnings as _warnings

from bioforge.cli.analyze import *          # noqa: F401,F403  (nombres públicos)
from bioforge.cli import analyze as _target

_warnings.warn(
    "bioforge.analyze se movió a bioforge.cli.analyze; esta ruta seguirá funcionando una versión "
    "más. Cambia el import cuando puedas.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name):       # también reenvía lo no público
    return getattr(_target, name)


def __dir__():
    return dir(_target)
