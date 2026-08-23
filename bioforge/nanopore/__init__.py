"""
bioforge.nanopore — basecaller de nanoporo desde cero (señal → bases).

Lectura de POD5/FAST5, detección de eventos, estimación del pore model y
decodificación Viterbi. NumPy puro: sin IA y sin GPU.
"""
from .basecaller import *              # noqa: F401,F403
from .basecaller import __all__ as _a
__all__ = list(_a) if isinstance(_a, (list, tuple)) else []


def __getattr__(name):
    """Compatibilidad: antes ``bioforge.nanopore`` era un módulo plano.

    Reenvía a :mod:`bioforge.nanopore.basecaller` lo que no exporte el paquete.
    """
    from . import basecaller as _bc
    try:
        return getattr(_bc, name)
    except AttributeError:
        raise AttributeError(
            f"module 'bioforge.nanopore' has no attribute {name!r}") from None
