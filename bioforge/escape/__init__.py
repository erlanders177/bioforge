"""Eje B — escape a anticuerpos (carga perezosa, Regla #11).

Pedir el veredicto no carga la quimica hasta que hace falta, y nada de esta
familia arrastra NumPy: son consultas por mutacion sobre entradas pequenas.
"""
from typing import TYPE_CHECKING

_EXPORTS = {
    "AMINOACIDOS": "bioforge.escape.chemistry",
    "NO_PREDICE_PROPAGACION": "bioforge.escape.chemistry",
    "escape_score": "bioforge.escape.chemistry",
    "score_sitio": "bioforge.escape.chemistry",
    "percentil_en_sitio": "bioforge.escape.chemistry",
    "EscapeVerdict": "bioforge.escape.verdict",
    "CALIBRACION": "bioforge.escape.verdict",
    "MITADES": "bioforge.escape.verdict",
    "evaluar": "bioforge.escape.verdict",
    "evaluar_muchas": "bioforge.escape.verdict",
    "probabilidad_de_propagacion": "bioforge.escape.verdict",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    modulo = _EXPORTS.get(name)
    if modulo is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    valor = getattr(importlib.import_module(modulo), name)
    globals()[name] = valor
    return valor


def __dir__():
    return __all__


if TYPE_CHECKING:                                    # solo para el editor
    from .chemistry import (AMINOACIDOS, NO_PREDICE_PROPAGACION, escape_score,
                            percentil_en_sitio, score_sitio)
    from .verdict import (CALIBRACION, MITADES, EscapeVerdict, evaluar, evaluar_muchas,
                          probabilidad_de_propagacion)
