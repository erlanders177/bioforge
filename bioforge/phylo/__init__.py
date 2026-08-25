"""
bioforge.phylo — árboles evolutivos (filogenia por distancias).

Completa el frente evolutivo: sobre el MSA que ya construye ``bioforge.align``,
esta familia responde a **quién desciende de quién**.

    secuencias → align_multiple → distance_matrix → neighbor_joining → Newick

Dos piezas separables:

* ``distance``  — matrices de distancia con corrección de sustituciones múltiples
  (Jukes-Cantor, Kimura 2 parámetros, Poisson). Vale sola: comparar cómo de
  parecidas son unas secuencias no exige construir ningún árbol.
* ``tree``      — Neighbor-Joining, UPGMA, salida Newick y **soporte por
  bootstrap** (cuánta confianza merece cada rama).

Carga perezosa (regla de oro nº11): pedir ``distance_matrix`` **no** carga el
constructor de árboles. Cada herramienta se activa solo cuando se usa.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# símbolo público → módulo que lo define (de este mapa salen a la vez __all__ y
# la resolución perezosa, así no se desincronizan nunca)
_EXPORTS = {
    "DistanceMatrix": "distance",
    "distance_matrix": "distance",
    "Clade": "tree",
    "Tree": "tree",
    "neighbor_joining": "tree",
    "upgma": "tree",
    "wpgma": "tree",
    "build_tree": "tree",
    "bootstrap_support": "tree",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """Resuelve el símbolo importando SOLO su módulo (PEP 562)."""
    modulo = _EXPORTS.get(name)
    if modulo is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    mod = importlib.import_module(f"{__name__}.{modulo}")
    valor = getattr(mod, name)
    globals()[name] = valor              # se cachea: la próxima vez es directo
    return valor


def __dir__() -> list[str]:
    return __all__


if TYPE_CHECKING:                         # solo para editores y mypy, nunca en runtime
    from .distance import DistanceMatrix, distance_matrix
    from .tree import (
        Clade,
        Tree,
        bootstrap_support,
        build_tree,
        neighbor_joining,
        upgma,
        wpgma,
    )
