"""
bioforge.lab — las herramientas del día a día en un laboratorio de biología molecular.

No son algoritmos de investigación: son las preguntas que alguien con una pipeta en
la mano se hace **antes de tocar nada**, y que hoy se resuelven a mano o con webs
que obligan a subir la secuencia a un servidor ajeno.

* ``restriction`` — ¿qué enzimas cortan mi secuencia, y en cuántos trozos? Sitios,
  digestión, fragmentos, cortadores únicos y simulación del gel.
* ``orf``         — ¿qué genes puede haber aquí? Marcos abiertos de lectura en los
  seis marcos, con su proteína.
* ``primers``     — ¿servirán estos cebadores? Temperatura de fusión por vecino más
  próximo, diseño de parejas y PCR *in silico*.

Contrastadas contra los estándares (regla nº12): las enzimas dan posiciones
**idénticas** a las de Biopython/REBASE (64/64), los ORFs coinciden al **100 %** con
``getorf`` de EMBOSS en sus dos modos, y la Tm coincide con Biopython **a precisión
de máquina**. Ver ``tools/bench_lab_vs_estandares.py``.

Carga perezosa (regla nº11): pedir una de las tres no carga las otras dos.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# símbolo público → módulo que lo define (de aquí salen a la vez __all__ y la
# resolución perezosa, así no se desincronizan nunca)
_EXPORTS = {
    # restricción
    "Enzyme": "restriction",
    "ENZYMES": "restriction",
    "get_enzyme": "restriction",
    "Site": "restriction",
    "Fragment": "restriction",
    "Digestion": "restriction",
    "find_sites": "restriction",
    "digest": "restriction",
    "unique_cutters": "restriction",
    "gel": "restriction",
    # ORFs
    "ORF": "orf",
    "find_orfs": "orf",
    "longest_orf": "orf",
    # cebadores
    "Primer": "primers",
    "Amplicon": "primers",
    "tm_nn": "primers",
    "tm_wallace": "primers",
    "gc_percent": "primers",
    "design_primers": "primers",
    "pcr": "primers",
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
    globals()[name] = valor
    return valor


def __dir__() -> list[str]:
    return __all__


if TYPE_CHECKING:                         # solo para editores y mypy
    from .orf import ORF, find_orfs, longest_orf
    from .primers import (
        Amplicon,
        Primer,
        design_primers,
        gc_percent,
        pcr,
        tm_nn,
        tm_wallace,
    )
    from .restriction import (
        ENZYMES,
        Digestion,
        Enzyme,
        Fragment,
        Site,
        digest,
        find_sites,
        gel,
        get_enzyme,
        unique_cutters,
    )
