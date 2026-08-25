"""
bioforge.variants — de lecturas mapeadas a mutaciones (llamada de variantes).

Cierra la tubería del análisis de secuenciación:

    FASTQ → GenomeAligner (mapeo) → pileup (evidencia) → call_variants → VCF

Dos piezas, separables a propósito:

* ``pileup``  — apila las lecturas sobre la referencia y da la **profundidad** de
  cobertura. Vale por sí solo para responder «¿he leído bastante?».
* ``caller``  — decide qué diferencias son mutaciones reales y no ruido, con una
  razón de verosimilitudes binomial, y escribe VCF.

Carga perezosa (regla de oro nº11): pedir ``Pileup`` **no** carga el llamador, y
al revés. Cada herramienta se activa solo cuando se usa.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# símbolo público → módulo que lo define (misma mecánica que bioforge/__init__.py:
# de este mapa salen a la vez __all__ y la resolución perezosa, así no se
# desincronizan nunca)
_EXPORTS = {
    "Pileup": "pileup",
    "pileup": "pileup",
    "pileup_from_mappings": "pileup",
    "Variant": "caller",
    "call_variants": "caller",
    "write_vcf": "caller",
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
    from .caller import Variant, call_variants, write_vcf
    from .pileup import Pileup, pileup, pileup_from_mappings
