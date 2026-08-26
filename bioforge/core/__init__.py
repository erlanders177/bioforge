"""
bioforge.core — el CIMIENTO: almacenamiento 5-bit, tipos y lectura de secuencias.

Aquí viven ``PackedSequence`` (las secuencias empaquetadas a 5 bits), el lector
``SmartImporter`` (FASTA/FASTQ, streaming y por lotes) y la jerarquía de errores.
Todo lo demás del motor se apoya en esto.

Carga perezosa, y por qué importa aquí más que en ningún sitio
--------------------------------------------------------------
Los **errores** viven en ``core.errors``, un módulo sin una sola dependencia; el
resto vive en ``core.biocore``, que carga NumPy y el motor C. Antes este
``__init__`` hacía ``from .biocore import *``, así que **pedir una excepción
cargaba NumPy entero**: cualquier herramienta de texto puro que solo quisiera
lanzar un ``SequenceValueError`` decente pagaba ~500 ms por siete ``class``.

Ahora se resuelve por PEP 562: las excepciones vienen de ``errors`` y no arrastran
nada. Es lo que permite que las enzimas de restricción o la temperatura de fusión
de un cebador arranquen al instante en un portátil modesto.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# La jerarquía de errores NO necesita NumPy: va en su propio módulo ligero.
_ERRORES = ("BioForgeError", "SequenceTypeError", "SequenceValueError",
            "TranslationError", "AlignmentError", "BioForgeIOError", "EngineError")

# Lo demás sí toca el almacenamiento 5-bit y el motor.
_PESADOS = ("AA_LUT", "NUC_LUT", "BioCode", "BitPacker", "FastqRecord",
            "PackedSequence", "SeqType", "SequenceStats", "SmartImporter",
            "compute_stats", "SequenceBatch", "ReadBatch")

_EXPORTS = {n: "errors" for n in _ERRORES}
_EXPORTS.update({n: "biocore" for n in _PESADOS})

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """Resuelve el símbolo cargando SOLO su módulo (PEP 562)."""
    modulo = _EXPORTS.get(name)
    if modulo is None:
        # puede ser un símbolo interno de biocore que no está en la lista pública
        import importlib
        mod = importlib.import_module(f"{__name__}.biocore")
        try:
            valor = getattr(mod, name)
        except AttributeError:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}") from None
        globals()[name] = valor
        return valor
    import importlib
    valor = getattr(importlib.import_module(f"{__name__}.{modulo}"), name)
    globals()[name] = valor              # se cachea: la próxima vez es directo
    return valor


def __dir__() -> list[str]:
    return __all__


if TYPE_CHECKING:                         # solo para editores y mypy
    from .biocore import (
        AA_LUT,
        NUC_LUT,
        BioCode,
        BitPacker,
        FastqRecord,
        PackedSequence,
        ReadBatch,
        SeqType,
        SequenceBatch,
        SequenceStats,
        SmartImporter,
        compute_stats,
    )
    from .errors import (
        AlignmentError,
        BioForgeError,
        BioForgeIOError,
        EngineError,
        SequenceTypeError,
        SequenceValueError,
        TranslationError,
    )
