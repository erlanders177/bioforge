"""
bioforge.core — el CIMIENTO: almacenamiento 5-bit, tipos y lectura de secuencias.

Aquí viven ``PackedSequence`` (las secuencias empaquetadas a 5 bits), el lector
``SmartImporter`` (FASTA/FASTQ, streaming y por lotes) y la jerarquía de errores.
Todo lo demás del motor se apoya en esto.
"""
from .biocore import *          # noqa: F401,F403
from .biocore import __all__ as _a
__all__ = list(_a) if isinstance(_a, (list, tuple)) else []
