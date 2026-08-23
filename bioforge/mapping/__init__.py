"""
bioforge.mapping — mapeo de lecturas largas contra un genoma.

Minimizers canónicos, índice de la referencia y el mapeador seed-chain-align
(estilo minimap2, con la tubería entera en C y salida PAF).
"""
from .genomemap import GenomeAligner, Mapping     # noqa: F401
__all__ = ["GenomeAligner", "Mapping"]
