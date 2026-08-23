"""
refindex.py
══════════════════════════════════════════════════════════════════════
Índice de una referencia por minimizers — Fase 2 del alineador de
genomas (v3.0).

Coge un genoma, le saca todos sus minimizers y los guarda en una tabla
ordenada por hash. Consultar un read es entonces una búsqueda binaria
vectorizada (``np.searchsorted``) sobre esa tabla: en vez de recorrer el
genoma, saltas directo a las posiciones candidatas — como el índice al
final de un libro.

Optimización de rendimiento
───────────────────────────
``max_occ`` descarta los minimizers hiper-frecuentes (los que caen en
repeticiones y aparecen miles de veces). Son ruido caro: multiplican las
anclas sin aportar señal. minimap2 hace lo mismo. Es la clase de atajo
que prima la velocidad sobre la exhaustividad.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

from .minimizers import encode_bases, minimizers


class LookupResult(NamedTuple):
    """Coincidencias de un lote de hashes contra el índice (expandidas).

    query_idx     : int64  — a qué hash de entrada corresponde cada coincidencia.
    ref_positions : int64  — posición del minimizer en la referencia.
    ref_strands   : uint8  — hebra (0/1) con que se registró en la referencia.
    """
    query_idx:     np.ndarray
    ref_positions: np.ndarray
    ref_strands:   np.ndarray

    def __len__(self) -> int:
        return int(self.ref_positions.size)


class ReferenceIndex:
    """Índice de minimizers de una referencia, consultable por hash."""

    def __init__(self, codes: np.ndarray, k: int = 15, w: int = 10,
                 max_occ: Optional[int] = None):
        sk = minimizers(codes, k=k, w=w)

        # Ordenar por hash → búsqueda binaria posterior con searchsorted.
        order = np.argsort(sk.hashes, kind="stable")
        hashes = sk.hashes[order]
        positions = sk.positions[order]
        strands = sk.strands[order]

        # Filtrar minimizers hiper-frecuentes (repeticiones).
        if max_occ is not None and hashes.size:
            _uniq, inv, cnts = np.unique(hashes, return_inverse=True,
                                         return_counts=True)
            keep = cnts[inv] <= max_occ
            hashes, positions, strands = hashes[keep], positions[keep], strands[keep]

        self.hashes = hashes
        self.positions = positions
        self.strands = strands
        self.k = k
        self.w = w
        self.ref_len = int(codes.size)
        self.max_occ = max_occ

    @classmethod
    def from_sequence(cls, seq, k: int = 15, w: int = 10,
                      max_occ: Optional[int] = None) -> "ReferenceIndex":
        """Construye el índice desde una secuencia (str/bytes)."""
        return cls(encode_bases(seq), k=k, w=w, max_occ=max_occ)

    @property
    def n_minimizers(self) -> int:
        return int(self.hashes.size)

    def lookup(self, query_hashes: np.ndarray) -> LookupResult:
        """Busca un lote de hashes y devuelve TODAS sus coincidencias.

        Todo vectorizado: dos ``searchsorted`` dan los rangos [lo, hi) de
        cada hash, y una expansión "ragged" (repeat + offsets) reúne las
        coincidencias sin un solo bucle Python sobre las consultas.
        """
        qh = np.ascontiguousarray(query_hashes, dtype=np.uint64)
        nq = qh.size
        if nq == 0 or self.hashes.size == 0:
            return LookupResult(np.empty(0, np.int64), np.empty(0, np.int64),
                                np.empty(0, np.uint8))

        lo = np.searchsorted(self.hashes, qh, side="left")
        hi = np.searchsorted(self.hashes, qh, side="right")
        counts = (hi - lo).astype(np.int64)
        total = int(counts.sum())
        if total == 0:
            return LookupResult(np.empty(0, np.int64), np.empty(0, np.int64),
                                np.empty(0, np.uint8))

        # Expansión ragged: para cada consulta i, sus entradas son [lo[i], hi[i]).
        query_idx = np.repeat(np.arange(nq, dtype=np.int64), counts)
        group_start = np.cumsum(counts) - counts          # inicio de cada grupo en la salida
        offsets = np.arange(total, dtype=np.int64) - np.repeat(group_start, counts)
        ref_entry = np.repeat(lo.astype(np.int64), counts) + offsets

        return LookupResult(query_idx, self.positions[ref_entry],
                            self.strands[ref_entry])
