"""
minimizers.py
══════════════════════════════════════════════════════════════════════
Muestreo de k-mers por *minimizers* (w, k) — Fase 1 del alineador de
genomas (v3.0). Es el cimiento de "seed-chain-align".

La idea
───────
Un genoma tiene demasiados k-mers para indexarlos todos. Un *minimizer*
es un muestreo inteligente: de cada ventana de ``w`` k-mers consecutivos
se guarda **solo el de menor hash**. Como la consulta (read) y la
referencia (genoma) usan la MISMA regla, siguen coincidiendo en los
mismos sitios, pero almacenas ~10× menos. Es el invento que hace viable
alinear reads largos contra genomas enteros.

Canónico (independiente de hebra)
─────────────────────────────────
Para cada k-mer se calcula su hash y el de su complementario inverso, y
se queda el **menor de los dos** (el "canónico"), anotando la hebra. Así
un read mapea igual venga de la hebra que venga — sin duplicar el índice.

Regla de oro respetada
──────────────────────
Cero bucles por símbolo. El único ``for`` es ``range(k)`` (Horner sobre
la longitud del k-mer, ~15): O(k) fijo y pequeño, operando sobre vectores
completos — análogo al bucle permitido de anti-diagonales del alineador.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .biocore import SequenceValueError
from .engine._loader import C_MINIMIZERS_AVAILABLE, c_minimizers

# ── LUT ASCII → código 2-bit (A=0, C=1, G=2, T=3; cualquier otra cosa = 4) ──────
_BASE_LUT = np.full(256, 4, dtype=np.uint8)
for _b, _v in ((b"A", 0), (b"C", 1), (b"G", 2), (b"T", 3)):
    _BASE_LUT[_b[0]] = _v
    _BASE_LUT[_b.lower()[0]] = _v

_SENTINEL = np.uint64(0xFFFFFFFFFFFFFFFF)   # hash imposible → nunca se elige


class MinimizerSketch(NamedTuple):
    """Minimizers de una secuencia.

    hashes    : uint64  — hash canónico de cada minimizer.
    positions : int64   — posición (0-based) del inicio del k-mer en la secuencia.
    strands   : uint8   — 0 si se eligió la hebra directa; 1 si la inversa.
    """
    hashes:    np.ndarray
    positions: np.ndarray
    strands:   np.ndarray

    def __len__(self) -> int:
        return int(self.positions.size)


def encode_bases(seq) -> np.ndarray:
    """Convierte una secuencia (str/bytes) a códigos 2-bit uint8.

    A/C/G/T → 0/1/2/3 (mayúsculas o minúsculas); cualquier otro carácter
    (N, ambigüedad, etc.) → 4, que marca la base como inválida.
    """
    if isinstance(seq, str):
        seq = seq.encode("ascii", "replace")
    arr = np.frombuffer(seq, dtype=np.uint8)
    return _BASE_LUT[arr]


def _hash64(x: np.ndarray, mask: np.uint64) -> np.ndarray:
    """Hash entero invertible de 64 bits (esquema de minimap2), vectorizado.

    Reparte los k-mers de forma uniforme para que el "menor hash" sea un
    muestreo pseudoaleatorio y no esté sesgado por la secuencia. Se opera
    en uint64: los desbordes envuelven mód 2^64 (justo lo que se quiere).
    """
    x = x & mask
    x = (~x + (x << np.uint64(21))) & mask
    x = x ^ (x >> np.uint64(24))
    x = ((x + (x << np.uint64(3))) + (x << np.uint64(8))) & mask
    x = x ^ (x >> np.uint64(14))
    x = ((x + (x << np.uint64(2))) + (x << np.uint64(4))) & mask
    x = x ^ (x >> np.uint64(28))
    x = (x + (x << np.uint64(31))) & mask
    return x


def minimizers(codes: np.ndarray, k: int = 15, w: int = 10) -> MinimizerSketch:
    """Calcula los (w, k)-minimizers canónicos de una secuencia.

    ``codes`` : array uint8 de códigos 2-bit (ver :func:`encode_bases`);
                los valores ≥ 4 marcan bases inválidas (N…).
    ``k``     : longitud del k-mer (1–31).
    ``w``     : nº de k-mers consecutivos por ventana (densidad ~ 2/(w+1)).
    """
    if not (1 <= k <= 31):
        raise SequenceValueError(f"k debe estar en 1..31 (se recibió {k}).")
    if w < 1:
        raise SequenceValueError(f"w debe ser ≥ 1 (se recibió {w}).")

    codes = np.ascontiguousarray(codes, dtype=np.uint8)
    # Motor C si está disponible (mucho más rápido en genomas grandes); si no,
    # el fallback NumPy, que produce EXACTAMENTE lo mismo (verificado en tests).
    if C_MINIMIZERS_AVAILABLE:
        h, pos, strand = c_minimizers(codes, k, w)
        return MinimizerSketch(h, pos, strand)
    return _minimizers_numpy(codes, k, w)


def _minimizers_numpy(codes: np.ndarray, k: int, w: int) -> MinimizerSketch:
    """Fallback NumPy de :func:`minimizers` (idéntico al C bio_minimizers)."""
    L = codes.size
    n = L - k + 1                       # nº de k-mers en la secuencia
    if n <= 0:
        empty_u64 = np.empty(0, dtype=np.uint64)
        return MinimizerSketch(empty_u64, np.empty(0, dtype=np.int64),
                               np.empty(0, dtype=np.uint8))

    valid = codes < 4
    c64  = np.where(valid, codes, 0).astype(np.uint64)   # base directa
    comp = np.uint64(3) - c64                            # complemento (A↔T, C↔G)

    # ── k-mer directo e inverso-complementario por Horner (O(k) vectorizado) ────
    fwd = np.zeros(n, dtype=np.uint64)
    rev = np.zeros(n, dtype=np.uint64)
    two = np.uint64(2)
    for j in range(k):
        fwd = (fwd << two) | c64[j : j + n]
        rev = (rev << two) | comp[(k - 1 - j) : (k - 1 - j) + n]

    # ── Validez de cada k-mer: todas sus bases deben ser ACGT (cumsum, O(L)) ────
    pref = np.concatenate(([0], np.cumsum(valid, dtype=np.int64)))
    kmer_valid = (pref[k:] - pref[:-k]) == k

    # ── Hash canónico: el menor entre hebra directa e inversa ───────────────────
    mask = np.uint64((1 << (2 * k)) - 1)
    hf = _hash64(fwd, mask)
    hr = _hash64(rev, mask)
    strand = (hr < hf).astype(np.uint8)
    h = np.minimum(hf, hr)
    h = np.where(kmer_valid, h, _SENTINEL)   # inválidos → nunca son el mínimo

    # ── Mínimo por ventana deslizante de w k-mers ───────────────────────────────
    if n >= w:
        win = sliding_window_view(h, w)          # (n-w+1, w) — vista, sin copia
        rel = win.argmin(axis=1)                 # índice del mínimo dentro de cada ventana
        pos = np.arange(n - w + 1, dtype=np.int64) + rel
    else:
        pos = np.array([int(h.argmin())], dtype=np.int64)   # una sola ventana

    pos = np.unique(pos)                         # ventanas solapadas repiten minimizer
    pos = pos[h[pos] != _SENTINEL]               # descartar ventanas todas-inválidas

    return MinimizerSketch(h[pos], pos, strand[pos])
