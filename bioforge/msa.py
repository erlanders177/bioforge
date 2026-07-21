"""
msa.py
══════════════════════════════════════════════════════════════════════
Alineamiento múltiple de secuencias (MSA) — método CENTER-STAR.

El alineamiento múltiple óptimo es NP-difícil, así que todos usan heurísticas.
Aquí implementamos **center-star**, la más simple y directa, ideal para conjuntos
de secuencias MUY parecidas (p. ej. el mismo gen de distintas cepas de un virus a
lo largo de los años — el caso del predictor de evolución):

  1. Se elige una secuencia CENTRAL (la más larga; heurística barata).
  2. Se alinea **cada** secuencia contra la central por pares (nuestro NW en C).
  3. Se FUSIONAN todas propagando los huecos: para cada posición de la central se
     reserva el máximo de columnas de inserción que pida cualquier alineamiento,
     y cada secuencia se proyecta sobre ese "esqueleto" común.

Resultado: todas las secuencias alineadas, misma longitud, columnas homólogas.

Relación con los pros: los alineadores serios (Clustal Omega, MAFFT, MUSCLE) usan
alineamiento PROGRESIVO (árbol guía + profile-profile) y refinamiento iterativo,
mejores para secuencias divergentes. Center-star es el escalón honesto de partida
para secuencias parecidas; el progresivo queda como upgrade futuro.

Regla de oro: el trabajo por símbolo (las alineaciones por pares) ocurre en C.
Los bucles de este módulo son POR SECUENCIA / POR COLUMNA (no por símbolo en la
ruta crítica), análogos al traceback permitido del alineador.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

from .aligner import SequenceAligner
from .biocore import SeqType, SequenceTypeError, SequenceValueError, SmartImporter


class MSAResult(NamedTuple):
    """Resultado de un alineamiento múltiple.

    aligned : lista de cadenas alineadas (todas de la misma longitud, con '-').
    center  : índice de la secuencia usada como central.
    length  : longitud del alineamiento (nº de columnas).
    """
    aligned: list[str]
    center: int
    length: int

    def __len__(self) -> int:
        return len(self.aligned)

    def consensus(self, threshold: float = 0.5) -> str:
        """Secuencia consenso: en cada columna, el símbolo más frecuente (sin
        contar huecos) si supera ``threshold``; si no, 'N'. Huecos → se omiten."""
        if not self.aligned:
            return ""
        cols = zip(*self.aligned, strict=True)   # todas las filas de igual longitud
        out = []
        n = len(self.aligned)
        for col in cols:
            counts: dict[str, int] = {}
            for ch in col:
                if ch != "-":
                    counts[ch] = counts.get(ch, 0) + 1
            if not counts:
                continue                      # columna toda huecos → se omite
            best = max(counts, key=counts.get)
            out.append(best if counts[best] >= threshold * n else "N")
        return "".join(out)


def _pack(seq: str):
    return SmartImporter.from_string(f">x\n{seq}\n",
                                     force_type=SeqType.NUCLEOTIDE)[0]


def _pairwise(a, b: str) -> tuple[str, str]:
    """Alineamiento global por pares (NW en C). Devuelve (a_alineada, b_alineada).

    ``a`` puede venir ya empaquetada (PackedSequence): en center-star la central se
    alinea contra las n−1 restantes, así que empaquetarla una sola vez evita n−1
    re-empaquetados idénticos (era el 2× de llamadas a _pack que vio el perfil)."""
    pa = a if not isinstance(a, str) else _pack(a)
    # band="auto": banda adaptativa — EXACTA (se ensancha si el camino roza el
    # borde) pero varias veces más rápida gracias al SIMD banded del motor C.
    # En un MSA las secuencias son parecidas, que es justo su mejor caso.
    res = SequenceAligner.align(pa, _pack(b), mode="global", band="auto",
                                detect_mutations=False)
    return res.aligned_a, res.aligned_b


def align_multiple(sequences, center: Optional[int] = None) -> MSAResult:
    """Alinea múltiples secuencias por el método center-star.

    ``sequences`` : iterable de cadenas (ADN). Se pasan a mayúsculas.
    ``center``    : índice de la secuencia central; si None, se usa la más larga.

    Lanza ``SequenceTypeError`` si algún elemento no es str, ``SequenceValueError``
    si no hay secuencias o alguna está vacía.
    """
    seqs = list(sequences)
    if not seqs:
        raise SequenceValueError("No hay secuencias que alinear.")
    for s in seqs:
        if not isinstance(s, str):
            raise SequenceTypeError(
                f"cada secuencia debe ser str, se recibió {type(s).__name__!r}.")
    seqs = [s.upper() for s in seqs]
    if any(len(s) == 0 for s in seqs):
        raise SequenceValueError("Ninguna secuencia puede estar vacía.")

    n = len(seqs)
    if n == 1:
        return MSAResult(aligned=[seqs[0]], center=0, length=len(seqs[0]))

    # ── 1. secuencia central (la más larga; ties → la primera) ──────────────────
    ci = center if center is not None else max(range(n), key=lambda i: len(seqs[i]))
    if not (0 <= ci < n):
        raise SequenceValueError(f"center fuera de rango: {ci} (n={n}).")
    C = seqs[ci]
    L = len(C)

    # ── 2. alinear cada secuencia contra la central; extraer inserciones y columnas
    # per_seq[i] = (ins_chars, col_char):
    #   ins_chars[p] (p=0..L)   : lista de residuos de i insertados ANTES de C[p]
    #   col_char[p] (p=0..L-1)  : el símbolo de i alineado a C[p] (residuo o '-')
    per_seq: list[tuple[list[list[str]], list[str]]] = []
    C_packed = _pack(C)                              # empaquetar el centro UNA vez
    for i, S in enumerate(seqs):
        if i == ci:
            per_seq.append(([[] for _ in range(L + 1)], list(C)))
            continue
        aC, aS = _pairwise(C_packed, S)
        ins_chars: list[list[str]] = [[] for _ in range(L + 1)]
        col_char: list[str] = ["-"] * L
        p = 0
        for cc, sc in zip(aC, aS, strict=True):
            if cc == "-":                     # inserción de S respecto a C
                ins_chars[p].append(sc)
            else:                             # cc == C[p]
                col_char[p] = sc
                p += 1
        per_seq.append((ins_chars, col_char))

    # ── 3. columnas de inserción por hueco: el máximo que pida cualquiera ────────
    master_ins = [0] * (L + 1)
    for ins_chars, _ in per_seq:
        for p in range(L + 1):
            if len(ins_chars[p]) > master_ins[p]:
                master_ins[p] = len(ins_chars[p])

    # ── 4. construir cada fila proyectada sobre el esqueleto común ──────────────
    rows: list[str] = []
    for ins_chars, col_char in per_seq:
        row: list[str] = []
        for p in range(L):
            slot = ins_chars[p]
            row.extend(slot)
            row.extend("-" * (master_ins[p] - len(slot)))
            row.append(col_char[p])
        slot = ins_chars[L]                   # inserciones tras el último C
        row.extend(slot)
        row.extend("-" * (master_ins[L] - len(slot)))
        rows.append("".join(row))

    return MSAResult(aligned=rows, center=ci, length=len(rows[0]))
