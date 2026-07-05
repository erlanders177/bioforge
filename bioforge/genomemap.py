"""
genomemap.py
══════════════════════════════════════════════════════════════════════
Alineador de genomas / mapeador de reads largos (v3.0) — el buque
insignia: "seed-chain-align" completo, la estrategia de minimap2/BWA.

Fases (todas aquí montadas sobre minimizers.py + refindex.py):
  3. SEEDING   — anclas (read↔referencia) en ambas hebras.
  4. CHAINING  — DP sobre anclas → mejor cadena colineal.
  5. EXTENSIÓN — alineamiento base a base (banded) en la región de la cadena.
  6. API       — GenomeAligner.map(read) → lista de Mapping (formato tipo PAF).

Idea central
────────────
No se alinea "todo contra todo" (imposible a escala de genoma). Se buscan
anclas (k-mers compartidos), se encadenan las colineales para localizar la
región, y solo ahí se corre el DP exacto en una banda estrecha.

Diagonal
────────
Para anclas en hebra directa, diagonal = ref_pos − read_pos: las que la
comparten están en línea. Para hebra inversa se transforma la coordenada
del read (yc = (Lr−k) − read_pos) para que el mismo chaining valga igual.

Nota de rendimiento
───────────────────
El chaining es un DP sobre ANCLAS (no por símbolo): un bucle acotado por
ventana de predecesores. Es el sitio natural para bajar a C más adelante;
en Python ya es correcto y rápido para el nº de anclas habitual.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import numpy as np

from .aligner import SequenceAligner
from .biocore import SeqType, SmartImporter
from .minimizers import encode_bases, minimizers
from .refindex import ReferenceIndex

_RC = str.maketrans("ACGTacgt", "TGCAtgca")


def _revcomp(s: str) -> str:
    return s.translate(_RC)[::-1]


# ══════════════════════════════════════════════════════════════════════════════
# §3  SEEDING — anclas
# ══════════════════════════════════════════════════════════════════════════════

class Anchors(NamedTuple):
    """Coincidencias k-mer read↔referencia. Todo vectorizado.

    ref_pos  : int64  — posición del k-mer en la referencia.
    read_pos : int64  — posición del k-mer en el read (coords. directas).
    strand   : uint8  — 0 = hebra directa, 1 = inversa (read↔ref).
    """
    ref_pos:  np.ndarray
    read_pos: np.ndarray
    strand:   np.ndarray
    read_len: int
    k:        int

    def __len__(self) -> int:
        return int(self.ref_pos.size)


def seed(index: ReferenceIndex, read_codes: np.ndarray) -> Anchors:
    """Genera las anclas de un read contra el índice (ambas hebras)."""
    mk = minimizers(read_codes, k=index.k, w=index.w)
    res = index.lookup(mk.hashes)
    q_pos = mk.positions[res.query_idx]
    q_str = mk.strands[res.query_idx]
    rel = (q_str ^ res.ref_strands).astype(np.uint8)   # hebra relativa
    return Anchors(res.ref_positions, q_pos, rel,
                   read_len=int(read_codes.size), k=index.k)


# ══════════════════════════════════════════════════════════════════════════════
# §4  CHAINING — DP sobre anclas
# ══════════════════════════════════════════════════════════════════════════════

class Chain(NamedTuple):
    score:      float
    strand:     int            # 0 directa, 1 inversa
    ref_start:  int
    ref_end:    int
    read_start: int
    read_end:   int
    n_anchors:  int
    anchor_ref:  np.ndarray    # posiciones ref de las anclas (ordenadas)
    anchor_read: np.ndarray    # posiciones read de las anclas


# Parámetros de chaining (razonables; ajustables).
_MAX_GAP = 5000      # distancia máxima entre anclas consecutivas
_WINDOW  = 64        # nº de predecesores que se examinan por ancla
_GAP_W   = 0.2       # peso lineal de la penalización por hueco
_MIN_ANCHORS = 2


def _chain_one(x: np.ndarray, y: np.ndarray, k: int,
               min_score: float) -> list[tuple]:
    """DP de chaining sobre un conjunto de anclas de UNA hebra.

    x = ref_pos, y = coordenada de chaining (ambas crecen a lo largo de una
    alineación válida). Devuelve caminos (listas de índices) por score desc.
    """
    n = x.size
    if n == 0:
        return []
    order = np.lexsort((y, x))          # ordenar por x, luego y
    xs, ys = x[order], y[order]

    f = np.full(n, float(k), dtype=np.float64)
    prev = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        xi, yi = xs[i], ys[i]
        best, bp = float(k), -1
        lo = max(0, i - _WINDOW)
        for j in range(i - 1, lo - 1, -1):
            dx = xi - xs[j]
            if dx > _MAX_GAP:
                break                    # xs ordenado → más atrás aún más lejos
            dy = yi - ys[j]
            if dy <= 0 or dx <= 0 or dy > _MAX_GAP:
                continue
            gap = abs(int(dx) - int(dy))
            if gap > _MAX_GAP:
                continue
            match = min(k, int(dx), int(dy))
            cost = _GAP_W * gap + (math.log2(gap + 1) if gap else 0.0)
            sc = f[j] + match - cost
            if sc > best:
                best, bp = sc, j
        f[i], prev[i] = best, bp

    # Backtrack de cadenas no solapadas, por score descendente.
    used = np.zeros(n, dtype=bool)
    paths: list[tuple] = []
    for start in np.argsort(-f):
        if f[start] < min_score:
            break
        if used[start]:
            continue
        path = []
        i = int(start)
        while i != -1 and not used[i]:
            used[i] = True
            path.append(i)
            i = int(prev[i])
        if len(path) >= _MIN_ANCHORS:
            path.reverse()                       # de inicio a fin
            paths.append((float(f[start]), order[np.array(path)]))
    return paths


def chain(anchors: Anchors, min_score: float = 40.0) -> list[Chain]:
    """Encadena las anclas en cadenas colineales, mejor primero."""
    if len(anchors) == 0:
        return []
    k = anchors.k
    Lr = anchors.read_len
    out: list[Chain] = []
    for strand in (0, 1):
        m = anchors.strand == strand
        if not m.any():
            continue
        rp = anchors.ref_pos[m]
        qp = anchors.read_pos[m]
        # coordenada de chaining: directa = qp; inversa = (Lr-k) - qp
        yc = qp if strand == 0 else (Lr - k) - qp
        for score, idx in _chain_one(rp, yc, k, min_score):
            # idx: índices (en el orden original de esta hebra) de la cadena
            a_ref = rp[idx]
            a_read = qp[idx]
            srt = np.argsort(a_ref)
            a_ref, a_read = a_ref[srt], a_read[srt]
            out.append(Chain(
                score=score, strand=strand,
                ref_start=int(a_ref.min()), ref_end=int(a_ref.max()) + k,
                read_start=int(a_read.min()), read_end=int(a_read.max()) + k,
                n_anchors=int(a_ref.size),
                anchor_ref=a_ref, anchor_read=a_read,
            ))
    out.sort(key=lambda c: c.score, reverse=True)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# §5-6  EXTENSIÓN + API
# ══════════════════════════════════════════════════════════════════════════════

class Mapping(NamedTuple):
    """Un mapeo de un read en la referencia (campos estilo PAF)."""
    query_len:    int
    query_start:  int
    query_end:    int
    strand:       str          # '+' o '-'
    target_len:   int
    target_start: int
    target_end:   int
    num_matches:  int          # residuos idénticos en el bloque alineado
    block_len:    int          # longitud del bloque de alineamiento
    mapq:         int
    identity:     float
    chain_score:  float
    cigar:        Optional[str]

    def to_paf(self, query_name: str = "query", target_name: str = "ref") -> str:
        """Serializa a una línea PAF (el formato de minimap2)."""
        fields = [
            query_name, self.query_len, self.query_start, self.query_end,
            self.strand, target_name, self.target_len,
            self.target_start, self.target_end,
            self.num_matches, self.block_len, self.mapq,
        ]
        line = "\t".join(str(f) for f in fields)
        if self.cigar:
            line += f"\tcg:Z:{self.cigar}"
        return line


def _pack(seq: str):
    return SmartImporter.from_string(f">x\n{seq}\n",
                                     force_type=SeqType.NUCLEOTIDE)[0]


def _cigar(aln_ref: str, aln_read: str) -> tuple[str, int, int]:
    """CIGAR + (nº matches, longitud de bloque) desde dos cadenas alineadas.

    aln_ref/aln_read llevan '-' en los huecos. M=col sin hueco, I=inserción
    en el read (hueco en ref), D=deleción (hueco en el read).
    """
    ops: list[str] = []
    n_match = 0
    for a, b in zip(aln_ref, aln_read, strict=True):
        if a != "-" and b != "-":
            op = "M"
            if a == b:
                n_match += 1
        elif a == "-":
            op = "I"
        else:
            op = "D"
        ops.append(op)
    # run-length encode
    if not ops:
        return "", 0, 0
    parts, run, cur = [], 1, ops[0]
    for op in ops[1:]:
        if op == cur:
            run += 1
        else:
            parts.append(f"{run}{cur}")
            cur, run = op, 1
    parts.append(f"{run}{cur}")
    return "".join(parts), n_match, len(ops)


def _extend(ref: str, read: str, ch: Chain) -> Optional[Mapping]:
    """Fase 5: alinea (banded) la región de la cadena y arma el Mapping."""
    ref_sub = ref[ch.ref_start:ch.ref_end]
    read_sub = read[ch.read_start:ch.read_end]
    if ch.strand == 1:
        read_sub = _revcomp(read_sub)
    if not ref_sub or not read_sub:
        return None

    band = min(256, abs(len(ref_sub) - len(read_sub)) + 32)
    try:
        res = SequenceAligner.align(_pack(ref_sub), _pack(read_sub),
                                    mode="global", band=band)
        cigar, n_match, block = _cigar(res.aligned_a, res.aligned_b)
        identity = res.identity
    except Exception:                       # noqa: BLE001 — extensión best-effort
        cigar, n_match, block, identity = None, ch.n_anchors * ch.k, \
            ch.ref_end - ch.ref_start, 1.0

    return Mapping(
        query_len=len(read),
        query_start=ch.read_start, query_end=ch.read_end,
        strand="+" if ch.strand == 0 else "-",
        target_len=len(ref),
        target_start=ch.ref_start, target_end=ch.ref_end,
        num_matches=n_match, block_len=block,
        mapq=0, identity=identity, chain_score=ch.score, cigar=cigar,
    )


def _mapq(chains: list[Chain], i: int) -> int:
    """Calidad de mapeo simple: primaria alta si domina a la secundaria."""
    if i > 0:
        return 5
    if len(chains) < 2:
        return 60
    ratio = chains[1].score / chains[0].score if chains[0].score else 0.0
    return int(max(0, min(60, round(60 * (1.0 - ratio)))))


class GenomeAligner:
    """Mapeador de reads contra una referencia (seed-chain-align)."""

    def __init__(self, reference: str, k: int = 15, w: int = 10,
                 max_occ: Optional[int] = 50, name: str = "ref"):
        self.reference = reference.upper()
        self.name = name
        self.index = ReferenceIndex.from_sequence(self.reference, k=k, w=w,
                                                  max_occ=max_occ)

    @property
    def k(self) -> int:
        return self.index.k

    def map(self, read: str, min_chain_score: float = 40.0,
            max_hits: int = 5) -> list[Mapping]:
        """Mapea un read → lista de Mapping (primaria primero)."""
        read = read.upper()
        anchors = seed(self.index, encode_bases(read))
        chains = chain(anchors, min_score=min_chain_score)[:max_hits]
        mappings: list[Mapping] = []
        for i, ch in enumerate(chains):
            mp = _extend(self.reference, read, ch)
            if mp is not None:
                mappings.append(mp._replace(mapq=_mapq(chains, i)))
        return mappings
