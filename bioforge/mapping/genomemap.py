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

import os
from multiprocessing import Pool
from typing import NamedTuple, Optional

import numpy as np

from ..align.pairwise import SequenceAligner
from ..core.biocore import (
    AlignmentError,
    SeqType,
    SequenceTypeError,
    SequenceValueError,
    SmartImporter,
)
from ..engine._loader import (
    C_CHAIN_AVAILABLE,
    C_INDEX_AVAILABLE,
    c_chain_dp,
    c_index_build,
    c_index_free,
    c_map_batch,
    c_map_read,
)
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
_GAP_COEF = 0.01     # coste lineal de hueco = 0.01·k (fórmula de minimap2)
_MIN_ANCHORS = 2


def _chain_fill_numpy(xs: np.ndarray, ys: np.ndarray, k: int, gap_w: float):
    """Fallback NumPy del DP de chaining (idéntico al C bio_chain_dp).

    Bucle EXTERNO secuencial (f[i] depende de f[j<i]); bucle INTERNO sobre la
    ventana de predecesores vectorizado. En empate gana el predecesor más
    cercano (mismo desempate que el C). Coste de hueco estilo minimap2:
    ``gap_w·|l| + 0.5·log2|l|``.
    """
    n = xs.size
    f = np.full(n, float(k), dtype=np.float64)
    prev = np.full(n, -1, dtype=np.int64)
    kf = float(k)
    for i in range(n):
        lo = i - _WINDOW
        if lo < 0:
            lo = 0
        if lo == i:                      # i == 0: sin predecesores
            continue
        dx = xs[i] - xs[lo:i]
        dy = ys[i] - ys[lo:i]
        gap = np.abs(dx - dy)
        valid = ((dx > 0) & (dy > 0) & (dx <= _MAX_GAP)
                 & (dy <= _MAX_GAP) & (gap <= _MAX_GAP))
        if not valid.any():
            continue
        match = np.minimum(np.minimum(dx, dy), k).astype(np.float64)
        logterm = np.where(gap > 0, 0.5 * np.log2(np.maximum(gap, 1)), 0.0)
        sc = np.where(valid, f[lo:i] + match - (gap_w * gap + logterm), -np.inf)
        b = sc.size - 1 - int(sc[::-1].argmax())   # empate → predecesor cercano
        if sc[b] > kf:
            f[i], prev[i] = float(sc[b]), lo + b
    return f, prev


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
    xs = np.ascontiguousarray(x[order], dtype=np.int64)
    ys = np.ascontiguousarray(y[order], dtype=np.int64)

    # Relleno del DP: C si está disponible (10-50× más rápido), si no NumPy.
    gap_w = _GAP_COEF * k                     # 0.01·k (coste de hueco minimap2)
    if C_CHAIN_AVAILABLE:
        f, prev = c_chain_dp(xs, ys, k, _MAX_GAP, _WINDOW, gap_w)
        prev = prev.astype(np.int64)
    else:
        f, prev = _chain_fill_numpy(xs, ys, k, gap_w)

    # Backtrack de cadenas no solapadas, por score descendente.
    used = np.zeros(n, dtype=bool)
    paths: list[tuple] = []
    for start in np.argsort(-f):
        if f[start] < min_score:
            break                                # f desc: el resto también cae
        if used[start]:
            continue
        path = []
        i = int(start)
        while i != -1 and not used[i]:
            used[i] = True
            path.append(i)
            i = int(prev[i])
        # Score REAL del fragmento extraído: f[start] menos el score acumulado
        # hasta el punto donde se cortó (una ancla ya usada, o el inicio). Sin
        # esto, un fragmento corto (cola de otra cadena) heredaría el score
        # inflado de la cadena larga.
        frag = float(f[start]) - (float(f[i]) if i != -1 else 0.0)
        if len(path) >= _MIN_ANCHORS and frag >= min_score:
            path.reverse()                       # de inicio a fin
            paths.append((frag, order[np.array(path)]))
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

    # Supresión de solapamientos: una cadena de menor score que solapa >50% en
    # el GENOMA con una ya aceptada (misma hebra) es un fragmento redundante del
    # mismo locus → se descarta. Loci distintos (copias) tienen spans separados
    # y se conservan ambos.
    kept: list[Chain] = []
    for c in out:
        redundant = False
        for kc in kept:
            if c.strand != kc.strand:
                continue
            inter = min(c.ref_end, kc.ref_end) - max(c.ref_start, kc.ref_start)
            shorter = min(c.ref_end - c.ref_start, kc.ref_end - kc.ref_start)
            if shorter > 0 and inter > 0.5 * shorter:
                redundant = True
                break
        if not redundant:
            kept.append(c)
    return kept


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
    target_name:  str = "ref"  # nombre del contig/cromosoma al que mapea

    def to_paf(self, query_name: str = "query",
               target_name: Optional[str] = None) -> str:
        """Serializa a una línea PAF (el formato de minimap2).

        Si no se pasa ``target_name``, se usa el del propio mapeo (el contig al
        que mapeó, útil con referencias multi-cromosoma).
        """
        tname = target_name if target_name is not None else self.target_name
        fields = [
            query_name, self.query_len, self.query_start, self.query_end,
            self.strand, tname, self.target_len,
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
    a = np.frombuffer(aln_ref.encode("ascii"), dtype=np.uint8)
    b = np.frombuffer(aln_read.encode("ascii"), dtype=np.uint8)
    if a.size == 0:
        return "", 0, 0
    gap = np.uint8(ord("-"))
    ag, bg = a == gap, b == gap
    n_match = int(np.count_nonzero(~ag & ~bg & (a == b)))
    # op: 0=M, 1=I (hueco en ref), 2=D (hueco en el read)  — vectorizado
    op = np.where(ag, 1, np.where(bg, 2, 0)).astype(np.int8)
    change = np.ones(op.size, dtype=bool)
    change[1:] = op[1:] != op[:-1]
    starts = np.flatnonzero(change)
    runs = np.diff(np.append(starts, op.size))
    # el join recorre solo los TRAMOS (pocos), no las columnas
    cigar = "".join(f"{int(r)}{'MID'[int(op[s])]}"
                    for r, s in zip(runs, starts, strict=True))
    return cigar, n_match, int(op.size)


def _extend(ref: str, read: str, ch: Chain, k: int,
            cstart: int, cend: int) -> Optional[Mapping]:
    """Fase 5: alinea el READ COMPLETO contra la referencia y arma el Mapping.

    La ventana de referencia se sitúa por la diagonal de la cadena
    (``d`` = posición ref donde cae el inicio del read) y se recorta al contig
    ``[cstart, cend)``. Así la alineación cubre todo el read (no solo la región
    de la cadena); si el read sobresale por un borde del contig, esa parte queda
    recortada (soft-clip) de forma natural.
    """
    Lr = len(read)
    if ch.strand == 0:
        d = int(np.median(ch.anchor_ref - ch.anchor_read))
        oriented = read
    else:
        # hebra inversa: se alinea revcomp(read) hacia delante; diagonal:
        d = int(np.median(ch.anchor_ref + ch.anchor_read)) - (Lr - k)
        oriented = _revcomp(read)

    rs = max(cstart, d)
    re = min(cend, d + Lr)
    if re - rs < 1:
        return None
    ref_sub = ref[rs:re]
    read_sub = oriented[rs - d:re - d]        # porción del read dentro de la ventana
    if not ref_sub or not read_sub:
        return None

    band = min(512, abs(len(ref_sub) - len(read_sub)) + 64)
    try:
        res = SequenceAligner.align(_pack(ref_sub), _pack(read_sub),
                                    mode="global", band=band,
                                    detect_mutations=False)   # el mapeo no las usa
    except (AlignmentError, MemoryError):
        # Extensión fallida (banda insuficiente, etc.): SIN mapeo. Nunca
        # inventar identidad — un 100% falso es peor que no reportar nada.
        return None
    cigar, n_match, block = _cigar(res.aligned_a, res.aligned_b)
    identity = res.identity

    # coords de query en el read ORIGINAL (hacia delante), aun en hebra inversa
    if ch.strand == 0:
        q_start, q_end = rs - d, re - d
    else:
        q_start, q_end = Lr - (re - d), Lr - (rs - d)

    return Mapping(
        query_len=Lr,
        query_start=q_start, query_end=q_end,
        strand="+" if ch.strand == 0 else "-",
        target_len=len(ref),
        target_start=rs, target_end=re,
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


# ── Workers de multiprocessing (a nivel de módulo → picklables) ─────────────────
_MP_ALIGNER: "Optional[GenomeAligner]" = None


def _mp_init(aligner: "GenomeAligner") -> None:
    """Inicializador de cada proceso: guarda el índice (se pasa una vez)."""
    global _MP_ALIGNER
    _MP_ALIGNER = aligner


def _mp_worker(args) -> "list[Mapping]":
    read, min_chain_score, max_hits = args
    return _MP_ALIGNER.map(read, min_chain_score, max_hits)


class GenomeAligner:
    """Mapeador de reads contra una referencia (seed-chain-align).

    La referencia puede ser:
      • una cadena (un solo contig), o
      • un dict ``{nombre: secuencia}`` o iterable de ``(nombre, secuencia)``
        para varios cromosomas/contigs (genomas reales).

    Multi-contig: los contigs se concatenan con separadores de ``N`` (que los
    minimizers excluyen → ningún k-mer cruza fronteras). Se indexa el conjunto y
    al mapear la posición global se traduce a (contig, posición local).
    """

    def __init__(self, reference, k: int = 15, w: int = 10,
                 max_occ: Optional[int] = 50, name: str = "ref"):
        contigs = self._normalize(reference, name)      # [(nombre, seq), ...]
        if not contigs:
            raise SequenceValueError("La referencia no contiene ninguna secuencia.")
        if sum(len(s) for _, s in contigs) == 0:
            raise SequenceValueError("La referencia está vacía.")
        sep = "N" * k                                   # rompe k-meros de frontera
        parts: list[str] = []
        self._names: list[str] = []
        self._starts_list: list[int] = []
        self._lengths: list[int] = []
        g = 0
        for i, (nm, seq) in enumerate(contigs):
            seq = seq.upper()
            if i > 0:
                parts.append(sep); g += len(sep)
            self._names.append(str(nm))
            self._starts_list.append(g)
            self._lengths.append(len(seq))
            parts.append(seq); g += len(seq)

        self.reference = "".join(parts)                 # concatenado (con N)
        self._starts = np.array(self._starts_list, dtype=np.int64)
        self.name = self._names[0] if len(self._names) == 1 else "multi"
        self._ref_codes = encode_bases(self.reference)
        self.index = ReferenceIndex(self._ref_codes, k=k, w=w, max_occ=max_occ)

        # Índice opaco en C: el pipeline entero (seed-chain-align) corre en C sin
        # marshalling por consulta. Se construye UNA vez y vive tras un handle.
        # Si el motor C no está, todo cae al camino NumPy (idéntico, verificado).
        self._c_index: Optional[int] = None
        if C_INDEX_AVAILABLE:
            try:
                self._c_index = c_index_build(
                    self._ref_codes, k, w, (max_occ or 0),
                    self._starts,
                    np.array(self._lengths, dtype=np.int64),
                )
            except MemoryError:
                self._c_index = None

    @staticmethod
    def _normalize(reference, name: str) -> "list[tuple[str, str]]":
        if isinstance(reference, str):
            return [(name, reference)]
        if isinstance(reference, dict):
            return list(reference.items())
        try:
            return [(n, s) for n, s in reference]       # iterable de pares
        except (TypeError, ValueError) as exc:
            raise SequenceTypeError(
                "reference debe ser str, dict {nombre: secuencia} o un iterable "
                f"de pares (nombre, secuencia); se recibió "
                f"{type(reference).__name__!r}.") from exc

    @property
    def k(self) -> int:
        return self.index.k

    @property
    def n_contigs(self) -> int:
        return len(self._names)

    def _contig_of(self, gpos: int) -> int:
        """Índice del contig que contiene la posición global ``gpos``."""
        i = int(np.searchsorted(self._starts, gpos, side="right")) - 1
        return max(0, i)

    def _localize(self, mp: Mapping) -> Mapping:
        """Traduce coords globales de un mapeo a (contig, coords locales)."""
        ci = self._contig_of(mp.target_start)
        cstart, clen = self._starts_list[ci], self._lengths[ci]
        return mp._replace(
            target_name=self._names[ci],
            target_len=clen,
            target_start=mp.target_start - cstart,
            target_end=min(mp.target_end - cstart, clen),
        )

    def __del__(self):
        h = getattr(self, "_c_index", None)
        if h:
            try:
                c_index_free(h)
            except Exception:       # noqa: BLE001 — el intérprete puede estar cerrando
                pass
            self._c_index = None

    def __getstate__(self):
        # El handle C es un puntero crudo, no picklable ni válido entre procesos.
        st = self.__dict__.copy()
        st["_c_index"] = None
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        if C_INDEX_AVAILABLE and self._c_index is None:
            try:
                self._c_index = c_index_build(
                    self._ref_codes, self.index.k, self.index.w,
                    (self.index.max_occ or 0), self._starts,
                    np.array(self._lengths, dtype=np.int64))
            except MemoryError:
                self._c_index = None

    def _mapping_from_dict(self, d: dict, query_len: int) -> Mapping:
        """Construye un Mapping (coords locales al contig) desde la salida C."""
        ci = d["contig"]
        cstart, clen = self._starts_list[ci], self._lengths[ci]
        return Mapping(
            query_len=query_len,
            query_start=d["query_start"], query_end=d["query_end"],
            strand="+" if d["strand"] == 0 else "-",
            target_len=clen,
            target_start=d["target_start"] - cstart,
            target_end=min(d["target_end"] - cstart, clen),
            num_matches=d["num_matches"], block_len=d["block_len"],
            mapq=d["mapq"], identity=d["identity"], chain_score=d["chain_score"],
            cigar=d["cigar"], target_name=self._names[ci],
        )

    def _mappings_from_columnar(self, out, counts, cbuf, mh, ups) -> list[list[Mapping]]:
        """Construye los Mapping desde la salida columnar del batch (v6.1).

        Lee cada campo como COLUMNA (``.tolist()`` es C-level y da ints/floats
        Python de una) → el bucle solo indexa listas y crea NamedTuples, sin
        acceso ctypes campo a campo ni dicts intermedios. Es lo que quita la cola
        serial que capaba el multinúcleo.
        """
        names = self._names
        starts = self._starts_list
        lengths = self._lengths
        # columnas → listas Python (rápido, una vez)
        ide = out["identity"].tolist();      chs = out["chain_score"].tolist()
        ts  = out["target_start"].tolist();  te  = out["target_end"].tolist()
        qs  = out["query_start"].tolist();   qe  = out["query_end"].tolist()
        st  = out["strand"].tolist();        nm  = out["num_matches"].tolist()
        bl  = out["block_len"].tolist();     mq  = out["mapq"].tolist()
        ct  = out["contig"].tolist();        co  = out["cigar_off"].tolist()
        cl  = out["cigar_len"].tolist()
        cnts = counts.tolist()
        results: list[list[Mapping]] = []
        for i, r in enumerate(ups):
            c = cnts[i]
            base = i * mh
            qlen = len(r)
            maps: list[Mapping] = []
            for j in range(base, base + c):
                ci = ct[j]
                cstart = starts[ci]; clen = lengths[ci]
                cig = bytes(cbuf[co[j]:co[j] + cl[j]]).decode("ascii")
                maps.append(Mapping(
                    query_len=qlen, query_start=qs[j], query_end=qe[j],
                    strand="+" if st[j] == 0 else "-",
                    target_len=clen, target_start=ts[j] - cstart,
                    target_end=min(te[j] - cstart, clen),
                    num_matches=nm[j], block_len=bl[j],
                    mapq=mq[j], identity=ide[j], chain_score=chs[j],
                    cigar=cig, target_name=names[ci],
                ))
            results.append(maps)
        return results

    def map(self, read: str, min_chain_score: float = 40.0,
            max_hits: int = 5) -> list[Mapping]:
        """Mapea un read → lista de Mapping (primaria primero)."""
        if not isinstance(read, str):
            raise SequenceTypeError(
                f"read debe ser str, se recibió {type(read).__name__!r}.")
        read = read.upper()

        # Camino C: pipeline entero (seed-chain-align) en una sola llamada.
        if self._c_index:
            dicts = c_map_read(self._c_index, encode_bases(read),
                               max_hits, min_chain_score)
            return [self._mapping_from_dict(d, len(read)) for d in dicts]

        # Fallback NumPy (idéntico, verificado con parity tests).
        anchors = seed(self.index, encode_bases(read))
        chains = chain(anchors, min_score=min_chain_score)[:max_hits]
        mappings: list[Mapping] = []
        for i, ch in enumerate(chains):
            # Contig de la cadena → sus límites acotan la ventana de extensión.
            ci = self._contig_of(ch.ref_start)
            cstart = self._starts_list[ci]
            cend = cstart + self._lengths[ci]
            mp = _extend(self.reference, read, ch, self.k, cstart, cend)
            if mp is not None:
                mappings.append(self._localize(mp._replace(mapq=_mapq(chains, i))))
        return mappings

    def map_batch(self, reads, n_processes: int = 0,
                  min_chain_score: float = 40.0,
                  max_hits: int = 5) -> list[list[Mapping]]:
        """Mapea muchos reads en PARALELO → lista de listas de Mapping.

        Los reads son independientes (paralelismo perfecto). Usa procesos
        (multiprocessing) porque el GIL impide que los hilos escalen: el trabajo
        por read es mayoritariamente Python. El índice se pasa una vez a cada
        proceso; el orden de salida coincide con el de ``reads``.

        ``n_processes``: 0 = todos los núcleos · 1 = secuencial · N = N procesos.

        Nota (Windows/macOS): el script que lo llame debe estar protegido por
        ``if __name__ == "__main__":`` (requisito de multiprocessing). Si el
        arranque de procesos falla, cae a secuencial con gracia.
        """
        reads = list(reads)

        # Camino C: OpenMP dentro del motor (sin GIL, sin coste de procesos).
        # Una sola llamada mapea todo el lote en paralelo. Es la vía rápida.
        if self._c_index and len(reads) > 1:
            for r in reads:
                if not isinstance(r, str):
                    raise SequenceTypeError(
                        f"cada read debe ser str, se recibió {type(r).__name__!r}.")
            ups = [r.upper() for r in reads]
            enc = [encode_bases(r) for r in ups]
            nt = 0 if n_processes <= 0 else n_processes
            out, counts, cbuf, mh = c_map_batch(self._c_index, enc, max_hits,
                                                min_chain_score, n_threads=nt)
            return self._mappings_from_columnar(out, counts, cbuf, mh, ups)

        if n_processes == 1 or len(reads) <= 1:
            return [self.map(r, min_chain_score, max_hits) for r in reads]
        nt = (os.cpu_count() or 1) if n_processes <= 0 else n_processes
        args = [(r, min_chain_score, max_hits) for r in reads]
        chunk = max(1, len(reads) // (nt * 4))
        try:
            with Pool(processes=nt, initializer=_mp_init, initargs=(self,)) as pool:
                return pool.map(_mp_worker, args, chunksize=chunk)
        except Exception:                       # noqa: BLE001 — fallback secuencial
            return [self.map(r, min_chain_score, max_hits) for r in reads]
