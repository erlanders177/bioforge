"""
tests/test_genomemap.py — Fases 3-6 del alineador de genomas (v3.0).

Prueba el mapeador completo seed-chain-align: anclas, chaining, extensión
y la API GenomeAligner.map con salida PAF. Casos: hebra directa e inversa,
mutaciones, reads que no mapean, copias múltiples y formato PAF.
"""

import numpy as np

from bioforge.genomemap import (
    Anchors,
    Chain,
    GenomeAligner,
    Mapping,
    _revcomp,
    chain,
    seed,
)
from bioforge.minimizers import encode_bases
from bioforge.refindex import ReferenceIndex


def _rng_seq(n: int, seed_: int = 0) -> str:
    rng = np.random.default_rng(seed_)
    return "".join("ACGT"[i] for i in rng.integers(0, 4, size=n))


def _mutate(seq: str, n_mut: int, seed_: int = 0) -> str:
    rng = np.random.default_rng(seed_)
    s = list(seq)
    for p in rng.choice(len(seq), n_mut, replace=False):
        s[p] = "ACGT"[("ACGT".index(s[p]) + 1) % 4]
    return "".join(s)


# ── Seeding + chaining (piezas) ─────────────────────────────────────────────────

def test_seed_genera_anclas():
    genoma = _rng_seq(3000, 1)
    idx = ReferenceIndex.from_sequence(genoma, k=15, w=10)
    read = genoma[1000:1400]
    anc = seed(idx, encode_bases(read))
    assert isinstance(anc, Anchors)
    assert len(anc) > 0
    # las anclas de un read directo caen en su diagonal de origen
    diagonales = anc.ref_pos[anc.strand == 0] - anc.read_pos[anc.strand == 0]
    assert (diagonales == 1000).sum() >= 1


def test_chain_encadena_colineales():
    genoma = _rng_seq(4000, 2)
    idx = ReferenceIndex.from_sequence(genoma, k=15, w=10)
    read = genoma[500:900]
    chains = chain(seed(idx, encode_bases(read)))
    assert len(chains) >= 1
    top = chains[0]
    assert isinstance(top, Chain)
    assert top.strand == 0
    assert top.n_anchors >= 2


# ── Mapeo completo ──────────────────────────────────────────────────────────────

def test_map_read_directo():
    genoma = _rng_seq(100_000, 7)
    ga = GenomeAligner(genoma, k=15, w=10)
    o = 54_321
    mps = ga.map(genoma[o : o + 500])
    assert len(mps) >= 1
    mp = mps[0]
    assert mp.strand == "+"
    # diagonal exacta: target_start - query_start == posición de origen
    assert mp.target_start - mp.query_start == o
    assert mp.identity > 0.99
    assert mp.mapq >= 50


def test_map_read_inverso():
    genoma = _rng_seq(100_000, 8)
    ga = GenomeAligner(genoma, k=15, w=10)
    o = 40_000
    mps = ga.map(_revcomp(genoma[o : o + 500]))
    assert len(mps) >= 1
    mp = mps[0]
    assert mp.strand == "-"
    assert mp.identity > 0.99
    # la región mapeada solapa el origen real
    assert mp.target_start < o + 500 and mp.target_end > o


def test_map_read_con_mutaciones():
    genoma = _rng_seq(80_000, 9)
    ga = GenomeAligner(genoma, k=15, w=10)
    o = 20_000
    read = _mutate(genoma[o : o + 600], n_mut=12, seed_=1)
    mps = ga.map(read)
    assert len(mps) >= 1
    mp = mps[0]
    assert mp.strand == "+"
    assert mp.target_start - mp.query_start == o    # localiza bien pese a mutaciones
    assert 0.90 < mp.identity < 1.0                 # alta pero no perfecta


def test_read_que_no_mapea():
    genoma = _rng_seq(50_000, 10)
    ga = GenomeAligner(genoma, k=15, w=10)
    ajeno = _rng_seq(500, 999)                       # de otra secuencia
    assert ga.map(ajeno) == []


def test_read_mas_corto_que_k():
    ga = GenomeAligner(_rng_seq(5000, 11), k=15, w=10)
    assert ga.map("ACGT") == []


def test_copias_multiples():
    seg = _rng_seq(500, 12)
    genoma = _rng_seq(2000, 13) + seg + _rng_seq(2000, 14) + seg + _rng_seq(2000, 15)
    ga = GenomeAligner(genoma, k=15, w=10, max_occ=100)
    mps = ga.map(seg)
    # el segmento aparece dos veces → dos mapeos
    starts = sorted(mp.target_start for mp in mps)
    assert len(mps) >= 2
    # uno cerca de 2000, otro cerca de 2000+500+2000=4500
    assert any(abs(s - 2000) < 50 for s in starts)
    assert any(abs(s - 4500) < 50 for s in starts)


# ── Salida PAF ──────────────────────────────────────────────────────────────────

def test_map_batch_igual_que_secuencial():
    genoma = _rng_seq(30_000, 20)
    ga = GenomeAligner(genoma, k=15, w=10)
    reads = [genoma[o : o + 400] for o in (1000, 5000, 12000, 20000)]
    seq = [ga.map(r) for r in reads]

    # ruta secuencial de la API (n_processes=1)
    b1 = ga.map_batch(reads, n_processes=1)
    assert [len(x) for x in b1] == [len(x) for x in seq]

    # ruta paralela (procesos): mismo resultado y mismo orden
    b2 = ga.map_batch(reads, n_processes=2)
    assert len(b2) == len(reads)
    for got, exp in zip(b2, seq, strict=True):
        assert [m.target_start for m in got] == [m.target_start for m in exp]


def test_formato_paf():
    genoma = _rng_seq(30_000, 16)
    ga = GenomeAligner(genoma, k=15, w=10)
    o = 10_000
    mp = ga.map(genoma[o : o + 400])[0]
    assert isinstance(mp, Mapping)
    paf = mp.to_paf(query_name="read1", target_name="chr1")
    campos = paf.split("\t")
    assert len(campos) >= 12
    assert campos[0] == "read1"
    assert campos[4] in ("+", "-")
    assert campos[5] == "chr1"
    assert "cg:Z:" in paf                            # CIGAR presente
