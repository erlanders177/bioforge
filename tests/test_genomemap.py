"""
tests/test_genomemap.py — Fases 3-6 del alineador de genomas (v3.0).

Prueba el mapeador completo seed-chain-align: anclas, chaining, extensión
y la API GenomeAligner.map con salida PAF. Casos: hebra directa e inversa,
mutaciones, reads que no mapean, copias múltiples y formato PAF.
"""

import numpy as np
import pytest

from bioforge.biocore import (
    BioForgeError,
    SequenceTypeError,
    SequenceValueError,
)
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


def test_multicontig_mapea_al_contig_correcto():
    rng = np.random.default_rng(30)
    chr1 = "".join("ACGT"[i] for i in rng.integers(0, 4, 40_000))
    chr2 = "".join("ACGT"[i] for i in rng.integers(0, 4, 25_000))
    plas = "".join("ACGT"[i] for i in rng.integers(0, 4, 6_000))
    ga = GenomeAligner({"chr1": chr1, "chr2": chr2, "plasmid": plas}, k=15, w=10)
    assert ga.n_contigs == 3

    # read de chr2: contig y coords LOCALES correctas
    mp = ga.map(chr2[12_000:12_500])[0]
    assert mp.target_name == "chr2"
    assert mp.target_len == len(chr2)
    assert mp.target_start - mp.query_start == 12_000     # diagonal local
    assert mp.identity > 0.99

    # read del plasmid en hebra inversa
    mp = ga.map(_revcomp(plas[2_000:2_400]))[0]
    assert mp.target_name == "plasmid"
    assert mp.strand == "-"
    assert mp.target_start < 2_400 and mp.target_end > 0

    # el PAF lleva el contig correcto sin pasar target_name
    assert ga.map(chr1[5_000:5_500])[0].to_paf().split("\t")[5] == "chr1"


def test_extension_cubre_el_read_entero():
    # La extensión alinea el read COMPLETO (no solo la región de la cadena).
    genoma = _rng_seq(60_000, 33)
    ga = GenomeAligner(genoma, k=15, w=10)
    o, Lr = 25_000, 500
    mp = ga.map(genoma[o : o + Lr])[0]
    assert mp.query_start == 0            # empieza en el inicio del read…
    assert mp.query_end == Lr             # …y llega al final
    assert mp.target_start == o           # posición exacta (no + offset de minimizer)
    assert mp.identity > 0.99
    # con mutaciones también cubre el read entero
    read = _mutate(genoma[o : o + 600], n_mut=12, seed_=2)
    mp = ga.map(read)[0]
    assert mp.query_start == 0 and mp.query_end == 600


def test_multicontig_lista_de_pares():
    # también acepta un iterable de (nombre, secuencia)
    a = _rng_seq(8000, 31)
    b = _rng_seq(8000, 32)
    ga = GenomeAligner([("A", a), ("B", b)], k=15, w=10)
    assert ga.n_contigs == 2
    assert ga.map(b[3000:3400])[0].target_name == "B"


# ── Robustez / sistema de errores (regla #8: todo fallo → BioForgeError) ────────

def test_map_valida_tipo_de_read():
    ga = GenomeAligner(_rng_seq(2000, 40), k=15, w=10)
    with pytest.raises(SequenceTypeError):
        ga.map(12345)                        # no es str
    with pytest.raises(BioForgeError):       # y capturable como BioForgeError
        ga.map(None)


def test_referencia_vacia_falla():
    with pytest.raises(SequenceValueError):
        GenomeAligner("", k=15, w=10)        # cadena vacía
    with pytest.raises(SequenceValueError):
        GenomeAligner({}, k=15, w=10)        # sin contigs
    with pytest.raises(SequenceValueError):
        GenomeAligner({"a": "", "b": ""}, k=15, w=10)   # contigs vacíos
    with pytest.raises(SequenceTypeError):
        GenomeAligner(12345, k=15, w=10)                # tipo inválido


def test_read_en_los_extremos_del_genoma():
    g = _rng_seq(5000, 41)
    ga = GenomeAligner(g, k=15, w=10)
    # read al inicio (posición 0)
    m0 = ga.map(g[0:400])
    assert m0 and m0[0].target_start == 0
    # read hasta el final
    mE = ga.map(g[4600:5000])
    assert mE and mE[0].target_end == 5000
    # read que sobresale por el final → mapea la parte válida, sin petar
    over = g[4700:5000] + _rng_seq(200, 99)
    mo = ga.map(over)
    assert isinstance(mo, list)              # no lanza; devuelve lista


def test_read_degenerado_no_peta():
    ga = GenomeAligner(_rng_seq(3000, 43), k=15, w=10)
    assert ga.map("") == []                  # vacío
    assert ga.map("ACGT") == []              # más corto que k
    assert ga.map("N" * 500) == []           # todo inválido


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
