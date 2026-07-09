"""
tests/test_cmap_parity.py — paridad exacta del pipeline de mapeo en C
frente al fallback NumPy (mapeador v5).

El camino C (bio_map_read / bio_map_batch) DEBE producir mapeos idénticos al
camino NumPy, campo por campo (coords, CIGAR, identidad, mapq, chain_score).
Este fuzz determinista lo garantiza en muchos tipos de read — es la red que
protege la equivalencia cuando el motor C evolucione (p.ej. el SIMD de v6.0).

Si el motor C no está compilado con el mapeador, se salta (el fallback ya está
cubierto por test_genomemap).
"""

import numpy as np
import pytest

from bioforge.engine import _loader as L
from bioforge.genomemap import GenomeAligner, _revcomp

pytestmark = pytest.mark.skipif(
    not L.C_INDEX_AVAILABLE, reason="motor C sin mapeador (bio_map_read)")


def _fields(mps):
    """Tupla comparable con TODOS los campos observables de cada Mapping."""
    return [(m.target_name, m.target_start, m.target_end, m.query_start,
             m.query_end, m.strand, m.num_matches, m.block_len,
             round(m.identity, 9), m.mapq, m.cigar, round(m.chain_score, 6))
            for m in mps]


def _fallback(ga, fn):
    """Ejecuta fn con el índice C desactivado (camino NumPy) y lo restaura."""
    h = ga._c_index
    ga._c_index = None
    try:
        return fn()
    finally:
        ga._c_index = h


def _rseq(n, rng):
    return "".join("ACGT"[i] for i in rng.integers(0, 4, n))


def _mutate(s, nmut, rng):
    s = list(s)
    for p in rng.choice(len(s), min(nmut, len(s)), replace=False):
        s[p] = "ACGT"[("ACGT".index(s[p]) + 1) % 4]
    return "".join(s)


def _indels(s, rng, n=4):
    s = list(s)
    for _ in range(n):
        if len(s) > 1:
            del s[int(rng.integers(0, len(s)))]
        s.insert(int(rng.integers(0, len(s) + 1)), "ACGT"[int(rng.integers(0, 4))])
    return "".join(s)


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_map_c_igual_que_fallback(seed):
    rng = np.random.default_rng(seed)
    genome = _rseq(80_000, rng)
    ga = GenomeAligner(genome, k=15, w=10, max_occ=50)
    for _ in range(60):
        kind = int(rng.integers(0, 6))
        o = int(rng.integers(0, len(genome) - 700))
        Lr = int(rng.integers(60, 700))
        read = genome[o:o + Lr]
        if kind == 1:
            read = _revcomp(read)
        elif kind == 2:
            read = _mutate(read, max(1, Lr // 30), rng)
        elif kind == 3:
            read = _indels(read, rng)
        elif kind == 4:
            read = _rseq(Lr, rng)              # ajeno (no mapea)
        elif kind == 5:
            read = "N" * 20 + read             # prefijo inválido
        c = _fields(ga.map(read))
        py = _fields(_fallback(ga, lambda: ga.map(read)))
        assert c == py


def test_map_batch_c_igual_que_fallback_multicontig():
    rng = np.random.default_rng(2024)
    contigs = {f"c{i}": _rseq(int(rng.integers(4000, 15000)), rng)
               for i in range(3)}
    ga = GenomeAligner(contigs, k=15, w=10, max_occ=50)
    reads = []
    for _ in range(200):
        ci = int(rng.integers(0, ga.n_contigs))
        cs, cl = ga._starts_list[ci], ga._lengths[ci]
        src = ga.reference[cs:cs + cl]
        o = int(rng.integers(0, max(1, cl - 400)))
        r = src[o:o + 300]
        if rng.integers(0, 2):
            r = _revcomp(r)
        reads.append(r)

    for max_hits in (1, 5):
        cb = [_fields(x) for x in ga.map_batch(reads, n_processes=0,
                                               max_hits=max_hits)]
        pb = _fallback(ga, lambda: [_fields(ga.map(r, 40.0, max_hits))
                                    for r in reads])
        assert cb == pb


def test_repetitivo_muchas_anclas():
    # Genoma hiper-repetitivo → muchas anclas y cadenas: estresa chaining/cigar.
    rng = np.random.default_rng(11)
    unit = _rseq(120, rng)
    ga = GenomeAligner(unit * 300 + _rseq(5000, rng), k=15, w=10, max_occ=100)
    for mult in (2, 4, 6):
        read = unit * mult
        c = _fields(ga.map(read))
        py = _fields(_fallback(ga, lambda: ga.map(read)))
        assert c == py
