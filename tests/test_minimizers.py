"""
tests/test_minimizers.py — Fase 1 del alineador de genomas (v3.0).

Comprueba el muestreo por minimizers: corrección, exclusión de bases
inválidas, densidad, y la propiedad canónica (invariancia ante hebra).
"""

import numpy as np
import pytest

from bioforge.biocore import SequenceValueError
from bioforge.minimizers import MinimizerSketch, encode_bases, minimizers

_COMP = str.maketrans("ACGTacgt", "TGCAtgca")


def _revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def _rng_seq(n: int, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    return "".join("ACGT"[i] for i in rng.integers(0, 4, size=n))


# ── Básico ──────────────────────────────────────────────────────────────────────

def test_encode_bases():
    assert list(encode_bases("ACGTN")) == [0, 1, 2, 3, 4]
    assert list(encode_bases("acgt")) == [0, 1, 2, 3]


def test_devuelve_sketch_valido():
    seq = _rng_seq(200, seed=1)
    mk = minimizers(encode_bases(seq), k=15, w=10)
    assert isinstance(mk, MinimizerSketch)
    assert mk.hashes.dtype == np.uint64
    assert mk.positions.dtype == np.int64
    assert mk.strands.dtype == np.uint8
    assert len(mk) == mk.hashes.size == mk.positions.size == mk.strands.size
    # posiciones dentro del rango de k-mers y ordenadas/únicas
    n_kmers = len(seq) - 15 + 1
    assert mk.positions.min() >= 0
    assert mk.positions.max() < n_kmers
    assert np.all(np.diff(mk.positions) > 0)


def test_secuencia_mas_corta_que_k():
    mk = minimizers(encode_bases("ACGT"), k=15, w=10)
    assert len(mk) == 0


def test_w1_todos_los_kmeros_son_candidatos():
    # Con w=1 cada k-mer es su propia ventana → todos los k-mers válidos aparecen.
    seq = _rng_seq(60, seed=2)
    n_kmers = len(seq) - 11 + 1
    mk = minimizers(encode_bases(seq), k=11, w=1)
    assert len(mk) == n_kmers


# ── Bases inválidas ─────────────────────────────────────────────────────────────

def test_excluye_kmeros_con_N():
    # Un tramo de N no puede aportar minimizers que lo solapen.
    seq = "A" * 20 + "N" * 20 + "C" * 20
    mk = minimizers(encode_bases(seq), k=15, w=5)
    # Ningún minimizer puede empezar en una posición cuyo k-mer toque una N.
    for p in mk.positions:
        ventana = seq[p : p + 15]
        assert "N" not in ventana


# ── Densidad ────────────────────────────────────────────────────────────────────

def test_densidad_aproximada():
    # Densidad esperada ~ 2/(w+1). Para una secuencia aleatoria larga debe rondar.
    seq = _rng_seq(5000, seed=3)
    k, w = 15, 10
    mk = minimizers(encode_bases(seq), k=k, w=w)
    n_kmers = len(seq) - k + 1
    densidad = len(mk) / n_kmers
    esperada = 2 / (w + 1)
    assert 0.4 * esperada < densidad < 2.0 * esperada


# ── Propiedad canónica: invariancia ante hebra (la clave del método) ────────────

def test_reverse_complement_mismos_minimizers():
    seq = _rng_seq(1000, seed=4)
    rc = _revcomp(seq)
    mk_f = minimizers(encode_bases(seq), k=15, w=10)
    mk_r = minimizers(encode_bases(rc), k=15, w=10)
    # El CONJUNTO de hashes debe coincidir: un read y su inverso-complementario
    # comparten los mismos minimizers → se puede sembrar en ambas hebras.
    assert set(mk_f.hashes.tolist()) == set(mk_r.hashes.tolist())


def test_hash_canonico_es_el_menor():
    # El hash guardado en cada posición debe ser min(directo, inverso).
    seq = _rng_seq(300, seed=5)
    codes = encode_bases(seq)
    # w=1 para inspeccionar todas las posiciones sin el filtro de ventana.
    mk = minimizers(codes, k=15, w=1)
    # Recalcular a mano el hash directo e inverso de un par de posiciones.
    from bioforge.minimizers import _hash64
    k = 15
    mask = np.uint64((1 << (2 * k)) - 1)
    for p in mk.positions[:20]:
        kmer = codes[p : p + k].astype(np.uint64)
        fwd = np.uint64(0)
        for c in kmer:
            fwd = (fwd << np.uint64(2)) | c
        comp = (np.uint64(3) - kmer)[::-1]
        rev = np.uint64(0)
        for c in comp:
            rev = (rev << np.uint64(2)) | c
        esperado = min(int(_hash64(np.array([fwd]), mask)[0]),
                       int(_hash64(np.array([rev]), mask)[0]))
        idx = np.where(mk.positions == p)[0][0]
        assert int(mk.hashes[idx]) == esperado


# ── Validación de argumentos ────────────────────────────────────────────────────

@pytest.mark.parametrize("k", [0, 32, -1])
def test_k_invalido(k):
    with pytest.raises(SequenceValueError):
        minimizers(encode_bases("ACGT" * 20), k=k, w=5)


def test_w_invalido():
    with pytest.raises(SequenceValueError):
        minimizers(encode_bases("ACGT" * 20), k=15, w=0)
