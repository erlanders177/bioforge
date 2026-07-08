"""
tests/test_cindex.py — índice opaco de la referencia en C (mapeador v5).

Fase C1 del traslado del pipeline de mapeo a C: el índice se construye una
sola vez en C y vive tras un handle opaco. Estos tests verifican PARIDAD
exacta con ReferenceIndex (mismo nº de minimizers, mismo filtro max_occ) y
que el ciclo build/free no rompe. Si el motor C no está compilado con el
índice, se saltan (fallback NumPy sigue cubierto por test_refindex).
"""

import numpy as np
import pytest

from bioforge.engine import _loader as L
from bioforge.minimizers import encode_bases
from bioforge.refindex import ReferenceIndex

pytestmark = pytest.mark.skipif(
    not L.C_INDEX_AVAILABLE, reason="motor C sin índice (bio_index_build)")


def _rng_seq(n: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    return "".join("ACGT"[i] for i in rng.integers(0, 4, n))


def _build_c(codes, k, w, max_occ, starts, lengths):
    return L.c_index_build(codes, k, w, max_occ,
                           np.asarray(starts, np.int64),
                           np.asarray(lengths, np.int64))


@pytest.mark.parametrize("max_occ", [0, 50, 100])
def test_paridad_n_minimizers_un_contig(max_occ):
    codes = encode_bases(_rng_seq(60_000, 3))
    py = ReferenceIndex(codes, k=15, w=10,
                        max_occ=(None if max_occ == 0 else max_occ))
    h = _build_c(codes, 15, 10, max_occ, [0], [codes.size])
    try:
        assert L.c_index_n_minimizers(h) == py.n_minimizers
    finally:
        L.c_index_free(h)


@pytest.mark.parametrize("max_occ", [0, 5, 20, 50])
def test_filtro_max_occ_con_repeticiones(max_occ):
    # Unidad repetida → minimizers hiper-frecuentes → max_occ debe recortar.
    rng = np.random.default_rng(1)
    unit = "".join("ACGT"[i] for i in rng.integers(0, 4, 300))
    ref = unit * 200 + _rng_seq(20_000, 2)
    codes = encode_bases(ref)
    py = ReferenceIndex(codes, k=15, w=10,
                        max_occ=(None if max_occ == 0 else max_occ))
    h = _build_c(codes, 15, 10, max_occ, [0], [codes.size])
    try:
        assert L.c_index_n_minimizers(h) == py.n_minimizers
    finally:
        L.c_index_free(h)


def test_paridad_multicontig():
    chr1 = _rng_seq(50_000, 10)
    chr2 = _rng_seq(30_000, 11)
    sep = "N" * 15
    codes = encode_bases(chr1 + sep + chr2)
    py = ReferenceIndex(codes, k=15, w=10, max_occ=50)
    starts = [0, len(chr1) + len(sep)]
    lengths = [len(chr1), len(chr2)]
    h = _build_c(codes, 15, 10, 50, starts, lengths)
    try:
        assert L.c_index_n_minimizers(h) == py.n_minimizers
    finally:
        L.c_index_free(h)


def test_free_de_handle_nulo_no_peta():
    # doble free / handle 0 debe ser inofensivo
    L.c_index_free(0)
    assert L.c_index_n_minimizers(0) == -1


def test_ref_muy_corta():
    codes = encode_bases("ACGT")           # más corta que k → 0 minimizers
    h = _build_c(codes, 15, 10, 0, [0], [codes.size])
    try:
        assert L.c_index_n_minimizers(h) == 0
    finally:
        L.c_index_free(h)
