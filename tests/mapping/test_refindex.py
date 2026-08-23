"""
tests/test_refindex.py — Fase 2 del alineador de genomas (v3.0).

Comprueba el índice de la referencia: construcción, búsqueda vectorizada,
identidad (cada minimizer se encuentra a sí mismo), mapeo de un fragmento
a su posición correcta, y el filtrado de minimizers hiper-frecuentes.
"""

import numpy as np

from bioforge.mapping.minimizers import encode_bases, minimizers
from bioforge.mapping.refindex import LookupResult, ReferenceIndex


def _rng_seq(n: int, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    return "".join("ACGT"[i] for i in rng.integers(0, 4, size=n))


def test_construccion_coincide_con_minimizers():
    seq = _rng_seq(2000, seed=1)
    idx = ReferenceIndex.from_sequence(seq, k=15, w=10)
    mk = minimizers(encode_bases(seq), k=15, w=10)
    assert idx.n_minimizers == len(mk)
    assert idx.ref_len == len(seq)
    # la tabla está ordenada por hash (invariante para searchsorted)
    assert np.all(np.diff(idx.hashes) >= 0)


def test_identidad_cada_minimizer_se_encuentra():
    # Buscar los propios hashes de la referencia: cada uno debe hallarse a sí
    # mismo (su posición original entre las coincidencias).
    seq = _rng_seq(1500, seed=2)
    idx = ReferenceIndex.from_sequence(seq, k=15, w=10)
    res = idx.lookup(idx.hashes)
    assert isinstance(res, LookupResult)
    # Para cada consulta i (hash idx.hashes[i], posición idx.positions[i]),
    # sus coincidencias deben incluir esa posición.
    for i in range(idx.n_minimizers):
        matches = res.ref_positions[res.query_idx == i]
        assert idx.positions[i] in matches


def test_fragmento_mapea_a_su_posicion():
    # Un fragmento del genoma: sus minimizers deben apuntar a su zona de origen.
    seq = _rng_seq(5000, seed=3)
    idx = ReferenceIndex.from_sequence(seq, k=15, w=10)
    a = 1200
    frag = seq[a : a + 400]
    mk = minimizers(encode_bases(frag), k=15, w=10)
    res = idx.lookup(mk.hashes)
    # Al menos un minimizer del fragmento debe coincidir…
    assert len(res) > 0
    # …y para los que coinciden, alguna posición de referencia debe ser
    # (a + posición_local): es el mismo k-mer, así que casa exacto.
    aciertos = 0
    for qi in np.unique(res.query_idx):
        local_pos = int(mk.positions[qi])
        ref_ok = (res.ref_positions[res.query_idx == qi] == a + local_pos)
        aciertos += int(ref_ok.any())
    # la mayoría de los minimizers compartidos deben mapear al offset correcto
    assert aciertos >= 0.5 * len(np.unique(res.query_idx))


def test_hash_inexistente_sin_coincidencias():
    seq = _rng_seq(500, seed=4)
    idx = ReferenceIndex.from_sequence(seq, k=15, w=10)
    # un hash imposible (no es minimizer de nada)
    res = idx.lookup(np.array([np.uint64(0)], dtype=np.uint64))
    assert len(res) == 0


def test_lookup_vacio():
    idx = ReferenceIndex.from_sequence(_rng_seq(300, seed=5), k=15, w=10)
    res = idx.lookup(np.empty(0, dtype=np.uint64))
    assert len(res) == 0


def test_max_occ_filtra_repeticiones():
    # Un motivo repetido crea un minimizer hiper-frecuente; max_occ lo descarta.
    motif = "ACGTACGTGGCC"
    seq = motif * 100 + _rng_seq(1000, seed=6)
    sin_filtro = ReferenceIndex.from_sequence(seq, k=11, w=5)
    con_filtro = ReferenceIndex.from_sequence(seq, k=11, w=5, max_occ=5)
    # el filtrado reduce el nº de entradas (elimina las hiper-frecuentes)
    assert con_filtro.n_minimizers < sin_filtro.n_minimizers
    # y ningún hash sobrevive con más de max_occ apariciones
    if con_filtro.n_minimizers:
        _uniq, cnts = np.unique(con_filtro.hashes, return_counts=True)
        assert cnts.max() <= 5


def test_strands_en_resultado():
    seq = _rng_seq(800, seed=7)
    idx = ReferenceIndex.from_sequence(seq, k=15, w=10)
    res = idx.lookup(idx.hashes[:10])
    assert res.ref_strands.dtype == np.uint8
    assert np.all((res.ref_strands == 0) | (res.ref_strands == 1))
