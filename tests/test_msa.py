"""
tests/test_msa.py — alineamiento múltiple (MSA, método center-star).

Verifica las propiedades que DEBE cumplir cualquier MSA correcto:
  - todas las filas tienen la misma longitud (columnas alineadas),
  - quitar los huecos de cada fila recupera EXACTAMENTE la secuencia original
    (el MSA no corrompe ni pierde datos),
  - secuencias idénticas se alinean sin huecos,
  - inserciones y deleciones se manejan de forma consistente,
  - consenso y errores.
"""

import numpy as np
import pytest

from bioforge.biocore import SequenceTypeError, SequenceValueError
from bioforge.msa import MSAResult, align_multiple


def _rng_seq(n, seed):
    rng = np.random.default_rng(seed)
    return "".join("ACGT"[i] for i in rng.integers(0, 4, n))


def _mutate(s, n_mut, seed):
    rng = np.random.default_rng(seed)
    s = list(s)
    for p in rng.choice(len(s), min(n_mut, len(s)), replace=False):
        s[p] = "ACGT"[("ACGT".index(s[p]) + 1) % 4]
    return "".join(s)


# ── Propiedades fundamentales ─────────────────────────────────────────────────

def test_todas_las_filas_misma_longitud():
    seqs = [_mutate(_rng_seq(300, 1), 10, i) for i in range(8)]
    r = align_multiple(seqs)
    assert isinstance(r, MSAResult)
    assert len({len(a) for a in r.aligned}) == 1
    assert r.length == len(r.aligned[0])
    assert len(r) == len(seqs)


def test_quitar_huecos_recupera_original():
    """La propiedad más importante: el MSA no corrompe las secuencias."""
    rng = np.random.default_rng(7)
    base = _rng_seq(400, 2)
    seqs = []
    for i in range(10):
        s = list(_mutate(base, 15, i))
        # meter algún indel
        if i % 2:
            del s[int(rng.integers(0, len(s)))]
        else:
            s.insert(int(rng.integers(0, len(s))), "ACGT"[int(rng.integers(0, 4))])
        seqs.append("".join(s))
    r = align_multiple(seqs)
    for original, fila in zip(seqs, r.aligned, strict=True):
        assert fila.replace("-", "") == original


def test_identicas_sin_huecos():
    s = _rng_seq(120, 3)
    r = align_multiple([s, s, s, s])
    assert all(a == s for a in r.aligned)
    assert r.length == len(s)


def test_insercion_se_alinea():
    # la del medio tiene una base extra → las otras reciben un hueco
    r = align_multiple(["ACGTACGT", "ACGTTACGT", "ACGTACGT"])
    assert len({len(a) for a in r.aligned}) == 1
    assert r.length == 9
    # cada fila sin huecos = su original
    assert r.aligned[0].replace("-", "") == "ACGTACGT"
    assert r.aligned[1].replace("-", "") == "ACGTTACGT"


def test_delecion_se_alinea():
    r = align_multiple(["ACGTACGT", "ACGACGT", "ACGTACGT"])
    assert len({len(a) for a in r.aligned}) == 1
    assert r.aligned[1].replace("-", "") == "ACGACGT"


def test_center_es_la_mas_larga_por_defecto():
    r = align_multiple(["ACGT", "ACGTACGT", "ACGA"])
    assert r.center == 1                       # la de 8 bases


def test_center_explicito():
    r = align_multiple(["ACGTACGT", "ACGTACGT", "ACGTACGT"], center=2)
    assert r.center == 2


# ── Consenso ──────────────────────────────────────────────────────────────────

def test_consenso_mayoria():
    # posición 3: T, T, A → mayoría T
    r = align_multiple(["ACGTACGT", "ACGTACGT", "ACGAACGT"])
    assert r.consensus() == "ACGTACGT"


def test_consenso_columna_ambigua():
    # 4 secuencias, pos 0: A,C,G,T (empate 25%) → con umbral alto = N
    r = align_multiple(["ACGT", "CCGT", "GCGT", "TCGT"])
    cons = r.consensus(threshold=0.5)
    assert cons[0] == "N"                      # ninguna supera el 50%
    assert cons[1:] == "CGT"


# ── Casos borde y errores ─────────────────────────────────────────────────────

def test_una_sola_secuencia():
    r = align_multiple(["ACGTACGT"])
    assert r.aligned == ["ACGTACGT"]
    assert r.center == 0


def test_lista_vacia_falla():
    with pytest.raises(SequenceValueError):
        align_multiple([])


def test_secuencia_vacia_falla():
    with pytest.raises(SequenceValueError):
        align_multiple(["ACGT", "", "ACGT"])


def test_tipo_invalido_falla():
    with pytest.raises(SequenceTypeError):
        align_multiple(["ACGT", 1234, "ACGT"])


def test_longitudes_muy_distintas():
    # robustez: secuencias de longitudes dispares no rompen
    seqs = ["ACGTACGTACGT", "ACGT", "ACGTACGTACGTACGTACGT", "ACGTACGT"]
    r = align_multiple(seqs)
    assert len({len(a) for a in r.aligned}) == 1
    for original, fila in zip(seqs, r.aligned, strict=True):
        assert fila.replace("-", "") == original
