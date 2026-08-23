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

from bioforge.core.biocore import SeqType, SequenceTypeError, SequenceValueError
from bioforge.align.msa import MSAResult, _infer_type, align_multiple


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


# ── Proteínas: NO corromper (regresión del bug del MSA-como-ADN) ──────────────
# Durante meses el MSA empaquetaba TODA secuencia como ADN: cada aminoácido que no
# fuera A/C/G/T se convertía en 'N'. Silencioso y devastador — corrompía la base
# entera del predictor L5. Toda la suite anterior usaba solo ACGT, así que nunca
# saltó. Estos tests prueban con residuos imposibles en ADN.

def _rng_prot(n, seed):
    rng = np.random.default_rng(seed)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    return "".join(aa[i] for i in rng.integers(0, len(aa), n))


def test_infer_type_distingue_proteina_de_adn():
    assert _infer_type(["ACGTACGT", "ACGTACGT"]) is SeqType.NUCLEOTIDE
    assert _infer_type(["MKLPWY", "MKLPWF"]) is SeqType.PROTEIN
    # basta que UNA del conjunto traiga una letra imposible en ADN
    assert _infer_type(["ACGTACGT", "ACGTACGW"]) is SeqType.PROTEIN


def test_proteina_no_se_corrompe():
    """La propiedad clave, ahora con AMINOÁCIDOS: quitar huecos = original."""
    base = _rng_prot(200, 11)
    seqs = []
    for i in range(8):
        s = list(base)
        rng = np.random.default_rng(100 + i)
        for p in rng.choice(len(s), 8, replace=False):
            s[p] = "ACDEFGHIKLMNPQRSTVWY"[int(rng.integers(0, 20))]
        seqs.append("".join(s))
    r = align_multiple(seqs)
    assert _infer_type(seqs) is SeqType.PROTEIN
    for original, fila in zip(seqs, r.aligned, strict=True):
        assert fila.replace("-", "") == original      # ni una 'N' fabricada
    # y ningún residuo raro W/E/F/P sobrevivió como 'N'
    juntas = "".join(r.aligned)
    assert set("WEFPYQ") & set(juntas)                 # los residuos siguen ahí


def test_proteina_forzada_explicita():
    """seq_type explícito manda sobre la inferencia."""
    seqs = ["MKTAC", "MKTAG", "MKTAA"]
    r = align_multiple(seqs, seq_type=SeqType.PROTEIN)
    for original, fila in zip(seqs, r.aligned, strict=True):
        assert fila.replace("-", "") == original


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
