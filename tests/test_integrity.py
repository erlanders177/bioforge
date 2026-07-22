"""
tests/test_integrity.py — RED DE INTEGRIDAD contra la corrupción silenciosa.

Este fichero existe por un bug concreto: durante meses el MSA empaquetó toda
proteína como ADN y cada residuo no-ACGT se convirtió en 'N'. No reventó: devolvió
basura *plausible*. Pasó desapercibido porque NINGÚN test verificaba la propiedad
más básica —"los datos que entran salen intactos"— sobre PROTEÍNAS (solo se probaba
ADN).

La lección no es "arreglar el MSA": es que la corrupción silenciosa se combate con
INVARIANTES probados sobre datos aleatorios de AMBOS alfabetos, no con ejemplos
elegidos a mano. Cada test de aquí es una propiedad que debe cumplirse SIEMPRE:

  1. lo que codificas y decodificas vuelve idéntico            (ida y vuelta)
  2. una secuencia válida NUNCA produce símbolos corruptos     (no-pérdida)
  3. forzar el tipo equivocado FALLA en vez de corromper       (el guard)
  4. el MSA no altera ninguna secuencia, sea ADN o proteína    (invariante MSA)
  5. la tubería completa (import→traducir) no confunde tipos   (cruce de módulos)

Si algo aquí se pone rojo, hay riesgo de pérdida de datos en algún nivel.
"""

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from bioforge.biocore import (
    SeqType,
    SequenceValueError,
    SmartImporter,
)
from bioforge.msa import _infer_type, align_multiple
from bioforge.smart_translator import SmartTranslator

# Alfabetos canónicos en mayúscula → to_string() devuelve exactamente lo que entró.
_DNA = "ACGT"
_PROT = "ACDEFGHIKLMNPQRSTVWY"
_PROT_EXCLUSIVE = "EFILPQ"      # residuos imposibles en el alfabeto de nucleótidos

dna_seq = st.text(alphabet=_DNA, min_size=1, max_size=400)
prot_seq = st.text(alphabet=_PROT, min_size=1, max_size=400)
_SETTINGS = settings(max_examples=200, deadline=None,
                     suppress_health_check=[HealthCheck.too_slow])


def _imp(seq: str, t: SeqType):
    return SmartImporter.from_string(f">x\n{seq}\n", force_type=t)[0]


# ── 1. Ida y vuelta: codificar y decodificar es la identidad ──────────────────

@given(dna_seq)
@_SETTINGS
def test_roundtrip_adn(seq):
    assert _imp(seq, SeqType.NUCLEOTIDE).to_string() == seq


@given(prot_seq)
@_SETTINGS
def test_roundtrip_proteina(seq):
    """La propiedad que habría cazado el bug del MSA en el minuto uno."""
    assert _imp(seq, SeqType.PROTEIN).to_string() == seq


@given(prot_seq)
@_SETTINGS
def test_roundtrip_proteina_autodeteccion(seq):
    # con al menos un residuo exclusivo, la auto-detección debe acertar y no perder
    s = "EFILPQ" + seq
    p = SmartImporter.from_string(f">x\n{s}\n")[0]
    assert p.seq_type is SeqType.PROTEIN
    assert p.to_string() == s


# ── 2. No-pérdida: una secuencia VÁLIDA nunca genera símbolos corruptos ────────

@given(prot_seq)
@_SETTINGS
def test_proteina_valida_sin_perdida(seq):
    """Cero 'X' fabricadas: si el residuo estaba, sigue estando."""
    out = _imp(seq, SeqType.PROTEIN).to_string()
    assert out.count("X") == seq.count("X")     # no aparecen X de la nada
    assert len(out) == len(seq)


@given(dna_seq)
@_SETTINGS
def test_adn_valido_sin_perdida(seq):
    out = _imp(seq, SeqType.NUCLEOTIDE).to_string()
    assert out.count("N") == seq.count("N")
    assert len(out) == len(seq)


# ── 3. El guard: forzar el tipo equivocado FALLA, no corrompe ──────────────────

@given(st.text(alphabet=_PROT_EXCLUSIVE, min_size=20, max_size=200))
@_SETTINGS
def test_guard_rechaza_proteina_como_adn(seq):
    """Una secuencia inequívocamente proteica forzada como ADN debe reventar."""
    with pytest.raises(SequenceValueError):
        _imp(seq, SeqType.NUCLEOTIDE)


def test_guard_mensaje_es_util():
    with pytest.raises(SequenceValueError) as e:
        _imp("MKLPQEFILPQWY" * 4, SeqType.NUCLEOTIDE)
    msg = str(e.value)
    assert "%" in msg and ("proteína" in msg or "NUCLEOTIDE" in msg)


def test_guard_no_molesta_a_adn_con_algunas_N():
    # ADN real con N dispersas (baja fracción) NO debe saltar
    seq = "ATGCNNATGCGTANCGTAGCTAGCTAGCTAGCNTAGC"
    p = _imp(seq, SeqType.NUCLEOTIDE)
    assert p.to_string() == seq


def test_guard_no_falla_con_arrays_vacios():
    # caso borde: la medición de pérdida sobre nada no puede romper ni dividir por 0
    empty = np.array([], dtype=np.uint8)
    assert SmartImporter._encoding_loss(empty, empty, SeqType.NUCLEOTIDE) == 0.0
    assert SmartImporter._encoding_loss(empty, empty, SeqType.PROTEIN) == 0.0


# ── 4. Invariante del MSA: no altera ninguna secuencia, en ningún alfabeto ─────

def _variants(base, n, alphabet, seed):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        s = list(base)
        for p in rng.choice(len(s), max(1, len(s) // 20), replace=False):
            s[p] = alphabet[int(rng.integers(0, len(alphabet)))]
        out.append("".join(s))
    return out


@given(st.integers(min_value=0, max_value=10_000))
@settings(max_examples=40, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_msa_no_corrompe_proteina(seed):
    base = "MKTIIALSYIFCLVFAQKLPGNDNSTATLCLGHHAV"
    seqs = _variants(base, 6, _PROT, seed)
    r = align_multiple(seqs)
    assert _infer_type(seqs) is SeqType.PROTEIN
    for original, fila in zip(seqs, r.aligned, strict=True):
        assert fila.replace("-", "") == original          # degap = original


@given(st.integers(min_value=0, max_value=10_000))
@settings(max_examples=40, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_msa_no_corrompe_adn(seed):
    base = "ATGCGTACGTAGCTAGCATCGATCGATCGTAGCTAGC"
    seqs = _variants(base, 6, _DNA, seed)
    r = align_multiple(seqs)
    for original, fila in zip(seqs, r.aligned, strict=True):
        assert fila.replace("-", "") == original


# ── 5. Cruce de módulos: import → traducir no confunde tipos ───────────────────

@given(st.text(alphabet=_DNA, min_size=3, max_size=90).map(lambda s: "ATG" + s))
@_SETTINGS
def test_pipeline_traduccion_no_corrompe(seq):
    """De ADN a proteína y la proteína se re-empaqueta intacta (sin caer en 'N')."""
    seq = seq[: len(seq) - len(seq) % 3]                   # múltiplo de 3
    nuc = _imp(seq, SeqType.NUCLEOTIDE)
    prot = SmartTranslator.translate(nuc)
    # la proteína resultante debe re-importarse sin pérdida como PROTEÍNA
    txt = prot.to_string()
    if txt:
        reimport = _imp(txt, SeqType.PROTEIN)
        assert reimport.to_string() == txt
