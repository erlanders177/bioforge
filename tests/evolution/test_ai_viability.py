"""
tests/test_ai_viability.py — eje B (viabilidad con ESM-2, extra opcional bioforge[ai]).

Se SALTAN si torch/transformers no están instalados (el núcleo no los requiere).
Prueban que ESM-2 da salidas en rango y coherentes sobre una proteína real corta.
"""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from bioforge.evolution.ai import grammaticality_profile, viability_scores  # noqa: E402
from bioforge.core.biocore import SequenceValueError  # noqa: E402

# lisozima de clara de huevo (fragmento N-terminal real) — contexto natural para ESM-2
SEQ = "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRW"


def test_gramaticalidad_en_rango():
    p = grammaticality_profile(SEQ)
    assert p.shape == (len(SEQ),)
    assert (p >= 0).all() and (p <= 1).all()


def test_viability_scores_en_rango():
    sc = viability_scores(SEQ, {5: SEQ[5], 20: SEQ[20]})
    assert set(sc) == {5, 20}
    assert all(0.0 <= v <= 1.0 for v in sc.values())


def test_posicion_fuera_de_rango_falla():
    with pytest.raises(SequenceValueError):
        viability_scores(SEQ, {9999: "A"})


def test_residuo_natural_no_menos_viable_que_disruptivo():
    # en un sitio dado, el residuo natural suele ser >= una sustitución rara
    pos = 15
    wt = viability_scores(SEQ, {pos: SEQ[pos]})[pos]
    alt = "P" if SEQ[pos] != "P" else "W"
    disruptive = viability_scores(SEQ, {pos: alt})[pos]
    assert wt >= disruptive
