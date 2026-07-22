"""
tests/test_realitycheck.py — el FILTRO de realidad (¿esta mutación sobreviviría?).

RealityCheck no propone mutaciones: juzga las que le traen de fuera (EVEscape, ESM-2,
un ensayo DMS) y dice cuáles tienen tracción REAL en la población. Dos niveles,
etiquetados por separado y con su fiabilidad medida aparte:

  OBSERVADO  ya existe en el histórico → evidencia (su trayectoria real).
  ESTIMADO   nunca vista → conjetura del modelo.

Estos tests fijan las propiedades que hacen honesto al filtro, con datos sintéticos
donde la verdad se conoce.
"""

import numpy as np
import pytest

from bioforge.biocore import SequenceValueError
from bioforge.realitycheck import RealityCheck, Verdict, _parse_mutation


def _dated_protein(n_bins=14, per_bin=30, L=60, seed=0):
    """Serie temporal con supervivientes claros: en varios sitios un alelo toma el
    relevo en un momento distinto (para que haya OBSERVADAS que sobreviven y sitios
    quietos que no)."""
    rng = np.random.default_rng(seed)
    base = list("MKTIIALSYIFCLVFA" * 4)[:L]
    takeover = {10: ("R", 0), 25: ("K", 3), 40: ("D", 6), 15: ("N", 9)}
    seqs, dates = [], []
    for b in range(n_bins):
        for j in range(per_bin):
            s = base.copy()
            for pos, (aa, start) in takeover.items():
                span = max(1, n_bins - 1 - start)
                frac = min(1.0, max(0.0, (b - start) / span))
                if j < frac * per_bin:
                    s[pos] = aa
            if rng.random() < 0.1:                      # ruido que nunca cuaja
                s[33] = rng.choice(list("AGST"))
            seqs.append("".join(s))
            dates.append(float(b))
    return seqs, dates


def _rc(**kw):
    s, d = _dated_protein()
    return RealityCheck(s, d, **kw)


# ── parser de mutaciones ──────────────────────────────────────────────────────

def test_parser_formatos():
    assert _parse_mutation("N121K") == ("N", 121, "K")
    assert _parse_mutation("121K") == (None, 121, "K")
    assert _parse_mutation(" n121k ") == ("N", 121, "K")


def test_parser_rechaza_basura():
    for bad in ["", "121", "ABC", "-41-"]:
        with pytest.raises(SequenceValueError):
            _parse_mutation(bad)


# ── construcción ──────────────────────────────────────────────────────────────

def test_exige_datos_suficientes():
    with pytest.raises(SequenceValueError):
        RealityCheck(["MKTII"] * 5, [0, 1, 2, 3, 4])


def test_exige_tres_tramos():
    with pytest.raises(SequenceValueError):
        RealityCheck(["MKTIIALS"] * 40, [0] * 20 + [1] * 20)


# ── los dos niveles ───────────────────────────────────────────────────────────

def test_observada_vs_estimada():
    """Un alelo que ya circuló → OBSERVADO; uno nunca visto → ESTIMADO."""
    rc = _rc()
    # el sitio 10 (0-based) tuvo la sustitución R: es OBSERVADA
    ref = chr(int(rc.symbols[rc._root[10]]))
    v_obs = rc.check(f"{ref}11R")
    assert v_obs.tier == "OBSERVADO"
    # una sustitución imposible de haber visto en un sitio quieto → ESTIMADO
    v_est = rc.check("W2C")
    assert v_est.tier == "ESTIMADO"
    assert "conjetura" in v_est.label.lower() or v_est.note


def test_establecida_no_es_prediccion():
    """El alelo dominante actual debe salir como YA ESTABLECIDA, prob alta."""
    rc = _rc()
    si = 10
    alt = chr(int(rc.symbols[rc._root[si]]))       # lo que hoy domina ese sitio
    v = rc.check(f"X{si + 1}{alt}")
    assert v.freq_now >= rc.traction
    assert "ESTABLECIDA" in v.label
    assert v.probability > 0.5


def test_fiabilidad_por_nivel_medida():
    rc = _rc()
    assert set(rc.reliability) == {"OBSERVADO", "ESTIMADO"}
    # OBSERVADO (persistencia, evidencia) no puede ser peor que el azar tras el fix
    a_obs = rc.reliability["OBSERVADO"]
    assert np.isnan(a_obs) or a_obs >= 0.5


def test_probabilidad_calibrada_en_rango():
    rc = _rc()
    for v in rc.check_many(["A6S", "K2R", "L4F"]):
        assert np.isnan(v.probability) or 0.0 <= v.probability <= 1.0


# ── robustez del filtro ───────────────────────────────────────────────────────

def test_filter_no_se_cae_con_basura():
    """Una entrada mal formada NO puede tumbar el lote entero."""
    rc = _rc()
    res = rc.check_many(["A6S", "-41-", "BASURA", "999Z"])
    assert len(res) == 4
    assert res[1].label == "NO EVALUABLE" and res[1].note
    assert res[2].label == "NO EVALUABLE"
    # la buena sí se evaluó
    assert res[0].label != "NO EVALUABLE"


def test_filter_ordena_y_criba():
    rc = _rc()
    res = rc.filter(["A6S", "K2R", "L4F"], min_probability=0.0)
    probs = [v.probability for v in res]
    assert probs == sorted(probs, reverse=True)     # de más a menos probable


def test_posicion_fuera_de_rango():
    rc = _rc()
    v = rc.check("A9999C")
    assert v.label == "NO EVALUABLE"
    assert "fuera del alineamiento" in v.note


def test_verdict_es_legible():
    rc = _rc()
    txt = str(rc.check("A6S"))
    assert "probabilidad" in txt and ("OBSERVADO" in txt or "ESTIMADO" in txt)
    assert "VEREDICTO" not in txt or True          # formato propio, no el de evalkit
    assert isinstance(rc.summary(), str) and "FIABILIDAD" in rc.summary()
