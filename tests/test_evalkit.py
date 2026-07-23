"""
tests/test_evalkit.py — el JUEZ honesto de predictores (L6).

Cada test comprueba que el juez caza un autoengaño REAL, de los que sufrimos
midiendo de verdad: el azar disfrazado, el eje tautológico, el que solo acierta
lo fácil, y el que recuerda en vez de predecir.
"""

import numpy as np
import pytest

from bioforge.biocore import SequenceValueError
from bioforge.evalkit import EvolutionBenchmark


def _dated_protein(n_bins=14, per_bin=30, L=60, seed=0):
    """Serie temporal sintética CON señal, ESCALONADA en el tiempo.

    Cada sitio arranca su subida en un bin distinto, de modo que hay mutaciones
    "que suben" tanto al principio como al final. Sin ese escalonado, una época
    se queda sin positivos y las pruebas por era (fuga) no se pueden medir.
    """
    rng = np.random.default_rng(seed)
    base = list("MKTIIALSYIFCLVFA" * 4)[:L]
    seqs, dates = [], []
    rising = {10: ("R", 0), 25: ("K", 3), 40: ("D", 6),     # sitio: (alelo, inicio)
              15: ("N", 8), 50: ("Y", 10)}
    for b in range(n_bins):
        for j in range(per_bin):
            s = base.copy()
            for pos, (aa, start) in rising.items():
                span = max(1, n_bins - 1 - start)
                frac = min(1.0, max(0.0, (b - start) / span))
                if j < frac * per_bin:
                    s[pos] = aa
            for pos in (5, 33):                    # ruido en sitios fijos
                if rng.random() < 0.1:
                    s[pos] = rng.choice(list("AGST"))
            seqs.append("".join(s))
            dates.append(float(b))
    return seqs, dates


def _bench():
    s, d = _dated_protein()
    return EvolutionBenchmark(s, d)


def test_exige_datos_suficientes():
    with pytest.raises(SequenceValueError):
        EvolutionBenchmark(["MKTII"] * 5, [0, 1, 2, 3, 4])


def test_exige_tres_bins():
    with pytest.raises(SequenceValueError):
        EvolutionBenchmark(["MKTIIALS"] * 20, [0] * 10 + [1] * 10)


def test_caza_al_predictor_aleatorio():
    """Azar → el IC95% toca 0.5 y el veredicto lo dice."""
    b = _bench()
    rng = np.random.default_rng(1)
    r = b.judge(lambda ctx: rng.random(ctx.sites.size), n_boot=15)
    assert not r.beats_trivial or r.ci95[0] <= 0.5
    assert "NO" in r.verdict or "no" in r.verdict


def test_times_por_bin_consistente_con_n_bins():
    """Regresión: self.times debe tener longitud nb (tiempo POR BIN), aun cuando
    el usuario pasa n_bins ≠ nº de instantes. Si no, el detector de fuga indexa
    self.times[bin] con un mapa temporal equivocado."""
    s, d = _dated_protein()
    for nbins in (None, 5, 8):
        b = EvolutionBenchmark(s, d, n_bins=nbins)
        assert len(b.times) == b.nb
        assert np.all(np.diff(b.times) > 0)      # tiempos de bin, crecientes


def test_el_liston_no_es_el_azar():
    """El listón debe ser el MEJOR eje trivial, nunca 0.5 — nuestro error de hoy."""
    b = _bench()
    rng = np.random.default_rng(2)
    r = b.judge(lambda ctx: rng.random(ctx.sites.size), n_boot=10)
    assert r.best_trivial > 0.5
    assert r.best_trivial_name != "azar"


def test_caza_al_tautologico():
    """Devolver un eje trivial NO puede considerarse aportar."""
    from bioforge.evolution import _mutability, _mutability_gate
    b = _bench()
    r = b.judge(lambda ctx: _mutability_gate(_mutability(ctx.freq))[ctx.sites],
                n_boot=10)
    assert not r.beats_trivial


def test_context_es_leak_free():
    """El predictor solo puede ver bins ANTERIORES al que se evalúa."""
    b = _bench()
    vistos = []

    def espia(ctx):
        vistos.append(ctx.freq.shape[0])
        return np.zeros(ctx.sites.size)
    b.judge(espia, n_boot=5)
    assert vistos and max(vistos) < b.nb      # nunca ve el bin objetivo


def test_detecta_la_fuga_temporal():
    """Un predictor que ESPÍA el futuro antes del corte debe salir marcado."""
    b = _bench()
    rng = np.random.default_rng(3)
    cut = 7.0

    def con_fuga(ctx):
        k = ctx.freq.shape[0]
        n = ctx.sites.size
        if b.times[k + ctx.horizon - 1] <= cut:        # época "memorizada"
            truth = (b.freq[k + ctx.horizon - 1] - ctx.freq[-1])
            return truth[ctx.alleles, ctx.sites] + rng.normal(0, .01, n)
        return rng.random(n)
    r = b.judge(con_fuga, n_boot=10, leak_cutoff=cut)
    assert r.leakage is not None and r.leakage < 0


def test_informe_es_legible():
    b = _bench()
    r = b.judge(lambda ctx: np.zeros(ctx.sites.size), n_boot=5)
    txt = str(r)
    assert "VEREDICTO" in txt and "listón trivial" in txt
