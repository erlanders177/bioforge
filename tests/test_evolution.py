"""
tests/test_evolution.py — predictor de evolución (Fase 1: base + backtesting).

Las propiedades que DEBE cumplir un predictor honesto y genoma-agnóstico:
  - ante un BARRIDO SELECTIVO (un alelo que sube en el tiempo), el método de
    tendencia debe **ganar a la baseline ingenua** (skill > 0) — el listón de "útil";
  - ante datos ESTÁTICOS, NO debe fingir mejora (skill ≈ 0, no negativo por ruido);
  - descubre de los datos los "sitios que cambian" (sin hardcodear organismo);
  - funciona igual sobre un alfabeto de PROTEÍNA (genoma-agnóstico de verdad);
  - errores y casos borde.
"""

import numpy as np
import pytest

from bioforge.biocore import SequenceTypeError, SequenceValueError
from bioforge.evolution import (
    BacktestResult,
    CladePrediction,
    EscapeResult,
    EvolutionResult,
    FusionResult,
    GrowthResult,
    _assign_lineages,
    _clade_labels,
    _dissimilarity,
    _own,
    _prepare,
    _project_freqs,
    _renest,
    _shrink_slopes,
    backtest_evolution,
    designate_lineages,
    escape_potential,
    escape_weights,
    estimate_growth,
    predict_clade,
    predict_evolution,
    predict_fusion,
    site_mutability,
)


def _clade_dataset():
    """Dos clados nítidos: A='AAA' y B='TTT' en sitios 0-2 (resto G). B sube 0.1→0.9."""
    fracB = [0.1, 0.3, 0.5, 0.7, 0.9]
    npb, L = 20, 6
    seqs, times = [], []
    for b, fr in enumerate(fracB):
        nB = round(fr * npb)
        for j in range(npb):
            row = ["G"] * L
            trip = "T" if j < nB else "A"      # clado B (T) o A (A)
            row[0] = row[1] = row[2] = trip
            seqs.append("".join(row))
            times.append(b)
    return seqs, times


def _two_site_protein():
    """Proteína, 2 sitios: 0 = D→K rápido+disruptivo; 1 = L→I lento+conservador."""
    f0 = [0.10, 0.35, 0.60, 0.80, 0.95]        # sitio 0: rápido
    f1 = [0.20, 0.35, 0.50, 0.62, 0.72]        # sitio 1: lento
    npb, L = 20, 4
    seqs, times = [], []
    for b in range(5):
        n0, n1 = round(f0[b] * npb), round(f1[b] * npb)
        for j in range(npb):
            row = ["M"] * L
            row[0] = "K" if j < n0 else "D"
            row[1] = "I" if j < n1 else "L"
            seqs.append("".join(row))
            times.append(b)
    return seqs, times


def _traj_dataset(fractions, L=6, site=0, fixed="G", new="A", old="C", npb=10):
    """Dataset con una trayectoria de frecuencia dada para `new` en `site`."""
    seqs, times = [], []
    for b, fr in enumerate(fractions):
        n_new = round(fr * npb)
        for j in range(npb):
            base = [fixed] * L
            base[site] = new if j < n_new else old
            seqs.append("".join(base))
            times.append(b)
    return seqs, times


def _sweep_dataset(n_per_bin=10, n_bins=10, L=10, sweep_site=0,
                   fixed="G", allele_new="A", allele_old="C"):
    """Barrido selectivo: en `sweep_site`, la fracción de `allele_new` sube 0→0.9
    a lo largo de los bins; el resto de sitios fijos en `fixed`. Consenso de ese
    sitio vuelca de old→new a mitad de camino."""
    seqs, times = [], []
    for b in range(n_bins):
        n_new = b                                   # 0,1,...,n_bins-1 de n_per_bin
        for j in range(n_per_bin):
            base = [fixed] * L
            base[sweep_site] = allele_new if j < n_new else allele_old
            seqs.append("".join(base))
            times.append(b)
    return seqs, times


# ── El listón de "útil": ganar a la baseline ─────────────────────────────────

def test_tendencia_gana_a_ingenua_en_barrido():
    seqs, times = _sweep_dataset()
    r = backtest_evolution(seqs, times, method="trend")
    assert isinstance(r, BacktestResult)
    assert r.beats_naive
    assert r.skill > 0.0
    assert r.method_accuracy > r.naive_accuracy


def test_no_finge_mejora_en_datos_estaticos():
    # todas idénticas en el tiempo → no hay nada que predecir mejor que persistir
    seqs = ["ACGTACGTAC"] * 60
    times = list(np.repeat(np.arange(6), 10))
    r = backtest_evolution(seqs, times, method="trend")
    assert r.naive_accuracy == 1.0            # persistir acierta todo
    assert r.method_accuracy == 1.0           # tendencia no lo estropea
    assert r.skill == 0.0                     # no hay mejora que reclamar (ni daño)


def test_ingenua_contra_si_misma_skill_cero():
    seqs, times = _sweep_dataset()
    r = backtest_evolution(seqs, times, method="naive")
    assert r.skill == 0.0                     # la ingenua no se gana a sí misma
    assert not r.beats_naive


# ── Predicción ────────────────────────────────────────────────────────────────

def test_predice_el_alelo_en_ascenso():
    # el barrido termina con 'A' dominante y subiendo → la predicción del próximo
    # consenso en el sitio 0 debe ser 'A'
    seqs, times = _sweep_dataset()
    r = predict_evolution(seqs, times, method="trend")
    assert isinstance(r, EvolutionResult)
    assert r.predicted_aligned[0] == "A"


def test_descubre_sitios_que_cambian():
    seqs, times = _sweep_dataset(sweep_site=3)
    r = predict_evolution(seqs, times)
    assert r.changing_sites == [3]            # solo el sitio barrido cambia


def test_longitud_prediccion_coincide_con_columnas():
    seqs, times = _sweep_dataset()
    r = predict_evolution(seqs, times)
    assert len(r.predicted_aligned) == r.n_sites
    assert r.n_sites == 10


# ── Genoma-agnóstico: proteína ────────────────────────────────────────────────

def test_funciona_sobre_alfabeto_proteina():
    # barrido con residuos de aminoácido (no ACGT) — no debe asumir nucleótidos
    seqs, times = _sweep_dataset(fixed="M", allele_new="K", allele_old="R")
    r = backtest_evolution(seqs, times, method="trend")
    assert r.beats_naive
    pred = predict_evolution(seqs, times)
    assert set(pred.predicted_aligned) <= {"M", "K", "R"}


# ── Alineamiento automático de longitudes dispares ───────────────────────────

def test_alinea_si_longitudes_difieren():
    # con align=True, secuencias de distinta longitud se alinean antes
    seqs = ["ACGTACGT", "ACGTTACGT", "ACGTACGT", "ACGTTACGT",
            "ACGTACGT", "ACGTTACGT"]
    times = [0, 0, 1, 1, 2, 2]
    r = predict_evolution(seqs, times, align=True)
    assert isinstance(r, EvolutionResult)


def test_longitudes_dispares_sin_align_falla():
    seqs = ["ACGT", "ACGTACGT", "ACGT", "ACGTACGT"]
    times = [0, 0, 1, 1]
    with pytest.raises(SequenceValueError):
        predict_evolution(seqs, times, align=False)


# ── Errores y casos borde ─────────────────────────────────────────────────────

def test_longitudes_desiguales_seq_times_falla():
    with pytest.raises(SequenceValueError):
        predict_evolution(["ACGT", "ACGT"], [0])


def test_menos_de_dos_secuencias_falla():
    with pytest.raises(SequenceValueError):
        predict_evolution(["ACGT"], [0])


def test_tipo_invalido_falla():
    with pytest.raises(SequenceTypeError):
        predict_evolution(["ACGT", 123, "ACGT"], [0, 1, 2])


def test_metodo_desconocido_falla():
    seqs, times = _sweep_dataset()
    with pytest.raises(SequenceValueError):
        predict_evolution(seqs, times, method="magia")


def test_backtesting_necesita_tres_bins():
    seqs = ["ACGT", "ACGT", "ACGT", "ACGT"]
    times = [0, 0, 1, 1]                       # solo 2 bins
    with pytest.raises(SequenceValueError):
        backtest_evolution(seqs, times)


# ══════════════════════════════════════════════════════════════════════════════
# Fase 2 — términos de fitness (eje A): crecimiento log-lineal (FGA/MLR) y GARW
# ══════════════════════════════════════════════════════════════════════════════

def test_fitness_recupera_direccion_del_crecimiento():
    # 'A' sube, 'C' baja → fitness(A) > 0 > fitness(C)
    seqs, times = _traj_dataset([0.1, 0.3, 0.5, 0.7, 0.9])
    g = estimate_growth(seqs, times)
    assert isinstance(g, GrowthResult)
    assert g.site_growth[0]["A"] > 0 > g.site_growth[0]["C"]
    assert g.rising[0] == "A"                   # el alelo en ascenso


def test_garw_capta_reversion_reciente():
    # sube y LUEGO cae: la tasa reciente (GARW) es más negativa que la fija (FGA)
    fr = [0.1, 0.3, 0.5, 0.7, 0.6, 0.5, 0.4, 0.3]
    seqs, times = _traj_dataset(fr)
    fga = estimate_growth(seqs, times, garw=False)
    garw = estimate_growth(seqs, times, garw=True)
    assert garw.site_growth[0]["A"] < fga.site_growth[0]["A"]


_EXP = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]


# HALLAZGO HONESTO: en el argmax de consenso a UN paso, la persistencia (naive) es
# muy fuerte; logistic/GARW NO la ganan de forma fiable (solo el modelo que coincide
# con la dinámica, p.ej. 'trend' en datos lineales). El valor de logistic/GARW es el
# FITNESS estimado (eje A) y la dinámica temporal, no el argmax de un paso. Testeamos
# eso — lo que es cierto — no una victoria inventada.

def test_trend_gana_a_naive_solo_donde_su_modelo_encaja():
    # el árbitro funcionando: en dinámica lineal gradual, el modelo lineal gana
    lin = _sweep_dataset()
    assert backtest_evolution(*lin, method="trend").skill > 0.0


def test_logistico_identifica_al_alelo_en_ascenso():
    # lo honesto que logistic SÍ cumple: señala al alelo que sube como el ganador
    seqs, times = _traj_dataset(_EXP, npb=100)
    assert predict_evolution(seqs, times, method="logistic").predicted_aligned[0] == "A"


def test_garw_identifica_al_alelo_en_ascenso():
    seqs, times = _traj_dataset(_EXP, npb=100)
    assert predict_evolution(seqs, times, method="garw").predicted_aligned[0] == "A"


def test_logistico_no_finge_mejora_en_estatico():
    seqs = ["ACGTAC"] * 60
    times = list(np.repeat(np.arange(6), 10))
    r = backtest_evolution(seqs, times, method="logistic")
    assert r.skill == 0.0                       # no daña ni inventa


def test_prediccion_logistica_es_agnostica_de_alfabeto():
    seqs, times = _sweep_dataset(fixed="M", allele_new="K", allele_old="R")
    r = predict_evolution(seqs, times, method="logistic")
    assert set(r.predicted_aligned) <= {"M", "K", "R"}


def test_estimate_growth_necesita_dos_bins():
    seqs = ["ACGT", "ACGT"]
    times = [0, 0]                              # 1 solo bin
    with pytest.raises(SequenceValueError):
        estimate_growth(seqs, times)


# ══════════════════════════════════════════════════════════════════════════════
# Fase 3 — término de escape (eje C): disimilitud físico-química (EVEscape)
# ══════════════════════════════════════════════════════════════════════════════

def test_disimilitud_identicos_es_cero():
    assert _dissimilarity("K", "K") == 0.0


def test_disimilitud_carga_opuesta_es_alta():
    # D (−) vs K (+): salto de carga máximo → escape alto
    assert _dissimilarity("D", "K") > 0.4


def test_disimilitud_similares_es_baja():
    # I y L: ambos hidrófobos, sin carga → cambio conservador
    assert _dissimilarity("I", "L") < 0.2


def test_escape_alto_para_cambio_disruptivo():
    # residuo original D (−) reemplazado por K (+) en ascenso → escape alto
    seqs, times = _traj_dataset(fr := [0.1, 0.3, 0.5, 0.7, 0.9],
                                fixed="M", new="K", old="D")
    e = escape_potential(seqs, times)
    assert isinstance(e, EscapeResult)
    assert e.change[0] == "D→K"
    assert e.site_escape[0] > 0.4
    assert e.ranked[0][0] == 0                  # el sitio disruptivo, arriba del ranking


def test_escape_bajo_para_cambio_conservador():
    seqs, times = _traj_dataset([0.1, 0.3, 0.5, 0.7, 0.9],
                                fixed="M", new="I", old="L")
    e = escape_potential(seqs, times)
    assert e.site_escape[0] < 0.2              # I↔L conservador → poco escape


def test_escape_rechaza_nucleotido():
    seqs, times = _sweep_dataset()             # ACGT → no es proteína
    with pytest.raises(SequenceValueError):
        escape_potential(seqs, times)


def test_escape_es_agnostico_entre_proteinas():
    # otro par de residuos, otro organismo hipotético — no asume nada
    seqs, times = _traj_dataset([0.2, 0.5, 0.8], fixed="A", new="R", old="E")
    e = escape_potential(seqs, times)
    assert e.change[0] == "E→R"                 # E(−) → R(+)
    assert e.site_escape[0] > 0.4


# ══════════════════════════════════════════════════════════════════════════════
# Fase 5 — fusión A+B+C (el predictor integrado, arquitectura tipo EVEscape)
# ══════════════════════════════════════════════════════════════════════════════

def test_fusion_prioriza_rapido_y_disruptivo():
    seqs, times = _two_site_protein()
    f = predict_fusion(seqs, times)
    assert isinstance(f, FusionResult)
    assert f.ranked[0][0] == 0                  # sitio 0 (rápido+disruptivo) arriba
    assert "growth" in f.used and "escape" in f.used
    assert "viability" not in f.used            # sin eje B si no se aporta


def test_fusion_degrada_con_gracia_en_nucleotido():
    seqs, times = _sweep_dataset()              # ADN → sin eje C
    f = predict_fusion(seqs, times)             # no debe fallar
    assert f.used == ["growth"]                 # solo eje A
    assert 0 in f.site_score


def test_fusion_enchufa_viabilidad_eje_b():
    seqs, times = _two_site_protein()
    f = predict_fusion(seqs, times, viability={0: 1.0, 1: 0.0})
    assert "viability" in f.used                # el eje B (ESM-2) se enchufa
    assert "viability" in f.terms[0]


def test_fusion_respeta_pesos():
    seqs, times = _two_site_protein()
    # solo crecimiento: el escape se ignora → score del sitio 0 = su crecimiento norm.
    f = predict_fusion(seqs, times, weights={"growth": 1.0, "escape": 0.0})
    assert f.site_score[0] == f.terms[0]["growth"]


# ══════════════════════════════════════════════════════════════════════════════
# Clados / linajes — la vía que compite con la baseline (estilo evofr/MLR)
# ══════════════════════════════════════════════════════════════════════════════

def test_clados_separa_dos_linajes():
    seqs, times = _clade_dataset()
    arr, _, symbols = _prepare(seqs, times, align=True)
    labels, m = _clade_labels(arr, symbols, n_clades=5, min_count=3, key_sites=6)
    assert m == 2                              # exactamente dos haplotipos definitorios
    assert len(set(labels.tolist())) == 2


def test_clados_predice_el_linaje_en_ascenso():
    # el clado B (TTT) sube → la predicción debe ser su consenso "TTTGGG"
    seqs, times = _clade_dataset()
    r = predict_clade(seqs, times, n_clades=5, key_sites=6)
    assert isinstance(r, CladePrediction)
    assert r.predicted_aligned == "TTTGGG"     # gana el clado en ascenso, mutaciones enlazadas
    assert r.n_clades == 2


def test_clados_frecuencia_proyectada_favorece_al_que_sube():
    seqs, times = _clade_dataset()
    r = predict_clade(seqs, times, n_clades=5, key_sites=6)
    dom = r.dominant_clade
    # el clado dominante previsto es el de mayor frecuencia proyectada
    assert r.clade_projected[dom] == max(r.clade_projected.values())


def test_clados_agnostico_de_alfabeto_proteina():
    # mismo patrón con aminoácidos → no asume nucleótidos
    seqs = [s.replace("A", "K").replace("T", "R").replace("G", "M")
            for s in _clade_dataset()[0]]
    times = _clade_dataset()[1]
    r = predict_clade(seqs, times, n_clades=5, key_sites=6)
    assert r.predicted_aligned == "RRRMMM"     # clado en ascenso (era T→R)


# ── mutabilidad por sitio (clado variable) ────────────────────────────────────

def test_mutabilidad_detecta_el_sitio_que_cambia():
    # en el barrido, solo el sitio 0 cambia en el tiempo → debe ser el más mutable
    seqs, times = _sweep_dataset()
    mut = site_mutability(seqs, times)
    assert 0 in mut                            # el sitio barrido aparece
    assert mut[0] == max(mut.values())         # y es el de mayor mutabilidad


def test_mutabilidad_estatico_vacio_o_bajo():
    # datos sin cambio temporal → no hay sitios notablemente mutables
    seqs = ["ACGTAC"] * 60
    times = list(np.repeat(np.arange(6), 10))
    mut = site_mutability(seqs, times)
    assert all(v < 1e-6 for v in mut.values()) or mut == {}


# ── linajes ESTABLES: definitorias + GRI (Pango/autolin sin árbol) ────────────

def _lineage_dataset(n_early=150, n_late=250, seed=0):
    """Estructura VERDADERA anidada: raíz → A[10,20] → A.1[+30,40]; raíz → B[50,60].
    Como en la realidad, la raíz domina al principio y los derivados llegan después."""
    rng = np.random.default_rng(seed)
    L = 120
    truth = {0: [], 1: [10, 20], 2: [10, 20, 30, 40], 3: [50, 60]}
    early = rng.choice(4, n_early, p=[0.75, 0.15, 0.05, 0.05])
    late = rng.choice(4, n_late, p=[0.05, 0.30, 0.45, 0.20])
    lab = np.concatenate([early, late])
    arr = np.full((len(lab), L), ord("A"), dtype=np.uint8)
    for i, t in enumerate(lab):
        arr[i, truth[int(t)]] = ord("G")
    arr[rng.random(arr.shape) < 0.01] = ord("C")          # 1% de ruido
    return arr, np.unique(arr), lab


def test_linajes_recuperan_la_estructura_verdadera():
    # designar con lo antiguo (fija el ancestro) y extender — el uso real
    arr, symbols, truth = _lineage_dataset()
    sysx = designate_lineages(arr[:150], symbols, min_size=10)
    sysx = designate_lineages(arr, symbols, prior=sysx, min_size=10)
    labels = _assign_lineages(arr, sysx)
    for t in range(4):                       # cada grupo verdadero → un linaje, puro
        m = truth == t
        c = np.bincount(labels[m], minlength=sysx.n)
        assert c.max() / m.sum() > 0.90


def test_linajes_definitorias_son_mutaciones_derivadas():
    # una definitoria es un cambio RESPECTO AL ANCESTRO, nunca el alelo ancestral
    arr, symbols, _ = _lineage_dataset()
    sysx = designate_lineages(arr[:150], symbols, min_size=10)
    sysx = designate_lineages(arr, symbols, prior=sysx, min_size=10)
    for i in range(1, sysx.n):
        assert sysx.sites[i].size > 0
        assert np.all(sysx.alleles[i] != sysx.root[sysx.sites[i]])


def test_linajes_estables_al_extender():
    # la disciplina Pango: los linajes ya designados NO cambian de identidad
    arr, symbols, _ = _lineage_dataset()
    base = designate_lineages(arr[:150], symbols, min_size=10)
    ext = designate_lineages(arr, symbols, prior=base, min_size=10)
    assert ext.n >= base.n
    assert np.array_equal(ext.root, base.root)                  # el ancla no se mueve
    for i in range(base.n):
        assert np.array_equal(ext.sites[i], base.sites[i])      # identidad congelada
        assert np.array_equal(ext.alleles[i], base.alleles[i])


def test_jerarquia_anidada_y_aciclica():
    # A.1 lleva todo lo de A y algo más ⇒ debe COLGAR de A (no de la raíz)
    arr, symbols, _ = _lineage_dataset()
    sysx = designate_lineages(arr[:150], symbols, min_size=10)
    sysx = designate_lineages(arr, symbols, prior=sysx, min_size=10)
    assert sysx.parents[0] == -1
    for i in range(1, sysx.n):
        p = int(sysx.parents[i])
        assert p < i or sysx.sites[p].size < sysx.sites[i].size  # padre ⊂ hijo estricto
        seen, cur = set(), i                                     # y no hay ciclos
        while cur >= 0:
            assert cur not in seen
            seen.add(cur)
            cur = int(sysx.parents[cur])
    deep = max(range(sysx.n), key=lambda i: sysx.sites[i].size)
    assert sysx.parents[deep] > 0              # el más profundo no cuelga de la raíz


def test_renest_por_contencion():
    # contención pura: {} ⊂ {10G} ⊂ {10G,20G}
    sites = [np.array([], dtype=np.intp), np.array([10]), np.array([10, 20])]
    alleles = [np.array([], dtype=np.uint8), np.array([71], dtype=np.uint8),
               np.array([71, 71], dtype=np.uint8)]
    parents = _renest(sites, alleles)
    assert list(parents) == [-1, 0, 1]


def test_own_quita_las_heredadas():
    arr, symbols, _ = _lineage_dataset()
    sysx = designate_lineages(arr[:150], symbols, min_size=10)
    sysx = designate_lineages(arr, symbols, prior=sysx, min_size=10)
    for i in range(1, sysx.n):
        own_s, _ = _own(sysx, i)
        p = int(sysx.parents[i])
        assert own_s.size == sysx.sites[i].size - sysx.sites[p].size


def test_numero_de_linajes_no_se_fija_a_dedo():
    # a diferencia del clustering tosco (n_clades), sale de los umbrales
    arr, symbols, _ = _lineage_dataset()
    pocos = designate_lineages(arr, symbols, min_size=250)      # umbral alto → menos
    muchos = designate_lineages(arr, symbols, min_size=10)
    assert pocos.n < muchos.n


def test_sin_variacion_un_solo_linaje():
    arr = np.full((50, 30), ord("A"), dtype=np.uint8)
    sysx = designate_lineages(arr, np.unique(arr), min_size=5)
    assert sysx.n == 1
    assert np.all(_assign_lineages(arr, sysx) == 0)


# ── shrinkage: crecimiento regularizado (evofr) y propagado (Łuksza) ──────────

def test_shrink_encoge_lo_poco_evidenciado_hacia_la_media():
    slope = np.array([2.0, -2.0])
    weight = np.array([1.0, 1.0])              # casi sin evidencia → casi todo prior
    out = _shrink_slopes(slope, weight, kappa=100.0)
    assert abs(out[0]) < 0.1 and abs(out[1]) < 0.1


def test_shrink_respeta_lo_muy_evidenciado():
    slope = np.array([2.0, -2.0])
    weight = np.array([1e6, 1e6])              # evidencia abrumadora → casi el dato
    out = _shrink_slopes(slope, weight, kappa=30.0)
    assert np.allclose(out, slope, atol=1e-3)


def test_shrink_por_arbol_hereda_del_padre():
    # hijo sin evidencia dentro de un padre que crece → hereda la tendencia del padre
    slope = np.array([0.0, 1.0, -5.0])         # el -5 del hijo es ruido
    weight = np.array([1e6, 1e6, 1.0])
    parents = np.array([-1, 0, 1])
    sizes = np.array([0, 1, 2])
    out = _shrink_slopes(slope, weight, 100.0, parents=parents, sizes=sizes)
    assert out[2] > 0.9                        # arrastrado hacia la tasa del padre (~1)


# ── GRI ponderado: la puerta del conocimiento externo (ejes B/C) ──────────────

def test_escape_weights_forma_y_rango():
    symbols = np.frombuffer(b"AKDEP", dtype=np.uint8)
    root = np.frombuffer(b"AAKKD", dtype=np.uint8)
    w = escape_weights(symbols, root)
    assert w.shape == (5, 5)
    assert np.all(w >= 1.0) and np.all(w <= 2.0)
    a_idx = int(np.where(symbols == ord("A"))[0][0])
    assert w[a_idx, 0] == 1.0                  # A→A no es mutación: sin disimilitud


def test_escape_weights_exige_proteina():
    symbols = np.frombuffer(b"ACGT", dtype=np.uint8)
    with pytest.raises(SequenceValueError):
        escape_weights(symbols, np.frombuffer(b"ACGT", dtype=np.uint8))


def test_mut_weights_inclinan_la_designacion():
    # pesar mucho un sitio hace que su mutación merezca definir linaje antes
    arr, symbols, _ = _lineage_dataset()
    raiz = designate_lineages(arr[:150], symbols, max_lineages=1)   # ancla el ancestro
    plano = designate_lineages(arr, symbols, prior=raiz, min_size=10, max_lineages=2)
    w = np.ones(arr.shape[1])
    w[[50, 60]] = 50.0                         # el clado B pasa a pesar mucho
    pesado = designate_lineages(arr, symbols, prior=raiz, min_size=10, max_lineages=2,
                                mut_weights=w)
    assert not np.array_equal(plano.sites[1], pesado.sites[1])
    assert set(pesado.sites[1].tolist()) == {50, 60}   # ahora B merece ser el primero


# ── proyección MLR: preserva ceros y se reduce a la ingenua sin señal ─────────

def test_proyeccion_preserva_los_linajes_extintos():
    # un linaje a 0 debe seguir a 0: el softmax sobre logits le regalaba masa
    cf = np.array([[0.5, 0.5, 0.0], [0.6, 0.4, 0.0], [0.7, 0.3, 0.0]])
    p = _project_freqs(cf, False)
    assert p[2] == 0.0
    assert abs(p.sum() - 1.0) < 1e-9


def test_proyeccion_sin_crecimiento_es_la_ingenua():
    # trayectoria plana → r=0 → el modelo NO puede perder contra persistir
    cf = np.array([[0.4, 0.6], [0.4, 0.6], [0.4, 0.6]])
    assert np.allclose(_project_freqs(cf, False), cf[-1], atol=1e-6)


def test_proyeccion_sigue_al_que_sube():
    cf = np.array([[0.9, 0.1], [0.7, 0.3], [0.5, 0.5]])
    p = _project_freqs(cf, False)
    assert p[1] > cf[-1][1]                    # el que crece, sigue creciendo
    assert abs(p.sum() - 1.0) < 1e-9


def test_shrink_total_devuelve_la_ingenua():
    # κ enorme → tasas ≈ 0 → la predicción colapsa a persistir (interpolación continua)
    cf = np.array([[0.9, 0.1], [0.7, 0.3], [0.5, 0.5]])
    counts = np.full((3, 2), 5.0)
    p = _project_freqs(cf, False, counts=counts, shrink=1e6)
    assert np.allclose(p, cf[-1], atol=1e-3)
