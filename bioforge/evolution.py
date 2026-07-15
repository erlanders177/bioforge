"""
evolution.py
══════════════════════════════════════════════════════════════════════
Predictor de evolución de secuencias — GENOMA-AGNÓSTICO (Fase 1: base + backtesting).

El norte del proyecto: no ganar en velocidad pura, sino ser la ÚNICA caja integrada
que además **predice hacia dónde evoluciona** un conjunto de secuencias fechadas
(cepas de un virus a lo largo del tiempo, un gen bajo presión selectiva, etc.).

Honestidad primero (regla del proyecto):
  - Esto NO es ciencia puntera. evofr (Bedford Lab) y EVEscape (Marks Lab, Nature
    2023) ya hacen forecasting de evolución mejor. Nuestro nicho es ACCESIBILIDAD +
    INTEGRACIÓN + HONESTIDAD: corre en un portátil, se entiende, y te dice su propia
    incertidumbre midiéndose contra la baseline trivial.
  - El listón de "útil, no inútil" NO es opinión, es MEDIDA: la predicción debe
    **ganar a la baseline ingenua** ("la próxima = la más reciente"). Si no la gana,
    no vale. Por eso el backtesting está HORNEADO aquí, no es un extra.

Genoma-agnóstico DE VERDAD (requisito del usuario — "que funcione en todo tipo de
genomas"): no se hardcodea nada de gripe ni de ningún organismo. El alfabeto
(4 nucleótidos o 20 aminoácidos, con o sin huecos) se infiere de los propios datos.
Los "sitios que importan" NO se dan por sabidos: se DESCUBREN como las columnas cuya
frecuencia alélica cambia de forma dirigida en el tiempo. Eso es lo que lo hace
universal, y es una mejora conceptual sobre los modelos que hardcodean los epítopos.

Método (Fase 1 — la base mecanicista, sin GPU):
  1. Se alinean las secuencias (nuestro MSA) y se ordenan/agrupan por tiempo en bins.
  2. Por cada columna del alineamiento se calcula la **trayectoria de frecuencia** de
     cada alelo a lo largo de los bins.
  3. Predicción del próximo consenso, dos métodos comparables:
       - "naive"  : el consenso más reciente persiste (la baseline a batir).
       - "trend"  : extrapolación lineal de cada trayectoria un paso al futuro
                    (el alelo en ascenso "gana" antes que en la persistencia).
  4. **Backtesting** rodante: entrena hasta el bin k, predice el bin k, compara con
     el real, y reporta la exactitud por-sitio del método **frente a la ingenua**.

Fases futuras (documentadas, no aquí): crecimiento logístico/fitness (evofr),
híbrido clado+sitio, y prior opcional de modelo de lenguaje de proteínas (ESM-2),
siempre medidos contra esta base antes de enviarse.

Regla de oro: el trabajo por SÍMBOLO (contar alelos por columna) es NumPy vectorizado.
Los bucles de este módulo son POR BIN o POR TIPO-DE-ALELO (alfabeto pequeño y fijo),
no por símbolo en la ruta crítica — análogos a los permitidos en MSA/aligner.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, Union

import numpy as np

from .biocore import SequenceTypeError, SequenceValueError
from .msa import align_multiple

Number = Union[int, float]


class EvolutionResult(NamedTuple):
    """Predicción de la 'próxima' secuencia de consenso.

    predicted          : consenso previsto SIN huecos (secuencia lista para usar).
    predicted_aligned  : consenso previsto sobre el esqueleto del alineamiento (con '-').
    method             : método usado ("naive" | "trend").
    n_bins             : nº de bins temporales.
    n_sites            : nº de columnas del alineamiento.
    changing_sites     : índices de columna cuyo consenso varía en el tiempo (los
                         sitios "bajo presión", descubiertos de los datos).
    """
    predicted: str
    predicted_aligned: str
    method: str
    n_bins: int
    n_sites: int
    changing_sites: list[int]


class GrowthResult(NamedTuple):
    """Fitness estimado (ventaja de crecimiento) por sitio — la señal del eje A.

    site_growth : {sitio → {alelo → tasa de crecimiento en escala log/bin}}. La tasa
                  es la ventaja de fitness: >0 el alelo sube, <0 baja. Solo alelos
                  presentes en ese sitio.
    rising      : {sitio → alelo con mayor tasa} — el candidato a dominar.
    method      : "logistic" (FGA/MLR, tasa fija) | "garw" (tasa variable en el tiempo).
    """
    site_growth: dict[int, dict[str, float]]
    rising: dict[int, str]
    method: str


class EscapeResult(NamedTuple):
    """Potencial de escape antigénico por sitio — la señal del eje C (EVEscape).

    Para cada sitio que cambia, mide cuán físico-químicamente DISTINTO es el residuo
    en ascenso respecto al que reemplaza (Δcarga + Δhidrofobicidad): un cambio grande
    tiene más probabilidad de romper el reconocimiento de un anticuerpo → escape.

    site_escape : {sitio → puntuación de escape en [0, 1]}. Alto = cambio disruptivo.
    ranked      : [(sitio, score)] ordenado de mayor a menor escape — los candidatos.
    change      : {sitio → "viejo→nuevo"} el reemplazo de residuo detectado.
    """
    site_escape: dict[int, float]
    ranked: list[tuple[int, float]]
    change: dict[int, str]


class FusionResult(NamedTuple):
    """Fusión de ejes A+B+C → prioridad por sitio (la arquitectura tipo EVEscape).

    Combina crecimiento (eje A, evofr) + escape (eje C, EVEscape) + viabilidad
    opcional (eje B, ESM-2, enchufable). Los sitios con mayor score son los candidatos
    a la próxima cepa dominante: los que **suben rápido Y son disruptivos**.

    site_score  : {sitio → prioridad combinada en [0, 1]}.
    ranked      : [(sitio, score)] de mayor a menor prioridad.
    terms       : {sitio → {"growth":…, "escape":…, "viability":…}} términos crudos.
    used        : ejes realmente usados (p. ej. sin escape si el alfabeto es ADN).
    weights     : pesos aplicados a cada eje.
    """
    site_score: dict[int, float]
    ranked: list[tuple[int, float]]
    terms: dict[int, dict[str, float]]
    used: list[str]
    weights: dict[str, float]


class CladePrediction(NamedTuple):
    """Predicción a nivel de CLADO (linaje) — estilo evofr/MLR.

    En vez de extrapolar sitio a sitio (ruidoso), agrupa las secuencias en clados por
    sus mutaciones compartidas, modela la frecuencia de cada clado en el tiempo, y
    predice el consenso del clado que dominará — acertando sus mutaciones ENLAZADAS
    de golpe. Esta es la vía que de verdad compite con la baseline ingenua.

    predicted         : consenso previsto del clado ganador, SIN huecos.
    predicted_aligned : el mismo, sobre el esqueleto del alineamiento (con '-').
    dominant_clade    : id del clado que se prevé dominante.
    n_clades          : nº de clados detectados.
    clade_last_freq   : {clado → frecuencia en el último bin}.
    clade_projected   : {clado → frecuencia proyectada al bin siguiente}.
    """
    predicted: str
    predicted_aligned: str
    dominant_clade: int
    n_clades: int
    clade_last_freq: dict[int, float]
    clade_projected: dict[int, float]


class BacktestResult(NamedTuple):
    """Resultado del backtesting: ¿le ganamos a la baseline ingenua?

    method_accuracy : exactitud media por-sitio del método evaluado.
    naive_accuracy  : exactitud media por-sitio de la baseline ingenua (persistencia).
    skill           : mejora normalizada sobre la ingenua en [−∞, 1].
                      skill = (method − naive) / (1 − naive); >0 significa que aporta.
    n_evaluations   : nº de puntos de corte evaluados (bins predichos).
    method          : método evaluado.
    """
    method_accuracy: float
    naive_accuracy: float
    skill: float
    n_evaluations: int
    method: str

    @property
    def beats_naive(self) -> bool:
        """True si el método supera estrictamente a la baseline trivial."""
        return self.method_accuracy > self.naive_accuracy


# ── utilidades internas (vectorizadas) ────────────────────────────────────────

def _encode(aligned: list[str]) -> np.ndarray:
    """Alineamiento (lista de str de igual longitud) → matriz uint8 (N, L).

    Vectorizado: una sola conversión de bytes, sin bucle por símbolo."""
    n = len(aligned)
    buf = "".join(aligned).encode("latin1")
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, len(aligned[0]))


def _bin_ids(times: np.ndarray, n_bins: Optional[int]) -> tuple[np.ndarray, int]:
    """Asigna cada secuencia a un bin temporal.

    n_bins=None → binning ADAPTATIVO: cada instante distinto es su propio bin
    (respeta el ritmo real del organismo, sin imponer una rejilla). En otro caso,
    n_bins intervalos iguales sobre [min, max]."""
    if n_bins is None:
        _, inv = np.unique(times, return_inverse=True)
        return inv.astype(np.intp), int(inv.max()) + 1
    lo, hi = float(times.min()), float(times.max())
    if hi == lo:
        return np.zeros(len(times), dtype=np.intp), 1
    edges = np.linspace(lo, hi, n_bins + 1)
    ids = np.clip(np.digitize(times, edges[1:-1]), 0, n_bins - 1)
    return ids.astype(np.intp), n_bins


def _freqs(arr: np.ndarray, bins: np.ndarray, n_bins: int,
           symbols: np.ndarray) -> np.ndarray:
    """Frecuencias alélicas por bin y columna → array (n_bins, n_symbols, L).

    Bucle por bin y por tipo-de-alelo (alfabeto pequeño), NO por símbolo: el conteo
    ``(sub == s).sum(axis=0)`` es una op NumPy vectorizada sobre toda la columna."""
    L = arr.shape[1]
    S = len(symbols)
    freq = np.zeros((n_bins, S, L), dtype=np.float64)
    for b in range(n_bins):
        sub = arr[bins == b]
        if sub.shape[0] == 0:
            continue
        for si in range(S):
            freq[b, si] = (sub == symbols[si]).sum(axis=0)
        tot = freq[b].sum(axis=0)
        tot[tot == 0] = 1.0
        freq[b] /= tot
    return freq


def _consensus_idx(freq_slice: np.ndarray) -> np.ndarray:
    """Índice del alelo mayoritario por columna (empates → alfabeto ascendente)."""
    return freq_slice.argmax(axis=0)


_METHODS = ("naive", "trend", "logistic", "garw")


def _loglinear_fit(freq: np.ndarray, weighted: bool
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Ajuste ROBUSTO en escala LOGIT de la frecuencia por (alelo, columna).

    Modela logit f_s(t) ≈ a_s + r_s·t (crecimiento logístico ⇒ ventaja de fitness
    r_s, el núcleo de evofr FGA/MLR). Defensas contra la inestabilidad del log crudo
    en frecuencias ~0 (que hacía SOBREDISPARAR la predicción — el árbitro lo cazó):
      - **logit con recorte** [0.02, 0.98] en vez de log con suelo minúsculo;
      - **pesado por la frecuencia** del propio alelo → los bins donde casi no existe
        no distorsionan la pendiente (como el MLR serio);
      - ``weighted=True`` = **GARW**: además favorece los bins recientes → la tasa
        varía en el tiempo (aceleraciones/reversiones).

    Devuelve (logit, slope). ``logit`` (nb, S, L) para ANCLAR la predicción en la
    última frecuencia real; ``slope`` (S, L) es la ventaja de fitness. Vectorizado
    sobre bins; sin bucle por símbolo."""
    nb = freq.shape[0]
    t = np.arange(nb, dtype=np.float64)[:, None, None]
    fc = np.clip(freq, 0.02, 0.98)
    logit = np.log(fc / (1.0 - fc))                   # (nb, S, L)
    w = freq + 1e-2                                    # confianza ∝ frecuencia
    if weighted:                                      # GARW: además, recencia
        w = w * (0.5 ** (nb - 1 - np.arange(nb)))[:, None, None]
    W = w.sum(axis=0)
    tw = (w * t).sum(axis=0) / W
    lm = (w * logit).sum(axis=0) / W
    tc = t - tw
    denom = (w * tc * tc).sum(axis=0) + 1e-12
    slope = (w * tc * (logit - lm)).sum(axis=0) / denom
    return logit, slope


def _predict_idx(freq: np.ndarray, method: str) -> np.ndarray:
    """Predice el índice de alelo por columna para el bin siguiente.

    freq : (n_bins, S, L). Devuelve (L,) con el índice de alelo previsto."""
    if method == "naive":
        return _consensus_idx(freq[-1])
    nb = freq.shape[0]
    if method == "trend":
        if nb < 2:                      # sin dos puntos no hay pendiente → ingenua
            return _consensus_idx(freq[-1])
        t = np.arange(nb, dtype=np.float64)
        tm = t - t.mean()
        denom = (tm * tm).sum()
        # pendiente por (alelo, columna): cov(t, f) / var(t), vectorizado sobre bins
        slope = (tm[:, None, None] * (freq - freq.mean(axis=0))).sum(axis=0) / denom
        projected = freq[-1] + slope    # un paso al futuro
        return projected.argmax(axis=0)
    if method in ("logistic", "garw"):
        if nb < 2:
            return _consensus_idx(freq[-1])
        logit, slope = _loglinear_fit(freq, weighted=(method == "garw"))
        step = np.clip(slope, -1.5, 1.5)  # crecimiento por paso acotado (anti-overshoot)
        proj = logit[-1] + step           # ANCLADO en la última frecuencia observada
        return proj.argmax(axis=0)        # el logit mayor → alelo previsto
    raise SequenceValueError(
        f"método desconocido: {method!r} (usa {'|'.join(_METHODS)}).")


def _validate(sequences: Sequence[str], times: Sequence[Number]) -> None:
    if len(sequences) != len(times):
        raise SequenceValueError(
            f"sequences y times deben tener igual longitud "
            f"({len(sequences)} vs {len(times)}).")
    if len(sequences) < 2:
        raise SequenceValueError("hacen falta al menos 2 secuencias fechadas.")
    for s in sequences:
        if not isinstance(s, str):
            raise SequenceTypeError(
                f"cada secuencia debe ser str, se recibió {type(s).__name__!r}.")
        if len(s) == 0:
            raise SequenceValueError("ninguna secuencia puede estar vacía.")


def _prepare(sequences: Sequence[str], times: Sequence[Number],
             align: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (arr uint8 (N,L), times float, symbols) alineado y ordenado por tiempo."""
    seqs = [s.upper() for s in sequences]
    same_len = len(set(map(len, seqs))) == 1
    if not same_len:
        if not align:
            raise SequenceValueError(
                "las secuencias no están alineadas (longitudes distintas) y "
                "align=False. Alinéalas antes o usa align=True.")
        seqs = list(align_multiple(seqs).aligned)
    arr = _encode(seqs)
    t = np.asarray(times, dtype=np.float64)
    order = np.argsort(t, kind="stable")            # cronológico
    arr, t = arr[order], t[order]
    symbols = np.unique(arr)
    return arr, t, symbols


# ── término de escape (eje C): disimilitud físico-química, EVEscape ───────────
# Hidrofobicidad de Kyte-Doolittle y carga a pH fisiológico. Coste cero: tablas.
_HYDRO = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
          "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
          "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
          "Y": -1.3, "V": 4.2}
_CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.5}
_HYDRO_SPAN = 9.0                                    # 4.5 − (−4.5), para normalizar
_NUC = set("ACGTUN-")                                # si el alfabeto ⊆ esto → ADN/ARN


def _dissimilarity(a: str, b: str) -> float:
    """Disimilitud físico-química entre dos residuos en [0, 1] (Δcarga + Δhidrofob.).

    0 = idénticos o muy parecidos; 1 = opuestos (p. ej. carga + vs −). El proxy de
    escape de EVEscape sin necesidad de estructura 3D."""
    if a == b:
        return 0.0
    dc = abs(_CHARGE.get(a, 0.0) - _CHARGE.get(b, 0.0)) / 2.0        # [0, 1]
    dh = abs(_HYDRO.get(a, 0.0) - _HYDRO.get(b, 0.0)) / _HYDRO_SPAN  # [0, 1]
    return 0.5 * dc + 0.5 * dh


def _require_protein(symbols: np.ndarray) -> None:
    chars = {chr(int(s)) for s in symbols}
    if chars <= _NUC:
        raise SequenceValueError(
            "el término de escape opera sobre PROTEÍNA (carga/hidrofobicidad de "
            "aminoácidos). Las secuencias parecen nucleótido — tradúcelas antes "
            "(SmartTranslator) o pasa la proteína directamente.")


# ── API pública ───────────────────────────────────────────────────────────────

def predict_evolution(sequences: Sequence[str], times: Sequence[Number], *,
                      method: str = "trend", align: bool = True,
                      n_bins: Optional[int] = None) -> EvolutionResult:
    """Predice la 'próxima' secuencia de consenso de un conjunto fechado.

    sequences : iterable de str (ADN o proteína; se pasan a mayúsculas).
    times     : instante de cada secuencia (nº; p. ej. año o fecha ordinal).
    method    : "trend" (extrapolación de tendencia) | "naive" (persistencia).
    align     : si True y las longitudes difieren, se alinean con el MSA.
    n_bins    : nº de bins temporales; None = uno por instante distinto (adaptativo).

    Genoma-agnóstico: el alfabeto se infiere de los datos.
    """
    _validate(sequences, times)
    arr, t, symbols = _prepare(sequences, times, align)
    bins, nb = _bin_ids(t, n_bins)
    freq = _freqs(arr, bins, nb, symbols)

    pred_idx = _predict_idx(freq, method)
    pred_syms = symbols[pred_idx]
    aligned_pred = pred_syms.tobytes().decode("latin1")
    gapless = aligned_pred.replace("-", "")

    # sitios "bajo presión": columnas cuyo consenso varía entre bins (de los datos)
    cons = np.stack([_consensus_idx(freq[b]) for b in range(nb)])  # (nb, L)
    changing = np.where((cons != cons[0]).any(axis=0))[0].tolist()

    return EvolutionResult(
        predicted=gapless,
        predicted_aligned=aligned_pred,
        method=method,
        n_bins=nb,
        n_sites=arr.shape[1],
        changing_sites=changing,
    )


def backtest_evolution(sequences: Sequence[str], times: Sequence[Number], *,
                       method: str = "trend", align: bool = True,
                       n_bins: Optional[int] = None) -> BacktestResult:
    """Backtesting rodante: ¿el método le gana a la baseline ingenua?

    Para cada punto de corte k (con ≥2 bins de entrenamiento) predice el consenso del
    bin k a partir de los bins < k y lo compara con el consenso real de ese bin. Mide
    la exactitud media por-sitio del método y de la ingenua, y el skill normalizado.

    El criterio de "útil": ``result.beats_naive`` / ``result.skill > 0``.
    """
    _validate(sequences, times)
    arr, t, symbols = _prepare(sequences, times, align)
    bins, nb = _bin_ids(t, n_bins)
    if nb < 3:
        raise SequenceValueError(
            f"hacen falta ≥3 bins temporales para backtesting (hay {nb}). "
            "Aporta secuencias de más instantes o reduce n_bins.")
    freq = _freqs(arr, bins, nb, symbols)

    L = arr.shape[1]
    m_correct = n_correct = 0
    evals = 0
    for k in range(2, nb):                       # ≥2 bins de entrenamiento
        actual = _consensus_idx(freq[k])          # consenso real del bin k
        m_pred = _predict_idx(freq[:k], method)   # método, entrenado con bins < k
        n_pred = _predict_idx(freq[:k], "naive")  # ingenua (persistencia)
        m_correct += int((m_pred == actual).sum())
        n_correct += int((n_pred == actual).sum())
        evals += 1

    total = evals * L
    m_acc = m_correct / total
    n_acc = n_correct / total
    skill = (m_acc - n_acc) / (1.0 - n_acc) if n_acc < 1.0 else 0.0
    return BacktestResult(
        method_accuracy=m_acc,
        naive_accuracy=n_acc,
        skill=skill,
        n_evaluations=evals,
        method=method,
    )


def estimate_growth(sequences: Sequence[str], times: Sequence[Number], *,
                    garw: bool = False, align: bool = True,
                    n_bins: Optional[int] = None) -> GrowthResult:
    """Estima el **fitness** (ventaja de crecimiento) de cada alelo en los sitios que
    cambian — la señal del eje A (evofr FGA/MLR; ``garw=True`` = tasa variable, GARW).

    Devuelve un ``GrowthResult`` con la tasa de crecimiento por sitio y alelo, y el
    alelo en ascenso de cada sitio. Solo se reportan los sitios cuyo consenso varía
    (los "bajo presión", descubiertos de los datos) — el resto es ruido plano.

    Genoma-agnóstico: alfabeto inferido de los datos.
    """
    _validate(sequences, times)
    arr, t, symbols = _prepare(sequences, times, align)
    bins, nb = _bin_ids(t, n_bins)
    if nb < 2:
        raise SequenceValueError(
            f"hacen falta ≥2 bins temporales para estimar crecimiento (hay {nb}).")
    freq = _freqs(arr, bins, nb, symbols)
    _, slope = _loglinear_fit(freq, weighted=garw)     # (S, L) ventaja de fitness

    cons = np.stack([_consensus_idx(freq[b]) for b in range(nb)])
    changing = np.where((cons != cons[0]).any(axis=0))[0]
    present = freq.max(axis=0) > 0                       # (S, L) alelo visto en el sitio

    site_growth: dict[int, dict[str, float]] = {}
    rising: dict[int, str] = {}
    for site in changing.tolist():
        alleles = {chr(int(symbols[si])): float(slope[si, site])
                   for si in range(len(symbols)) if present[si, site]}
        site_growth[site] = alleles
        rising[site] = max(alleles, key=alleles.get)
    method = "garw" if garw else "logistic"
    return GrowthResult(site_growth=site_growth, rising=rising, method=method)


def escape_potential(sequences: Sequence[str], times: Sequence[Number], *,
                     garw: bool = False, align: bool = True,
                     n_bins: Optional[int] = None) -> EscapeResult:
    """Potencial de **escape antigénico** por sitio — la señal del eje C (EVEscape).

    Para cada sitio bajo presión, mide cuán físico-químicamente distinto es el residuo
    en ascenso respecto al que reemplaza (Δcarga + Δhidrofobicidad). Un cambio grande
    es más probable que rompa el reconocimiento de un anticuerpo → escape inmune.
    **Coste cero** (tablas de propiedades), sin estructura 3D.

    Requiere PROTEÍNA (traduce antes si tienes nucleótido). Devuelve un ``EscapeResult``
    con la puntuación por sitio, el ranking de candidatos, y el reemplazo detectado.
    """
    _validate(sequences, times)
    arr, t, symbols = _prepare(sequences, times, align)
    _require_protein(symbols)
    bins, nb = _bin_ids(t, n_bins)
    if nb < 2:
        raise SequenceValueError(
            f"hacen falta ≥2 bins temporales para el escape (hay {nb}).")
    freq = _freqs(arr, bins, nb, symbols)
    _, slope = _loglinear_fit(freq, weighted=garw)

    cons0 = _consensus_idx(freq[0])                     # residuo original (primer bin)
    consN = np.stack([_consensus_idx(freq[b]) for b in range(nb)])
    changing = np.where((consN != consN[0]).any(axis=0))[0]
    present = freq.max(axis=0) > 0

    site_escape: dict[int, float] = {}
    change: dict[int, str] = {}
    for site in changing.tolist():
        old = chr(int(symbols[cons0[site]]))            # el que reemplaza
        cand = {si: slope[si, site] for si in range(len(symbols)) if present[si, site]}
        new = chr(int(symbols[max(cand, key=cand.get)]))  # el residuo en ascenso
        site_escape[site] = _dissimilarity(old, new)
        change[site] = f"{old}→{new}"
    ranked = sorted(site_escape.items(), key=lambda kv: kv[1], reverse=True)
    return EscapeResult(site_escape=site_escape, ranked=ranked, change=change)


def predict_fusion(sequences: Sequence[str], times: Sequence[Number], *,
                   garw: bool = False, align: bool = True,
                   n_bins: Optional[int] = None,
                   viability: Optional[dict[int, float]] = None,
                   weights: Optional[dict[str, float]] = None) -> FusionResult:
    """Fusión **A+B+C** → prioridad por sitio (el predictor integrado, tipo EVEscape).

    Combina, por cada sitio bajo presión: crecimiento (eje A), escape (eje C) y —si se
    aporta— viabilidad (eje B, p. ej. de un ESM-2 externo: ``{sitio: score}``). Cada
    eje se estandariza a [0, 1] y se combina con ``weights`` (por defecto reparto igual
    entre los ejes disponibles).

    **Degradación con gracia** (principio genoma-agnóstico): si el alfabeto es
    nucleótido, el eje C (escape) no aplica y la fusión usa solo A (+ B si se aporta).
    """
    _validate(sequences, times)
    arr, t, symbols = _prepare(sequences, times, align)
    is_protein = not ({chr(int(s)) for s in symbols} <= _NUC)
    bins, nb = _bin_ids(t, n_bins)
    if nb < 2:
        raise SequenceValueError(f"hacen falta ≥2 bins temporales (hay {nb}).")
    freq = _freqs(arr, bins, nb, symbols)
    _, slope = _loglinear_fit(freq, weighted=garw)

    cons0 = _consensus_idx(freq[0])
    consN = np.stack([_consensus_idx(freq[b]) for b in range(nb)])
    changing = np.where((consN != consN[0]).any(axis=0))[0].tolist()
    present = freq.max(axis=0) > 0

    # término A (crecimiento del alelo en ascenso) y C (escape) por sitio
    growth: dict[int, float] = {}
    escape: dict[int, float] = {}
    for site in changing:
        cand = {si: slope[si, site] for si in range(len(symbols)) if present[si, site]}
        top = max(cand, key=cand.get)
        growth[site] = max(float(slope[top, site]), 0.0)         # parte positiva
        if is_protein:
            old = chr(int(symbols[cons0[site]]))
            escape[site] = _dissimilarity(old, chr(int(symbols[top])))

    gmax = max(growth.values(), default=0.0) or 1.0              # normaliza crecimiento

    used = ["growth"]
    if is_protein:
        used.append("escape")
    if viability:
        used.append("viability")
    w = weights or {k: 1.0 / len(used) for k in used}
    wsum = sum(w.get(k, 0.0) for k in used) or 1.0

    site_score: dict[int, float] = {}
    terms: dict[int, dict[str, float]] = {}
    for site in changing:
        parts = {"growth": growth[site] / gmax}
        if "escape" in used:
            parts["escape"] = escape[site]
        if "viability" in used:
            parts["viability"] = float(viability.get(site, 0.0))
        terms[site] = parts
        site_score[site] = sum(w.get(k, 0.0) * v for k, v in parts.items()) / wsum
    ranked = sorted(site_score.items(), key=lambda kv: kv[1], reverse=True)
    return FusionResult(site_score=site_score, ranked=ranked, terms=terms,
                        used=used, weights={k: w.get(k, 0.0) for k in used})


# ── clados / haplotipos: eje A a nivel de LINAJE (estilo evofr/MLR) ────────────

def _clade_labels(arr: np.ndarray, symbols: np.ndarray, n_clades: int,
                  min_count: int, key_sites: int,
                  counts: Optional[np.ndarray] = None) -> tuple[np.ndarray, int]:
    """Agrupa secuencias en clados por sus alelos en los sitios más polimórficos.

    Método robusto y sin dependencias: (1) puntúa cada columna por cuántas secuencias
    se salen de la mayoría; (2) toma los ``key_sites`` sitios más variables como
    "definitorios"; (3) siembra clados con los HAPLOTIPOS más frecuentes en esos sitios
    (las variantes que de verdad circulan); (4) asigna cada secuencia al haplotipo-
    semilla más cercano (Hamming). Devuelve (labels, n_clados_reales).

    ``counts`` (S, L): conteos por alelo/sitio ya calculados (optimización — evita
    recontar el array completo; p. ej. el evaluador los pasa vía cumsum)."""
    if counts is None:
        counts = np.stack([(arr == s).sum(axis=0) for s in symbols])   # (S, L)
    minor = arr.shape[0] - counts.max(axis=0)                       # fuera de mayoría
    var_idx = np.where(minor >= min_count)[0]
    if var_idx.size == 0:
        return np.zeros(arr.shape[0], dtype=np.intp), 1
    key = var_idx[np.argsort(-minor[var_idx])[:key_sites]]          # sitios clave
    genos = arr[:, key]                                             # (N, K)
    uniq, cnt = np.unique(genos, axis=0, return_counts=True)
    m = int(min(n_clades, uniq.shape[0]))
    seeds = uniq[np.argsort(-cnt)[:m]]                             # (m, K) frecuentes
    dist = (genos[:, None, :] != seeds[None, :, :]).sum(axis=2)    # (N, m) Hamming
    return dist.argmin(axis=1).astype(np.intp), m


def _clade_consensus_idx(arr: np.ndarray, symbols: np.ndarray,
                         mask: np.ndarray) -> np.ndarray:
    """Índice de alelo mayoritario por columna dentro de un clado (subconjunto)."""
    counts = np.stack([(arr[mask] == s).sum(axis=0) for s in symbols])
    return counts.argmax(axis=0)


def _clade_freqs(labels: np.ndarray, bins: np.ndarray, nb: int, m: int) -> np.ndarray:
    """Frecuencia de cada clado por bin temporal → (nb, m)."""
    cf = np.zeros((nb, m), dtype=np.float64)
    for b in range(nb):
        lb = labels[bins == b]
        if lb.size:
            cf[b] = np.bincount(lb, minlength=m) / lb.size
    return cf


def _project_dominant(clade_freq: np.ndarray, garw: bool) -> np.ndarray:
    """Frecuencia proyectada de cada clado al bin siguiente (mismo ajuste anclado)."""
    nb = clade_freq.shape[0]
    if nb < 2:
        return clade_freq[-1]
    logit, slope = _loglinear_fit(clade_freq[:, :, None], weighted=garw)  # (m, 1)
    return (logit[-1] + np.clip(slope, -1.5, 1.5))[:, 0]


# ── mutabilidad por sitio: "clado variable" (idea propia, estilo beth-1) ───────

def _mutability(freq: np.ndarray) -> np.ndarray:
    """Mutabilidad por sitio (L,) desde el cambio temporal, REGULARIZADA.

    Mide cuánto se mueve la composición de cada sitio año a año (variación total
    media sobre los bins). Se encoge hacia la media global (shrinkage bayesiano) para
    que la escasez de datos no invente volatilidad. Alta = propenso a cambiar; ~0 =
    estable. Es la señal de "transition time" de beth-1, pero data-driven y regularizada."""
    nb = freq.shape[0]
    if nb < 2:
        return np.zeros(freq.shape[2])
    tv = np.abs(np.diff(freq, axis=0)).sum(axis=1).mean(axis=0)   # (L,) cambio medio
    prior = float(tv.mean())                                      # tasa global
    return (tv * (nb - 1) + prior) / ((nb - 1) + 1.0)            # shrink hacia el global


def _mutability_gate(mut: np.ndarray) -> np.ndarray:
    """Puerta [0,1) por sitio: ~0 en sitios estables (→ persistir/naive), →1 en los
    volátiles (→ confiar en el modelo). Escala relativa a la mutabilidad mediana."""
    pos = mut[mut > 1e-9]
    tau = float(np.median(pos)) if pos.size else 1.0
    return mut / (mut + tau + 1e-12)


def site_mutability(sequences: Sequence[str], times: Sequence[Number], *,
                    align: bool = True, n_bins: Optional[int] = None,
                    top: int = 20) -> dict[int, float]:
    """Sitios más **propensos a cambiar** (mutabilidad regularizada), interpretables.

    Devuelve los ``top`` sitios de mayor mutabilidad → {sitio: score}. Son los que
    están bajo presión de cambio (candidatos antigénicos), descubiertos de los datos
    sin hardcodear nada. Genoma-agnóstico."""
    _validate(sequences, times)
    arr, t, symbols = _prepare(sequences, times, align)
    bins, nb = _bin_ids(t, n_bins)
    mut = _mutability(_freqs(arr, bins, nb, symbols))
    order = np.argsort(-mut)[:top]
    return {int(i): float(mut[i]) for i in order if mut[i] > 0}


def predict_clade(sequences: Sequence[str], times: Sequence[Number], *,
                  align: bool = True, n_clades: int = 12, min_count: int = 3,
                  key_sites: int = 40, garw: bool = False) -> CladePrediction:
    """Predice la próxima cepa a nivel de **clado** (linaje) — la vía que compite.

    Agrupa en clados, modela la frecuencia de cada uno en el tiempo, proyecta cuál
    dominará, y devuelve su consenso (que acierta las mutaciones enlazadas de golpe).
    """
    _validate(sequences, times)
    arr, t, symbols = _prepare(sequences, times, align)
    labels, m = _clade_labels(arr, symbols, n_clades, min_count, key_sites)
    bins, nb = _bin_ids(t, None)
    cf = _clade_freqs(labels, bins, nb, m)
    proj = _project_dominant(cf, garw)
    dom = int(proj.argmax())

    pred_idx = _clade_consensus_idx(arr, symbols, labels == dom)
    aligned = symbols[pred_idx].tobytes().decode("latin1")
    return CladePrediction(
        predicted=aligned.replace("-", ""),
        predicted_aligned=aligned,
        dominant_clade=dom,
        n_clades=m,
        clade_last_freq={c: float(cf[-1, c]) for c in range(m)},
        clade_projected={c: float(proj[c]) for c in range(m)},
    )
