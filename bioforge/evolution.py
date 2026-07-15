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

def _clade_model(arr: np.ndarray, symbols: np.ndarray, n_clades: int,
                 min_count: int, key_sites: int,
                 counts: Optional[np.ndarray] = None):
    """Ajusta el modelo de clados y devuelve (labels, m, key_cols, seeds).

    (1) puntúa cada columna por cuántas secuencias se salen de la mayoría; (2) toma los
    ``key_sites`` sitios más variables como "definitorios"; (3) siembra clados con los
    HAPLOTIPOS más frecuentes en esos sitios; (4) asigna al más cercano (Hamming).
    Exponer ``key_cols``/``seeds`` permite asignar secuencias NUEVAS al mismo modelo
    (necesario para medir la frecuencia real del próximo periodo — nivel linaje)."""
    if counts is None:
        counts = np.stack([(arr == s).sum(axis=0) for s in symbols])   # (S, L)
    minor = arr.shape[0] - counts.max(axis=0)                       # fuera de mayoría
    var_idx = np.where(minor >= min_count)[0]
    if var_idx.size == 0:
        z = np.zeros(arr.shape[0], dtype=np.intp)
        return z, 1, np.empty(0, dtype=np.intp), None
    key = var_idx[np.argsort(-minor[var_idx])[:key_sites]]          # sitios clave
    genos = arr[:, key]                                             # (N, K)
    uniq, cnt = np.unique(genos, axis=0, return_counts=True)
    m = int(min(n_clades, uniq.shape[0]))
    seeds = uniq[np.argsort(-cnt)[:m]]                             # (m, K) frecuentes
    dist = (genos[:, None, :] != seeds[None, :, :]).sum(axis=2)    # (N, m) Hamming
    return dist.argmin(axis=1).astype(np.intp), m, key, seeds


def _assign_clades(arr: np.ndarray, key: np.ndarray, seeds) -> np.ndarray:
    """Asigna secuencias NUEVAS al clado-semilla más cercano (mismo modelo)."""
    if seeds is None or seeds.shape[0] == 0:
        return np.zeros(arr.shape[0], dtype=np.intp)
    genos = arr[:, key]
    dist = (genos[:, None, :] != seeds[None, :, :]).sum(axis=2)
    return dist.argmin(axis=1).astype(np.intp)


def _clade_labels(arr: np.ndarray, symbols: np.ndarray, n_clades: int,
                  min_count: int, key_sites: int,
                  counts: Optional[np.ndarray] = None) -> tuple[np.ndarray, int]:
    """Agrupa secuencias en clados. Devuelve (labels, n_clados). Ver ``_clade_model``."""
    labels, m, _, _ = _clade_model(arr, symbols, n_clades, min_count, key_sites, counts)
    return labels, m


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


def _clade_counts(labels: np.ndarray, bins: np.ndarray, nb: int, m: int) -> np.ndarray:
    """Nº de secuencias de cada clado por bin → (nb, m). Es la EVIDENCIA: un linaje
    visto 3 veces no merece el mismo crédito que uno visto 300."""
    cc = np.zeros((nb, m), dtype=np.float64)
    for b in range(nb):
        lb = labels[bins == b]
        if lb.size:
            cc[b] = np.bincount(lb, minlength=m)
    return cc


def _shrink_slopes(slope: np.ndarray, weight: np.ndarray, kappa: float,
                   parents: Optional[np.ndarray] = None,
                   sizes: Optional[np.ndarray] = None) -> np.ndarray:
    """Encoge las tasas de crecimiento hacia un prior (Bayes empírico, forma cerrada).

    ``slope`` crudo de un linaje con pocas secuencias es casi ruido; encogerlo hacia un
    prior en proporción a su evidencia es lo que hace evofr con priors bayesianos —
    aquí sin MCMC ni JAX: ``w = n/(n+κ)``, el estimador clásico. κ = nº de secuencias
    a partir del cual te fías del dato más que del prior.

      - ``parents=None`` → prior = media global ponderada (**palanca 4, estilo evofr**):
        "sin evidencia, no hay ventaja de crecimiento".
      - ``parents`` dado → prior = la tasa YA ENCOGIDA DEL PADRE (**palanca 5, estilo
        Łuksza-Lässig**): el fitness se propaga por la jerarquía, así que un sub-linaje
        con pocos datos hereda la tendencia de su padre en vez de inventarse la suya.
        Se recorre de padres a hijos (orden topológico = conjunto definitorio creciente).
    """
    w = weight / np.maximum(weight + kappa, 1e-9)
    if parents is None:
        mu = float((slope * weight).sum() / max(weight.sum(), 1e-9))
        return mu + (slope - mu) * w
    out = slope.copy()
    order = np.argsort(sizes, kind="stable") if sizes is not None \
        else np.arange(len(slope))
    for i in order:                                   # padres antes que hijos
        p = int(parents[i])
        prior = 0.0 if p < 0 else float(out[p])       # la raíz se encoge hacia "0"
        out[i] = prior + (slope[i] - prior) * w[i]
    return out


def _project_freqs(clade_freq: np.ndarray, garw: bool, *,
                   counts: Optional[np.ndarray] = None,
                   shrink: float = 0.0,
                   parents: Optional[np.ndarray] = None,
                   sizes: Optional[np.ndarray] = None) -> np.ndarray:
    """Frecuencias de clado proyectadas al bin siguiente — forma MULTIPLICATIVA
    ``f' ∝ f·exp(r)``, que es el modelo MLR de evofr.

    Por qué no un softmax sobre logits proyectados: el ajuste recorta a [0.02, 0.98]
    para estabilizarse, así que un softmax REGALA un suelo de probabilidad a cada
    linaje EXTINTO. Con nomenclatura estable (que acumula linajes históricos, como
    Pango) eso son decenas de muertos robando masa a los vivos. La forma
    multiplicativa preserva los ceros: extinto por 0 sigue siendo 0.

    Además tiene la inducción correcta: con r=0 el modelo **se reduce exactamente a
    la ingenua**, así que encoger las tasas (``shrink``) interpola de forma continua
    entre "predigo crecimiento" y "persisto" — nunca se puede perder por inventarse
    una ventaja sin evidencia.
    """
    nb = clade_freq.shape[0]
    if nb < 2:
        return clade_freq[-1]
    _, slope = _loglinear_fit(clade_freq[:, :, None], weighted=garw)
    s = slope[:, 0]
    if shrink > 0.0 and counts is not None:
        s = _shrink_slopes(s, counts.sum(axis=0), shrink, parents, sizes)
    pred = clade_freq[-1] * np.exp(np.clip(s, -1.5, 1.5))
    tot = pred.sum()
    return pred / tot if tot > 0 else clade_freq[-1]


def _project_dominant(clade_freq: np.ndarray, garw: bool, *,
                      counts: Optional[np.ndarray] = None,
                      shrink: float = 0.0,
                      parents: Optional[np.ndarray] = None,
                      sizes: Optional[np.ndarray] = None) -> np.ndarray:
    """Frecuencia proyectada de cada clado al bin siguiente (mismo ajuste anclado).

    Con ``shrink>0`` (κ) las tasas se regularizan según la evidencia de cada linaje;
    con ``parents`` además se propagan por la jerarquía. Ver ``_shrink_slopes``."""
    nb = clade_freq.shape[0]
    if nb < 2:
        return clade_freq[-1]
    logit, slope = _loglinear_fit(clade_freq[:, :, None], weighted=garw)  # (m, 1)
    s = slope[:, 0]
    if shrink > 0.0 and counts is not None:
        s = _shrink_slopes(s, counts.sum(axis=0), shrink, parents, sizes)
    return logit[-1, :, 0] + np.clip(s, -1.5, 1.5)


# ── linajes ESTABLES: mutaciones definitorias + GRI (estilo Pango/autolin) ────
#
# El error que arregla esto: re-agrupar los clados en cada fold hace que las etiquetas
# bailen (el linaje "3" de hoy no es el de ayer) → el modelo de crecimiento recibe
# trayectorias sin sentido. NADIE en el campo hace eso: Pango/Nextstrain DESIGNAN los
# linajes una vez (y solo AÑADEN nuevos), y luego únicamente ASIGNAN. evofr recibe los
# clados ya hechos: su acierto hereda la calidad de esas definiciones.
#
# Prestado y adaptado:
#   - Pango      : linaje = conjunto de MUTACIONES DEFINITORIAS respecto a un ancestro,
#                  conservadas en >70% del linaje. Jerárquico (B.1.1.7 hereda de B.1.1).
#   - autolin    : designación AUTOMÁTICA y patógeno-agnóstica maximizando el
#                  Genotype Representation Index  GRI = N·D / (S + N + D)
#                  N = tamaño, D = distinción del padre, S = diversidad interna.
#   - Nextclade  : asignar comparando conjuntos de mutaciones, descendiendo por la
#                  jerarquía (incluidos nodos internos/ancestrales).
#
# NUESTRA VUELTA: autolin necesita un árbol filogenético enraizado con longitudes de
# rama (IQ-TREE: caro, dependencia pesada, mata Edge Computing). Aquí los tres términos
# del GRI se calculan DIRECTO del MSA con una matriz de co-ocurrencia (un solo matmul):
#   N = portadores · D = definitorias nuevas vs el padre · S = Σ (N − alelo mayoritario).
# Honesto: sin árbol no distinguimos monofilia real de homoplasia (mutación recurrente);
# la co-ocurrencia la aproxima. Es una aproximación, y se dice.

_DEFINING = 0.7          # regla de Pango: definitoria si está en >70% del linaje
_MATCH = 0.5             # se asigna al hijo si lleva la mayoría de sus definitorias


class LineageSystem(NamedTuple):
    """Definición ESTABLE de linajes (estilo Pango), designada una vez y EXTENDIDA.

    La IDENTIDAD de un linaje es su conjunto COMPLETO de mutaciones definitorias
    respecto al ancestro, y se congela al designarlo: nunca cambia → las etiquetas son
    comparables en el tiempo, que es justo lo que necesita el modelo de crecimiento
    (y lo que nuestro clustering tosco rompía). ``parents`` se reconstruye por
    contención de conjuntos (si las definitorias de A ⊂ las de A.1, entonces A.1
    desciende de A), lo que reordena la jerarquía sin tocar identidades.
    """
    root: np.ndarray         # (L,) alelo ancestral por sitio (consenso de lo más antiguo)
    sites: list              # sites[i] = (k_i,) sitios definitorios COMPLETOS del linaje i
    alleles: list            # alleles[i] = (k_i,) alelos definitorios completos
    parents: np.ndarray      # (m,) índice del padre; -1 = raíz
    n: int                   # nº de linajes designados


def _keys(sites: np.ndarray, alleles: np.ndarray) -> np.ndarray:
    """Mutaciones (sitio, alelo) → claves int64, para operar con conjuntos."""
    return sites.astype(np.int64) * 256 + alleles.astype(np.int64)


def _own(system: LineageSystem, i: int) -> tuple[np.ndarray, np.ndarray]:
    """Mutaciones PROPIAS del linaje i = las suyas menos las heredadas del padre."""
    p = int(system.parents[i])
    if p < 0 or system.sites[p].size == 0:
        return system.sites[i], system.alleles[i]
    m = ~np.isin(_keys(system.sites[i], system.alleles[i]),
                 _keys(system.sites[p], system.alleles[p]))
    return system.sites[i][m], system.alleles[i][m]


def _renest(sites: list, alleles: list) -> np.ndarray:
    """Jerarquía por CONTENCIÓN: el padre de i es el linaje de conjunto definitorio
    más grande estrictamente contenido en el de i.

    Sin árbol filogenético, la contención de mutaciones definitorias es la señal de
    descendencia (A.1 lleva todo lo de A y algo más ⇒ desciende de A). Como el padre
    siempre tiene un conjunto ESTRICTAMENTE menor, la jerarquía es acíclica (requisito
    de augur). Bucle por linaje (decenas), no por símbolo."""
    ks = [_keys(s, a) for s, a in zip(sites, alleles)]
    parents = np.zeros(len(ks), dtype=np.intp)
    parents[0] = -1                                   # 0 = raíz (conjunto vacío)
    for i in range(1, len(ks)):
        best, size = 0, -1
        for j in range(len(ks)):
            if j == i or ks[j].size >= ks[i].size or ks[j].size <= size:
                continue
            if bool(np.isin(ks[j], ks[i]).all()):     # j ⊂ i  ⇒ j es ancestro de i
                best, size = j, ks[j].size
        parents[i] = best
    return parents


def _root_consensus(arr: np.ndarray, symbols: np.ndarray) -> np.ndarray:
    """Alelo mayoritario por sitio — aproxima el ancestro (se calcula con lo más
    antiguo disponible y NO vuelve a cambiar: es el ancla de la estabilidad)."""
    counts = np.stack([(arr == s).sum(axis=0) for s in symbols])
    return symbols[counts.argmax(axis=0)]


def _candidates(arr: np.ndarray, symbols: np.ndarray, root: np.ndarray,
                min_size: int, key_sites: int,
                mut_weights: Optional[np.ndarray] = None):
    """Espacio de búsqueda: mutaciones (sitio, alelo) que podrían definir un linaje.

    Devuelve (sitios, alelos, X, Y, starts, w), con ``w`` = peso de cada mutación:
      - X (N, C) = portadores de cada mutación **DERIVADA** (alelo ≠ ancestro). Solo
        estas definen linajes: llevar el alelo ancestral no es una mutación, es una
        ausencia — confundirlo fusiona la raíz con sus hijos.
      - Y (N, P) = perfil con TODOS los alelos (ancestral incluido), necesario para
        saber cuál es el mayoritario dentro de un linaje (término S del GRI).
      - starts = inicio de cada sitio en las columnas de Y (para el max por sitio).
    Se restringe a los ``key_sites`` sitios más variables: una definitoria siempre está
    en un sitio variable, y acotar las columnas mantiene el matmul barato."""
    counts = np.stack([(arr == s).sum(axis=0) for s in symbols]).astype(np.int64)
    n = arr.shape[0]
    minor = n - counts.max(axis=0)
    var = np.where(minor >= min_size)[0]
    e = np.empty(0, dtype=np.intp)
    if var.size == 0:                                    # nada variable → sin candidatos
        return e, np.empty(0, dtype=arr.dtype), np.zeros((n, 0), dtype=bool), \
            np.zeros((n, 0), dtype=bool), e, np.empty(0, dtype=np.float64)
    key = np.sort(var[np.argsort(-minor[var])[:key_sites]])
    si, ki = np.nonzero(counts[:, key] >= min_size)      # alelos con presencia real
    order = np.argsort(ki, kind="stable")                # ordenar POR SITIO
    si, ki = si[order], ki[order]
    li, al = key[ki], symbols[si]
    Y = arr[:, li] == al[None, :]                        # perfil completo
    _, starts = np.unique(ki, return_index=True)
    derived = al != root[li]                             # ← mutación DE VERDAD
    if mut_weights is None:
        w = np.ones(int(derived.sum()), dtype=np.float64)
    elif mut_weights.ndim == 1:                          # (L,) peso por SITIO
        w = mut_weights[li[derived]].astype(np.float64)
    else:                                                # (S, L) peso por MUTACIÓN
        w = mut_weights[si[derived], li[derived]].astype(np.float64)
    return li[derived], al[derived], Y[:, derived], Y, starts.astype(np.intp), w


def _best_split(Xg: np.ndarray, Yg: np.ndarray, starts: np.ndarray,
                min_size: int, min_dist: int, w: np.ndarray):
    """Mejor división del grupo según GRI → (gri, cand, defining_mask) o ``None``.

    Dos matmuls dan los tres términos del GRI a la vez para TODOS los candidatos
    (sin bucle por candidato): co-ocurrencia entre mutaciones (N, D) y conteo de
    alelos dentro de cada hijo (S). ``w`` pondera cada mutación en el término D."""
    ng = Xg.shape[0]
    if ng < 2 * min_size or Xg.shape[1] == 0:
        return None
    Xf = Xg.astype(np.float32)
    co = Xf.T @ Xf                                    # (C, C) portadores de a Y de b
    na = np.diag(co).copy()                           # (C,) tamaño de cada candidato
    ok = (na >= min_size) & (na <= ng - 1)            # ha de dividir, no copiar al padre
    if not ok.any():
        return None
    safe = np.maximum(na, 1.0)
    child = co / safe[:, None]                        # (C, C) frecuencia dentro del hijo
    parent = na / ng                                  # (C,) frecuencia dentro del padre
    define = (child >= _DEFINING) & (parent < _DEFINING)[None, :]   # definitorias NUEVAS
    d = define @ w                                    # D: distinción PONDERADA del padre
    cross = Xf.T @ Yg.astype(np.float32)              # (C, P) alelos dentro del hijo
    smax = np.maximum.reduceat(cross, starts, axis=1)  # mayoritario por sitio
    s = (na[:, None] - smax).sum(axis=1)              # S: diversidad interna del hijo
    gri = na * d / np.maximum(s + na + d, 1e-9)       # ← Genotype Representation Index
    gri[~ok | (define.sum(axis=1) < min_dist)] = -1.0
    best = int(gri.argmax())
    if gri[best] <= 0.0:
        return None
    return float(gri[best]), best, define[best]


def _assign_lineages(arr: np.ndarray, system: LineageSystem) -> np.ndarray:
    """Asigna secuencias al linaje más específico cuyas definitorias llevan.

    Desciende por la jerarquía (estilo Nextclade/augur): en cada nodo compara con los
    conjuntos de mutaciones de sus hijos y baja por el que mejor encaja. Bucle POR
    LINAJE (~decenas), no por símbolo."""
    labels = np.zeros(arr.shape[0], dtype=np.intp)
    if system.n < 2:
        return labels
    kids: dict[int, list[int]] = {}
    for i in range(1, system.n):
        kids.setdefault(int(system.parents[i]), []).append(i)
    owns = [_own(system, i) for i in range(system.n)]     # propias vs el padre
    queue = [0]
    while queue:
        p = queue.pop(0)
        ch = kids.get(p)
        if not ch:
            continue
        idx = np.where(labels == p)[0]
        if idx.size:
            sub = arr[idx]
            score = np.stack([(sub[:, owns[c][0]] == owns[c][1]).mean(axis=1)
                              if owns[c][0].size else np.zeros(idx.size)
                              for c in ch])                       # (n_hijos, n_seqs)
            best, top = score.argmax(axis=0), score.max(axis=0)
            take = top >= _MATCH
            labels[idx[take]] = np.asarray(ch)[best[take]]
        queue.extend(ch)
    return labels


def designate_lineages(arr: np.ndarray, symbols: np.ndarray, *,
                       max_lineages: int = 20, min_size: int = 10,
                       min_dist: int = 1, key_sites: int = 100,
                       mut_weights: Optional[np.ndarray] = None,
                       prior: Optional[LineageSystem] = None) -> LineageSystem:
    """Designa linajes maximizando el GRI de forma voraz (autolin sin árbol).

    Con ``prior`` EXTIENDE un sistema existente: conserva raíz y definiciones (los
    linajes viejos jamás cambian) y solo añade los nuevos que los datos justifiquen —
    la disciplina "dinámica pero estable" de Pango. Sin ``prior``, designa desde cero
    tomando como ancestro el consenso de ``arr`` (llamar con lo más antiguo).

    A diferencia del clustering tosco, el número de linajes NO se fija a dedo: sale de
    los umbrales ``min_size`` (tamaño mínimo) y ``min_dist`` (mutaciones mínimas que lo
    distinguen del padre), como en autolin.

    ``mut_weights`` (opcional) pondera cuánto "distingue" cada mutación: (L,) por sitio
    o (n_símbolos, L) por mutación. Es la opción de pesos por mutación de autolin, y la
    puerta por la que entra el conocimiento externo SIN romper el agnosticismo (por
    defecto todo pesa 1): ``escape_weights`` (eje C, físico-química), sitios epítopo
    conocidos (prior antigénico), o cualquier puntuación de una IA (eje B). Así los
    linajes se definen por las mutaciones que IMPORTAN, no por la deriva neutral.
    """
    if prior is None:
        root = _root_consensus(arr, symbols)
        sites: list = [np.empty(0, dtype=np.intp)]
        alleles: list = [np.empty(0, dtype=arr.dtype)]
        parents: list = [-1]
    else:
        root = prior.root
        sites, alleles, parents = list(prior.sites), list(prior.alleles), \
            list(prior.parents)

    li, al, X, Y, starts, w = _candidates(arr, symbols, root, min_size, key_sites,
                                          mut_weights)
    if li.size == 0:
        return LineageSystem(root, sites, alleles, np.asarray(parents, dtype=np.intp),
                             len(sites))

    labels = _assign_lineages(arr, LineageSystem(
        root, sites, alleles, np.asarray(parents, dtype=np.intp), len(sites)))
    cache: dict = {}                     # mejor división por grupo: dividir uno NO
    while len(sites) < max_lineages:     # cambia a los demás → sus cálculos siguen
        best = None                      # valiendo (voraz O(m) en vez de O(m²))
        for g in range(len(sites)):
            if g not in cache:
                mask = labels == g
                cache[g] = (_best_split(X[mask], Y[mask], starts, min_size, min_dist, w)
                            if int(mask.sum()) >= 2 * min_size else None)
            r = cache[g]
            if r is not None and (best is None or r[0] > best[1][0]):
                best = (g, r)
        if best is None:
            break
        g, (_, cand, define) = best
        new = len(sites)
        sites.append(np.concatenate([sites[g], li[define]]))     # identidad = conjunto
        alleles.append(np.concatenate([alleles[g], al[define]]))  # COMPLETO vs ancestro
        parents.append(g)
        member = (labels == g) & X[:, cand]           # los portadores pasan al hijo
        labels[member] = new
        cache.pop(g)                                  # solo g cambió de miembros
    return LineageSystem(root, sites, alleles, _renest(sites, alleles), len(sites))


def escape_weights(symbols: np.ndarray, root: np.ndarray, *,
                   base: float = 1.0) -> np.ndarray:
    """Pesos de mutación (n_símbolos, L) = 1 + disimilitud físico-química vs el ancestro.

    Para ``designate_lineages(mut_weights=...)``: una mutación que cambia carga o
    hidrofobicidad (candidata a escape inmune, eje C de EVEscape) pesa hasta el doble
    al definir un linaje que una sustitución conservadora. Requiere PROTEÍNA.
    """
    _require_protein(symbols)
    tab = np.array([[_dissimilarity(chr(int(a)), chr(int(b))) for b in symbols]
                    for a in symbols])                       # (S, S) entre símbolos
    idx = {int(s): i for i, s in enumerate(symbols)}
    ridx = np.array([idx.get(int(r), 0) for r in root])      # (L,) alelo raíz por sitio
    return base + tab[:, ridx]                               # (S, L)


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
