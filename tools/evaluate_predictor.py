"""
tools/evaluate_predictor.py
══════════════════════════════════════════════════════════════════════
El MEDIDOR HONESTO — evaluación rigurosa del predictor de evolución.

La lección del fracaso: un número suelto de exactitud de consenso (argmax) baila
±0.09 con la muestra → no se puede confiar. Aquí se arregla el MÉTODO, no el modelo:

  1. Métrica de FRECUENCIAS (como evofr), no consenso argmax. Se predice la frecuencia
     de cada alelo por sitio y se mide el error medio (MAE) frente a la realidad.
     Continua y agregada → deja de oscilar.
  2. SKILL = 1 − error_modelo / error_naive. >0 = el modelo aporta sobre lo trivial.
  3. BOOTSTRAP: se remuestrean las secuencias (con reemplazo, estratificado por año)
     B veces → una DISTRIBUCIÓN de skill con intervalo de confianza al 95%. Si el
     intervalo cruza 0, NO hay win (y lo sabemos). Se alinea UNA vez; el bootstrap
     trabaja sobre el alineamiento (barato).
  4. Backtest leak-free: cada fold usa solo datos < k para ajustar.

No se afirma nada que el intervalo de confianza no respalde.
"""

import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from bioforge.evolution import (  # noqa: E402
    _bin_ids,
    _clade_freqs,
    _clade_labels,
    _freqs,
    _loglinear_fit,
    _mutability,
    _mutability_gate,
    _prepare,
    _project_dominant,
)
from bioforge.fetch import fetch_dated, fetch_dated_precise  # noqa: E402

ORGANISMS = [
    ("H3N2 (gripe A)",
     "Influenza A virus[Organism] AND H3N2 AND hemagglutinin[Title] "
     "AND 1650:1780[SLEN] AND {year}"),
    ("H1N1 (gripe A)",
     "Influenza A virus[Organism] AND H1N1 AND hemagglutinin[Title] "
     "AND 1650:1780[SLEN] AND {year}"),
    ("gripe B",
     "Influenza B virus[Organism] AND hemagglutinin[Title] "
     "AND 1700:1800[SLEN] AND {year}"),
]
YEARS = range(2011, 2024)
PER_YEAR = 150                                 # más datos → IC más estrecho
N_BOOT = 50
MIN_COUNT = 3
METHODS = ("site", "clade", "clade-var")       # clade-var = clado + puerta de mutabilidad
FINE = True                                    # B: bins TRIMESTRALES (fecha de colecta real)


def _softmax(x, axis=0):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _predict_freq(freq_train, arr_train, bins_train, k, symbols, method, counts=None):
    """Frecuencia por-sitio (S, L) prevista para el bin siguiente, según el método.
    ``counts`` (S, L): conteos de entrenamiento ya calculados (optimización)."""
    if method == "naive":
        return freq_train[-1]
    if method == "site":                              # logístico por-sitio
        logit, slope = _loglinear_fit(freq_train, weighted=False)
        return _softmax(logit[-1] + np.clip(slope, -1.5, 1.5), axis=0)
    if method == "clade":                             # proyección a nivel de clado
        labels, m = _clade_labels(arr_train, symbols, 15, MIN_COUNT, 50, counts)
        cf = _clade_freqs(labels, bins_train, k, m)
        w = _softmax(_project_dominant(cf, False))    # pesos de clado (suman 1)
        pred = np.zeros_like(freq_train[-1])
        wsum = 0.0
        for c in range(m):
            sub = arr_train[labels == c]
            if sub.shape[0] == 0:
                continue
            comp = np.stack([(sub == s).sum(0) for s in symbols]).astype(float)
            tot = comp.sum(0); tot[tot == 0] = 1.0
            pred += w[c] * (comp / tot)
            wsum += w[c]
        return pred / wsum if wsum else pred
    if method == "clade-var":
        # clado, pero SOLO nos desviamos de naive donde el sitio es mutable
        base = _predict_freq(freq_train, arr_train, bins_train, k, symbols, "clade", counts)
        naive = freq_train[-1]
        gate = _mutability_gate(_mutability(freq_train))          # (L,) en [0,1)
        return gate[None, :] * base + (1.0 - gate[None, :]) * naive
    raise ValueError(method)


def _one_eval_fast(arr, bins, nb, symbols, idx):
    """Igual que _one_eval pero SIN recontar el array completo en cada fold: cuenta
    por bin una vez y acumula (cumsum). Mismos números, mucho más rápido."""
    a, b = arr[idx], bins[idx]
    S, L = len(symbols), a.shape[1]
    binc = np.zeros((nb, S, L))                        # conteos por bin (una vez)
    for bn in range(nb):
        sub = a[b == bn]
        if sub.shape[0]:
            for si in range(S):
                binc[bn, si] = (sub == symbols[si]).sum(0)
    tot = binc.sum(axis=1, keepdims=True)
    freq = binc / np.maximum(tot, 1.0)                 # frecuencias por bin
    cum = binc.cumsum(axis=0)                          # cum[k-1] = conteos de bins < k

    errs = {m: [] for m in ("naive", *METHODS)}
    for k in range(2, nb):
        tr = b < k
        at, bt = a[tr], b[tr]
        if at.shape[0] == 0:
            continue
        ctr = cum[k - 1]                               # conteos de entrenamiento (S, L)
        variable = (at.shape[0] - ctr.max(0)) >= MIN_COUNT
        if not variable.any():
            continue
        actual = freq[k][:, variable]
        ftr = freq[:k]
        clade_pred = None                             # se calcula una vez, se reusa
        for m in ("naive", *METHODS):
            if m == "clade-var" and clade_pred is not None:
                gate = _mutability_gate(_mutability(ftr))
                pred = gate[None, :] * clade_pred + (1.0 - gate[None, :]) * ftr[-1]
            else:
                pred = _predict_freq(ftr, at, bt, k, symbols, m, ctr)
                if m == "clade":
                    clade_pred = pred
            errs[m].append(float(np.abs(pred[:, variable] - actual).mean()))
    return {m: (np.mean(v) if v else np.nan) for m, v in errs.items()}


def _one_eval(arr, bins, nb, symbols, idx):
    """Un remuestreo → error MAE medio por método (sobre sitios variables)."""
    a, b = arr[idx], bins[idx]
    freq = _freqs(a, b, nb, symbols)
    errs = {m: [] for m in ("naive", *METHODS)}
    for k in range(2, nb):
        tr = b < k
        at, bt = a[tr], b[tr]
        if at.shape[0] == 0:
            continue
        counts = np.stack([(at == s).sum(0) for s in symbols])
        variable = (at.shape[0] - counts.max(0)) >= MIN_COUNT
        if not variable.any():
            continue
        actual = freq[k][:, variable]
        for m in ("naive", *METHODS):
            pred = _predict_freq(freq[:k], at, bt, k, symbols, m)[:, variable]
            errs[m].append(float(np.abs(pred - actual).mean()))
    return {m: (np.mean(v) if v else np.nan) for m, v in errs.items()}


def _resample(bins, nb, rng):
    idx = []
    for bn in range(nb):
        pool = np.where(bins == bn)[0]
        if len(pool):
            idx.append(rng.choice(pool, len(pool), replace=True))
    return np.concatenate(idx)


def evaluate(label, term):
    data = (fetch_dated_precise(term, YEARS, per_year=PER_YEAR) if FINE
            else fetch_dated(term, YEARS, per_year=PER_YEAR))
    if len(data) < 60:
        print(f"  {label}: pocas secuencias ({len(data)}) — saltando.\n")
        return
    seqs = [s for s, _ in data]
    times = [y for _, y in data]
    arr, t, symbols = _prepare(seqs, times, align=True)
    bins, nb = _bin_ids(t, None)

    rng = np.random.default_rng(0)
    idx_all = np.arange(len(seqs))
    skills = {m: [] for m in METHODS}
    for it in range(N_BOOT):
        idx = idx_all if it == 0 else _resample(bins, nb, rng)
        e = _one_eval_fast(arr, bins, nb, symbols, idx)
        for m in METHODS:
            if not np.isnan(e[m]) and not np.isnan(e["naive"]) and e["naive"] > 0:
                skills[m].append(1.0 - e[m] / e["naive"])

    print(f"  {label}  (n={len(seqs)}, años={nb}, bootstrap={N_BOOT})")
    for m in METHODS:
        s = np.array(skills[m])
        lo, hi = np.percentile(s, [2.5, 97.5])
        robust = "✓ WIN robusto" if lo > 0 else ("~ empate" if hi > 0 else "✗ pierde")
        print(f"    {m:6s} skill = {s.mean():+.4f}  IC95% [{lo:+.4f}, {hi:+.4f}]  {robust}")
    print()


def main():
    print("EVALUADOR RIGUROSO — skill de frecuencias vs naive, con IC95% (bootstrap)")
    print(f"Resolución: {'TRIMESTRAL (B, fecha real)' if FINE else 'anual'}. "
          "Un WIN solo cuenta si el IC95% está ENTERO por encima de 0.\n")
    for label, term in ORGANISMS:              # los 3 virus (prueba de generalidad)
        evaluate(label, term)


if __name__ == "__main__":
    main()
