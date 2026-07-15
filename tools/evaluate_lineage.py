"""
tools/evaluate_lineage.py
══════════════════════════════════════════════════════════════════════
Evaluación a NIVEL DE LINAJE — la pregunta que hace de verdad el campo (evofr).

Estábamos midiendo a nivel de SITIO (las ~1800 posiciones), donde la señal de los
~15 sitios que importan se ahoga en el ruido de los ~1785 que no, y donde persistir
("mañana = hoy") es casi imbatible. Eso era dar puñetazos a toda la pared.

Aquí medimos lo que importa: **¿predijo bien qué CLADO crecería el próximo trimestre?**
  - Modelo: proyecta la frecuencia de cada clado (crecimiento logístico).
  - Naive: la frecuencia de clados PERSISTE (la del último trimestre).
  - Real: se asignan las secuencias del próximo trimestre a los clados de entrenamiento.
  - Error: distancia L1 entre distribución predicha y real. Skill = 1 − modelo/naive.

Pocas opciones, mucha señal. Backtest leak-free + bootstrap con IC95%.
"""

import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from bioforge.evolution import (  # noqa: E402
    _assign_clades,
    _assign_lineages,
    _bin_ids,
    _clade_counts,
    _clade_freqs,
    _clade_model,
    _prepare,
    _project_freqs,
    designate_lineages,
)
from bioforge.fetch import fetch_dated_precise  # noqa: E402

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
PER_YEAR = 150
N_BOOT = 50
N_CLADES, MIN_COUNT, KEY = 15, 3, 50           # clustering tosco (el de antes)
MAX_LIN, MIN_SIZE, KEY_LIN = 20, 10, 100       # linajes estables (Pango/autolin)
KAPPA = 30.0                                   # κ del shrinkage — fijado A PRIORI
#                                              (≈1 bin de datos), NO ajustado al test
METHODS = ("crudo", "estable", "+shrink", "+arbol")   # acumulativas: 3 → 4 → 5


def _one_lineage(arr, bins, nb, symbols, idx, method):
    """Error L1 medio (modelo, naive) prediciendo la distribución de clados del bin k.

    ``crudo``   = re-agrupar desde cero en cada fold (las etiquetas bailan).
    ``estable`` = designar linajes UNA vez y solo EXTENDER (estilo Pango) → las
                  trayectorias de linaje son comparables en el tiempo.  [palanca 3]
    ``+shrink`` = además, tasas de crecimiento encogidas por evidencia.   [palanca 4]
    ``+arbol``  = además, fitness propagado por la jerarquía de linajes.  [palanca 5]
    """
    a, b = arr[idx], bins[idx]
    em = en = 0.0
    folds = 0
    system = None                                      # se ARRASTRA entre folds
    for k in range(2, nb):
        at, bt = a[b < k], b[b < k]
        aq = a[b == k]                                 # secuencias del próximo trimestre
        if at.shape[0] == 0 or aq.shape[0] == 0:
            continue
        kw = {}
        if method == "crudo":
            labels, m, key, seeds = _clade_model(at, symbols, N_CLADES, MIN_COUNT, KEY)
            actual_lab = _assign_clades(aq, key, seeds)
        else:
            system = designate_lineages(at, symbols, prior=system, key_sites=KEY_LIN,
                                        max_lineages=MAX_LIN, min_size=MIN_SIZE)
            m = system.n
            labels = _assign_lineages(at, system)
            actual_lab = _assign_lineages(aq, system)
            if method != "estable":
                kw["counts"] = _clade_counts(labels, bt, k, m)
                kw["shrink"] = KAPPA
            if method == "+arbol":
                kw["parents"] = system.parents
                kw["sizes"] = np.array([s.size for s in system.sites])
        if m < 2:
            continue
        cf = _clade_freqs(labels, bt, k, m)            # (k, m) trayectoria de clados
        pm = _project_freqs(cf, False, **kw)           # predicho por el modelo (MLR)
        pn = cf[-1]                                    # naive: persistir el último
        actual = np.bincount(actual_lab, minlength=m) / aq.shape[0]
        em += float(np.abs(pm - actual).sum())         # L1 sobre la distribución
        en += float(np.abs(pn - actual).sum())
        folds += 1
    return (em / folds, en / folds) if folds else (np.nan, np.nan)


def _resample(bins, nb, rng):
    idx = []
    for bn in range(nb):
        pool = np.where(bins == bn)[0]
        if len(pool):
            idx.append(rng.choice(pool, len(pool), replace=True))
    return np.concatenate(idx)


def evaluate(label, term):
    data = fetch_dated_precise(term, YEARS, per_year=PER_YEAR)
    if len(data) < 60:
        print(f"  {label}: pocas secuencias ({len(data)}).")
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
        for m in METHODS:
            em, en = _one_lineage(arr, bins, nb, symbols, idx, m)
            if not np.isnan(em) and en > 0:
                skills[m].append(1.0 - em / en)
    # diagnóstico: ¿cuántos linajes designa el sistema estable, y se topa con el cap?
    sysd = None
    for k in range(2, nb):
        at = arr[bins < k]
        if at.shape[0]:
            sysd = designate_lineages(at, symbols, prior=sysd, key_sites=KEY_LIN,
                                      max_lineages=MAX_LIN, min_size=MIN_SIZE)
    prof = max(s.size for s in sysd.sites) if sysd else 0
    print(f"  {label}  (n={len(seqs)}, bins={nb}) · linajes designados: {sysd.n}"
          f"/{MAX_LIN} · profundidad máx: {prof} mutaciones")
    for m in METHODS:
        s = np.array(skills[m])
        lo, hi = np.percentile(s, [2.5, 97.5])
        robust = "✓ WIN robusto" if lo > 0 else ("~ empate" if hi > 0 else "✗ pierde")
        print(f"    {m:8s} skill = {s.mean():+.4f}  IC95% [{lo:+.4f}, {hi:+.4f}]  {robust}")
    print()


def main():
    print("NIVEL LINAJE — ¿predice qué CLADO crece mejor que persistir? (como evofr)")
    print("crudo = re-agrupar cada fold (lo de antes) · estable = Pango/autolin (nuevo)")
    print("Skill sobre la distribución de clados. WIN si el IC95% está entero > 0.\n")
    only = sys.argv[1].lower() if len(sys.argv) > 1 else None
    for label, term in ORGANISMS:
        if only is None or only in label.lower():
            evaluate(label, term)


if __name__ == "__main__":
    main()
