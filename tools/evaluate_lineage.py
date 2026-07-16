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
MAX_LIN, MIN_SIZE, KEY_LIN = 500, 10, 100   # tope = solo válvula; el control es min_size
KAPPA = 30.0                                   # κ del shrinkage — fijado A PRIORI
#                                              (≈1 bin de datos), NO ajustado al test
METHODS = ("crudo", "estable", "+shrink", "+arbol")   # acumulativas: 3 → 4 → 5
HORIZONS = (1, 2, 4, 6)                        # trimestres vista: 3, 6, 12 y 18 meses
#                                              (4 ≈ la próxima temporada = lo útil)


def _one_lineage(arr, bins, nb, symbols, idx, horizon=1):
    """Un remuestreo → {método: (error_modelo, error_naive)} para TODOS los métodos.

    ``horizon`` = cuántos bins al futuro se predice. A 1 trimestre la ingenua es casi
    imbatible (al virus no le ha dado tiempo a cambiar); el campo predice a la próxima
    TEMPORADA, que es donde la ingenua se rompe y donde está la utilidad real.

    ``crudo``   = re-agrupar desde cero en cada fold (las etiquetas bailan).
    ``estable`` = designar linajes UNA vez y solo EXTENDER (estilo Pango) → las
                  trayectorias de linaje son comparables en el tiempo.  [palanca 3]
    ``+shrink`` = además, tasas de crecimiento encogidas por evidencia.   [palanca 4]
    ``+arbol``  = además, fitness propagado por la jerarquía de linajes.  [palanca 5]

    Las tres variantes estables comparten EL MISMO sistema de linajes — solo cambia la
    proyección final — así que se designa una vez por fold y se reusa (3× más rápido).
    La naive se calcula sobre CADA partición: el listón de "persistir" no es el mismo
    con clados toscos que con linajes estables.
    """
    a, b = arr[idx], bins[idx]
    acc = {m: [0.0, 0.0] for m in METHODS}
    folds = {"crudo": 0, "estable": 0}
    system = None                                      # se ARRASTRA entre folds
    for k in range(2, nb - horizon + 1):
        at, bt = a[b < k], b[b < k]
        aq = a[b == k + horizon - 1]                   # el futuro que hay que acertar
        if at.shape[0] == 0 or aq.shape[0] == 0:
            continue

        labels, m, key, seeds = _clade_model(at, symbols, N_CLADES, MIN_COUNT, KEY)
        if m >= 2:
            cf = _clade_freqs(labels, bt, k, m)
            actual = np.bincount(_assign_clades(aq, key, seeds), minlength=m) / aq.shape[0]
            acc["crudo"][0] += float(np.abs(_project_freqs(cf, False, steps=horizon)
                                            - actual).sum())
            acc["crudo"][1] += float(np.abs(cf[-1] - actual).sum())
            folds["crudo"] += 1

        system = designate_lineages(at, symbols, prior=system, key_sites=KEY_LIN,
                                    max_lineages=MAX_LIN, min_size=MIN_SIZE)
        m = system.n
        if m < 2:
            continue
        labels = _assign_lineages(at, system)
        cf = _clade_freqs(labels, bt, k, m)            # (k, m) trayectoria de linajes
        cc = _clade_counts(labels, bt, k, m)           # evidencia por linaje
        actual = np.bincount(_assign_lineages(aq, system), minlength=m) / aq.shape[0]
        naive = float(np.abs(cf[-1] - actual).sum())   # persistir el último
        sizes = np.array([s.size for s in system.sites])
        variants = {
            "estable": {},
            "+shrink": {"counts": cc, "shrink": KAPPA},
            "+arbol": {"counts": cc, "shrink": KAPPA, "parents": system.parents,
                       "sizes": sizes},
        }
        for name, kw in variants.items():
            acc[name][0] += float(np.abs(_project_freqs(cf, False, steps=horizon, **kw)
                                         - actual).sum())
            acc[name][1] += naive
        folds["estable"] += 1

    out = {}
    for name in METHODS:
        f = folds["crudo"] if name == "crudo" else folds["estable"]
        out[name] = (acc[name][0] / f, acc[name][1] / f) if f else (np.nan, np.nan)
    return out


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
    idx_all = np.arange(len(seqs))
    print(f"  {label}  (n={len(seqs)}, bins={nb})")
    for h in HORIZONS:
        rng = np.random.default_rng(0)                 # misma semilla por horizonte
        skills = {m: [] for m in METHODS}
        for it in range(N_BOOT):
            idx = idx_all if it == 0 else _resample(bins, nb, rng)
            res = _one_lineage(arr, bins, nb, symbols, idx, horizon=h)
            for m in METHODS:
                em, en = res[m]
                if not np.isnan(em) and en > 0:
                    skills[m].append(1.0 - em / en)
        print(f"    ── horizonte {h} trimestre(s) ≈ {h * 3} meses "
              f"{'← la temporada siguiente (terreno de Łuksza)' if h == 4 else ''}")
        for m in METHODS:
            s = np.array(skills[m])
            if s.size == 0:
                continue
            lo, hi = np.percentile(s, [2.5, 97.5])
            robust = "✓ WIN robusto" if lo > 0 else ("~ empate" if hi > 0 else "✗ pierde")
            print(f"       {m:8s} skill = {s.mean():+.4f}  "
                  f"IC95% [{lo:+.4f}, {hi:+.4f}]  {robust}")
    print()


def main():
    print("NIVEL LINAJE — ¿predice qué CLADO crece mejor que persistir? (como evofr)")
    print("crudo = re-agrupar cada fold (lo de antes) · estable = Pango/autolin (nuevo)")
    print("HORIZONTE: a 1 trimestre la ingenua es casi imbatible (al virus no le ha dado")
    print("tiempo a cambiar). El campo predice la próxima TEMPORADA — la OMS elige la")
    print("cepa vacunal 9-12 meses antes. Ahí es donde la ingenua se rompe.")
    print("Skill sobre la distribución de clados. WIN si el IC95% está entero > 0.\n")
    only = sys.argv[1].lower() if len(sys.argv) > 1 else None
    for label, term in ORGANISMS:
        if only is None or only in label.lower():
            evaluate(label, term)


if __name__ == "__main__":
    main()
