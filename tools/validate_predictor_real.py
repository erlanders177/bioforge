"""
tools/validate_predictor_real.py
══════════════════════════════════════════════════════════════════════
La PRUEBA DE FUEGO: valida el predictor de evolución con gripe H3N2 real de NCBI.

Baja la hemaglutinina (HA) de varios años, la alinea, y pasa el árbitro de
backtesting. Reporta el número HONESTO en dos vistas:
  - GLOBAL: exactitud media sobre TODAS las posiciones (dominada por las que nunca
    cambian → naive sale altísimo; difícil de batir, y así debe ser).
  - SITIOS QUE CAMBIAN: exactitud solo donde el consenso varía en el entrenamiento
    (los sitios "bajo presión") → aquí es donde el método gana o pierde de verdad.

No se maquilla nada: se imprime lo que salga.
"""

import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from bioforge.evolution import (  # noqa: E402
    _bin_ids,
    _clade_consensus_idx,
    _clade_freqs,
    _clade_labels,
    _consensus_idx,
    _freqs,
    _predict_idx,
    _prepare,
    _project_dominant,
)
from bioforge.evolution.fetch import fetch_dated  # noqa: E402

# Varios virus → prueba de GENERALIDAD (la promesa de la caja de herramientas).
# Todos usan el nombre de cepa con /AÑO, así que fetch_dated los parsea igual.
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
YEARS = range(2011, 2024)          # más años → más folds (robustez)
PER_YEAR = 40                      # secuencias/año


def backtest_detailed(seqs, times, method):
    arr, t, symbols = _prepare(seqs, times, align=True)
    bins, nb = _bin_ids(t, None)
    freq = _freqs(arr, bins, nb, symbols)
    L = arr.shape[1]
    om = on = ot = 0                # global: método, naive, total
    cm = cn = ct = 0               # sitios que cambian
    for k in range(2, nb):
        actual = _consensus_idx(freq[k])
        m_pred = _predict_idx(freq[:k], method)
        n_pred = _predict_idx(freq[:k], "naive")
        cons_train = np.stack([_consensus_idx(freq[b]) for b in range(k)])
        changing = (cons_train != cons_train[0]).any(axis=0)     # (L,)
        om += int((m_pred == actual).sum()); on += int((n_pred == actual).sum()); ot += L
        if changing.any():
            cm += int((m_pred[changing] == actual[changing]).sum())
            cn += int((n_pred[changing] == actual[changing]).sum())
            ct += int(changing.sum())
    return dict(nb=nb, L=L,
                g_m=om / ot, g_n=on / ot,
                c_m=cm / ct if ct else float("nan"),
                c_n=cn / ct if ct else float("nan"),
                c_total=ct)


def backtest_clade(seqs, times, *, n_clades=15, min_count=3, key_sites=50, garw=False):
    """Backtest a nivel de clado, LEAK-FREE: clusteriza solo con datos < k en cada fold."""
    arr, t, symbols = _prepare(seqs, times, align=True)
    bins, nb = _bin_ids(t, None)
    freq = _freqs(arr, bins, nb, symbols)          # para consenso real/naive por bin
    L = arr.shape[1]
    # dos predictores: clado puro, e híbrido (naive de base + clado en sitios variables)
    acc = {n: dict(om=0, cm=0) for n in ("clade", "hybrid")}
    on = ot = cn = ct = 0
    for k in range(2, nb):
        train = bins < k
        sub = arr[train]
        labels, m = _clade_labels(sub, symbols, n_clades, min_count, key_sites)
        cf = _clade_freqs(labels, bins[train], k, m)
        dom = int(_project_dominant(cf, garw).argmax())
        clade = _clade_consensus_idx(sub, symbols, labels == dom)   # consenso del clado
        naive = _consensus_idx(freq[k - 1])
        # sitios variables en el entrenamiento (criterio leak-free) → pisar solo ahí
        counts_tr = np.stack([(sub == s).sum(axis=0) for s in symbols])
        variable = (sub.shape[0] - counts_tr.max(axis=0)) >= min_count
        hybrid = np.where(variable, clade, naive)

        actual = _consensus_idx(freq[k])
        cons_train = np.stack([_consensus_idx(freq[b]) for b in range(k)])
        changing = (cons_train != cons_train[0]).any(axis=0)
        on += int((naive == actual).sum()); ot += L
        for name, pred in (("clade", clade), ("hybrid", hybrid)):
            acc[name]["om"] += int((pred == actual).sum())
            if changing.any():
                acc[name]["cm"] += int((pred[changing] == actual[changing]).sum())
        if changing.any():
            cn += int((naive[changing] == actual[changing]).sum())
            ct += int(changing.sum())
    out = {}
    for name in ("clade", "hybrid"):
        out[name] = dict(g_m=acc[name]["om"] / ot,
                         c_m=acc[name]["cm"] / ct if ct else float("nan"))
    out.update(nb=nb, L=L, g_n=on / ot,
               c_n=cn / ct if ct else float("nan"), c_total=ct)
    return out


def run_organism(label, term):
    data = fetch_dated(term, YEARS, per_year=PER_YEAR)
    if len(data) < 40:
        print(f"  {label}: muy pocas secuencias ({len(data)}) — saltando.")
        return None
    seqs = [s for s, _ in data]
    times = [y for _, y in data]
    cl = backtest_clade(seqs, times, garw=False)
    c_sk = (cl["clade"]["c_m"] - cl["c_n"]) / (1 - cl["c_n"]) if cl["c_n"] < 1 else 0.0
    print(f"  {label:16s} n={len(seqs):4d} años={cl['nb']:2d}  "
          f"naive={cl['c_n']:.4f}  clado={cl['clade']['c_m']:.4f}  "
          f"Δ={cl['clade']['c_m']-cl['c_n']:+.4f}  skill={c_sk:+.3f}"
          f"  {'✓ GANA' if c_sk > 0 else '✗ pierde'}")
    return c_sk


def main():
    print("PRUEBA DE GENERALIDAD — ¿el modelo de clados bate a naive en varios virus?")
    print("(métrica: exactitud en los SITIOS QUE CAMBIAN, backtest leak-free)\n")
    print("Descargando de NCBI y pasando el árbitro por organismo...\n", flush=True)
    results = {}
    for label, term in ORGANISMS:
        results[label] = run_organism(label, term)
    wins = [v for v in results.values() if v is not None and v > 0]
    print(f"\nResumen: {len(wins)}/{sum(v is not None for v in results.values())} "
          f"organismos con skill > 0 (clado bate a la baseline ingenua).")


if __name__ == "__main__":
    main()
