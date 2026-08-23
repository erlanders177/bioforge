"""
tools/bench_predictor_vs_sota.py
══════════════════════════════════════════════════════════════════════
El predictor de BioForge contra el ESTADO DEL ARTE, en su terreno y con su métrica.

Comparar nuestro AUC (¿qué mutación sube en frecuencia?) con el Spearman de EVEscape
o DERIVE (¿cuánto escapa esta mutación, medido en laboratorio?) sería tramposo: son
tareas distintas. Así que nos medimos donde ellos se miden:

  Benchmark : ProteinGym · A0A2Z5U3Z0_9INFA_Doud_2016
              10.715 mutaciones de hemaglutinina de gripe medidas EN LABORATORIO
              (deep mutational scanning, Doud & Bloom 2016)
  Métrica   : Spearman ρ entre la puntuación del modelo y el efecto experimental
  Rivales   : los 97 modelos de la tabla pública de ProteinGym (ESM, EVE, ESM-1v,
              Tranception, AIDO-16B…), con sus cifras oficiales.

Nuestros ejes se aplican en CERO-SHOT: no se ha entrenado nada sobre este ensayo.
La mutabilidad sale de secuencias de gripe REALES (caché de bioforge.evolution.fetch).
"""

import csv
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, ".")

from bioforge.evolution import (  # noqa: E402
    _bin_ids,
    _conservation_table,
    _freqs,
    _mutability,
    _mutability_gate,
    _prepare,
)

DMS = "data_real/dms_flu_ha.csv"
BOARD = "data_real/pg_leaderboard.csv"
ASSAY = "A0A2Z5U3Z0_9INFA_Doud_2016"


def spearman(x, y):
    """Correlación de Spearman sin scipy (rangos con empates promediados)."""
    def rank(v):
        order = np.argsort(v, kind="stable")
        r = np.empty(v.size, dtype=np.float64)
        r[order] = np.arange(1, v.size + 1)
        s = v[order]
        i = 0
        while i < s.size:
            j = i
            while j + 1 < s.size and s[j + 1] == s[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + j + 2) / 2.0
            i = j + 1
        return r
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d else 0.0


def load_dms():
    muts, scores = [], []
    with open(DMS, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            muts.append(row["mutant"])
            scores.append(float(row["DMS_score"]))
    return muts, np.array(scores)


def real_flu_axes():
    """Mutabilidad por sitio y frecuencias, desde secuencias de gripe REALES."""
    from bioforge.evolution.fetch import fetch_dated_precise
    term = ("Influenza A virus[Organism] AND H1N1 AND hemagglutinin[Title] "
            "AND 1650:1780[SLEN] AND {year}")
    data = fetch_dated_precise(term, range(2011, 2020), per_year=120)
    prot = []
    from bioforge.core.biocore import SeqType, SmartImporter
    from bioforge.sequence.translator import SmartTranslator as T
    packed = []
    for s, _ in data:
        try:
            packed.append(SmartImporter.from_string(
                f">x\n{s}\n", force_type=SeqType.NUCLEOTIDE)[0])
        except Exception:
            pass
    for p in T.translate_many(packed, warn_short=False):
        if p is not None and p.n_symbols > 400:
            prot.append(p.to_string())
    if len(prot) < 30:
        return None, None, None
    times = list(range(len(prot)))
    arr, t, symbols = _prepare(prot[:400], times[:400], align=True)
    bins, nb = _bin_ids(np.asarray(t, float), 8)
    freq = _freqs(arr, bins, nb, symbols)
    mut = _mutability_gate(_mutability(freq))
    return freq[-1], mut, symbols


def main():
    muts, dms = load_dms()
    print(f"Benchmark: {ASSAY}")
    print(f"  {len(muts):,} mutaciones de HA medidas en laboratorio\n")

    wt = np.array([m[0] for m in muts])
    alt = np.array([m[-1] for m in muts])
    pos = np.array([int(m[1:-1]) for m in muts])

    # ── Eje 1: conservación físico-química (sin datos, puro BioForge) ──────────
    syms = np.frombuffer("ACDEFGHIKLMNPQRSTVWY".encode(), dtype=np.uint8)
    tab = _conservation_table(syms)
    idx = {chr(int(s)): i for i, s in enumerate(syms)}
    cons = np.array([tab[idx[a], idx[w]] if a in idx and w in idx else 0.5
                     for w, a in zip(wt, alt)])
    print(f"  conservación (físico-química)      ρ = {spearman(cons, dms):+.3f}")

    # ── Eje 2: mutabilidad por sitio, de gripe REAL ───────────────────────────
    last, mut, symbols = real_flu_axes()
    if mut is not None:
        L = mut.size
        mv = np.array([mut[p - 1] if 1 <= p <= L else mut.mean() for p in pos])
        print(f"  mutabilidad (gripe real, {L} sitios)  ρ = {spearman(mv, dms):+.3f}")
        comb = 0.5 * (cons - cons.mean()) / (cons.std() or 1) + \
               0.5 * (mv - mv.mean()) / (mv.std() or 1)
        print(f"  COMBINADO (cons + mutabilidad)     ρ = {spearman(comb, dms):+.3f}")

    # ── La tabla pública ──────────────────────────────────────────────────────
    skip = {"DMS ID", "Number of Mutants", "Selection Type", "UniProt ID",
            "MSA Neff L div", "Taxon"}
    with open(BOARD, encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f) if r["DMS ID"] == ASSAY)
    vals = []
    for k, v in row.items():
        if k in skip:
            continue
        try:
            x = float(v)
            if -1 <= x <= 1:
                vals.append((k, x))
        except (TypeError, ValueError):
            pass
    vals.sort(key=lambda a: -a[1])
    print(f"\n  ── estado del arte ({len(vals)} modelos) ──")
    for k, v in vals[:5]:
        print(f"    {k:32s} {v:.3f}")
    print(f"    {'…':32s}")
    print(f"    mediana del campo               {np.median([v for _, v in vals]):.3f}")
    print(f"    peor                            {vals[-1][1]:.3f}")


if __name__ == "__main__":
    main()
