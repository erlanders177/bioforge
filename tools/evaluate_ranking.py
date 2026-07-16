"""
tools/evaluate_ranking.py
══════════════════════════════════════════════════════════════════════
ORDENACIÓN DE MUTACIONES — la pregunta que el campo SÍ responde.

Llevábamos semanas midiendo *"¿qué frecuencia tendrá cada cosa?"* → una REGRESIÓN.
Y ahí la ingenua ("mañana = hoy") es imbatible: el ~95% de los sitios no se mueve y
los acierta gratis. Medido: skill ≈ 0 en los 3 virus. No es que fuéramos malos —
es que ese deporte no se puede ganar, y por eso NADIE lo publica.

Aquí se mide lo que miden de verdad EVEscape (AUC de mutaciones que subieron) y
Łuksza (ordenación de clados por fitness): **¿CUÁLES mutaciones van a subir, y a QUÉ
alelo?** En ese tablero la ingenua dice "no cambia nada" → **no puntúa: ni juega**.

La unidad es la MUTACIÓN (sitio, alelo), no el sitio: un sitio no tiene carga, una
sustitución sí. Así una sola métrica responde al "cuál" y al "cómo" a la vez.

HONESTIDAD (lo que hace que esto valga algo):
  - Esto NO es "batir a la ingenua": es OTRA pregunta. La ingenua sigue siendo
    imbatible prediciendo frecuencias, y así se dice en el README.
  - El listón no es el azar (AUC 0.5), que es un listón de mentira. El listón es
    **la frecuencia actual**: "la mutación que ya está subiendo, seguirá subiendo".
    Si no batimos a CONTAR, no aportamos nada.
  - Se parte el resultado en dos regímenes, porque no valen lo mismo:
      · "ya circulaba" → fácil, casi contar. Un AUC alto aquí no es mérito.
      · "NUEVA" (frecuencia 0) → la frecuencia no puede decir nada (todas valen 0).
        Solo se puede acertar preguntando si la sustitución es VIABLE (eje B) y si
        ESCAPA (eje C). Ahí es donde se gana o se pierde de verdad.
"""

import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from bioforge.biocore import (  # noqa: E402
    SeqType,
    SequenceTypeError,
    SequenceValueError,
    SmartImporter,
)
from bioforge.evolution import (  # noqa: E402
    _bin_ids,
    _dissimilarity,
    _freqs,
    _loglinear_fit,
    _prepare,
)
from bioforge.fetch import fetch_dated_precise  # noqa: E402
from bioforge.smart_translator import SmartTranslator  # noqa: E402

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
HORIZONS = (1, 4)                  # 3 meses y la próxima temporada (~1 año)
RISE = 0.05                        # "subió" = ganó ≥5 puntos de frecuencia
MINOR = 0.5                        # candidata = no es el alelo mayoritario


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    """AUC = P(puntuación de una que sube > puntuación de una que no).

    Mann-Whitney sobre rangos, con empates promediados (sin scipy). 0.5 = azar."""
    npos = int(label.sum())
    nneg = int(label.size - npos)
    if npos == 0 or nneg == 0:
        return np.nan
    order = np.argsort(score, kind="stable")
    ranks = np.empty(score.size, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1)
    s = score[order]                                   # empates → rango promedio
    start = 0
    for i in range(1, s.size + 1):
        if i == s.size or s[i] != s[start]:
            if i - start > 1:
                ranks[order[start:i]] = (start + i + 1) / 2.0
            start = i
    return (ranks[label].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def _to_protein(seqs):
    """Traduce con NUESTRO traductor (dogfooding) y descarta lo que no cuadra.

    ``SmartTranslator`` arranca en el primer ATG y corta en el primer STOP en marco,
    que es justo lo que queremos para un CDS de HA completo. Solo se capturan los
    errores ESPERADOS (sin ATG / demasiado corta); cualquier otro debe explotar y
    verse — un ``except Exception`` mudo aquí ya me escondió un fallo una vez."""
    out, fallos = [], 0
    for s in seqs:
        try:
            packed = SmartImporter.from_string(f">s\n{s}\n",
                                               force_type=SeqType.NUCLEOTIDE)[0]
            p = SmartTranslator.translate(packed, warn_short=False).to_string()
        except (SequenceValueError, SequenceTypeError, ValueError, TypeError):
            out.append(None)
            fallos += 1
            continue
        out.append(p if len(p) >= 400 else None)       # HA completa ≈ 550-570 aa
    if fallos:
        print(f"    ({fallos}/{len(seqs)} sin ORF traducible — descartadas)")
    return out


def _scorers(freq, k, symbols, is_protein):
    """Puntuaciones (S, L) de cada mutación candidata, por método."""
    ftr = freq[:k]
    last = ftr[-1]                                     # frecuencia actual (S, L)
    _, slope = _loglinear_fit(ftr, weighted=False)     # eje A: crecimiento (S, L)
    out = {"frecuencia": last, "crecimiento(A)": slope}
    if is_protein:
        cons = last.argmax(axis=0)                     # alelo mayoritario por sitio
        tab = np.array([[_dissimilarity(chr(int(a)), chr(int(b))) for b in symbols]
                        for a in symbols])             # (S, S)
        esc = tab[:, cons]                             # (S, L) escape de cada mutación
        out["escape(C)"] = esc
        # fusión A+C: crecimiento normalizado × (1 + escape) — la mutación que sube Y
        # además es disruptiva es la candidata de verdad (lógica de EVEscape)
        g = slope - slope.min()
        g = g / (g.max() or 1.0)
        out["fusion(A+C)"] = g * (1.0 + esc)
    return out


def evaluate(label, term, horizon):
    data = fetch_dated_precise(term, YEARS, per_year=PER_YEAR)
    seqs = [s for s, _ in data]
    times = [t for _, t in data]
    prot = _to_protein(seqs)
    keep = [i for i, p in enumerate(prot) if p is not None]
    if len(keep) < 60:
        print(f"  {label}: solo {len(keep)} traducciones válidas.")
        return
    arr, t, symbols = _prepare([prot[i] for i in keep], [times[i] for i in keep],
                               align=True)
    bins, nb = _bin_ids(t, None)
    freq = _freqs(arr, bins, nb, symbols)              # (nb, S, L)
    is_protein = True

    names = ["frecuencia", "crecimiento(A)", "escape(C)", "fusion(A+C)"]
    aucs = {r: {n: [] for n in names} for r in ("todas", "ya circulaba", "NUEVA")}
    for k in range(2, nb - horizon + 1):
        last = freq[:k][-1]
        target = freq[k + horizon - 1]
        if target.sum() == 0:
            continue
        cand = last < MINOR                            # no es el alelo mayoritario
        subio = (target - last) >= RISE                # la verdad: ¿ganó terreno?
        sc = _scorers(freq, k, symbols, is_protein)
        regimes = {"todas": cand,
                   "ya circulaba": cand & (last > 0),
                   "NUEVA": cand & (last == 0)}
        for rname, mask in regimes.items():
            y = subio[mask]
            if y.sum() < 3 or (~y).sum() < 3:
                continue
            for n in names:
                a = _auc(sc[n][mask], y)
                if not np.isnan(a):
                    aucs[rname][n].append(a)

    print(f"  {label} · horizonte {horizon} ({horizon * 3} meses) · "
          f"n={len(keep)} proteínas, {nb} bins")
    for rname in ("todas", "ya circulaba", "NUEVA"):
        vals = aucs[rname]
        if not any(vals[n] for n in names):
            continue
        nota = {"ya circulaba": "  (fácil: casi contar)",
                "NUEVA": "  ← AQUÍ se gana o se pierde (la frecuencia no puede ayudar)",
                "todas": ""}[rname]
        print(f"    [{rname}]{nota}")
        for n in names:
            if vals[n]:
                a = np.array(vals[n])
                marca = "✓" if a.mean() > 0.55 else ("~" if a.mean() > 0.45 else "✗")
                print(f"       {n:16s} AUC = {a.mean():.3f} "
                      f"(±{a.std():.3f}, {len(a)} folds)  {marca}")
    print()


def main():
    print("ORDENACIÓN DE MUTACIONES — ¿cuáles suben, y a qué alelo? (como EVEscape)")
    print("La ingenua NO juega aquí: decir 'no cambia nada' no ordena nada.")
    print("El listón NO es el azar (0.5) sino LA FRECUENCIA ACTUAL: si no batimos a")
    print("contar, no aportamos. AUC 0.5 = azar · 1.0 = perfecto.\n")
    only = sys.argv[1].lower() if len(sys.argv) > 1 else None
    for label, term in ORGANISMS:
        if only is None or only in label.lower():
            for h in HORIZONS:
                evaluate(label, term, h)


if __name__ == "__main__":
    main()
