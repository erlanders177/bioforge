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
    _conservation_table,
    _freqs,
    _loglinear_fit,
    _mutability,
    _mutability_gate,
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


def _norm(x):
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def _scorers(freq, k, symbols, viab=None):
    """Puntuaciones (S, L) de cada mutación candidata, por método."""
    ftr = freq[:k]
    last = ftr[-1]                                     # frecuencia actual (S, L)
    _, slope = _loglinear_fit(ftr, weighted=False)     # eje A: crecimiento (S, L)
    cons_idx = last.argmax(axis=0)                     # alelo mayoritario por sitio
    conserv = _conservation_table(symbols)[:, cons_idx]   # (S, L) 1 − disimilitud
    mut = _mutability_gate(_mutability(ftr))           # (L,) accesibilidad histórica
    out = {
        "frecuencia": last,                            # el listón: casi contar
        "crecimiento(A)": slope,                       # medido: ruido (0.40-0.44)
        "conservacion(B')": conserv,                   # el escape CON EL SIGNO BUENO
        "mutabilidad": np.broadcast_to(mut, last.shape),   # accesibilidad sin 3D
        # fusión honesta: viable Y en un sitio que tolera cambios. SIN crecimiento,
        # porque está medido que es ruido — meterlo solo ensuciaría.
        "fusion(B'+mut)": _norm(conserv) * _norm(np.broadcast_to(mut, last.shape)),
    }
    if viab is not None:
        out["viabilidad(B:ESM-2)"] = viab
        out["fusion(ESM+mut)"] = _norm(viab) * _norm(np.broadcast_to(mut, last.shape))
    return out


def _esm_matrix(freq_train, symbols):
    """(S, L) de viabilidad ESM-2 sobre el consenso actual, alineada a las columnas.

    ESM no entiende huecos: se traduce el consenso a secuencia real, se pregunta una
    vez, y se devuelve el resultado a las columnas del MSA (las columnas de hueco se
    quedan a 0 = inviable, que es lo correcto)."""
    from bioforge.ai.viability import viability_matrix

    cons = "".join(chr(int(symbols[i])) for i in freq_train[-1].argmax(axis=0))
    cols = [j for j, c in enumerate(cons) if c != "-"]        # columna → posición real
    seq = cons.replace("-", "")
    alpha = [chr(int(s)) for s in symbols]
    sub = viability_matrix(seq, alpha)                        # (S, len(seq))
    out = np.zeros((len(symbols), len(cons)))
    out[:, cols] = sub
    return out


def evaluate(label, term, horizon, use_esm=False):
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

    names = ["frecuencia", "crecimiento(A)", "conservacion(B')", "mutabilidad",
             "fusion(B'+mut)"]
    if use_esm:
        names += ["viabilidad(B:ESM-2)", "fusion(ESM+mut)"]
    aucs = {r: {n: [] for n in names}
            for r in ("todas", "ya circulaba", "NUEVA", "NUEVA en sitio vivo")}
    for k in range(2, nb - horizon + 1):
        last = freq[:k][-1]
        target = freq[k + horizon - 1]
        if target.sum() == 0:
            continue
        cand = last < MINOR                            # no es el alelo mayoritario
        subio = (target - last) >= RISE                # la verdad: ¿ganó terreno?
        sc = _scorers(freq, k, symbols,
                      _esm_matrix(freq[:k], symbols) if use_esm else None)
        # El sitio ya varía = tiene HISTORIAL de cambio (solo datos < k, leak-free).
        # Es el test antitrampa de la mutabilidad: separar sitios congelados de sitios
        # vivos es FÁCIL y ya da un AUC enorme sin haber predicho nada interesante.
        # Si dentro de los sitios vivos la mutabilidad se cae a 0.5, toda su fuerza
        # era esa separación trivial.
        vivo = (freq[:k].max(axis=0) - freq[:k].min(axis=0) > 0).any(axis=0)
        regimes = {"todas": cand,
                   "ya circulaba": cand & (last > 0),
                   "NUEVA": cand & (last == 0),
                   "NUEVA en sitio vivo": cand & (last == 0) & vivo[None, :]}
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
    for rname in ("todas", "ya circulaba", "NUEVA", "NUEVA en sitio vivo"):
        vals = aucs[rname]
        if not any(vals[n] for n in names):
            continue
        nota = {"ya circulaba": "  (fácil: casi contar)",
                "NUEVA": "  ← la frecuencia no puede ayudar (0.5 por construcción)",
                "NUEVA en sitio vivo":
                    "  ← EL TEST ANTITRAMPA: sin sitios congelados que regalen AUC",
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
    args = [a.lower() for a in sys.argv[1:]]
    esm = "esm" in args                            # eje B real (pip install bioforge[ai])
    only = next((a for a in args if a != "esm"), None)
    if esm:
        print("Eje B = ESM-2 ACTIVO (una pasada por fold; la primera baja el modelo).\n")
    for label, term in ORGANISMS:
        if only is None or only in label.lower():
            for h in HORIZONS:
                evaluate(label, term, h, use_esm=esm)


if __name__ == "__main__":
    main()
