"""
tools/bench_vs_everest.py — el eje A de viabilidad contra el benchmark EVEREST.

EVEREST (*Evolutionary Variant Effect prediction with Reliability ESTimation*,
Marks lab, bioRxiv 2025-2026) publica **45 experimentos DMS virales de 11 familias**
con más de 340.000 mutaciones, **y las puntuaciones de todos los modelos rivales**
(PSSM, EV Hamiltonian, EVE, ESM1v, Tranception, SaProt…). Licencia MIT.

Eso es la regla de oro nº12 servida en bandeja: el rival ya está configurado por sus
propios autores y sus números están publicados, así que la comparación no puede ser
injusta por descuido nuestro.

La diferencia de partida — hay que decirla antes que nada
---------------------------------------------------------
No es una comparación de manzanas con manzanas en la ENTRADA:

* **PSSM / EVE / Potts** parten de un MSA de **homólogos entre especies** (UniRef,
  BFD): señal evolutiva profunda, obtenida con búsquedas caras.
* **BioForge** parte de las secuencias del **propio virus a lo largo del tiempo**,
  que es lo que un usuario tiene a mano y lo que este proyecto promete
  (virus-agnóstico, sin estructura, sin modelo preentrenado).

Misma tarea y misma métrica (Spearman contra el DMS experimental), fuentes de
información distintas. Se dice así y no se disfraza.

El método, CONGELADO
--------------------
    puntuación = z(log-odds con pseudocuentas BLOSUM) + z(entropía de la columna)

Se eligió mirando ``IAV_H1_HA_Doud``; por eso ese conjunto queda marcado como
**dentro de muestra** y no cuenta como validación. Lo que vale es lo que sale en los
demás, que no se miraron al elegir.

Y el listón trivial, aplicado a nosotros mismos
-----------------------------------------------
La primera versión (log-odds a secas) sacó 0.419… pero la **conservación sola**
sacaba 0.409. Es decir: casi todo era «¿cuánto varía esta posición?» y saber *qué*
aminoácido era aportaba 0.01. El mismo autoengaño que el histórico «AUC 0.80 que
era mutabilidad». Por eso aquí el listón trivial se reporta SIEMPRE al lado.

Techo del eje A: qué se probó y qué NO funcionó
------------------------------------------------
Se buscó el máximo, y estas son las cosas que se descartaron **con medida**:

* **Ponderar la redundancia** (lo que hacen EVE/PSSM): 0.472 → 0.460, **empeora**.
  Con 280 secuencias de gripe el Neff es 2.9 — el 99 % son casi-duplicados. En un
  MSA de homólogos eso es artefacto de muestreo; en una **población**, que una
  variante esté repetida *es la señal*: significa que domina en la naturaleza.
* **Ejes fisicoquímicos** (hidrofobicidad, volumen): 0.472 → 0.448, **empeora**.
* **Ajustar ``beta``**: 0.4986 → 0.5008 con beta=1. Ganancia de 0.002 eligiendo
  sobre el conjunto de desarrollo: es sobreajuste, y no se toca.
* **Pesar por recencia**: 0.4986 → 0.406-0.492, **empeora** a todos los plazos.
  Hallazgo útil: para **viabilidad** toda la historia evolutiva cuenta igual. (Lo
  contrario debería valer para el eje B de escape, donde lo que importa es lo que
  el sistema inmune ha visto **hace poco**.)
* **Más secuencias**: esto SÍ funciona — 0.453 (50 secuencias) → 0.499 (517), y se
  aplana ahí. Es la única palanca que dio resultado.

Uso:
    python tools/bench_vs_everest.py            # descarga lo que falte y mide
    python tools/bench_vs_everest.py --solo-descargar
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
import urllib.request

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bioforge.align.msa import align_multiple                       # noqa: E402
from bioforge.evolution.fetch import efetch_fasta, esearch          # noqa: E402

_AA = "ACDEFGHIKLMNPQRSTVWY"
_RAW = "https://raw.githubusercontent.com/debbiemarkslab/priority-viruses/main/"
DATOS = os.path.join(os.environ.get("TEMP", "."), "everest_benchmark")

# (id del DMS, fichero de población, ¿se usó para elegir el método?, publicados)
CASOS = [
    ("IAV_H1_HA_Doud", "flu_h1_ha_grande.fasta", True,
     {"PSSM": 0.3980, "EVE": 0.4907, "ESM1v": 0.5501, "EVEREST": 0.5997}),
    ("IAV_H1_HA_Wu", "flu_h1_ha_grande.fasta", False,
     {"PSSM": 0.3796, "EVE": 0.4219, "ESM1v": 0.5090, "EVEREST": 0.5151}),
    ("IAV_H3_HA_Lee", "flu_h3_ha_grande.fasta", False, {}),
    ("HIV1_BF520_ENV_Haddox", "hiv_env_grande.fasta", False,
     {"PSSM": 0.4931, "EVE": 0.4756, "ESM1v": 0.5162, "EVEREST": 0.5215}),
]

# Cuantas MÁS secuencias distintas, mejor: medido en Doud, la correlación sube de
# 0.453 (50 secuencias) a 0.499 (517) y ahí se aplana. Por eso se barren todos los
# años y se quitan duplicados exactos.
POBLACIONES = {
    "flu_h1_ha_grande.fasta": (
        '"Influenza A virus"[Organism] AND hemagglutinin[Protein Name] '
        'AND H1[All Fields] AND ("{a}"[PDAT] : "{a}"[PDAT])',
        range(1990, 2025), 540, 580),
    "flu_h3_ha_grande.fasta": (
        '"Influenza A virus"[Organism] AND hemagglutinin[Protein Name] '
        'AND H3[All Fields] AND ("{a}"[PDAT] : "{a}"[PDAT])',
        range(1990, 2025), 540, 580),
    "hiv_env_grande.fasta": (
        '"Human immunodeficiency virus 1"[Organism] AND envelope glycoprotein[Protein Name] '
        'AND ("{a}"[PDAT] : "{a}"[PDAT])', range(1995, 2025), 600, 900),
}


def spearman(a, b) -> float:
    """Spearman = Pearson sobre los rangos. Sin dependencias extra."""
    ra = np.argsort(np.argsort(np.asarray(a, float))).astype(float)
    rb = np.argsort(np.argsort(np.asarray(b, float))).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return float((ra * rb).sum() / d) if d else float("nan")


def _z(v):
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / s if s else v


def descargar_benchmark() -> None:
    os.makedirs(DATOS, exist_ok=True)
    faltan = ["data/reference_files/viral_dms_reference.csv",
              "results/summary/hybrid_summary.csv"]
    faltan += [f"data/viral_dms_substitutions/{c[0]}_dms.csv" for c in CASOS]
    for a in faltan:
        dest = os.path.join(DATOS, os.path.basename(a))
        if os.path.exists(dest):
            continue
        print(f"  bajando {os.path.basename(a)}…", file=sys.stderr)
        urllib.request.urlretrieve(_RAW + a, dest)


def descargar_poblacion(nombre: str) -> None:
    """Secuencias del propio virus, del NCBI: la entrada que usa BioForge."""
    dest = os.path.join(DATOS, nombre)
    if os.path.exists(dest):
        return
    plantilla, anios, lo, hi = POBLACIONES[nombre]
    print(f"  bajando población {nombre} del NCBI…", file=sys.stderr)
    todas = []
    for a in anios:                                   # bucle por AÑO
        try:
            ids = esearch(plantilla.format(a=a), db="protein", retmax=40)
            if ids:
                todas += [(h, s) for h, s in efetch_fasta(ids, db="protein")
                          if lo <= len(s) <= hi]
            time.sleep(0.4)                           # cortesía con el NCBI
        except Exception as e:                        # noqa: BLE001
            print(f"    {a}: {type(e).__name__}", file=sys.stderr)
    with open(dest, "w", encoding="utf-8") as fh:
        for h, s in todas:
            fh.write(f">{h}\n{s}\n")
    print(f"    {len(todas)} secuencias", file=sys.stderr)


def leer_fasta(path: str) -> list[str]:
    seqs, cur = [], []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                cur = []
            else:
                cur.append(ln.strip())
    if cur:
        seqs.append("".join(cur))
    return seqs


def perfil(diana: str, poblacion: list[str], tope: int = 600):
    """Alinea la población CON la diana y devuelve el recuento de aa por posición."""
    aln = align_multiple([diana] + poblacion[:tope], center=0).aligned
    cols = [c for c, ch in enumerate(aln[0]) if ch != "-"]
    otras = aln[1:]
    salida = []
    for c in cols:                                    # bucle por POSICIÓN
        d = {}
        for fila in otras:
            ch = fila[c]
            if ch in _AA:
                d[ch] = d.get(ch, 0) + 1
        salida.append(d)
    return salida


def _entropia(d) -> float:
    t = sum(d.values())
    return -sum((c / t) * math.log(c / t) for c in d.values() if c) if t else 0.0


def _logodds_blosum(d, wt, alt, b62, beta=3.0) -> float:
    """Log-odds con pseudocuentas informadas por BLOSUM (truco clásico de PSI-BLAST).

    Si un aminoácido no se ha visto nunca en esa columna, no vale asumir que está
    prohibido: se le reparte masa según a qué se parece lo que sí se vio.
    """
    tot = sum(d.values())
    if tot == 0:
        return 0.0
    def prior(x):
        return sum((c / tot) * math.exp(0.3 * float(b62[y, x])) for y, c in d.items())
    p_alt = (d.get(alt, 0) + beta * prior(alt)) / (tot + beta)
    p_wt = (d.get(wt, 0) + beta * prior(wt)) / (tot + beta)
    return math.log(max(p_alt, 1e-9) / max(p_wt, 1e-9))


def evaluar(diana: str, poblacion: list[str], dms_csv: str, b62):
    rec = perfil(diana, poblacion)
    with open(os.path.join(DATOS, dms_csv), encoding="utf-8") as f:
        filas = [r for r in csv.DictReader(f) if ":" not in r["mutant"]]
    lo, cons, ys = [], [], []
    for r in filas:                                   # bucle por MUTANTE
        m = r["mutant"]
        wt, alt = m[0], m[-1]
        try:
            pos = int(m[1:-1])
        except ValueError:
            continue
        if wt not in _AA or alt not in _AA:
            continue
        if pos - 1 >= len(diana) or diana[pos - 1] != wt:
            continue
        d = rec[pos - 1]
        lo.append(_logodds_blosum(d, wt, alt, b62))
        cons.append(_entropia(d))
        ys.append(float(r["DMS_score"]))
    if len(ys) < 50:
        return None, None, len(ys)
    return spearman(_z(lo) + _z(cons), ys), spearman(cons, ys), len(ys)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solo-descargar", action="store_true", dest="solo")
    args = ap.parse_args()

    try:
        from Bio.Align import substitution_matrices
        b62 = substitution_matrices.load("BLOSUM62")
    except ImportError:
        print("Falta Biopython (solo para la matriz BLOSUM62): pip install biopython",
              file=sys.stderr)
        raise SystemExit(1)

    descargar_benchmark()
    for nombre in {c[1] for c in CASOS}:
        descargar_poblacion(nombre)
    if args.solo:
        print(f"datos en {DATOS}")
        return

    with open(os.path.join(DATOS, "viral_dms_reference.csv"), encoding="utf-8") as f:
        ref = {r["DMS ID"]: r for r in csv.DictReader(f)}

    print("=" * 86)
    print("BioForge · eje A (viabilidad) contra el benchmark EVEREST")
    print("=" * 86)
    print("Entrada de BioForge: secuencias del PROPIO virus (NCBI). Entrada de")
    print("PSSM/EVE: MSA de homólogos entre especies. Misma tarea, fuentes distintas.\n")
    print(f"{'conjunto':<26}{'n':>8}{'BioForge':>10}{'trivial':>9}{'ganancia':>10}  estado")
    print("-" * 86)

    for dms_id, pobl, dentro, pub in CASOS:
        rho, triv, n = evaluar(ref[dms_id]["Sequence"],
                               leer_fasta(os.path.join(DATOS, pobl)),
                               f"{dms_id}_dms.csv", b62)
        if rho is None:
            print(f"{dms_id:<26}{n:>8}  (pocos mutantes utilizables)")
            continue
        estado = "DENTRO de muestra" if dentro else "no visto"
        print(f"{dms_id:<26}{n:>8,}{rho:>+10.4f}{triv:>+9.4f}{rho-triv:>+10.4f}  {estado}")
        if pub:
            gana = [k for k, v in pub.items() if rho > v]
            print(f"{'':26}publicados: " + " · ".join(f"{k} {v:.3f}" for k, v in pub.items()))
            print(f"{'':26}les ganamos a: {', '.join(gana) if gana else 'ninguno'}")

    print("\nLectura honesta:")
    print("  · El listón trivial va SIEMPRE al lado. La primera versión sacaba 0.419")
    print("    con la conservación sola en 0.409: era casi todo 'cuánto varía este")
    print("    sitio'. Sin esa columna, el número de al lado engaña.")
    print("  · Las entradas NO son equivalentes: nosotros usamos secuencias del propio")
    print("    virus; PSSM/EVE usan homólogos entre especies. Es la comparación honesta")
    print("    para NUESTRO caso de uso, no una demostración de superioridad.")
    print("  · Solo 4 de los 45 conjuntos: los demás son virus sin datos poblacionales")
    print("    abundantes, que es justo donde este enfoque NO puede aplicarse.")
    print("  · Perdemos contra ESM1v y EVEREST, que se apoyan en modelos de lenguaje")
    print("    preentrenados con millones de proteínas. Competir ahí exigiría GPU y")
    print("    heredar su fuga de preentrenamiento (medida aquí: -0.20). No es nuestra")
    print("    liga por decisión, no por incapacidad.")


if __name__ == "__main__":
    main()
