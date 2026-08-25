"""
tools/bench_vs_biopython_phylo.py — nuestra filogenia CONTRA el estándar.

Construir un árbol es fácil; construirlo **bien** es lo que hay que demostrar. Y
la única demostración que vale es contrastar contra una implementación
independiente y reconocida: **Biopython** (``Bio.Phylo.TreeConstruction``), que es
la referencia de facto en Python y lleva dos décadas en uso.

Se comparan tres cosas, y en este orden de importancia:

1. **¿La misma topología?** — es lo único que de verdad importa en un árbol. Se
   comparan las *biparticiones* (qué hojas quedan a cada lado de cada rama), que
   es la forma canónica de decir si dos árboles son el mismo.
2. **¿Las mismas distancias?** — error máximo entre ambas matrices.
3. **¿A qué velocidad?** — lo último, porque un árbol rápido y equivocado no sirve.

Uso:
    python tools/bench_vs_biopython_phylo.py
    python tools/bench_vs_biopython_phylo.py --rapido
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, __file__.rsplit("tools", 1)[0])

from bioforge.phylo.distance import distance_matrix          # noqa: E402
from bioforge.phylo.tree import (  # noqa: E402
    _particiones, neighbor_joining, upgma, wpgma)

try:
    from Bio.Align import MultipleSeqAlignment
    from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
except ImportError:                                          # pragma: no cover
    print("Falta Biopython. Instálalo con:  pip install biopython", file=sys.stderr)
    raise SystemExit(1)


def simular(n_taxa: int, largo: int, semilla: int) -> tuple[list[str], list[str]]:
    """Secuencias con una genealogía real: se van partiendo linajes desde un ancestro."""
    rng = np.random.default_rng(semilla)

    def mutar(s: str, tasa: float) -> str:
        a = np.array(list(s))
        m = rng.random(len(s)) < tasa
        if m.any():
            a[m] = rng.choice(list("ACGT"), size=int(m.sum()))
        return "".join(a)

    poblacion = ["".join(rng.choice(list("ACGT"), size=largo))]
    while len(poblacion) < n_taxa:                   # bucle por LINAJE
        i = int(rng.integers(0, len(poblacion)))
        padre = poblacion.pop(i)
        poblacion += [mutar(padre, 0.04), mutar(padre, 0.04)]
    seqs = [mutar(s, 0.02) for s in poblacion[:n_taxa]]
    return seqs, [f"t{i:02d}" for i in range(len(seqs))]


def _particiones_biopython(arbol, hojas: set[str]) -> set[frozenset[str]]:
    """Biparticiones informativas de un árbol de Biopython (misma forma canónica)."""
    ancla = min(hojas)
    salida = set()
    for clado in arbol.get_nonterminals():
        abajo = {t.name for t in clado.get_terminals()}
        if min(len(abajo), len(hojas) - len(abajo)) >= 2:
            salida.add(frozenset(abajo if ancla not in abajo else hojas - abajo))
    return salida


def comparar(n_taxa: int, largo: int, semilla: int) -> dict:
    seqs, nombres = simular(n_taxa, largo, semilla)
    hojas = set(nombres)

    # ── BioForge ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    dm_bf = distance_matrix(seqs, model="jc", names=nombres)
    t_dist_bf = time.perf_counter() - t0
    t0 = time.perf_counter()
    nj_bf = neighbor_joining(dm_bf)
    t_nj_bf = time.perf_counter() - t0
    up_bf = upgma(dm_bf)
    wp_bf = wpgma(dm_bf)

    # ── Biopython ─────────────────────────────────────────────────────────────
    aln = MultipleSeqAlignment(
        [SeqRecord(Seq(s), id=n) for s, n in zip(seqs, nombres, strict=True)])
    calc = DistanceCalculator("identity")            # p-distance cruda
    t0 = time.perf_counter()
    dm_bp = calc.get_distance(aln)
    t_dist_bp = time.perf_counter() - t0
    cons = DistanceTreeConstructor()
    t0 = time.perf_counter()
    nj_bp = cons.nj(dm_bp)
    t_nj_bp = time.perf_counter() - t0
    up_bp = cons.upgma(calc.get_distance(aln))

    # ── 1. topología: ¿el MISMO árbol? ───────────────────────────────────────
    p_nj_bf = _particiones(nj_bf)
    p_nj_bp = _particiones_biopython(nj_bp, hojas)
    p_up_bf = _particiones(up_bf)
    p_wp_bf = _particiones(wp_bf)
    p_up_bp = _particiones_biopython(up_bp, hojas)

    def acuerdo(a: set, b: set) -> float:
        return 100.0 * len(a & b) / max(len(a | b), 1)

    # ── 2. distancias: BioForge 'p' contra la 'identity' de Biopython ────────
    dm_bf_p = distance_matrix(seqs, model="p", names=nombres).matrix
    D_bp = np.array([[dm_bp[i, j] for j in range(n_taxa)] for i in range(n_taxa)])
    err = float(np.max(np.abs(dm_bf_p - D_bp)))

    return {
        "n_taxa": n_taxa, "largo": largo,
        "nj_acuerdo": acuerdo(p_nj_bf, p_nj_bp),
        "up_acuerdo": acuerdo(p_up_bf, p_up_bp),
        "nj_identico": p_nj_bf == p_nj_bp,
        "up_identico": p_up_bf == p_up_bp,
        "wp_identico": p_wp_bf == p_up_bp,
        "err_dist": err,
        "t_dist_bf": t_dist_bf, "t_dist_bp": t_dist_bp,
        "t_nj_bf": t_nj_bf, "t_nj_bp": t_nj_bp,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rapido", action="store_true")
    args = ap.parse_args()

    casos = ([(8, 500), (20, 800)] if args.rapido
             else [(6, 400), (10, 600), (20, 800), (40, 1000), (60, 1200)])

    print("=" * 84)
    print("BioForge vs Biopython — filogenia por distancias")
    print("=" * 84)
    print("Biopython 'identity' = p-distance cruda · BioForge 'p' para comparar matrices,")
    print("'jc' (Jukes-Cantor) para construir el árbol.\n")

    print(f"{'taxones':>8} {'cols':>6} | {'NJ':>6} {'ns.UPGMA':>9} {'ns.WPGMA':>9} "
          f"{'err.dist':>10} | {'distancias':>18} {'NJ':>16}")
    print("-" * 84)

    filas = []
    for n, L in casos:
        r = comparar(n, L, semilla=100 + n)
        filas.append(r)
        nj_txt = "SÍ" if r["nj_identico"] else f"{r['nj_acuerdo']:.0f}%"
        up_txt = "SÍ" if r["up_identico"] else f"{r['up_acuerdo']:.0f}%"
        wp_txt = "SÍ" if r["wp_identico"] else "no"
        print(f"{n:>8} {L:>6} | {nj_txt:>6} {up_txt:>9} {wp_txt:>9} "
              f"{r['err_dist']:>10.2e} | "
              f"{r['t_dist_bf']*1000:>7.1f} vs {r['t_dist_bp']*1000:>6.1f} ms "
              f"{r['t_nj_bf']*1000:>6.1f} vs {r['t_nj_bp']*1000:>5.1f} ms")

    print("\nLectura honesta:")
    todos_nj = all(f["nj_identico"] for f in filas)
    todos_up = all(f["up_identico"] for f in filas)
    err_max = max(f["err_dist"] for f in filas)
    print(f"  · Topología NJ idéntica a Biopython en {sum(f['nj_identico'] for f in filas)}"
          f"/{len(filas)} casos" + ("  ← concordancia total" if todos_nj else ""))
    n_wp = sum(f["wp_identico"] for f in filas)
    print(f"  · Topología UPGMA idéntica en {sum(f['up_identico'] for f in filas)}"
          f"/{len(filas)} casos — y NO es un fallo nuestro:")
    print(f"    el 'upgma()' de Biopython promedia (d(k,i)+d(k,j))/2, SIN ponderar por")
    print(f"    el tamaño del grupo. Eso es WPGMA, no UPGMA. Nuestro wpgma() reproduce")
    print(f"    su salida en {n_wp}/{len(filas)} casos; nuestro upgma() usa la media")
    print(f"    ponderada de la definición original (Sokal & Michener, 1958).")
    print(f"    Si necesitas reproducir Biopython exactamente, usa method='wpgma'.")
    print(f"  · Error máximo entre matrices de distancia: {err_max:.2e}"
          + ("  (precisión de máquina: son la MISMA matriz)" if err_max < 1e-9 else ""))
    d_med = np.median([f["t_dist_bp"] / max(f["t_dist_bf"], 1e-9) for f in filas])
    n_med = np.median([f["t_nj_bp"] / max(f["t_nj_bf"], 1e-9) for f in filas])
    print(f"  · Velocidad (mediana): distancias {d_med:.1f}× · NJ {n_med:.1f}× "
          f"{'más rápido' if d_med > 1 else 'MÁS LENTO'} que Biopython")
    print("\n  Lo que NO demuestra esto: que el árbol sea biológicamente correcto.")
    print("  Coincidir con Biopython prueba que el ALGORITMO está bien implementado,")
    print("  no que los métodos de distancia sean los mejores. Para eso está el")
    print("  bootstrap, y para máxima verosimilitud harían falta RAxML o IQ-TREE,")
    print("  que juegan en otra liga y no pretendemos igualar.")


if __name__ == "__main__":
    main()
