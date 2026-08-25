"""
bioforge/cli/phylo.py — árboles evolutivos desde la terminal (bioforge-phylo).

Toma un FASTA de secuencias emparentadas, las alinea, calcula las distancias,
construye el árbol y lo dibuja en la propia terminal, además de escribirlo en
**Newick** para abrirlo en MEGA, FigTree o iTOL.

Ejemplos
--------
    # árbol por Neighbor-Joining, dibujado en pantalla
    bioforge-phylo secuencias.fasta

    # con soporte por bootstrap (la confianza de cada rama) y guardado
    bioforge-phylo secuencias.fasta --bootstrap 500 -o arbol.newick

    # comparar métodos y modelos de sustitución
    bioforge-phylo secuencias.fasta --method upgma --model k2p

    # solo la matriz de distancias, en formato PHYLIP
    bioforge-phylo secuencias.fasta --solo-distancias
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from ..core.biocore import BioForgeError, SmartImporter


def _dibujar(nodo, prefijo: str = "", ultimo: bool = True,
             lineas: Optional[list[str]] = None, raiz: bool = True) -> list[str]:
    """Dibuja el árbol con caracteres de caja, estilo ``tree``.

    Bucle por NODO (no por símbolo): permitido, y además esto es presentación.
    """
    if lineas is None:
        lineas = []
    if raiz:
        lineas.append("·")
    else:
        rama = "└─" if ultimo else "├─"
        if nodo.is_leaf:
            lineas.append(f"{prefijo}{rama} {nodo.name}  ({nodo.length:.4f})")
        else:
            sop = f"  [{nodo.support:.0f}%]" if nodo.support is not None else ""
            lineas.append(f"{prefijo}{rama}┐{sop}")
    hijos = nodo.children
    for i, h in enumerate(hijos):
        es_ultimo = i == len(hijos) - 1
        if raiz:
            nuevo_prefijo = ""
        else:
            nuevo_prefijo = prefijo + ("   " if ultimo else "│  ")
        _dibujar(h, nuevo_prefijo, es_ultimo, lineas, raiz=False)
    return lineas


def main(argv: Optional[list[str]] = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(
        prog="bioforge-phylo",
        description="Construye un árbol evolutivo a partir de un FASTA de "
                    "secuencias emparentadas.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="El soporte por bootstrap es lo que separa un dibujo bonito de un "
               "resultado: una rama por debajo del 70%% no la sostienen los datos.")
    p.add_argument("fasta", help="FASTA con las secuencias a comparar")
    p.add_argument("-o", "--salida", help="archivo Newick (por defecto, pantalla)")
    p.add_argument("--method", default="nj", choices=["nj", "upgma"],
                   help="nj = Neighbor-Joining (por defecto) · upgma = por media")
    p.add_argument("--model", default="jc", choices=["p", "jc", "k2p", "poisson"],
                   help="modelo de sustitución: jc (defecto), k2p (solo ADN), "
                        "poisson (proteínas), p (sin corregir)")
    p.add_argument("--bootstrap", type=int, default=0, metavar="N",
                   help="réplicas de bootstrap para el soporte (0 = ninguno; 100 típico)")
    p.add_argument("--seed", type=int, default=None, help="semilla, para reproducir")
    p.add_argument("--solo-distancias", action="store_true", dest="solo_dist",
                   help="imprime la matriz de distancias (PHYLIP) y termina")
    p.add_argument("--sin-alinear", action="store_true", dest="sin_alinear",
                   help="las secuencias YA vienen alineadas (misma longitud)")
    args = p.parse_args(argv)

    # imports perezosos: '--help' no debe cargar el motor
    from ..align.msa import align_multiple
    from ..phylo.distance import distance_matrix
    from ..phylo.tree import bootstrap_support, build_tree

    try:
        registros = list(SmartImporter.from_file(args.fasta))
    except BioForgeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if len(registros) < 3:
        print(f"Error: hacen falta al menos 3 secuencias para un árbol "
              f"(el archivo tiene {len(registros)}).", file=sys.stderr)
        return 1

    nombres = [(r.header.split()[0] if r.header else f"seq{i+1}")[:30]
               for i, r in enumerate(registros)]
    seqs = [r.to_string().upper() for r in registros]
    print(f"secuencias : {len(seqs)}", file=sys.stderr)

    if args.sin_alinear:
        largos = {len(s) for s in seqs}
        if len(largos) != 1:
            print(f"Error: con --sin-alinear todas deben medir lo mismo "
                  f"(hay longitudes {sorted(largos)}).", file=sys.stderr)
            return 1
        alineadas = seqs
    else:
        print("alineando  : …", file=sys.stderr)
        alineadas = align_multiple(seqs).aligned
    print(f"columnas   : {len(alineadas[0])}", file=sys.stderr)

    try:
        dm = distance_matrix(alineadas, model=args.model, names=nombres)
    except BioForgeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if dm.saturated:
        print(f"[aviso] {dm.saturated} parejas están tan divergentes que el modelo "
              f"'{args.model}' no puede corregirlas: su distancia se recortó. "
              f"El árbol en esa zona es poco fiable.", file=sys.stderr)

    if args.solo_dist:
        sys.stdout.write(dm.to_text())
        return 0

    try:
        if args.bootstrap > 0:
            print(f"bootstrap  : {args.bootstrap} réplicas…", file=sys.stderr)
            arbol = bootstrap_support(alineadas, names=nombres, method=args.method,
                                      model=args.model, replicates=args.bootstrap,
                                      seed=args.seed)
        else:
            arbol = build_tree(alineadas, names=nombres, method=args.method,
                               model=args.model)
    except BioForgeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\nárbol ({args.method.upper()}, modelo {args.model}, "
          f"{'sin raíz' if not arbol.rooted else 'con raíz'}):", file=sys.stderr)
    for linea in _dibujar(arbol.root):
        print(linea, file=sys.stderr)

    if args.bootstrap > 0:
        flojas = []
        def revisar(n):
            if n.support is not None and n.support < 70:
                flojas.append(n.support)
            for h in n.children:
                revisar(h)
        revisar(arbol.root)
        if flojas:
            print(f"\n[aviso] {len(flojas)} rama(s) con soporte < 70 %: los datos no "
                  f"las sostienen. No las presentes como resultado.", file=sys.stderr)
        else:
            print("\nTodas las ramas internas superan el 70 % de soporte.",
                  file=sys.stderr)

    newick = arbol.newick() + "\n"
    if args.salida:
        with open(args.salida, "w", encoding="utf-8") as fh:
            fh.write(newick)
        print(f"\nescrito    : {args.salida}  (ábrelo en MEGA, FigTree o iTOL)",
              file=sys.stderr)
    else:
        sys.stdout.write(newick)
    return 0


if __name__ == "__main__":
    sys.exit(main())
