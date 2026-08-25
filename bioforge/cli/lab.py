"""
bioforge/cli/lab.py — herramientas de laboratorio en la terminal (bioforge-lab).

Las tres preguntas que se hace alguien antes de tocar una pipeta:

    bioforge-lab enzimas   secuencia.fasta          ¿qué enzimas la cortan?
    bioforge-lab orfs      secuencia.fasta          ¿qué genes puede haber?
    bioforge-lab primers   secuencia.fasta          ¿sirven estos cebadores?

Ejemplos
--------
    # sitios de EcoRI y BamHI, con el gel de la digestión
    bioforge-lab enzimas plasmido.fasta --enzimas EcoRI,BamHI --circular

    # solo las enzimas que cortan UNA vez (las útiles para clonar)
    bioforge-lab enzimas plasmido.fasta --unicas

    # los genes candidatos más largos, con su proteína
    bioforge-lab orfs genoma.fasta --min 300 --top 5

    # diseñar una pareja de cebadores y simular la PCR
    bioforge-lab primers gen.fasta
    bioforge-lab primers gen.fasta --pcr ACGTTGCATGCAAGCT,TTGCATGCAAGCTTGG
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from ..core.biocore import BioForgeError, SmartImporter


def _leer(path: str) -> tuple[str, str]:
    registros = list(SmartImporter.from_file(path))
    if not registros:
        raise BioForgeError(f"{path!r} no contiene secuencias")
    r = registros[0]
    if len(registros) > 1:
        print(f"[aviso] {len(registros)} secuencias; se usa la primera.", file=sys.stderr)
    return (r.header.split()[0] if r.header else "seq"), r.to_string().upper()


def _cli_enzimas(args) -> int:
    from ..lab.restriction import digest, find_sites, gel, unique_cutters

    nombre, seq = _leer(args.fasta)
    print(f"{nombre}: {len(seq):,} pb"
          + (" (circular)" if args.circular else " (lineal)"), file=sys.stderr)

    if args.unicas:
        u = unique_cutters(seq, circular=args.circular)
        print(f"\n{len(u)} enzimas cortan EXACTAMENTE una vez "
              f"(las útiles para clonar):\n", file=sys.stderr)
        for i in range(0, len(u), 6):
            print("  " + "  ".join(f"{x:<10}" for x in u[i:i + 6]))
        return 0

    enzimas = args.enzimas.split(",") if args.enzimas else None
    if enzimas is None:
        sitios = find_sites(seq, circular=args.circular)
        cuenta: dict[str, int] = {}
        for s in sitios:
            cuenta[s.enzyme] = cuenta.get(s.enzyme, 0) + 1
        print(f"\n{len(cuenta)} enzimas del catálogo cortan esta secuencia:\n",
              file=sys.stderr)
        for e, c in sorted(cuenta.items(), key=lambda x: (x[1], x[0])):
            print(f"  {e:<10} {c:>3} corte{'s' if c > 1 else ''}")
        return 0

    d = digest(seq, enzimas, circular=args.circular)
    print(f"\ndigestión con {' + '.join(d.enzymes)}:", file=sys.stderr)
    print(f"  {d.n_cuts} cortes → {len(d.fragments)} fragmentos", file=sys.stderr)
    for s in d.sites:
        print(f"    {s.enzyme:<10} corta en {s.position:,}")
    print("\ngel:", file=sys.stderr)
    print(gel(d))
    return 0


def _cli_orfs(args) -> int:
    from ..lab.orf import find_orfs

    nombre, seq = _leer(args.fasta)
    orfs = find_orfs(seq, min_length=args.min, require_start=not args.sin_atg)
    print(f"{nombre}: {len(seq):,} pb → {len(orfs)} ORFs de ≥{args.min} nt",
          file=sys.stderr)
    if not orfs:
        print("  (ninguno: prueba a bajar --min o usar --sin-atg)", file=sys.stderr)
        return 0
    print(file=sys.stderr)
    for i, o in enumerate(orfs[:args.top], 1):
        marca = "" if o.has_stop else "  [TRUNCADO: quizá continúa]"
        print(f"  {i}. marco {o.strand}{abs(o.frame)}  {o.start:,}–{o.end:,}  "
              f"{o.length:,} nt  {o.n_aa} aa{marca}")
        if args.proteinas:
            for j in range(0, len(o.protein), 60):
                print(f"       {o.protein[j:j + 60]}")
    return 0


def _cli_primers(args) -> int:
    from ..lab.primers import design_primers, gc_percent, pcr, tm_nn

    nombre, seq = _leer(args.fasta)
    print(f"{nombre}: {len(seq):,} pb", file=sys.stderr)

    if args.pcr:
        partes = args.pcr.split(",")
        if len(partes) != 2:
            print("Error: --pcr necesita DOS cebadores separados por coma.",
                  file=sys.stderr)
            return 1
        f, r = partes[0].strip().upper(), partes[1].strip().upper()
        for etiqueta, c in (("directo", f), ("inverso", r)):
            try:
                print(f"  cebador {etiqueta}: {c}  Tm {tm_nn(c):.1f}°C  "
                      f"GC {gc_percent(c):.0f}%", file=sys.stderr)
            except BioForgeError as e:
                print(f"  cebador {etiqueta}: {e}", file=sys.stderr)
        productos = pcr(seq, f, r, max_mismatches=args.fallos, circular=args.circular)
        print(f"\n{len(productos)} producto(s) de PCR:", file=sys.stderr)
        if not productos:
            print("  ninguno: los cebadores no pegan (prueba --fallos 1)",
                  file=sys.stderr)
        for p in productos:
            print(f"  {p.start:,}–{p.end:,}  {p.length:,} pb")
        if len(productos) > 1:
            print("\n[aviso] más de un producto: en el gel saldrían varias bandas.",
                  file=sys.stderr)
        return 0

    par = design_primers(seq, target_tm=args.tm)
    if par is None:
        print("Error: la secuencia es demasiado corta para diseñar cebadores.",
              file=sys.stderr)
        return 1
    print(f"\npareja propuesta (objetivo Tm {args.tm}°C):\n", file=sys.stderr)
    for etiqueta, p in (("directo", par[0]), ("inverso", par[1])):
        print(f"  {etiqueta:<8} {p.sequence}")
        print(f"           {p.length} nt · Tm {p.tm:.1f}°C · GC {p.gc:.0f}%",
              file=sys.stderr)
        for a in p.warnings:
            print(f"           ⚠ {a}", file=sys.stderr)
    if not par[0].warnings and not par[1].warnings:
        print("\n  sin pegas: buenos candidatos.", file=sys.stderr)
    print(f"\n  diferencia de Tm entre los dos: {abs(par[0].tm - par[1].tm):.1f}°C"
          + ("  (bien: por debajo de 5°C)" if abs(par[0].tm - par[1].tm) < 5
             else "  ⚠ más de 5°C: uno pegará y el otro no"), file=sys.stderr)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(
        prog="bioforge-lab",
        description="Herramientas de laboratorio: enzimas de restricción, marcos "
                    "abiertos de lectura y cebadores de PCR.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="orden", required=True)

    pe = sub.add_parser("enzimas", help="dónde cortan las enzimas de restricción")
    pe.add_argument("fasta")
    pe.add_argument("--enzimas", help="lista separada por comas (ej. EcoRI,BamHI)")
    pe.add_argument("--circular", action="store_true", help="ADN circular (plásmido)")
    pe.add_argument("--unicas", action="store_true",
                    help="solo las que cortan una vez (para clonar)")
    pe.set_defaults(func=_cli_enzimas)

    po = sub.add_parser("orfs", help="marcos abiertos de lectura (genes candidatos)")
    po.add_argument("fasta")
    po.add_argument("--min", type=int, default=90, help="longitud mínima en nt (90)")
    po.add_argument("--top", type=int, default=10, help="cuántos mostrar (10)")
    po.add_argument("--sin-atg", action="store_true", dest="sin_atg",
                    help="tramos entre paradas, sin exigir ATG")
    po.add_argument("--proteinas", action="store_true", help="mostrar la proteína")
    po.set_defaults(func=_cli_orfs)

    pp = sub.add_parser("primers", help="diseñar cebadores o simular una PCR")
    pp.add_argument("fasta")
    pp.add_argument("--tm", type=float, default=60.0, help="Tm objetivo (60)")
    pp.add_argument("--pcr", help="simular con estos dos cebadores: DIRECTO,INVERSO")
    pp.add_argument("--fallos", type=int, default=0, dest="fallos",
                    help="bases mal apareadas permitidas (0)")
    pp.add_argument("--circular", action="store_true")
    pp.set_defaults(func=_cli_primers)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except BioForgeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
