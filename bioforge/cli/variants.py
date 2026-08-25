"""
bioforge/cli/variants.py — llamada de variantes desde la terminal (bioforge-variants).

Toma un genoma de referencia y un archivo de lecturas, corre la tubería entera
(mapeo → pileup → llamada) y escribe un VCF. Es la cara de línea de comandos de
:mod:`bioforge.variants`, pensada para meterla en un script o un pipeline.

Ejemplos
--------
    # variantes de unas lecturas contra una referencia, a la salida estándar
    bioforge-variants ref.fasta lecturas.fastq

    # guardar el VCF y bajar el umbral para buscar variantes minoritarias
    bioforge-variants ref.fasta lecturas.fastq -o llamadas.vcf --min-af 0.05

    # datos ruidosos (nanoporo): ajusta la tasa de error asumida
    bioforge-variants ref.fasta reads.fastq --error-rate 0.05

    # solo el informe de cobertura, sin llamar variantes
    bioforge-variants ref.fasta lecturas.fastq --solo-cobertura
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from ..core.biocore import BioForgeError, SmartImporter


def _leer_referencia(path: str) -> tuple[str, str]:
    """Primera secuencia del FASTA de referencia → (nombre, secuencia)."""
    registros = list(SmartImporter.from_file(path))
    if not registros:
        raise BioForgeError(f"la referencia {path!r} no contiene secuencias")
    if len(registros) > 1:
        print(f"[aviso] {path} trae {len(registros)} secuencias; se usa la primera "
              f"({registros[0].header[:40]}). El multi-contig llegará más adelante.",
              file=sys.stderr)
    r = registros[0]
    return r.header.split()[0] if r.header else "ref", r.to_string().upper()


def _leer_lecturas(path: str) -> list[str]:
    """Lecturas de un FASTQ o FASTA."""
    ext = path.lower()
    if ext.endswith((".fastq", ".fq", ".fastq.gz", ".fq.gz")):
        return [r.sequence.to_string().upper() for r in SmartImporter.stream_fastq(path)]
    return [r.to_string().upper() for r in SmartImporter.from_file(path)]


def _barra(frac: float, ancho: int = 28) -> str:
    lleno = int(round(frac * ancho))
    return "█" * lleno + "·" * (ancho - lleno)


def main(argv: Optional[list[str]] = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(
        prog="bioforge-variants",
        description="Encuentra las mutaciones de unas lecturas frente a un genoma "
                    "de referencia y las escribe en VCF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="La tubería es: mapeo → pileup → llamada. Con datos ruidosos "
               "(nanoporo, ~5%% de error) sube --error-rate: es lo que evita los "
               "falsos positivos.")
    p.add_argument("referencia", help="FASTA del genoma de referencia")
    p.add_argument("lecturas", help="FASTQ/FASTA con las lecturas a comparar")
    p.add_argument("-o", "--salida", help="archivo VCF (por defecto, salida estándar)")
    p.add_argument("--min-depth", type=int, default=5, dest="min_depth",
                   help="profundidad mínima para llamar (5)")
    p.add_argument("--min-af", type=float, default=0.2, dest="min_af",
                   help="fracción alélica mínima (0.2); bájala para minoritarias")
    p.add_argument("--min-qual", type=float, default=20.0, dest="min_qual",
                   help="calidad Phred mínima (20)")
    p.add_argument("--error-rate", type=float, default=0.01, dest="error_rate",
                   help="tasa de error asumida del secuenciador (0.01 ≈ Illumina; "
                        "usa 0.05 para nanoporo)")
    p.add_argument("--min-mapq", type=int, default=0, dest="min_mapq",
                   help="descarta mapeos por debajo de esta calidad (0)")
    p.add_argument("--sin-indels", action="store_true", dest="sin_indels",
                   help="llamar solo sustituciones")
    p.add_argument("--solo-cobertura", action="store_true", dest="solo_cobertura",
                   help="solo el informe de cobertura, sin llamar variantes")
    args = p.parse_args(argv)

    # imports aquí dentro: así 'bioforge-variants --help' no carga el motor entero
    from ..mapping.genomemap import GenomeAligner
    from ..variants.caller import call_variants, write_vcf
    from ..variants.pileup import pileup

    try:
        nombre, ref = _leer_referencia(args.referencia)
        lecturas = _leer_lecturas(args.lecturas)
    except BioForgeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if not lecturas:
        print(f"Error: {args.lecturas} no contiene lecturas.", file=sys.stderr)
        return 1

    print(f"referencia : {nombre}  ({len(ref):,} pb)", file=sys.stderr)
    print(f"lecturas   : {len(lecturas):,}", file=sys.stderr)

    aligner = GenomeAligner(ref, name=nombre)   # mismo nombre de contig en todo
    pares = []
    sin_mapear = 0
    for lectura in lecturas:                         # bucle por LECTURA (registro)
        hits = aligner.map(lectura)
        if hits:
            pares.append((lectura, hits[0]))
        else:
            sin_mapear += 1
    pct = 100.0 * len(pares) / len(lecturas)
    print(f"mapeadas   : {len(pares):,} ({pct:.1f}%)"
          + (f" · {sin_mapear:,} sin mapear" if sin_mapear else ""), file=sys.stderr)

    pile = pileup(ref, pares, contig=nombre, min_mapq=args.min_mapq)
    if pile.n_reads == 0 and pares:
        print("Error: no se apiló ninguna lectura. Revisa que la referencia sea la "
              "correcta para estas lecturas.", file=sys.stderr)
        return 1
    if pile.n_skipped:
        print(f"[aviso] {pile.n_skipped:,} mapeos descartados "
              f"(sin CIGAR o mapq < {args.min_mapq})", file=sys.stderr)

    print(f"\ncobertura del genoma", file=sys.stderr)
    print(f"  profundidad media : {pile.mean_depth:.1f}×", file=sys.stderr)
    for umbral in (1, 5, 10, 30):
        frac = pile.covered(umbral)
        print(f"  ≥{umbral:>3}× : {_barra(frac)} {frac*100:5.1f}%", file=sys.stderr)

    if args.solo_cobertura:
        return 0
    if pile.mean_depth < args.min_depth:
        print(f"\n[aviso] la profundidad media ({pile.mean_depth:.1f}×) está por debajo "
              f"de --min-depth ({args.min_depth}): probablemente no se llame casi nada.",
              file=sys.stderr)

    variantes = call_variants(
        pile, ref, min_depth=args.min_depth, min_af=args.min_af,
        min_qual=args.min_qual, error_rate=args.error_rate,
        indels=not args.sin_indels)

    n_snv = sum(1 for v in variantes if v.kind == "SNV")
    n_ind = len(variantes) - n_snv
    print(f"\nvariantes  : {len(variantes)}  ({n_snv} sustituciones, {n_ind} indels)",
          file=sys.stderr)

    texto = write_vcf(variantes, contigs=[(nombre, len(ref))])
    if args.salida:
        with open(args.salida, "w", encoding="utf-8") as fh:
            fh.write(texto)
        print(f"escrito    : {args.salida}", file=sys.stderr)
    else:
        sys.stdout.write(texto)
    return 0


if __name__ == "__main__":
    sys.exit(main())
