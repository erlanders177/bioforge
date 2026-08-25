"""
tools/bench_vs_bcftools.py — nuestro llamador de variantes CONTRA el estándar.

``bcftools`` (sobre htslib) es *la* referencia del campo para llamar variantes: lo
usan prácticamente todos los laboratorios del mundo. Comparar contra él es la única
forma de saber si nuestro llamador vale algo o solo se lo parece.

Se compara de DOS formas, y la distinción importa:

1. **Tubería contra tubería** — nuestro mapeador + nuestro llamador, frente a
   ``minimap2`` + ``bcftools``. Es lo que vive el usuario final, pero mezcla dos
   variables (mapeo y llamada).
2. **Llamador contra llamador** (*el contraste justo*) — los DOS partiendo de los
   **mismos alineamientos** de minimap2. Aquí solo cambia la estadística de la
   llamada, así que si hay diferencias son del llamador y de nadie más.

Poder hacer (2) es consecuencia de una decisión de diseño: nuestro llamador no
depende del mapeador, consume cualquier objeto con los atributos de un ``Mapping``.

Requisitos (se comprueban al arrancar):
    WSL con ``minimap2``, ``samtools`` y ``bcftools`` instalados.

Uso:
    python tools/bench_vs_bcftools.py
    python tools/bench_vs_bcftools.py --reads 2000 --cobertura 30
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bioforge.mapping.genomemap import GenomeAligner            # noqa: E402
from bioforge.variants.caller import call_variants              # noqa: E402
from bioforge.variants.pileup import pileup                     # noqa: E402

RC = str.maketrans("ACGT", "TGCA")


class MapeoSAM:
    """Un alineamiento leído de un SAM, con la forma que espera nuestro pileup.

    No hereda de ``Mapping`` a propósito: demuestra que el llamador funciona con
    cualquier objeto que exponga estos atributos (tipado pato).
    """

    __slots__ = ("cigar", "target_start", "query_start", "strand", "mapq", "target_name")

    def __init__(self, cigar, target_start, target_name, mapq):
        self.cigar = cigar
        self.target_start = target_start
        self.query_start = 0        # el SAM guarda la lectura entera; los 'S' del
        self.strand = "+"           # CIGAR ya colocan el desplazamiento, y SEQ viene
        self.mapq = mapq            # ya orientada a la referencia
        self.target_name = target_name


def wsl(cmd: str, cwd_win: str | None = None) -> subprocess.CompletedProcess:
    """Ejecuta una orden en WSL, opcionalmente dentro de una carpeta de Windows."""
    if cwd_win:
        ruta = subprocess.run(["wsl.exe", "wslpath", "-a", cwd_win.replace("\\", "/")],
                              capture_output=True, text=True).stdout.strip()
        cmd = f"cd '{ruta}' && {cmd}"
    return subprocess.run(["wsl.exe", "-e", "bash", "-lc", cmd],
                          capture_output=True, text=True, errors="replace")


def comprobar_entorno() -> None:
    r = wsl("for t in minimap2 samtools bcftools; do command -v $t >/dev/null "
            "|| echo FALTA:$t; done")
    faltan = [x.split(":")[1] for x in r.stdout.split() if x.startswith("FALTA:")]
    if faltan:
        print(f"Faltan en WSL: {', '.join(faltan)}", file=sys.stderr)
        print("Instálalas con:  wsl -u root -e bash -lc "
              "'apt-get update && apt-get install -y minimap2 samtools bcftools'",
              file=sys.stderr)
        raise SystemExit(1)


def generar(carpeta: str, largo: int, n_mut: int, n_reads: int, largo_read: int,
            error: float, semilla: int) -> dict[int, str]:
    """Crea referencia, muestra con mutaciones CONOCIDAS y lecturas. Devuelve la verdad."""
    rng = np.random.default_rng(semilla)
    ref = "".join(rng.choice(list("ACGT"), size=largo))
    posiciones = rng.choice(np.arange(500, largo - 500), size=n_mut, replace=False)
    muestra = list(ref)
    verdad: dict[int, str] = {}
    for p in sorted(int(x) for x in posiciones):
        nueva = str(rng.choice([b for b in "ACGT" if b != ref[p]]))
        muestra[p] = nueva
        verdad[p + 1] = nueva                        # 1-based, como el VCF
    muestra = "".join(muestra)

    with open(os.path.join(carpeta, "ref.fa"), "w", encoding="utf-8") as fh:
        fh.write(">cromosoma\n")
        for i in range(0, largo, 70):
            fh.write(ref[i:i + 70] + "\n")

    with open(os.path.join(carpeta, "reads.fq"), "w", encoding="utf-8") as fh:
        for i in range(n_reads):
            s = int(rng.integers(0, largo - largo_read))
            r = list(muestra[s:s + largo_read])
            for j in range(len(r)):
                if rng.random() < error:
                    r[j] = str(rng.choice(list("ACGT")))
            seq = "".join(r)
            if rng.random() < 0.5:
                seq = seq.translate(RC)[::-1]
            fh.write(f"@r{i}\n{seq}\n+\n{'I' * len(seq)}\n")
    return verdad


def leer_fasta(path: str) -> tuple[str, str]:
    nombre, trozos = "ref", []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            if ln.startswith(">"):
                nombre = ln[1:].split()[0]
            else:
                trozos.append(ln.strip())
    return nombre, "".join(trozos).upper()


def leer_fastq(path: str) -> list[str]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for k, ln in enumerate(fh):
            if k % 4 == 1:
                out.append(ln.strip().upper())
    return out


def parsear_sam(path: str) -> list[tuple[str, MapeoSAM]]:
    """SAM → [(secuencia, MapeoSAM)] quedándose solo con los alineamientos primarios."""
    pares = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            if ln.startswith("@"):
                continue
            c = ln.rstrip("\n").split("\t")
            if len(c) < 11:
                continue
            flag = int(c[1])
            if flag & 0x4 or flag & 0x100 or flag & 0x800:   # sin mapear/secundario/suplementario
                continue
            cigar, seq = c[5], c[9]
            if cigar == "*" or seq == "*":
                continue
            pares.append((seq, MapeoSAM(cigar, int(c[3]) - 1, c[2], int(c[4]))))
    return pares


def leer_vcf(path: str) -> set[tuple[int, str]]:
    """VCF → {(posición, alelo alternativo)} quedándose con las sustituciones."""
    fuera = set()
    with open(path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            if ln.startswith("#"):
                continue
            c = ln.split("\t")
            if len(c) < 5:
                continue
            ref, alts = c[3], c[4].split(",")
            for alt in alts:
                if len(ref) == 1 and len(alt) == 1 and alt not in (".", "<*>"):
                    fuera.add((int(c[1]), alt))
    return fuera


def evaluar(llamadas: set, verdad: dict) -> tuple[float, float, int, int]:
    reales = set(verdad.items())
    tp = len(llamadas & reales)
    fp = len(llamadas - reales)
    sens = 100.0 * tp / max(len(reales), 1)
    prec = 100.0 * tp / max(len(llamadas), 1)
    return sens, prec, tp, fp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--largo", type=int, default=20000, help="tamaño del genoma (20000)")
    ap.add_argument("--mutaciones", type=int, default=40)
    ap.add_argument("--reads", type=int, default=1500)
    ap.add_argument("--largo-read", type=int, default=250, dest="largo_read")
    ap.add_argument("--error", type=float, default=0.01)
    ap.add_argument("--semilla", type=int, default=2026)
    args = ap.parse_args()

    comprobar_entorno()

    base = os.path.join(os.environ.get("TEMP", "."), "bf_vs_bcftools")
    os.makedirs(base, exist_ok=True)

    print("=" * 82)
    print("BioForge vs bcftools — llamada de variantes")
    print("=" * 82)
    verdad = generar(base, args.largo, args.mutaciones, args.reads,
                     args.largo_read, args.error, args.semilla)
    print(f"genoma {args.largo:,} pb · {len(verdad)} sustituciones conocidas · "
          f"{args.reads:,} lecturas de {args.largo_read} pb · {args.error:.0%} de error\n")

    nombre, ref = leer_fasta(os.path.join(base, "ref.fa"))
    lecturas = leer_fastq(os.path.join(base, "reads.fq"))

    # ── el estándar: minimap2 → samtools → bcftools ──────────────────────────
    print("corriendo minimap2 + bcftools…", file=sys.stderr)
    t0 = time.perf_counter()
    # Se corre bcftools de DOS maneras, porque compararlo mal sería hacer trampa:
    #  (a) por defecto — diploide y SIN filtrar, que es como sale de la caja;
    #  (b) en igualdad — haploide (--ploidy 1) y filtrado con los MISMOS umbrales
    #      que aplicamos nosotros (QUAL>=20, profundidad>=5). Esta es la que vale.
    r = wsl("minimap2 -ax sr ref.fa reads.fq 2>/dev/null > aln.sam && "
            "samtools sort -o aln.bam aln.sam 2>/dev/null && "
            "samtools index aln.bam && "
            "bcftools mpileup -f ref.fa aln.bam -Ou 2>/dev/null | "
            "bcftools call -mv -Ov -o bcftools_defecto.vcf 2>/dev/null && "
            "bcftools mpileup -f ref.fa aln.bam -Ou 2>/dev/null | "
            "bcftools call -mv --ploidy 1 -Ou 2>/dev/null | "
            "bcftools filter -i 'QUAL>=20 && INFO/DP>=5' -Ov "
            "-o bcftools_justo.vcf 2>/dev/null && echo LISTO", base)
    t_std = time.perf_counter() - t0
    if "LISTO" not in r.stdout:
        print("fallo la tuberia estandar:", r.stdout[-400:], r.stderr[-400:],
              file=sys.stderr)
        raise SystemExit(1)
    v_bcf_def = leer_vcf(os.path.join(base, "bcftools_defecto.vcf"))
    v_bcf = leer_vcf(os.path.join(base, "bcftools_justo.vcf"))

    # ── nosotros, tubería completa ───────────────────────────────────────────
    print("corriendo BioForge (tubería completa)…", file=sys.stderr)
    t0 = time.perf_counter()
    ga = GenomeAligner(ref, name=nombre)
    pares = [(x, m[0]) for x in lecturas for m in [ga.map(x)] if m]
    pile = pileup(ref, pares)
    v_bf = {(v.pos, v.alt) for v in call_variants(pile, ref) if v.kind == "SNV"}
    t_bf = time.perf_counter() - t0

    # ── nosotros, sobre los MISMOS alineamientos de minimap2 (contraste justo) ─
    print("corriendo BioForge sobre los alineamientos de minimap2…", file=sys.stderr)
    pares_sam = parsear_sam(os.path.join(base, "aln.sam"))
    t0 = time.perf_counter()
    pile2 = pileup(ref, pares_sam)
    v_bf2 = {(v.pos, v.alt) for v in call_variants(pile2, ref) if v.kind == "SNV"}
    t_bf2 = time.perf_counter() - t0

    # ── resultados ───────────────────────────────────────────────────────────
    print(f"\n{'':34} {'sensib.':>9} {'precisión':>10} {'ciertas':>8} "
          f"{'falsas':>7} {'tiempo':>9}")
    print("-" * 82)
    for etiqueta, llamadas, t in (
            ("bcftools — por defecto (diploide)", v_bcf_def, t_std),
            ("bcftools — en igualdad (haploide+filtro)", v_bcf, t_std),
            ("BioForge (tubería propia)", v_bf, t_bf),
            ("BioForge (alineamientos de minimap2)", v_bf2, t_bf2)):
        s, p, tp, fp = evaluar(llamadas, verdad)
        print(f"{etiqueta:34} {s:>8.1f}% {p:>9.1f}% {tp:>8} {fp:>7} {t:>8.2f}s")

    print(f"\nlecturas apiladas: BioForge propia {pile.n_reads:,} · "
          f"desde el SAM de minimap2 {pile2.n_reads:,}")

    print("\nCONTRASTE JUSTO (mismos alineamientos, solo cambia el llamador):")
    comunes = v_bf2 & v_bcf
    union = v_bf2 | v_bcf
    print(f"  concordancia: {100.0*len(comunes)/max(len(union),1):.1f}%  "
          f"({len(comunes)} en común de {len(union)})")
    solo_bf = sorted(v_bf2 - v_bcf)[:5]
    solo_bcf = sorted(v_bcf - v_bf2)[:5]
    if solo_bf:
        ciertas = sum(1 for x in (v_bf2 - v_bcf) if x in set(verdad.items()))
        print(f"  solo BioForge: {len(v_bf2 - v_bcf)} (de ellas {ciertas} reales) "
              f"p.ej. {solo_bf}")
    if solo_bcf:
        ciertas = sum(1 for x in (v_bcf - v_bf2) if x in set(verdad.items()))
        print(f"  solo bcftools: {len(v_bcf - v_bf2)} (de ellas {ciertas} reales) "
              f"p.ej. {solo_bcf}")

    print("\nLectura honesta:")
    print("  · bcftools 'por defecto' llama en DIPLOIDE y sin filtrar; compararnos")
    print("    contra esa fila sería hacer trampa, porque nuestra salida ya viene")
    print("    filtrada. La fila honesta es 'en igualdad' (haploide + mismos umbrales).")
    print("  · La comparación limpia es 'en igualdad' vs la ÚLTIMA fila: mismos")
    print("    alineamientos y mismos umbrales, solo cambia la estadística de llamada.")
    print("  · bcftools trae 20 años de casos raros resueltos (sesgo de hebra,")
    print("    realineamiento de indels, calidades por base, modelo diploide).")
    print("    Aquí solo se comparan SUSTITUCIONES en muestra haploide, que es")
    print("    donde nuestro llamador está pensado para competir.")
    print("  · Los indels quedan fuera de la tabla a propósito: nuestro alineador")
    print("    usa hueco lineal y los parte (ver variants/caller.py).")


if __name__ == "__main__":
    main()
