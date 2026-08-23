#!/usr/bin/env python3
"""
tools/accuracy_vs_minimap2.py — PRECISIÓN de BioForge vs minimap2 en un genoma REAL.

No mide velocidad (eso es bench_vs_minimap2.py), mide **corrección**: simula reads
desde una referencia real GUARDANDO la posición verdadera de cada uno, los mapea
con ambas herramientas, y cuenta qué fracción cae en la posición correcta (dentro
de una tolerancia). Los genomas reales tienen repeticiones — el caso difícil.
Responde a "¿es correcto, no solo rápido?".

Uso (Linux/WSL, con minimap2 instalado):
    python3 tools/accuracy_vs_minimap2.py --ref _benchdata/ecoli.fa \
        [--reads 5000] [--read-len 2000] [--error 0.05] [--tol 50] [--seed 1]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bioforge.mapping.genomemap import GenomeAligner  # noqa: E402

_VALID = set("ACGT")


def load_fasta(path: str) -> str:
    parts = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith(">"):
                parts.append(line.strip())
    return "".join(parts).upper()


def simulate(genome: str, n: int, read_len: int, error: float, rng):
    """Reads con error (subs+indels), ambas hebras. truth[i] = (start0based, strand)."""
    comp = str.maketrans("ACGT", "TGCA")
    L = len(genome)
    reads, truth = [], []
    for _ in range(n):
        rl = int(rng.integers(read_len // 2, read_len + 1))
        o = int(rng.integers(0, L - rl))
        out = []
        for ch in genome[o:o + rl]:
            if ch not in _VALID:
                out.append(ch); continue
            r = rng.random()
            if r > error:
                out.append(ch)
            elif r < error * 0.7:
                out.append("ACGT"[("ACGT".index(ch) + 1 + int(rng.integers(0, 3))) % 4])
            elif r < error * 0.85:
                out.append(ch); out.append("ACGT"[int(rng.integers(0, 4))])
            # else: deleción
        read = "".join(out)
        strand = int(rng.integers(0, 2))
        if strand:
            read = read.translate(comp)[::-1]
        reads.append(read)
        truth.append((o, strand))
    return reads, truth


def write_fastq(path: Path, reads):
    with open(path, "w") as fh:
        for i, r in enumerate(reads):
            fh.write(f"@r{i}\n{r}\n+\n{'I' * len(r)}\n")


def minimap2_positions(mm2, ref_fa, reads_fq, n):
    """Devuelve pos0[i] = posición forward (0-based) del mapeo PRIMARIO, o None."""
    res = subprocess.run([mm2, "-ax", "map-ont", "-t", "4", str(ref_fa), str(reads_fq)],
                         check=True, capture_output=True, text=True)
    pos = [None] * n
    for ln in res.stdout.splitlines():
        if not ln or ln.startswith("@"):
            continue
        f = ln.split("\t")
        qname, flag = f[0], int(f[1])
        if flag & 4:               # sin mapear
            continue
        if flag & 0x100 or flag & 0x800:   # secundario / suplementario → saltar
            continue
        try:
            i = int(qname[1:])
        except ValueError:
            continue
        pos[i] = int(f[3]) - 1     # SAM POS es 1-based
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--reads", type=int, default=5000)
    ap.add_argument("--read-len", type=int, default=2000)
    ap.add_argument("--error", type=float, default=0.05)
    ap.add_argument("--tol", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    mm2 = shutil.which("minimap2")
    if not mm2:
        sys.exit("minimap2 no encontrado (sudo apt install minimap2)")

    rng = np.random.default_rng(args.seed)
    genome = load_fasta(args.ref)
    print(f"Referencia: {len(genome)/1e6:.2f} Mb · {args.reads} reads "
          f"(~{args.read_len} bp, error {args.error:.0%}, tolerancia ±{args.tol} bp)")

    reads, truth = simulate(genome, args.reads, args.read_len, args.error, rng)
    tol = args.tol

    # ── minimap2 ──
    tmp = Path(tempfile.mkdtemp(prefix="acc_"))
    reads_fq = tmp / "reads.fq"
    write_fastq(reads_fq, reads)
    mm_pos = minimap2_positions(mm2, args.ref, reads_fq, args.reads)

    # ── BioForge ──
    ga = GenomeAligner(genome, k=15, w=10, max_occ=50)
    bf = ga.map_batch(reads, n_processes=0)
    bf_pos = [mps[0].target_start if mps else None for mps in bf]

    # ── métricas ──
    def stats(pos):
        mapped = sum(1 for p in pos if p is not None)
        correct = sum(1 for p, (o, _) in zip(pos, truth)
                      if p is not None and abs(p - o) <= tol)
        return mapped, correct

    mm_mapped, mm_correct = stats(mm_pos)
    bf_mapped, bf_correct = stats(bf_pos)
    N = args.reads

    # concordancia: ambos mapean y a menos de tol el uno del otro
    concord = sum(1 for a, b in zip(mm_pos, bf_pos)
                  if a is not None and b is not None and abs(a - b) <= tol)
    both_mapped = sum(1 for a, b in zip(mm_pos, bf_pos)
                      if a is not None and b is not None)

    print(f"\n{'':10} | {'mapeados':>16} | {'en pos. correcta':>18}")
    print(f"{'minimap2':10} | {mm_mapped}/{N} ({mm_mapped/N:6.1%}) | "
          f"{mm_correct}/{N} ({mm_correct/N:6.1%})")
    print(f"{'BioForge':10} | {bf_mapped}/{N} ({bf_mapped/N:6.1%}) | "
          f"{bf_correct}/{N} ({bf_correct/N:6.1%})")
    print(f"\nConcordancia (ambos mapean a la misma posición ±{tol}): "
          f"{concord}/{both_mapped} ({concord/both_mapped:.1%} de los que ambos mapean)")
    print("\nNota: 'posición correcta' = a ±tol de la posición real de simulación.")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
