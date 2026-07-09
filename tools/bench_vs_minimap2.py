#!/usr/bin/env python3
"""
tools/bench_vs_minimap2.py — duelo honesto BioForge vs minimap2.

Mismo genoma, mismos reads, misma máquina. Mide throughput de MAPEO (reads/s,
Mb/s) de ambos, en 1 hilo y en todos los núcleos, aislando la construcción del
índice del mapeo en sí. Pensado para correr en Linux/WSL, donde minimap2 existe
de verdad (`apt install minimap2`).

Uso:
    python3 tools/bench_vs_minimap2.py [--genome 4_800_000] [--reads 2000]
        [--read-len 2000] [--error 0.05] [--threads 4] [--seed 1]

Honestidad: los reads se simulan con error (sustituciones + indels) para no dar
a nadie el caso trivial. Se reporta cuántos reads mapea cada uno (sanity), no
solo la velocidad — mapear rápido pero peor no es ganar.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bioforge.genomemap import GenomeAligner  # noqa: E402


def _rng_genome(n: int, rng) -> str:
    return "".join("ACGT"[i] for i in rng.integers(0, 4, n))


def _simulate_reads(genome: str, n: int, read_len: int, error: float, rng):
    """Reads de posiciones aleatorias, ambas hebras, con sustituciones+indels."""
    comp = str.maketrans("ACGT", "TGCA")
    reads, truth = [], []
    L = len(genome)
    for _ in range(n):
        rl = int(rng.integers(read_len // 2, read_len + 1))
        o = int(rng.integers(0, max(1, L - rl)))
        s = list(genome[o:o + rl])
        # errores: por base, prob error → sub / ins / del
        out = []
        for ch in s:
            r = rng.random()
            if r > error:
                out.append(ch)
            elif r < error * 0.7:                       # sustitución
                out.append("ACGT"[("ACGT".index(ch) + 1 + rng.integers(0, 3)) % 4])
            elif r < error * 0.85:                      # inserción
                out.append(ch); out.append("ACGT"[rng.integers(0, 4)])
            # else: deleción (omitir la base)
        read = "".join(out)
        if rng.integers(0, 2):                           # hebra inversa
            read = read.translate(comp)[::-1]
        reads.append(read); truth.append(o)
    return reads, truth


def _write_fasta(path: Path, name: str, seq: str) -> None:
    with open(path, "w") as fh:
        fh.write(f">{name}\n")
        for i in range(0, len(seq), 70):
            fh.write(seq[i:i + 70] + "\n")


def _write_fastq(path: Path, reads) -> None:
    with open(path, "w") as fh:
        for i, r in enumerate(reads):
            fh.write(f"@r{i}\n{r}\n+\n{'I' * len(r)}\n")


def _run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", type=int, default=4_800_000)
    ap.add_argument("--reads", type=int, default=2000)
    ap.add_argument("--read-len", type=int, default=2000)
    ap.add_argument("--error", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=0)      # 0 = todos
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    mm2 = shutil.which("minimap2")
    if not mm2:
        sys.exit("minimap2 no encontrado. En WSL/Linux: sudo apt install minimap2")

    import os
    nthreads = args.threads or (os.cpu_count() or 1)
    rng = np.random.default_rng(args.seed)

    print(f"Generando genoma ({args.genome/1e6:.1f} Mb) y {args.reads} reads "
          f"(~{args.read_len} bp, error {args.error:.0%})…")
    genome = _rng_genome(args.genome, rng)
    reads, _ = _simulate_reads(genome, args.reads, args.read_len, args.error, rng)
    total_mb = sum(len(r) for r in reads) / 1e6

    tmp = Path(tempfile.mkdtemp(prefix="bfbench_"))
    ref_fa = tmp / "ref.fa"; reads_fq = tmp / "reads.fq"; mmi = tmp / "ref.mmi"
    _write_fasta(ref_fa, "chr1", genome)
    _write_fastq(reads_fq, reads)

    print("\n=== minimap2 ===")
    _run([mm2, "-x", "map-ont", "-d", str(mmi), str(ref_fa)])       # índice aparte
    for t in (1, nthreads):
        t0 = time.perf_counter()
        res = _run([mm2, "-x", "map-ont", "-t", str(t), "-a",
                    str(mmi), str(reads_fq)])
        dt = time.perf_counter() - t0
        mapped = sum(1 for ln in res.stdout.splitlines()
                     if ln and not ln.startswith("@")
                     and ln.split("\t")[1] != "4")            # flag 4 = unmapped
        print(f"  -t {t:<2}: {dt:6.3f} s  {len(reads)/dt:8.0f} reads/s  "
              f"{total_mb/dt:6.2f} Mb/s   (mapeados ~{mapped})")

    print("\n=== BioForge ===")
    t0 = time.perf_counter()
    ga = GenomeAligner(genome, k=15, w=10, max_occ=50)
    print(f"  índice: {time.perf_counter()-t0:.2f} s")
    for t in (1, nthreads):
        np_ = 1 if t == 1 else 0
        t0 = time.perf_counter()
        out = ga.map_batch(reads, n_processes=np_)
        dt = time.perf_counter() - t0
        mapped = sum(1 for m in out if m)
        print(f"  {('1 hilo' if t==1 else str(nthreads)+' nucleos'):<9}: "
              f"{dt:6.3f} s  {len(reads)/dt:8.0f} reads/s  "
              f"{total_mb/dt:6.2f} Mb/s   (mapeados {mapped})")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\nNota: throughput de mapeo (índice aparte). Reads simulados con "
          "error; 'mapeados' es señal de calidad, no solo velocidad.")


if __name__ == "__main__":
    main()
