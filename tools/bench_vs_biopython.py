"""
tools/bench_vs_biopython.py — BioForge vs Biopython: tiempo y RAM.

Compara la ingesta y el control de calidad (QC) de FASTQ entre BioForge y
Biopython. Cada herramienta se ejecuta en su PROPIO subproceso para medir el
pico de memoria de forma justa (Biopython retiene todos los objetos; BioForge
trabaja en columnas).

Uso:
    python tools/bench_vs_biopython.py
    python tools/bench_vs_biopython.py --reads 500000 --length 150
    python tools/bench_vs_biopython.py --gz          # también prueba .gz

Requiere: biopython, psutil  (pip install biopython psutil)
"""

import argparse
import gzip
import os
import random
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Medición de pico de RAM (multiplataforma) ───────────────────────────────
def _peak_rss_bytes() -> int:
    try:
        import psutil
        mi = psutil.Process().memory_info()
        # En Windows existe peak_wset (pico real del working set).
        return int(getattr(mi, "peak_wset", mi.rss))
    except Exception:
        try:
            import resource
            kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux: KB · macOS: bytes
            return kb * 1024 if sys.platform != "darwin" else kb
        except Exception:
            return 0


# ════════════════════════════════════════════════════════════════════════════
# Workloads (se ejecutan dentro del subproceso hijo)
# ════════════════════════════════════════════════════════════════════════════
def _task_bioforge_parse(path: str) -> str:
    from bioforge import SmartImporter
    bases = 0
    for rec in SmartImporter.stream_fastq(path):
        bases += rec.sequence.n_symbols
    return f"{bases} bases"


def _task_bioforge_qc(path: str) -> str:
    from bioforge import SmartImporter
    total = passed = 0
    for batch in SmartImporter.stream_fastq_batches(path):
        total += len(batch)
        passed += int(batch.passes(20).sum())
    return f"{passed}/{total} pasan"


def _task_bioforge_load(path: str) -> str:
    # Mantiene TODAS las lecturas en RAM (5-bit empaquetado + calidades 2-D).
    from bioforge import SmartImporter
    batches = list(SmartImporter.stream_fastq_batches(path))
    n = sum(len(b) for b in batches)
    return f"{n} lecturas en RAM"


def _task_biopython_load(path: str) -> str:
    # Mantiene TODOS los SeqRecord en RAM (strings + listas de calidad).
    from Bio import SeqIO
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        records = list(SeqIO.parse(fh, "fastq"))
    return f"{len(records)} lecturas en RAM"


def _task_biopython_parse(path: str) -> str:
    from Bio import SeqIO
    opener = gzip.open if path.endswith(".gz") else open
    bases = 0
    with opener(path, "rt") as fh:
        for rec in SeqIO.parse(fh, "fastq"):
            bases += len(rec.seq)
    return f"{bases} bases"


def _task_biopython_qc(path: str) -> str:
    from Bio import SeqIO
    opener = gzip.open if path.endswith(".gz") else open
    total = passed = 0
    with opener(path, "rt") as fh:
        for rec in SeqIO.parse(fh, "fastq"):
            q = rec.letter_annotations["phred_quality"]
            if q and (sum(q) / len(q)) >= 20:
                passed += 1
            total += 1
    return f"{passed}/{total} pasan"


_TASKS = {
    "bioforge_parse":  _task_bioforge_parse,
    "bioforge_qc":     _task_bioforge_qc,
    "bioforge_load":   _task_bioforge_load,
    "biopython_parse": _task_biopython_parse,
    "biopython_qc":    _task_biopython_qc,
    "biopython_load":  _task_biopython_load,
}


def _run_child(task: str, path: str) -> None:
    """Ejecuta un workload y emite 'RESULT <segundos> <peak_bytes> <info>'."""
    fn = _TASKS[task]
    t0 = time.perf_counter()
    info = fn(path)
    dt = time.perf_counter() - t0
    peak = _peak_rss_bytes()
    print(f"RESULT {dt:.6f} {peak} {info}")


# ════════════════════════════════════════════════════════════════════════════
# Orquestación (proceso padre)
# ════════════════════════════════════════════════════════════════════════════
def _spawn(task: str, path: str) -> "tuple[float, int, str]":
    cmd = [sys.executable, __file__, "--child", task, "--file", path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    line = ""
    for ln in res.stdout.splitlines():
        if ln.startswith("RESULT "):
            line = ln
    if not line:
        sys.stderr.write(res.stdout + res.stderr)
        raise RuntimeError(f"El subproceso '{task}' no devolvió RESULT")
    _, secs, peak, *rest = line.split(maxsplit=3)
    return float(secs), int(peak), (rest[0] if rest else "")


def _make_fastq(path: str, n_reads: int, length: int, gz: bool) -> None:
    rng = random.Random(2024)
    bases = "ACGT"
    qchars = "".join(chr(33 + q) for q in range(40))
    seq = "".join(rng.choice(bases) for _ in range(length))
    opener = (lambda p: gzip.open(p, "wt", newline="\n")) if gz else \
             (lambda p: open(p, "w", newline="\n"))
    with opener(path) as f:
        for i in range(n_reads):
            q = "".join(qchars[rng.randint(0, 39)] for _ in range(length))
            f.write(f"@read{i}\n{seq}\n+\n{q}\n")


def _fmt_mb(b: int) -> str:
    return f"{b / 1e6:7.1f} MB"


def _row(label, secs, peak):
    print(f"  {label:<22} {secs*1000:9.0f} ms   {_fmt_mb(peak)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="BioForge vs Biopython")
    ap.add_argument("--child", choices=list(_TASKS))
    ap.add_argument("--file")
    ap.add_argument("--reads", type=int, default=300_000)
    ap.add_argument("--length", type=int, default=150)
    ap.add_argument("--gz", action="store_true")
    args = ap.parse_args()

    if args.child:
        _run_child(args.child, args.file)
        return 0

    # ¿Está Biopython disponible?
    try:
        import Bio  # noqa: F401
        have_bio = True
    except ImportError:
        have_bio = False
        print("[aviso] Biopython no está instalado — solo se medirá BioForge.")
        print("        Instálalo con:  pip install biopython psutil\n")

    from bioforge.engine._loader import C_AVAILABLE, C_BATCH_AVAILABLE, C_PARSER_AVAILABLE
    print(f"Motor C: disponible={C_AVAILABLE}  parser={C_PARSER_AVAILABLE}  "
          f"batch={C_BATCH_AVAILABLE}")

    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="bioforge_bench_")
    suffix = ".fastq.gz" if args.gz else ".fastq"
    path = os.path.join(tmpdir, "reads" + suffix)

    mb = args.reads * args.length / 1e6
    print(f"\nGenerando FASTQ: {args.reads:,} lecturas × {args.length} bp "
          f"= {mb:.0f} M bases{' (.gz)' if args.gz else ''} ...")
    _make_fastq(path, args.reads, args.length, args.gz)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Tamaño en disco: {size_mb:.1f} MB\n")

    print("═" * 60)
    print(f"  {'TAREA':<22} {'TIEMPO':>9}      {'RAM PICO':>10}")
    print("═" * 60)

    print("\n[ Parsing: leer todas las secuencias ]")
    bf_t, bf_m, bf_i = _spawn("bioforge_parse", path)
    _row("BioForge", bf_t, bf_m)
    if have_bio:
        bp_t, bp_m, bp_i = _spawn("biopython_parse", path)
        _row("Biopython", bp_t, bp_m)
        print(f"  → BioForge es {bp_t/bf_t:5.1f}× más rápido, "
              f"{bp_m/max(bf_m,1):4.1f}× menos RAM")

    print("\n[ QC: filtrar lecturas con calidad media ≥ 20 ]")
    bf_t, bf_m, bf_i = _spawn("bioforge_qc", path)
    _row("BioForge (columnar)", bf_t, bf_m)
    if have_bio:
        bp_t, bp_m, bp_i = _spawn("biopython_qc", path)
        _row("Biopython", bp_t, bp_m)
        print(f"  → BioForge es {bp_t/bf_t:5.1f}× más rápido, "
              f"{bp_m/max(bf_m,1):4.1f}× menos RAM")
        # sanity: mismos resultados
        ok = bf_i == bp_i
        print(f"  resultado idéntico: {'OK' if ok else f'{bf_i!r} vs {bp_i!r}'}")

    print("\n[ Cargar TODO en RAM (aquí pesa el almacenamiento 5-bit) ]")
    bf_t, bf_m, bf_i = _spawn("bioforge_load", path)
    _row("BioForge (5-bit)", bf_t, bf_m)
    if have_bio:
        bp_t, bp_m, bp_i = _spawn("biopython_load", path)
        _row("Biopython", bp_t, bp_m)
        print(f"  → BioForge usa {bp_m/max(bf_m,1):4.1f}× menos RAM, "
              f"{bp_t/bf_t:5.1f}× más rápido")

    os.remove(path)
    os.rmdir(tmpdir)
    print("\nListo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
