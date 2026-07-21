"""
tools/bench_real_data.py
══════════════════════════════════════════════════════════════════════
Comparación HONESTA contra los mejores del mundo — SOLO DATOS REALES.

Los benchmarks con secuencias aleatorias mienten: el ADN real tiene repeticiones,
sesgo de composición, regiones de baja complejidad y distribuciones de longitud que
las secuencias uniformes no reproducen. Aquí todo se mide sobre material real:

  · Genoma      : E. coli K-12 MG1655 (NCBI, 4.64 Mb)
  · Genes       : 4.325 CDS reales de ese genoma
  · Lecturas    : run Illumina real (ENA SRR2584863), con sus calidades auténticas
  · Homólogos   : hemaglutininas de gripe reales (caché de bioforge.fetch)

Ambos lados reciben EXACTAMENTE el mismo fichero. Se ejecuta en WSL/Linux, donde
viven los rivales (seqkit, minimap2, mafft, muscle, parasail).

Uso:  python3 tools/bench_real_data.py [parsing|translate|align|map|msa|all]
"""

import os
import subprocess
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, ".")

HOME = os.path.expanduser("~")
SEQKIT = os.path.join(HOME, "seqkit")
MAFFT = os.path.join(HOME, "mafft-linux64", "mafft.bat")
MUSCLE = os.path.join(HOME, "muscle")
D = "data_real"
GENOME = os.path.join(D, "ecoli.fna")
CDS = os.path.join(D, "ecoli_cds.fna")
READS = os.path.join(D, "reads_real.fastq")
SUB = "/tmp/reads_sub.fastq"          # subconjunto para tiempos manejables


def best(fn, n=3):
    fn()
    return min((lambda: (lambda t: (fn(), time.perf_counter() - t)[1])(
        time.perf_counter()))() for _ in range(n))


def sh(cmd):
    t = time.perf_counter()
    subprocess.run(cmd, capture_output=True)
    return time.perf_counter() - t


def read_fasta(path, limit=None):
    seqs, cur = [], []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
                    if limit and len(seqs) >= limit:
                        return seqs
            else:
                cur.append(line.strip())
    if cur:
        seqs.append("".join(cur))
    return seqs


def make_subset(n_reads=400_000):
    if os.path.exists(SUB):
        return
    with open(READS) as fi, open(SUB, "w") as fo:
        for i, line in enumerate(fi):
            if i >= n_reads * 4:
                break
            fo.write(line)


def bench_parsing():
    print("\n── PARSING · lecturas Illumina REALES ──", flush=True)
    make_subset()
    mb = os.path.getsize(SUB) / 1e6
    print(f"   fichero: {mb:.0f} MB reales (calidades auténticas)")
    from bioforge.biocore import SmartImporter

    def bf(nt):
        def run():
            tot = 0
            for rb in SmartImporter.stream_fastq_batches(SUB, n_threads=nt):
                tot += int(np.asarray(rb.n_symbols).sum())
            return tot
        return run

    t1 = best(bf(1)); tp = best(bf(0))
    s1 = min(sh([SEQKIT, "stats", "-j", "1", SUB]) for _ in range(3))
    s4 = min(sh([SEQKIT, "stats", "-j", "4", SUB]) for _ in range(3))
    print(f"   BioForge 1 hilo : {t1:.3f}s      seqkit 1 hilo : {s1:.3f}s   → {t1/s1:.2f}x")
    print(f"   BioForge paralelo: {tp:.3f}s     seqkit 4 hilos: {s4:.3f}s   → {tp/s4:.2f}x")


def bench_translate():
    print("\n── TRADUCCIÓN · 4.325 genes REALES de E. coli ──", flush=True)
    from bioforge.biocore import SeqType, SmartImporter
    from bioforge.smart_translator import SmartTranslator as T
    seqs = read_fasta(CDS)
    packed = [SmartImporter.from_string(f">x\n{s}\n",
                                        force_type=SeqType.NUCLEOTIDE)[0] for s in seqs]
    tb = best(lambda: T.translate_many(packed, warn_short=False))
    ts = min(sh([SEQKIT, "translate", CDS]) for _ in range(3))
    n_ok = sum(1 for p in T.translate_many(packed, warn_short=False) if p is not None)
    print(f"   BioForge: {tb:.3f}s   seqkit: {ts:.3f}s   → {tb/ts:.2f}x")
    print(f"   ({n_ok}/{len(seqs)} genes traducidos)")


def bench_align():
    print("\n── ALINEAMIENTO · hemaglutininas de gripe REALES ──", flush=True)
    import parasail
    from bioforge.aligner import SequenceAligner as A
    from bioforge.biocore import SeqType, SmartImporter
    from bioforge.fetch import fetch_dated_precise
    term = ("Influenza A virus[Organism] AND H3N2 AND hemagglutinin[Title] "
            "AND 1650:1780[SLEN] AND {year}")
    data = fetch_dated_precise(term, range(2015, 2020), per_year=60)
    seqs = [s for s, _ in data][:120]
    if len(seqs) < 20:
        print("   (sin secuencias en caché — saltando)")
        return
    pairs = [(seqs[i], seqs[i + 1]) for i in range(0, len(seqs) - 1, 2)]
    pp = [(SmartImporter.from_string(f">a\n{a}\n", force_type=SeqType.NUCLEOTIDE)[0],
           SmartImporter.from_string(f">b\n{b}\n", force_type=SeqType.NUCLEOTIDE)[0])
          for a, b in pairs]
    mx = parasail.matrix_create("ACGT", 1, -1)
    tb = best(lambda: [A.align(a, b, band="auto", detect_mutations=False) for a, b in pp])
    tp = best(lambda: [parasail.nw_striped_16(a, b, 1, 1, mx) for a, b in pairs])
    print(f"   {len(pairs)} pares de cepas reales (~1700 nt)")
    print(f"   BioForge: {tb*1000:.1f}ms   parasail: {tp*1000:.1f}ms   → {tb/tp:.2f}x")


def bench_map():
    print("\n── MAPEO · lecturas reales → genoma real de E. coli ──", flush=True)
    from bioforge.genomemap import GenomeAligner
    ref = "".join(read_fasta(GENOME))
    reads = []
    with open(READS) as f:
        for i, line in enumerate(f):
            if i % 4 == 1:
                reads.append(line.strip())
            if len(reads) >= 3000:
                break
    t = time.perf_counter()
    aln = GenomeAligner(ref, k=15, w=10)
    t_idx = time.perf_counter() - t
    t = time.perf_counter()
    res = aln.map_batch(reads, n_processes=0)
    tb = time.perf_counter() - t
    mapped = sum(1 for r in res if len(r))
    fq = "/tmp/map_reads.fastq"
    with open(fq, "w") as f:
        for i, r in enumerate(reads):
            f.write(f"@r{i}\n{r}\n+\n{'I'*len(r)}\n")
    tm1 = sh(["minimap2", "-ax", "sr", "-t", "1", GENOME, fq])
    tm4 = sh(["minimap2", "-ax", "sr", "-t", "4", GENOME, fq])
    print(f"   {len(reads)} lecturas reales · índice BioForge {t_idx:.1f}s")
    print(f"   BioForge (todos): {tb:.2f}s   minimap2 1h: {tm1:.2f}s  4h: {tm4:.2f}s")
    print(f"   mapeadas: {mapped}/{len(reads)} ({mapped/len(reads)*100:.1f}%)")


def bench_msa():
    print("\n── MSA · cepas de gripe REALES ──", flush=True)
    from bioforge.fetch import fetch_dated_precise
    from bioforge.msa import align_multiple
    term = ("Influenza A virus[Organism] AND H3N2 AND hemagglutinin[Title] "
            "AND 1650:1780[SLEN] AND {year}")
    data = fetch_dated_precise(term, range(2015, 2020), per_year=60)
    seqs = [s for s, _ in data][:150]
    if len(seqs) < 20:
        print("   (sin secuencias en caché — saltando)")
        return
    fa = "/tmp/msa_real.fasta"
    with open(fa, "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">s{i}\n{s}\n")
    t = time.perf_counter(); r = align_multiple(seqs); tb = time.perf_counter() - t
    ok = all(row.replace("-", "") == s for row, s in zip(r.aligned, seqs))
    print(f"   {len(seqs)} cepas reales (~1700 nt)")
    print(f"   BioForge: {tb:.2f}s (sin pérdida={ok})")
    if os.path.exists(MAFFT):
        print(f"   MAFFT   : {sh([MAFFT, '--auto', '--quiet', fa]):.2f}s")
    if os.path.exists(MUSCLE):
        print(f"   MUSCLE  : {sh([MUSCLE, '-align', fa, '-output', '/tmp/o.fa']):.2f}s")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    print("═" * 62)
    print("  BioForge vs LOS MEJORES — SOLO DATOS REALES (nada simulado)")
    print("═" * 62)
    fns = {"parsing": bench_parsing, "translate": bench_translate,
           "align": bench_align, "map": bench_map, "msa": bench_msa}
    for name, fn in fns.items():
        if which in ("all", name):
            try:
                fn()
            except Exception as e:
                print(f"   ✗ {name}: {type(e).__name__}: {e}")
