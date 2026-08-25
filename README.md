# BioForge — High-Performance Bioinformatics Engine

[![Tests](https://github.com/erlanders177/bioforge/actions/workflows/tests.yml/badge.svg)](https://github.com/erlanders177/bioforge/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-PolyForm_NC_1.0-blue)](LICENSE)

A bioinformatics engine built for **Edge Computing**.  
No Biopython. No heavy dependencies. NumPy core + optional C engine for maximum speed.

---

## Why this exists

Most bioinformatics tools are built for servers with gigabytes of RAM.  
BioForge was built for the opposite: low-power hardware, minimal footprint,
maximum speed — running genetic analysis **offline and locally**.

Two core rules:
- **Zero Python loops** in the hot path — every operation is vectorised with NumPy.
- **5-bit encoding** — every biological symbol fits in 5 bits, saving 37.5% memory vs ASCII.

---

## What's in the box

BioForge is **one lightweight engine bundling several tools** — a single `pip
install`, a single dependency (NumPy), a shared C backend. Each tool has its own
section (with examples) further down.

| Category | Tools |
|----------|-------|
| **Storage & I/O** | 5-bit sequence storage · streaming FASTA/FASTQ parser (C) · `.gz` / BGZF |
| **Sequence transforms** | DNA→protein translation · reverse complement · 6-frame translation |
| **Alignment** | pairwise (Needleman–Wunsch / banded / Smith–Waterman) · **multiple sequence alignment** (center-star) |
| **Genome mapping** | long-read seed-chain-align mapper, whole pipeline in C, PAF output — *on par with minimap2 on multi-core, ~99.8% accurate* |
| **Analysis & QC** | FastQC-style quality report · GC content · k-mer spectrum |
| **Phylogenetics** *(v10.2)* | distance matrices (p / Jukes-Cantor / Kimura-2P / Poisson) · Neighbor-Joining · UPGMA · WPGMA · Newick output · **bootstrap support** — *NJ topology identical to Biopython, 15× faster* |
| **Variant calling** *(v10.2)* | pileup (depth & coverage) · SNV/indel calling by binomial likelihood ratio · VCF 4.2 output — closes the pipeline: reads → mapping → **variants** |
| **Evolution** *(v7.0)* | mutation ranking · stable lineage designation (Pango/autolin-style, no tree) · honest backtesting — `bioforge-evolution` |
| **Evaluation & reality-check** *(v8.0)* | `EvolutionBenchmark` — judge any evolution predictor honestly (trivial-baseline bar, novel-regime split, bootstrap CI, pretraining-leakage detector) · `RealityCheck` — filter another tool's mutation hits by real-world traction |
| **Nanopore basecalling** *(v9.0–9.1)* | raw electrical signal → bases, from scratch (POD5/FAST5 readers · event detection · own pore-model estimation · Viterbi with stay/skip · iterative rescaling). Pure NumPy, no AI — **~74% on real R9.4 signal** |
| **Desktop app** *(v10.0)* | the whole engine behind a friendly local window — 7 tabs for non-coders. Double-click `.exe` or `pip install "bioforge[app]" && bioforge-app`. Offline, private (*DNA Edge*) |

Why one engine instead of a pile of separate tools? **Fewer resources and less
friction** — no piping data between programs, no format conversions, one install
that runs on low-power/edge hardware. Competitive at each task, and unique in
combining them (especially the evolution front).

---

## Key numbers

| Operation | Result |
|-----------|--------|
| Memory (30M bases) | **18.75 MB** (37.5% less than plain ASCII) |
| Translation throughput | **~5 M amino acids / second** (NumPy) · **~27× faster** with C engine |
| Bulk translation — `translate_many()` | **~5.5× faster** than one-by-one on 20 K sequences (columnar: one C crossing per batch for unpack, translate and pack; ATG/STOP resolved vectorised) |
| NW alignment 1000×1000 nt | **~165 ms** (NumPy) · **~29× faster** with C engine |
| NW with `band="auto"` (adaptive, **exact**) | **~1.3× of parasail** (the SIMD specialist) on 1000 nt — adaptive band + AVX2 banded kernel; widens until the optimal path no longer touches the edge, so the result is provably identical to full NW |
| MSA 200 × 1000 nt (center-star) | **0.27 s** — 5.3× faster than before the SIMD/adaptive-band work |
| Genome mapping — speed vs minimap2 | **on par on multi-core**, ~1.18× behind single-thread (*E. coli* scale, `minimap2 -a`; `tools/bench_vs_minimap2.py`) |
| Genome mapping — accuracy vs minimap2 | **~99.8% of reads mapped to the correct position** on real *E. coli*, matching minimap2 (`tools/accuracy_vs_minimap2.py`) |
| FASTA ingestion (C batch parser) | **~80 M bases / second** |
| FASTQ ingestion (C batch parser) | **~14 M bases / s · ~94 K reads / s** |
| QC filter 200 K reads (columnar) | **0.28 s** — **18.6× faster** than per-record |
| vs Biopython — QC filter | **~5–6× faster**, identical result |
| vs Biopython — load all in RAM | **~6.9× less memory** (115 MB vs 801 MB) · ~9.5× faster |
| Compressed input | **`.gz` read transparently in C** (zlib, static-linked) |
| FASTQ parsing — parallel (`n_threads=0`) | **~1.9× faster** than single-thread (500 K reads: 0.40 s → 0.21 s) — on par with `seqkit stats` single-threaded |
| Evolution — mutation ranking | **cross-virus AUC ~0.74–0.91** on flu HA, beats the best trivial axis on all 6 held-out tests (trained model runs in pure NumPy) |
| Evolution — judged honestly (Level 6) | on real H3N2: model **AUC 0.837 global / 0.631 on novel mutations**, over the trivial mutability bar (0.793) — *measured by our own `EvolutionBenchmark`* |
| Data integrity | **anti-corruption guard** refuses to mis-encode a mistyped sequence · property-based invariants over both alphabets · `tools/integrity_check.py` certificate |
| Nanopore basecaller (Level 7) | **~74% identity on real R9.4 signal** (E. coli, n=36, vs production Guppy; v9.0 was 70%, v9.1 lifted it) — from-scratch classical Viterbi, pure NumPy, no AI. *Reproducible: `tools/bench_basecaller.py`* |
| Import cost *(v10.1)* | **`import bioforge` in 4.7 ms** — was 75 ms (**16× faster**). The package loads **lazily**: 1 submodule instead of 15, so translating DNA never loads the basecaller, mapper or evolution code |
| Desktop app memory *(v10.1)* | **flat RAM with many files open** — only the ACTIVE file is materialised; the others keep just their summary and are re-read on demand (20 files: ~0.2 MB, previously growing linearly) |
| Phylogenetics — vs Biopython (the reference) | **NJ topology identical in 5/5 cases** (6–60 taxa) · distance matrices agree to **1.5e-8** (machine precision) · **15× faster** distances, **3.8× faster** NJ. *`tools/bench_vs_biopython_phylo.py`* |
| Phylogenetics — a finding | Biopython's `upgma()` averages `(d(k,i)+d(k,j))/2` **without weighting by cluster size** — that is **WPGMA, not UPGMA**. Ours weights (Sokal & Michener 1958); we also ship `wpgma()`, which reproduces Biopython's output in 5/5 cases |
| Phylogenetics — own proof | on an **additive** matrix NJ recovers the exact branch lengths (theoretical guarantee, in the test suite); bootstrap gives <70% support on random sequences — it does not invent genealogies |
| Variant calling — vs **bcftools** (the reference) | On the **same minimap2 alignments** and matched thresholds: **100% concordance** — the exact same 40 calls out of 40, 100% sensitivity and 100% precision for both. BioForge's caller takes **0.16 s vs 4.73 s** for the standard pipeline. *`tools/bench_vs_bcftools.py`* |
| Variant calling — what the comparison fixed | Connecting to real SAM exposed that our CIGAR reader silently ignored soft clips (`S`), which would misplace every base of a `bwa`/`minimap2` alignment. Now the **full SAM CIGAR alphabet** is supported. It also justified raising `min_alt_count` 2→3 (measured over 10 runs: removes every false positive, loses no true mutation) |
| Variant calling — SNV accuracy | **100% sensitivity and 100% precision from 10× coverage** (0.1–1% read error, 5 kb genome, 25 known SNVs). At 5× sensitivity drops to 64–72% but precision stays 100% — it prefers silence over invention. *`tools/bench_variants.py`* |
| Variant calling — noisy reads | at 5% error the default (`error_rate=0.01`) misfires; setting it to 0.05 lifts precision **71% → 100%** at 10× with no loss of sensitivity |
| Dependencies | **NumPy** (C engine + trained ranker included, pre-compiled) |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Level 10 — phylo/ (distance · tree)     Who descends from whom│
│  NJ · UPGMA · WPGMA · bootstrap support · Newick               │
├──────────────────────────────────────────────────────────────┤
│  Level 9 — variants/ (pileup · caller)   Reads → what changed │
│  binomial likelihood ratio · Phred QUAL · VCF 4.2 output      │
├──────────────────────────────────────────────────────────────┤
│  Level 7 — nanopore.py                   Signal → bases       │
│  from-scratch classical basecaller (Viterbi, pure NumPy)      │
├──────────────────────────────────────────────────────────────┤
│  Level 6 — evalkit · realitycheck        Honest judgment      │
│  judge a predictor · filter mutations by real-world traction  │
├──────────────────────────────────────────────────────────────┤
│  Level 5 — evolution · msa · fetch · ai  Evolution predictor  │
│  mutation ranking · stable lineages · trained MLP (pure NumPy)│
├──────────────────────────────────────────────────────────────┤
│  Level 4 — genomemap · minimizers · refindex   Genome mapper  │
│  seed-chain-align (minimap2-style) · whole pipeline in C      │
├──────────────────────────────────────────────────────────────┤
│  Level 3 — bioforge/aligner.py           NW alignment         │
│  Anti-diagonal wavefront O(m+n) · mutation detection         │
├──────────────────────────────────────────────────────────────┤
│  Level 2 — bioforge/smart_translator.py  DNA → Protein       │
│  CODON_LUT + sliding_window_view · first-ATG ORF detection   │
├──────────────────────────────────────────────────────────────┤
│  Level 1 — bioforge/biocore.py           5-bit storage        │
│  BitPacker · PackedSequence · SmartImporter · LUTs           │
├──────────────────────────────────────────────────────────────┤
│  C engine — bioforge/engine/engine.c     Optional backend     │
│  GCC -O3 -fopenmp · auto-loaded via ctypes · NumPy fallback  │
└──────────────────────────────────────────────────────────────┘
```

### The 5-bit unified alphabet

Every biological symbol — nucleotides, amino acids, gaps, stop codons and
ambiguous bases — fits in a single 5-bit scheme (32 states).  
One encoding covers DNA, RNA, and proteins in the same pipeline.

```
State  Symbol            State  Symbol
  0    Adenine   (A)      14    Methionine    (M)
  1    Cytosine  (C)      ...   (all 20 amino acids: 4–23)
  2    Guanine   (G)      24    STOP codon    (*)
  3    Thymine / Uracil   25    Alignment gap (-)
 4–23  Amino acids        31    Unknown / ambiguous
```

---

## Installation

```bash
pip install bioforge
```

Native wheels ship for **Windows, Linux and macOS** with the C engine
pre-compiled inside — no compiler needed. On any other platform BioForge falls
back to the pure-NumPy path automatically.

**From source** (latest `main`):
```bash
git clone https://github.com/erlanders177/bioforge.git
cd bioforge
pip install -e .          # only needs NumPy
```

**Requirements**
- Python ≥ 3.10
- NumPy ≥ 1.24 — the only runtime dependency
- The C engine ships pre-compiled (OpenMP, zlib and libdeflate statically linked
  inside the binary). If it can't load on your platform, BioForge falls back to
  NumPy automatically.

**Optional — recompile the C engine** (needed only if you build from source on an
unsupported platform, or change `engine.c`):
```bash
python bioforge/engine/build.py
```
Requires GCC. On Windows: [MinGW-w64](https://www.mingw-w64.org/) / MSYS2. On Linux/Mac: `sudo apt install gcc` / `brew install gcc`.  
If not compiled, BioForge falls back to NumPy automatically — no code changes needed.

For development and testing:
```bash
pip install hypothesis pytest pytest-benchmark
```

---

## Desktop app — no coding required *(v10.0)*

BioForge also has a **desktop app**: the same engine behind a friendly window, for
people who don't write code. Everything runs **locally and offline** — your DNA never
leaves your machine (*DNA Edge*). Seven tabs, each with a plain-language explanation:

- **🧬 Sequences** — browse your FASTA/FASTQ, see each sequence's type and size, and
  translate DNA → protein (codon by codon, colour-coded by amino-acid type).
- **📊 Quality** — a FastQC-style report for FASTQ (per-position quality, GC, Phred),
  drawn as inline charts.
- **⚗️ Align** — compare two sequences and see their differences (mutations) highlighted.
- **🔍 Variants** — stack many reads on a reference genome, see the coverage and get the
  mutations, with a one-click **VCF** export.
- **🌳 Tree** — reconstruct the evolutionary tree of several sequences, drawn with
  branch lengths and **bootstrap support**, exportable as **Newick**.
- **〜 Nanopore** — turn raw electrical signal (POD5/FAST5) into DNA bases with our own
  classical basecaller, then reuse those bases anywhere in the app.
- **🔮 Evolution** — rank which protein mutations may rise, and reality-check a specific one.

**Get it — two faces, same code at the same version:**

- **Just double-click (no Python):** download `BioForge-<version>-windows.zip` from the
  [latest release](https://github.com/erlanders177/bioforge/releases), unzip, and run
  `BioForge.exe`. Self-contained — nothing to install. *(The `.exe` is built
  automatically and attached to each release.)*
- **From the package (for coders):**
  ```bash
  pip install "bioforge[app]"
  bioforge-app
  ```

---

## Quick start

### Import and encode a FASTA sequence

```python
from bioforge import SmartImporter, SeqType

records = SmartImporter.from_string(""">gene_1
ATGGTGCACCTGACTCCTGAGGAGAAGTCTGCC
""")

seq = records[0]
print(seq.n_symbols)      # 33
print(len(seq.data))      # 21  (37.5% smaller than ASCII)
print(seq.to_string())    # ATGGTGCACCTGACTCCTGAGGAGAAGTCTGCC
```

### Stream a huge FASTA/FASTQ with constant RAM

```python
from bioforge import SmartImporter

# One PackedSequence at a time — never loads the whole file
for seq in SmartImporter.stream("genome.fa"):
    print(seq.header, seq.n_symbols)

# FASTQ yields FastqRecord (sequence + Phred qualities)
for rec in SmartImporter.stream_fastq("reads.fastq"):
    if rec.passes_quality(20):
        process(rec.sequence)
```

### Quality-filter millions of reads — the fast lane (columnar)

```python
from bioforge import SmartImporter

total = passed = 0
for batch in SmartImporter.stream_fastq_batches("reads.fastq"):
    mask = batch.passes(20)          # ONE NumPy op for thousands of reads
    total  += len(batch)
    passed += int(mask.sum())
    kept = batch.filter(mask)        # new ReadBatch, no per-read objects
print(f"{passed}/{total} reads with mean quality >= 20")
```

`stream_fastq_batches` keeps a whole batch as contiguous matrices instead of
one object per read, so filtering 200 000 reads drops from ~5.3 s to ~0.28 s.
Materialise a single read only when you need it: `batch[i]` → `FastqRecord`.

Compressed `.gz` files are read transparently (decompressed in C):

```python
for rec in SmartImporter.stream_fastq("reads.fastq.gz"):   # no manual gunzip
    ...
```

Pass `n_threads` to go multi-core (an adaptive dispatcher picks the best path):

```python
# plain → parallel parse · .gz → libdeflate (~2× faster) + parse
for batch in SmartImporter.stream_fastq_batches("reads.fastq.gz", n_threads=0):
    ...   # n_threads: 1 = sequential (constant RAM) · >1 = threads · 0 = all cores
```

Reading compressed FASTQ is **~1.6× faster** this way (libdeflate beats zlib);
plain-file parse parallelism is memory-bandwidth bound, so its gain is modest on
few cores but scales on many-core servers.

### BGZF — parallel-decompressible `.gz` (~2× faster reads)

A BGZF file is a **valid `.gz`** (any `gunzip` reads it) but split into
independent 64 KB blocks, so BioForge decompresses it across all cores. Convert
once a file you'll process repeatedly:

```bash
python -m bioforge.bgzf reads.fastq        # or: bioforge-bgzip reads.fastq
# → reads.fastq.gz (BGZF). Reads at ~113 M bases/s vs ~58 for plain .gz.
```

BioForge auto-detects BGZF and routes to the parallel path; plain `.gz` keeps
using single-thread libdeflate.

### GC content and k-mer spectrum — vectorised over a whole batch

```python
from bioforge import SmartImporter

spectrum = None
for batch in SmartImporter.stream_fastq_batches("reads.fastq"):
    gc = batch.gc_content()              # GC fraction per read (NumPy array)
    s  = batch.kmer_spectrum(k=4)        # counts of all 4^4 k-mers in the batch
    spectrum = s if spectrum is None else spectrum + s
# spectrum[i] = how many times k-mer #i appears across the whole file
```

Both run with zero per-read objects; ambiguous bases (N) are skipped from k-mers.

### Fast FASTQ quality report (FastQC-style)

```bash
python -m bioforge.qcreport reads.fastq.gz        # or: bioforge-qc reads.fastq.gz
```

One pass, constant RAM. Reports read/base counts, length, overall GC, mean
quality, %reads ≥ Q20/Q30, plus per-read quality and GC histograms,
**per-position mean quality** (the FastQC signature plot) and per-base
composition — all built on the columnar API. Use `-o report.txt` to save it.

### Translate DNA to protein

```python
from bioforge import SmartTranslator

protein = SmartTranslator.translate(seq)
print(protein.to_string())   # MVHLTPEEKSA
```

### Detect mutations between two sequences

```python
from bioforge import SequenceAligner, format_alignment

result = SequenceAligner.align(seq_ref, seq_query)

print(f"Identity: {result.identity:.1%}")
print(format_alignment(result))

for mut in result.mutations:
    print(mut)
# Mutation(kind='substitution', pos_a=18, pos_b=18, sym_a='A', sym_b='T')
```

### Map long reads to a genome (Level 4 — seed-chain-align)

Locate reads in a reference far beyond what the O(m·n) aligner can handle,
minimap2-style: minimizer seeding → chaining → banded extension of the full read.
The **entire pipeline runs in C** behind an opaque index handle; Python is a thin
cover (with a verified, identical NumPy fallback).

```python
from bioforge import GenomeAligner

# Single sequence, or a whole multi-contig genome:
mapper = GenomeAligner({"chr1": chr1_seq, "chr2": chr2_seq, "plasmid": p_seq})

for m in mapper.map(read):
    print(m.to_paf())                # standard PAF, one line per mapping
    print(m.target_name, m.strand, m.target_start, f"{m.identity:.1%}")

# Map many reads in parallel — OpenMP inside the C engine (GIL-free):
results = mapper.map_batch(reads, n_processes=0)   # 0 = all cores
```

Handles multi-chromosome references (reports the contig + local coordinates),
both strands, aligns the full read, tolerates mismatches/indels, and reports a
mapping quality. Built once, the C index is reused for every query;
`map_batch` maps the whole batch in a single C call parallelised with OpenMP.

> **Speed, honestly.** The whole pipeline runs in C (SIMD banded extension +
> OpenMP batch). In a same-machine head-to-head (`tools/bench_vs_minimap2.py`,
> 4.8 Mb genome, 6000 simulated reads at 5% error, `minimap2 -a`), BioForge is
> **on par with minimap2 on multiple cores** (~4.3–5.0 vs ~4.3–4.9 Mb/s, sometimes
> ahead) and **~1.18× behind single-threaded** (~1.87 vs ~2.2 Mb/s) — both map all
> reads. Please reproduce it yourself and tell me where it breaks.
>
> Honest caveats: this is *E. coli* scale with simulated reads; minimap2 — years
> of hand-tuning, by a team — may well pull ahead at human-genome scale, on real
> noisy data, or with many cores. This is not "we beat minimap2"; it's "a
> from-scratch, `pip install`-and-go engine got competitive," and the goal from
> here is a niche it *doesn't* occupy (see Roadmap).

**Reproduce the benchmark yourself** (Linux/WSL, with `minimap2` installed):

```bash
pip install bioforge
git clone https://github.com/erlanders177/bioforge.git && cd bioforge
python3 tools/bench_vs_minimap2.py --genome 4800000 --reads 6000 --error 0.05
# prints Mb/s for minimap2 and BioForge at 1 thread and all cores, same reads.
# Numbers are relative to your machine — report back what you get.
```

> **Accuracy — fast is worthless if it's wrong.** On a **real *E. coli* K-12
> genome** (4.64 Mb, 5000 simulated reads, recording each read's true origin,
> ±50 bp tolerance): BioForge maps **~99.8%** of reads to the *correct* position
> — matching minimap2 (99.8% at 5% error, 99.7% vs 99.9% at 10%), with **99.8%
> concordance** between the two. So it's not fast-at-the-cost-of-correctness.
> Reproduce with `tools/accuracy_vs_minimap2.py` (grab a real genome from NCBI
> first). Honest note: at higher error minimap2 is marginally ahead, and this is
> *E. coli* scale — larger genomes may differ.

### Call variants: what actually changed (Level 9 — v10.2)

Mapping tells you *where* each read came from. Variant calling tells you **what
changed** — the step that turns a pile of reads into an answer. This closes the
pipeline: `FASTQ → mapping → pileup → variants → VCF`.

```python
from bioforge import GenomeAligner, pileup, call_variants, write_vcf

aligner = GenomeAligner(reference)                  # reference: str
pairs = [(read, hits[0]) for read in reads
         for hits in [aligner.map(read)] if hits]   # keep the primary hit

pile = pileup(reference, pairs, contig="ref")
print(pile)                       # Pileup('ref', 5000 bp, 200 reads, mean depth 10.0×)
print(f"{pile.covered(10)*100:.0f}% of the genome at 10× or more")

variants = call_variants(pile, reference, min_depth=5, min_af=0.2)
for v in variants:
    print(v)                      # Variant(ref:1501 C>A SNV AF=1.00 DP=46 Q=920)

open("calls.vcf", "w").write(write_vcf(variants, contigs=[("ref", len(reference))]))
```

From the terminal, the same pipeline in one command — it also prints a coverage report:

```bash
bioforge-variants reference.fasta reads.fastq -o calls.vcf
#   reference : chromosome_1  (4,000 bp)
#   mapped    : 500 (100.0%)
#   mean depth: 31.2×    ≥10×: 97.2% of the genome
#   variants  : 3  (3 substitutions, 0 indels)

bioforge-variants ref.fasta reads.fastq --error-rate 0.05   # noisy (nanopore) reads
bioforge-variants ref.fasta reads.fastq --solo-cobertura    # coverage report only
```

**How it decides.** For every position it weighs two hypotheses with a binomial
**likelihood ratio**: *"these odd reads are sequencing errors"* (rate `error_rate`)
versus *"this base is really there"* (at the observed frequency). `QUAL` is
`10·log₁₀ LR`, which is exactly the Phred scale VCF expects — so Q30 means the
variant is a thousand times likelier than error. The binomial coefficient cancels
in the ratio, so it's plain logarithms over NumPy arrays: no special functions, no
new dependencies, no loop over positions.

**Measured** (`tools/bench_variants.py` — 5 kb genome, 25 known SNVs):

| read error | coverage | sensitivity | precision |
|-----------|----------|-------------|-----------|
| 0.1% / 1% | 10× and up | **100%** | **100%** |
| 0.1% / 1% | 5× | 64–72% | **100%** |
| 5% | 10× | 100% | 71% → **100%** with `error_rate=0.05` |

At low coverage it loses sensitivity but *never* precision: **it prefers silence
over invention**, which is the deliberate design. With noisy reads, match
`error_rate` to your sequencer — the parameter is not decorative.

> **Honest limitation — long indels come out split, and the cause is upstream.**
> The aligner uses a **linear** gap model (`GAP = −2` per base), so a 5 bp gap
> costs exactly the same whole as split into 3+2 — nothing pushes it to stay
> together. Measured: a 5 bp deletion is reported as 3 bp + 2 bp, a 4 bp insertion
> as 1 + 3. The indel *is* detected, but its exact coordinates may be spread. The
> real fix is **affine** gap penalties in the aligner (open expensive, extend
> cheap), not a patch in the caller — it's on the roadmap. SNVs are unaffected.

It's a **haploid, single-sample** caller (viruses, bacteria, amplicons): it does
not model diploid genotypes and does not compete with GATK there.

### Align many sequences at once (Level 4 — multiple sequence alignment)

Line up several sequences column-by-column — the basis for consensus, phylogeny
and the evolution front. Uses the **center-star** heuristic (align all to a
central sequence via the C aligner, then merge gaps), ideal for sets of similar
sequences (e.g. the same gene across strains over time).

```python
from bioforge import align_multiple

msa = align_multiple([
    "ATGGCCTTAGGCTA",
    "ATGGCGTTAGGCTA",
    "ATGGCCTTAGCTA",     # a deletion
    "ATGGCCTTAGGCTAA",   # an insertion
])
for row in msa.aligned:
    print(row)            # all rows same length, homologous columns
print(msa.consensus())    # majority consensus (N where ambiguous)
```

Every row with its gaps removed reproduces the original sequence exactly (no data
loss). *Honest scope:* center-star is the simple, correct starting point and
shines on similar sequences; serious aligners (Clustal Omega, MAFFT, MUSCLE) use
progressive + iterative refinement for divergent sets — a future upgrade.

### Evolution: rank the mutations that will rise (Level 5 — v7.0)

Given dated sequences of a gene under selection (e.g. a flu HA across seasons),
BioForge ranks **which mutations are most likely to rise next**, designates
**stable lineages**, and **backtests every prediction against the trivial "tomorrow
= today" baseline**. Genome-agnostic: nothing about flu or any organism is
hard-coded. The date goes in the FASTA header (a year, or `YYYY-MM`).

```bash
# Rank candidate mutations (site, target residue) by probability of rising:
bioforge-evolution rank strains.fasta --top 20

# Only mutations never seen before — where counting can't help and this earns its keep:
bioforge-evolution rank strains.fasta --novel --translate

# Is the model actually better than "tomorrow = today"?  (the honest judge)
bioforge-evolution backtest strains.fasta

# Designate stable lineages (Pango/autolin-style) with their defining mutations:
bioforge-evolution lineages strains.fasta
```

```python
from bioforge import rank_mutations
r = rank_mutations(protein_seqs, years, novel_only=True)
for site, residue, score in r.ranked[:10]:
    print(site + 1, residue, round(score, 3))
```

**How it works, and the honesty that comes with it** (this is a research tool, and
its own measured limits are baked in):

- **The right question.** Predicting exact *frequencies* is a dead end — it ties the
  naive baseline at every horizon we tested (3–18 months), because the naive
  baseline is nearly optimal there. So instead we do what the field actually does
  (EVEscape, Łuksza): **rank mutations** (AUC). Here the naive baseline doesn't even
  play — "nothing changes" ranks nothing.
- **Three genome-agnostic axes** feed a small trained model: how a site has changed
  in the past (a data-driven stand-in for structural accessibility), the
  physico-chemical (dis)similarity of the substitution, and its recent growth.
  Notably, the "escape" dissimilarity axis is **inverted** — in flu HA the
  substitutions that rise are *conservative*, replicated across H3N2, H1N1 and B.
  It measures **viability**, not escape (without a structural-accessibility term,
  most of a domain is core: "disruptive" means "breaks the protein", not "escapes").
- **The model is a tiny neural net (MLP 2×64) run in pure NumPy** — training used
  PyTorch as scaffolding, but the shipped model is a 39 KB `.npz` and inference is
  three matrix multiplies. No PyTorch, no GPU, runs on a laptop. It beats a plain
  linear model on all six held-out tests (per-virus **and** cross-virus — trained on
  two influenza types, tested on a third), which is where it helps most.
- **What it is not.** None of this is scientifically novel — DERIVE, EVEscape and
  Hie et al. already rank escape mutations and cross viruses, with more resources and
  usually better. An optional ESM-2 axis (`pip install bioforge[ai]`) exists but
  suffers **pretraining leakage** (its AUC drops ~0.20 on data after its training
  cutoff — measured, and off by default). The value here is the *integrated, honest,
  laptop-runnable box*, not a new state of the art.

### Judge a predictor honestly (Level 6 — v8.0)

Building the Level 5 predictor was mostly a fight against **our own measurements
flattering us**. Every check below exists because it caught us:

```python
from bioforge import EvolutionBenchmark

bench = EvolutionBenchmark(sequences, dates)      # real, dated sequences
report = bench.judge(my_predictor)                # any f(Context) -> scores
print(report)
```

```
── VEREDICTO ─────────────────────────────────────────────
  AUC global            0.837
  AUC en NUEVAS         0.631   (el régimen difícil)
  listón trivial        0.793   (mutabilidad del sitio)
  IC95% del AUC         [0.830, 0.848]
  (220,092 candidatas · 20 cortes)

  → APORTA de verdad: supera el listón trivial y aguanta en NUEVAS.
```

- **The bar is not 0.5.** It is the best *free* axis (current frequency, site
  mutability, physico-chemical conservation). Our own headline "AUC 0.80" turned
  out to be **exactly** the mutability axis — a tautology, not a prediction.
- **Two regimes.** Mutations already circulating (where counting is enough) are
  split from genuinely new ones. Our hand-tuned axes score ~0.52 on new mutations
  — chance. The trained model scores 0.631. That gap *is* what the AI buys.
- **Bootstrap CI.** Small wins evaporate when you resample. If the interval
  touches 0.5, the verdict says *not demonstrated*.
- **Pretraining-leakage detector.** Compares AUC before/after a model's training
  cutoff, *subtracting* the drop of a trivial control axis. In a planted test a
  cheating predictor scored **AUC 0.863 with a clean CI [0.859, 0.866] — better
  than the trivial bar (0.793), and 0.857 on the hard regime — and was still
  flagged** (leak −0.438). Without this check we would have published that
  number. It is the same signature we measured on ESM-2 (−0.20 after its cutoff).
- **`Context` is leak-free by construction**: a predictor is only ever handed
  time bins *before* the one being scored, so lookahead is not possible by
  accident.
- **`cross_validate`** re-runs the battery on other organisms to test whether a
  model generalises or memorised one virus.

The point of Level 6 is not to win a benchmark. It is that a claim about
predicting evolution should be *hard to make by mistake*.

> These numbers are measured on real H3N2 hemagglutinin (900 sequences, 2013–2022,
> 220k candidate mutations). An earlier draft reported 0.861/0.613 — computed, we
> later found, on a **corrupted multiple alignment** (a silent bug that packed every
> protein as DNA, turning non-ACGT residues into `N`). The bug is fixed, the model
> retrained on clean alignments, and every figure above re-measured. The two
> honesty tools in this section are what surfaced the corruption in the first place.

### Filter another tool's hits by reality (Level 6 — v8.0)

`evalkit` judges a whole *predictor*. `RealityCheck` judges individual *mutations*:
of the "concerning" variants some other tool (EVEscape, ESM-2, a DMS assay) hands
you, which ones have **real traction in the population**? It plugs in behind any of
them.

```python
from bioforge import RealityCheck

rc = RealityCheck(sequences, dates)          # real, dated sequences
survivors = rc.filter(candidates_from_another_tool)   # keeps only what has traction
print(rc.check("N145K"))
```

- **Two tiers, never mixed.** `OBSERVADO` — the mutation already exists in the
  record, so we return its real trajectory (this is *evidence*). `ESTIMADO` — never
  seen, so the trained model *estimates* it (this is a *conjecture*, and it says so).
- **Each tier's reliability is measured separately** and travels with the verdict.
  On real H3N2: observed **AUC 0.97** (survival of already-circulating variants is
  largely *seen*, not guessed), estimated **AUC 0.72** (genuine prediction for the
  novel ones).
- **"Survival" means reaching or holding real presence**, not "rising 5%". A variant
  already at 98 % cannot rise but is the clearest survivor there is — scoring *rise*
  would mark it a failure (ceiling effect). That is exactly the question you asked:
  *would this mutation survive out there?*
- **Calibrated** against the historical record: 0.70 means ≈70 % of past mutations
  that scored ~0.70 went on to establish. **Resilient**: one malformed entry in a
  batch never sinks the rest (it comes back `NO EVALUABLE`).

> Coordinates note: positions are alignment-column indices of *your* data, not a
> standard scheme (e.g. H3 numbering). If you paste `N145K` from a paper, the tool
> tells you when the wildtype residue it expected there disagrees — a hint you are
> in a different numbering.

### Basecall a nanopore read from raw signal (Level 7 — v9.0)

Oxford Nanopore devices push DNA through a pore and measure the **ionic current** —
raw electrical signal. Turning that signal back into bases is *basecalling*. BioForge
does it **from scratch, in pure NumPy, with no neural network** — the classical
route (a Viterbi HMM over a k-mer pore model), the same algorithmic family as the
Level 3 aligner.

```python
from bioforge import read_fast5, basecall            # pip install "bioforge[nanopore]"

read = next(iter(read_fast5("read.fast5")))          # POD5 also: read_pod5(...)
bases = basecall(read.to_picoamperes(), pore_model, k=6)
```

- **Reads real signal**: `read_pod5` (modern) and `read_fast5` (legacy) — the only
  place an optional dependency is used, purely to open the file format. Everything
  after (normalise · detect events · estimate the pore model · Viterbi) is pure NumPy.
- **Learns its own pore model** (`estimate_pore_model`) from labelled signal rather
  than shipping someone else's table.
- **Viterbi with stay/step/skip** (`viterbi_basecall`) so imperfect segmentation no
  longer breaks the call — the classical fix that took us from 58 % to 70 %.
- **Iterative rescaling** (v9.1): a first pass tells us which k-mer each event is, then
  the per-read scale is refit to the *actual* levels of those k-mers — that plus a
  better stay probability lifted real-signal accuracy from ~70 % to ~74 %.

**Honest, reproducible numbers.** On **real captured R9.4** signal (E. coli, n=36,
identity to the production Guppy basecall, aligned locally with our own aligner):
**mean ~74.5 %, median ~74 %, range 68–82 %**. That is squarely in the range of the
historical classical basecallers (nanocall ~68–85 %) and far below the ~99 % of the
neural Dorado — which is the honest point: *the classical route is an R9-era method,
and without AI it has a ceiling*. Reproduce it yourself: `python tools/bench_basecaller.py`.

> **Why not R10 (modern chemistry)?** It reads ~9 bases at once → 4⁹ = 262 144 hidden
> states, so an O(T·states) Viterbi is infeasible on a laptop; and ONT ships no flat
> k-mer table for R10 (its models are neural, inside Dorado). That wall is the real
> reason the field moved to neural basecalling. BioForge reads R10 signal fine; it
> just doesn't pretend a classical decoder can call it well.

### Full mutation analysis pipeline (DNA + protein)

```python
from bioforge import run, build_report

result = run("reference.fa", "query.fa", mode="both")
print(build_report(result))
```

### Error handling

```python
from bioforge import BioForgeError, TranslationError, SmartImporter, SmartTranslator

try:
    for rec in SmartImporter.stream_fastq("reads.fastq.gz"):
        protein = SmartTranslator.translate(rec.sequence)
except TranslationError as e:
    print(f"Translation failed: {e}")   # e.g. no ATG found
except BioForgeError as e:
    print(f"BioForge error: {e}")       # ANY engine error: parse, I/O, decompress…
```

**One exception family.** Every engine error subclasses `BioForgeError`, so a
single `except BioForgeError` catches them all — translation, alignment, parsing,
file I/O (`BioForgeIOError`), engine/decompression (`EngineError`). Each also
subclasses the matching builtin (`ValueError`, `OSError`, `RuntimeError`…), so
existing `except OSError`-style code keeps working.

### Run the verifier (no coding knowledge required)

```bash
python check.py
```

---

## Project structure

```
bioforge/               organised BY FUNCTION (v10.1); tests mirror it
  __init__.py           Public API + LAZY loading (_EXPORTS map, PEP 562) + version
  core/biocore.py       THE FOUNDATION — 5-bit storage, BitPacker, PackedSequence,
                        SmartImporter (FASTA/FASTQ reader), columnar batches, errors
  sequence/translator.py   Level 2 — DNA → protein, 6-frame, reverse complement
  align/
    pairwise.py         Level 3 — NW global/semi-global, banded, Smith-Waterman
    msa.py              Multiple sequence alignment (center-star)
  mapping/
    minimizers.py       Level 4 — canonical (w, k) minimizers (C + NumPy)
    refindex.py         Level 4 — reference minimizer index (hash-sorted lookup)
    genomemap.py        Level 4 — GenomeAligner: seed-chain-align → PAF
  phylo/
    distance.py         Level 10 — distance matrices (p/JC/K2P/Poisson), matmul-based
    tree.py             Level 10 — Neighbor-Joining, UPGMA, WPGMA, Newick, bootstrap
  variants/
    pileup.py           Level 9 — stacks reads on the reference: depth, coverage
    caller.py           Level 9 — SNV/indel calling (binomial LR) → VCF 4.2
  evolution/
    predict.py          Level 5 — mutation ranking, stable lineages, backtesting
    evalkit.py          Level 6 — honest predictor judge (EvolutionBenchmark)
    realitycheck.py     Level 6 — mutation reality filter (RealityCheck)
    fetch.py            Level 5 — dated NCBI Entrez download (stdlib, cached)
    ai/viability.py     Level 5 — optional ESM-2 axis (bioforge[ai], lazy-loaded)
  nanopore/basecaller.py   Level 7 — from-scratch basecaller (signal → bases)
  io/
    qcreport.py         Fast FASTQ quality report (FastQC-style) — bioforge-qc
    bgzf.py             BGZF converter (parallel block gzip) — bioforge-bgzip
  cli/
    analyze.py          Full DNA + protein pipeline — bioforge-analyze
    evolution.py        Evolution CLI (rank/backtest/lineages) — bioforge-evolution
  app/                  Desktop app (v10.0) — local window over the engine
    main.py             PyWebview launcher (window + native file dialogs)
    backend.py          The bridge the UI calls (Api) — tested without a window
    index.html          The whole interface (vanilla JS, offline, inline SVG charts)
    data/               UI resources: pore model + app icon
  data/                 Trained mutation-ranker weights (.npz, in the wheel)
  engine/
    engine.c            C source — pack/unpack, NW, translate, parser, mapper
    engine.dll          Compiled C backend (Windows; .so on Linux/macOS)
    _loader.py          ctypes wrapper with automatic NumPy fallback
    build.py            Compiles the DLL/SO (auto-detects GCC)
  aligner.py biocore.py …  14 compatibility BRIDGES at the old flat paths, so code
                        written against <=10.0.0 keeps working (DeprecationWarning)
check.py                Non-programmer verifier (runs all checks automatically)
conftest.py             Pytest fixtures shared across all tests

tools/
  visor.py              Interactive step-by-step translator (CLI)
  comparador.py         Sequence comparator tool (CLI)
  stress_test.py        30M-base performance benchmark
  bench_vs_biopython.py BioForge vs Biopython: time + RAM (FASTQ parse/QC/load)

tests/                  mirrors the package layout (630 tests)
  core/                 5-bit storage, streaming/columnar, errors, integrity net
  sequence/             genetic code correctness + error paths
  align/                alignment properties, MSA, SIMD kernel parity
  mapping/              minimizers, index, seed-chain-align, C parity
  phylo/                distances, NJ exactness on additive matrices, bootstrap honesty,
                        head-to-head against Biopython
  variants/             pileup, SNV/indel calling, VCF, no-false-positive guarantees
  evolution/            ranking, honest judge, reality filter, ESM-2 axis, CLI
  nanopore/             signal I/O, event detection, Viterbi basecaller
  io/                   FASTQ quality report, BGZF
  cli/                  full pipeline integration + CLI
  app/                  desktop app bridge, tested without opening a window
  test_isolation.py     architecture guard: each tool loads without waking the others
docs/
  architecture.md       Design rules, levels, encoding details
  api_reference.md      Code examples for every module
  benchmarks.md         Measured numbers and methodology
  roadmap.md            Status and planned extensions
```

---

## How the vectorisation works

### Translation (Level 2)

```
① decode PackedSequence → uint8 array  [0–3 per nucleotide]
② find first ATG        → C engine scan / NumPy sliding_window_view
③ extract ORF, reshape  → (N, 3) codon matrix
④ base-4 index          → idx = n₁×16 + n₂×4 + n₃  (vectorised)
⑤ CODON_LUT[idx]        → amino acid array  (single fancy-index)
⑥ argmax on STOP mask   → truncate at stop codon
```

### Alignment (Level 3)

Needleman-Wunsch has a cell-level data dependency that prevents full 2D
vectorisation. The solution: **anti-diagonal wavefront**.

Cells on the same anti-diagonal (`i + j = d`) are mutually independent,
so each diagonal is a single vectorised operation.  
Python-level iterations: **O(m+n)** instead of O(m·n).

When the C engine is available, the entire DP matrix is computed in C
with OpenMP, giving **~29× speedup** over the NumPy wavefront.

### C engine

`bioforge/engine/engine.c` provides optimised implementations of all hot-path
operations. Loaded automatically via `ctypes` at import time.  
If `engine.dll` is missing, all code falls back to NumPy silently.

```python
from bioforge.engine._loader import C_AVAILABLE
print(C_AVAILABLE)   # True if C engine loaded, False if using NumPy fallback
```

---

## Running the tests

```bash
# Full test suite (630 tests)
pytest tests/ -v

# Benchmarks only
pytest tests/ --benchmark-only

# Quick smoke check (no coding knowledge required)
python check.py
```

---

## Known limitations

| Limitation | Detail |
|------------|--------|
| Aligner memory (full NW) | O(m·n) matrix — sequences > 15 000 bp may exhaust RAM. Use `band=N` for large sequences. |
| Protein auto-detection | Sequences without E/F/I/L/P/Q/* are classified as nucleotides. Use `force_type=SeqType.PROTEIN` to override. Since v8.0, mis-encoding is no longer silent: the encoder **refuses** a sequence that is mostly invalid for its type (would-be corruption raises `SequenceValueError`). |
| RealityCheck coordinates | Positions are alignment-column indices of *your* data, not a standard scheme (e.g. H3 numbering). The tool warns when the wildtype it expected at a position disagrees with what you passed. |
| C engine | Ships pre-compiled in the PyPI wheels. Building from source on an unsupported platform needs GCC (`python bioforge/engine/build.py`). |
| Banded NW (NumPy fallback) | Without the C engine, banded NW uses the full matrix with NEG_INF masking — same result, standard RAM. |
| Variant caller — long indels | Reported **split** (a 5 bp deletion as 3+2), because the aligner's **linear** gap model prices a contiguous gap the same as a broken one. The indel is detected but its coordinates may be spread; SNVs are unaffected. Real fix: affine gap penalties in the aligner (on the roadmap). |
| Variant caller — ploidy | Haploid / single sample (viruses, bacteria, amplicons). No diploid genotype model (0/1, 1/1), no strand-bias filter, no per-base quality integration — it does not compete with GATK on human genomes. |
| Genome mapper — tested scale | Benchmarked on par with minimap2 on multi-core and ~1.18× behind single-threaded at *E. coli* scale with simulated reads (`tools/bench_vs_minimap2.py`). Not yet validated at human-genome scale or on real noisy data, where minimap2 may pull ahead. |

---

## Roadmap

- [x] Level 1 — 5-bit storage, FASTA parser, SmartImporter
- [x] Level 2 — vectorised genetic code translation (C + NumPy)
- [x] Level 3 — Needleman-Wunsch alignment + mutation detection (C + NumPy)
- [x] Full mutation analysis pipeline (DNA + protein, 3 modes)
- [x] BioForgeError exception hierarchy for library users
- [x] Reverse complement vectorised — `PackedSequence.reverse_complement()`
- [x] 6-frame translation — `SmartTranslator.translate_all_frames()`
- [x] Banded NW — `SequenceAligner.align(seq_a, seq_b, band=N)`
- [x] Smith-Waterman local alignment — `SequenceAligner.align_local()`
- [x] Streaming FASTA/FASTQ parser in C — `SmartImporter.stream()` / `stream_fastq()`
- [x] Batch parser (5-bit encoding in C) — ~80 M bases/s FASTA, ~94 K reads/s FASTQ
- [x] Columnar QC API — `stream_fastq_batches()` · `ReadBatch.passes()` / `filter()`
- [x] Compressed `.gz` decoded in C (zlib, static-linked, transparent)
- [x] Object-free columnar k-mer spectrum + per-read GC — `kmer_spectrum()` / `gc_content()`
- [x] Benchmark vs Biopython — `tools/bench_vs_biopython.py`
- [x] Fast FASTQ quality report (FastQC-style) — `bioforge-qc` / `bioforge.qcreport`
- [x] Adaptive multi-core dispatcher — `n_threads=`: parallel parse + libdeflate `.gz`
- [x] BGZF parallel-decompressible `.gz` + converter — `bioforge-bgzip`
- [x] Native per-platform wheels on PyPI (cibuildwheel) — `pip install bioforge`
- [x] Long-read / genome-scale aligner — `GenomeAligner` (seed-chain-align, PAF)
- [x] Whole mapping pipeline in C behind an opaque index — `bio_map_read` / `bio_map_batch` (OpenMP)
- [x] SIMD banded extension (AVX2, int32 + int16) — `_nw_banded_diag_simd` *(v6.0 / v6.2)*
- [x] Columnar `map_batch` output → full multi-core scaling *(v6.1)*
- [x] Head-to-head benchmark vs minimap2 (`tools/bench_vs_minimap2.py`, WSL) — on par multi-core
- [x] Multiple sequence alignment (center-star) — `align_multiple` *(v6.3)*
- [x] **Evolution front — mutation ranking, stable lineages, honest backtesting** — `bioforge-evolution` *(v7.0)*
- [x] **Trained mutation-ranker** (MLP in pure NumPy, no PyTorch at inference) + optional ESM-2 axis *(v7.0)*
- [x] **Level 6 — honest predictor judge** `EvolutionBenchmark` (trivial-baseline bar · novel-regime split · bootstrap CI · pretraining-leakage detector) *(v8.0)*
- [x] **Level 6 — reality filter** `RealityCheck`: rank another tool's mutation hits by real-world traction, observed vs estimated tiers *(v8.0)*
- [x] **Anti-corruption integrity net** — encode guard + property-based invariants (both alphabets) + `tools/integrity_check.py` *(v8.0)*
- [x] **Level 7 — nanopore basecaller from scratch** (POD5/FAST5 readers · event detection · own pore-model estimation · Viterbi stay/skip · pure NumPy) — **~70 % on real R9.4** *(v9.0)*
- [x] Nanopore: **iterative per-read rescaling + tuned transitions → ~74 %** on real R9.4 *(v9.1)*
- [x] **Variant calling** — `pileup` + `call_variants` → VCF 4.2, binomial likelihood-ratio QUAL, honest sensitivity/precision benchmark *(v10.2)*
- [x] **Phylogenetics** — distance matrices, Neighbor-Joining, UPGMA/WPGMA, Newick, bootstrap support; validated head-to-head against Biopython *(v10.2)*
- [x] **Head-to-head for the variant caller against `bcftools`** (WSL) — 100% concordance on identical alignments *(v10.2)*
- [ ] Affine gap penalties in the aligner (open/extend) so long indels stop being split — the measured root cause behind imprecise indel coordinates
- [ ] Nanopore: keep lifting (drift term, homopolymers, trained transition/emission model); pluggable Dorado backend when present
- [ ] Structural-accessibility axis (to separate escape from viability) — the term EVEscape has and we don't
- [ ] Validate the mapper at human-genome scale on real (non-simulated) reads
- [x] **Phase 2 — desktop application** *(v10.0)*: the whole engine behind a friendly local window (7 tabs), for non-coders. Ships **inside the package** (`bioforge.app`, launch with `bioforge-app`) **and** as a self-contained `.exe`, auto-built and attached to each release. Local, no servers, privacy-first — your DNA never leaves the machine
- [x] **Organised by function + lazy loading** *(v10.1)*: subpackages per domain (`core/ sequence/ align/ mapping/ evolution/ nanopore/ io/ cli/ app/`), tests mirroring them, and a package that loads only what you use (`import bioforge` 75 ms → 4.7 ms). Old import paths keep working through compatibility bridges
- [ ] **Phase 3 — a true predictor**, built to beat the honest bar Level 6 now measures (0.631 on novel mutations)

---

## References & inspiration

BioForge's genome mapper (Level 4) is an **independent, from-scratch
implementation** of well-established, published algorithms. No third-party
source code is included or copied — only the *ideas* from the scientific
literature, which is what publishing them is for. With gratitude to:

- **Minimap2** — Li, H. (2018). *Minimap2: pairwise alignment for nucleotide
  sequences.* Bioinformatics, 34(18), 3094–3100. The seed-chain-align strategy
  and the chaining dynamic program that inspired `genomemap.py`.
  ([paper](https://doi.org/10.1093/bioinformatics/bty191) ·
  [MIT-licensed source](https://github.com/lh3/minimap2))
- **Minimizers** — Roberts, M., Hayes, W., Hunt, B. R., Mount, S. M., &
  Yorke, J. A. (2004). *Reducing storage requirements for biological sequence
  comparison.* Bioinformatics, 20(18), 3363–3369. The (w, k) minimizer sampling
  behind `minimizers.py`.
- **Needleman–Wunsch** (1970) and **Smith–Waterman** (1981) — the classic
  dynamic-programming alignments behind Level 3.

BioForge is not affiliated with or endorsed by the authors of the above.

---

## Author

**Aarón Aranda Torrijos** — [github.com/erlanders177](https://github.com/erlanders177)

---

## License

PolyForm Noncommercial 1.0.0 — free for personal, academic and research use.  
Commercial use requires explicit permission from the author.

See [LICENSE](LICENSE) for full terms.
