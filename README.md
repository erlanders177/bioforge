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

## Key numbers

| Operation | Result |
|-----------|--------|
| Memory (30M bases) | **18.75 MB** (37.5% less than plain ASCII) |
| Translation throughput | **~5 M amino acids / second** (NumPy) · **~27× faster** with C engine |
| NW alignment 1000×1000 nt | **~165 ms** (NumPy) · **~29× faster** with C engine |
| Genome mapping — speed vs minimap2 | **on par on multi-core**, ~1.18× behind single-thread (*E. coli* scale, `minimap2 -a`; `tools/bench_vs_minimap2.py`) |
| Genome mapping — accuracy vs minimap2 | **~99.8% of reads mapped to the correct position** on real *E. coli*, matching minimap2 (`tools/accuracy_vs_minimap2.py`) |
| FASTA ingestion (C batch parser) | **~80 M bases / second** |
| FASTQ ingestion (C batch parser) | **~14 M bases / s · ~94 K reads / s** |
| QC filter 200 K reads (columnar) | **0.28 s** — **18.6× faster** than per-record |
| vs Biopython — QC filter | **~5–6× faster**, identical result |
| vs Biopython — load all in RAM | **~6.9× less memory** (115 MB vs 801 MB) · ~9.5× faster |
| Compressed input | **`.gz` read transparently in C** (zlib, static-linked) |
| Dependencies | **NumPy** (C engine included, pre-compiled) |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
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
bioforge/               Python package — all core modules
  __init__.py           Public API entry point (from bioforge import ...)
  biocore.py            Level 1 — 5-bit storage engine
  smart_translator.py   Level 2 — DNA → protein translation
  aligner.py            Level 3 — pairwise alignment + mutation detection
  minimizers.py         Level 4 — canonical (w, k) minimizers (C + NumPy)
  refindex.py           Level 4 — reference minimizer index (hash-sorted lookup)
  genomemap.py          Level 4 — GenomeAligner: seed-chain-align → PAF
  analyze.py            Full pipeline: DNA + protein analysis, report generation
  qcreport.py           Fast FASTQ quality report (FastQC-style, columnar)
  bgzf.py               BGZF converter (parallel block gzip) — bioforge-bgzip
  engine/
    engine.c            C source — pack/unpack, NW, translate, parser, mapper
    engine.dll          Compiled C backend (Windows; .so on Linux/macOS)
    _loader.py          ctypes wrapper with automatic NumPy fallback
    build.py            Compiles the DLL/SO (auto-detects GCC)

check.py                Non-programmer verifier (runs all checks automatically)
conftest.py             Pytest fixtures shared across all tests

tools/
  visor.py              Interactive step-by-step translator (CLI)
  comparador.py         Sequence comparator tool (CLI)
  stress_test.py        30M-base performance benchmark
  bench_vs_biopython.py BioForge vs Biopython: time + RAM (FASTQ parse/QC/load)

tests/
  test_biocore.py       L1: property-based tests (Hypothesis) + benchmarks
  test_translator.py    L2: genetic code correctness + error paths
  test_aligner.py       L3: alignment properties + mutation detection
  test_analyze.py       Pipeline: full integration tests + CLI tests
  test_streaming.py     Streaming/batch parser + columnar API (Sequence/ReadBatch)
  test_qcreport.py      FASTQ quality report (qcreport.py)
  test_minimizers.py    L4: canonical minimizers (C == NumPy parity)
  test_refindex.py      L4: reference index lookup
  test_genomemap.py     L4: seed-chain-align, multi-contig, PAF, robustness
  test_cindex.py        L4: opaque C index parity (bio_index_build)

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
# Full test suite (375 tests)
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
| Protein auto-detection | Sequences without E/F/I/L/P/Q/* are classified as nucleotides. Use `force_type=SeqType.PROTEIN` to override. |
| C engine | Ships pre-compiled in the PyPI wheels. Building from source on an unsupported platform needs GCC (`python bioforge/engine/build.py`). |
| Banded NW (NumPy fallback) | Without the C engine, banded NW uses the full matrix with NEG_INF masking — same result, standard RAM. |
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
- [ ] **Evolution front — Markov substitution baselines + backtesting** (the differentiator)
- [ ] **Strain forecasting** with protein language models (ESM-2) — model how a sequence evolves
- [ ] Validate the mapper at human-genome scale on real (non-simulated) reads

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
