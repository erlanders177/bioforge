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
| Dependencies | **NumPy** (C engine included, pre-compiled) |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Level 3 — aligner.py          Needleman-Wunsch alignment    │
│  Anti-diagonal wavefront O(m+n) · mutation detection         │
├──────────────────────────────────────────────────────────────┤
│  Level 2 — smart_translator.py  DNA → Protein translation    │
│  CODON_LUT + sliding_window_view · first-ATG ORF detection   │
├──────────────────────────────────────────────────────────────┤
│  Level 1 — biocore.py           5-bit storage engine         │
│  BitPacker · PackedSequence · SmartImporter · LUTs           │
├──────────────────────────────────────────────────────────────┤
│  C engine — engine/engine.c     Optional compiled backend     │
│  GCC -O3 -march=native -fopenmp · auto-loaded via ctypes     │
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
git clone https://github.com/erlanders177/bioforge.git
cd bioforge
pip install numpy
```

**Optional — compile the C engine** (27–29× faster on translation and alignment):
```bash
python engine/build.py
```
Requires GCC. On Windows: [MinGW-w64](https://www.mingw-w64.org/). On Linux/Mac: `sudo apt install gcc` / `brew install gcc`.  
If not compiled, BioForge falls back to NumPy automatically — no code changes needed.

For development and testing:
```bash
pip install hypothesis pytest pytest-benchmark
```

---

## Quick start

### Import and encode a FASTA sequence

```python
from biocore import SmartImporter, SeqType

records = SmartImporter.from_string(""">gene_1
ATGGTGCACCTGACTCCTGAGGAGAAGTCTGCC
""")

seq = records[0]
print(seq.n_symbols)      # 33
print(len(seq.data))      # 21  (37.5% smaller than ASCII)
print(seq.to_string())    # ATGGTGCACCTGACTCCTGAGGAGAAGTCTGCC
```

### Translate DNA to protein

```python
from smart_translator import SmartTranslator

protein = SmartTranslator.translate(seq)
print(protein.to_string())   # MVHLTPEEKSA
```

### Detect mutations between two sequences

```python
from aligner import SequenceAligner, format_alignment

result = SequenceAligner.align(seq_ref, seq_query)

print(f"Identity: {result.identity:.1%}")
print(format_alignment(result))

for mut in result.mutations:
    print(mut)
# Mutation(kind='substitution', pos_a=18, pos_b=18, sym_a='A', sym_b='T')
```

### Full mutation analysis pipeline (DNA + protein)

```python
from analyze import run, build_report

result = run("reference.fa", "query.fa", mode="both")
print(build_report(result))
```

### Error handling

```python
from biocore import BioForgeError, TranslationError

try:
    protein = SmartTranslator.translate(my_seq)
except TranslationError as e:
    print(f"Translation failed: {e}")   # e.g. no ATG found
except BioForgeError as e:
    print(f"BioForge error: {e}")       # any other engine error
```

### Run the verifier (no coding knowledge required)

```bash
python check.py
```

---

## Project structure

```
biocore.py              Level 1 — 5-bit storage engine
smart_translator.py     Level 2 — DNA → protein translation
aligner.py              Level 3 — pairwise alignment + mutation detection
analyze.py              Full pipeline: DNA + protein analysis, report generation
check.py                Non-programmer verifier (runs all checks automatically)
conftest.py             Pytest fixtures shared across all tests

engine/
  engine.c              C source — pack, unpack, NW align, translate (GCC -O3)
  engine.dll            Compiled C backend (Windows)
  _loader.py            ctypes wrapper with automatic NumPy fallback

tools/
  visor.py              Interactive step-by-step translator (CLI)
  comparador.py         Sequence comparator tool (CLI)
  stress_test.py        30M-base performance benchmark

tests/
  test_biocore.py       L1: property-based tests (Hypothesis) + benchmarks
  test_translator.py    L2: genetic code correctness + error paths
  test_aligner.py       L3: alignment properties + mutation detection
  test_analyze.py       Pipeline: full integration tests + CLI tests

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

`engine/engine.c` provides optimised implementations of all hot-path
operations. Loaded automatically via `ctypes` at import time.  
If `engine.dll` is missing, all code falls back to NumPy silently.

```python
from engine._loader import C_AVAILABLE
print(C_AVAILABLE)   # True if C engine loaded, False if using NumPy fallback
```

---

## Running the tests

```bash
# Full test suite (209 tests)
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
| Aligner memory | O(m·n) matrix — sequences > 15 000 bp may exhaust RAM. Banded NW planned for v1.2. |
| Protein auto-detection | Sequences without E/F/I/L/P/Q/* are classified as nucleotides. Use `force_type=SeqType.PROTEIN` to override. |
| C engine | Pre-compiled `.dll`/`.so` not included. Run `python engine/build.py` to compile. Requires GCC. |
| Translation | Single-frame only (forward strand, first ATG). 6-frame translation planned for v1.1. |

---

## Roadmap

- [x] Level 1 — 5-bit storage, FASTA parser, SmartImporter
- [x] Level 2 — vectorised genetic code translation (C + NumPy)
- [x] Level 3 — Needleman-Wunsch alignment + mutation detection (C + NumPy)
- [x] Full mutation analysis pipeline (DNA + protein, 3 modes)
- [x] BioForgeError exception hierarchy for library users
- [ ] Reverse complement (vectorised)
- [ ] 6-frame translation
- [ ] Banded NW for sequences > 15 000 bp

---

## Author

**Aarón Aranda Torrijos** — [github.com/erlanders177](https://github.com/erlanders177)

---

## License

PolyForm Noncommercial 1.0.0 — free for personal, academic and research use.  
Commercial use requires explicit permission from the author.

See [LICENSE](LICENSE) for full terms.
