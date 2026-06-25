# BioCore — High-Performance Bioinformatics Engine

A NumPy-only bioinformatics engine designed for **Edge Computing**.  
No Biopython. No heavy dependencies. Just NumPy and fast math.

---

## Why this exists

Most bioinformatics tools are built for servers with gigabytes of RAM.  
This engine was built for the opposite: low-power hardware, minimal footprint,
maximum speed — running genetic analysis **offline and locally**.

The core design rule: **zero Python loops in the hot path**.  
Every operation is vectorised with NumPy.

---

## Key numbers

| Operation | Result |
|-----------|--------|
| Memory usage (30M bases) | **18.75 MB** (37.5% less than plain ASCII) |
| Translation throughput | **~5 million amino acids / second** |
| NW alignment (1000 × 1000 nt) | **~165 ms** |
| Dependencies | **NumPy only** (+ optional Numba) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Level 3 — aligner.py        Mutation detection         │
│  Needleman-Wunsch, anti-diagonal wavefront O(m+n)        │
├─────────────────────────────────────────────────────────┤
│  Level 2 — smart_translator.py   DNA → Protein          │
│  sliding_window_view + base-4 CODON_LUT, zero loops      │
├─────────────────────────────────────────────────────────┤
│  Level 1 — biocore.py        5-bit storage engine       │
│  BitPacker · PackedSequence · SmartImporter · LUTs       │
└─────────────────────────────────────────────────────────┘
```

### The 5-bit unified alphabet

Every biological symbol — nucleotides, amino acids, gaps, stop codons and
ambiguous bases — fits in a single 5-bit encoding (32 states).  
One scheme covers DNA, RNA, and proteins in the same pipeline.

```
State  Symbol          State  Symbol
  0    Adenine (A)      14    Methionine (M)
  1    Cytosine (C)     ...
  2    Guanine (G)      24    STOP codon (*)
  3    Thymine/Uracil   25    Alignment gap (-)
 4–23  Amino acids      31    Unknown / ambiguous
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/biocore-engine.git
cd biocore-engine
pip install numpy
```

Optional (for faster alignment on large sequences):
```bash
pip install numba
```

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
print(seq.packed_bytes)   # 21  (37.5% smaller than ASCII)
print(seq.to_string())    # ATGGTGCACCTGACTCCTGAGGAGAAGTCTGCC
```

### Translate DNA to protein

```python
from smart_translator import SmartTranslator

protein = SmartTranslator.translate(seq)
print(protein.to_string())   # MVHLTPEEKS...
```

### Detect mutations between two sequences

```python
from aligner import SequenceAligner, format_alignment

# seq_a = reference allele, seq_b = variant
result = SequenceAligner.align(seq_a, seq_b)

print(f"Identity: {result.identity:.1%}")
print(format_alignment(result))

for mut in result.mutations:
    print(mut)
# SUB  a[19]='A' → b[19]='T'   ← sickle cell mutation
```

### Run a full benchmark (30M bases)

```bash
python tools/stress_test.py
```

---

## Project structure

```
biocore.py              Level 1 — 5-bit storage engine
smart_translator.py     Level 2 — DNA → protein translation
aligner.py              Level 3 — pairwise alignment + mutation detection

tools/
  visor.py              interactive step-by-step translator
  stress_test.py        30M-base performance benchmark

tests/
  test_biocore.py       property-based tests (Hypothesis)

docs/
  architecture.md       full design notes and rules
  api_reference.md      code examples for every module
  benchmarks.md         measured numbers and corrections
  roadmap.md            status and planned extensions
```

---

## How the vectorisation works

### Translation (Level 2)

Instead of looping over every codon:

```
① Decode PackedSequence  →  uint8 array
② sliding_window_view    →  find first ATG (no loop)
③ reshape to (N, 3)      →  codon matrix
④ idx = n1×16 + n2×4 + n3  →  base-4 index (vectorised)
⑤ CODON_LUT[idx]         →  amino acid array (single fancy-index)
⑥ argmax on STOP hits    →  truncate at stop codon
```

### Alignment (Level 3)

Needleman-Wunsch has a data dependency that prevents full 2D vectorisation.
The solution: **anti-diagonal wavefront**.

Cells on the same anti-diagonal (`i + j = d`) are independent of each other,
so each diagonal is computed as a single NumPy operation.  
This reduces Python-level iterations from O(m·n) to **O(m+n)**.

---

## Running the tests

```bash
# Full test suite
pytest tests/ -v

# Benchmarks only
pytest tests/ --benchmark-only
```

---

## Roadmap

- [x] Level 1 — 5-bit storage, FASTA parser
- [x] Level 2 — vectorised genetic code translation
- [x] Level 3 — Needleman-Wunsch alignment + mutation detection
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
