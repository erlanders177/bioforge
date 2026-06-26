# Changelog

All notable changes to BioForge are documented here.  
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) · Versioning: [SemVer](https://semver.org/).

---

## [1.0.0] — 2026-06-26

First stable release.

### Added

**L1 — Core storage engine (`biocore.py`)**
- Unified 5-bit biological alphabet (32 states): nucleotides 0–3, amino acids 4–23, STOP 24, GAP 25, UNK 31
- `BitPacker`: vectorised 5-bit pack/unpack — zero Python loops, NumPy + optional C backend
- `PackedSequence`: immutable, write-locked sequence container with O(1) random access and memory ratio 0.625
- `SmartImporter`: FASTA parser with auto-detection (NUCLEOTIDE vs PROTEIN), multi-record and chunked file reading
- `BioCode` and `SeqType` enumerations
- `compute_stats()`: composition, GC content, length statistics

**L2 — Translator (`smart_translator.py`)**
- `SmartTranslator.translate()`: DNA/RNA → Protein using the Standard Genetic Code (NCBI table #1)
- ATG/AUG detection via `sliding_window_view` — no Python loop
- ORF extraction and STOP codon truncation
- ~5 M amino acids/second (NumPy) · ~27× faster with C engine

**L3 — Aligner (`aligner.py`)**
- `SequenceAligner.align()`: global and semi-global Needleman-Wunsch
- Anti-diagonal wavefront: O(m+n) Python iterations instead of O(m·n)
- Linear gap model: match +2, mismatch −1, gap −2
- `format_alignment()`: human-readable block alignment with match/mismatch/gap symbols
- Mutation detection: substitutions, insertions, deletions with positions and symbols
- ~165 ms for 1000×1000 nt on NumPy · ~29× faster with C engine

**Pipeline (`analyze.py`)**
- Three modes: `dna` · `protein` · `both`
- Conservative vs radical amino acid substitution classification
- Silent (synonymous) mutation detection and labelling
- Full text report via `build_report()`
- CLI: `python analyze.py reference.fa query.fa [--mode dna|protein|both] [--output file]`

**C engine (`engine/engine.c`)**
- `bio_pack5`, `bio_unpack5`, `bio_getitem5`: 5-bit storage operations
- `bio_find_atg`: linear scan for first ATG codon
- `bio_translate`: full codon→amino acid translation
- `nw_global`, `nw_semiglobal`: complete NW alignment in C with OpenMP
- Compiled with GCC -O3 -march=native -fopenmp
- Auto-loaded via ctypes · transparent NumPy fallback if not compiled
- `engine/build.py`: cross-platform build script (Windows .dll / Linux·Mac .so)

**Error handling**
- `BioForgeError` base exception — catch all engine errors in one `except` clause
- `SequenceTypeError` · `SequenceValueError` · `TranslationError` · `AlignmentError`
- All subclasses inherit from both `BioForgeError` and the matching standard exception (backwards-compatible)
- Actionable error messages with recovery suggestions

**Tools**
- `visor.py`: interactive step-by-step DNA→Protein translator (CLI)
- `comparador.py`: sequence comparator with alignment report (CLI)
- `check.py`: non-programmer verifier — runs all checks automatically

**Tests**
- 209 tests passing across all modules
- Hypothesis property-based tests (round-trip, mathematical properties)
- pytest-benchmark (pack, unpack, translate, align at various sizes)
- Full pipeline integration tests
- Error-path tests for all public API entry points
- Exception hierarchy tests

**Documentation**
- `docs/architecture.md`: design rules, vectorisation strategy, encoding details
- `docs/api_reference.md`: code examples for every public API
- `docs/benchmarks.md`: measured performance numbers and methodology
- `docs/roadmap.md`: status, known limitations, planned extensions

---

## Roadmap — planned for future releases

- **v1.1** — Reverse complement (vectorised) · 6-frame translation
- **v1.2** — Banded Needleman-Wunsch for sequences > 15 000 bp
- **v2.0** — Package restructure (`from bioforge import ...`)
