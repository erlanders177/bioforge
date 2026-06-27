# Changelog

All notable changes to BioForge are documented here.  
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) · Versioning: [SemVer](https://semver.org/).

---

## [2.0.1] — 2026-06-27

Correcciones encontradas en una auditoría completa del código tras v2.0.0.

### Fixed

- **Registros vacíos truncaban el archivo** (`engine.c`, `_parse_one`): un registro
  FASTA/FASTQ sin secuencia hacía que el parser devolviera `0`, indistinguible del
  fin de archivo. Resultado: un registro vacío al inicio de un lote (en el peor
  caso, el primero del fichero) **detenía la lectura y descartaba el resto**.
  Ahora los registros vacíos se **saltan**; `0` solo significa EOF real.
- **FASTQ malformado (calidad ≠ longitud de secuencia)** provocaba un
  `ValueError` críptico al hacer `reshape` en la ruta columnar de longitud fija
  (`biocore.py`, `_stream_columnar`). Ahora se detecta el descuadre y se usa la
  ruta irregular, sin fallo.

### Performance

- **`bio_unpack5` ahora es seguro en los límites** (`engine.c`): se eliminó la
  copia completa del array empaquetado que `c_unpack5` hacía en **cada** llamada
  para un "byte de seguridad". Afecta a toda la ruta de `decode()` — alineador,
  traductor, GC/k-meros irregulares. Unpack ≈ 229 M símbolos/s.
- **Copia de cabeceras** (`biocore.py`): el streaming/columnar copiaba los 2 MB
  completos del buffer de cabeceras por lote; ahora usa `ctypes.string_at` y
  copia solo los bytes realmente usados.
- **GC + k-meros comparten una sola decodificación** por lote (`_decode_cached`):
  llamar a `gc_content()` y `kmer_spectrum()` sobre el mismo lote ya no
  desempaqueta dos veces.

### Tests
- 275 tests (desde 269): 6 nuevos de regresión para registros vacíos (FASTA/FASTQ,
  en medio y como primero) y FASTQ con calidad de longitud incorrecta.

---

## [2.0.0] — 2026-06-27

Versión centrada en **velocidad de ingesta**: el objetivo es procesar secuencias
más rápido que la célula que las produce. El cuello de botella ya no es leer y
codificar (eso vive en C), sino fabricar objetos Python por registro — y la API
columnar lo elimina para los flujos de control de calidad.

### Added

**Parser de streaming en C (`engine/engine.c`)**
- `bio_parser_open` / `bio_parser_next` / `bio_parser_close`: parser FASTA/FASTQ
  con buffer de 64 KB, `memchr` (SIMD de la libc) para saltos de línea, y
  codificación a BioCode 5-bit **dentro de C** — la secuencia nunca pasa por un
  `str` de Python
- `SmartImporter.stream(path)` — generador FASTA de RAM constante
- `SmartImporter.stream_fastq(path)` — generador FASTQ; produce `FastqRecord`
  (secuencia 5-bit + calidades Phred 0–93 ya decodificadas)
- `FastqRecord` con `mean_quality` y `passes_quality(min_q)`

**Parser por lotes en C (`bio_parser_next_batch`)**
- Una sola llamada parsea hasta 8 192 registros y empaqueta cada secuencia a
  5-bit en C, devolviendo buffers contiguos + tablas de offset
- Elimina los dos cuellos de botella medidos: el peaje de `ctypes` por registro
  y el `pack` de NumPy por registro
- Stash interno para registros que no caben en el lote (se emiten en la
  siguiente llamada)
- FASTA: **20.8 → 80 M bases/s** (3.8×). FASTQ: **2.1 → 14 M bases/s, 14 K → 94 K
  lecturas/s** (6.7×)

**API columnar (`biocore.py`)**
- `SequenceBatch` / `ReadBatch` — un lote de registros como matrices contiguas,
  sin un objeto Python por registro
- `SmartImporter.stream_batches(path)` (FASTA) / `stream_fastq_batches(path)` (FASTQ)
- `ReadBatch.mean_quality()`, `passes(min_q)`, `filter(mask)` — vectorizados
  sobre todo el lote; caso Illumina (longitud fija) usa una matriz 2-D limpia,
  caso Nanopore (irregular) usa `reduceat` sobre offsets
- Acceso perezoso: `batch[i]` materializa un `PackedSequence`/`FastqRecord` solo
  cuando se pide
- **Filtrar 200 000 lecturas por calidad media: 5.3 s → 0.28 s (18.6×)**,
  resultado idéntico al filtrado por registro
- Fallback en Python puro (`_columnar_fallback`) si el motor C por lotes no está

**Composición vectorizada en los lotes (`biocore.py`)**
- `ReadBatch.gc_content()` / `SequenceBatch.gc_content()` — fracción GC por
  registro; una sola `unpackbits` para todo el lote cuando la longitud es fija
- `ReadBatch.kmer_spectrum(k)` / `SequenceBatch.kmer_spectrum(k)` — espectro de
  k-meros del lote (`int64`, longitud `4**k`); k-meros con bases ambiguas
  descartados; vectorizado con `sliding_window_view` + `bincount`
- `SequenceBatch` lanza `SequenceTypeError` si se piden GC/k-meros sobre proteínas

**Lectura de archivos comprimidos (`engine/engine.c`)**
- El parser lee `.gz` de forma transparente vía zlib (`gzopen`/`gzread`): el
  mismo código sirve para archivos planos y comprimidos (autodetección del
  magic gzip). `stream("x.fastq.gz")`, `stream_fastq(...)`, etc. funcionan sin
  paso de descompresión manual
- Compilación condicional `-DBIO_USE_ZLIB`: si zlib no está, se compila sin él
  y los archivos planos siguen funcionando. En Windows zlib se enlaza **estático**
  (`-l:libz.a`) → el DLL es autocontenido, sin dependencia de `zlib1.dll`

**Detección de capacidades del motor (`engine/_loader.py`)**
- Banderas separadas `C_PARSER_AVAILABLE` y `C_BATCH_AVAILABLE`: un DLL antiguo
  sin las funciones nuevas degrada con gracia en vez de fallar

**Empaquetado e instalación**
- `pyproject.toml` actualizado: versión dinámica desde `bioforge.__version__`
  (fuente única), backend estándar `setuptools.build_meta`, y el motor C
  (`*.dll`/`*.so`/`*.c`) se incluye en el wheel vía `package-data`
- `build.py` detecta GCC automáticamente (incl. ruta típica de MSYS2) e intenta
  enlazar zlib, con fallback sin zlib si no está

**Benchmark contra Biopython (`tools/bench_vs_biopython.py`)**
- Mide tiempo y RAM pico (aislamiento por subproceso) en parsing, QC y carga
  total. Resultados medidos (300 000 lecturas × 150 bp):
  - QC (filtrar por calidad media): **~5–6× más rápido**, resultado idéntico
  - Cargar todo en RAM: **~6.9× menos memoria** (115 MB vs 801 MB) y **~9.5×
    más rápido** — aquí pesa el almacenamiento 5-bit

### Tests
- 269 tests (desde 239): `tests/test_streaming.py` añade 30 tests del parser
  streaming/batch, la API columnar, GC, k-meros, `.gz` y rutas de error —
  correctitud frente a `from_file` y a referencias ingenuas, longitud fija e
  irregular, calidades Phred exactas, `filter()` y descarte de bases ambiguas

---

## [1.1.1] — 2026-06-27

### Fixed

- `__init__.py`: `__version__` reported `"1.0.0"` instead of the correct version
- `aligner.py`: C engine imports were unconditional — if the `engine/` directory were missing, `aligner.py` would crash with `ImportError` instead of falling back to NumPy (inconsistent with `biocore.py` which used `try/except`)
- `engine.c`: Semi-global NW (`nw_semiglobal`) only searched the last column for the best traceback start; it now searches both the last row and the last column, matching the NumPy fallback behaviour

---

## [1.1.0] — 2026-06-27

### Added

**Reverse complement vectorised (`biocore.py`)**
- `PackedSequence.reverse_complement()` — applies Watson-Crick pairing (A↔T/U, C↔G) and reverses the sequence
- Implemented as two NumPy ops: `_NUC_COMPLEMENT` LUT + `np.flip`; zero Python loops
- Result header prefixed with `[RC]`; raises `SequenceTypeError` for protein input
- RC(RC(x)) == x guaranteed for all nucleotide sequences

**6-frame translation (`smart_translator.py`)**
- `SmartTranslator.translate_all_frames(seq)` — translates all 6 reading frames (+1/+2/+3/-1/-2/-3)
- Returns `list[PackedSequence]` — one entry per frame that contains an ATG, frames without ORF skipped silently
- Header format: `[PROT | frame +1 | ORF@N] <original_header>`
- Optional `warn_short=False` to suppress short-protein warnings

**Smith-Waterman local alignment (`aligner.py`)**
- `SequenceAligner.align_local(seq_a, seq_b)` — finds the highest-scoring local sub-region
- Returns `AlignmentResult` with `mode='local'`
- Score floored at 0; traceback stops when H cell reaches 0
- C engine path via `sw_align()` in `engine.c`; NumPy anti-diagonal wavefront fallback

**Banded Needleman-Wunsch (`aligner.py`)**
- `SequenceAligner.align(seq_a, seq_b, band=N)` — restricts DP to ±N cells around the main diagonal
- C engine: true banded storage O(m·N) via `nw_banded()` / `nw_banded_semiglobal()` in `engine.c`
- NumPy fallback: full matrix with NEG_INF masking outside band — same result, standard RAM
- `band=0` raises `AlignmentError`

**C engine additions (`engine/engine.c`, `engine/_loader.py`)**
- `sw_align()`: Smith-Waterman with calloc zeros, floor-at-0 fill, max-cell traceback
- `nw_banded()` / `nw_banded_semiglobal()`: banded NW with macros `_BH(i,k)` / `_BTB(i,k)`, W=2·band+1
- Python wrappers `c_sw_align()` and `c_nw_banded()` added to `_loader.py`

**Bug fix**
- `SmartTranslator._find_orf_start()`: was raising bare `ValueError` instead of `TranslationError`; fixed so `BioForgeError` catches it

**Tests**
- 239 tests passing (up from 172 in v1.0.0)
- 10 new tests for `reverse_complement()` (correctness, round-trip, palindromes, error paths)
- 8 new tests for `translate_all_frames()` (all frames, no-ATG, strand detection, error paths)
- 13 new tests for `align_local()` Smith-Waterman (mode, identity, local region, errors)
- 6 new tests for banded NW (coherence with full NW, error paths)

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

- **v1.1** — Reverse complement (vectorised) · 6-frame translation ✓
- **v1.2** — Banded Needleman-Wunsch for sequences > 15 000 bp ✓
- **v2.0** — Streaming/batch parser in C · columnar API for QC ✓
- **futuro** — API columnar 100% sin objetos (k-meros vectorizados, GC por lote);
  lectura de FASTQ comprimido (gzip) en C; SIMD AVX2 en pack/unpack
