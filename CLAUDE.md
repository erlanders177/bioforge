# BioForge — Contexto para Claude Code

## Qué es este proyecto

BioForge: motor bioinformático de alto rendimiento para Edge Computing (hardware limitado).
Sin Biopython. NumPy core + motor C opcional (ctypes). Python 3.13, Windows 10.
Es un paquete instalable: `from bioforge import ...` (versión actual **2.2.0**).

Niveles implementados y validados:
- **L1** `bioforge/biocore.py` — almacenamiento 5-bit, LUTs, BitPacker, PackedSequence, SmartImporter
- **L2** `bioforge/smart_translator.py` — traducción ADN→Proteína vectorizada (CODON_LUT + sliding_window_view); 6-frame + reverse complement
- **L3** `bioforge/aligner.py` — Needleman-Wunsch wavefront (global/semi-global), banded NW, Smith-Waterman local
- **Ingesta v2.0** `bioforge/engine/engine.c` + `biocore.py` — parser FASTA/FASTQ en C (streaming + por lotes), API columnar, `.gz`

Motor C en `bioforge/engine/engine.c` (compilado a `engine.dll`/`.so`), cargado vía
ctypes con fallback NumPy transparente. Documentación detallada en `docs/`.

---

## Reglas de oro — OBLIGATORIAS en todo el código del motor

### 1. Cero loops Python en la ruta crítica
**Prohibido** en `biocore.py`, `smart_translator.py`, y `_fill_matrix` de `aligner.py`:
- Cualquier `for` o `while` que itere **símbolo a símbolo o celda a celda**.

**Obligatorio:** operaciones NumPy vectorizadas — fancy indexing, `packbits`, `unpackbits`,
`where`, `sliding_window_view`, `argmax`, `bincount`, etc.

**Aclaración (v2.0):** los bucles **por registro** (no por símbolo) SÍ están permitidos
—p.ej. el streaming/columnar de `SmartImporter` itera registros, y `ReadBatch.filter`
en el caso irregular itera supervivientes. Todo el trabajo por símbolo (parse, encode,
pack, GC, k-meros) ocurre en C o en una sola op NumPy sobre el lote.

### 2. Excepciones conocidas y aceptadas
- `visor.py` — loops permitidos (frontend de display, no procesamiento)
- `aligner._traceback` — loop O(m+n) permitido (dependencia de datos inevitable)
- `aligner._fill_matrix` — UN loop O(m+n) sobre anti-diagonales (no O(m·n))

### 3. Nunca almacenar secuencias como str
Las secuencias biológicas existen únicamente como `PackedSequence` con `data` uint8
write-locked. El único lugar donde existe un `str` de secuencia es dentro de
`SmartImporter._encode()` como variable local temporal.

### 4. Force_type para proteínas sin marcadores obvios
Si una proteína no tiene E/F/I/L/P/Q/* en su secuencia, la auto-detección la
clasificará silenciosamente como ADN. Usar siempre `force_type=SeqType.PROTEIN`.

### 5. Benchmark después de cada optimización
Ejecutar `python tools/stress_test.py` antes y después de cualquier cambio en el
motor para verificar que no empeora RAM ni velocidad. Para la ingesta, comparar
con `python tools/bench_vs_biopython.py`.

### 6. Recompilar el motor C tras tocar engine.c
`python bioforge/engine/build.py` (autodetecta GCC, incl. MSYS2 en
`C:\msys64\mingw64\bin\gcc.exe`). Enlaza **estático**: OpenMP (libgomp), zlib y
libdeflate DENTRO del DLL → motor autocontenido, sin dependencias de runtime.
Degrada con gracia si falta libdeflate (solo zlib) o zlib (sin `.gz`). El DLL
compilado se versiona en git para que el usuario no necesite GCC.

### 7. Procesamiento multinúcleo (v2.2) — despachador adaptativo
`stream_batches`/`stream_fastq_batches` aceptan `n_threads` (1=secuencial RAM
constante; >1=hilos; 0=todos los núcleos). El motor enruta: plano→parseo paralelo
(OpenMP, mmap sin copia); `.gz`→libdeflate (~2×) + parseo; fallback a zlib
secuencial. El parseo paralelo está limitado por ancho de banda de memoria
(poco en pocos núcleos); el win real es libdeflate en `.gz` (~1.6× end-to-end).
**BGZF (palanca 3):** `.gz` por bloques independientes → descompresión
paralela (~1.95×, la vía más rápida). Conversor `bioforge-bgzip` (bgzf.py);
salida compatible con gunzip. El despachador detecta BGZF (subcampo `BC`) y
enruta; `.gz` normal → libdeflate 1 hilo.

### 8. Sistema de errores unificado — todo bajo BioForgeError
Cualquier fallo del motor DEBE lanzar una subclase de `BioForgeError`, para que
`except BioForgeError` lo capture todo. Cada subclase hereda además del builtin
estándar adecuado (compatibilidad). Jerarquía (en `biocore.py`):
`SequenceTypeError`(+TypeError), `SequenceValueError`/`TranslationError`/
`AlignmentError`(+ValueError), `BioForgeIOError`(+OSError, apertura de archivo),
`EngineError`(+RuntimeError, parser/(de)compresión/BGZF). Los errores de
**uso/argumento** (p.ej. `mode` inválido, salida==entrada) sí pueden ser
`ValueError` plano, como ya hace `analyze.py`.

---

## Números correctos del proyecto

| Métrica | Valor correcto |
|---------|---------------|
| Ahorro de memoria (5-bit) | **37.5%** (memory_ratio = 0.625) |
| RAM para 30M bases | **18.75 MB** |
| Throughput traducción | **~5 M aa/s** |
| Benchmark alineador 1000×1000 nt | **~165 ms** |
| Ingesta FASTA (parser C por lotes) | **~80 M bases/s** |
| Ingesta FASTQ (parser C por lotes) | **~14 M bases/s · ~94 K lecturas/s** |
| Filtrar 200K lecturas por calidad (columnar) | **~0.28 s** (18.6× vs por registro) |
| vs Biopython — cargar todo en RAM | **~6.9× menos RAM** (115 vs 801 MB), ~9.5× más rápido |
| Leer FASTQ `.gz` (libdeflate + paralelo, n_threads≠1) | **~89 M bases/s** (1.59× vs zlib) |
| Descompresión gzip libdeflate vs zlib | **2.15×** (379 vs 176 MB/s) |
| Leer FASTQ **BGZF** (descompresión paralela) | **~113 M bases/s** (~1.95× vs baseline) |

⚠️ El resumen ejecutivo original cita "60-70%" — ese número es incorrecto.
Correspondería a 2-bit packing, no al esquema 5-bit implementado.

---

## Estructura de archivos

```
bioforge/                  paquete instalable (from bioforge import ...)
  __init__.py              API pública + __version__ (fuente única de versión)
  biocore.py               L1 — almacenamiento 5-bit, SmartImporter, FastqRecord,
                           SequenceBatch/ReadBatch (API columnar) — no tocar sin impacto global
  smart_translator.py      L2 — traducción ADN→Proteína, 6-frame, reverse complement
  aligner.py               L3 — NW global/semi-global, banded, Smith-Waterman
  analyze.py               pipeline CLI (dna/protein/both)
  qcreport.py              informe de calidad FASTQ (tipo FastQC, columnar) — CLI bioforge-qc
  bgzf.py                  conversor a BGZF (gzip por bloques, paralelo) — CLI bioforge-bgzip
  engine/
    engine.c               motor C — pack/unpack, NW/SW, parser FASTA/FASTQ + batch + .gz
    engine.dll             binario compilado (versionado en git)
    _loader.py             ctypes + banderas C_AVAILABLE/C_PARSER_AVAILABLE/C_BATCH_AVAILABLE
    build.py               compila el DLL/SO (autodetecta GCC, enlaza zlib)
tools/
  visor.py                 frontend interactivo (loops permitidos aquí)
  comparador.py            comparador de secuencias (CLI)
  stress_test.py           benchmark de 30M bases
  bench_vs_biopython.py    BioForge vs Biopython (tiempo + RAM)
tests/                     test_biocore / _translator / _aligner / _analyze / _streaming
docs/                      architecture · api_reference · benchmarks · roadmap
pyproject.toml             empaquetado (versión dinámica, incluye el DLL en el wheel)
```

---

## Limitaciones conocidas del estado actual (v2.0)

- Alineador: solo viable para secuencias ≤ 15 000 símbolos (O(m·n) RAM; usar `band=N`)
- Auto-detección de tipo: puede fallar en proteínas sin residuos exclusivos (usar `force_type`)
- API columnar: `batch[i]` materializa un objeto; el modo 100% sin objetos solo cubre
  por ahora GC y k-meros
- Wheel PyPI `py3-none-any` con DLL de Windows: en otras plataformas cae a fallback
  NumPy o requiere recompilar. Faltan wheels nativos por plataforma (cibuildwheel)

---

## Próximas extensiones priorizadas (post-v2.0)

1. Wheels nativos por plataforma (cibuildwheel) para `pip install bioforge` real
2. Columnar 100% sin objetos en más operaciones (la "frontera taquión")
3. SIMD AVX2 explícito en pack/unpack y `bio_find_atg`

---

## Compatibilidad con consola Windows

Añadir al inicio de cada bloque `if __name__ == "__main__"`:
```python
import sys
sys.stdout.reconfigure(encoding="utf-8")
```
La consola Windows usa cp1252 y no puede mostrar caracteres ═ ─ sin esto.
