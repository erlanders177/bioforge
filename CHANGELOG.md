# Changelog

All notable changes to BioForge are documented here.  
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) · Versioning: [SemVer](https://semver.org/).

---

## [6.2.0] — 2026-07-09

**Extensión SIMD int16 (16 carriles) — recorta el gap de 1 hilo.** El kernel
antidiagonal gana una variante de 16 bits que procesa el doble de celdas por
instrucción cuando los scores caben (reads ≤ 12 000 bp); reads más largos siguen
en int32. Multinúcleo sigue a la par de minimap2.

### Added
- **`_nw_banded_diag_simd_i16`** — versión AVX2 int16 (16× carriles) del kernel
  banded antidiagonal, con inversión de 16 int16 (shuffle por-carril + swap de
  mitades). Un dispatcher enruta por tamaño: int16 (m,n ≤ 12000) → int32 (8×) →
  escalar (sin AVX2). Bit-idéntica al escalar (mismo DP y empate).

### Performance (honesto — WSL, 4.8 Mb, 6000 reads, 5% error, minimap2 -a)
- Kernel extensión 2000×2000: **1.42×** sobre int32 (el 2× teórico se diluye por
  costes fijos: malloc, traceback, bordes, que no vectorizan).
- 1 hilo: BioForge ~1.87 vs minimap2 ~2.2 Mb/s → **~1.18×** por detrás (antes ~1.3×).
- 4 núcleos: **a la par / por delante** (~4.3-5.0 vs ~4.3-4.9). Ambos mapean 6000.
- El resto del gap de 1 hilo se reparte entre seeding (~25%) y chaining (~34%);
  cerrarlo del todo es rendimiento decreciente.

### Tests
- Paridad kernel int16↔core **0/muchos** (incl. N, bordes, banda estrecha);
  **valgrind 0 errores / 0 fugas** sobre el int16 (WSL, con su inversión de 16
  carriles). **361 tests**.

---

## [6.1.0] — 2026-07-09

**Salida columnar en `map_batch` — el multinúcleo alcanza a minimap2.** Se elimina
la cola serial de Python que reconstruía los `Mapping`, así el escalado ya no se
capa. En 4 núcleos, BioForge queda **a la par de minimap2** (dentro del ruido, a
veces por delante) en el benchmark de referencia.

### Changed
- `bio_map_batch` (vía `c_map_batch`) escribe ahora en un **array estructurado
  NumPy** (mismo layout que el struct C `MapOut`, verificado offset a offset) en
  vez de un array ctypes. La cubierta Python construye los `Mapping` leyendo cada
  campo como **columna** (`.tolist()`, C-level) — sin acceso ctypes campo a campo
  ni dicts intermedios. Resultado **idéntico** (verificado por `test_cmap_parity`).

### Benchmark (WSL, 4.8 Mb, 6000 reads, 5% error, minimap2 -a)
- 4 núcleos: minimap2 ~4.0-5.7 vs BioForge ~4.2-4.7 Mb/s → **a la par** (el motor
  C puro daba 4.85; la cola serial lo bajaba a ~3.8, ahora ~4.4-4.7).
- 1 hilo: ~1.2-1.3× por detrás aún (minimap2 ~2.2 vs BioForge ~1.8 Mb/s).
- Ambos mapean los 6000. A escala E. coli; a escala mayor minimap2 puede separarse.

### Tests
- 361 tests; `map_batch` idéntico con 1/2/3/4/0 hilos y == `map()` secuencial.

---

## [6.0.0] — 2026-07-09

**Extensión SIMD + escalado multinúcleo — el mapeador se vuelve competitivo.**
Head-to-head medido contra minimap2 en la misma máquina (WSL): de ~4× por debajo
a **~1.3×**, tanto en 1 hilo como en 4 núcleos, mapeando lo mismo. Sin promoción
todavía (medición honesta, no titular).

### Added
- **Extensión banded vectorizada con AVX2** (`_nw_banded_diag_simd`): el DP se
  recorre por antidiagonales (celdas independientes) y procesa **8 celdas int32
  por instrucción**. El kernel pasa de **88 a 529 M celdas/s (6×)**; la extensión
  es el 88% del tiempo de mapeo → **~4× en 1 hilo** en el mapeador completo.
  Bordes en escalar, desempate `diag>up>left` replicado exacto → salida
  **bit-idéntica** al kernel escalar. Fallback escalar automático si no hay AVX2.

### Fixed
- **Escalado multinúcleo real en `bio_map_batch`:** el nº de hilos de OpenMP no se
  reseteaba, así que tras una llamada a 1 hilo las siguientes se quedaban en 1 →
  `map_batch` parecía no escalar. Ahora se fija siempre (`n<=0` → todos los
  núcleos). Efecto: **~2.3× en 4 núcleos**.

### Benchmark (honesto — WSL, 4.8 Mb, 6000 reads, 5% error, minimap2 -a)
- 1 hilo: minimap2 ~2.4 vs BioForge ~1.8 Mb/s (**~1.3×**).
- 4 núcleos: minimap2 ~4.0 vs BioForge ~3.0 Mb/s (**~1.3×**). Ambos mapean 6000.
- A escala E. coli. A escala genoma humano / millones de reads minimap2 podría
  separarse más (no medido). Queda una cola serial de reconstrucción `Mapping` en
  Python (el motor C puro ya da ~4.85 Mb/s) → margen para igualar en multinúcleo.

### Tests
- Paridad SIMD↔escalar **0/10 000** (incl. banda patológica); **valgrind 0
  errores / 0 fugas** sobre el kernel SIMD (WSL); `map_batch` idéntico con
  1/2/3/4/0 hilos y == `map()` secuencial. Nueva herramienta
  `tools/bench_vs_minimap2.py`. **359 tests**.

---

## [5.0.0] — 2026-07-08

**La tubería de mapeo entera en C, tras un índice opaco.** El motor deja de
orquestar el mapeo desde Python: ahora seed-chain-align corre completo en C y
Python es solo una cubierta fina. Es el paso arquitectónico que pedía el
roadmap; la velocidad *aún* no rivaliza con minimap2 (ver nota honesta abajo).

### Added
- **Índice opaco de la referencia en C** (`bio_index_build`). Se construye una
  sola vez y retiene la tabla de minimizers ordenada por hash (búsqueda binaria),
  una copia de los codes de la referencia (para la extensión) y las fronteras de
  contigs. Python solo guarda el handle → sin re-serializar el índice por consulta.
- **`bio_map_read`** — pipeline entero de un read en una sola llamada C:
  minimizers → lookup → chaining (DP + backtrack + supresión de solapamientos) →
  extensión banded del read completo → `Mapping`. Réplica fiel del camino Python.
- **`bio_map_batch`** — mapea un lote de reads en paralelo con **OpenMP**, sin
  GIL y sin coste de procesos. `GenomeAligner.map_batch` lo usa cuando hay motor C
  (antes: `multiprocessing`, que exigía guard `if __name__ == "__main__"`).

### Changed
- `GenomeAligner` construye el índice C en `__init__` y enruta `map`/`map_batch`
  por C, con **fallback NumPy transparente** (idéntico, verificado). Handle liberado
  en `__del__`; excluido del pickling (`__getstate__`) por ser un puntero crudo.

### Tests
- Paridad **exacta** del camino C contra el pipeline NumPy, campo por campo
  (coords, CIGAR, identidad, mapq) en 200+ reads directos/inversos/mutados,
  multi-contig, copias múltiples y bordes. +tests del índice C. **354 tests**.

### Rendimiento (honesto — sin promocionar)
- El pipeline entero en C da **~1.7× en 1 hilo** (el DP banded de la extensión
  *ya* estaba en C: es el 88% del tiempo). `map_batch` escala ~2.3× en 4 núcleos.
- Sigue **~30-50× por debajo de minimap2 en 1 hilo** (~0.8 vs ~20-40 Mb/s). El
  muro es la alineación base a base escalar. El siguiente paso (v6.0) es **SIMD**
  (KSW2/SSE) sobre ese DP — el 8-16× que falta. **No se promociona velocidad.**

---

## [4.0.1] — 2026-07-08

**Robustez y sistema de errores del mapeador** (antes de llevar la tubería a C).

### Fixed
- **La extensión ya no inventa `identity=1.0`** cuando la alineación falla:
  devuelve "sin mapeo" (un 100% falso es peor que no reportar nada). El
  `except` se acota a `(AlignmentError, MemoryError)` — antes capturaba
  cualquier excepción y ocultaba bugs. Más holgura en la banda de la extensión.

### Changed
- **Sistema de errores unificado en el mapeador (regla #8):** las entradas
  inválidas lanzan subclases de `BioForgeError` en vez de errores crudos de
  Python. `SequenceTypeError` (read no-`str`, referencia de tipo inválido) y
  `SequenceValueError` (referencia vacía / sin contigs).

### Tests
- +tests de robustez: reads en los extremos del genoma, que sobresalen, y
  degenerados (vacío / < k / todo N); jerarquía de errores capturable como
  `BioForgeError`. Limpieza de linter en la suite. 344 tests.

---

## [4.0.0] — 2026-07-08

**El mapeador de genomas, usable en datos reales.** Referencias
multi-cromosoma y alineación del read completo.

### Added
- **Referencia multi-cromosoma.** `GenomeAligner` acepta una cadena (un contig)
  **o** un `dict {nombre: secuencia}` / iterable de `(nombre, secuencia)` para
  varios cromosomas/plásmidos/contigs. Los contigs se concatenan con
  separadores `N` (los minimizers los excluyen → ningún k-mer cruza fronteras).
  Los mapeos reportan el contig (`Mapping.target_name`) y **coordenadas
  locales**; el PAF sale con el nombre correcto. `GenomeAligner.n_contigs`.

### Changed
- **Extensión del read completo.** La alineación cubre ahora todo el read (antes
  solo la región de la cadena): mejor cobertura, posición exacta, y soft-clip
  natural si el read sobresale por un borde del contig. `Mapping` gana el campo
  `target_name`.

### Notes
- **Honestidad de velocidad:** correcto y funcional, pero el mapeo **aún no
  compite en velocidad con minimap2** (~1–2 órdenes de magnitud más lento; el
  índice sí es rápido). El siguiente gran paso es llevar toda la tubería de
  mapeo a C. No se promociona como rival de velocidad todavía.

---

## [3.4.0] — 2026-07-07

**Coste de hueco del chaining, fórmula de minimap2 (4/6).**

### Changed
- La penalización por hueco del chaining pasa a `γ(l) = 0.01·k·|l| + 0.5·log₂|l|`
  (Li 2018), en vez de la aproximación anterior (`0.2·gap + log₂(gap+1)`). Mejora
  la precisión al encadenar anclas con indels. Aplicado por igual en el motor C
  y en el fallback NumPy → siguen dando el mismo resultado (verificado, f_diff 0).

---

## [3.3.0] — 2026-07-07

**Minimizers en C (3/6).** El cálculo de minimizers —el otro punto O(n·w) que
quedaba en NumPy— pasa al motor C con hash rodante.

### Performance
- Indexar/muestrear una secuencia: **~17× más rápido** en 2 Mb (1219 → 70 ms).
  Es lo que hacía falta para indexar genomas grandes sin ahogarse en memoria y
  tiempo (`sliding_window_view` materializaba una vista (n, w)).

### Added
- `bio_minimizers` en el motor C (`C_MINIMIZERS_AVAILABLE`, `c_minimizers`).
  Réplica exacta del cálculo NumPy (mismo hash, canónico y desempate) → C ==
  fallback verificado con y sin bases N. `minimizers()` usa C si está; si no,
  `_minimizers_numpy` (idéntico).

---

## [3.2.0] — 2026-07-07

**Mapeo por lotes en paralelo (2/6).**

### Added
- `GenomeAligner.map_batch(reads, n_processes=0)` → mapea muchos reads en
  paralelo (0 = todos los núcleos · 1 = secuencial · N = N procesos). Usa
  **procesos** (multiprocessing): con hilos el GIL no deja escalar porque el
  trabajo por read es mayoritariamente Python (medido: 1.0× con hilos). El
  índice se pasa una vez a cada proceso; el orden de salida se conserva; cae a
  secuencial con gracia si el arranque de procesos falla.

### Performance
- ~**1.6×** en 4 núcleos (2000 reads de 1000 bp). Escala con más reads, reads
  más largos y más núcleos (el arranque de procesos se amortiza).

### Notes
- Requisito de multiprocessing: el script que llame a `map_batch` con
  `n_processes != 1` debe estar bajo `if __name__ == "__main__":`.

---

## [3.1.0] — 2026-07-07

**Optimización del mapeo (1/6).** El alineador puede saltar la detección de
mutaciones cuando no se necesita.

### Added
- `SequenceAligner.align(..., detect_mutations=True)` — por defecto `True`
  (compatible). Con `False`, no construye la lista de `Mutation` (identidad,
  score, CIGAR y matches quedan igual).

### Performance
- El mapeador de genomas usa `detect_mutations=False` en la extensión (no las
  necesita) → **~2,9 → ~2,3 ms/read** (~20% más rápido) en el micro-benchmark.

---

## [3.0.0] — 2026-07-07

**Level 4 — Genome mapper.** Un alineador de reads largos contra genomas al
estilo seed-chain-align (minimap2), que escala mucho más allá del DP O(m·n) del
Level 3. Implementación propia desde cero de algoritmos publicados (ver
`docs/references.md`).

### Added
- **`GenomeAligner.map(read)`** → mapeos en formato PAF, con hebra, identidad,
  CIGAR y calidad de mapeo. API pública: `GenomeAligner`, `Mapping`.
- **`minimizers.py`** — minimizers canónicos (w, k) vectorizados (hash invertible
  estilo minimap2; exclusión de N; independiente de hebra).
- **`refindex.py`** — índice de la referencia (tabla ordenada por hash +
  `searchsorted`; filtrado de minimizers hiper-frecuentes vía `max_occ`).
- **`genomemap.py`** — seeding (anclas en ambas hebras) → chaining (DP colineal)
  → extensión banded (reutiliza el aligner) → salida PAF.
- **Chaining DP en C** (`bio_chain_dp`) con fallback NumPy idéntico verificado.
- `docs/references.md` — citación de las obras que inspiraron cada nivel.

### Performance
- Mapeo ~**9,4× más rápido** que la primera versión del chaining tras
  vectorizar el bucle interno y portar el DP a C (14,5 → 2,9 ms/read en el
  micro-benchmark de reads de 1000 bp; C 5× sobre NumPy).

### Fixed
- Scoring de cadenas secundarias: se calcula el score real del fragmento y se
  suprimen solapamientos en el genoma (fragmentos redundantes del mismo locus).

### Notes
- Primera versión del Level 4: la extensión cubre la región de la cadena
  (extremos soft-clipped) y la referencia es una sola secuencia. El benchmark
  frente a herramientas de referencia y el mapeo por lotes en paralelo quedan
  para 3.x.

---

## [2.3.0] — 2026-06-30

**Wheels nativos multiplataforma.** El motor C ahora se compila para Windows,
Linux y macOS, así que `pip install bioforge` da el motor rápido en las tres
plataformas (antes solo Windows; el resto caía al fallback NumPy). Verificado en
CI: el motor C carga y traduce en cada sistema.

### Added
- Compilación de wheels por plataforma con **cibuildwheel** + GitHub Actions
  (`.github/workflows/wheels.yml`). Wheels `py3-none-<plataforma>`: un wheel por
  SO vale para todas las versiones de Python 3 (el motor es ctypes, no depende
  de la ABI de Python).
- `setup.py`: compila el motor al construir el wheel y lo etiqueta por
  plataforma; en Windows reutiliza el `engine.dll` precompilado.
- `tools/ci_check.py`: verificación de humo que falla si el motor C no carga.
- Herramientas de desarrollo: Ruff, mypy, pyright, pytest-cov, pre-commit
  (config en `pyproject.toml` y `.pre-commit-config.yaml`); Scalene y py-spy
  para perfilado (extra `[profile]`).

### Changed
- `build.py`: compilación **portátil** (`BIOFORGE_PORTABLE`, sin `-march=native`)
  para wheels que funcionan en cualquier CPU, y **autocontenida**
  (`BIOFORGE_STATIC`): OpenMP estático en Linux/macOS para no depender de
  librerías fuera de la lista blanca de manylinux. Soporte de macOS (clang +
  libomp).
- Limpieza con `ruff --fix`: imports muertos, f-strings, `zip(..., strict=True)`.

### Notes
- En Linux/macOS, los wheels llevan zlib (`.gz`) + OpenMP (paralelo); el extra
  de libdeflate (`.gz` ~2×) queda pendiente de compilarse desde fuente en CI.

---

## [2.2.1] — 2026-06-27

**Actualización de emergencia (hotfix).** Una auditoría línea por línea tras
publicar v2.2.0 encontró un **deadlock**; este parche lo corrige, más dos
arreglos de robustez y la unificación del sistema de errores.

### Fixed
- **Deadlock OpenMP (crítico)** en `bio_bgzf_decompress_parallel` y
  `bio_bgzf_compress`: el `#pragma omp for` estaba dentro de un `if`; si a un
  hilo le fallaba la reserva del (des)compresor, los demás quedaban colgados en
  la barrera para siempre. Movido el bucle fuera del `if`.
- **Conversor BGZF**: con entrada ya `.gz` y sin `-o`, el destino coincidía con
  la entrada y la sobrescribía (pérdida de datos). Ahora se rechaza out == in.
- **RAM en la vía rápida `.gz`**: guard de tamaño (>512 MB comprimido → ruta
  secuencial de RAM constante) para no agotar memoria con archivos enormes.

### Changed
- **Sistema de errores unificado**: toda la ruta de ingesta lanza ahora
  excepciones de la jerarquía `BioForgeError`, cumpliendo la promesa de "captura
  todos los errores del motor con un solo `except BioForgeError`". Dos nuevas:
  `BioForgeIOError(BioForgeError, OSError)` (apertura/lectura) y
  `EngineError(BioForgeError, RuntimeError)` (parser/(de)compresión). Ambas
  heredan también del builtin estándar, así que el código que ya atrapaba
  `OSError`/`RuntimeError` sigue funcionando.

### Tests
- 303 tests (+5): `tests/test_errors.py` — jerarquía completa, archivo inexistente
  (`BioForgeIOError`), informe QC vacío, y guard de sobrescritura del conversor.

---

## [2.2.0] — 2026-06-27

Ingesta **multinúcleo** con un **despachador adaptativo**: el motor elige la mejor
estrategia según la entrada (FASTA/FASTQ, plano, `.gz`, **BGZF**) y solo llama a
lo necesario. Incluye descompresión `.gz` rápida (libdeflate) y descompresión
**BGZF en paralelo** + conversor.

### Added

**Lector BGZF paralelo — `.gz` por bloques, descomprimible en todos los núcleos**
- `bio_is_bgzf` detecta el formato (subcampo extra `BC`); `bio_bgzf_usize` da el
  tamaño; `bio_bgzf_decompress_parallel` descomprime los bloques en paralelo
  (OpenMP, 1 descompresor libdeflate por hilo). Un BGZF es un `.gz` 100 % válido.
- El despachador detecta BGZF y enruta a la vía paralela; los `.gz` normales
  siguen con libdeflate (1 hilo). Medido: BGZF **113 M bases/s** vs 58 del
  baseline (~1.95×).

**Conversor a BGZF (`bioforge/bgzf.py`)**
- `bgzf.compress_file(path)` / `compress_bytes(data)` — comprime a BGZF **en
  paralelo**; salida compatible con `gunzip` estándar. CLI `bioforge-bgzip`.
- Idea: convierte una vez un FASTQ que procesas muchas veces y léelo siempre por
  la vía más rápida.

**Descompresión `.gz` rápida con libdeflate (la victoria real)**
- Para archivos `.gz`, el motor descomprime el archivo entero con **libdeflate**
  (SIMD, ~2× más rápido que zlib) y parsea el resultado en memoria. Medido:
  zlib 56 → **89 M bases/s end-to-end (1.59×)** leyendo FASTQ comprimido; la
  descompresión sola es **2.15× más rápida** (379 vs 176 MB/s).
- Fallback robusto a zlib en streaming (RAM constante) si el tamaño es
  inesperado (gzip multi-miembro, etc.).

**Parser paralelo (OpenMP)**
- `bio_parse_mem_parallel` (`engine.c`): trocea un bloque en N rangos alineados
  a límites de registro y los parsea en paralelo, con buffers por hilo y fusión
  serial. Salida idéntica al parser secuencial.
- `_stream_parallel`: mmap + vistas NumPy sin copia, troceo por ventanas.
- Nota honesta: en hardware de **pocos núcleos** el parseo paralelo da poco
  (~1.1×) — está limitado por ancho de banda de memoria, no por CPU (el C escala
  2.2× en aislado). Se conserva como opción; rinde más en servidores multinúcleo.

**Despachador adaptativo + API**
- `stream_batches` / `stream_fastq_batches` aceptan ``n_threads`` (1 = secuencial
  y RAM constante; >1 = nº de hilos; 0 = todos los núcleos). El motor enruta:
  plano → parseo paralelo; `.gz` → libdeflate + parseo; si algo falta, cae a la
  ruta secuencial con zlib.
- Banderas `C_PARALLEL_AVAILABLE`, `C_LIBDEFLATE_AVAILABLE` (`engine/_loader.py`).
- `ReadBatch.decoded_2d()` / `quality_matrix()` ya estaban (v2.1).

**Empaquetado / build**
- `build.py` enlaza **estáticamente** OpenMP (libgomp), zlib y libdeflate dentro
  del DLL → motor C **autocontenido** (sin dependencias de runtime). Degrada con
  gracia: si libdeflate no está, compila con zlib; si zlib no está, sin `.gz`.

### Tests
- 298 tests (desde 284): parser paralelo == secuencial (FASTQ fijo/variable,
  FASTA, muchas ventanas, registros vacíos, fallback `.gz`), `.gz` rápida con
  libdeflate == zlib, y BGZF (`tests/test_bgzf.py`) — round-trip, compatibilidad
  con `gunzip`, lectura paralela.

---

## [2.1.0] — 2026-06-27

Primera **aplicación de cara al usuario** construida sobre el motor v2.0: un
informe de calidad de FASTQ rápido (estilo FastQC) que aprovecha la API columnar.

### Added

**Informe de calidad FASTQ (`bioforge/qcreport.py`)**
- `qcreport.run(path)` — calcula todas las métricas en **una sola pasada** sobre
  `stream_fastq_batches` (RAM constante, sin objeto por lectura). Lee `.gz`.
- Métricas: nº lecturas, bases, longitud (min/media/max), GC global, calidad
  media global, % de lecturas con Q media ≥ 20 y ≥ 30, histograma de calidad por
  lectura, histograma de %GC, **calidad media por posición** (el gráfico estrella
  de FastQC), y composición A/C/G/T/N por posición.
- `qcreport.build_report(r)` — informe de texto con histogramas y sparkline ASCII.
- CLI: `python -m bioforge.qcreport reads.fastq.gz [-o informe.txt]` y entry point
  `bioforge-qc`.

**API columnar (`biocore.py`)**
- `ReadBatch.decoded_2d()` — códigos como matriz `(m, L)` (longitud fija) o `None`.
- `ReadBatch.quality_matrix()` — calidades como matriz `(m, L)` o `None`.

### Tests
- 284 tests (desde 275): `tests/test_qcreport.py` añade 9 tests del informe —
  métricas contra valores a mano, calidad por posición, composición por base,
  `.gz` == plano, longitud irregular, CLI y errores.

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
