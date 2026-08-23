# BioForge — Contexto para Claude Code

## Qué es este proyecto

BioForge: motor bioinformático de alto rendimiento para Edge Computing (hardware limitado).
Sin Biopython. NumPy core + motor C opcional (ctypes). Python 3.13, Windows 10.
Es un paquete instalable: `from bioforge import ...` (versión actual **10.1.0**,
publicada en PyPI con wheels nativos Win/Linux/Mac). Desde la v10.0 tiene además una
**app de escritorio** (`bioforge.app`, "la otra cara del motor"): misma versión, dos
rostros — un `.exe` autocontenido (doble clic, sin Python) y el comando `bioforge-app`
(`pip install "bioforge[app]"`). También tiene web pública (GitHub Pages, `docs/`,
bilingüe EN/ES) en https://erlanders177.github.io/bioforge/.

**v10.1 — organización por FUNCIONES + carga perezosa.** El paquete dejó de ser 17
`.py` planos: ahora son subpaquetes por función (`core/ sequence/ align/ mapping/
evolution/ nanopore/ io/ cli/ app/ engine/`), y los tests van en espejo. Además,
`import bioforge` **ya no carga el motor entero**: cada nombre trae su módulo solo
cuando se usa (PEP 562, mapa `_EXPORTS` en `__init__.py`). Medido: **75 ms → 4.7 ms**
y **1 submódulo cargado en vez de 15**; pedir `SmartTranslator` no carga nanoporo,
evolución ni mapeo. La API pública NO cambió y las 14 rutas viejas siguen funcionando
vía módulos-puente (con `DeprecationWarning`) — nada de lo publicado en 10.0.0 se rompe.

Niveles implementados y validados:
- **L1** `bioforge/core/biocore.py` — almacenamiento 5-bit, LUTs, BitPacker, PackedSequence, SmartImporter
- **L2** `bioforge/sequence/translator.py` — traducción ADN→Proteína vectorizada (CODON_LUT + sliding_window_view); 6-frame + reverse complement
- **L3** `bioforge/align/pairwise.py` — Needleman-Wunsch wavefront (global/semi-global), banded NW, Smith-Waterman local
- **L4 (v5.0)** mapeador de genomas / reads largos — seed-chain-align estilo
  minimap2, **tubería ENTERA en C tras un índice opaco** (cubierta Python fina):
  `bio_index_build` (índice opaco: tabla de minimizers ordenada + codes de la
  referencia + fronteras de contigs), `bio_map_read` (minimizers → lookup →
  chaining DP+backtrack+supresión → extensión banded del read completo → `Mapping`,
  todo en una llamada C), `bio_map_batch` (lote en paralelo con OpenMP, sin GIL).
  `GenomeAligner.map`/`map_batch` enrutan por C con **fallback NumPy idéntico**
  (verificado con paridad exacta). Módulos Python `mapping/minimizers.py`/`refindex.py`/
  `genomemap.py` siguen como fallback y utilidades. Multi-cromosoma; salida PAF.
  **v6.0 (SIMD + multihilo) — ya competitivo.** Head-to-head real medido en WSL
  (`tools/bench_vs_minimap2.py`, 4.8 Mb, 6000 reads, 5% error, minimap2 -a):
  **~1.3× por debajo de minimap2** tanto en 1 hilo (~1.8 vs ~2.4 Mb/s) como en 4
  núcleos (~3.0 vs ~4.0 Mb/s), ambos mapean los 6000. Se llegó desde ~4×/~3× con:
  (a) **extensión banded SIMD AVX2** (`_nw_banded_diag_simd`, 8× int32 antidiagonal
  → kernel 88→529 M celdas/s, 6×; mapeador 4× en 1 hilo; bit-idéntico al escalar,
  fallback sin AVX2); (b) **fix del reset de hilos OpenMP** en `bio_map_batch`
  (n<=0 → todos los núcleos siempre) → escalado 2.3× real. **v6.1:** salida
  **columnar** en `map_batch` (array estructurado NumPy en vez de dicts/ctypes) →
  mata la cola serial Python → en 4 núcleos **a la par de minimap2** (~4.2-4.7 vs
  ~4.0-5.7 Mb/s, a veces por delante). Single-thread aún ~1.2-1.3× por detrás.
  Red `test_cmap_parity.py` garantiza que nada de esto cambia resultados.
- **Ingesta v2.0** `bioforge/engine/engine.c` + `core/biocore.py` — parser FASTA/FASTQ en C (streaming + por lotes), API columnar, `.gz`
- **L5 (v7.0) — predicción de evolución** `bioforge/evolution/predict.py` (+ `fetch.py`, `ai/`).
  Genoma-agnóstico, sobre el MSA. Piezas: (a) **backtest honesto** horneado —toda
  predicción se mide contra la ingenua "mañana = hoy"; (b) **linajes ESTABLES** estilo
  Pango/autolin SIN árbol filogenético (GRI = N·D/(S+N+D) por co-ocurrencia, dos
  matmuls sobre el MSA; `designate_lineages`); (c) **rankeador de MUTACIONES**
  (`rank_mutations`) que ordena qué sustitución subirá — la pregunta que el campo sí
  responde (AUC), donde la ingenua no juega; (d) **modelo entrenado** (regresión
  logística + interacciones, Newton/IRLS, pesos en `data/ranker_weights.npz`,
  inferencia NumPy pura sin torch); (e) **eje B opcional** ESM-2 (`bioforge[ai]`).
  **Honestidad medida y no negociable:** predecir FRECUENCIAS empata con la ingenua a
  todo horizonte (callejón cerrado); ORDENAR mutaciones sí tiene señal. La disimilitud
  físico-química va INVERTIDA (mide viabilidad, no escape) — replicado en H3N2/H1N1/B.
  ESM-2 sufre FUGA de preentrenamiento (−0.20 tras su corte). Nada de esto es novedad
  científica (DERIVE/EVEscape/Hie ya existen y son mejores); el valor es la CAJA
  integrada, accesible, en portátil y honesta. Ver `docs/` y memoria del proyecto.

- **L6 (v8.0) — JUEZ de predictores + FILTRO de realidad** `bioforge/evolution/evalkit.py` y
  `bioforge/evolution/realitycheck.py`. No predictores: las herramientas que deciden.
  **`EvolutionBenchmark(seqs, dates).judge(f)`** somete cualquier `f(Context)->scores`
  a la batería completa, cada prueba nacida de un autoengaño real: (a) **listón
  trivial** — el rival no es 0.5 sino el mejor eje gratis (frecuencia/mutabilidad/
  conservación); nuestro "AUC 0.80" era exactamente la mutabilidad; (b) **régimen
  NUEVAS** — separa lo ya circulante (basta contar) de lo nuevo; (c) **IC95%
  bootstrap**; (d) **detector de FUGA** — antes/después del corte DESCONTANDO la
  caída de un eje trivial de control; (e) `cross_validate`. `Context` es leak-free.
  **`RealityCheck(seqs, dates)`** juzga MUTACIONES concretas venidas de fuera
  (EVEscape/ESM-2/DMS): `.check("N145K")` / `.filter(lista)`. Dos niveles nunca
  mezclados — OBSERVADO (evidencia, su trayectoria) vs ESTIMADO (conjetura del
  modelo), cada uno con su fiabilidad aparte. "Sobrevivir" = alcanzar/mantener
  presencia real (NO "subir 5%": una variante al 98% no sube pero es la superviviente
  más clara — efecto techo). Calibrado contra el histórico; resiliente a basura.
  **Validado en H3N2 real limpio** (900 secs, 220.092 candidatas): evalkit caza azar/
  tautológico/solo-fácil, y a un tramposo con AUC 0.863 e IC limpio (fuga −0.438);
  RealityCheck da OBSERVADO AUC 0.97 / ESTIMADO 0.72. **Nuestro v7.0 juzgado:** MLP
  0.837 global / **0.631 NUEVAS** (APORTA), ejes manuales ~0.52 en NUEVAS (azar) →
  eso aporta la IA. ⚠ **Estas cifras son las LIMPIAS**: un draft previo citaba
  0.861/0.613, medidas sobre un MSA CORRUPTO (bug: empaquetaba proteínas como ADN,
  todo residuo no-ACGT→'N'; ver `git log msa.py`). Bug corregido, modelo reentrenado,
  todo re-medido. Fueron evalkit y RealityCheck los que destaparon la corrupción.

- **L7 (v9.0) — BASECALLER de nanoporo desde cero** `bioforge/nanopore/basecaller.py`. Señal
  eléctrica cruda de Oxford Nanopore → bases, **NumPy puro, sin IA, sin GPU**, vía la
  ruta CLÁSICA (HMM/Viterbi, no red neuronal). Piezas: `read_pod5`/`read_fast5` (ingesta
  de señal, dep OPCIONAL `bioforge[nanopore]` = pod5+h5py, solo la fontanería del
  formato); `normalize_signal` (mediana/MAD); `detect_events` (segmentación por t-stat,
  cumsum O(n)); `estimate_pore_model` (aprende la tabla k-mero→corriente de datos, no la
  copia); `viterbi_decode` (move-only) y `viterbi_basecall` (STAY/STEP/SKIP — el HMM
  completo, absorbe errores de segmentación); `basecall` (entry point: normaliza →
  sobre-segmenta → escala por-read por MOMENTOS → Viterbi). **Números REALES medidos:**
  decodificador 100% sobre niveles ideales; **74.5% sobre señal R9.4 REAL capturada**
  (E. coli, n=36, vs Guppy, identidad local con nuestro alineador; v9.0 daba 70.3%, la
  v9.1 lo subió con refit iterativo + p_stay=0.5) — en el rango de los
  clásicos históricos (nanocall ~68-85%), lejos del ~99% neuronal de Dorado. **Verdad
  no negociable:** es la vía clásica de la era R9; **R10 queda fuera por diseño** (9-mer
  = 4⁹ = 262.144 estados → Viterbi O(T·estados) inviable en portátil; ONT no publica
  tabla plana R10, sus modelos son neuronales dentro de Dorado) — la razón real de por
  qué el campo pasó al basecalling neuronal. Backend enchufable pendiente (usar Dorado
  si está). Benchmark REPRODUCIBLE en `tools/bench_basecaller.py` (baja modelo+datos
  públicos y mide de cero). El valor no es ganar a Dorado: es correr sin instalar nada.

- **L8 (v10.0) — APP DE ESCRITORIO** `bioforge/app/`. "La otra cara del motor": una
  ventana NATIVA y LOCAL (PyWebview) sobre el mismo `bioforge`, para no-programadores —
  analizar ADN a clics, sin código y sin que los datos salgan de la máquina (ADN Edge,
  sin servidor ni red). Cinco pestañas, cada una con explicación "para todos":
  **Secuencias** (listar/traducir ADN→proteína codón a codón), **Calidad** (informe QC
  estilo FastQC con gráficos SVG), **Alinear** (dos secuencias → mutaciones), **Nanoporo**
  (señal cruda → bases con nuestro basecaller, y "usar en otras pestañas") y **Evolución**
  (rank_mutations + RealityCheck). Piezas: `main.py` (lanzador PyWebview + diálogos de
  archivo nativos), `backend.py` (`Api`, el PUENTE que la interfaz invoca — cada método
  devuelve dicts, NUNCA lanza a la UI vía `@_guard`; se prueba SIN abrir ventana:
  `tests/test_app_backend.py`), `index.html` (toda la interfaz: JS vanilla, offline,
  gráficos SVG inline). **Dos rostros, mismo código y misma versión:** el paquete la trae
  dentro (`pip install "bioforge[app]"` → comando `bioforge-app`) y se distribuye también
  como `.exe` autocontenido (PyInstaller, `BioForge.spec`) que un workflow
  (`build-app.yml`, al PUBLICAR la Release) compila y adjunta SOLO. `app_dir()` resuelve
  recursos desde `sys._MEIPASS` (.exe) o `bioforge/app/` (paquete). Icono propio
  (`data/icon.ico`, doble hélice). El motor y la CLI funcionan sin el extra `app`.

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
- `minimizers.minimizers` — loop `range(k)` (Horner, O(k) fijo sobre vectores)
- `genomemap._chain_one` — DP **por ancla** (no por símbolo), ventana acotada;
  candidato a bajar a C. `_cigar` itera columnas alineadas (dependencia de datos,
  como el traceback). Permitidos.

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
secuencial. **El parseo paralelo da ~1.9× en plano** (500k reads: 0.40s→0.213s)
desde que el buffer de cabeceras dejó de cero-inicializarse (`np.empty` en vez de
`create_string_buffer`: los 48 MB de ceros costaban ~39 ms por llamada y se comían
la ventaja del multinúcleo). Con eso BioForge en paralelo **empata con seqkit a
1 hilo** y queda a ~1.33× de seqkit a 4 hilos. El otro win es libdeflate en `.gz`.
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

### 9. Actualizar el README en cada versión — ANTES de taggear
El `README.md` es el escaparate público (GitHub **y** la página de PyPI, que se
construye del tag). En cada versión, actualizar el README **en el mismo commit de
release, antes de crear el tag**, para que GitHub y PyPI muestren siempre el
estado real. Cifras clave, cuenta de tests, roadmap y limitaciones deben cuadrar
con el código. **Nunca** dejar contradicciones (p.ej. "30-50× más lento" cuando
ya se es competitivo): la honestidad del escaparate es tan importante como la del
benchmark. Si el README se corrige *después* de un tag, PyPI seguirá mostrando el
viejo hasta la siguiente versión → publicar un parche solo-docs para sincronizar.

### 10. Organización por funciones y carga perezosa (v10.1) — no degradar
- **Cada módulo va en su subpaquete por FUNCIÓN** (`core/ sequence/ align/ mapping/
  evolution/ nanopore/ io/ cli/ app/`). Nada de `.py` sueltos nuevos en la raíz del
  paquete: los que hay son **puentes de compatibilidad** y no se tocan ni se amplían.
  Los tests van **en espejo** (`tests/align/…`).
- **Nada de imports pesados en el nivel superior de `__init__.py`.** La API pública se
  declara en el mapa `_EXPORTS` de `bioforge/__init__.py`, del que salen *a la vez*
  `__all__` y la resolución perezosa (PEP 562) — así no se desincronizan. Al añadir un
  símbolo público, se registra ahí; **no** se añade un `from .x import y` arriba.
  Regla práctica: `import bioforge` debe seguir cargando **un solo** submódulo.
- **Rutas canónicas en código propio** (motor, tests, tools, app): los puentes existen
  solo para quien instaló ≤10.0.0. Si aparece un `DeprecationWarning` nuestro en los
  tests, es un import que hay que corregir.
- **La app mantiene la RAM plana**: solo el archivo ACTIVO materializado; de los demás,
  su ficha (`meta`). Listar pestañas nunca debe forzar la carga de archivos.

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
| `import bioforge` (carga perezosa, v10.1) | **4.7 ms** (antes 75 ms, 16×) · **1** submódulo vs 15 |
| App con N archivos abiertos (v10.1) | **RAM plana** (~0.2 MB con 20; antes crecía lineal a 1.07) |

⚠️ El resumen ejecutivo original cita "60-70%" — ese número es incorrecto.
Correspondería a 2-bit packing, no al esquema 5-bit implementado.

---

## Estructura de archivos

Organizado **por FUNCIONES** (v10.1), no por capas. Los tests van en espejo.

```
bioforge/                  paquete instalable (from bioforge import ...)
  __init__.py              API pública + CARGA PEREZOSA (mapa _EXPORTS -> PEP 562)
                           + __version__ (fuente única de versión)
  core/biocore.py          EL CIMIENTO — almacenamiento 5-bit, LUTs, BitPacker,
                           PackedSequence, SmartImporter (lector FASTA/FASTQ),
                           FastqRecord, SequenceBatch/ReadBatch (API columnar),
                           jerarquía de errores — no tocar sin impacto global
  sequence/translator.py   L2 — traducción ADN→Proteína, 6-frame, reverse complement,
                           translate_many() COLUMNAR
  align/
    pairwise.py            L3 — NW global/semi-global, banded, Smith-Waterman,
                           band="auto" (banda ADAPTATIVA exacta)
    msa.py                 MSA center-star — soporte del predictor
  mapping/
    minimizers.py          L4 — minimizers canónicos (w,k) vectorizados
    refindex.py            L4 — índice de la referencia (hash ordenado + searchsorted)
    genomemap.py           L4 — seed-chain-align: GenomeAligner.map → PAF
  evolution/
    predict.py             L5 — backtest, linajes estables (designate_lineages),
                           rank_mutations, score_mutations
    evalkit.py             L6 — el JUEZ honesto: EvolutionBenchmark.judge/cross_validate,
                           Context (leak-free), Report (veredicto)
    realitycheck.py        L6 — FILTRO de realidad: RealityCheck.check/filter,
                           Verdict (OBSERVADO=evidencia | ESTIMADO=conjetura)
    fetch.py               L5 — descarga NCBI Entrez fechada (stdlib, caché + reintentos)
    ai/viability.py        L5 — eje B opcional: ESM-2 (bioforge[ai], carga perezosa)
  nanopore/basecaller.py   L7 — basecaller desde cero: read_pod5/read_fast5,
                           detect_events, estimate_pore_model, viterbi_basecall
                           (STAY/STEP/SKIP), basecall. NumPy puro. 74.5% en R9.4 real
  io/
    qcreport.py            informe de calidad FASTQ (tipo FastQC) — CLI bioforge-qc
    bgzf.py                conversor a BGZF (gzip por bloques) — CLI bioforge-bgzip
  cli/
    analyze.py             pipeline CLI (dna/protein/both) — bioforge-analyze
    evolution.py           CLI de evolución (rank/backtest/linajes) — bioforge-evolution
  app/                     L8 (v10.0) — app de escritorio (PyWebview, local, sin servidor)
    main.py                lanzador: ventana + diálogos nativos (comando bioforge-app)
    backend.py             Api: el PUENTE que la UI invoca (dicts, @_guard). RAM PLANA:
                           solo el archivo ACTIVO materializado (v10.1)
    index.html             toda la interfaz (JS vanilla, offline, gráficos SVG inline)
    data/                  recursos de la UI: pore model + icon.ico (doble hélice)
  data/ranker_weights.npz  L5 — pesos del rankeador entrenado (2.2 KB, en el wheel)
  engine/
    engine.c               motor C — pack/unpack, NW/SW, parser FASTA/FASTQ + batch + .gz
    engine.dll             binario compilado (versionado en git)
    _loader.py             ctypes + banderas C_AVAILABLE/C_PARSER_AVAILABLE/…
    build.py               compila el DLL/SO (autodetecta GCC, enlaza zlib)
  aligner.py biocore.py …  14 módulos-PUENTE en las rutas viejas (compatibilidad
                           con 10.0.0; avisan con DeprecationWarning). No usarlos
                           en código nuevo: el motor ya usa solo rutas canónicas.
tools/
  visor.py                 frontend interactivo (loops permitidos aquí)
  comparador.py            comparador de secuencias (CLI)
  stress_test.py           benchmark de 30M bases
  bench_vs_biopython.py    BioForge vs Biopython (tiempo + RAM)
tests/                     EN ESPEJO del paquete: core/ sequence/ align/ mapping/
                           evolution/ nanopore/ io/ cli/ app/  (548 tests)
docs/                      documentación técnica (.md) + LA WEB pública (GitHub Pages,
                           index.html EN, es/index.html ES, style.css, sitemap, og.png)
pyproject.toml             empaquetado (versión dinámica; incluye DLL + app en el wheel)
BioForge.spec              PyInstaller: empaqueta la app en dist/BioForge/BioForge.exe
BioForge.bat               lanzador dev (doble clic) — python -m bioforge.app.main
.github/workflows/         tests · wheels (tag vX→PyPI) · build-app (Release→.exe adjunto)
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
