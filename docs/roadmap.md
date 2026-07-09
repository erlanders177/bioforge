# Roadmap del Proyecto

## Estado actual (2026-07-09) — v5.0.0

| Nivel | Módulo | Descripción | Estado |
|-------|--------|-------------|--------|
| L1 | biocore.py | Almacenamiento 5-bit, parser FASTA/FASTQ en C, API columnar, `.gz`/BGZF, reverse complement | ✅ Completo |
| L2 | smart_translator.py | Traducción ADN→Proteína, 6-frame | ✅ Completo |
| L3 | aligner.py | NW global/semi-global, banded NW, Smith-Waterman local | ✅ Completo |
| L4 | genomemap · minimizers · refindex | Mapeador seed-chain-align, **tubería entera en C** tras índice opaco, OpenMP | ✅ Completo (v5.0) |
| — | analyze / qcreport / bgzf | Pipeline CLI, informe QC tipo FastQC, conversor BGZF | ✅ Funcional |

Publicado en PyPI (`pip install bioforge`) con wheels nativos Win/Linux/Mac.
359 tests. Motor C con fallback NumPy verificado (paridad exacta).

---

## La tesis del proyecto — hacia dónde vamos y por qué

**No vamos a ganar la guerra de la velocidad pura.** minimap2 y compañía tienen
más manos y más años; empatarles en throughput bruto es una guerra de recursos.
La jugada no es competir en su terreno — es **abrir uno nuevo donde ellos no
juegan**. Un mapeador rápido es una *commodity*; un motor que además **modela
cómo evoluciona una secuencia y propone la siguiente cepa probable** tiene
identidad propia. Ahí está el legado.

Regla que gobierna todo lo de abajo: **honestidad radical**. Nada se promociona
sin backtesting y números reales. "No siempre acierta" no es un defecto que
ocultar — es la naturaleza del problema, y decirlo es lo que nos hace serios.

---

## Fases planificadas (en orden)

### v6.0 — SIMD en la alineación base a base  *(EN CURSO)*
El muro de velocidad del mapeador es el DP banded escalar (~88% del tiempo, ya
en C → por eso pasar seed/chain a C solo dio 1.7×). La solución: **SIMD
(KSW2/SSE)** — procesar la antidiagonal en lotes de 8-16 celdas por instrucción.
- **Baseline honesto medido (WSL, `tools/bench_vs_minimap2.py`, 4.8 Mb, 2000
  reads, 5% error, minimap2 -a):** gap real **~4× en 1 hilo** (minimap2 1.35 vs
  BioForge 0.34 Mb/s), ambos mapean los 2000. NO los ~30-50× estimados de memoria.
  A gran escala minimap2 se separa (su multihilo escala; el nuestro apenas 1.5×
  por la cola serial de reconstruir Mapping en Python — otra palanca a atacar).
- **Objetivo realista:** con un 3-4× del SIMD en la extensión, rozar/igualar a
  minimap2 en 1 hilo en cargas de este tamaño. Entorno WSL ya montado (minimap2,
  gcc, valgrind).
- La red `test_cmap_parity.py` protege la corrección durante la reescritura.

### Mejorar el comparador
Ya existe `tools/comparador.py`. Antes del salto evolutivo, robustecerlo y
prepararlo como cimiento del análisis de cepas (comparación de muchas
secuencias, no solo pares).

### Motor de cadenas de Markov de evolución
Las cadenas de Markov **son** la base legítima de la biología molecular: los
modelos de sustitución (Jukes-Cantor, HKY, GTR) son cadenas de Markov sobre las
bases. Desde un conjunto de secuencias **alineadas y fechadas** construir
matrices de transición:
- **Por-posición**: qué sitios mutan y hacia qué (site-specific).
- **Por-contexto**: Markov de orden-N sobre k-meros (dependencias locales).

### Predicción de cepas futuras — el horizonte
**La idea:** dado cómo ha evolucionado un virus (p.ej. la gripe), generar la
**próxima cepa probable**. Serán intentos, no profecías.
- Dada la última secuencia + el modelo Markov → muestrear/elegir el descendiente
  más probable. Framing honesto: **hipótesis probabilística**.
- **Backtesting = el guardián de la honestidad.** Entrenar con cepas hasta el año
  T, predecir T+1, medir el parecido con la que *de verdad* apareció
  (identidad/Hamming). Convierte "creo que predice" en un número. Sin esto no se
  promociona nada.
- Es forecasting para salud pública (como Nextstrain / la selección de cepas de
  la OMS), **análisis y anticipación, no síntesis de patógenos**. Terreno limpio.
- **Límite honesto conocido**: un Markov crudo modela mutación *neutral*; la
  evolución viral real la manda la *selección* (escape inmune). Mejoras futuras:
  tasas por-posición, señal de *fitness*, contexto filogenético. Hasta los
  profesionales fallan (años de vacuna mal emparejada) — eso es esperado.
- **Prerrequisito de datos**: series temporales reales fechadas (HA de gripe de
  NCBI/GISAID). Sin serie temporal no hay entrenamiento ni backtesting.

---

## Decisiones de diseño cerradas

No reabrir sin un caso de uso concreto que las justifique:

| Decisión | Alternativa descartada | Razón |
|----------|----------------------|-------|
| Alfabeto 5-bit unificado | 2-bit solo para ADN | Unificación de pipeline nucleótido+proteína |
| NW global para mutaciones | Smith-Waterman (local) | SW busca motivos; NW compara alelos |
| NumPy puro (sin Numba/Cython) | Numba JIT | Portabilidad, cero dependencias extra |
| Anti-diagonal wavefront en NW | Loops O(m·n) Python | Regla de oro del proyecto |
| Motor entero en C tras índice opaco | Orquestar el mapeo desde Python | Sin marshalling por consulta; cubierta Python fina |
| Diferenciación evolutiva (Markov→cepas) | Competir en velocidad pura | Terreno nuevo donde los gigantes no juegan |
