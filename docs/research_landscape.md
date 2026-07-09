# Panorama de herramientas — investigación (2026-07-09)

Sondeo del estado del arte para extraer **lo único de cada herramienta** que sea
adaptable a nuestra arquitectura limpia. Dos frentes: **velocidad** (mapeo/
alineamiento) y **evolución** (predicción de cepas). Todo con la regla de siempre:
honestidad sobre el esfuerzo real y la incertidumbre. Fuentes al final.

Ventaja de partir de 0: la base no está cimentada, podemos incorporar las mejores
ideas sin lastre heredado. El campo de batalla que elegimos: **la herramienta
integrada** (mapeo rápido + evolución), no ganar a cada especialista en su nicho.

---

## FRENTE VELOCIDAD

### 1. WFA / BiWFA — Wavefront Alignment  ⭐ (lead #1)
- **Idea única:** explora la matriz DP **por score creciente**, no por banda fija.
  Es una "banda adaptativa" que se calcula sola, sin conocer el error de antemano;
  poda las diagonales sin potencial de llevar al óptimo.
- **Complejidad:** O(ns) tiempo (n=longitud, s=score/error), memoria O(s²) en WFA
  → **O(s) en BiWFA** (bidireccional). Para secuencias **parecidas** (bajo error,
  s pequeño) es dramáticamente rápido.
- **Velocidad publicada:** ~200× vs DP clásico; **6-7× vs métodos de banda
  adaptativa**. (El 200× es vs DP ingenuo; el número relevante para NOSOTROS, que
  ya hacemos SIMD banded, es el 6-7× vs banda adaptativa — potencial real, pero
  **hay que medirlo** contra nuestro `_nw_banded_diag_simd`.)
- **WFA2-lib** (smarco): **licencia MIT**, soporta global / semi-global / extensión,
  gap-affine/linear/edit, produce **CIGAR**, **auto-vectoriza (SIMD)** sin código
  específico de ISA, y modo `ultralow` (BiWFA, memoria O(s)).
- **Encaje BioForge:** el candidato más fuerte al próximo salto de velocidad, sobre
  todo en reads de bajo error. Dos vías: (a) **reimplementar** el algoritmo (mantiene
  el ADN "desde 0"; el núcleo no es enorme); (b) **enlazar WFA2-lib** (MIT, compatible
  con nuestra licencia noncommercial, pero añade una dependencia C). La red
  `test_cmap_parity` / `test_simd_kernel` protege la corrección al cambiar el kernel.
- **Riesgo:** medio-alto (algoritmo nuevo + traceback). Medir antes de comprometer.

### 2. Seeding de nueva generación — syncmers · strobemers · MCS
- **Problema de los minimizers (lo nuestro):** repetitividad (caen en repeticiones,
  inflan anclas).
- **Syncmers:** submuestreo alternativo con **mejor conservación** bajo mutación.
- **Strobemers:** enlazan 2+ k-mers separados (gapped) → más sensibles a indels y
  **reducen la repetitividad ~10×** vs minimizers en genomas grandes.
- **Multi-context seeds (MCS, 2024, en strobealign):** **superan la precisión de
  minimap2** en modo extensión y son más rápidos; comparables a BWA-MEM en alta
  diversidad con mucho menos tiempo de indexado/mapeo.
- **Encaje:** cambiar nuestro seeding (minimizers → syncmers/strobemers) mejoraría
  **sensibilidad Y velocidad**. Esfuerzo **moderado** (tenemos `minimizers.py` +
  `refindex.py` aislados). **Bajo riesgo.** Buen win paralelo al WFA.

### 3. FM-index / BWT (BWA, bowtie2)
- **Idea única:** índice comprimido Burrows-Wheeler para búsqueda exacta de
  sub-cadenas con **poca memoria**. Paradigma distinto al de minimizers.
- **Encaje:** útil si el objetivo fuera short-read exacto / footprint mínimo. Menos
  alineado con nuestro nicho long-read seed-chain-align. **Nota, no prioridad.**

### 4. SIMD avanzado — KSW2 y trucos baratos
- **KSW2** (minimap2): recurrencia de diferencias (Suzuki-Kasahara) + **Z-drop**
  (aborta la alineación cuando diverge demasiado → ahorra celdas).
- **Encaje:** ya hacemos SIMD antidiagonal. **Z-drop es un truco barato y adoptable**
  (cortar temprano). AVX-512 / int16 = más carriles (2× sobre AVX2 int32), a costa
  de complejidad y de un límite de longitud por el rango de int16.

---

## FRENTE EVOLUCIÓN

### 1. Modelos de lenguaje de proteínas (ESM-2, Nucleotide Transformer)  ⭐
- **Grammaticality + semantic change** (Hie et al., *Science* 2021): la base del
  "lenguaje del escape viral" — una mutación candidata es la que el modelo ve
  probable (gramática) Y que cambia mucho el significado (escape).
- **Nuance honesta (evaluación sistemática, 2025):** la gramaticalidad mide
  viabilidad, PERO los métodos entrenados **explícitamente** para predecir el efecto
  de la mutación resultan **más efectivos** que la gramaticalidad cruda. → No basta
  ESM-2 zero-shot; conviene un modelo/ajuste específico al efecto de mutación.
- **Encaje:** escalera — Markov + backtesting (base local) → ESM-2 pre-entrenado
  (inferencia, sin entrenar) → fine-tuning/objetivo explícito del efecto. Modelos
  pequeños corren en portátil / Colab gratis.

### 2. Forecasting específico de gripe
- **MetaFluAD (2024):** meta-learning para distancias antigénicas entre cepas.
- **Predicción antigénica estacional de H3N2** con ML (aplicación práctica directa).
- **Framework unificado deep learning** (Nature Machine Intelligence, ene 2025) para
  predicción de variación viral (COVID/gripe/VIH; mejoras notables de precisión).
- **Modelos de fitness** (Łuksza-Lässig, *Nature* 2014): pocos parámetros, predijeron
  la evolución de la gripe. **Nuestra base honesta y alcanzable** sin GPU.

### 3. Nextstrain / filodinámica
- Estándar de rastreo evolutivo en tiempo real (árboles filogenéticos + fitness +
  epidemiología). Referencia de "cómo se hace de verdad" y con qué contrastar el
  backtesting.

---

## Prioridad recomendada (honesta)

**Velocidad (para cerrar y superar a minimap2 en NUESTRO terreno):**
1. **WFA/BiWFA** — próximo gran salto; **medir vs nuestro SIMD** antes de comprometer.
2. **Seeding strobemer/syncmer** — win moderado, bajo riesgo, mejora sensibilidad+velocidad.
3. **Z-drop** — truco barato; **cola serial Python en `map_batch`** (salida columnar) para el multinúcleo.

**Evolución (el frente diferenciador):**
1. **Markov + backtesting** (base local, sin GPU) → primer número honesto.
2. **ESM-2 pre-entrenado** (grammaticality+semantic change) → reproducir *Science* en Colab.
3. **Objetivo explícito de efecto de mutación** (el nuance 2025) → donde de verdad mejora.

**El hilo conductor:** integrar seeding moderno + WFA + evolución en UNA arquitectura
limpia. Ese es el campo de batalla elegido — ser la herramienta *integrada*, no el
#1 aislado de cada especialista.

---

## Fuentes

- Marco-Sola et al. *Fast gap-affine pairwise alignment using the wavefront
  algorithm.* Bioinformatics 2021. https://academic.oup.com/bioinformatics/article/37/4/456/5904262
- *Optimal gap-affine alignment in O(s) space* (BiWFA). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9940620/
- WFA2-lib (MIT). https://github.com/smarco/WFA2-lib
- Sahlin. *Effective sequence similarity detection with strobemers.* Genome Res 2021. https://pmc.ncbi.nlm.nih.gov/articles/PMC8559714/
- *Strobealign: flexible seed size…* Genome Biology 2022. https://link.springer.com/article/10.1186/s13059-022-02831-7
- *Multi-context seeds enable fast and high-accuracy read mapping* (2024). https://www.biorxiv.org/content/10.1101/2024.10.29.620855
- Hie et al. *Learning the language of viral evolution and escape.* Science 2021.
- *A systematic evaluation of the language-of-viral-escape model…* (2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC12040448/
- *A unified evolution-driven deep learning framework for virus variation driver
  prediction.* Nature Machine Intelligence, ene 2025.
- MetaFluAD (2024) — meta-learning para distancias antigénicas de gripe.
- Łuksza & Lässig. *A predictive fitness model for influenza.* Nature 2014.
