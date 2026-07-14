# Panorama de las CAJAS DE HERRAMIENTAS generalistas — investigación (2026-07-10)

No los especialistas de un algoritmo (eso está en `research_landscape.md`), sino los
**generalistas**: las suites que intentan hacer de todo. Objetivo: extraer **la cosa
única** de cada una y ver cómo fusionarlas aprovechando que partimos de cero — sin
heredar su lastre (código de los 90, dependencias, GUIs pesadas).

Tesis: ninguna caja reúne *nuestro* conjunto (5-bit + mapeo a la par de minimap2 +
MSA + predictor de evolución integrado + edge + honestidad). Los generalistas son
anchos pero lentos/viejos; los especialistas son rápidos pero aislados. El hueco es
**la fusión limpia**.

---

## Ficha por herramienta: qué la hace ÚNICA · qué robamos · cómo lo mejoramos

### 1. Biotite — el "NumPy-native" moderno  ⭐ (nuestro espejo)
- **Único:** las secuencias y estructuras **SON arrays NumPy**; el indexado usa la
  misma semántica de NumPy. Vectoriza (traducción, geometría); Cython donde no llega.
  Cuatro subpaquetes limpios: `sequence`, `structure`, `database`, `application`.
- **Robamos:** la filosofía NumPy-native (ya es la nuestra), el split limpio de
  subpaquetes, y sobre todo el concepto de **`application`** (interfaz a herramientas
  externas) y **`database`** (bajar de bases de datos) como ciudadanos de primera.
- **Mejoramos (desde 0):** ellos usan Cython; nosotros un **DLL C autocontenido con
  SIMD + OpenMP + fallback NumPy** → rutas calientes más rápidas Y cero dependencias.
  Nuestro **5-bit** gana en RAM a sus arrays crudos. Y añadimos el frente evolución
  que ellos no tienen.

### 2. seqkit — la navaja suiza de línea de comandos  ⭐
- **Único:** **un solo binario** sin dependencias, multiplataforma, **38 subcomandos**
  para FASTA/Q, ultrarrápido (Go), **streaming STDIN/STDOUT**, se encadena en pipes
  Unix, lee gzip/xz/zstd/bz2/lz4. Cero configuración.
- **Robamos:** el ethos de **"una cosa autocontenida, sin deps, componible en pipes"**
  (ya tenemos DLL autocontenido + wheels). La **amplitud de operaciones pequeñas**
  (subseq, sliding, dedup, sample, stats, grep, translate) bajo un CLI unificado.
  El **streaming pipe-friendly**.
- **Mejoramos:** seqkit es **solo CLI** (Go). Nosotros damos **API Python limpia +
  CLI**, respaldados por el motor C + 5-bit + salida columnar. Un `bioforge` CLI que
  imita la ergonomía de seqkit pero enchufa a nuestra API.

### 3. scikit-bio — el alto nivel educativo
- **Único:** alto nivel, **recursos educativos** de primera, amplitud "ómica"
  (genómica, microbioma, ecología, evolución), BSD, se vende como "el Biopython de
  alto rendimiento". Estructuras de datos primero.
- **Robamos:** el ángulo **educativo** (encaja clavado con nuestro nicho
  accesibilidad/edge/"un estudiante lo entiende"), la API limpia de alto nivel.
- **Mejoramos:** bajamos más (5-bit, C SIMD) Y tenemos mapeo de genomas + predictor
  de evolución que ellos no enfatizan.

### 4. Biopython — el patriarca (amplitud + pegamento)
- **Único:** **madurez y amplitud**: I/O de casi todo formato, integración con
  herramientas externas (BLAST, Clustal, EMBOSS), **acceso a bases de datos** (NCBI
  Entrez), es el que **todo el mundo ya conoce**.
- **Robamos:** la **amplitud de I/O de formatos**, el **pegamento a bases de datos**
  (bajar de NCBI/GISAID — *prerrequisito del predictor*, que necesita secuencias
  fechadas reales), y hacer nuestra API **familiar** a quien viene de Biopython
  (baja el coste de cambio).
- **Mejoramos:** Biopython es **pre-NumPy y lento**; nosotros NumPy+C, ya medido
  **~6.9× menos RAM y ~9.5× más rápido** cargando. API familiar pero mucho más ligera.

### 5. EMBOSS — la suite compilada exhaustiva
- **Único:** **~200+ herramientas** de línea de comandos en C, exhaustiva, invocable
  desde varios lenguajes.
- **Robamos:** el modelo de **suite de herramientas compiladas y componibles**.
- **Mejoramos:** EMBOSS es viejo, disperso y difícil de instalar; nosotros **un solo
  paquete `pip` limpio** con API unificada.

### 6. UGENE — el workbench LOCAL-FIRST  ⭐
- **Único:** workbench todo-en-uno (GUI + CLI) que integra decenas de tools, con un
  **Workflow Designer que corre EN LOCAL** — sin nube, sin transferir datos, offline.
  Privacidad y funcionamiento sin internet como rasgo distintivo.
- **Robamos:** el ethos **local-first + offline + privacidad** (encaja perfecto con
  edge/hardware limitado), y la idea de **workflow** (encadenar nuestras tools:
  importar → mapear → MSA → predecir).
- **Mejoramos:** UGENE es una app Qt **pesada**; nosotros una **librería ligera
  embebible + CLI**. Un "pipeline" local (import→map→MSA→predict) sin el GUI pesado.

### 7. Geneious / CLC — los comerciales pulidos
- **Único:** todo-en-uno **pulido**, GUI cómoda, alineamiento + filogenia + diseño de
  primers. UX de pago.
- **Robamos:** la **aspiración de integración y UX pulida**.
- **Mejoramos:** son **cerrados y caros**; nosotros abiertos, gratis, licencia
  no-comercial, y corremos en una patata.

---

## La síntesis: qué fusionamos (la ventaja de partir de 0)

| De… | Fusionamos su rasgo único |
|-----|---------------------------|
| **Biotite** | núcleo NumPy-native limpio → lo llevamos a 5-bit + C SIMD autocontenido |
| **seqkit** | un paquete sin deps, componible en pipes, streaming, CLI ergonómico |
| **scikit-bio** | accesibilidad + recursos educativos |
| **Biopython** | amplitud de I/O + **pegamento a bases de datos** + API familiar |
| **EMBOSS** | suite de herramientas compiladas componibles |
| **UGENE** | local-first, offline, privacidad, **workflow** integrado |
| **BioForge (único)** | 5-bit · mapeo a la par de minimap2 · **predictor de evolución** · edge · honestidad/backtesting |

**El resultado que NO existe en ningún sitio:** rendimiento NumPy-native (Biotite) +
autocontenido componible (seqkit) + accesibilidad educativa (scikit-bio) + pegamento
de formatos/BD (Biopython) + workflow local-privado (UGENE), **más las dos cosas que
NINGÚN generalista tiene**: (1) mapeo de genomas a la par del especialista (minimap2)
y (2) un **predictor de evolución integrado**. Esa combinación es genuinamente nueva.

---

## Ideas concretas y accionables robadas (para el roadmap)

1. **`bioforge.fetch` — pegamento a bases de datos** (de Biopython/Biotite `database`).
   Bajar secuencias fechadas de NCBI (Entrez). **Es prerrequisito del predictor**
   (necesita datos reales con fecha) Y una feature general útil. **Alta prioridad.**
2. **CLI navaja-suiza** (de seqkit): unificar operaciones bajo `bioforge <subcomando>`
   (stats, subseq, translate, sliding, sample, grep), streaming, pipe-friendly.
3. **Workflow local** (de UGENE): un "pipeline" fino que encadena
   import→map→MSA→predict en local, offline.
4. **Recursos educativos** (de scikit-bio): tutoriales/notebooks — encaja con el nicho
   accesibilidad.
5. **Arquitectura de subpaquetes limpia** (de Biotite): ya la tenemos; mantener la
   disciplina al crecer.

---

## Lo honesto (para no engañarnos)

Estos generalistas son **maduros y enormes** (Biopython = décadas, cientos de
contribuidores; miles de formatos y utilidades de borde). **No vamos a igualar su
amplitud.** Nuestra ventaja no es "más features que Biopython" — es la **fusión
específica** de sus mejores rasgos en una base limpia + los **dos frentes que ellos
no tienen** (mapeo competitivo + evolución) + honestidad + edge. No prometer
reemplazar a Biopython; prometer ser **la caja integrada, rápida y honesta** para
nuestro nicho.

---

## Fuentes

- Biotite: *a unifying open source computational biology framework in Python.* BMC
  Bioinformatics 2018. https://link.springer.com/article/10.1186/s12859-018-2367-z ·
  *new tools…* 2023. https://link.springer.com/article/10.1186/s12859-023-05345-6
- SeqKit2: *A Swiss Army Knife for Sequence and Alignment Processing.*
  https://bioinf.shenwei.me/seqkit/ · https://github.com/shenwei356/seqkit
- scikit-bio: *a fundamental Python library for biological omic data analysis.*
  Nature Methods 2025. https://www.nature.com/articles/s41592-025-02981-z ·
  https://scikit.bio/
- Biopython: https://biopython.org/
- EMBOSS: https://emboss.sourceforge.net/
- Unipro UGENE: *a unified bioinformatics toolkit.* Bioinformatics 2012.
  https://academic.oup.com/bioinformatics/article/28/8/1166/195474 · http://ugene.net/
- Geneious: https://www.geneious.com/
</content>
</invoke>
