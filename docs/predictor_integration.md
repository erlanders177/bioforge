# Predictor de evolución — diseño de INTEGRACIÓN (2026-07-10)

Cómo lo hacen los mejores del mundo *dentro del campo de predicción de evolución*
—especialistas de un virus Y generalistas de todos los genomas—, cómo fusionarlos
en UN predictor, y cómo mejorarlos aprovechando que partimos de cero. Meta: máxima
precisión + velocidad en pocos recursos. Honestidad primero: no inventamos la
ciencia, la **integramos** de una forma que nadie ha hecho.

---

## El insight clave: 3 ejes complementarios, y nadie fusiona los dos primeros

Cada método puntero mide un eje distinto del fitness. **Son complementarios, no
rivales.** Y el hueco real: los de dinámica temporal y los de fitness-de-secuencia
**viven separados** — nadie los une limpio.

| Eje | Pregunta que responde | Quién lo hace | Necesita |
|-----|----------------------|---------------|----------|
| **A. Dinámica de frecuencia** | ¿qué está subiendo YA? | evofr (MLR/GARW), beth-1 | datos de vigilancia fechados |
| **B. Fitness de secuencia** | ¿qué es viable aunque NO se haya visto? | EVE, ESM-2 (PLM) | solo secuencias evolutivas |
| **C. Escape antigénico** | ¿qué escapa a la inmunidad? | EVEscape, Łuksza-Lässig | biofísica / estructura (opcional) |

- **Eje A** es fuerte a corto plazo pero **ciego a lo nuevo** (arranque en frío).
- **Eje B** resuelve el arranque en frío y da **alerta temprana** (pre-vigilancia).
- **Eje C** captura la presión selectiva que *dirige* el cambio.

**El especialista que nadie ha construido = fusionar los tres.** EVEscape ya fusiona
B+C (producto de términos) pero **carece del eje A** (es escape pre-vigilancia).
evofr domina A pero ignora B y C. **Unir A+B+C es el terreno vacío.**

---

## Cómo funciona cada uno (mecanismo, para poder robarlo)

**evofr — Bedford Lab (eje A, generalista).** Fitness por dinámica de frecuencias.
- *MLR*: ventaja de crecimiento FIJA por variante, desde conteos de secuencias.
- *GARW* (growth advantage random walk): la ventaja **varía suavemente en el tiempo**
  (paseo aleatorio) → captura fitness cambiante. **Idea a robar: fitness dependiente
  del tiempo con incertidumbre.** Forecasts de corto plazo precisos, SARS-CoV-2+gripe.

**beth-1 — Nature Comms 2024 (eje A, por-sitio).** Fitness **por-sitio**, no por clado.
- Define *transition time*: cuánto tarda una mutación en llegar a frecuencia
  influyente. Proyecta el paisaje de fitness al futuro. **Supera en genetic matching.**
- **Idea a robar:** el *transition time* por sitio — más granular que el clado.

**EVEscape — Marks Lab, Nature 2023 (ejes B+C, generalista de virus).**
- `P(escape) = fitness × accessibility × dissimilarity`, estandarizados → logística
  con temperatura → log del producto.
  - *fitness*: EVE (modelo generativo no supervisado del efecto de mutación).
  - *accessibility*: número de contacto ponderado (WCN) desde estructura PDB.
  - *dissimilarity*: cambio de **carga + hidrofobicidad** mutante vs original.
- **Funciona pre-pandemia** (alerta temprana). Generaliza a gripe, VIH, Lassa, Nipah.
- **Ideas a robar:** (1) la **arquitectura modular producto-de-términos** —elegante,
  cada término de una fuente distinta; (2) `dissimilarity` es **GRATIS** (tablas de
  propiedades de aminoácidos, microsegundos, sin estructura) → oro para pocos recursos.

**ESM-2 pequeño (eje B, bajo recurso).** El efecto de mutación = razón de
log-verosimilitud mutante/original.
- ESM-2 **8M corre en CPU**, 5-10× más rápido que 150M. Perplejidad óptima 3-6.
- **Distilación**: 8M pasa de 65%→**88% AUC** (ClinVar). LoRA sube 8M 58%→62%.
- **Evidencia clave: bajo recurso ≠ baja precisión.** Un 8M destilado es preciso y
  corre en portátil. Esto **valida nuestro nicho.**

---

## La arquitectura BioForge: FITNESS MODULAR (fusión A+B+C)

Robamos la estructura **producto-de-términos de EVEscape** y le añadimos el **eje A
temporal** que a EVEscape le falta. Fitness de una variante/mutación:

```
fitness = w_A·(crecimiento)  +  w_B·(viabilidad)  +  w_C·(escape)      [log-lineal]
```

Cuatro términos **enchufables e independientes**, cada uno degrada con gracia si
falta su fuente de datos:

| Término | De dónde | Coste | Bajo recurso |
|---------|----------|-------|--------------|
| **1. crecimiento** (eje A) | evofr GARW: ventaja de crecimiento variable en el tiempo, de las trayectorias | NumPy barato | ✅ nativo |
| **2. transición** (eje A) | beth-1: *transition time* por sitio | NumPy barato | ✅ nativo |
| **3. viabilidad** (eje B) | ESM-2 8M destilado: log-verosimilitud de la mutación | pesado pero OPCIONAL | ✅ 8M en CPU + caché |
| **4. escape** (eje C) | EVEscape dissimilarity: Δcarga+Δhidrofobicidad (tablas) | **gratis** | ✅ microsegundos |

El núcleo (términos 1, 2, 4) corre **en una patata**. El término 3 (IA) es un
`bioforge[ai]` opcional. Estructura/accessibility (eje C completo) = opcional si hay
PDB; sin ella, degradamos al proxy dissimilarity gratis.

---

## Cómo lo MEJORAMOS (la vuelta de tuerca, ventaja de partir de 0)

1. **Fusión A+B+C que nadie tiene.** evofr (A) y EVEscape (B+C) viven separados.
   Nuestro fitness modular los une → precisión de dos frentes en un solo número.
2. **Bajo recurso POR DISEÑO.** Los labs top usan clústeres GPU. Nosotros: términos
   1,2,4 en NumPy/C SIMD (candidatos a bajar a C como el resto del motor), IA opcional
   y pequeña (8M destilado, CPU, embeddings **cacheados una vez**). Corre en edge.
3. **Backtesting como ÁRBITRO, horneado.** Cada término debe **ganarse su sitio**:
   solo entra si sube el skill vs la baseline ingenua **en ese genoma**. Términos que
   no ayudan se apagan (auto-ajuste por organismo). Es "descubrir de los datos" llevado
   a los pesos w_A/w_B/w_C.
4. **Genoma-agnóstico de verdad.** Los términos universales (1,2,4 + PLM) funcionan en
   cualquier virus/genoma; los de dominio (estructura) son opcionales. EVEscape necesita
   estructura; nosotros degradamos con gracia sin ella.
5. **Incertidumbre honesta.** GARW da crecimiento con bandas de confianza → propagamos
   → reportamos intervalos, no un punto. Honesto por construcción.

---

## Precisión + velocidad en pocos recursos (la evidencia de que es posible)

- `dissimilarity` (término 4): tablas de propiedades → **0 coste**, aporta señal C.
- ESM-2 8M destilado: **88% AUC** en CPU → precisión sin GPU.
- Todo lo demás: NumPy vectorizado / C SIMD (nuestro terreno) → microsegundos.
- La caché de embeddings del PLM convierte el único paso caro en un one-shot.

→ Objetivo realista: **precisión competitiva con un modelo modular fusionado**, en un
portátil, midiéndolo honestamente contra la baseline y contra evofr/EVEscape donde se
pueda comparar.

---

## Roadmap de integración (fases, cada una medida antes de avanzar)

- **F1 (hecho, en pausa):** trayectorias + baseline + backtesting = el árbitro.
- **F2:** término **crecimiento** (GARW) + término **transición** (beth-1) → primer
  fitness real, eje A. Medir skill vs naive en ≥3 genomas.
- **F3:** término **escape** gratis (dissimilarity) → eje C sin coste. Medir aporte.
- **F4:** término **viabilidad** ESM-2 8M destilado, opcional `bioforge[ai]` → eje B,
  arranque en frío. Medir si supera a F2+F3 (si no, no entra al núcleo).
- **F5:** fusión ponderada A+B+C con pesos auto-ajustados por backtesting + intervalos.

---

## Fuentes

- evofr / fitness models SARS-CoV-2 (MLR, FGA, GARW). PLOS Comp Biol 2024.
  https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012443 ·
  https://github.com/blab/evofr
- beth-1: *site-based dynamics of mutations.* Nature Comms 2024.
  https://www.nature.com/articles/s41467-024-46918-0 · https://github.com/mwanglab/beth-1
- EVEscape: *Learning from pre-pandemic data to forecast viral escape.* Nature 2023.
  https://www.nature.com/articles/s41586-023-06617-0 · https://github.com/OATML-Markslab/EVEscape
- Łuksza & Lässig. *A predictive fitness model for influenza.* Nature 2014.
- *Understanding Protein Language Model Scaling on Mutation Effect Prediction.*
  bioRxiv 2025. https://www.biorxiv.org/content/10.1101/2025.04.25.650688v1.full
- *Compressing the collective knowledge of ESM into a single PLM* (distilación).
  Nature Methods 2026. https://www.nature.com/articles/s41592-026-03050-9
</content>
