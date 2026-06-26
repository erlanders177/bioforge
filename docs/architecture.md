# BioForge — Arquitectura

## Resumen ejecutivo

BioForge: motor bioinformático de alto rendimiento para Edge Computing, sin Biopython.
Basado exclusivamente en NumPy. Diseñado para ejecutarse en hardware de
recursos limitados (i5-U, dispositivos embebidos).

**Principio rector:** cero loops Python en la ruta crítica de cálculo.
Todo vectorizado con NumPy. Ver sección "Reglas de oro".

---

## Niveles de abstracción

```
┌─────────────────────────────────────────────────────────┐
│  Nivel 3 — aligner.py       Detección de mutaciones     │
│  Needleman-Wunsch anti-diagonal wavefront O(m+n)         │
├─────────────────────────────────────────────────────────┤
│  Nivel 2 — smart_translator.py   Traducción ADN→Prot    │
│  sliding_window_view + CODON_LUT base-4                  │
├─────────────────────────────────────────────────────────┤
│  Nivel 1 — biocore.py       Almacenamiento 5-bit        │
│  BitPacker · PackedSequence · SmartImporter · LUTs       │
└─────────────────────────────────────────────────────────┘
```

---

## Nivel 1 — Almacenamiento (biocore.py)

**Estado:** Completo y validado.

### Alfabeto unificado 5-bit (BioCode)

| Rango   | Contenido                          |
|---------|------------------------------------|
| 0–3     | Nucleótidos: A, C, G, T/U          |
| 4–23    | Aminoácidos (orden IUPAC)          |
| 24      | STOP codon / terminador de cadena  |
| 25      | GAP (alineamiento)                 |
| 26–30   | Reservados                         |
| 31      | UNK / ambiguo (N en ADN, X en prot)|

Un solo esquema cubre nucleótidos y proteínas. Coste: 37.5% de ahorro vs
ASCII. El 60-70% que citan algunos resúmenes es incorrecto — eso sería
2-bit packing, no 5-bit.

### Componentes

- **`NUC_LUT` / `AA_LUT`**: tablas 256 uint8 para traducir ASCII→BioCode en
  una sola indexación vectorizada sin loops.
- **`BitPacker`**: empaquetado/desempaquetado 5-bit con `np.packbits` /
  `np.unpackbits`. Sin loops Python.
- **`PackedSequence`**: contenedor inmutable. Array `data` write-locked tras
  construcción. Acceso O(1) por posición sin desempaquetar todo el array.
- **`SmartImporter`**: parser FASTA con auto-detección de tipo. Modo chunked
  para archivos que no caben en RAM.

---

## Nivel 2 — Traducción ADN→Proteína (smart_translator.py)

**Estado:** Completo y validado.

### Pipeline (cero loops Python)

```
① decode PackedSequence(NUC)  →  uint8 array [0–3]
② sliding_window_view         →  localizar primer ATG (Fail-Fast)
③ reshape (N, 3)              →  matriz de codones
④ idx = n1×16 + n2×4 + n3    →  índice base-4 vectorizado
⑤ CODON_LUT[idx]              →  array de BioCode aminoácidos
⑥ truncar en primer STOP      →  argmax sobre hits
⑦ UserWarning si < 50 aa      →  posible ncRNA o fragmento roto
⑧ BitPacker.pack              →  nuevo PackedSequence(PROTEIN)
```

**Limitaciones conocidas:**
- Solo 1 frame de lectura (hebra directa, primer ATG).
- Sin 6 frames ni reverse complement (extensión futura).

---

## Nivel 3 — Alineamiento y Mutaciones (aligner.py)

**Estado:** Completo y validado (2026-05-31).

### Por qué no se puede vectorizar completamente

La recurrencia de Needleman-Wunsch tiene dependencia de datos:

```
H[i,j] = max( H[i-1,j-1] + score,
               H[i-1,j  ] + gap,
               H[i  ,j-1] + gap )
```

Cada celda depende de tres vecinos → imposible vectorizar en 2D.

### Solución: wavefront anti-diagonal

Las celdas de la misma anti-diagonal (`i + j = d`) son **independientes entre sí**:

```
d=0: (0,0)
d=1: (0,1) (1,0)       ← inicialización
d=2: (1,1)             ← primer computable
d=3: (1,2) (2,1)
d=4: (1,3) (2,2) (3,1)
...
```

- **UN** loop Python externo: O(m+n) iteraciones sobre anti-diagonales.
- **NumPy vectorizado** dentro de cada anti-diagonal.
- Reducción: de O(m·n) loops Python a O(m+n).

El traceback es O(m+n) secuencial — inevitable, pero rápido (~2000 iters para 1000bp).

### Scoring

| Evento     | Puntuación |
|------------|-----------|
| Match      | +2        |
| Mismatch   | −1        |
| Gap (open) | −2        |

Modelo lineal (sin penalización de extensión diferenciada).

### Límite de secuencias

La matriz DP es O(m·n) int32 en RAM:
- 1 000 × 1 000 bp → ~4 MB → viable
- 15 000 × 15 000 bp → ~3.4 GB → límite (se emite UserWarning)
- Genomas completos → NO viable. Requiere banded alignment.

---

## Relaciones entre módulos

```
biocore.py  ←── smart_translator.py
biocore.py  ←── aligner.py
biocore.py  ←── visor.py
biocore.py  ←── stress_test.py
```

---

## Reglas de oro (obligatorias para contribuciones)

1. **Prohibido** cualquier loop `for`/`while` que itere símbolo a símbolo
   en biocore.py, smart_translator.py, o `_fill_matrix` de aligner.py.

2. **Permitido** en `_traceback` de aligner.py: un loop O(m+n).

3. **Permitido** en visor.py: loops para display de resultados al usuario.

4. **Nunca** almacenar secuencias biológicas como `str` Python después de
   la importación. Siempre usar `PackedSequence`.

5. **Siempre** ejecutar stress_test.py al optimizar funciones del motor.

6. Para proteínas sin E/F/I/L/P/Q/* en la secuencia, pasar siempre
   `force_type=SeqType.PROTEIN` explícitamente al importer.
