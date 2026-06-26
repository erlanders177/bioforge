# Roadmap del Proyecto

## Estado actual (2026-06-27) — v1.1.0

| Nivel | Módulo | Descripción | Estado |
|-------|--------|-------------|--------|
| L1 | biocore.py | Almacenamiento 5-bit, FASTA parser, LUTs, reverse complement | ✅ Completo |
| L2 | smart_translator.py | Traducción ADN→Proteína, 6-frame translation | ✅ Completo |
| L3 | aligner.py | NW global/semi-global, banded NW, Smith-Waterman local | ✅ Completo |
| — | visor.py | Frontend de display interactivo | ✅ Funcional |
| — | stress_test.py | Benchmark de rendimiento con 30M bases | ✅ Funcional |

### Completado en v1.1.0

**Reverse complement vectorizado** ✅
- `PackedSequence.reverse_complement()` — dos ops NumPy: `_NUC_COMPLEMENT` LUT + `np.flip`.
- Sin loops Python. RC(RC(x)) == x garantizado.

**6-frame translation** ✅
- `SmartTranslator.translate_all_frames(seq)` — devuelve lista de proteínas por marco.
- Frames +1/+2/+3 (hebra directa) y -1/-2/-3 (reverse complement).
- Frames sin ATG omitidos silenciosamente.

**Banded Needleman-Wunsch** ✅
- `SequenceAligner.align(seq_a, seq_b, band=N)`.
- Motor C: O(m·N) memoria real. Fallback NumPy: máscara NEG_INF sobre matriz completa.

**Smith-Waterman (alineamiento local)** ✅
- `SequenceAligner.align_local(seq_a, seq_b)`.
- Para dominios/motivos o secuencias con flancos no homólogos.

---

## Próximas extensiones (priorizadas)

### Alta prioridad

**FASTA export**
- `to_fasta(seq, line_width=60) → str` — genera texto FASTA bien formateado.
- Trivial de implementar, útil para interoperabilidad con otras herramientas.

### Media prioridad

**FASTA export**
- `to_fasta(seq, line_width=60) → str` — genera texto FASTA bien formateado.
- Trivial de implementar, pero útil para interoperabilidad.

**Detección de ORFs múltiples**
- Encontrar todos los ATG en los 6 frames con sus ORFs completos.
- Devolver lista de `PackedSequence(PROTEIN)` candidatos, ordenados por longitud.

### Baja prioridad

**Almacenamiento 2-bit para ADN puro**
- Para secuencias sin ambigüedades ni gaps: A=00, C=01, G=10, T=11.
- Daría 75% de ahorro vs ASCII (frente al 37.5% actual).
- Requiere un segundo tipo de contenedor o un flag en PackedSequence.
- Solo merece la pena si el footprint de RAM se convierte en el cuello de botella real.

**Lectura de archivos .gz**
- `SmartImporter.from_file("archivo.fa.gz")` sin descomprimir manualmente.
- Implementación: `gzip.open()` en lugar de `open()`.

---

## Decisiones de diseño cerradas

Estas decisiones fueron tomadas y no deben reabrirse sin un caso de uso
concreto que las justifique:

| Decisión | Alternativa descartada | Razón |
|----------|----------------------|-------|
| Alfabeto 5-bit unificado | 2-bit solo para ADN | Unificación de pipeline nucleótido+proteína |
| NW global para mutaciones | Smith-Waterman (local) | SW busca motivos; NW compara alelos |
| NumPy puro (sin Numba/Cython) | Numba JIT | Portabilidad, cero dependencias extra |
| Anti-diagonal wavefront en NW | Loops O(m·n) Python | Regla de oro del proyecto |
