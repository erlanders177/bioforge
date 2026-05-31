# Roadmap del Proyecto

## Estado actual (2026-05-31)

| Nivel | Módulo | Descripción | Estado |
|-------|--------|-------------|--------|
| L1 | biocore.py | Almacenamiento 5-bit, FASTA parser, LUTs | ✅ Completo |
| L2 | smart_translator.py | Traducción ADN→Proteína vectorizada | ✅ Completo |
| L3 | aligner.py | Alineamiento NW + detección de mutaciones | ✅ Completo |
| — | visor.py | Frontend de display interactivo | ✅ Funcional |
| — | stress_test.py | Benchmark de rendimiento con 30M bases | ✅ Funcional |

---

## Próximas extensiones (priorizadas)

### Alta prioridad

**Banded Needleman-Wunsch**
- Problema: el alineador actual requiere O(m·n) RAM → inviable para secuencias largas.
- Solución: solo rellenar una banda de anchura `k` alrededor de la diagonal principal.
- Beneficio: memoria O((m+n)·k), tiempo O((m+n)·k).
- Cuándo implementar: cuando haya necesidad de comparar secuencias > 15 000 bp.

**Reverse complement vectorizado**
- Función `reverse_complement(seq: PackedSequence) → PackedSequence` en biocore.
- Implementación: flip del array + LUT de complemento (A↔T, C↔G).
- Sin loops Python. Prerequisito para 6-frame translation.

**6-frame translation**
- Traducir los 6 marcos de lectura (3 directos + 3 en reverse complement).
- Depende del reverse complement implementado.
- Útil para anotar genes en secuencias genómicas.

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
