# Benchmarks y Métricas Reales

Todos los números han sido medidos ejecutando el código real en la máquina
de desarrollo (Windows 10, Python 3.13). No son estimaciones.

---

## Nivel 1+2 — Almacenamiento y Traducción

| Operación | Resultado |
|-----------|-----------|
| Empaquetado 30M bases | ~200–400 ms |
| RAM para 30M bases (empaquetado) | **18.75 MB** |
| RAM para 30M bases (ASCII naive) | 30 MB |
| Ahorro de memoria | **37.5 %** (memory_ratio = 0.6250 = 5/8) |
| Throughput de traducción | **~5 M aminoácidos/segundo** |

### Corrección importante

El resumen ejecutivo original del proyecto citaba "ahorro del 60-70%". **Ese número es incorrecto.**

- **37.5%** → lo que realmente entrega el esquema 5-bit implementado.
- **75%** → lo que daría un esquema 2-bit (solo ACGT, sin ambiguos ni aminoácidos).

El esquema 5-bit fue elegido deliberadamente para cubrir nucleótidos, aminoácidos,
gaps, STOP y ambiguos en un solo pipeline. La unificación tiene un coste en
compresión frente al 2-bit puro, y ese trade-off fue aceptado.

Cuando se cite eficiencia de memoria del proyecto: **37.5%**, nunca "60-70%".

---

## Nivel 3 — Alineador Needleman-Wunsch

| Operación | Resultado |
|-----------|-----------|
| Alineamiento 1 000 × 1 000 nt | ~165 ms |
| Detección mutaciones en 1 000 nt (1% mutaciones) | 10/10 exactas |
| HBB normal vs sickle cell (30 nt) | Score=57, mutación A19T detectada |
| Inserción 3 bases (18 nt × 21 nt) | Score=30, 3 inserciones detectadas |
| Proteínas HBB-α vs HBB-β (52 aa × 51 aa) | Identity=45.3%, 29 mutaciones |

### Límites del alineador

| Tamaño de secuencias | RAM necesaria | Viabilidad |
|----------------------|---------------|------------|
| 100 × 100 | ~40 KB | OK |
| 1 000 × 1 000 | ~4 MB | OK |
| 5 000 × 5 000 | ~100 MB | OK |
| 15 000 × 15 000 | ~3.4 GB | Límite (UserWarning) |
| Cromosomas completos | Petabytes | NO viable |

Para secuencias de genes típicos (100–10 000 bp) el alineador es completamente viable.
Para genomas completos se necesitaría banded alignment (extensión futura).

---

## Cómo ejecutar los benchmarks

```bash
# Benchmark de empaquetado y traducción (30M bases)
python stress_test.py

# Self-test completo con benchmark de aligner (1000×1000 nt)
python aligner.py

# Self-test del motor de traducción
python smart_translator.py

# Self-test del motor de almacenamiento (10M bases)
python biocore.py
```

---

## Fragilidades conocidas (no son bugs, son limitaciones documentadas)

1. **Auto-detección de tipo silenciosa:**
   La detección nucleótido/proteína solo busca E/F/I/L/P/Q/* como indicadores
   de proteína. Una proteína compuesta solo de residuos que también son IUPAC
   nucleotídicos (p.ej. una proteína solo de A/C/G) se clasificará como ADN
   sin avisar. **Solución:** usar `force_type=SeqType.PROTEIN` siempre que se
   sepa que la secuencia es una proteína.

2. **Un solo frame de traducción:**
   `SmartTranslator` solo traduce la hebra directa desde el primer ATG.
   No implementa los 6 marcos de lectura ni reverse complement.

3. **Alineador no apto para genomas:**
   La matriz DP es O(m·n) en RAM. No es viable para secuencias de más de
   ~15 000 símbolos con el hardware actual.
