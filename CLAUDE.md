# BioForge — Contexto para Claude Code

## Qué es este proyecto

BioForge: motor bioinformático de alto rendimiento para Edge Computing (hardware limitado).
Sin Biopython. Basado exclusivamente en NumPy. Python 3.13, Windows 10.

Tres niveles implementados y validados:
- **L1** `biocore.py` — almacenamiento 5-bit, LUTs, BitPacker, PackedSequence, SmartImporter
- **L2** `smart_translator.py` — traducción ADN→Proteína vectorizada (CODON_LUT + sliding_window_view)
- **L3** `aligner.py` — Needleman-Wunsch con vectorización anti-diagonal (wavefront)

Documentación detallada en `docs/`.

---

## Reglas de oro — OBLIGATORIAS en todo el código del motor

### 1. Cero loops Python en la ruta crítica
**Prohibido** en `biocore.py`, `smart_translator.py`, y `_fill_matrix` de `aligner.py`:
- Cualquier `for` o `while` que itere símbolo a símbolo o celda a celda.

**Obligatorio:** operaciones NumPy vectorizadas — fancy indexing, `packbits`, `unpackbits`,
`where`, `sliding_window_view`, `argmax`, `bincount`, etc.

### 2. Excepciones conocidas y aceptadas
- `visor.py` — loops permitidos (frontend de display, no procesamiento)
- `aligner._traceback` — loop O(m+n) permitido (dependencia de datos inevitable)
- `aligner._fill_matrix` — UN loop O(m+n) sobre anti-diagonales (no O(m·n))

### 3. Nunca almacenar secuencias como str
Las secuencias biológicas existen únicamente como `PackedSequence` con `data` uint8
write-locked. El único lugar donde existe un `str` de secuencia es dentro de
`SmartImporter._encode()` como variable local temporal.

### 4. Force_type para proteínas sin marcadores obvios
Si una proteína no tiene E/F/I/L/P/Q/* en su secuencia, la auto-detección la
clasificará silenciosamente como ADN. Usar siempre `force_type=SeqType.PROTEIN`.

### 5. Benchmark después de cada optimización
Ejecutar `python stress_test.py` antes y después de cualquier cambio en el motor
para verificar que no empeora RAM ni velocidad.

---

## Números correctos del proyecto

| Métrica | Valor correcto |
|---------|---------------|
| Ahorro de memoria (5-bit) | **37.5%** (memory_ratio = 0.625) |
| RAM para 30M bases | **18.75 MB** |
| Throughput traducción | **~5 M aa/s** |
| Benchmark alineador 1000×1000 nt | **~165 ms** |

⚠️ El resumen ejecutivo original cita "60-70%" — ese número es incorrecto.
Correspondería a 2-bit packing, no al esquema 5-bit implementado.

---

## Estructura de archivos

```
biocore.py            L1 — almacenamiento (no modificar sin impacto en todo)
smart_translator.py   L2 — traducción ADN→Proteína
aligner.py            L3 — alineamiento NW + detección de mutaciones
visor.py              frontend interactivo (loops permitidos aquí)
stress_test.py        benchmark de 30M bases
docs/
  architecture.md     arquitectura detallada por niveles
  api_reference.md    ejemplos de uso de todos los módulos
  benchmarks.md       métricas reales y correcciones
  roadmap.md          estado, extensiones pendientes, decisiones cerradas
```

---

## Limitaciones conocidas del estado actual

- Alineador: solo viable para secuencias ≤ 15 000 símbolos (O(m·n) RAM)
- Traductor: solo 1 frame (hebra directa, primer ATG). Sin 6 frames ni reverse complement.
- Auto-detección de tipo: puede fallar en proteínas sin residuos exclusivos

---

## Próximas extensiones priorizadas

1. Banded NW — para secuencias > 15 000 bp
2. Reverse complement vectorizado — prerequisito para 6-frame translation
3. 6-frame translation

---

## Compatibilidad con consola Windows

Añadir al inicio de cada bloque `if __name__ == "__main__"`:
```python
import sys
sys.stdout.reconfigure(encoding="utf-8")
```
La consola Windows usa cp1252 y no puede mostrar caracteres ═ ─ sin esto.
