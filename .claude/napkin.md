# Napkin Runbook — Motor Bioinformático 5-bit

## Curation Rules
- Re-prioritize on every read. Max 10 items per category.
- Each item: date + rule + "Do instead" action.

## User Directives (Highest Priority)
1. **[2026-05-31] Checkpoint antes de cualquier cambio**
   Do instead: `git add -A && git commit -m "checkpoint: antes de <cambio>"` SIEMPRE antes de tocar código.

2. **[2026-05-31] Mutaciones solo a nivel de aminoácido**
   Do instead: ignorar cambios sinónimos en ADN. Solo reportar lo que cambia el aminoácido.

3. **[2026-05-31] Cero loops Python en la ruta crítica**
   Do instead: usar operaciones NumPy vectorizadas. Loops solo permitidos en visor.py, traceback del alineador (O(m+n)), y tools/.

## Execution & Validation
1. **[2026-05-31] Verificar tests antes de commit**
   Do instead: `pytest tests/ -q` — si falla algo, no hacer commit hasta resolverlo.

2. **[2026-05-31] Consola Windows no soporta caracteres Unicode ═ ─**
   Do instead: añadir `sys.stdout.reconfigure(encoding="utf-8")` al inicio de cada bloque `if __name__ == "__main__"`.

3. **[2026-05-31] pytest no encuentra biocore sin conftest.py**
   Do instead: conftest.py en raíz ya lo resuelve — no eliminar ese archivo.

4. **[2026-05-31] Comandos Bash (del, 2>$null) no funcionan en PowerShell**
   Do instead: usar PowerShell nativo: `Remove-Item`, `Out-Null`, etc.

## Domain Behavior Guardrails
1. **[2026-05-31] Auto-detección de tipo falla en proteínas sin E/F/I/L/P/Q/***
   Do instead: pasar siempre `force_type=SeqType.PROTEIN` cuando se sabe que es proteína.

2. **[2026-05-31] Alineador no apto para secuencias > 15.000 símbolos**
   Do instead: emite UserWarning automáticamente. Para genomas usar banded NW (no implementado).

3. **[2026-05-31] Número de ahorro de memoria es 37.5%, no 60-70%**
   Do instead: citar siempre 37.5% o memory_ratio=0.625. El 60-70% sería 2-bit packing, no 5-bit.

## Shell & Command Reliability
1. **[2026-05-31] Git push falla si GitHub tiene commits que no tenemos**
   Do instead: usar `git push --force` solo cuando el repositorio remoto se inicializó con archivos automáticos de GitHub (README, .gitignore). No en otros casos.

2. **[2026-05-31] tools/ necesitan sys.path.insert para importar biocore**
   Do instead: ya está puesto en visor.py, stress_test.py y comparador.py — no quitarlo.
