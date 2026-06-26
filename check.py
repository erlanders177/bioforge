"""
check.py — Verificador del motor bioinformático

Comprueba que todo el proyecto funciona correctamente.
No necesitas saber programación para usarlo.

Cómo usarlo:
    python check.py

Qué significa el resultado:
    ✅  Todo funciona correctamente.
    ❌  Algo está roto. El programa te dirá qué y cómo solucionarlo.
"""

import subprocess
import sys
import time


# ── Colores para la terminal ───────────────────────────────────────────────────
_VERDE   = "\033[92m"
_ROJO    = "\033[91m"
_AMARILLO= "\033[93m"
_NEGRITA = "\033[1m"
_RESET   = "\033[0m"

def _v(t): return f"{_VERDE}{t}{_RESET}"
def _r(t): return f"{_ROJO}{t}{_RESET}"
def _a(t): return f"{_AMARILLO}{t}{_RESET}"
def _n(t): return f"{_NEGRITA}{t}{_RESET}"


# ── Descripción humana de cada suite de tests ──────────────────────────────────
_SUITES = [
    (
        "Almacenamiento de secuencias",
        "tests/test_biocore.py",
        (
            "Comprueba que el motor guarda las secuencias correctamente en memoria,\n"
            "  que el empaquetado 5-bit funciona, que los archivos FASTA se leen bien\n"
            "  y que la compresión de datos produce los resultados esperados."
        ),
    ),
    (
        "Traducción de ADN a proteína",
        "tests/test_translator.py",
        (
            "Verifica que los 61 codones del código genético se traducen al\n"
            "  aminoácido correcto, que los 3 codones STOP detienen la traducción,\n"
            "  y que el motor detecta el punto de inicio (ATG) correctamente."
        ),
    ),
    (
        "Detección de mutaciones",
        "tests/test_aligner.py",
        (
            "Confirma que el alineador encuentra diferencias entre secuencias,\n"
            "  incluyendo la mutación de la anemia falciforme (Glu→Val),\n"
            "  inserciones, deleciones y propiedades matemáticas del algoritmo."
        ),
    ),
    (
        "Pipeline de análisis completo",
        "tests/test_analyze.py",
        (
            "Prueba el flujo completo de principio a fin: cargar FASTA,\n"
            "  traducir, alinear y generar el informe en los tres modos\n"
            "  (nucleótido, aminoácido, ambos). También verifica que las\n"
            "  mutaciones sinónimas se ignoran correctamente."
        ),
    ),
]


def _ejecutar_suite(archivo: str) -> tuple[bool, int, int, list[str]]:
    """Ejecuta una suite de tests. Devuelve (ok, pasados, fallados, errores)."""
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest", archivo,
            "-q", "--tb=line",
            "-k", "not benchmark",
            "--no-header",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    stdout = result.stdout
    pasados = 0
    fallados = 0
    errores_legibles: list[str] = []

    for linea in stdout.splitlines():
        if "passed" in linea:
            partes = linea.strip().split()
            for i, p in enumerate(partes):
                if p == "passed" and i > 0:
                    try:
                        pasados = int(partes[i - 1])
                    except ValueError:
                        pass
        if "failed" in linea and "passed" in linea:
            partes = linea.strip().split()
            for i, p in enumerate(partes):
                if p == "failed" and i > 0:
                    try:
                        fallados = int(partes[i - 1])
                    except ValueError:
                        pass
        if "FAILED" in linea or "AssertionError" in linea:
            errores_legibles.append(linea.strip())

    ok = result.returncode == 0
    return ok, pasados, fallados, errores_legibles


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    W = 62
    print()
    print(_n("  ╔" + "═" * W + "╗"))
    print(_n("  ║") + "  VERIFICACIÓN DEL MOTOR BIOINFORMÁTICO" + " " * (W - 39) + _n("║"))
    print(_n("  ║") + "  Aarón Aranda Torrijos · github.com/erlanders177" + " " * (W - 49) + _n("║"))
    print(_n("  ╚" + "═" * W + "╝"))
    print()
    print("  Comprobando que todo funciona correctamente...")
    print()

    total_pasados  = 0
    total_fallados = 0
    hubo_fallo     = False
    t_inicio       = time.perf_counter()

    for idx, (nombre, archivo, descripcion) in enumerate(_SUITES, 1):
        print(f"  [{idx}/{len(_SUITES)}] {_n(nombre)}")
        print(f"  {descripcion}")
        print(f"  Comprobando", end="", flush=True)

        ok, pasados, fallados, errores = _ejecutar_suite(archivo)

        print("\r", end="")   # borrar la línea "Comprobando..."

        if ok:
            print(f"  [{idx}/{len(_SUITES)}] {_n(nombre)}")
            print(f"  {descripcion}")
            print(f"  {_v(f'✅  {pasados} comprobaciones superadas')}")
        else:
            hubo_fallo = True
            print(f"  [{idx}/{len(_SUITES)}] {_n(nombre)}")
            print(f"  {descripcion}")
            print(f"  {_r(f'❌  FALLO — {fallados} comprobación(es) no superadas')}")
            if errores:
                print()
                print(f"  {_a('  Qué falló:')}")
                for e in errores[:5]:
                    print(f"  {_a('  →')} {e}")
                if len(errores) > 5:
                    print(f"  {_a(f'  ... y {len(errores)-5} más')}")
            print()
            print(f"  {_a('  Cómo ver el detalle técnico:')}")
            print(f"  {_a(f'  pytest {archivo} -v --tb=short')}")

        total_pasados  += pasados
        total_fallados += fallados
        print()

    t_total = time.perf_counter() - t_inicio

    print("  " + "─" * W)
    print()

    if not hubo_fallo:
        print(f"  {_v(_n('✅  TODO FUNCIONA CORRECTAMENTE'))}")
        print()
        print(f"  {_v(str(total_pasados))} comprobaciones superadas  ·  "
              f"0 fallos  ·  {t_total:.1f}s")
    else:
        print(f"  {_r(_n('❌  SE ENCONTRARON PROBLEMAS'))}")
        print()
        print(f"  {_v(str(total_pasados))} superadas  ·  "
              f"{_r(str(total_fallados))} falladas  ·  {t_total:.1f}s")
        print()
        print(f"  {_a('Para ver todos los errores con detalle:')}")
        print(f"  {_a('  pytest tests/ -v --tb=short -k \"not benchmark\"')}")

    print()
    print("  " + "─" * W)
    print()

    return 1 if hubo_fallo else 0


if __name__ == "__main__":
    sys.exit(main())
