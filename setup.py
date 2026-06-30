"""
setup.py — Compila el motor C al construir el wheel y etiqueta el wheel como
específico de plataforma.

Casi toda la metadata vive en ``pyproject.toml`` (fuente de verdad). Este
archivo solo añade dos comportamientos que pyproject no puede expresar:

1. **Compilar el motor** (`engine.c` → `engine.dll` / `engine.so`) durante el
   build, de forma portátil (`BIOFORGE_PORTABLE=1`, sin `-march=native`) para
   que el binario funcione en cualquier CPU del usuario, no solo la del CI.
   Si la compilación falla pero ya existe un binario precompilado (p.ej. el
   `engine.dll` versionado en git para Windows), se conserva ese — degradación
   con gracia, nunca rompe el build.

2. **Etiquetar el wheel por plataforma** como ``py3-none-<plataforma>``. El
   motor se carga con ctypes y no depende de la ABI de Python, así que un único
   wheel por plataforma vale para todas las versiones de Python 3. Sin esto,
   setuptools generaría un wheel ``py3-none-any`` (universal) que llevaría el
   binario equivocado a otros sistemas.
"""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

_ROOT = Path(__file__).parent
_BUILD_ENGINE = _ROOT / "bioforge" / "engine" / "build.py"
_ENGINE_DIR = _ROOT / "bioforge" / "engine"


def _engine_binary_exists() -> bool:
    return any(_ENGINE_DIR.glob("engine.dll")) or any(_ENGINE_DIR.glob("engine.so"))


class BuildWithEngine(build_py):
    """Compila el motor C antes de copiar los datos del paquete."""

    def run(self) -> None:
        # Windows: reutilizar SIEMPRE el engine.dll precompilado y autocontenido
        # (versionado en git). Recompilar en CI es arriesgado —los runners traen
        # MSYS2 pero no libz.a/libdeflate.a, lo que daría un DLL sin .gz. Para
        # rebuild local tras tocar engine.c se usa `python engine/build.py`.
        force = os.environ.get("BIOFORGE_FORCE_BUILD", "") not in ("", "0", "false")
        if sys.platform == "win32" and any(_ENGINE_DIR.glob("engine.dll")) \
                and not force:
            print(">> Windows: se usa el engine.dll precompilado (sin recompilar).")
            super().run()
            return

        env = os.environ.copy()
        env["BIOFORGE_PORTABLE"] = "1"   # wheels distribuidos → CPU genérica
        print(">> Compilando el motor C (portátil) para el wheel...")
        try:
            res = subprocess.run([sys.executable, str(_BUILD_ENGINE)], env=env)
            ok = res.returncode == 0
        except Exception as exc:                    # noqa: BLE001
            print(f">> No se pudo lanzar la compilación: {exc}")
            ok = False

        if not ok:
            if _engine_binary_exists():
                print(">> Compilación no disponible; se usa el binario "
                      "precompilado existente (p.ej. engine.dll versionado).")
            else:
                print(">> AVISO: sin motor C compilado; el wheel funcionará con "
                      "el fallback NumPy (más lento).")
        super().run()


# ── Etiquetar el wheel como específico de plataforma (py3-none-<plat>) ──────────
cmdclass = {"build_py": BuildWithEngine}

try:
    # setuptools recientes traen bdist_wheel; versiones viejas lo toman de wheel.
    try:
        from setuptools.command.bdist_wheel import bdist_wheel
    except ImportError:
        from wheel.bdist_wheel import bdist_wheel

    class BDistWheel(bdist_wheel):
        def finalize_options(self) -> None:
            super().finalize_options()
            self.root_is_pure = False        # contiene un binario → no es "pure"

        def get_tag(self):
            # ABI-independiente (ctypes): py3-none-<plataforma>.
            _python, _abi, plat = super().get_tag()
            return "py3", "none", plat

    cmdclass["bdist_wheel"] = BDistWheel
except ImportError:
    # Sin wheel/bdist_wheel disponible (p.ej. solo sdist); no pasa nada.
    pass


setup(cmdclass=cmdclass)
