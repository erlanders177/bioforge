"""
engine/build.py — Compila el motor C a DLL/SO.

Uso:
    python engine/build.py

Detecta GCC automáticamente (incluida la ubicación típica de MSYS2 en Windows)
e intenta enlazar zlib para soporte de archivos .gz. Si zlib no está disponible,
compila sin él (los archivos planos siguen funcionando).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).parent
SRC = ENGINE_DIR / "engine.c"

# Ubicaciones habituales de GCC en Windows (MSYS2) si no está en el PATH.
_WIN_GCC_CANDIDATES = [
    r"C:\msys64\mingw64\bin\gcc.exe",
    r"C:\msys64\ucrt64\bin\gcc.exe",
    r"C:\mingw64\bin\gcc.exe",
]


def _find_gcc() -> str | None:
    found = shutil.which("gcc")
    if found:
        return found
    if sys.platform == "win32":
        for cand in _WIN_GCC_CANDIDATES:
            if Path(cand).exists():
                return cand
    return None


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    # En Windows, MSYS2 necesita su carpeta bin en el PATH para emitir errores
    # y para que el .a de zlib se resuelva.
    env = os.environ.copy()
    gcc = cmd[0]
    if sys.platform == "win32" and gcc.lower().endswith("gcc.exe"):
        env["PATH"] = str(Path(gcc).parent) + os.pathsep + env.get("PATH", "")
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def build() -> bool:
    gcc = _find_gcc()
    if not gcc:
        print("[ERROR] No se encontró GCC. Instala MinGW-w64 (Windows) o "
              "gcc (Linux/Mac) y vuelve a intentarlo.")
        return False

    base = [gcc, "-O3", "-march=native", "-fopenmp", "-shared"]
    if sys.platform == "win32":
        out = ENGINE_DIR / "engine.dll"
        zlib_link = ["-l:libz.a"]   # estático → DLL autocontenido, sin zlib1.dll
        # -static enlaza libgomp (OpenMP), libgcc y winpthread DENTRO del DLL,
        # para que no dependa de libgomp-1.dll en tiempo de carga.
        base += ["-static"]
    else:
        out = ENGINE_DIR / "engine.so"
        base += ["-fPIC"]
        zlib_link = ["-lz"]         # zlib es estándar del sistema en Linux/Mac

    common = base + ["-o", str(out), str(SRC)]

    if sys.platform == "win32":
        deflate_link = ["-l:libdeflate.a"]
    else:
        deflate_link = ["-ldeflate"]

    # Se intenta de más a menos capaz; cada intento degrada con gracia.
    #   1. zlib (.gz) + libdeflate (.gz ~2x más rápido)
    #   2. zlib (.gz) solo
    #   3. sin nada (archivos planos siguen funcionando)
    attempts = [
        ("CON .gz + libdeflate (rápido)",
         ["-DBIO_USE_ZLIB", "-DBIO_USE_LIBDEFLATE"] + zlib_link + deflate_link),
        ("CON .gz (zlib)",      ["-DBIO_USE_ZLIB"] + zlib_link),
        ("SIN soporte .gz",     []),
    ]

    print(f"Compilando {SRC.name} -> {out.name} ...")
    last = None
    for label, extra in attempts:
        res = _run(common + extra)
        last = res
        if res.returncode == 0:
            size_kb = out.stat().st_size // 1024
            print(f"[OK] Motor compilado [{label}]: {out}  ({size_kb} KB)")
            return True
        print(f"[aviso] No se pudo compilar [{label}]; probando alternativa.")

    print("[ERROR] Error de compilacion:")
    print(last.stderr if last else "(sin salida)")
    return False


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
