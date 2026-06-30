"""
engine/build.py — Compila el motor C a DLL/SO (Windows / Linux / macOS).

Uso:
    python engine/build.py

Detecta el compilador automáticamente (GCC, incluida la ubicación típica de
MSYS2 en Windows; clang en macOS) e intenta enlazar zlib (.gz) y libdeflate
(.gz ~2× más rápido). Cada dependencia degrada con gracia si no está.

Variables de entorno
────────────────────
  BIOFORGE_PORTABLE=1   Compila para CPU genérica (sin `-march=native`).
                        OBLIGATORIO al construir wheels que se distribuyen:
                        `-march=native` genera código para la CPU exacta de la
                        máquina que compila y crashea ("instrucción ilegal") en
                        CPUs distintas. En tu propio PC, sin esta variable, se
                        usa `-march=native` para máxima velocidad local.
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


def _portable() -> bool:
    return os.environ.get("BIOFORGE_PORTABLE", "") not in ("", "0", "false")


def _find_compiler() -> str | None:
    """GCC en Windows/Linux; clang (o gcc) en macOS."""
    if sys.platform == "darwin":
        return shutil.which("clang") or shutil.which("gcc")
    found = shutil.which("gcc")
    if found:
        return found
    if sys.platform == "win32":
        for cand in _WIN_GCC_CANDIDATES:
            if Path(cand).exists():
                return cand
    return None


def _brew_prefix(pkg: str) -> str | None:
    """Prefijo de instalación de un paquete Homebrew (macOS), o None."""
    try:
        res = subprocess.run(["brew", "--prefix", pkg],
                             capture_output=True, text=True)
        if res.returncode == 0:
            p = res.stdout.strip()
            return p if p and Path(p).exists() else None
    except FileNotFoundError:
        pass
    return None


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    # En Windows, MSYS2 necesita su carpeta bin en el PATH para emitir errores
    # y para que el .a de zlib se resuelva.
    env = os.environ.copy()
    cc = cmd[0]
    if sys.platform == "win32" and cc.lower().endswith("gcc.exe"):
        env["PATH"] = str(Path(cc).parent) + os.pathsep + env.get("PATH", "")
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def build() -> bool:
    cc = _find_compiler()
    if not cc:
        print("[ERROR] No se encontró compilador. Instala MinGW-w64 (Windows), "
              "gcc (Linux) o Xcode/clang (macOS) y vuelve a intentarlo.")
        return False

    # Arquitectura: nativa para uso local (máxima velocidad), genérica para
    # wheels distribuidos (compatibilidad con cualquier CPU).
    arch = ["-O3"] if _portable() else ["-O3", "-march=native"]

    # ── Banderas OpenMP + salida + enlace de zlib/libdeflate, por plataforma ──
    omp = ["-fopenmp"]
    extra_inc: list[str] = []
    extra_lib: list[str] = []

    if sys.platform == "win32":
        out = ENGINE_DIR / "engine.dll"
        # -static enlaza libgomp (OpenMP), libgcc y winpthread DENTRO del DLL,
        # para que no dependa de libgomp-1.dll en tiempo de carga.
        base = [cc, *arch, *omp, "-shared", "-static"]
        zlib_link = ["-l:libz.a"]         # estático → DLL autocontenido
        deflate_link = ["-l:libdeflate.a"]
    elif sys.platform == "darwin":
        out = ENGINE_DIR / "engine.so"
        base = [cc, *arch, "-shared", "-fPIC"]
        # clang no trae OpenMP: se usa libomp de Homebrew.
        libomp = _brew_prefix("libomp")
        if libomp:
            omp = ["-Xpreprocessor", "-fopenmp"]
            extra_inc += [f"-I{libomp}/include"]
            extra_lib += [f"-L{libomp}/lib", "-lomp"]
        else:
            print("[aviso] libomp (Homebrew) no encontrado; sin OpenMP "
                  "(el motor funciona, sin parseo paralelo).")
            omp = []
        # libdeflate/zlib de Homebrew si están.
        deflate_pfx = _brew_prefix("libdeflate")
        zlib_pfx = _brew_prefix("zlib")
        if deflate_pfx:
            extra_inc += [f"-I{deflate_pfx}/include"]
            extra_lib += [f"-L{deflate_pfx}/lib"]
        if zlib_pfx:
            extra_inc += [f"-I{zlib_pfx}/include"]
            extra_lib += [f"-L{zlib_pfx}/lib"]
        base = [cc, *arch, *omp, "-shared", "-fPIC"]
        zlib_link = ["-lz"]
        deflate_link = ["-ldeflate"]
    else:  # Linux y otros Unix
        out = ENGINE_DIR / "engine.so"
        base = [cc, *arch, *omp, "-shared", "-fPIC"]
        zlib_link = ["-lz"]               # zlib es estándar del sistema
        deflate_link = ["-ldeflate"]

    common = base + extra_inc + ["-o", str(out), str(SRC)] + extra_lib

    # Se intenta de más a menos capaz; cada intento degrada con gracia.
    #   1. zlib (.gz) + libdeflate (.gz ~2× más rápido)
    #   2. zlib (.gz) solo
    #   3. sin nada (archivos planos siguen funcionando)
    attempts = [
        ("CON .gz + libdeflate (rápido)",
         ["-DBIO_USE_ZLIB", "-DBIO_USE_LIBDEFLATE"] + zlib_link + deflate_link),
        ("CON .gz (zlib)",      ["-DBIO_USE_ZLIB"] + zlib_link),
        ("SIN soporte .gz",     []),
    ]

    mode = "portátil/genérica" if _portable() else "nativa (-march=native)"
    print(f"Compilando {SRC.name} -> {out.name}  [arquitectura {mode}] ...")
    last = None
    for label, link in attempts:
        res = _run(common + link)
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
