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
                        OBLIGATORIO al construir wheels distribuidos.
  BIOFORGE_STATIC=1     Enlace AUTOCONTENIDO para wheels: en Linux/macOS enlaza
                        OpenMP (libgomp/libomp) de forma estática para que el
                        .so no dependa de librerías fuera de la lista blanca de
                        manylinux. Así el paso "repair" (auditwheel/delocate)
                        sobra y se puede saltar. zlib se deja dinámico (está en
                        la lista blanca / es del sistema en macOS).
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


def _flag(name: str) -> bool:
    return os.environ.get(name, "") not in ("", "0", "false")


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
    env = os.environ.copy()
    cc = cmd[0]
    if sys.platform == "win32" and cc.lower().endswith("gcc.exe"):
        env["PATH"] = str(Path(cc).parent) + os.pathsep + env.get("PATH", "")
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def _arch() -> list[str]:
    # Nativa para uso local (máxima velocidad), genérica para wheels.
    return ["-O3"] if _flag("BIOFORGE_PORTABLE") else ["-O3", "-march=native"]


# ──────────────────────────────────────────────────────────────────────────────
# Ruta AUTOCONTENIDA para wheels (Linux/macOS): compilar + enlazar en dos pasos,
# con OpenMP estático. Devuelve True si produjo engine.so.
# ──────────────────────────────────────────────────────────────────────────────
def _build_static_unix(cc: str) -> bool:
    out = ENGINE_DIR / "engine.so"
    obj = ENGINE_DIR / "engine.o"
    is_mac = sys.platform == "darwin"
    arch = _arch()

    # ── Paso 1: compilar a objeto (pragmas OpenMP activos) ──────────────────────
    cflags = [cc, "-c", *arch, "-fPIC", "-DBIO_USE_ZLIB"]
    libomp = None
    if is_mac:
        libomp = _brew_prefix("libomp")
        if libomp:
            cflags += ["-Xpreprocessor", "-fopenmp", f"-I{libomp}/include"]
        else:
            print("[aviso] libomp no encontrado; motor sin parseo paralelo.")
    else:
        cflags += ["-fopenmp"]
    cflags += ["-o", str(obj), str(SRC)]

    print("Compilando engine.c -> engine.o (objeto) ...")
    res = _run(cflags)
    if res.returncode != 0:
        # Reintentar sin zlib (p.ej. faltan cabeceras de zlib).
        cflags = [c for c in cflags if c != "-DBIO_USE_ZLIB"]
        res = _run(cflags)
        if res.returncode != 0:
            print("[ERROR] No se pudo compilar el objeto:")
            print(res.stderr)
            return False
        zlib_on = False
    else:
        zlib_on = True

    # ── Paso 2: enlazar a .so autocontenido (OpenMP estático) ───────────────────
    link = [cc, "-shared", "-o", str(out), str(obj)]
    if is_mac:
        if libomp:
            link += [f"{libomp}/lib/libomp.a"]   # OpenMP estático (full path)
        if zlib_on:
            link += ["-lz"]                       # zlib del sistema (dinámico)
    else:
        link += ["-static-libgcc"]
        # libgomp ESTÁTICO (no está en la lista blanca de manylinux);
        # zlib y pthread DINÁMICOS (sí están en la lista blanca).
        link += ["-Wl,--whole-archive", "-l:libgomp.a", "-Wl,--no-whole-archive"]
        if zlib_on:
            link += ["-lz"]
        link += ["-lpthread", "-lm", "-ldl"]

    print("Enlazando engine.o -> engine.so (OpenMP estático) ...")
    res = _run(link)
    if res.returncode != 0 and not is_mac:
        # Fallback: libgomp dinámico (wheel funcionará donde haya libgomp.so.1).
        print("[aviso] Enlace estático de OpenMP falló; probando dinámico.")
        link = [cc, "-shared", "-o", str(out), str(obj), "-fopenmp"]
        if zlib_on:
            link += ["-lz"]
        res = _run(link)

    try:
        obj.unlink()
    except OSError:
        pass

    if res.returncode == 0 and out.exists():
        size_kb = out.stat().st_size // 1024
        omp = "con" if (is_mac and libomp) or not is_mac else "sin"
        gz = "con" if zlib_on else "sin"
        print(f"[OK] Motor autocontenido: {out} ({size_kb} KB) "
              f"[{omp} OpenMP · {gz} .gz]")
        return True

    print("[ERROR] Error de enlace:")
    print(res.stderr)
    return False


def build() -> bool:
    cc = _find_compiler()
    if not cc:
        print("[ERROR] No se encontró compilador. Instala MinGW-w64 (Windows), "
              "gcc (Linux) o Xcode/clang (macOS) y vuelve a intentarlo.")
        return False

    # Wheels distribuidos en Linux/macOS → ruta autocontenida (OpenMP estático).
    if _flag("BIOFORGE_STATIC") and sys.platform != "win32":
        return _build_static_unix(cc)

    arch = _arch()
    omp = ["-fopenmp"]

    if sys.platform == "win32":
        out = ENGINE_DIR / "engine.dll"
        # -static enlaza libgomp, libgcc y winpthread DENTRO del DLL.
        base = [cc, *arch, *omp, "-shared", "-static"]
        zlib_link = ["-l:libz.a"]
        deflate_link = ["-l:libdeflate.a"]
    elif sys.platform == "darwin":
        out = ENGINE_DIR / "engine.so"
        libomp = _brew_prefix("libomp")
        extra_inc: list[str] = []
        extra_lib: list[str] = []
        if libomp:
            omp = ["-Xpreprocessor", "-fopenmp"]
            extra_inc += [f"-I{libomp}/include"]
            extra_lib += [f"-L{libomp}/lib", "-lomp"]
        else:
            omp = []
        base = [cc, *arch, *omp, "-shared", "-fPIC", *extra_inc, *extra_lib]
        zlib_link = ["-lz"]
        deflate_link = ["-ldeflate"]
    else:  # Linux (uso local, no-wheel)
        out = ENGINE_DIR / "engine.so"
        base = [cc, *arch, *omp, "-shared", "-fPIC"]
        zlib_link = ["-lz"]
        deflate_link = ["-ldeflate"]

    common = base + ["-o", str(out), str(SRC)]
    attempts = [
        ("CON .gz + libdeflate (rápido)",
         ["-DBIO_USE_ZLIB", "-DBIO_USE_LIBDEFLATE"] + zlib_link + deflate_link),
        ("CON .gz (zlib)",      ["-DBIO_USE_ZLIB"] + zlib_link),
        ("SIN soporte .gz",     []),
    ]

    mode = "portátil/genérica" if _flag("BIOFORGE_PORTABLE") else "nativa (-march=native)"
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
