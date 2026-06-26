"""
engine/build.py — Compila el motor C a DLL/SO.

Uso:
    python engine/build.py
"""

import subprocess
import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).parent
SRC = ENGINE_DIR / "engine.c"


def build() -> bool:
    if sys.platform == "win32":
        out = ENGINE_DIR / "engine.dll"
        cmd = [
            "gcc", "-O3", "-march=native", "-fopenmp",
            "-shared",
            "-o", str(out), str(SRC),
        ]
    else:
        out = ENGINE_DIR / "engine.so"
        cmd = [
            "gcc", "-O3", "-march=native", "-fopenmp",
            "-shared", "-fPIC",
            "-o", str(out), str(SRC),
        ]

    print(f"Compilando {SRC.name} -> {out.name} ...")
    res = subprocess.run(cmd, capture_output=True, text=True)

    if res.returncode == 0:
        size_kb = out.stat().st_size // 1024
        print(f"[OK] Motor compilado: {out}  ({size_kb} KB)")
        return True
    else:
        print("[ERROR] Error de compilacion:")
        print(res.stderr)
        return False


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
