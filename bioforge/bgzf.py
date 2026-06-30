"""
bgzf.py
══════════════════════════════════════════════════════════════════════
Conversor a BGZF — gzip por bloques independientes, descomprimible en
PARALELO.

Un archivo BGZF es un `.gz` 100 % válido (cualquier herramienta lo lee con
`gunzip`/zlib), pero sus bloques de 64 KB se pueden descomprimir en paralelo.
Convertir una vez un FASTQ que vas a procesar muchas veces hace que BioForge
lo lea con la vía más rápida (palanca 3).

Uso
───
  python -m bioforge.bgzf reads.fastq                 # -> reads.fastq.gz (BGZF)
  python -m bioforge.bgzf reads.fastq -o out.gz -l 9 -t 0
  bioforge-bgzip reads.fastq                           # si el paquete está instalado

Requiere el motor C compilado con libdeflate (build.py lo enlaza si está).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from .engine._loader import C_LIBDEFLATE_AVAILABLE as _C_LIBDEFLATE_AVAILABLE
    from .engine._loader import c_bgzf_compress as _c_bgzf_compress
    from .engine._loader import c_is_bgzf as _c_is_bgzf
except ImportError:                                    # pragma: no cover
    _C_LIBDEFLATE_AVAILABLE = False


def compress_bytes(data: bytes, level: int = 6, n_threads: int = 0) -> bytes:
    """Comprime ``data`` a BGZF y devuelve los bytes comprimidos.

    ``n_threads``: 0 = todos los núcleos. ``level``: 1–12 (libdeflate).
    """
    if not _C_LIBDEFLATE_AVAILABLE:
        raise RuntimeError(
            "El motor C no tiene libdeflate; recompila con "
            "`python bioforge/engine/build.py` (necesita la librería libdeflate).")
    nt = (os.cpu_count() or 1) if n_threads <= 0 else n_threads
    inbuf = np.frombuffer(data, dtype=np.uint8)
    # Holgura: overhead de framing por bloque + posible expansión de deflate.
    nblocks = (len(data) + 0xFF00 - 1) // 0xFF00
    cap = len(data) + len(data) // 8 + nblocks * 64 + 1024
    out = np.empty(cap, dtype=np.uint8)
    n = _c_bgzf_compress(inbuf, out, level, nt)
    if n < 0:                                          # holgura insuficiente: reintentar
        out = np.empty(cap * 2 + (1 << 20), dtype=np.uint8)
        n = _c_bgzf_compress(inbuf, out, level, nt)
        if n < 0:
            raise RuntimeError("Fallo al comprimir a BGZF.")
    return out[:n].tobytes()


def compress_file(in_path: str, out_path: Optional[str] = None,
                  level: int = 6, n_threads: int = 0) -> str:
    """Convierte ``in_path`` a un archivo BGZF. Devuelve la ruta de salida."""
    if out_path is None:
        out_path = in_path + ".gz"
    if os.path.abspath(out_path) == os.path.abspath(in_path):
        raise ValueError(
            f"La salida coincide con la entrada ({in_path!r}); indica otra ruta "
            "con -o para no sobrescribir el archivo original.")
    with open(in_path, "rb") as f:
        data = f.read()
    comp = compress_bytes(data, level=level, n_threads=n_threads)
    with open(out_path, "wb") as f:
        f.write(comp)
    return out_path


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="bioforge-bgzip",
        description="Convierte un archivo a BGZF (gzip por bloques, paralelizable).")
    p.add_argument("input", help="Archivo a comprimir (p.ej. reads.fastq)")
    p.add_argument("--output", "-o", help="Ruta de salida (por defecto: input + .gz)")
    p.add_argument("--level", "-l", type=int, default=6,
                   help="Nivel de compresión 1–12 (libdeflate, por defecto 6)")
    p.add_argument("--threads", "-t", type=int, default=0,
                   help="Hilos (0 = todos los núcleos)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    args = _parse_args(argv)
    try:
        import time
        t0 = time.perf_counter()
        out = compress_file(args.input, args.output, args.level, args.threads)
        dt = time.perf_counter() - t0
        ins = os.path.getsize(args.input)
        outs = os.path.getsize(out)
        print(f"BGZF: {args.input} -> {out}")
        print(f"  {ins/1e6:.1f} MB -> {outs/1e6:.1f} MB "
              f"({outs/ins*100:.0f}%) en {dt*1000:.0f} ms")
        print("  Léelo en paralelo con n_threads en SmartImporter.stream_fastq_batches.")
    except FileNotFoundError:
        print(f"Archivo no encontrado: {args.input}", file=sys.stderr)
        return 1
    except (RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
