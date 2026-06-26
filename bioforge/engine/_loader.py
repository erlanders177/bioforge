"""
engine/_loader.py — Carga el motor C y expone funciones Python.

Si el DLL no está compilado, C_AVAILABLE = False y los módulos
usan el fallback NumPy automáticamente.
"""

import ctypes
import sys
from pathlib import Path

import numpy as np

_ENGINE_DIR = Path(__file__).parent
_DLL_PATH   = _ENGINE_DIR / ("engine.dll" if sys.platform == "win32" else "engine.so")

# ── Tipos ctypes frecuentes ────────────────────────────────────────────────────
_U8P  = ctypes.POINTER(ctypes.c_uint8)
_I32P = ctypes.POINTER(ctypes.c_int32)
_I32  = ctypes.c_int32
_CHARP = ctypes.c_char_p

# ── Carga del DLL ──────────────────────────────────────────────────────────────
_lib: ctypes.CDLL | None = None
C_AVAILABLE: bool = False

def _load() -> bool:
    global _lib, C_AVAILABLE
    if not _DLL_PATH.exists():
        return False
    try:
        _lib = ctypes.CDLL(str(_DLL_PATH))
        _setup_signatures()
        C_AVAILABLE = True
        return True
    except Exception:
        return False


def _setup_signatures() -> None:
    assert _lib is not None

    # ── bio_getitem5 ───────────────────────────────────────────────────────────
    _lib.bio_getitem5.restype  = ctypes.c_uint8
    _lib.bio_getitem5.argtypes = [_U8P, _I32]

    # ── bio_pack5 ─────────────────────────────────────────────────────────────
    _lib.bio_pack5.restype  = None
    _lib.bio_pack5.argtypes = [_U8P, _I32, _U8P]

    # ── bio_unpack5 ───────────────────────────────────────────────────────────
    _lib.bio_unpack5.restype  = None
    _lib.bio_unpack5.argtypes = [_U8P, _I32, _U8P]

    # ── bio_find_atg ──────────────────────────────────────────────────────────
    _lib.bio_find_atg.restype  = _I32
    _lib.bio_find_atg.argtypes = [_U8P, _I32]

    # ── bio_translate ─────────────────────────────────────────────────────────
    _lib.bio_translate.restype  = None
    _lib.bio_translate.argtypes = [_U8P, _U8P, _I32, _U8P]

    # ── nw_global / nw_semiglobal (misma firma) ────────────────────────────────
    _nw_args = [
        _U8P, _I32,     # codes_a, m
        _U8P, _I32,     # codes_b, n
        _CHARP,         # decode (32 bytes)
        _I32, _I32, _I32,  # match, mismatch, gap
        _CHARP, _CHARP, # out_a, out_b
        _I32P, _I32P, _I32P, _I32P,  # score, matches, mismatches, gaps
    ]
    _lib.nw_global.restype     = _I32
    _lib.nw_global.argtypes    = _nw_args
    _lib.nw_semiglobal.restype  = _I32
    _lib.nw_semiglobal.argtypes = _nw_args

    # ── sw_align (Smith-Waterman, misma firma que nw) ──────────────────────────
    _lib.sw_align.restype  = _I32
    _lib.sw_align.argtypes = _nw_args

    # ── nw_banded / nw_banded_semiglobal (band extra) ─────────────────────────
    _nw_banded_args = [
        _U8P, _I32,     # codes_a, m
        _U8P, _I32,     # codes_b, n
        _CHARP,         # decode
        _I32, _I32, _I32, _I32,  # match, mismatch, gap, band
        _CHARP, _CHARP,           # out_a, out_b
        _I32P, _I32P, _I32P, _I32P,
    ]
    _lib.nw_banded.restype           = _I32
    _lib.nw_banded.argtypes          = _nw_banded_args
    _lib.nw_banded_semiglobal.restype  = _I32
    _lib.nw_banded_semiglobal.argtypes = _nw_banded_args


_load()


# ── Wrappers Python ────────────────────────────────────────────────────────────

def c_getitem5(packed: np.ndarray, i: int) -> int:
    return int(_lib.bio_getitem5(
        packed.ctypes.data_as(_U8P),
        _I32(i),
    ))


def c_pack5(codes: np.ndarray) -> np.ndarray:
    n       = len(codes)
    out_len = (n * 5 + 7) // 8 + 1   # +1 para lecturas seguras
    out     = np.zeros(out_len, dtype=np.uint8)
    _lib.bio_pack5(
        codes.ctypes.data_as(_U8P),
        _I32(n),
        out.ctypes.data_as(_U8P),
    )
    return out[:out_len - 1]   # recortar el byte extra


def c_unpack5(packed: np.ndarray, n: int) -> np.ndarray:
    # Asegurar byte extra para lecturas seguras
    safe = np.zeros(len(packed) + 1, dtype=np.uint8)
    safe[:len(packed)] = packed
    out = np.empty(n, dtype=np.uint8)
    _lib.bio_unpack5(
        safe.ctypes.data_as(_U8P),
        _I32(n),
        out.ctypes.data_as(_U8P),
    )
    return out


def c_find_atg(codes: np.ndarray) -> int:
    """Devuelve el indice del primer ATG en codes, o -1 si no existe."""
    safe = np.ascontiguousarray(codes, dtype=np.uint8)
    return int(_lib.bio_find_atg(safe.ctypes.data_as(_U8P), _I32(len(safe))))


def c_translate(codon_lut: np.ndarray, nuc_codes: np.ndarray, n_codons: int) -> np.ndarray:
    """Traduce n_codons codones usando el LUT; devuelve array uint8 de AAs."""
    lut  = np.ascontiguousarray(codon_lut, dtype=np.uint8)
    safe = np.ascontiguousarray(nuc_codes[:n_codons * 3], dtype=np.uint8)
    out  = np.empty(n_codons, dtype=np.uint8)
    _lib.bio_translate(
        lut.ctypes.data_as(_U8P),
        safe.ctypes.data_as(_U8P),
        _I32(n_codons),
        out.ctypes.data_as(_U8P),
    )
    return out


def c_sw_align(
    codes_a: np.ndarray,
    codes_b: np.ndarray,
    decode_bytes: bytes,
    match: int, mismatch: int, gap: int,
) -> tuple[str, str, int, int, int, int]:
    """Smith-Waterman local alignment en C."""
    m, n     = len(codes_a), len(codes_b)
    buf_size = m + n + 2
    out_a = ctypes.create_string_buffer(buf_size)
    out_b = ctypes.create_string_buffer(buf_size)
    score = _I32(0); nm = _I32(0); nmi = _I32(0); ng = _I32(0)
    ca = np.ascontiguousarray(codes_a, dtype=np.uint8)
    cb = np.ascontiguousarray(codes_b, dtype=np.uint8)
    aln_len = _lib.sw_align(
        ca.ctypes.data_as(_U8P), _I32(m),
        cb.ctypes.data_as(_U8P), _I32(n),
        decode_bytes,
        _I32(match), _I32(mismatch), _I32(gap),
        out_a, out_b,
        ctypes.byref(score), ctypes.byref(nm),
        ctypes.byref(nmi), ctypes.byref(ng),
    )
    if aln_len < 0:
        raise MemoryError("Motor C: fallo de memoria en SW")
    return (
        out_a.value.decode("ascii"), out_b.value.decode("ascii"),
        score.value, nm.value, nmi.value, ng.value,
    )


def c_nw_banded(
    codes_a: np.ndarray,
    codes_b: np.ndarray,
    decode_bytes: bytes,
    match: int, mismatch: int, gap: int,
    band: int, mode: str,
) -> tuple[str, str, int, int, int, int]:
    """Banded NW en C. Memoria O(m*band)."""
    m, n     = len(codes_a), len(codes_b)
    buf_size = m + n + 2
    out_a = ctypes.create_string_buffer(buf_size)
    out_b = ctypes.create_string_buffer(buf_size)
    score = _I32(0); nm = _I32(0); nmi = _I32(0); ng = _I32(0)
    ca = np.ascontiguousarray(codes_a, dtype=np.uint8)
    cb = np.ascontiguousarray(codes_b, dtype=np.uint8)
    fn = _lib.nw_banded_semiglobal if mode == "semi-global" else _lib.nw_banded
    aln_len = fn(
        ca.ctypes.data_as(_U8P), _I32(m),
        cb.ctypes.data_as(_U8P), _I32(n),
        decode_bytes,
        _I32(match), _I32(mismatch), _I32(gap), _I32(band),
        out_a, out_b,
        ctypes.byref(score), ctypes.byref(nm),
        ctypes.byref(nmi), ctypes.byref(ng),
    )
    if aln_len < 0:
        raise MemoryError("Motor C: fallo de memoria en NW banded")
    return (
        out_a.value.decode("ascii"), out_b.value.decode("ascii"),
        score.value, nm.value, nmi.value, ng.value,
    )


def c_nw_align(
    codes_a: np.ndarray,
    codes_b: np.ndarray,
    decode_bytes: bytes,
    match: int, mismatch: int, gap: int,
    mode: str,
) -> tuple[str, str, int, int, int, int]:
    """
    Alineamiento NW completo en C.
    Devuelve (aligned_a, aligned_b, score, n_matches, n_mismatches, n_gaps).
    """
    m, n     = len(codes_a), len(codes_b)
    buf_size = m + n + 2

    out_a = ctypes.create_string_buffer(buf_size)
    out_b = ctypes.create_string_buffer(buf_size)
    score = _I32(0)
    nm    = _I32(0)
    nmi   = _I32(0)
    ng    = _I32(0)

    # Asegurar que los arrays son C-contiguos uint8
    ca = np.ascontiguousarray(codes_a, dtype=np.uint8)
    cb = np.ascontiguousarray(codes_b, dtype=np.uint8)

    fn = _lib.nw_semiglobal if mode == "semi-global" else _lib.nw_global

    aln_len = fn(
        ca.ctypes.data_as(_U8P), _I32(m),
        cb.ctypes.data_as(_U8P), _I32(n),
        decode_bytes,
        _I32(match), _I32(mismatch), _I32(gap),
        out_a, out_b,
        ctypes.byref(score),
        ctypes.byref(nm),
        ctypes.byref(nmi),
        ctypes.byref(ng),
    )

    if aln_len < 0:
        raise MemoryError("Motor C: fallo de asignación de memoria en NW")

    return (
        out_a.value.decode("ascii"),
        out_b.value.decode("ascii"),
        score.value,
        nm.value,
        nmi.value,
        ng.value,
    )
