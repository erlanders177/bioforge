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

# ── Verificar si el motor tiene las funciones del parser (requiere recompilación) ─
C_PARSER_AVAILABLE: bool = False

def _check_parser() -> None:
    global C_PARSER_AVAILABLE
    if not C_AVAILABLE or _lib is None:
        return
    try:
        # in_dll fuerza la resolución del símbolo en el DLL ahora mismo.
        # Si el DLL es antiguo (sin bio_parser_open), lanza OSError aquí
        # en vez de colapsar más tarde al llamar la función.
        ctypes.c_void_p.in_dll(_lib, "bio_parser_open")
        _lib.bio_parser_open.restype  = ctypes.c_void_p
        _lib.bio_parser_open.argtypes = [ctypes.c_char_p]

        _lib.bio_parser_next.restype  = _I32
        _lib.bio_parser_next.argtypes = [
            ctypes.c_void_p,         # handle
            ctypes.c_char_p, _I32,   # hdr, hdr_max
            _U8P, _I32, _I32P,       # codes, codes_max, n_out
            _I32,                     # force_type (-1 auto | 0 nuc | 1 prot)
            _I32P,                    # type_out
            _U8P, _I32P,             # qual, qual_out  (NULL para FASTA)
        ]

        _lib.bio_parser_close.restype  = None
        _lib.bio_parser_close.argtypes = [ctypes.c_void_p]

        C_PARSER_AVAILABLE = True
    except (AttributeError, OSError):
        pass


C_BATCH_AVAILABLE: bool = False

def _check_batch() -> None:
    """El parser por lotes es opcional: DLLs antiguos solo tienen next()."""
    global C_BATCH_AVAILABLE
    if not C_PARSER_AVAILABLE or _lib is None:
        return
    try:
        ctypes.c_void_p.in_dll(_lib, "bio_parser_next_batch")
        _lib.bio_parser_next_batch.restype  = _I32
        _lib.bio_parser_next_batch.argtypes = [
            ctypes.c_void_p, _I32, _I32,     # handle, max_records, force_type
            ctypes.c_char_p, _I32, _I32P,    # hdr_buf, hdr_buf_max, hdr_off
            _U8P, _I32, _I32P,               # pack_buf, pack_buf_max, pack_off
            _I32P, _I32P,                    # n_syms, types
            _U8P, _I32, _I32P,               # qual_buf, qual_buf_max, qual_off
        ]
        C_BATCH_AVAILABLE = True
    except (AttributeError, OSError):
        pass


_check_parser()
_check_batch()


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


def c_parser_open(path: str) -> int:
    """Abre un archivo FASTA/FASTQ y devuelve un handle opaco (c_void_p)."""
    raw = path.encode("utf-8") if isinstance(path, str) else path
    return _lib.bio_parser_open(raw)   # devuelve c_void_p (int en Python)


def c_parser_next(
    handle: int,
    hdr_buf: "ctypes.Array",
    codes_buf: np.ndarray,
    force_type: int,
) -> "tuple[int, int, int]":
    """Lee el siguiente registro FASTA.
    Retorna (ret, n_symbols, seq_type): ret 1=OK 0=EOF -1=error -2=overflow."""
    n_out    = _I32(0)
    type_out = _I32(0)
    ret = _lib.bio_parser_next(
        ctypes.c_void_p(handle),
        hdr_buf, _I32(len(hdr_buf)),
        codes_buf.ctypes.data_as(_U8P), _I32(len(codes_buf)),
        ctypes.byref(n_out),
        _I32(force_type),
        ctypes.byref(type_out),
        None, None,   # sin calidades — FASTA
    )
    return ret, n_out.value, type_out.value


def c_parser_next_fastq(
    handle: int,
    hdr_buf: "ctypes.Array",
    codes_buf: np.ndarray,
    qual_buf: np.ndarray,
) -> "tuple[int, int, int]":
    """Lee el siguiente registro FASTQ.
    Retorna (ret, n_symbols, n_qual): ret 1=OK 0=EOF -1=error -2=overflow."""
    n_out    = _I32(0)
    type_out = _I32(0)
    q_out    = _I32(0)
    ret = _lib.bio_parser_next(
        ctypes.c_void_p(handle),
        hdr_buf, _I32(len(hdr_buf)),
        codes_buf.ctypes.data_as(_U8P), _I32(len(codes_buf)),
        ctypes.byref(n_out),
        _I32(0),   # FASTQ siempre nucleótido
        ctypes.byref(type_out),
        qual_buf.ctypes.data_as(_U8P), ctypes.byref(q_out),
    )
    return ret, n_out.value, q_out.value


def c_parser_next_batch(
    handle: int,
    max_records: int,
    force_type: int,
    hdr_buf:  "ctypes.Array",
    hdr_off:  np.ndarray,
    pack_buf: np.ndarray,
    pack_off: np.ndarray,
    n_syms:   np.ndarray,
    types:    np.ndarray,
    qual_buf: "np.ndarray | None" = None,
    qual_off: "np.ndarray | None" = None,
) -> int:
    """Parsea hasta ``max_records`` registros en una sola llamada.

    Empaqueta cada secuencia a 5-bit dentro de C. Rellena los arrays de salida
    (que el llamante reutiliza entre lotes). Retorna el nº de registros
    parseados (>=0), 0 = EOF, o negativo en error (-2 = registro demasiado
    grande para el buffer).
    """
    q_ptr     = qual_buf.ctypes.data_as(_U8P) if qual_buf is not None else None
    q_max     = _I32(len(qual_buf)) if qual_buf is not None else _I32(0)
    q_off_ptr = qual_off.ctypes.data_as(_I32P) if qual_off is not None else None
    return _lib.bio_parser_next_batch(
        ctypes.c_void_p(handle), _I32(max_records), _I32(force_type),
        hdr_buf, _I32(len(hdr_buf)), hdr_off.ctypes.data_as(_I32P),
        pack_buf.ctypes.data_as(_U8P), _I32(len(pack_buf)),
        pack_off.ctypes.data_as(_I32P),
        n_syms.ctypes.data_as(_I32P), types.ctypes.data_as(_I32P),
        q_ptr, q_max, q_off_ptr,
    )


def c_parser_close(handle: int) -> None:
    """Libera el handle del parser y cierra el archivo."""
    _lib.bio_parser_close(ctypes.c_void_p(handle))


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
