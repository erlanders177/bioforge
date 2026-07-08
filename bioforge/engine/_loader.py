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
_I64P = ctypes.POINTER(ctypes.c_int64)
_U64P = ctypes.POINTER(ctypes.c_uint64)
_F64P = ctypes.POINTER(ctypes.c_double)
_I32  = ctypes.c_int32
_I64  = ctypes.c_int64
_F64  = ctypes.c_double
_CHARP = ctypes.c_char_p


class MapOut(ctypes.Structure):
    """Un mapeo devuelto por bio_map_read (layout replicado del struct C).

    Orden de campos idéntico al de engine.c (doubles/int64 primero para evitar
    padding sorpresa). Las coords de target son GLOBALES; la cubierta Python las
    localiza al contig y adjunta el nombre.
    """
    _fields_ = [
        ("identity",     ctypes.c_double),
        ("chain_score",  ctypes.c_double),
        ("target_start", ctypes.c_int64),
        ("target_end",   ctypes.c_int64),
        ("query_start",  ctypes.c_int32),
        ("query_end",    ctypes.c_int32),
        ("strand",       ctypes.c_int32),
        ("num_matches",  ctypes.c_int32),
        ("block_len",    ctypes.c_int32),
        ("mapq",         ctypes.c_int32),
        ("contig",       ctypes.c_int32),
        ("cigar_off",    ctypes.c_int32),
        ("cigar_len",    ctypes.c_int32),
    ]


_MAPOUTP = ctypes.POINTER(MapOut)

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


C_PARALLEL_AVAILABLE: bool = False

def _check_parallel() -> None:
    """Parser paralelo en memoria (OpenMP). Opcional."""
    global C_PARALLEL_AVAILABLE
    if not C_BATCH_AVAILABLE or _lib is None:
        return
    try:
        ctypes.c_void_p.in_dll(_lib, "bio_parse_mem_parallel")
        _lib.bio_parse_mem_parallel.restype  = _I32
        _lib.bio_parse_mem_parallel.argtypes = [
            _U8P, _I64, _I32, _I32, _I32,    # data, len, fmt, n_threads, force_type
            ctypes.c_char_p, _I32, _I32P,    # hdr_buf, hdr_buf_max, hdr_off
            _U8P, _I64, _I32P,               # pack_buf, pack_buf_max, pack_off
            _I32P, _I32P,                    # n_syms, types
            _U8P, _I64, _I32P,               # qual_buf, qual_buf_max, qual_off
            _I32,                            # max_records
        ]
        C_PARALLEL_AVAILABLE = True
    except (AttributeError, OSError):
        pass


C_LIBDEFLATE_AVAILABLE: bool = False

def _check_libdeflate() -> None:
    """Descompresor gzip rápido (libdeflate). Opcional."""
    global C_LIBDEFLATE_AVAILABLE
    if not C_AVAILABLE or _lib is None:
        return
    try:
        ctypes.c_void_p.in_dll(_lib, "bio_has_libdeflate")
        _lib.bio_has_libdeflate.restype = ctypes.c_int
        _lib.bio_has_libdeflate.argtypes = []
        _lib.bio_gzip_decompress.restype = _I64
        _lib.bio_gzip_decompress.argtypes = [_U8P, _I64, _U8P, _I64]
        _lib.bio_is_bgzf.restype = ctypes.c_int
        _lib.bio_is_bgzf.argtypes = [_U8P, _I64]
        _lib.bio_bgzf_usize.restype = _I64
        _lib.bio_bgzf_usize.argtypes = [_U8P, _I64]
        _lib.bio_bgzf_decompress_parallel.restype = _I64
        _lib.bio_bgzf_decompress_parallel.argtypes = [_U8P, _I64, _U8P, _I64, _I32]
        _lib.bio_bgzf_compress.restype = _I64
        _lib.bio_bgzf_compress.argtypes = [_U8P, _I64, _U8P, _I64, _I32, _I32]
        C_LIBDEFLATE_AVAILABLE = bool(_lib.bio_has_libdeflate())
    except (AttributeError, OSError):
        pass


C_CHAIN_AVAILABLE: bool = False

def _check_chain() -> None:
    """DP de chaining del alineador de genomas (v3). Opcional."""
    global C_CHAIN_AVAILABLE
    if not C_AVAILABLE or _lib is None:
        return
    try:
        _lib.bio_chain_dp.restype = None
        _lib.bio_chain_dp.argtypes = [
            _I64P, _I64P, _I32,        # x, y, n
            _I32, _I64, _I32, _F64,    # k, max_gap, window, gap_w
            _F64P, _I32P,              # f (out), prev (out)
        ]
        C_CHAIN_AVAILABLE = True
    except (AttributeError, OSError):
        pass


C_MINIMIZERS_AVAILABLE: bool = False

def _check_minimizers() -> None:
    """Minimizers (w,k) del alineador de genomas (v3). Opcional."""
    global C_MINIMIZERS_AVAILABLE
    if not C_AVAILABLE or _lib is None:
        return
    try:
        _lib.bio_minimizers.restype = _I64
        _lib.bio_minimizers.argtypes = [
            _U8P, _I64, _I32, _I32,        # codes, n, k, w
            _U64P, _I64P, _U8P,            # out_hash, out_pos, out_strand
        ]
        C_MINIMIZERS_AVAILABLE = True
    except (AttributeError, OSError):
        pass


C_INDEX_AVAILABLE: bool = False

def _check_index() -> None:
    """Índice opaco de la referencia (mapeador v5, pipeline en C). Opcional."""
    global C_INDEX_AVAILABLE
    if not C_AVAILABLE or _lib is None:
        return
    try:
        ctypes.c_void_p.in_dll(_lib, "bio_index_build")
        _lib.bio_index_build.restype = ctypes.c_void_p
        _lib.bio_index_build.argtypes = [
            _U8P, _I64, _I32, _I32, _I32,    # ref_codes, n, k, w, max_occ
            _I64P, _I64P, _I32,              # ctg_starts, ctg_lengths, n_contigs
        ]
        _lib.bio_index_free.restype  = None
        _lib.bio_index_free.argtypes = [ctypes.c_void_p]
        _lib.bio_index_n_minimizers.restype  = _I64
        _lib.bio_index_n_minimizers.argtypes = [ctypes.c_void_p]

        _lib.bio_map_read.restype  = _I32
        _lib.bio_map_read.argtypes = [
            ctypes.c_void_p, _U8P, _I32,     # handle, read_codes, Lr
            _I32, _F64,                      # max_hits, min_score
            _MAPOUTP, _I32,                  # out, max_out
            _CHARP, _I32,                    # cigar_buf, cigar_cap
        ]
        _lib.bio_map_batch.restype  = _I32
        _lib.bio_map_batch.argtypes = [
            ctypes.c_void_p, _U8P, _I64P, _I32,   # handle, reads, read_off, n_reads
            _I32, _F64,                           # max_hits, min_score
            _MAPOUTP, _I32P,                      # out, counts
            _CHARP, _I64P, _I32,                  # cigar_buf, cig_off, n_threads
        ]
        C_INDEX_AVAILABLE = True
    except (AttributeError, OSError):
        pass


_check_parser()
_check_batch()
_check_parallel()
_check_libdeflate()
_check_chain()
_check_minimizers()
_check_index()


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
    # bio_unpack5 es seguro en los límites → no hace falta copiar un byte extra.
    # ascontiguousarray no copia si ya es uint8 contiguo (el caso normal).
    safe = np.ascontiguousarray(packed, dtype=np.uint8)
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


def c_minimizers(codes: np.ndarray, k: int, w: int):
    """Minimizers (w,k) canónicos en C. Devuelve (hashes, positions, strands)."""
    codes = np.ascontiguousarray(codes, dtype=np.uint8)
    n = int(codes.size)
    nk = max(0, n - k + 1)
    out_hash = np.empty(nk, dtype=np.uint64)
    out_pos = np.empty(nk, dtype=np.int64)
    out_strand = np.empty(nk, dtype=np.uint8)
    cnt = _lib.bio_minimizers(
        codes.ctypes.data_as(_U8P), _I64(n), _I32(int(k)), _I32(int(w)),
        out_hash.ctypes.data_as(_U64P), out_pos.ctypes.data_as(_I64P),
        out_strand.ctypes.data_as(_U8P),
    )
    if cnt < 0:
        raise MemoryError("Motor C: fallo de memoria en minimizers")
    return out_hash[:cnt], out_pos[:cnt], out_strand[:cnt]


def c_index_build(ref_codes: np.ndarray, k: int, w: int, max_occ: int,
                  ctg_starts: np.ndarray, ctg_lengths: np.ndarray) -> int:
    """Construye el índice opaco de la referencia en C. Devuelve un handle (int).

    ``ref_codes`` : codes 2-bit uint8 de la referencia concatenada (>=4 = N).
    ``max_occ``   : 0 = sin filtro; >0 = descarta hashes con más de max_occ hits.
    Lanza MemoryError si el índice no cabe en memoria.
    """
    codes = np.ascontiguousarray(ref_codes, dtype=np.uint8)
    cs = np.ascontiguousarray(ctg_starts, dtype=np.int64)
    cl = np.ascontiguousarray(ctg_lengths, dtype=np.int64)
    handle = _lib.bio_index_build(
        codes.ctypes.data_as(_U8P), _I64(int(codes.size)),
        _I32(int(k)), _I32(int(w)), _I32(int(max_occ)),
        cs.ctypes.data_as(_I64P), cl.ctypes.data_as(_I64P), _I32(int(cs.size)),
    )
    if not handle:
        raise MemoryError("Motor C: fallo al construir el índice de la referencia")
    return handle


def c_index_free(handle: int) -> None:
    """Libera el índice opaco."""
    if handle:
        _lib.bio_index_free(ctypes.c_void_p(handle))


def c_index_n_minimizers(handle: int) -> int:
    """nº de minimizers del índice (verificación de paridad)."""
    return int(_lib.bio_index_n_minimizers(ctypes.c_void_p(handle)))


def c_map_read(handle: int, read_codes: np.ndarray, max_hits: int,
               min_score: float) -> list[dict]:
    """Mapea un read entero en C (seed-chain-align). Devuelve lista de dicts.

    Cada dict lleva los campos numéricos del mapeo (coords de target GLOBALES)
    más ``contig`` (índice) y ``cigar`` (str). La cubierta Python los convierte
    a Mapping localizando al contig y adjuntando el nombre.
    """
    codes = np.ascontiguousarray(read_codes, dtype=np.uint8)
    Lr = int(codes.size)
    max_out = int(max_hits)
    out = (MapOut * max_out)()
    cigar_cap = max_out * (2 * Lr + 1200) + 1024
    cbuf = ctypes.create_string_buffer(cigar_cap)
    n = _lib.bio_map_read(
        ctypes.c_void_p(handle), codes.ctypes.data_as(_U8P), _I32(Lr),
        _I32(max_out), _F64(float(min_score)),
        out, _I32(max_out), cbuf, _I32(cigar_cap),
    )
    if n < 0:
        raise MemoryError("Motor C: fallo en bio_map_read")
    raw = cbuf.raw
    results: list[dict] = []
    for i in range(n):
        o = out[i]
        cig = raw[o.cigar_off:o.cigar_off + o.cigar_len].decode("ascii")
        results.append({
            "identity": o.identity, "chain_score": o.chain_score,
            "target_start": o.target_start, "target_end": o.target_end,
            "query_start": o.query_start, "query_end": o.query_end,
            "strand": o.strand, "num_matches": o.num_matches,
            "block_len": o.block_len, "mapq": o.mapq,
            "contig": o.contig, "cigar": cig,
        })
    return results


def _map_out_to_dict(o: "MapOut", cigar: str) -> dict:
    return {
        "identity": o.identity, "chain_score": o.chain_score,
        "target_start": o.target_start, "target_end": o.target_end,
        "query_start": o.query_start, "query_end": o.query_end,
        "strand": o.strand, "num_matches": o.num_matches,
        "block_len": o.block_len, "mapq": o.mapq,
        "contig": o.contig, "cigar": cigar,
    }


def c_map_batch(handle: int, read_codes_list, max_hits: int, min_score: float,
                n_threads: int = 0) -> list[list[dict]]:
    """Mapea muchos reads en paralelo en C (OpenMP). Devuelve lista de listas.

    ``read_codes_list`` : iterable de arrays uint8 (codes 2-bit) por read.
    ``n_threads``       : 0 = todos los núcleos · N = N hilos.
    Cada read escribe hasta ``max_hits`` mapeos; los cigars van a un buffer
    compartido delimitado por read. Sin GIL: el trabajo pesado ocurre en C.
    """
    reads = [np.ascontiguousarray(r, dtype=np.uint8) for r in read_codes_list]
    n = len(reads)
    if n == 0:
        return []
    lens = np.fromiter((r.size for r in reads), dtype=np.int64, count=n)
    read_off = np.empty(n + 1, dtype=np.int64)
    read_off[0] = 0
    np.cumsum(lens, out=read_off[1:])
    concat = np.concatenate(reads) if n else np.empty(0, np.uint8)

    mh = int(max_hits)
    # región de cigar por read: cota generosa (el C trunca con seguridad si no cabe)
    cig_sizes = (mh * (2 * lens + 1200) + 1024).astype(np.int64)
    cig_off = np.empty(n + 1, dtype=np.int64)
    cig_off[0] = 0
    np.cumsum(cig_sizes, out=cig_off[1:])

    out = (MapOut * (n * mh))()
    counts = np.zeros(n, dtype=np.int32)
    cbuf = ctypes.create_string_buffer(int(cig_off[-1]))
    rc = _lib.bio_map_batch(
        ctypes.c_void_p(handle),
        concat.ctypes.data_as(_U8P), read_off.ctypes.data_as(_I64P), _I32(n),
        _I32(mh), _F64(float(min_score)),
        out, counts.ctypes.data_as(_I32P),
        cbuf, cig_off.ctypes.data_as(_I64P), _I32(int(n_threads)),
    )
    if rc < 0:
        raise MemoryError("Motor C: fallo en bio_map_batch")
    raw = cbuf.raw
    results: list[list[dict]] = []
    for i in range(n):
        maps = []
        for j in range(int(counts[i])):
            o = out[i * mh + j]
            cig = raw[o.cigar_off:o.cigar_off + o.cigar_len].decode("ascii")
            maps.append(_map_out_to_dict(o, cig))
        results.append(maps)
    return results


def c_chain_dp(xs: np.ndarray, ys: np.ndarray, k: int,
               max_gap: int, window: int, gap_w: float):
    """DP de chaining en C. Rellena y devuelve (f, prev).

    ``xs``/``ys`` deben ser int64 contiguos y estar ordenados por (x, y).
    """
    n = int(xs.size)
    f = np.empty(n, dtype=np.float64)
    prev = np.empty(n, dtype=np.int32)
    _lib.bio_chain_dp(
        xs.ctypes.data_as(_I64P), ys.ctypes.data_as(_I64P), _I32(n),
        _I32(int(k)), _I64(int(max_gap)), _I32(int(window)), _F64(float(gap_w)),
        f.ctypes.data_as(_F64P), prev.ctypes.data_as(_I32P),
    )
    return f, prev


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


def c_parse_mem_parallel(
    data:     np.ndarray,
    fmt:      int,
    n_threads: int,
    force_type: int,
    hdr_buf:  "ctypes.Array",
    hdr_off:  np.ndarray,
    pack_buf: np.ndarray,
    pack_off: np.ndarray,
    n_syms:   np.ndarray,
    types:    np.ndarray,
    qual_buf: "np.ndarray | None",
    qual_off: "np.ndarray | None",
    max_records: int,
) -> int:
    """Parsea un bloque de memoria (solo registros completos) en paralelo.

    ``data`` es un array uint8 contiguo. Rellena los buffers de salida (que el
    llamante reutiliza). Devuelve nº de registros (>=0), o negativo en error.
    """
    q_ptr = qual_buf.ctypes.data_as(_U8P) if qual_buf is not None else None
    q_max = _I64(len(qual_buf)) if qual_buf is not None else _I64(0)
    q_off = qual_off.ctypes.data_as(_I32P) if qual_off is not None else None
    return _lib.bio_parse_mem_parallel(
        data.ctypes.data_as(_U8P), _I64(len(data)),
        _I32(fmt), _I32(n_threads), _I32(force_type),
        hdr_buf, _I32(len(hdr_buf)), hdr_off.ctypes.data_as(_I32P),
        pack_buf.ctypes.data_as(_U8P), _I64(len(pack_buf)),
        pack_off.ctypes.data_as(_I32P),
        n_syms.ctypes.data_as(_I32P), types.ctypes.data_as(_I32P),
        q_ptr, q_max, q_off,
        _I32(max_records),
    )


def c_gzip_decompress(cbuf: np.ndarray, obuf: np.ndarray) -> int:
    """Descomprime gzip ``cbuf`` (uint8) en ``obuf`` (uint8) con libdeflate.

    Devuelve nº de bytes descomprimidos, o -1 si no caben o hay error.
    """
    return int(_lib.bio_gzip_decompress(
        cbuf.ctypes.data_as(_U8P), _I64(len(cbuf)),
        obuf.ctypes.data_as(_U8P), _I64(len(obuf)),
    ))


def c_is_bgzf(cbuf: np.ndarray) -> bool:
    """True si el buffer comprimido tiene formato BGZF (gzip por bloques)."""
    return bool(_lib.bio_is_bgzf(cbuf.ctypes.data_as(_U8P), _I64(len(cbuf))))


def c_bgzf_usize(cbuf: np.ndarray) -> int:
    """Tamaño total descomprimido de un BGZF, o -1 si malformado."""
    return int(_lib.bio_bgzf_usize(cbuf.ctypes.data_as(_U8P), _I64(len(cbuf))))


def c_bgzf_decompress_parallel(cbuf: np.ndarray, obuf: np.ndarray,
                               n_threads: int) -> int:
    """Descomprime un BGZF en paralelo (bloques independientes).

    Devuelve nº de bytes descomprimidos, o -1 en error.
    """
    return int(_lib.bio_bgzf_decompress_parallel(
        cbuf.ctypes.data_as(_U8P), _I64(len(cbuf)),
        obuf.ctypes.data_as(_U8P), _I64(len(obuf)), _I32(n_threads),
    ))


def c_bgzf_compress(inbuf: np.ndarray, obuf: np.ndarray,
                    level: int, n_threads: int) -> int:
    """Comprime ``inbuf`` a BGZF en paralelo. Devuelve bytes comprimidos o -1."""
    return int(_lib.bio_bgzf_compress(
        inbuf.ctypes.data_as(_U8P), _I64(len(inbuf)),
        obuf.ctypes.data_as(_U8P), _I64(len(obuf)),
        _I32(level), _I32(n_threads),
    ))


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
