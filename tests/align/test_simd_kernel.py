"""
tests/test_simd_kernel.py — paridad del kernel banded SIMD (v6.0).

La extensión del mapeador usa `nw_banded_diag_simd` (AVX2, 8× int32). DEBE dar
resultados BIT-IDÉNTICOS al banded NW escalar del core (`nw_banded`) y al
antidiagonal escalar (`nw_banded_diag`). Este fuzz determinista lo garantiza,
incluyendo códigos N (=4), tamaños < 8 (donde el SIMD no entra y todo es escalar)
y bandas extremas. Es la red que protege la corrección cuando el SIMD evolucione
(AVX-512, int16…). Si no hay motor C, se salta.
"""

import ctypes

import numpy as np
import pytest

from bioforge.engine import _loader as L

pytestmark = pytest.mark.skipif(not L.C_AVAILABLE, reason="motor C no disponible")

_U8P = ctypes.POINTER(ctypes.c_uint8); _I32 = ctypes.c_int32
_I32P = ctypes.POINTER(ctypes.c_int32); _CH = ctypes.c_char_p
_DEC = b'ACGTN' + b'N' * 27


def _get(name):
    fn = getattr(L._lib, name, None)
    if fn is None:
        return None
    fn.restype = _I32
    fn.argtypes = [_U8P, _I32, _U8P, _I32, _CH, _I32, _I32, _I32, _I32,
                   _CH, _CH, _I32P, _I32P, _I32P, _I32P]
    return fn


def _run(fn, a, b, band):
    m, n = len(a), len(b)
    ca = np.ascontiguousarray(a, dtype=np.uint8)
    cb = np.ascontiguousarray(b, dtype=np.uint8)
    oa = ctypes.create_string_buffer(m + n + 2)
    ob = ctypes.create_string_buffer(m + n + 2)
    sc = _I32(0); nm = _I32(0); nmi = _I32(0); ng = _I32(0)
    r = fn(ca.ctypes.data_as(_U8P), m, cb.ctypes.data_as(_U8P), n, _DEC,
           2, -1, -2, band, oa, ob,
           ctypes.byref(sc), ctypes.byref(nm), ctypes.byref(nmi), ctypes.byref(ng))
    return r, oa.value, ob.value, sc.value, nm.value, nmi.value, ng.value


def test_simd_igual_que_core_y_escalar():
    core = _get("nw_banded")
    diag = _get("nw_banded_diag")
    simd = _get("nw_banded_diag_simd")
    if not (core and diag and simd):
        pytest.skip("motor C sin los kernels banded del v6.0")

    rng = np.random.default_rng(2027)
    for _ in range(4000):
        m = int(rng.integers(1, 300))
        n = int(rng.integers(1, 300))
        pN = float(rng.choice([0.0, 0.0, 0.05, 0.2]))     # densidad de N
        a = np.where(rng.random(m) < pN, 4, rng.integers(0, 4, m)).astype(np.uint8)
        b = np.where(rng.random(n) < pN, 4, rng.integers(0, 4, n)).astype(np.uint8)
        band = int(rng.choice([1, 2, 3, 4, 7, 8, 9, 16, 31, 64, 200, 400]))
        rc = _run(core, a, b, band)
        rd = _run(diag, a, b, band)
        rs = _run(simd, a, b, band)
        assert rc == rd == rs, f"m={m} n={n} band={band} pN={pN}"


def test_simd_caso_extension_mapper():
    """Réplica del caso del mapeador: band = min(512, |diff|+64) — nunca falla."""
    core = _get("nw_banded")
    simd = _get("nw_banded_diag_simd")
    if not (core and simd):
        pytest.skip("motor C sin los kernels banded del v6.0")
    rng = np.random.default_rng(5)
    for _ in range(1000):
        m = int(rng.integers(100, 3000))
        a = rng.integers(0, 4, m).astype(np.uint8)
        # b = a con mutaciones + indels
        b = a.copy()
        for p in rng.choice(m, max(1, m // 25), replace=False):
            b[p] = (b[p] + 1) % 4
        if rng.integers(0, 2):
            cut = int(rng.integers(0, m))
            b = np.delete(b, cut)
        band = min(512, abs(m - len(b)) + 64)
        assert _run(core, a, b, band) == _run(simd, a, b, band)
