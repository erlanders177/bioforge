"""
tests/test_bgzf.py
Pruebas del soporte BGZF (palanca 3): conversor + lector paralelo.

BGZF es un .gz válido por bloques independientes, descomprimible en paralelo.
Se verifica el round-trip, la compatibilidad con gunzip estándar, la detección,
y que BioForge lo lee igual que la ruta secuencial.
"""

import gzip
import random

import numpy as np
import pytest

from bioforge import SmartImporter, bgzf
from bioforge.engine._loader import (
    C_LIBDEFLATE_AVAILABLE, C_PARALLEL_AVAILABLE, c_is_bgzf,
)

_bg = pytest.mark.skipif(
    not (C_LIBDEFLATE_AVAILABLE and C_PARALLEL_AVAILABLE),
    reason="libdeflate/paralelo no disponible")

QCHARS = "".join(chr(33 + q) for q in range(40))
BASES = "ACGT"


def _fastq_bytes(n, rng):
    parts = []
    recs = []
    for i in range(n):
        L = rng.randint(40, 200)
        s = "".join(rng.choice(BASES) for _ in range(L))
        q = "".join(rng.choice(QCHARS) for _ in range(L))
        recs.append((f"r{i}", s))
        parts.append(f"@r{i}\n{s}\n+\n{q}\n")
    return "".join(parts).encode(), recs


@_bg
def test_bgzf_roundtrip_and_gunzip_compatible():
    rng = random.Random(1)
    data, _ = _fastq_bytes(2000, rng)
    comp = bgzf.compress_bytes(data, level=6, n_threads=0)
    # es BGZF
    assert c_is_bgzf(np.frombuffer(comp, dtype=np.uint8))
    # gunzip estándar lo lee idéntico al original
    assert gzip.decompress(comp) == data


@_bg
def test_bgzf_read_equals_sequential(tmp_path):
    rng = random.Random(2)
    data, recs = _fastq_bytes(4000, rng)
    p = tmp_path / "c.fastq.gz"
    p.write_bytes(bgzf.compress_bytes(data, level=6, n_threads=0))

    def collect(nt):
        out = []
        for b in SmartImporter.stream_fastq_batches(str(p), n_threads=nt):
            for i in range(len(b)):
                out.append(b[i].sequence.to_string())
        return out

    seq = collect(1)            # ruta secuencial (zlib)
    par = collect(4)            # ruta BGZF paralela
    assert par == seq == [s for _, s in recs]


@_bg
def test_bgzf_compress_file(tmp_path):
    rng = random.Random(3)
    data, recs = _fastq_bytes(1000, rng)
    src = tmp_path / "in.fastq"
    src.write_bytes(data)
    out = bgzf.compress_file(str(src))
    assert out.endswith(".gz")
    assert c_is_bgzf(np.frombuffer(open(out, "rb").read(), dtype=np.uint8))
    # lectura coherente
    n = sum(len(b) for b in SmartImporter.stream_fastq_batches(out, n_threads=2))
    assert n == len(recs)


@_bg
def test_plain_gzip_not_detected_as_bgzf():
    data = b"@r1\nACGT\n+\nIIII\n" * 100
    plain = gzip.compress(data)
    assert not c_is_bgzf(np.frombuffer(plain, dtype=np.uint8))


@_bg
def test_bgzf_fasta(tmp_path):
    rng = random.Random(4)
    recs = [(f"g{i}", "".join(rng.choice(BASES) for _ in range(rng.randint(20, 150))))
            for i in range(2000)]
    content = "".join(f">{h}\n{s}\n" for h, s in recs).encode()
    p = tmp_path / "x.fasta.gz"
    p.write_bytes(bgzf.compress_bytes(content, n_threads=0))
    got = []
    for b in SmartImporter.stream_batches(str(p), n_threads=4):
        for i in range(len(b)):
            got.append((b[i].header, b[i].to_string()))
    assert got == recs
