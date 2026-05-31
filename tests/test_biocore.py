"""
tests/test_biocore.py
Pruebas de propiedad con Hypothesis + pytest-benchmark para biocore.py.

Ejecutar:
    pytest tests/ -v
    pytest tests/ --benchmark-only   # solo los benchmarks
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as np_arrays

from biocore import (
    BitPacker, BioCode, PackedSequence, SeqType,
    SmartImporter, compute_stats, NUC_LUT, AA_LUT,
)


# ── Estrategias Hypothesis ─────────────────────────────────────────────────────

# Arrays de códigos nucleotídicos válidos (0–3)
nuc_codes = np_arrays(
    dtype=np.uint8,
    shape=st.integers(min_value=1, max_value=2000),
    elements=st.integers(min_value=0, max_value=3),
)

# Arrays de códigos de aminoácidos válidos (4–23) + STOP(24) + GAP(25) + UNK(31)
valid_aa_values = list(range(4, 26)) + [31]
aa_codes = np_arrays(
    dtype=np.uint8,
    shape=st.integers(min_value=1, max_value=2000),
    elements=st.sampled_from(valid_aa_values),
)

# Arrays con cualquier código BioCode válido (0–31, excluyendo 26–30 reservados)
all_valid = list(range(26)) + [31]
any_codes = np_arrays(
    dtype=np.uint8,
    shape=st.integers(min_value=1, max_value=2000),
    elements=st.sampled_from(all_valid),
)


# ══════════════════════════════════════════════════════════════════════════════
# §1  ROUND-TRIP: pack → unpack → idéntico al original
# ══════════════════════════════════════════════════════════════════════════════

@given(any_codes)
def test_pack_unpack_roundtrip(codes):
    """Para cualquier array de BioCode, unpack(pack(x)) == x."""
    packed   = BitPacker.pack(codes)
    restored = BitPacker.unpack(packed, len(codes))
    assert np.array_equal(codes, restored), (
        f"Round-trip falló para array de {len(codes)} códigos"
    )


@given(any_codes)
def test_packed_size_matches_actual(codes):
    """El tamaño empaquetado real coincide con la fórmula ⌈n×5/8⌉."""
    packed = BitPacker.pack(codes)
    assert len(packed) == BitPacker.packed_size(len(codes))


@given(any_codes)
def test_packed_sequence_roundtrip(codes):
    """PackedSequence.decode() devuelve exactamente los mismos códigos originales."""
    seq = PackedSequence(
        header="test",
        seq_type=SeqType.NUCLEOTIDE,
        n_symbols=len(codes),
        data=BitPacker.pack(codes),
    )
    assert np.array_equal(seq.decode(), codes)


# ══════════════════════════════════════════════════════════════════════════════
# §2  PROPIEDADES DE PackedSequence
# ══════════════════════════════════════════════════════════════════════════════

@given(any_codes)
def test_getitem_single_matches_decode(codes):
    """seq[i] debe coincidir con seq.decode()[i] para todo i."""
    seq      = PackedSequence("h", SeqType.NUCLEOTIDE, len(codes), BitPacker.pack(codes))
    decoded  = seq.decode()
    n_check  = min(len(codes), 50)  # limitar a 50 para no ralentizar hypothesis
    for i in range(n_check):
        assert seq[i] == int(decoded[i]), f"Fallo en posición {i}"


@given(any_codes)
def test_data_is_readonly(codes):
    """El array data de PackedSequence debe estar write-locked."""
    seq = PackedSequence("h", SeqType.NUCLEOTIDE, len(codes), BitPacker.pack(codes))
    assert not seq.data.flags.writeable, "data debería ser read-only"


@given(any_codes)
def test_memory_ratio_near_ideal(codes):
    """El ratio de memoria debe ser ≥ 0.625 (ideal 5-bit) y nunca > 1.0."""
    seq = PackedSequence("h", SeqType.NUCLEOTIDE, len(codes), BitPacker.pack(codes))
    assert 0.625 <= seq.memory_ratio <= 1.0


@given(any_codes)
def test_len_equals_n_symbols(codes):
    """len(seq) == seq.n_symbols siempre."""
    seq = PackedSequence("h", SeqType.NUCLEOTIDE, len(codes), BitPacker.pack(codes))
    assert len(seq) == seq.n_symbols == len(codes)


# ══════════════════════════════════════════════════════════════════════════════
# §3  LOOKUP TABLES — sin caracteres perdidos
# ══════════════════════════════════════════════════════════════════════════════

def test_nuc_lut_acgtu_mapped():
    """A, C, G, T, U deben mapearse a valores 0–3, no a UNK."""
    for ch in "ACGTU":
        code = int(NUC_LUT[ord(ch)])
        assert code != BioCode.UNK, f"'{ch}' mapeó a UNK inesperadamente"


def test_nuc_lut_lowercase():
    """Las minúsculas deben mapear igual que las mayúsculas."""
    for ch in "acgtu":
        assert NUC_LUT[ord(ch)] == NUC_LUT[ord(ch.upper())]


def test_aa_lut_all_20_amino_acids():
    """Los 20 aminoácidos estándar deben mapearse, no a UNK."""
    for ch in "ACDEFGHIKLMNPQRSTVWY":
        code = int(AA_LUT[ord(ch)])
        assert code != BioCode.UNK, f"Aminoácido '{ch}' mapeó a UNK"


def test_aa_lut_stop_and_gap():
    """* debe mapearse a STOP y - a GAP en la tabla de aminoácidos."""
    assert AA_LUT[ord('*')] == BioCode.STOP
    assert AA_LUT[ord('-')] == BioCode.GAP


# ══════════════════════════════════════════════════════════════════════════════
# §4  IMPORTADOR FASTA
# ══════════════════════════════════════════════════════════════════════════════

def test_from_string_single_record():
    fasta = ">test\nATGCATGC\n"
    recs  = SmartImporter.from_string(fasta, force_type=SeqType.NUCLEOTIDE)
    assert len(recs) == 1
    assert recs[0].n_symbols == 8
    assert recs[0].to_string() == "ATGCATGC"


def test_from_string_multiple_records():
    fasta = ">a\nACGT\n>b\nTGCA\n>c\nGGGG\n"
    recs  = SmartImporter.from_string(fasta, force_type=SeqType.NUCLEOTIDE)
    assert len(recs) == 3
    assert [r.to_string() for r in recs] == ["ACGT", "TGCA", "GGGG"]


def test_autodetect_protein():
    """Una secuencia con E/F/I/L/P/Q debe detectarse como proteína."""
    fasta = ">prot\nMVHLTPEEKSAVTALWGKV\n"
    recs  = SmartImporter.from_string(fasta)
    assert recs[0].seq_type == SeqType.PROTEIN


def test_autodetect_nucleotide():
    """Una secuencia sin E/F/I/L/P/Q/* debe detectarse como nucleótido."""
    fasta = ">nuc\nATGCATGCATGC\n"
    recs  = SmartImporter.from_string(fasta)
    assert recs[0].seq_type == SeqType.NUCLEOTIDE


def test_fasta_mixed_case_and_spaces():
    """El importador debe limpiar mayúsculas/minúsculas y normalizar."""
    fasta = ">test\natgc\nATGC\n"
    recs  = SmartImporter.from_string(fasta, force_type=SeqType.NUCLEOTIDE)
    assert recs[0].n_symbols == 8
    assert recs[0].to_string() == "ATGCATGC"


# ══════════════════════════════════════════════════════════════════════════════
# §5  BENCHMARKS (pytest-benchmark)
# ══════════════════════════════════════════════════════════════════════════════

def _make_random_codes(n: int, max_val: int = 3) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, max_val + 1, size=n, dtype=np.uint8)


def test_benchmark_pack_1m(benchmark):
    """Benchmark: empaquetar 1M códigos nucleotídicos."""
    codes = _make_random_codes(1_000_000)
    benchmark(BitPacker.pack, codes)


def test_benchmark_unpack_1m(benchmark):
    """Benchmark: desempaquetar 1M códigos nucleotídicos."""
    codes  = _make_random_codes(1_000_000)
    packed = BitPacker.pack(codes)
    benchmark(BitPacker.unpack, packed, 1_000_000)


def test_benchmark_getitem_single(benchmark):
    """Benchmark: acceso O(1) a una posición en secuencia de 100K símbolos."""
    codes = _make_random_codes(100_000)
    seq   = PackedSequence("b", SeqType.NUCLEOTIDE, 100_000, BitPacker.pack(codes))
    benchmark(seq.__getitem__, 50_000)
