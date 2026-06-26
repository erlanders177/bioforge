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

from bioforge import (
    BitPacker, BioCode, PackedSequence, SeqType,
    SmartImporter, compute_stats, NUC_LUT, AA_LUT,
    BioForgeError, SequenceTypeError, SequenceValueError,
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


# ══════════════════════════════════════════════════════════════════════════════
# §5  TESTS OPCIONALES — casos límite y características adicionales
# ══════════════════════════════════════════════════════════════════════════════

def test_to_string_nucleotide():
    """to_string() debe devolver la secuencia IUPAC correcta para nucleótidos."""
    fasta = ">test\nACGT\n"
    seq = SmartImporter.from_string(fasta, force_type=SeqType.NUCLEOTIDE)[0]
    assert seq.to_string() == "ACGT"


def test_to_string_protein():
    """to_string() debe devolver la secuencia correcta para proteínas."""
    fasta = ">test\nMVHL*\n"
    seq = SmartImporter.from_string(fasta, force_type=SeqType.PROTEIN)[0]
    assert seq.to_string() == "MVHL*"


def test_negative_index():
    """Indexación negativa debe funcionar igual que en listas de Python."""
    fasta = ">test\nACGT\n"
    seq = SmartImporter.from_string(fasta, force_type=SeqType.NUCLEOTIDE)[0]
    assert seq[-1] == seq[3]   # última base
    assert seq[-4] == seq[0]   # primera base


def test_slice_devuelve_packed_sequence():
    """Un slice de PackedSequence debe devolver otro PackedSequence."""
    fasta = ">test\nATGCATGCATGC\n"
    seq = SmartImporter.from_string(fasta, force_type=SeqType.NUCLEOTIDE)[0]
    sub = seq[4:8]
    assert isinstance(sub, PackedSequence)
    assert sub.n_symbols == 4
    assert sub.to_string() == "ATGC"


def test_igualdad_secuencias_identicas():
    """Dos secuencias con el mismo contenido deben ser iguales."""
    fasta = ">test\nATGC\n"
    s1 = SmartImporter.from_string(fasta, force_type=SeqType.NUCLEOTIDE)[0]
    s2 = SmartImporter.from_string(fasta, force_type=SeqType.NUCLEOTIDE)[0]
    assert s1 == s2


def test_igualdad_secuencias_distintas():
    """Dos secuencias distintas no deben ser iguales."""
    s1 = SmartImporter.from_string(">a\nACGT\n", force_type=SeqType.NUCLEOTIDE)[0]
    s2 = SmartImporter.from_string(">a\nTGCA\n", force_type=SeqType.NUCLEOTIDE)[0]
    assert s1 != s2


def test_compute_stats_composicion():
    """compute_stats debe contar correctamente la composición de bases."""
    from bioforge import compute_stats
    fasta = ">test\nAAAACCGT\n"
    seq   = SmartImporter.from_string(fasta, force_type=SeqType.NUCLEOTIDE)[0]
    stats = compute_stats(seq)
    assert stats.composition["A"] == 4
    assert stats.composition["C"] == 2
    assert stats.composition["G"] == 1
    assert stats.composition["T"] == 1


def test_from_string_vacia_no_registros():
    """Un FASTA vacío debe devolver lista vacía."""
    records = SmartImporter.from_string("")
    assert records == []


def test_minusculas_se_codifican_igual_que_mayusculas():
    """La secuencia en minúsculas debe producir el mismo resultado que en mayúsculas."""
    s_upper = SmartImporter.from_string(">a\nACGT\n", force_type=SeqType.NUCLEOTIDE)[0]
    s_lower = SmartImporter.from_string(">a\nacgt\n", force_type=SeqType.NUCLEOTIDE)[0]
    assert np.array_equal(s_upper.decode(), s_lower.decode())


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


# ══════════════════════════════════════════════════════════════════════════════
# §6  RUTAS DE ERROR — validaciones de entrada
# ══════════════════════════════════════════════════════════════════════════════

def test_pack_array_multidimensional():
    """BitPacker.pack con array 2-D debe lanzar ValueError."""
    codes_2d = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    with pytest.raises(ValueError, match="1-D"):
        BitPacker.pack(codes_2d)


def test_unpack_n_negativo():
    """BitPacker.unpack con n negativo debe lanzar ValueError."""
    packed = BitPacker.pack(np.array([0, 1, 2], dtype=np.uint8))
    with pytest.raises(ValueError):
        BitPacker.unpack(packed, -1)


def test_unpack_packed_insuficiente():
    """BitPacker.unpack con packed demasiado pequeño debe lanzar ValueError."""
    packed = np.array([0xFF], dtype=np.uint8)   # 1 byte = max 1 símbolo
    with pytest.raises(ValueError):
        BitPacker.unpack(packed, 10)


def test_packed_sequence_n_symbols_negativo():
    """PackedSequence con n_symbols < 0 debe lanzar ValueError."""
    with pytest.raises(ValueError):
        PackedSequence(
            header="bad", seq_type=SeqType.NUCLEOTIDE, n_symbols=-1,
            data=np.array([], dtype=np.uint8),
        )


def test_packed_sequence_seq_type_invalido():
    """PackedSequence con seq_type que no es SeqType debe lanzar TypeError."""
    with pytest.raises(TypeError):
        PackedSequence(
            header="bad", seq_type="NUCLEOTIDE",
            n_symbols=0, data=np.array([], dtype=np.uint8),
        )


def test_getitem_out_of_range():
    """Acceder fuera de rango debe lanzar IndexError."""
    seq = SmartImporter.from_string(">t\nACGT\n", force_type=SeqType.NUCLEOTIDE)[0]
    with pytest.raises(IndexError):
        _ = seq[10]
    with pytest.raises(IndexError):
        _ = seq[-10]


def test_getitem_tipo_invalido():
    """Indexar con tipo no válido debe lanzar TypeError."""
    seq = SmartImporter.from_string(">t\nACGT\n", force_type=SeqType.NUCLEOTIDE)[0]
    with pytest.raises(TypeError):
        _ = seq[1.5]


# ══════════════════════════════════════════════════════════════════════════════
# §7  JERARQUÍA DE EXCEPCIONES — BioForgeError como base común
# ══════════════════════════════════════════════════════════════════════════════

def test_bioengine_error_es_base_de_sequence_type_error():
    """SequenceTypeError debe ser instancia de BioForgeError."""
    assert issubclass(SequenceTypeError, BioForgeError)


def test_bioengine_error_es_base_de_sequence_value_error():
    """SequenceValueError debe ser instancia de BioForgeError."""
    assert issubclass(SequenceValueError, BioForgeError)


def test_sequence_type_error_es_type_error():
    """SequenceTypeError también debe ser TypeError (retrocompatibilidad)."""
    assert issubclass(SequenceTypeError, TypeError)


def test_sequence_value_error_es_value_error():
    """SequenceValueError también debe ser ValueError (retrocompatibilidad)."""
    assert issubclass(SequenceValueError, ValueError)


def test_pack_error_capturado_como_bioengine_error():
    """BitPacker.pack con array 2-D debe poder capturarse con BioForgeError."""
    codes_2d = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    with pytest.raises(BioForgeError):
        BitPacker.pack(codes_2d)


def test_packed_sequence_type_error_capturado_como_bioengine_error():
    """PackedSequence con seq_type inválido debe capturarse con BioForgeError."""
    with pytest.raises(BioForgeError):
        PackedSequence(
            header="bad", seq_type="PROTEIN",
            n_symbols=0, data=np.array([], dtype=np.uint8),
        )


# ══════════════════════════════════════════════════════════════════════════════
# §8  REVERSE COMPLEMENT
# ══════════════════════════════════════════════════════════════════════════════

def _nuc(seq_str: str) -> PackedSequence:
    return SmartImporter.from_string(f">t\n{seq_str}\n", force_type=SeqType.NUCLEOTIDE)[0]


def test_reverse_complement_basico():
    """ATGC → GCAT (complemento invertido)."""
    seq = _nuc("ATGC")
    rc  = seq.reverse_complement()
    assert rc.to_string() == "GCAT"


def test_reverse_complement_palindromo():
    """Un palindromo Watson-Crick es idéntico a su RC."""
    seq = _nuc("ATAT")
    rc  = seq.reverse_complement()
    assert rc.to_string() == "ATAT"


def test_reverse_complement_doble_es_identidad():
    """RC(RC(x)) == x para cualquier secuencia nucleotídica."""
    seq = _nuc("ATGCGTACNNTTAA")
    assert seq.reverse_complement().reverse_complement().to_string() == seq.to_string()


@given(nuc_codes)
def test_reverse_complement_doble_es_identidad_property(codes):
    """RC(RC(x)) == x para cualquier array de códigos nucleotídicos."""
    packed = PackedSequence(
        header="t", seq_type=SeqType.NUCLEOTIDE,
        n_symbols=len(codes), data=BitPacker.pack(codes),
    )
    rc2 = packed.reverse_complement().reverse_complement()
    assert np.array_equal(packed.decode(), rc2.decode())


def test_reverse_complement_longitud_preservada():
    """RC mantiene la longitud de la secuencia."""
    seq = _nuc("ATGCGTACGT")
    rc  = seq.reverse_complement()
    assert rc.n_symbols == seq.n_symbols


def test_reverse_complement_preserva_n():
    """Las bases N (UNK) se mantienen como N en el RC."""
    seq = _nuc("ATGN")
    rc  = seq.reverse_complement()
    assert rc.to_string() == "NCAT"


def test_reverse_complement_header_prefijado():
    """El header del RC tiene el prefijo [RC]."""
    seq = _nuc("ATGC")
    rc  = seq.reverse_complement()
    assert rc.header.startswith("[RC]")


def test_reverse_complement_error_en_proteina():
    """reverse_complement en una proteína lanza SequenceTypeError."""
    prot = SmartImporter.from_string(">p\nMKGFEI\n")[0]
    with pytest.raises(SequenceTypeError):
        prot.reverse_complement()


def test_reverse_complement_pares_complementarios():
    """Cada base mapea al complemento correcto: A↔T, C↔G."""
    seq = _nuc("ACGT")
    rc  = seq.reverse_complement()
    # RC de ACGT: reverse es TGCA → complement es ACGT
    assert rc.to_string() == "ACGT"
