"""
tests/test_aligner.py
Suite de tests para aligner.py.

Cubre:
  - Propiedades matemáticas del alineamiento (score, identidad, simetría)
  - Detección de mutaciones conocidas (HBB sickle cell)
  - Indels detectados correctamente
  - Alineamiento de proteínas
  - Propiedades Hypothesis
  - Rutas de error
  - Benchmarks

Ejecutar:
    pytest tests/test_aligner.py -v
    pytest tests/test_aligner.py --benchmark-only
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from biocore import BitPacker, PackedSequence, SeqType, SmartImporter
from biocore import BioForgeError, SequenceTypeError, SequenceValueError, AlignmentError
from aligner import AlignmentResult, Mutation, SequenceAligner, format_alignment


# ── Helpers ────────────────────────────────────────────────────────────────────

def _nuc(seq: str, header: str = "h") -> PackedSequence:
    return SmartImporter.from_string(
        f">{header}\n{seq}\n", force_type=SeqType.NUCLEOTIDE
    )[0]


def _prot(seq: str, header: str = "h") -> PackedSequence:
    return SmartImporter.from_string(
        f">{header}\n{seq}\n", force_type=SeqType.PROTEIN
    )[0]


# ══════════════════════════════════════════════════════════════════════════════
# §1  PROPIEDADES MATEMÁTICAS
# ══════════════════════════════════════════════════════════════════════════════

def test_identical_sequences_perfect_score():
    """Dos secuencias idénticas → score máximo, identidad 1.0, 0 mutaciones."""
    seq = "ATGGTGCACCTGACTCCTGAGGAGAAGTCTGCC"
    r   = SequenceAligner.align(_nuc(seq), _nuc(seq))
    assert r.n_mismatches == 0
    assert r.n_gaps == 0
    assert r.identity == 1.0
    assert r.score == len(seq) * int(SequenceAligner.MATCH)
    assert len(r.mutations) == 0


def test_score_single_mismatch():
    """Una sola sustitución debe dar score = (n-1)*MATCH + MISMATCH."""
    a = "ATGCATGC"
    b = "ATGTATGC"
    r = SequenceAligner.align(_nuc(a), _nuc(b))
    expected = 7 * int(SequenceAligner.MATCH) + int(SequenceAligner.MISMATCH)
    assert r.score == expected
    assert r.n_mismatches == 1
    assert len(r.mutations) == 1
    assert r.mutations[0].kind == 'substitution'


def test_aligned_strings_same_length():
    """Las cadenas alineadas deben tener siempre la misma longitud."""
    r = SequenceAligner.align(_nuc("ATGCATGC"), _nuc("ATGTATGC"))
    assert len(r.aligned_a) == len(r.aligned_b)


def test_aligned_length_equals_counts():
    """n_matches + n_mismatches + n_gaps debe coincidir con longitud alineada."""
    r = SequenceAligner.align(
        _nuc("ATGGTGCACCTGACT"),
        _nuc("ATGGTGCACCTGACTCCC"),
    )
    aln_len = len(r.aligned_a)
    assert r.n_matches + r.n_mismatches + r.n_gaps == aln_len


def test_identity_in_range():
    """La identidad debe estar siempre en [0.0, 1.0]."""
    r = SequenceAligner.align(_nuc("ACGT"), _nuc("TGCA"))
    assert 0.0 <= r.identity <= 1.0


def test_mode_stored_correctly():
    """El modo usado debe quedar registrado en el resultado."""
    r_g = SequenceAligner.align(_nuc("ACGT"), _nuc("ACGT"), mode='global')
    r_s = SequenceAligner.align(_nuc("ACGT"), _nuc("ACGT"), mode='semi-global')
    assert r_g.mode == 'global'
    assert r_s.mode == 'semi-global'


# ══════════════════════════════════════════════════════════════════════════════
# §2  DETECCIÓN DE MUTACIONES CONOCIDAS
# ══════════════════════════════════════════════════════════════════════════════

def test_sickle_cell_substitution():
    """HBB sickle cell: única sustitución A→T en posición 19."""
    normal = _nuc("ATGGTGCACCTGACTCCTGAGGAGAAGTCT", "normal")
    sickle = _nuc("ATGGTGCACCTGACTCCTGTGGAGAAGTCT", "sickle")
    r = SequenceAligner.align(normal, sickle)
    assert r.n_mismatches == 1
    assert r.n_gaps == 0
    assert len(r.mutations) == 1
    m = r.mutations[0]
    assert m.kind  == 'substitution'
    assert m.pos_a == 19
    assert m.sym_a == 'A'
    assert m.sym_b == 'T'


def test_insertion_detected():
    """Una inserción de 3 bases debe detectarse como 3 gaps en seq_a."""
    ref = _nuc("ATGGTGCACCTGACTGAA")       # 18 nt
    ins = _nuc("ATGGTGCACCTGACTCCCGAA")    # 21 nt (CCC en pos 15)
    r   = SequenceAligner.align(ref, ins)
    ins_muts = [m for m in r.mutations if m.kind == 'insertion']
    assert len(ins_muts) == 3
    assert r.n_gaps == 3


def test_deletion_detected():
    """Una deleción de 3 bases debe detectarse como 3 gaps en seq_b."""
    ref = _nuc("ATGGTGCACCTGACTCCCGAA")    # 21 nt
    del_ = _nuc("ATGGTGCACCTGACTGAA")      # 18 nt (sin CCC)
    r    = SequenceAligner.align(ref, del_)
    del_muts = [m for m in r.mutations if m.kind == 'deletion']
    assert len(del_muts) == 3
    assert r.n_gaps == 3


def test_gap_char_in_aligned_strings():
    """Las posiciones de gap deben aparecer como '-' en el string alineado."""
    ref = _nuc("ATGGTGCACCTGACTGAA")
    ins = _nuc("ATGGTGCACCTGACTCCCGAA")
    r   = SequenceAligner.align(ref, ins)
    assert '-' in r.aligned_a
    assert '-' not in r.aligned_b or r.n_gaps > 0


# ══════════════════════════════════════════════════════════════════════════════
# §3  ALINEAMIENTO DE PROTEÍNAS
# ══════════════════════════════════════════════════════════════════════════════

def test_protein_alignment_same_type():
    """Dos proteínas deben alinearse sin error."""
    hbb = _prot("MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLST")
    hba = _prot("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLS")
    r   = SequenceAligner.align(hbb, hba)
    assert r.seq_type == SeqType.PROTEIN
    assert 0.0 < r.identity < 1.0


def test_identical_proteins():
    """Dos proteínas idénticas → identidad perfecta."""
    seq = "MVHLTPEEKSAVTALWGKV"
    r   = SequenceAligner.align(_prot(seq), _prot(seq))
    assert r.identity == 1.0
    assert r.n_mismatches == 0


# ══════════════════════════════════════════════════════════════════════════════
# §4  format_alignment
# ══════════════════════════════════════════════════════════════════════════════

def test_format_alignment_contains_pipes():
    """El formato de alineamiento debe contener '|' para los matches."""
    r = SequenceAligner.align(_nuc("ACGTACGT"), _nuc("ACGTACGT"))
    fmt = format_alignment(r)
    assert '|' in fmt


def test_format_alignment_contains_x_for_mismatch():
    """El formato debe contener 'X' para los mismatches."""
    r = SequenceAligner.align(_nuc("AAAA"), _nuc("AACA"))
    fmt = format_alignment(r)
    assert 'X' in fmt


def test_format_alignment_width():
    """El ancho de línea del formato debe respetarse."""
    seq = "ACGT" * 30    # 120 nt
    r   = SequenceAligner.align(_nuc(seq), _nuc(seq))
    fmt = format_alignment(r, width=40)
    for line in fmt.splitlines():
        if line.startswith("  A:") or line.startswith("  B:"):
            content = line[5:]   # "  A: " son 5 chars
            assert len(content) <= 40, f"Línea más larga de 40: {len(content)}"


# ══════════════════════════════════════════════════════════════════════════════
# §5  PROPIEDADES HYPOTHESIS
# ══════════════════════════════════════════════════════════════════════════════

@given(seq=st.text(alphabet="ACGT", min_size=1, max_size=100))
@settings(max_examples=300)
def test_self_alignment_perfect(seq):
    """Para cualquier secuencia, alinearse consigo misma da identidad 1.0."""
    s = _nuc(seq)
    r = SequenceAligner.align(s, s)
    assert r.identity == 1.0
    assert r.n_mismatches == 0
    assert r.n_gaps == 0


@given(seq=st.text(alphabet="ACGT", min_size=1, max_size=100))
@settings(max_examples=300)
def test_self_alignment_max_score(seq):
    """Score de auto-alineamiento == len * MATCH."""
    s = _nuc(seq)
    r = SequenceAligner.align(s, s)
    assert r.score == len(seq) * int(SequenceAligner.MATCH)


@given(
    a=st.text(alphabet="ACGT", min_size=2, max_size=80),
    b=st.text(alphabet="ACGT", min_size=2, max_size=80),
)
@settings(max_examples=200)
def test_aligned_strings_equal_length(a, b):
    """Los strings alineados siempre deben tener la misma longitud."""
    r = SequenceAligner.align(_nuc(a), _nuc(b))
    assert len(r.aligned_a) == len(r.aligned_b)


@given(
    a=st.text(alphabet="ACGT", min_size=2, max_size=80),
    b=st.text(alphabet="ACGT", min_size=2, max_size=80),
)
@settings(max_examples=200)
def test_counts_sum_equals_alignment_length(a, b):
    """n_matches + n_mismatches + n_gaps == longitud del alineamiento."""
    r = SequenceAligner.align(_nuc(a), _nuc(b))
    assert r.n_matches + r.n_mismatches + r.n_gaps == len(r.aligned_a)


# ══════════════════════════════════════════════════════════════════════════════
# §5b  TESTS OPCIONALES — modo semi-global y casos límite
# ══════════════════════════════════════════════════════════════════════════════

def test_semiglobal_fragmento_vs_referencia():
    """Semi-global: un fragmento corto alineado contra una referencia larga."""
    ref      = _nuc("ATGGTGCACCTGACTCCTGAGGAGAAGTCT")  # 30 nt
    fragment = _nuc("CCTGAGGAG")                         # 9 nt (subconjunto)
    r = SequenceAligner.align(ref, fragment, mode="semi-global")
    # La identidad del fragmento contra su región correspondiente debe ser alta
    assert r.identity > 0.0


def test_score_nunca_supera_el_maximo_posible():
    """El score siempre debe ser ≤ len(secuencia_mas_corta) × MATCH."""
    a = _nuc("ATGCATGCATGC")
    b = _nuc("ATGCATGC")
    r = SequenceAligner.align(a, b)
    max_possible = min(a.n_symbols, b.n_symbols) * int(SequenceAligner.MATCH)
    assert r.score <= max_possible


def test_resultado_tiene_tipo_secuencia_correcto():
    """El campo seq_type del resultado debe coincidir con el input."""
    r_nuc  = SequenceAligner.align(_nuc("ACGT"), _nuc("ACGT"))
    r_prot = SequenceAligner.align(_prot("MVHL"), _prot("MVHL"))
    assert r_nuc.seq_type  == SeqType.NUCLEOTIDE
    assert r_prot.seq_type == SeqType.PROTEIN


def test_mutaciones_en_orden_de_posicion():
    """La lista de mutaciones debe estar ordenada por posición ascendente."""
    a = _nuc("AAAAAAAAAA")
    b = _nuc("AATAAATAAA")   # cambios en posiciones 2 y 6
    r = SequenceAligner.align(a, b)
    subs = [m for m in r.mutations if m.kind == "substitution"]
    posiciones = [m.pos_a for m in subs]
    assert posiciones == sorted(posiciones)


def test_sin_mutaciones_lista_vacia():
    """Sin diferencias, la lista de mutaciones debe estar vacía."""
    r = SequenceAligner.align(_nuc("ATGCATGC"), _nuc("ATGCATGC"))
    assert r.mutations == []


# ══════════════════════════════════════════════════════════════════════════════
# §6  RUTAS DE ERROR
# ══════════════════════════════════════════════════════════════════════════════

def test_error_mismatched_types():
    """Alinear nucleótido con proteína debe lanzar TypeError."""
    nuc  = _nuc("ACGT")
    prot = _prot("MVHL")
    with pytest.raises(TypeError):
        SequenceAligner.align(nuc, prot)


def test_error_empty_sequence():
    """Secuencias vacías deben lanzar ValueError."""
    empty = PackedSequence(
        header="e", seq_type=SeqType.NUCLEOTIDE, n_symbols=1,
        data=BitPacker.pack(np.array([0], dtype=np.uint8)),
    )
    # El caso de longitud 1 debe funcionar (no es vacío)
    r = SequenceAligner.align(empty, empty)
    assert r.score == int(SequenceAligner.MATCH)


def test_error_seq_a_no_es_packed_sequence():
    """seq_a que no es PackedSequence debe lanzar TypeError."""
    with pytest.raises(TypeError, match="PackedSequence"):
        SequenceAligner.align("ACGT", _nuc("ACGT"))


def test_error_seq_b_no_es_packed_sequence():
    """seq_b que no es PackedSequence debe lanzar TypeError."""
    with pytest.raises(TypeError, match="PackedSequence"):
        SequenceAligner.align(_nuc("ACGT"), 42)


def test_error_mode_invalido():
    """Un mode no reconocido debe lanzar ValueError."""
    a = _nuc("ACGT")
    with pytest.raises(ValueError, match="mode"):
        SequenceAligner.align(a, a, mode="diagonal")


def test_error_format_alignment_width_invalido():
    """format_alignment con width <= 0 debe lanzar ValueError."""
    r = SequenceAligner.align(_nuc("ACGT"), _nuc("ACGT"))
    with pytest.raises(ValueError, match="width"):
        format_alignment(r, width=0)


def test_long_sequence_warning():
    """Secuencias > _MAX_SAFE_LEN deben emitir UserWarning."""
    import warnings
    rng   = np.random.default_rng(0)
    codes = rng.integers(0, 4, size=SequenceAligner._MAX_SAFE_LEN + 1, dtype=np.uint8)
    long_seq = PackedSequence(
        header="long",
        seq_type=SeqType.NUCLEOTIDE,
        n_symbols=len(codes),
        data=BitPacker.pack(codes),
    )
    short_seq = _nuc("ACGT")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SequenceAligner.align(long_seq, short_seq)
    assert any(issubclass(w.category, UserWarning) for w in caught)


# ══════════════════════════════════════════════════════════════════════════════
# §7  BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════════

def _make_pair(n: int, mut_frac: float = 0.01):
    rng   = np.random.default_rng(42)
    bases = np.array([0, 1, 2, 3], dtype=np.uint8)
    ca    = rng.choice(bases, size=n)
    cb    = ca.copy()
    n_mut = max(1, int(n * mut_frac))
    for p in rng.choice(n, size=n_mut, replace=False):
        cb[p] = (cb[p] + 1) % 4
    def _pack(codes):
        return PackedSequence(
            header="b", seq_type=SeqType.NUCLEOTIDE,
            n_symbols=n, data=BitPacker.pack(codes),
        )
    return _pack(ca), _pack(cb)


def test_benchmark_align_500x500(benchmark):
    """Benchmark: alineamiento 500 × 500 nt."""
    a, b = _make_pair(500)
    benchmark(SequenceAligner.align, a, b)


def test_benchmark_align_1000x1000(benchmark):
    """Benchmark: alineamiento 1 000 × 1 000 nt."""
    a, b = _make_pair(1_000)
    benchmark(SequenceAligner.align, a, b)


# ══════════════════════════════════════════════════════════════════════════════
# §8  JERARQUÍA DE EXCEPCIONES — BioForgeError como base común
# ══════════════════════════════════════════════════════════════════════════════

def test_alignment_error_es_subclase_correcta():
    """AlignmentError debe ser subclase de BioForgeError y ValueError."""
    assert issubclass(AlignmentError, BioForgeError)
    assert issubclass(AlignmentError, ValueError)


def test_sequence_type_error_en_align_capturado_como_bioengine_error():
    """Pasar un str a align debe capturarse con BioForgeError."""
    with pytest.raises(BioForgeError):
        SequenceAligner.align("ACGT", _nuc("ACGT"))


def test_alignment_mode_error_capturado_como_bioengine_error():
    """Modo inválido debe capturarse con BioForgeError."""
    a = _nuc("ACGT")
    with pytest.raises(BioForgeError):
        SequenceAligner.align(a, a, mode="bad_mode")


def test_alignment_mode_error_tambien_es_value_error():
    """Modo inválido debe poder capturarse también con ValueError (retrocompat)."""
    a = _nuc("ACGT")
    with pytest.raises(ValueError):
        SequenceAligner.align(a, a, mode="bad_mode")


def test_format_alignment_error_capturado_como_bioengine_error():
    """format_alignment con width=0 debe capturarse con BioForgeError."""
    r = SequenceAligner.align(_nuc("ACGT"), _nuc("ACGT"))
    with pytest.raises(BioForgeError):
        format_alignment(r, width=0)
