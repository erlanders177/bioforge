"""
tests/test_translator.py
Suite de tests para smart_translator.py.

Cubre:
  - Corrección biológica del código genético (todos los codones)
  - Detección de ATG y truncado en STOP
  - Propiedades Hypothesis: round-trip y longitud mínima de proteína
  - Rutas de error (tipo incorrecto, sin ATG, secuencia corta)
  - Benchmark con pytest-benchmark

Ejecutar:
    pytest tests/test_translator.py -v
    pytest tests/test_translator.py --benchmark-only
"""

import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as np_arrays

from bioforge import BitPacker, PackedSequence, SeqType, SmartImporter
from bioforge import BioForgeError, SequenceTypeError, TranslationError
from bioforge import SmartTranslator


# ── Estrategias Hypothesis ─────────────────────────────────────────────────────

# Secuencias nucleotídicas aleatorias que contienen al menos un ATG
def _fasta_with_atg(draw):
    prefix = draw(st.text(alphabet="ACGT", min_size=0, max_size=30))
    suffix = draw(st.text(alphabet="ACGT", min_size=3, max_size=200))
    return f">test\n{prefix}ATG{suffix}\n"

fasta_with_atg = st.builds(
    lambda: None
).flatmap(lambda _: st.builds(
    lambda prefix, suffix: f">test\n{prefix}ATG{suffix}\n",
    prefix=st.text(alphabet="ACGT", min_size=0, max_size=30),
    suffix=st.text(alphabet="ACGT", min_size=3, max_size=150),
))


# ══════════════════════════════════════════════════════════════════════════════
# §1  CORRECCIÓN BIOLÓGICA
# ══════════════════════════════════════════════════════════════════════════════

# Todos los codones del Código Genético Estándar con su AA esperado
CODON_TABLE = {
    # Fenilalanina
    "TTT": "F", "TTC": "F",
    # Leucina
    "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    # Isoleucina
    "ATT": "I", "ATC": "I", "ATA": "I",
    # Metionina (inicio)
    "ATG": "M",
    # Valina
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    # Serina
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "AGT": "S", "AGC": "S",
    # Prolina
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    # Treonina
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    # Alanina
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    # Tirosina
    "TAT": "Y", "TAC": "Y",
    # Histidina
    "CAT": "H", "CAC": "H",
    # Glutamina
    "CAA": "Q", "CAG": "Q",
    # Asparagina
    "AAT": "N", "AAC": "N",
    # Lisina
    "AAA": "K", "AAG": "K",
    # Ácido aspártico
    "GAT": "D", "GAC": "D",
    # Ácido glutámico
    "GAA": "E", "GAG": "E",
    # Cisteína
    "TGT": "C", "TGC": "C",
    # Triptófano
    "TGG": "W",
    # Arginina
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGA": "R", "AGG": "R",
    # Glicina
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    # STOP
    "TAA": "*", "TAG": "*", "TGA": "*",
}

@pytest.mark.parametrize("codon,expected_aa", [
    (codon, aa) for codon, aa in CODON_TABLE.items() if aa != "*"
])
def test_single_codon_translation(codon, expected_aa):
    """Cada codón sense debe traducirse al aminoácido correcto."""
    fasta = f">test\nATG{codon}TAA\n"
    nuc = SmartImporter.from_string(fasta, force_type=SeqType.NUCLEOTIDE)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        prot = SmartTranslator.translate(nuc, warn_short=False)
    result = prot.to_string()
    # primer AA siempre M (el ATG de inicio), segundo es el codón testeado
    assert result[0] == "M", f"Primer AA debe ser M, got {result[0]!r}"
    assert result[1] == expected_aa, (
        f"Codón {codon}: esperado {expected_aa!r}, obtenido {result[1]!r}"
    )


@pytest.mark.parametrize("stop_codon", ["TAA", "TAG", "TGA"])
def test_stop_codons_terminate_translation(stop_codon):
    """Los tres codones STOP deben truncar la proteína correctamente."""
    fasta = f">test\nATGAAAGGG{stop_codon}CCCGGG\n"
    nuc  = SmartImporter.from_string(fasta, force_type=SeqType.NUCLEOTIDE)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        prot = SmartTranslator.translate(nuc, warn_short=False)
    assert prot.to_string() == "MKG", (
        f"STOP {stop_codon}: proteína debe ser 'MKG', got {prot.to_string()!r}"
    )


def test_hbb_human_translation():
    """HBB_HUMAN: secuencia real de hemoglobina beta, resultado conocido."""
    fasta = """\
>NM_000518.5|HBB_HUMAN|partial CDS
NNNAACCCATGGTGCACCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGT
GGATGAAGTTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGGTTCTTTGAG
TCCTTTGGGGATCTGTCCACTCCTGATGCTGTTATGGGCAACCCTAAGGTGAAGGCTCATGGCAAGAAAG
TGCTCGGTGCCTTTAGTGATGGCCTGGCTCACCTGGACAACCTCAAGGGCACCTTTGCCACACTGAGTGA
GCTGCACTGTGACAAGCTGCACGTGGATCCTGAGAACTTCAGGCTCCTGGGCAACGTGCTGGTCTGTGTG
CTGGCCCATCACTTTGGCAAAGAATTCACCCCACCAGTGCAGGCTGCCTATCAGAAAGTGGTGGCTGGTGT
GGCTAATGCCCTGGCCCACAAGTATCACTAA
"""
    nuc  = SmartImporter.from_string(fasta, force_type=SeqType.NUCLEOTIDE)[0]
    prot = SmartTranslator.translate(nuc)
    seq  = prot.to_string()
    assert seq.startswith("MVHLTPEEKS"), f"HBB debe empezar con MVHLTPEEKS, got {seq[:10]!r}"
    assert len(seq) > 100, "HBB debe tener > 100 aminoácidos"


def test_atg_search_skips_upstream_noise():
    """El ORF finder debe ignorar bases antes del primer ATG."""
    fasta = ">test\nNNNNNNATGAAAGGGTAA\n"
    nuc  = SmartImporter.from_string(fasta, force_type=SeqType.NUCLEOTIDE)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        prot = SmartTranslator.translate(nuc, warn_short=False)
    assert prot.to_string() == "MKG"


def test_short_orf_triggers_warning():
    """Proteínas < 50 aa deben emitir UserWarning."""
    fasta = ">test\nATGAAAGGGTAA\n"
    nuc = SmartImporter.from_string(fasta, force_type=SeqType.NUCLEOTIDE)[0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SmartTranslator.translate(nuc)
    assert any(issubclass(w.category, UserWarning) for w in caught)


def test_protein_type_in_result():
    """La proteína resultante debe ser SeqType.PROTEIN."""
    fasta = ">test\nATGAAAGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGTAA\n"
    nuc  = SmartImporter.from_string(fasta, force_type=SeqType.NUCLEOTIDE)[0]
    prot = SmartTranslator.translate(nuc)
    assert prot.seq_type == SeqType.PROTEIN


def test_header_contains_orf_position():
    """La cabecera de la proteína debe incluir la posición del ORF."""
    fasta = ">mi_gen\nNNNATGAAAGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGTAA\n"
    nuc  = SmartImporter.from_string(fasta, force_type=SeqType.NUCLEOTIDE)[0]
    prot = SmartTranslator.translate(nuc)
    assert "[PROT | ORF@" in prot.header
    assert "mi_gen" in prot.header


# ══════════════════════════════════════════════════════════════════════════════
# §2  PROPIEDADES HYPOTHESIS
# ══════════════════════════════════════════════════════════════════════════════

@given(
    prefix=st.text(alphabet="ACGT", min_size=0, max_size=20),
    body=st.text(alphabet="ACGT", min_size=6, max_size=120),
)
@settings(max_examples=200)
def test_translation_result_is_protein(prefix, body):
    """Para cualquier CDS válido, el resultado siempre es SeqType.PROTEIN."""
    fasta = f">h\n{prefix}ATG{body}TAA\n"
    nuc = SmartImporter.from_string(fasta, force_type=SeqType.NUCLEOTIDE)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prot = SmartTranslator.translate(nuc, warn_short=False)
    assert prot.seq_type == SeqType.PROTEIN


@given(
    prefix=st.text(alphabet="ACGT", min_size=0, max_size=20),
    body=st.text(alphabet="ACGT", min_size=6, max_size=120),
)
@settings(max_examples=200)
def test_protein_starts_with_methionine(prefix, body):
    """Toda proteína traducida debe empezar con M (Metionina)."""
    fasta = f">h\n{prefix}ATG{body}TAA\n"
    nuc = SmartImporter.from_string(fasta, force_type=SeqType.NUCLEOTIDE)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prot = SmartTranslator.translate(nuc, warn_short=False)
    assert prot.to_string()[0] == "M", "Toda proteína debe comenzar con M"


@given(
    prefix=st.text(alphabet="ACGT", min_size=0, max_size=20),
    body=st.text(alphabet="ACGT", min_size=6, max_size=120),
)
@settings(max_examples=200)
def test_no_stop_in_protein_string(prefix, body):
    """La proteína resultante nunca debe contener '*' (STOP excluido)."""
    fasta = f">h\n{prefix}ATG{body}TAA\n"
    nuc = SmartImporter.from_string(fasta, force_type=SeqType.NUCLEOTIDE)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prot = SmartTranslator.translate(nuc, warn_short=False)
    assert "*" not in prot.to_string(), "STOP no debe aparecer en la cadena proteica"


# ══════════════════════════════════════════════════════════════════════════════
# §3  RUTAS DE ERROR
# ══════════════════════════════════════════════════════════════════════════════

def test_error_protein_input():
    """Pasar una proteína al traductor debe lanzar TypeError."""
    prot = PackedSequence(
        header="p", seq_type=SeqType.PROTEIN, n_symbols=4,
        data=BitPacker.pack(np.array([4, 5, 6, 7], dtype=np.uint8)),
    )
    with pytest.raises(TypeError):
        SmartTranslator.translate(prot)


def test_error_no_atg():
    """Una secuencia sin ATG debe lanzar ValueError."""
    nuc = SmartImporter.from_string(
        ">no_atg\nCCCGGGTTTACCCACC\n", force_type=SeqType.NUCLEOTIDE
    )[0]
    with pytest.raises(ValueError, match="ATG"):
        SmartTranslator.translate(nuc)


def test_error_sequence_too_short():
    """Una secuencia de < 3 nucleótidos debe lanzar ValueError."""
    nuc = PackedSequence(
        header="short", seq_type=SeqType.NUCLEOTIDE, n_symbols=2,
        data=BitPacker.pack(np.array([0, 1], dtype=np.uint8)),
    )
    with pytest.raises(ValueError):
        SmartTranslator.translate(nuc)


def test_error_seq_no_es_packed_sequence():
    """Pasar algo que no es PackedSequence debe lanzar TypeError."""
    with pytest.raises(TypeError, match="PackedSequence"):
        SmartTranslator.translate("ATGAAATAA")


# ══════════════════════════════════════════════════════════════════════════════
# §3b  TESTS OPCIONALES — comportamientos adicionales
# ══════════════════════════════════════════════════════════════════════════════

def test_multiples_atg_usa_el_primero():
    """Si hay varios ATG, la traducción empieza en el primero."""
    # ATG en pos 0 y otro en pos 9. Debe usar el de pos 0.
    fasta = ">test\nATGAAAGGGATGCCCTAA\n"
    nuc  = SmartImporter.from_string(fasta, force_type=SeqType.NUCLEOTIDE)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prot = SmartTranslator.translate(nuc, warn_short=False)
    assert prot.to_string().startswith("MKG")


def test_traduccion_sin_stop_va_hasta_el_final():
    """Sin codón STOP, la traducción debe llegar al último codón completo."""
    # Secuencia sin STOP — debe traducir todos los codones disponibles
    fasta = ">test\nATGAAAGGGCCCTTT\n"   # 15 nt = 5 codones: ATG AAA GGG CCC TTT
    nuc  = SmartImporter.from_string(fasta, force_type=SeqType.NUCLEOTIDE)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prot = SmartTranslator.translate(nuc, warn_short=False)
    assert "*" not in prot.to_string()
    assert len(prot.to_string()) == 5


def test_cabecera_proteina_contiene_posicion_orf():
    """La cabecera de la proteína debe incluir la posición donde empieza el ORF."""
    fasta = ">mi_gen\nGGGATGAAAGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGTAA\n"
    nuc  = SmartImporter.from_string(fasta, force_type=SeqType.NUCLEOTIDE)[0]
    prot = SmartTranslator.translate(nuc)
    assert "ORF@3" in prot.header   # ATG empieza en posición 3


def test_resultado_es_packed_sequence_proteina():
    """El resultado siempre debe ser un PackedSequence de tipo PROTEIN."""
    fasta = ">test\nATGAAAGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGTAA\n"
    nuc  = SmartImporter.from_string(fasta, force_type=SeqType.NUCLEOTIDE)[0]
    prot = SmartTranslator.translate(nuc)
    assert isinstance(prot, PackedSequence)
    assert prot.seq_type == SeqType.PROTEIN


# ══════════════════════════════════════════════════════════════════════════════
# §4  BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════════

def _make_cds(n_codons: int) -> PackedSequence:
    """Genera una CDS sintética de n_codons codones (+ ATG inicio + TAA stop)."""
    rng    = np.random.default_rng(42)
    bases  = np.array([0, 1, 2, 3], dtype=np.uint8)
    body   = rng.choice(bases, size=n_codons * 3)
    start  = np.array([0, 3, 2], dtype=np.uint8)   # ATG
    stop   = np.array([3, 0, 0], dtype=np.uint8)   # TAA
    codes  = np.concatenate([start, body, stop])
    return PackedSequence(
        header="bench_cds",
        seq_type=SeqType.NUCLEOTIDE,
        n_symbols=len(codes),
        data=BitPacker.pack(codes),
    )


def test_benchmark_translate_1k_codons(benchmark):
    """Benchmark: traducir CDS de 1 000 codones."""
    cds = _make_cds(1_000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        benchmark(SmartTranslator.translate, cds, False)


def test_benchmark_translate_10k_codons(benchmark):
    """Benchmark: traducir CDS de 10 000 codones."""
    cds = _make_cds(10_000)
    benchmark(SmartTranslator.translate, cds, False)


# ══════════════════════════════════════════════════════════════════════════════
# §5  JERARQUÍA DE EXCEPCIONES — BioForgeError como base común
# ══════════════════════════════════════════════════════════════════════════════

def test_translation_error_capturado_como_bioengine_error():
    """Un error de traducción (sin ATG) debe poder capturarse con BioForgeError."""
    seq = SmartImporter.from_string(">no_atg\nCCCCCCCCCCCC\n",
                                    force_type=SeqType.NUCLEOTIDE)[0]
    with pytest.raises(BioForgeError):
        SmartTranslator.translate(seq)


def test_translation_error_es_subclase_correcta():
    """TranslationError debe ser subclase de BioForgeError y ValueError."""
    assert issubclass(TranslationError, BioForgeError)
    assert issubclass(TranslationError, ValueError)


def test_sequence_type_error_capturado_como_bioengine_error():
    """Pasar un str a translate debe capturarse tanto con TypeError como BioForgeError."""
    with pytest.raises(BioForgeError):
        SmartTranslator.translate("ATGAAATAA")
    with pytest.raises(TypeError):
        SmartTranslator.translate("ATGAAATAA")
