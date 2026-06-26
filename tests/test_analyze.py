"""
tests/test_analyze.py
Suite de tests para analyze.py — pipeline de análisis de mutaciones.

Cubre:
  - Los tres modos: dna, protein, both
  - Mutaciones sinónimas (ADN cambia, aminoácido no)
  - Pipeline completo con secuencias reales (HBB sickle cell)
  - Generación y contenido del informe (build_report)
  - Clasificación conservativa/radical de sustituciones
  - Gestión de errores (archivo no encontrado, tipos incompatibles)
  - Interfaz de línea de comandos (main)

Ejecutar:
    pytest tests/test_analyze.py -v
"""

import sys
from pathlib import Path
import pytest

from biocore import SeqType
from analyze import run, build_report, AnalysisResult, _change_type, main


# ══════════════════════════════════════════════════════════════════════════════
# §0  FIXTURES — archivos FASTA temporales
# ══════════════════════════════════════════════════════════════════════════════

# Secuencia HBB normal (hemoglobina beta, 135 nt, 44 aa)
_SEQ_NORMAL   = "ATGGTGCACCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGGTTCTTTGAGTAA"
# Mutación sickle cell: posición 19, A→T (GAG→GTG, Glu→Val — RADICAL)
_SEQ_SICKLE   = "ATGGTGCACCTGACTCCTGTGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGGTTCTTTGAGTAA"
# Mutación sinónima: posición 17, T→C (CCT→CCC, Pro→Pro — sin cambio de AA)
_SEQ_SYNONYMOUS = _SEQ_NORMAL[:17] + "C" + _SEQ_NORMAL[18:]


@pytest.fixture
def fa(tmp_path):
    """Crea archivos FASTA temporales y devuelve función helper."""
    def _make(name: str, header: str, sequence: str) -> str:
        p = tmp_path / name
        p.write_text(f">{header}\n{sequence}\n", encoding="utf-8")
        return str(p)
    return _make


@pytest.fixture
def hbb_normal(fa):
    return fa("normal.fa", "HBB_normal", _SEQ_NORMAL)

@pytest.fixture
def hbb_sickle(fa):
    return fa("sickle.fa", "HBB_sickle", _SEQ_SICKLE)

@pytest.fixture
def hbb_synonymous(fa):
    return fa("synonymous.fa", "HBB_synonymous", _SEQ_SYNONYMOUS)

@pytest.fixture
def insulin_ref(fa):
    return fa("ins_ref.fa", "INS_human", "FVNQHLCGSHLVEALYLVCGERGFFYTPKT")

@pytest.fixture
def insulin_var(fa):
    return fa("ins_var.fa", "INS_variant", "FVNQHLCGSDLVEALYLVCGERGFFYTPKT")


# ══════════════════════════════════════════════════════════════════════════════
# §1  MODO ADN — análisis a nivel de nucleótido
# ══════════════════════════════════════════════════════════════════════════════

def test_modo_dna_detecta_mutacion_nucleotidica(hbb_normal, hbb_sickle):
    """El modo dna debe encontrar el cambio de base A→T en la posición 19."""
    r = run(hbb_normal, hbb_sickle, mode="dna")
    assert r.dna_alignment is not None
    assert r.aa_alignment is None
    assert r.dna_alignment.n_mismatches == 1
    assert r.dna_alignment.mutations[0].pos_a == 19
    assert r.dna_alignment.mutations[0].sym_a == "A"
    assert r.dna_alignment.mutations[0].sym_b == "T"


def test_modo_dna_detecta_mutacion_sinonima(hbb_normal, hbb_synonymous):
    """El modo dna debe detectar cambios sinónimos (Pro→Pro) a nivel nucleotídico."""
    r = run(hbb_normal, hbb_synonymous, mode="dna")
    assert r.dna_alignment.n_mismatches == 1
    assert r.dna_alignment.mutations[0].pos_a == 17


def test_modo_dna_secuencias_identicas(hbb_normal, fa):
    """Con secuencias idénticas en modo dna, no debe haber mutaciones."""
    copia = fa("copia.fa", "copia", _SEQ_NORMAL)
    r = run(hbb_normal, copia, mode="dna")
    assert r.dna_alignment.n_mismatches == 0
    assert r.dna_alignment.n_gaps == 0
    assert len(r.dna_alignment.mutations) == 0


def test_modo_dna_proteina_como_entrada_hace_fallback(insulin_ref, insulin_var):
    """Con entrada proteína, el modo dna cae automáticamente a protein."""
    r = run(insulin_ref, insulin_var, mode="dna")
    assert r.mode == "protein"
    assert r.dna_alignment is None
    assert r.aa_alignment is not None


# ══════════════════════════════════════════════════════════════════════════════
# §2  MODO PROTEÍNA — solo cambios de aminoácido
# ══════════════════════════════════════════════════════════════════════════════

def test_modo_protein_detecta_cambio_de_aminoacido(hbb_normal, hbb_sickle):
    """El modo protein debe encontrar la sustitución Glu→Val en posición 7."""
    r = run(hbb_normal, hbb_sickle, mode="protein")
    assert r.aa_alignment is not None
    assert r.dna_alignment is None
    assert r.aa_alignment.n_mismatches == 1
    m = r.aa_alignment.mutations[0]
    assert m.kind == "substitution"
    assert m.sym_a == "E"   # Glu — alelo normal
    assert m.sym_b == "V"   # Val — alelo sickle cell


def test_modo_protein_ignora_mutacion_sinonima(hbb_normal, hbb_synonymous):
    """El modo protein NO debe reportar mutaciones cuando el aminoácido no cambia."""
    r = run(hbb_normal, hbb_synonymous, mode="protein")
    assert r.aa_alignment.n_mismatches == 0
    assert len(r.aa_alignment.mutations) == 0


def test_modo_protein_con_proteinas_directas(insulin_ref, insulin_var):
    """El modo protein funciona con secuencias de proteína como entrada directa."""
    r = run(insulin_ref, insulin_var, mode="protein")
    assert r.was_dna is False
    assert r.aa_alignment is not None
    assert r.aa_alignment.n_mismatches == 1


def test_modo_protein_secuencias_identicas(hbb_normal, fa):
    """Sin diferencias de aminoácidos, el resultado debe ser identidad perfecta."""
    copia = fa("copia.fa", "copia", _SEQ_NORMAL)
    r = run(hbb_normal, copia, mode="protein")
    assert r.aa_alignment.identity == 1.0
    assert len(r.aa_alignment.mutations) == 0


# ══════════════════════════════════════════════════════════════════════════════
# §3  MODO BOTH — ADN + aminoácido en un solo análisis
# ══════════════════════════════════════════════════════════════════════════════

def test_modo_both_tiene_ambas_secciones(hbb_normal, hbb_sickle):
    """El modo both debe producir tanto el análisis de ADN como el de proteína."""
    r = run(hbb_normal, hbb_sickle, mode="both")
    assert r.dna_alignment is not None
    assert r.aa_alignment is not None


def test_modo_both_sickle_cell_completo(hbb_normal, hbb_sickle):
    """Pipeline completo sickle cell: 1 cambio ADN → 1 cambio AA (Glu→Val)."""
    r = run(hbb_normal, hbb_sickle, mode="both")
    assert r.dna_alignment.n_mismatches == 1   # A→T en posición 19
    assert r.aa_alignment.n_mismatches == 1    # Glu→Val en posición 7
    assert r.was_dna is True


def test_modo_both_sinonima_adn_si_proteina_no(hbb_normal, hbb_synonymous):
    """Mutación sinónima: aparece en ADN (1 cambio) pero no en proteína (0 cambios)."""
    r = run(hbb_normal, hbb_synonymous, mode="both")
    assert r.dna_alignment.n_mismatches == 1   # el cambio nucleotídico está ahí
    assert r.aa_alignment.n_mismatches == 0    # pero el aminoácido no cambió


def test_modo_both_default(hbb_normal, hbb_sickle):
    """El modo por defecto de run() debe ser 'both'."""
    r = run(hbb_normal, hbb_sickle)
    assert r.dna_alignment is not None
    assert r.aa_alignment is not None


# ══════════════════════════════════════════════════════════════════════════════
# §4  RESULTADO — campos y tipos
# ══════════════════════════════════════════════════════════════════════════════

def test_resultado_contiene_cabeceras(hbb_normal, hbb_sickle):
    """El resultado debe incluir los headers de las secuencias de entrada."""
    r = run(hbb_normal, hbb_sickle, mode="both")
    assert "HBB_normal" in r.ref_header
    assert "HBB_sickle" in r.query_header


def test_resultado_was_dna_correcto_para_adn(hbb_normal, hbb_sickle):
    r = run(hbb_normal, hbb_sickle, mode="both")
    assert r.was_dna is True


def test_resultado_was_dna_correcto_para_proteina(insulin_ref, insulin_var):
    r = run(insulin_ref, insulin_var, mode="protein")
    assert r.was_dna is False


def test_resultado_proteinas_presentes_en_modo_both(hbb_normal, hbb_sickle):
    """En modo both, las proteínas traducidas deben estar disponibles."""
    r = run(hbb_normal, hbb_sickle, mode="both")
    assert r.ref_protein is not None
    assert r.qry_protein is not None
    assert r.ref_protein.seq_type == SeqType.PROTEIN
    assert r.ref_protein.to_string().startswith("M")   # empieza con Met


# ══════════════════════════════════════════════════════════════════════════════
# §5  INFORME — build_report
# ══════════════════════════════════════════════════════════════════════════════

def test_informe_modo_dna_contiene_seccion_adn(hbb_normal, hbb_sickle):
    r = run(hbb_normal, hbb_sickle, mode="dna")
    report = build_report(r)
    assert "DNA ALIGNMENT" in report
    assert "NUCLEOTIDE MUTATIONS" in report
    assert "PROTEIN ALIGNMENT" not in report


def test_informe_modo_protein_contiene_seccion_proteina(hbb_normal, hbb_sickle):
    r = run(hbb_normal, hbb_sickle, mode="protein")
    report = build_report(r)
    assert "PROTEIN ALIGNMENT" in report
    assert "AMINO ACID MUTATIONS" in report
    assert "DNA ALIGNMENT" not in report


def test_informe_modo_both_contiene_ambas_secciones(hbb_normal, hbb_sickle):
    r = run(hbb_normal, hbb_sickle, mode="both")
    report = build_report(r)
    assert "DNA ALIGNMENT" in report
    assert "PROTEIN ALIGNMENT" in report


def test_informe_sin_mutaciones_dice_identico(hbb_normal, fa):
    copia = fa("copia.fa", "copia", _SEQ_NORMAL)
    r = run(hbb_normal, copia, mode="protein")
    report = build_report(r)
    assert "functionally identical" in report.lower() or "identical" in report.lower()


def test_informe_radical_clasificado_correctamente(hbb_normal, hbb_sickle):
    """La mutación Glu→Val debe aparecer como RADICAL en el informe."""
    r = run(hbb_normal, hbb_sickle, mode="protein")
    report = build_report(r)
    assert "RADICAL" in report


def test_informe_sinonima_mencionada_en_modo_both(hbb_normal, hbb_synonymous):
    """En modo both con mutación sinónima, el informe debe mencionar que hay
    diferencias en ADN pero no en proteína."""
    r = run(hbb_normal, hbb_synonymous, mode="both")
    report = build_report(r)
    assert "synonymous" in report.lower() or "sinónim" in report.lower() or "silent" in report.lower()


def test_informe_contiene_cabeceras_de_secuencias(hbb_normal, hbb_sickle):
    r = run(hbb_normal, hbb_sickle, mode="both")
    report = build_report(r)
    assert "HBB_normal" in report
    assert "HBB_sickle" in report


# ══════════════════════════════════════════════════════════════════════════════
# §6  CLASIFICACIÓN DE SUSTITUCIONES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("a,b", [
    ("E", "D"),   # ambos ácidos
    ("K", "R"),   # ambos básicos
    ("L", "V"),   # ambos alifáticos hidrofóbicos
    ("F", "Y"),   # ambos aromáticos
    ("S", "T"),   # ambos hidroxílicos pequeños
    ("N", "Q"),   # ambos amidas
])
def test_cambio_conservativo(a, b):
    assert _change_type(a, b) == "conservative"


@pytest.mark.parametrize("a,b", [
    ("E", "V"),   # ácido → hidrofóbico (sickle cell)
    ("G", "W"),   # tiny → aromático grande
    ("K", "D"),   # básico → ácido
    ("A", "F"),   # pequeño → aromático grande
])
def test_cambio_radical(a, b):
    assert _change_type(a, b) == "radical"


# ══════════════════════════════════════════════════════════════════════════════
# §7  GESTIÓN DE ERRORES
# ══════════════════════════════════════════════════════════════════════════════

def test_error_archivo_no_encontrado():
    """Archivo inexistente debe lanzar FileNotFoundError."""
    with pytest.raises((FileNotFoundError, OSError)):
        run("no_existe.fa", "tampoco_este.fa")


def test_error_tipos_incompatibles(hbb_normal, insulin_ref):
    """ADN vs proteína debe lanzar TypeError."""
    with pytest.raises(TypeError):
        run(hbb_normal, insulin_ref, mode="protein")


def test_error_archivo_vacio(tmp_path):
    """Archivo FASTA vacío debe lanzar ValueError."""
    vacio = str(tmp_path / "vacio.fa")
    Path(vacio).write_text("", encoding="utf-8")
    normal = str(tmp_path / "normal.fa")
    Path(normal).write_text(f">ref\n{_SEQ_NORMAL}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        run(vacio, normal)


# ══════════════════════════════════════════════════════════════════════════════
# §8  CLI — main()
# ══════════════════════════════════════════════════════════════════════════════

def test_cli_modo_dna(hbb_normal, hbb_sickle, capsys):
    """La CLI con --mode dna debe ejecutarse sin errores."""
    ret = main([hbb_normal, hbb_sickle, "--mode", "dna"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "DNA ALIGNMENT" in captured.out


def test_cli_modo_protein(hbb_normal, hbb_sickle, capsys):
    ret = main([hbb_normal, hbb_sickle, "--mode", "protein"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "PROTEIN ALIGNMENT" in captured.out


def test_cli_modo_both(hbb_normal, hbb_sickle, capsys):
    ret = main([hbb_normal, hbb_sickle, "--mode", "both"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "DNA ALIGNMENT" in captured.out
    assert "PROTEIN ALIGNMENT" in captured.out


def test_cli_guarda_archivo(hbb_normal, hbb_sickle, tmp_path):
    """Con --output, el informe debe guardarse en el archivo indicado."""
    salida = str(tmp_path / "informe.md")
    ret = main([hbb_normal, hbb_sickle, "--output", salida])
    assert ret == 0
    assert Path(salida).exists()
    contenido = Path(salida).read_text(encoding="utf-8")
    assert "MUTATION ANALYSIS REPORT" in contenido


def test_cli_error_devuelve_codigo_1(tmp_path):
    """CLI con archivo inexistente debe devolver código 1."""
    ret = main(["no_existe.fa", "tampoco.fa"])
    assert ret == 1
