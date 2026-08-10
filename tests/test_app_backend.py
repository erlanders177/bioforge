"""
tests/test_app_backend.py — el PUENTE de la app (app/backend.py).

La ventana gráfica no se puede probar sin pantalla, pero TODA la lógica que la
interfaz invoca sí: son métodos que reciben tipos simples y devuelven diccionarios.
Aquí se prueban enteros, sin abrir ninguna ventana — que es justo el diseño.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from backend import Api  # noqa: E402


@pytest.fixture()
def fasta(tmp_path):
    p = tmp_path / "muestra.fasta"
    p.write_text(
        ">seq1 referencia\nATGAAAGGGTTTCCCATGAAAGGGTTTCCC\n"
        ">seq2 variante\nATGAAAGCGTTTCCCATGAAAGGGTTTCCC\n"
        ">prot1 proteina\nMKGFPMKGFPWYEQ\n",
        encoding="utf-8")
    return str(p)


def test_open_file_resume_el_conjunto(fasta):
    api = Api()
    ws = api.open_file(fasta)                         # ahora devuelve el workspace
    assert ws["n_files"] == 1 and ws["active"] == 0
    s = api.summary()
    assert s["loaded"] and s["count"] == 3
    assert s["nucleotide"] == 2 and s["protein"] == 1
    assert s["filename"] == "muestra.fasta"


def test_multiples_archivos_abiertos(fasta, tmp_path):
    """Se pueden tener VARIOS genomas abiertos a la vez y cambiar entre ellos."""
    p2 = tmp_path / "otro.fasta"
    p2.write_text(">x descripcion\nACGTACGTACGT\n", encoding="utf-8")
    api = Api()
    api.open_file(fasta)
    ws = api.open_file(str(p2))
    assert ws["n_files"] == 2 and ws["active"] == 1   # el recién abierto queda activo
    assert api.summary()["filename"] == "otro.fasta"
    # volver al primero por su índice
    s = api.select_file(0)
    assert s["filename"] == "muestra.fasta" and s["count"] == 3
    # cerrar el archivo 0 → queda solo el otro, y sigue accesible
    ws2 = api.close_file(0)
    assert ws2["n_files"] == 1
    assert api.summary()["filename"] == "otro.fasta"


def test_sin_archivo_activo_es_error():
    api = Api()
    assert "error" in api.records_page()              # nada abierto → dato, no crash
    assert api.summary()["loaded"] is False


def test_archivo_inexistente_devuelve_error_no_crash():
    api = Api()
    r = api.open_file("no_existe_12345.fasta")
    assert "error" in r                              # dato, no excepción


def test_records_page_pagina(fasta):
    api = Api(); api.open_file(fasta)
    page = api.records_page(0, 2)
    assert page["total"] == 3 and len(page["items"]) == 2
    assert page["items"][0]["type"] == "ADN"
    assert page["items"][0]["length"] == 30


def test_translate_adn_a_proteina(fasta):
    api = Api(); api.open_file(fasta)
    r = api.translate(0)
    assert r["protein"].startswith("MKGFP")          # ATG AAA GGG TTT CCC → M K G F P


def test_translate_proteina_da_error_amable(fasta):
    api = Api(); api.open_file(fasta)
    r = api.translate(2)                              # la #2 ya es proteína
    assert "error" in r


def test_align_detecta_la_mutacion(fasta):
    api = Api(); api.open_file(fasta)
    r = api.align(0, 1)                               # seq1 vs seq2: una sustitución
    assert r["n_mutations"] >= 1
    assert r["identity"] > 90.0
    assert any(m["kind"] == "substitution" for m in r["mutations"])


def test_sequence_detail(fasta):
    api = Api(); api.open_file(fasta)
    d = api.sequence_detail(0)
    assert d["length"] == 30 and "A" in d["composition"]


def test_indice_fuera_de_rango_es_error(fasta):
    api = Api(); api.open_file(fasta)
    assert "error" in api.sequence_detail(99)


def test_open_fastq(tmp_path):
    """Regresión: la app dice cargar FASTA *y FASTQ*; el FASTQ debe cargar de verdad."""
    p = tmp_path / "reads.fastq"
    p.write_text(
        "@r1 lectura\nACGTACGTAC\n+\nIIIIIIIIII\n"
        "@r2 lectura\nTTGGCCAATT\n+\nIIIIFFFFII\n",
        encoding="utf-8")
    api = Api()
    api.open_file(str(p))
    s = api.summary()
    assert s["loaded"] and s["count"] == 2
    assert s["has_quality"] is True                  # FASTQ trae calidades
    assert api.records_page(0, 5)["items"][0]["length"] == 10


def test_qc_report_fastq(tmp_path):
    """El informe de calidad devuelve métricas y series para los gráficos."""
    p = tmp_path / "reads.fastq"
    p.write_text("".join(
        f"@r{i}\nACGTACGTAC\n+\nIIIIFFFF,,\n" for i in range(20)), encoding="utf-8")
    api = Api(); api.open_file(str(p))
    q = api.qc_report()
    assert q["n_reads"] == 20
    assert 0 <= q["mean_q"] <= 60
    assert len(q["pos_q_mean"]) == 10                # una calidad por posición
    assert len(q["meanq_hist"]) >= 40 and len(q["gc_hist"]) == 101
    assert len(q["base_frac"]) == 5                  # A/C/G/T/N por posición


def test_qc_report_fasta_da_error_amable(fasta):
    api = Api(); api.open_file(fasta)                # FASTA no tiene calidades
    assert "error" in api.qc_report()


def test_nanoporo_open_signal_y_basecall(tmp_path):
    """Cargar una señal FAST5 y basecallearla dentro de la app (extremo a extremo)."""
    h5py = pytest.importorskip("h5py")
    import numpy as np
    sig = (np.sin(np.arange(9000) / 12) * 90 + 300).astype(np.int16)
    p = tmp_path / "s.fast5"
    with h5py.File(str(p), "w") as f:
        ch = f.create_group("UniqueGlobalKey/channel_id")
        ch.attrs["digitisation"] = 8192.0
        ch.attrs["offset"] = 26.0
        ch.attrs["range"] = 1444.86
        ch.attrs["sampling_rate"] = 4000.0
        rd = f.create_group("Raw/Reads/Read_1")
        rd.attrs["read_id"] = b"read-nano-0001"
        rd.create_dataset("Signal", data=sig)

    api = Api()
    r = api.open_signal(str(p))
    assert r["n"] == 1 and r["reads"][0]["samples"] == 9000
    bc = api.basecall_read(0)
    assert bc["n_bases"] > 0 and isinstance(bc["bases"], str)
    assert len(bc["signal"]) > 0                     # puntos para el gráfico


def test_nanoporo_formato_no_valido():
    api = Api()
    # extensión no reconocida → error amable (creamos un archivo cualquiera)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as t:
        t.write(b"no soy senal")
        name = t.name
    try:
        assert "error" in api.open_signal(name)
    finally:
        os.unlink(name)


def test_add_sequence_usable_en_otras_partes(fasta):
    """Un basecall (o cualquier secuencia) se añade como genoma y sirve en todo."""
    api = Api()
    api.open_file(fasta)                              # ya hay 1 archivo (3 secuencias)
    ws = api.add_sequence("basecall_test", "ATGAAAGGGTTTCCCTAA")
    assert ws["n_files"] == 2 and ws["active"] == 1   # se añadió y quedó activo
    # ahora es un genoma de primera: se lista, se traduce, se alinea
    assert api.records_page(0, 5)["items"][0]["header"] == "basecall_test"
    assert api.translate(0)["protein"].startswith("MKG")
    # se puede alinear contra sí mismo (identidad 100)
    assert api.align(0, 0)["identity"] == 100.0


def test_ping():
    assert Api().ping()["ok"] is True
