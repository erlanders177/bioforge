"""
tests/test_app_backend.py — el PUENTE de la app (app/backend.py).

La ventana gráfica no se puede probar sin pantalla, pero TODA la lógica que la
interfaz invoca sí: son métodos que reciben tipos simples y devuelven diccionarios.
Aquí se prueban enteros, sin abrir ninguna ventana — que es justo el diseño.
"""

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
    s = api.open_file(fasta)
    assert s["loaded"] and s["count"] == 3
    assert s["nucleotide"] == 2 and s["protein"] == 1
    assert s["filename"] == "muestra.fasta"


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


def test_ping():
    assert Api().ping()["ok"] is True
