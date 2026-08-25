"""
tests/test_app_backend.py — el PUENTE de la app (bioforge/app/backend.py).

La ventana gráfica no se puede probar sin pantalla, pero TODA la lógica que la
interfaz invoca sí: son métodos que reciben tipos simples y devuelven diccionarios.
Aquí se prueban enteros, sin abrir ninguna ventana — que es justo el diseño.
"""

import os

import pytest

from bioforge.app.backend import Api


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


def _protein_series_fasta(tmp_path, n_time=24):
    """Serie temporal de una proteína con un aminoácido que sube en un sitio."""
    import numpy as np
    rng = np.random.default_rng(5)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    base = list("".join(rng.choice(list(aa), size=40)))
    lines = []
    for t in range(n_time):
        for rep in range(3):
            s = base.copy()
            if rng.random() < t / (n_time - 1):
                s[15] = "K"                          # el alelo nuevo sube con el tiempo
            lines.append(f">p_t{t:02d}_{rep}\n{''.join(s)}")
    p = tmp_path / "evo.fasta"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


def test_evolucion_predict_mutations(tmp_path):
    api = Api(); api.open_file(_protein_series_fasta(tmp_path))
    r = api.predict_mutations(10)
    assert r["n_sequences"] == 72
    assert len(r["mutations"]) == 10
    assert all("site" in m and "score" in m for m in r["mutations"])


def test_evolucion_check_mutation(tmp_path):
    api = Api(); api.open_file(_protein_series_fasta(tmp_path))
    v = api.check_mutation("K16E")                   # sitio 16 (1-based), a E
    assert v["tier"] in ("OBSERVADO", "ESTIMADO")
    assert "label" in v


def test_evolucion_exige_proteina(fasta):
    api = Api(); api.open_file(fasta)                # FASTA de ADN
    assert "error" in api.predict_mutations()        # evolución es sobre proteína


def test_open_example_carga_y_es_usable():
    """El botón 'Probar con un ejemplo' carga un FASTA en memoria, listo para todo."""
    api = Api()
    ws = api.open_example()
    assert ws["n_files"] == 1 and ws["active"] == 0
    s = api.summary()
    assert s["loaded"] and s["count"] == 3 and s["nucleotide"] == 3
    assert s["filename"] == "ejemplo_adn.fasta"
    # el ejemplo está pensado para lucir la app: se traduce y se alinea con una mutación
    assert api.translate(0)["protein"].startswith("MARK")
    al = api.align(0, 1)                              # gen A vs su variante
    assert al["n_mutations"] == 1 and al["identity"] > 98.0


def test_ram_plana_con_muchos_archivos(tmp_path):
    """Solo el archivo ACTIVO vive en RAM: abrir muchos no debe acumularlos.

    Es la promesa Edge de la app (v10.1): con 500 archivos abiertos la memoria no
    crece con ellos, porque de los inactivos se guarda solo su ficha y se releen
    del disco al volver.
    """
    paths = []
    for i in range(8):
        p = tmp_path / f"g{i}.fasta"
        p.write_text("".join(f">s{j}\n{'ACGT' * 60}\n" for j in range(20)),
                     encoding="utf-8")
        paths.append(str(p))

    api = Api()
    for p in paths:
        api.open_file(p)

    # exactamente UNO materializado, aunque haya 8 abiertos
    vivos = sum(1 for ds in api.datasets if ds["records"] is not None)
    assert vivos == 1 and len(api.datasets) == 8

    # y las pestañas se listan sin necesidad de cargar los archivos
    ws = api.workspace()
    assert ws["n_files"] == 8 and all(f["count"] == 20 for f in ws["files"])
    assert sum(1 for ds in api.datasets if ds["records"] is not None) == 1

    # volver a uno soltado lo relee y sigue funcionando igual
    s = api.select_file(2)
    assert s["count"] == 20 and s["filename"] == "g2.fasta"
    assert api.records_page(0, 3)["items"][0]["length"] == 240
    assert sum(1 for ds in api.datasets if ds["records"] is not None) == 1


def test_secuencia_en_memoria_no_se_suelta(fasta):
    """Un basecall añadido NO tiene archivo en disco: no debe soltarse nunca."""
    api = Api()
    api.add_sequence("basecall_x", "ATGAAAGGGTTTCCCTAA")   # sin path
    api.open_file(fasta)                                    # otro archivo pasa a activo
    en_memoria = api.datasets[0]
    assert en_memoria["path"] == ""
    assert en_memoria["records"] is not None                # sigue vivo: no hay de dónde releerlo
    assert api.select_file(0)["count"] == 1                 # y se puede volver a él


def test_ping():
    assert Api().ping()["ok"] is True


# ── variantes en la app: la tubería completa detrás de un botón ──────────────
@pytest.fixture()
def genoma_y_lecturas(tmp_path):
    """Un genoma de referencia y unas lecturas con 2 mutaciones conocidas."""
    import numpy as np
    rng = np.random.default_rng(31)
    L = 2500
    ref = "".join(rng.choice(list("ACGT"), size=L))
    verdad = {}
    muestra = list(ref)
    for p in (800, 1600):
        nueva = "A" if ref[p] != "A" else "G"
        muestra[p] = nueva
        verdad[p + 1] = nueva                        # 1-based, como el VCF
    muestra = "".join(muestra)

    rp = tmp_path / "genoma.fasta"
    rp.write_text(f">contig_prueba genoma\n{ref}\n", encoding="utf-8")

    comp = str.maketrans("ACGT", "TGCA")
    lp = tmp_path / "lecturas.fastq"
    with open(lp, "w", encoding="utf-8") as fh:
        for i in range(250):
            s = int(rng.integers(0, L - 200))
            r = list(muestra[s:s + 200])
            for j in range(len(r)):
                if rng.random() < 0.01:              # 1 % de error
                    r[j] = rng.choice(list("ACGT"))
            seq = "".join(r)
            if rng.random() < 0.5:
                seq = seq.translate(comp)[::-1]
            fh.write(f"@lectura_{i}\n{seq}\n+\n{'I' * len(seq)}\n")
    return str(rp), str(lp), verdad


def test_variant_sources_no_materializa_nada(genoma_y_lecturas):
    """Listar las opciones se apoya en las fichas: nunca debe cargar los archivos."""
    ref_path, reads_path, _ = genoma_y_lecturas
    api = Api()
    api.open_file(ref_path)
    api.open_file(reads_path)                        # este queda activo

    src = api.variant_sources()
    assert len(src["files"]) == 2
    # el archivo NO activo sigue sin materializar (RAM plana, regla nº10)
    assert api.datasets[0]["records"] is None

    ref_f = next(f for f in src["files"] if f["filename"] == "genoma.fasta")
    reads_f = next(f for f in src["files"] if f["filename"] == "lecturas.fastq")
    assert reads_f["looks_like_reads"] is True       # heurística de los desplegables
    assert ref_f["looks_like_reads"] is False


def test_reference_options_lista_las_secuencias(genoma_y_lecturas):
    ref_path, reads_path, _ = genoma_y_lecturas
    api = Api()
    api.open_file(ref_path)
    api.open_file(reads_path)
    o = api.reference_options(0)
    assert o["filename"] == "genoma.fasta"
    assert len(o["options"]) == 1
    assert o["options"][0]["length"] == 2500


def test_call_variants_app_encuentra_las_mutaciones(genoma_y_lecturas):
    """Extremo a extremo desde la app: encuentra las 2 reales y ninguna falsa."""
    ref_path, reads_path, verdad = genoma_y_lecturas
    api = Api()
    api.open_file(ref_path)
    api.open_file(reads_path)

    r = api.call_variants_app(ref_file=0, ref_index=0, reads_file=1)
    assert "error" not in r, r.get("error")
    assert r["reference"] == "contig_prueba"
    assert r["n_mapped"] > 200
    assert r["mean_depth"] > 5
    assert len(r["depth_series"]) > 0                # datos para el gráfico
    assert r["coverage"]["1"] > 90

    encontradas = {(v["pos"], v["alt"]) for v in r["variants"] if v["kind"] == "SNV"}
    assert encontradas == set(verdad.items()), (
        f"esperaba {set(verdad.items())}, encontró {encontradas}")


def test_call_variants_app_rechaza_proteina(tmp_path):
    """Pedir variantes con una proteína de referencia da un error amable, no un crash."""
    p = tmp_path / "prot.fasta"
    p.write_text(">p1 proteina\nMKGFPWYEQLLIPMKGFPWYEQLLIP\n", encoding="utf-8")
    q = tmp_path / "reads.fastq"
    q.write_text("@r1\nACGTACGTAC\n+\nIIIIIIIIII\n", encoding="utf-8")
    api = Api()
    api.open_file(str(p))
    api.open_file(str(q))
    r = api.call_variants_app(ref_file=0, ref_index=0, reads_file=1)
    assert "error" in r and "ADN" in r["error"]


def test_call_variants_app_indices_invalidos(genoma_y_lecturas):
    ref_path, reads_path, _ = genoma_y_lecturas
    api = Api()
    api.open_file(ref_path)
    api.open_file(reads_path)
    assert "error" in api.call_variants_app(ref_file=9, ref_index=0, reads_file=1)
    assert "error" in api.call_variants_app(ref_file=0, ref_index=9, reads_file=1)


def test_vcf_text_antes_y_despues(genoma_y_lecturas):
    """El VCF solo existe tras analizar; es lo que guarda el botón de la interfaz."""
    ref_path, reads_path, _ = genoma_y_lecturas
    api = Api()
    assert "error" in api.vcf_text()                 # todavía no se ha analizado

    api.open_file(ref_path)
    api.open_file(reads_path)
    api.call_variants_app(ref_file=0, ref_index=0, reads_file=1)

    texto = api.vcf_text()["vcf"]
    assert texto.startswith("##fileformat=VCFv4.2")
    assert "##contig=<ID=contig_prueba,length=2500>" in texto
    assert "#CHROM\tPOS\tID\tREF\tALT" in texto
    datos = [x for x in texto.strip().split("\n") if not x.startswith("#")]
    assert len(datos) >= 2 and all(len(x.split("\t")) == 8 for x in datos)
