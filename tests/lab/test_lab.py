"""
tests/lab/test_lab.py — herramientas de laboratorio: enzimas, ORFs y cebadores.

Estas tres son las que más se parecen a «2+2=4»: no hay estadística ni modelos,
solo hay que **acertar la posición exacta**. Y precisamente por eso el listón es
más alto: en una enzima de restricción, equivocarse en una base es cortar el gen
por el sitio equivocado y tirar un experimento de una semana.

Por eso aquí, además de los casos construidos a mano, hay contraste directo contra
los estándares del campo (Biopython/REBASE para las enzimas y la Tm), que se saltan
solos si Biopython no está instalado.
"""

from __future__ import annotations

import numpy as np
import pytest

from bioforge import (
    ENZYMES,
    design_primers,
    digest,
    find_orfs,
    find_sites,
    gc_percent,
    get_enzyme,
    longest_orf,
    pcr,
    tm_nn,
    tm_wallace,
    unique_cutters,
)
from bioforge.core.biocore import SequenceValueError

RC = str.maketrans("ACGT", "TGCA")


def _rc(s: str) -> str:
    return s.translate(RC)[::-1]


@pytest.fixture()
def relleno():
    """Secuencia aleatoria reproducible, para poner sitios dentro a propósito."""
    rng = np.random.default_rng(2026)
    return lambda n: "".join(rng.choice(list("ACGT"), size=n))


# ── enzimas de restricción ───────────────────────────────────────────────────
def test_encuentra_el_sitio_donde_se_puso(relleno):
    seq = relleno(300) + "GAATTC" + relleno(300)
    sitios = find_sites(seq, ["EcoRI"])
    assert len(sitios) == 1
    s = sitios[0]
    assert s.site_start == 300
    assert s.position == 301                    # G^AATTC: corta tras 1 base


def test_enzima_desconocida_avisa():
    with pytest.raises(SequenceValueError, match="desconocida"):
        get_enzyme("TijerasMagicasI")


def test_propiedades_de_los_extremos():
    """El saliente determina si dos fragmentos se pueden pegar: hay que acertarlo."""
    assert get_enzyme("EcoRI").ends == "cohesivo 5'"     # G^AATTC deja 4 nt salientes
    assert get_enzyme("SmaI").ends == "romo"             # CCC^GGG corta en el centro
    assert get_enzyme("PstI").ends == "cohesivo 3'"      # CTGCA^G
    assert get_enzyme("EcoRI").pretty_site() == "G^AATTC"


def test_digestion_lineal_y_circular(relleno):
    """n cortes dan n+1 fragmentos en lineal, pero exactamente n en circular."""
    seq = relleno(200) + "GAATTC" + relleno(400) + "GAATTC" + relleno(200)
    lineal = digest(seq, "EcoRI")
    assert lineal.n_cuts == 2 and len(lineal.fragments) == 3
    assert sum(lineal.sizes()) == len(seq)

    circular = digest(seq, "EcoRI", circular=True)
    assert circular.n_cuts == 2 and len(circular.fragments) == 2
    assert sum(circular.sizes()) == len(seq)     # nada se pierde ni se duplica


def test_doble_digestion(relleno):
    seq = relleno(150) + "GAATTC" + relleno(300) + "GGATCC" + relleno(150)
    d = digest(seq, ["EcoRI", "BamHI"])
    assert d.n_cuts == 2 and len(d.fragments) == 3
    assert set(d.enzymes) == {"EcoRI", "BamHI"}


def test_sin_sitios_da_un_solo_fragmento():
    seq = "AAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    d = digest(seq, "EcoRI")
    assert d.n_cuts == 0 and len(d.fragments) == 1
    assert d.fragments[0].sequence == seq


def test_codigos_ambiguos_iupac():
    """HincII reconoce GTYRAC: Y = C o T, R = A o G. Deben valer las cuatro combinaciones."""
    for y in "CT":
        for r in "AG":
            seq = "AAAAAAAAAA" + f"GT{y}{r}AC" + "TTTTTTTTTT"
            assert len(find_sites(seq, ["HincII"])) == 1, f"falló con GT{y}{r}AC"


def test_cortadores_unicos(relleno):
    seq = relleno(400) + "GCGGCCGC" + relleno(400)      # un solo sitio NotI
    unicos = unique_cutters(seq)
    assert "NotI" in unicos


def test_catalogo_es_coherente():
    """Cada enzima debe tener el corte DENTRO de su sitio y datos consistentes."""
    for nombre, e in ENZYMES.items():
        assert e.name == nombre
        assert 0 <= e.cut5 <= len(e.site), f"{nombre}: cut5 fuera del sitio"
        assert 0 <= e.cut3 <= len(e.site), f"{nombre}: cut3 fuera del sitio"
        assert set(e.site) <= set("ACGTRYSWKMBDHVN"), f"{nombre}: sitio con letras raras"


@pytest.mark.parametrize("enzima", ["EcoRI", "BamHI", "HindIII", "PstI", "SmaI",
                                    "NotI", "HinfI", "HincII", "Sau3AI", "AluI"])
def test_posiciones_identicas_a_biopython(enzima, relleno):
    """CONTRASTE: las posiciones deben coincidir EXACTO con Biopython (datos REBASE).

    No vale «parecido»: una posición mal es cortar por el sitio equivocado.
    """
    BR = pytest.importorskip("Bio.Restriction")
    from Bio.Seq import Seq

    seq = relleno(4000)
    nuestro = sorted({s.position for s in find_sites(seq, [enzima])})
    suyo = sorted(x - 1 for x in getattr(BR, enzima).search(Seq(seq), linear=True))
    assert nuestro == suyo, f"{enzima}: nuestras {len(nuestro)} vs sus {len(suyo)}"


# ── ORFs ─────────────────────────────────────────────────────────────────────
def test_encuentra_el_gen_colocado(relleno):
    gen = "ATG" + "GCA" * 40 + "TAA"
    seq = relleno(300) + gen + relleno(300)
    orfs = find_orfs(seq, min_length=90)
    encontrado = [o for o in orfs if o.start == 300 and o.strand == "+"]
    assert encontrado, f"no encontró el gen colocado en 300; halló {orfs[:3]}"
    o = encontrado[0]
    assert o.length == len(gen)
    assert o.protein.startswith("MAAA")
    assert o.has_stop is True


def test_busca_en_la_hebra_inversa(relleno):
    """Un gen en la hebra de abajo debe encontrarse igual: son SEIS marcos."""
    gen = "ATG" + "GCA" * 40 + "TAA"
    seq = relleno(300) + _rc(gen) + relleno(300)
    orfs = find_orfs(seq, min_length=90, both_strands=True)
    encontrado = [o for o in orfs if o.strand == "-" and o.start == 300]
    assert encontrado, f"no halló el gen en la hebra inversa; halló {orfs[:3]}"
    # el ORF puede ser MÁS LARGO que el gen colocado: si el relleno aleatorio trae
    # un ATG en el mismo marco sin ninguna parada en medio, el marco empieza antes.
    # Eso es correcto (el ORF va del primer ATG tras la parada anterior); lo que se
    # exige es que TERMINE exactamente en el gen que se puso.
    leido = _rc(seq[encontrado[0].start:encontrado[0].end])
    assert leido.endswith(gen), "el ORF de la hebra inversa no acaba en el gen colocado"

    # sin mirar la hebra inversa, ese ORF no puede aparecer
    solo_directa = find_orfs(seq, min_length=90, both_strands=False)
    assert all(o.strand == "+" for o in solo_directa)


def test_min_length_filtra():
    gen = "ATG" + "GCA" * 5 + "TAA"          # 21 nt, muy corto
    seq = "TTTTTTTTTT" + gen + "TTTTTTTTTT"
    assert find_orfs(seq, min_length=90) == []
    assert find_orfs(seq, min_length=15)


def test_sin_atg_da_tramos_entre_paradas():
    """``require_start=False`` devuelve el tramo entero, como ``getorf -find 0``."""
    # solo la hebra directa: al complementar, "TTATGC…" sí contiene un ATG
    seq = "TAA" + "GCA" * 40 + "TAA"
    con_atg = find_orfs(seq, min_length=60, require_start=True, both_strands=False)
    sin_atg = find_orfs(seq, min_length=60, require_start=False, both_strands=False)
    assert not con_atg                        # no hay ningún ATG en esta hebra
    assert sin_atg                            # pero sí un tramo largo sin paradas


def test_orf_truncado_se_marca():
    """Un ORF que llega al final sin parada puede continuar: hay que avisarlo."""
    seq = "ATG" + "GCA" * 50                  # sin codón de parada
    o = longest_orf(seq, min_length=90)
    assert o is not None and o.has_stop is False


def test_secuencia_vacia_es_error():
    with pytest.raises(SequenceValueError):
        find_orfs("")


def test_las_coordenadas_apuntan_al_gen_de_verdad(relleno):
    """Las coordenadas devueltas deben recortar exactamente el gen en la secuencia."""
    gen = "ATG" + "CGT" * 40 + "TGA"
    seq = relleno(250) + gen + relleno(250)
    # ojo: el relleno aleatorio puede contener ORFs MÁS LARGOS por casualidad,
    # así que se busca el que empieza donde se colocó el gen, no el mayor
    o = [x for x in find_orfs(seq, min_length=90)
         if x.strand == "+" and x.start == 250][0]
    assert seq[o.start:o.end] == gen
    assert o.length == len(gen)


# ── cebadores y PCR ──────────────────────────────────────────────────────────
def test_tm_sube_con_el_contenido_gc():
    """Más G+C, más estable la hélice, más temperatura hace falta para separarla."""
    assert tm_nn("ATATATATATATATAT") < tm_nn("ACGTACGTACGTACGT") < tm_nn("GCGCGCGCGCGCGCGC")


def test_tm_depende_del_ORDEN_no_solo_de_la_composicion():
    """Es la razón de usar vecino más próximo: dos secuencias con las MISMAS bases
    en distinto orden tienen Tm distinta. La regla casera no lo capta."""
    a, b = "GGGGCCCCGGGGCCCC", "GCGCGCGCGCGCGCGC"
    assert gc_percent(a) == gc_percent(b) == 100.0
    assert tm_wallace(a) == tm_wallace(b)         # la regla casera no las distingue
    assert abs(tm_nn(a) - tm_nn(b)) > 0.5, "el vecino más próximo SÍ debe distinguirlas"


def test_tm_rechaza_bases_raras():
    with pytest.raises(SequenceValueError, match="A, C, G y T"):
        tm_nn("ACGTNNNACGT")
    with pytest.raises(SequenceValueError, match="al menos 2"):
        tm_nn("A")


@pytest.mark.parametrize("cebador", [
    "ATGCGCATGCGCATGCGCAT",            # autocomplementaria
    "GGGGCCCCGGGGCCCC",                # autocomplementaria
    "TATCGGCTACCGCAAAAATAGTACC",
    "TTTTTTTTTTTTTTTT",
    "ACGTTGCATGCAAGCTTGGC",
])
def test_tm_identica_a_biopython(cebador):
    """CONTRASTE: misma Tm que Biopython, a precisión de máquina.

    Ojo con el matiz que destapó el contraste: Biopython **no detecta** si la
    secuencia es autocomplementaria — hay que decírselo con ``selfcomp=True``. El
    nuestro lo detecta solo, que es más seguro: un cebador palindrómico al que se
    le olvide el parámetro sale con la Tm equivocada sin ningún aviso.
    """
    mt = pytest.importorskip("Bio.SeqUtils.MeltingTemp")
    auto = cebador == _rc(cebador)
    suyo = mt.Tm_NN(cebador, nn_table=mt.DNA_NN3, dnac1=25, dnac2=25,
                    Na=50, saltcorr=5, selfcomp=auto)
    assert tm_nn(cebador) == pytest.approx(suyo, abs=1e-9)


def test_disena_cebadores_utilizables(relleno):
    seq = relleno(600)
    par = design_primers(seq)
    assert par is not None
    d, i = par
    assert d.strand == "+" and i.strand == "-"
    assert 18 <= d.length <= 27 and 18 <= i.length <= 27
    assert 45 <= d.tm <= 75 and 45 <= i.tm <= 75


def test_secuencia_corta_no_da_cebadores():
    assert design_primers("ACGTACGTACGT") is None


def test_pcr_amplifica_lo_esperado(relleno):
    """La prueba de la PCR: el producto debe ser exactamente el trozo esperado."""
    seq = relleno(1000)
    directo = seq[100:120]
    inverso = _rc(seq[400:420])
    productos = pcr(seq, directo, inverso)
    assert len(productos) == 1
    p = productos[0]
    assert p.start == 100 and p.end == 420
    assert p.length == 320
    assert p.sequence == seq[100:420]


def test_pcr_sin_sitio_no_amplifica(relleno):
    seq = relleno(800)
    assert pcr(seq, "GGGGGGGGGGGGGGGGGGGG", "CCCCCCCCCCCCCCCCCCCC") == []


def test_pcr_avisa_de_productos_multiples(relleno):
    """Si el cebador pega en dos sitios salen bandas inesperadas: hay que verlo."""
    repetido = "ACGTTGCATGCAAGCT"
    seq = (relleno(200) + repetido + relleno(300) + repetido + relleno(200))
    inverso_seq = _rc(seq[-40:-20])            # el inverso, cerca del final
    productos = pcr(seq, repetido, inverso_seq)
    assert len(productos) >= 2, "debería detectar que el directo pega en dos sitios"


def test_pcr_exige_los_dos_cebadores():
    with pytest.raises(SequenceValueError):
        pcr("ACGTACGTACGT", "", "ACGT")
