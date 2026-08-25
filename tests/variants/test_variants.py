"""
tests/variants/test_variants.py — llamada de variantes (pileup + caller).

Se prueban dos niveles:

* **Unitario** — construyendo los ``Mapping`` a mano, para aislar el llamador del
  mapeador y que los fallos señalen a un único culpable.
* **Extremo a extremo** — genoma → lecturas simuladas → mapeo real → VCF, que es
  lo único que demuestra que la tubería entera encaja.

Lo que más importa aquí no es encontrar la mutación (eso es fácil con 40× de
profundidad), sino **no inventarse ninguna**: un llamador que grita a la mínima es
peor que no tener llamador.
"""

from __future__ import annotations

import numpy as np
import pytest

from bioforge import (
    GenomeAligner,
    Mapping,
    Variant,
    call_variants,
    pileup,
    write_vcf,
)
from bioforge.core.biocore import SequenceValueError

RC = str.maketrans("ACGT", "TGCA")


def _rc(s: str) -> str:
    return s.translate(RC)[::-1]


def _mk(read: str, ref_start: int, cigar: str, *, strand: str = "+",
        mapq: int = 60, ref_len: int = 1000) -> Mapping:
    """Un Mapping sintético: evita depender del mapeador en los tests unitarios."""
    return Mapping(
        query_len=len(read), query_start=0, query_end=len(read), strand=strand,
        target_len=ref_len, target_start=ref_start, target_end=ref_start + len(read),
        num_matches=len(read), block_len=len(read), mapq=mapq, identity=1.0,
        chain_score=100.0, cigar=cigar, target_name="ref",
    )


@pytest.fixture()
def ref() -> str:
    """Referencia reproducible de 1000 pb."""
    rng = np.random.default_rng(1234)
    return "".join(rng.choice(list("ACGT"), size=1000))


# ── pileup: la evidencia ─────────────────────────────────────────────────────
def test_pileup_cuenta_las_bases(ref):
    """Cada lectura suma 1 a la base que vio en cada posición."""
    trozo = ref[100:150]
    pares = [(trozo, _mk(trozo, 100, "50M")) for _ in range(10)]
    pile = pileup(ref, pares, contig="ref")

    assert pile.n_reads == 10
    assert pile.counts.shape == (1000, 6)
    assert pile.depth[100] == 10 and pile.depth[149] == 10
    assert pile.depth[99] == 0 and pile.depth[150] == 0   # fuera, nadie cubre


def test_pileup_profundidad_y_cobertura(ref):
    """Las utilidades de cobertura responden '¿he leído bastante?'."""
    trozo = ref[0:100]
    pile = pileup(ref, [(trozo, _mk(trozo, 0, "100M"))] * 4, contig="ref")

    assert pile.covered(min_depth=1) == pytest.approx(0.1)   # 100 de 1000 pb
    assert pile.covered(min_depth=5) == 0.0                  # solo hay 4×
    assert pile.mean_depth == pytest.approx(0.4)


def test_pileup_hebra_inversa_se_orienta(ref):
    """Una lectura en hebra '-' se complementa antes de apilar; si no, todo sería ruido."""
    trozo = ref[200:260]
    inv = _rc(trozo)
    pile = pileup(ref, [(inv, _mk(inv, 200, "60M", strand="-"))] * 6, contig="ref")

    from bioforge.variants.pileup import _CODE
    esperado = _CODE[np.frombuffer(ref[200:260].encode(), dtype=np.uint8)]
    observado = pile.counts[200:260, :4].argmax(axis=1)
    assert np.array_equal(observado, esperado), "la hebra inversa no se orientó bien"


def test_pileup_filtra_por_mapq(ref):
    """Los mapeos dudosos se descartan: apilarlos contamina la evidencia."""
    trozo = ref[300:340]
    pares = ([(trozo, _mk(trozo, 300, "40M", mapq=60))] * 3
             + [(trozo, _mk(trozo, 300, "40M", mapq=2))] * 5)
    pile = pileup(ref, pares, contig="ref", min_mapq=30)
    assert pile.n_reads == 3 and pile.n_skipped == 5


def test_pileup_no_filtra_por_defecto_regresion(ref):
    """REGRESIÓN: el contig por defecto no debe descartar todas las lecturas.

    El defecto era ``contig="ref"``, que filtraba SIEMPRE por ``target_name``. Si el
    mapeador etiquetaba el contig con otro nombre (p.ej. el del FASTA), se
    descartaba el 100 % de las lecturas EN SILENCIO: la profundidad salía 0× sin
    ninguna pista de por qué. Lo destapó integrar la CLI. Ahora el defecto es
    ``None`` = no filtrar y tomar el nombre del primer mapeo.
    """
    trozo = ref[100:160]
    mp = _mk(trozo, 100, "60M")._replace(target_name="cromosoma_raro")
    pile = pileup(ref, [(trozo, mp)] * 8)          # sin pasar contig

    assert pile.n_reads == 8 and pile.n_skipped == 0
    assert pile.contig == "cromosoma_raro"          # hereda el nombre del mapeo
    assert pile.depth[100] == 8


def test_pileup_filtra_si_se_pide_contig_explicito(ref):
    """Con varios contigs sí hace falta filtrar, y entonces se pide explícitamente."""
    trozo = ref[100:160]
    a = _mk(trozo, 100, "60M")._replace(target_name="chr1")
    b = _mk(trozo, 100, "60M")._replace(target_name="chr2")
    pile = pileup(ref, [(trozo, a)] * 5 + [(trozo, b)] * 3, contig="chr1")

    assert pile.n_reads == 5 and pile.n_skipped == 3
    assert pile.contig == "chr1"


def test_pileup_rechaza_longitud_invalida():
    with pytest.raises(SequenceValueError):
        pileup(0, [], contig="ref")


# ── el llamador: SNVs ────────────────────────────────────────────────────────
def _lecturas_con_snv(ref: str, pos: int, nueva: str, n: int, inicio: int = 400,
                      largo: int = 120):
    """n lecturas idénticas que cubren `pos` y llevan `nueva` en esa posición."""
    trozo = list(ref[inicio:inicio + largo])
    trozo[pos - inicio] = nueva
    lectura = "".join(trozo)
    return [(lectura, _mk(lectura, inicio, f"{largo}M")) for _ in range(n)]


def test_llama_la_sustitucion(ref):
    """El caso base: 20 lecturas de acuerdo en un cambio → una variante."""
    pos = 450
    nueva = "A" if ref[pos] != "A" else "G"
    pile = pileup(ref, _lecturas_con_snv(ref, pos, nueva, 20), contig="ref")
    vs = call_variants(pile, ref)

    assert len(vs) == 1
    v = vs[0]
    assert v.pos == pos + 1                      # el VCF es 1-based
    assert v.ref == ref[pos] and v.alt == nueva
    assert v.kind == "SNV" and v.af == pytest.approx(1.0)
    assert v.depth == 20 and v.qual > 100


def test_no_inventa_variantes_sin_mutacion(ref):
    """Lecturas idénticas a la referencia → CERO llamadas. Lo más importante."""
    trozo = ref[400:520]
    pile = pileup(ref, [(trozo, _mk(trozo, 400, "120M"))] * 30, contig="ref")
    assert call_variants(pile, ref) == []


def test_una_lectura_discrepante_no_basta(ref):
    """Un solo error de secuenciación no puede producir una variante."""
    pos = 450
    nueva = "A" if ref[pos] != "A" else "G"
    pares = (_lecturas_con_snv(ref, pos, nueva, 1)
             + [(ref[400:520], _mk(ref[400:520], 400, "120M"))] * 29)
    pile = pileup(ref, pares, contig="ref")
    assert call_variants(pile, ref) == []


def test_profundidad_insuficiente_se_calla(ref):
    """Con poca evidencia, callar es lo honesto."""
    pos = 450
    nueva = "A" if ref[pos] != "A" else "G"
    pile = pileup(ref, _lecturas_con_snv(ref, pos, nueva, 3), contig="ref")
    assert call_variants(pile, ref, min_depth=10) == []
    assert len(call_variants(pile, ref, min_depth=2)) == 1      # bajando el listón, sí


def test_variante_minoritaria_depende_de_min_af(ref):
    """Una variante al 25 % aparece o no según el umbral que pidas."""
    pos, nueva = 450, ("A" if ref[450] != "A" else "G")
    ok = ref[400:520]
    pares = _lecturas_con_snv(ref, pos, nueva, 10) + [(ok, _mk(ok, 400, "120M"))] * 30
    pile = pileup(ref, pares, contig="ref")

    assert call_variants(pile, ref, min_af=0.5) == []           # 0.25 < 0.5
    vs = call_variants(pile, ref, min_af=0.1)
    assert len(vs) == 1 and vs[0].af == pytest.approx(0.25)


def test_qual_crece_con_la_evidencia(ref):
    """Más lecturas de acuerdo ⇒ más confianza. La razón de verosimilitudes lo refleja."""
    pos, nueva = 450, ("A" if ref[450] != "A" else "G")
    quals = []
    for n in (5, 20, 60):
        pile = pileup(ref, _lecturas_con_snv(ref, pos, nueva, n), contig="ref")
        quals.append(call_variants(pile, ref, min_depth=2)[0].qual)
    assert quals[0] < quals[1] < quals[2]


# ── indels ───────────────────────────────────────────────────────────────────
def test_llama_una_delecion(ref):
    """Deleción de 3 pb: se representa con la base ancla previa, como manda el VCF."""
    ini, largo = 500, 3
    lectura = ref[450:ini] + ref[ini + largo:550 + largo]
    mp = _mk(lectura, 450, f"{ini - 450}M{largo}D{100 - (ini - 450)}M")
    pile = pileup(ref, [(lectura, mp)] * 20, contig="ref")
    vs = [v for v in call_variants(pile, ref) if v.kind == "DEL"]

    assert len(vs) == 1
    v = vs[0]
    assert v.pos == ini                          # 1-based de la base ancla
    assert v.ref == ref[ini - 1:ini + largo] and v.alt == ref[ini - 1]
    assert len(v.ref) - len(v.alt) == largo


def test_llama_una_insercion(ref):
    """Inserción: REF es la base ancla, ALT la base ancla + lo insertado."""
    punto, insertado = 600, "GGTT"
    lectura = ref[550:punto] + insertado + ref[punto:650]
    mp = _mk(lectura, 550, f"{punto - 550}M{len(insertado)}I{650 - punto}M")
    pile = pileup(ref, [(lectura, mp)] * 20, contig="ref")
    vs = [v for v in call_variants(pile, ref) if v.kind == "INS"]

    assert len(vs) == 1
    v = vs[0]
    assert v.alt[1:] == insertado or len(v.alt) - len(v.ref) == len(insertado)
    assert v.ref == ref[v.pos - 1]


def test_indels_desactivables(ref):
    """``indels=False`` deja solo sustituciones."""
    ini, largo = 500, 3
    lectura = ref[450:ini] + ref[ini + largo:550 + largo]
    mp = _mk(lectura, 450, f"{ini - 450}M{largo}D{100 - (ini - 450)}M")
    pile = pileup(ref, [(lectura, mp)] * 20, contig="ref")
    assert all(v.kind == "SNV" for v in call_variants(pile, ref, indels=False))


def test_normalizacion_a_la_izquierda_unifica_indels():
    """La misma deleción escrita en dos sitios de un homopolímero es UNA variante.

    Sin normalizar saldrían dos llamadas distintas; con la referencia disponible,
    ambas se empujan a la izquierda y coinciden (es lo que hace ``bcftools norm``).
    """
    from bioforge.variants.pileup import _izquierda_del, _izquierda_ins

    ref = "CCCC" + "AAAAAAAA" + "GGGG"          # homopolímero de 8 A en [4, 12)
    # borrar la 1ª, la 4ª o la última A son la misma variante → todas a la izquierda
    assert _izquierda_del(ref, 4, 1) == 4
    assert _izquierda_del(ref, 7, 1) == 4
    assert _izquierda_del(ref, 11, 1) == 4
    # insertar una A dentro del homopolímero: misma idea, y la secuencia rota
    assert _izquierda_ins(ref, 9, "A") == (4, "A")


# ── VCF ──────────────────────────────────────────────────────────────────────
def test_vcf_tiene_forma_valida(ref):
    pos, nueva = 450, ("A" if ref[450] != "A" else "G")
    pile = pileup(ref, _lecturas_con_snv(ref, pos, nueva, 20), contig="ref")
    texto = write_vcf(call_variants(pile, ref), contigs=[("ref", len(ref))])

    lineas = texto.strip().split("\n")
    assert lineas[0] == "##fileformat=VCFv4.2"
    assert any(x.startswith("##contig=<ID=ref,length=1000") for x in lineas)
    cabecera = [x for x in lineas if x.startswith("#CHROM")][0]
    assert cabecera.split("\t")[:8] == ["#CHROM", "POS", "ID", "REF", "ALT",
                                        "QUAL", "FILTER", "INFO"]
    datos = [x for x in lineas if not x.startswith("#")]
    assert len(datos) == 1
    campos = datos[0].split("\t")
    assert len(campos) == 8 and campos[0] == "ref" and campos[1] == str(pos + 1)
    assert "DP=20" in campos[7] and "TYPE=SNV" in campos[7]


def test_variant_to_vcf_es_una_linea():
    v = Variant("chr1", 42, "A", "T", 99.5, 30, 28, 0.9333, "SNV")
    campos = v.to_vcf().split("\t")
    assert campos[0] == "chr1" and campos[1] == "42"
    assert campos[3] == "A" and campos[4] == "T" and campos[5] == "99.5"


# ── errores de uso ───────────────────────────────────────────────────────────
def test_referencia_corta_es_error(ref):
    pile = pileup(ref, [], contig="ref")
    with pytest.raises(SequenceValueError, match="más corta"):
        call_variants(pile, ref[:100])


def test_error_rate_invalido(ref):
    pile = pileup(ref, [], contig="ref")
    for malo in (0.0, 1.0, -0.1, 2.0):
        with pytest.raises(SequenceValueError):
            call_variants(pile, ref, error_rate=malo)


def test_pileup_vacio_no_llama_nada(ref):
    assert call_variants(pileup(ref, [], contig="ref"), ref) == []


# ── extremo a extremo: la tubería entera ─────────────────────────────────────
def test_tuberia_completa_encuentra_la_mutacion_sin_falsos_positivos():
    """Genoma → lecturas con 1 % de error → mapeo real → VCF.

    Es la prueba que justifica la herramienta: con ~1200 bases erróneas repartidas,
    debe encontrar la única mutación real y no inventarse ninguna.
    """
    rng = np.random.default_rng(7)
    L = 2000
    ref = "".join(rng.choice(list("ACGT"), size=L))
    pos = 1000
    nueva = "A" if ref[pos] != "A" else "G"
    muestra = ref[:pos] + nueva + ref[pos + 1:]

    lecturas = []
    for _ in range(200):
        s = int(rng.integers(0, L - 250))
        r = list(muestra[s:s + 250])
        for i in range(len(r)):
            if rng.random() < 0.01:
                r[i] = rng.choice(list("ACGT"))
        seq = "".join(r)
        lecturas.append(_rc(seq) if rng.random() < 0.5 else seq)

    ga = GenomeAligner(ref)
    pares = [(r, m[0]) for r in lecturas for m in [ga.map(r)] if m]
    assert len(pares) > 150, "el mapeador debería colocar casi todas las lecturas"

    pile = pileup(ref, pares, contig="ref")
    assert pile.mean_depth > 10

    vs = call_variants(pile, ref)
    acertadas = [v for v in vs if v.pos == pos + 1 and v.alt == nueva]
    assert len(acertadas) == 1, f"no encontró la mutación real; llamó {vs}"
    assert [v for v in vs if v.kind == "SNV"] == acertadas, (
        f"inventó sustituciones falsas: {[v for v in vs if v not in acertadas]}")
