"""
tests/phylo/test_phylo.py — filogenia: distancias, árboles y soporte.

Un árbol siempre "sale", aunque los datos sean basura: dale cuatro secuencias al
azar y te dibujará una genealogía con toda la confianza del mundo. Por eso aquí
no basta con comprobar que devuelve algo — se comprueba que devuelve **lo
correcto**, y que avisa cuando no lo sabe.

Tres niveles:

* **Matemático exacto** — sobre una matriz *aditiva* (una que procede de un árbol
  de verdad), Neighbor-Joining tiene garantía teórica de recuperar las longitudes
  de rama exactas. Si eso falla, la implementación está mal, sin discusión.
* **Biológico simulado** — secuencias evolucionadas desde un ancestro con una
  topología conocida.
* **Honestidad** — el bootstrap debe dar soporte alto en datos claros y bajo en
  datos sin señal.
"""

from __future__ import annotations

import numpy as np
import pytest

from bioforge import (
    DistanceMatrix,
    bootstrap_support,
    build_tree,
    distance_matrix,
    neighbor_joining,
    upgma,
)
from bioforge.core.biocore import SequenceValueError
from bioforge.phylo.tree import _particiones
from bioforge import wpgma


def _mutar(seq: str, tasa: float, rng) -> str:
    a = np.array(list(seq))
    m = rng.random(len(seq)) < tasa
    if m.any():
        a[m] = rng.choice(list("ACGT"), size=int(m.sum()))
    return "".join(a)


@pytest.fixture()
def cuarteto():
    """Cuatro secuencias con topología VERDADERA conocida: (A,B) y (C,D)."""
    rng = np.random.default_rng(7)
    L = 1200
    anc = "".join(rng.choice(list("ACGT"), size=L))
    izq, der = _mutar(anc, 0.10, rng), _mutar(anc, 0.10, rng)
    seqs = [_mutar(izq, 0.03, rng), _mutar(izq, 0.03, rng),
            _mutar(der, 0.03, rng), _mutar(der, 0.03, rng)]
    return seqs, ["A", "B", "C", "D"]


# ── matriz de distancias ─────────────────────────────────────────────────────
def test_matriz_es_simetrica_y_diagonal_cero(cuarteto):
    seqs, nombres = cuarteto
    dm = distance_matrix(seqs, names=nombres)
    assert dm.matrix.shape == (4, 4)
    assert np.allclose(dm.matrix, dm.matrix.T)
    assert np.allclose(np.diag(dm.matrix), 0.0)


def test_parientes_mas_cerca_que_extranos(cuarteto):
    """A y B (hermanos) deben estar más cerca entre sí que de C o D."""
    seqs, nombres = cuarteto
    d = distance_matrix(seqs, names=nombres).matrix
    assert d[0, 1] < d[0, 2] and d[0, 1] < d[0, 3]
    assert d[2, 3] < d[0, 2] and d[2, 3] < d[1, 3]


def test_secuencias_identicas_distancia_cero():
    s = "ACGTACGTACGTACGTACGT"
    dm = distance_matrix([s, s, s], names=["x", "y", "z"])
    assert np.allclose(dm.matrix, 0.0)


def test_correccion_aumenta_la_distancia(cuarteto):
    """Jukes-Cantor corrige hacia ARRIBA: estima las mutaciones que no se ven."""
    seqs, nombres = cuarteto
    cruda = distance_matrix(seqs, model="p", names=nombres).matrix[0, 2]
    jc = distance_matrix(seqs, model="jc", names=nombres).matrix[0, 2]
    assert jc > cruda, "la corrección debe estimar más mutaciones que las observadas"


def test_huecos_se_excluyen_por_parejas():
    """Una columna con hueco no cuenta como diferencia: es ausencia de dato."""
    a = "ACGTACGT"
    b = "ACGT--GT"          # dos huecos, el resto idéntico
    dm = distance_matrix([a, b], names=["a", "b"], model="p")
    assert dm.matrix[0, 1] == pytest.approx(0.0), "los huecos no son diferencias"


def test_k2p_rechaza_proteina():
    prot = ["MKGFPWYEQLL", "MKGFPWYEQLI"]
    with pytest.raises(SequenceValueError, match="k2p"):
        distance_matrix(prot, model="k2p", protein=True)


def test_proteina_con_poisson():
    a = "MKGFPWYEQLLIPMKGFPWYE"
    b = "MKGFPWYEQLLIPMKGFPWYQ"
    dm = distance_matrix([a, b], model="poisson", names=["p1", "p2"], protein=True)
    assert 0 < dm.matrix[0, 1] < 1


def test_errores_de_uso():
    with pytest.raises(SequenceValueError, match="al menos 2"):
        distance_matrix(["ACGT"], names=["a"])
    with pytest.raises(SequenceValueError, match="ALINEADAS"):
        distance_matrix(["ACGT", "ACG"], names=["a", "b"])
    with pytest.raises(SequenceValueError, match="modelo"):
        distance_matrix(["ACGT", "ACGA"], model="inventado", names=["a", "b"])


def test_matriz_a_texto_phylip(cuarteto):
    seqs, nombres = cuarteto
    txt = distance_matrix(seqs, names=nombres).to_text()
    assert txt.splitlines()[0] == "4"
    assert txt.splitlines()[1].startswith("A")


# ── Neighbor-Joining: la prueba matemática exacta ────────────────────────────
def test_nj_recupera_longitudes_exactas_en_matriz_aditiva():
    """Garantía teórica de NJ: sobre una matriz ADITIVA recupera el árbol exacto.

    Se parte de un árbol sin raíz con longitudes conocidas
    (A-x, B-x, x-y interna, C-y, D-y) y se construye la matriz sumando ramas.
    NJ debe devolver esas mismas longitudes, no una aproximación.
    """
    a, b, c, d, e = 0.10, 0.20, 0.15, 0.25, 0.30     # e = rama interna
    D = np.array([
        [0.0,    a + b,      a + e + c,  a + e + d],
        [a + b,  0.0,        b + e + c,  b + e + d],
        [a + e + c, b + e + c, 0.0,      c + d],
        [a + e + d, b + e + d, c + d,    0.0],
    ])
    dm = DistanceMatrix(names=["A", "B", "C", "D"], matrix=D, model="p")
    arbol = neighbor_joining(dm)

    # la única partición informativa debe ser {A,B} | {C,D}
    assert [sorted(p) for p in _particiones(arbol)] == [["C", "D"]]

    # y las longitudes de rama deben ser las de partida
    largos = {}
    def recorrer(nodo):
        if nodo.is_leaf:
            largos[nodo.name] = nodo.length
        for h in nodo.children:
            recorrer(h)
    recorrer(arbol.root)
    assert largos["A"] == pytest.approx(a, abs=1e-9)
    assert largos["B"] == pytest.approx(b, abs=1e-9)
    assert largos["C"] == pytest.approx(c, abs=1e-9)
    assert largos["D"] == pytest.approx(d, abs=1e-9)

    # la rama interna: el nodo que agrupa a A y B mide e
    interna = [n for n in arbol.root.children if not n.is_leaf]
    assert len(interna) == 1
    assert interna[0].length == pytest.approx(e, abs=1e-9)


def test_nj_recupera_la_topologia_simulada(cuarteto):
    seqs, nombres = cuarteto
    arbol = build_tree(seqs, names=nombres, method="nj")
    assert [sorted(p) for p in _particiones(arbol)] == [["C", "D"]]
    assert arbol.n_leaves == 4 and arbol.rooted is False


def test_nj_aguanta_ritmos_desiguales_y_upgma_no():
    """La diferencia de libro entre los dos métodos, medida.

    Con dos linajes que mutan mucho más rápido (A y C), UPGMA —que supone reloj
    molecular— tiende a agruparlos por su parecido en "rama larga". NJ descuenta
    la distancia al resto y no cae. Este test fija esa propiedad: es la razón por
    la que NJ es el método por defecto.
    """
    rng = np.random.default_rng(11)
    L = 1500
    anc = "".join(rng.choice(list("ACGT"), size=L))
    izq, der = _mutar(anc, 0.05, rng), _mutar(anc, 0.05, rng)
    seqs = [_mutar(izq, 0.35, rng), _mutar(izq, 0.01, rng),    # A rápida, B lenta
            _mutar(der, 0.35, rng), _mutar(der, 0.01, rng)]    # C rápida, D lenta
    nombres = ["A", "B", "C", "D"]

    nj = [set(p) for p in _particiones(build_tree(seqs, names=nombres, method="nj"))]
    up = [set(p) for p in _particiones(build_tree(seqs, names=nombres, method="upgma"))]
    correcta = ({"C", "D"}, {"A", "B"})
    assert any(p in correcta for p in nj), "NJ debería resistir la atracción de ramas largas"
    assert not any(p in correcta for p in up), (
        "si UPGMA acierta aquí, el caso ya no ilustra su limitación: revisar el test")


def test_upgma_con_reloj_molecular(cuarteto):
    """Cuando el ritmo SÍ es constante, UPGMA acierta y además da raíz."""
    seqs, nombres = cuarteto
    arbol = build_tree(seqs, names=nombres, method="upgma")
    assert [sorted(p) for p in _particiones(arbol)] == [["C", "D"]]
    assert arbol.rooted is True


def test_pocas_secuencias_es_error():
    with pytest.raises(SequenceValueError, match="al menos 3"):
        neighbor_joining(DistanceMatrix(names=["a", "b"],
                                        matrix=np.array([[0.0, 0.1], [0.1, 0.0]]),
                                        model="p"))


def test_metodo_desconocido(cuarteto):
    seqs, nombres = cuarteto
    with pytest.raises(SequenceValueError, match="método"):
        build_tree(seqs, names=nombres, method="magia")


# ── Newick y export ──────────────────────────────────────────────────────────
def test_newick_bien_formado(cuarteto):
    seqs, nombres = cuarteto
    nw = build_tree(seqs, names=nombres).newick()
    assert nw.endswith(";")
    assert nw.count("(") == nw.count(")")
    for n in nombres:
        assert n in nw


def test_to_dict_anidado(cuarteto):
    seqs, nombres = cuarteto
    d = build_tree(seqs, names=nombres).to_dict()
    assert "children" in d
    hojas = []
    def rec(x):
        if "children" in x:
            for h in x["children"]:
                rec(h)
        else:
            hojas.append(x["name"])
    rec(d)
    assert sorted(hojas) == sorted(nombres)


# ── bootstrap: la parte honesta ──────────────────────────────────────────────
def test_bootstrap_da_soporte_alto_con_senal_clara(cuarteto):
    seqs, nombres = cuarteto
    arbol = bootstrap_support(seqs, names=nombres, replicates=100, seed=1)
    soportes = []
    def rec(n):
        if n.support is not None:
            soportes.append(n.support)
        for h in n.children:
            rec(h)
    rec(arbol.root)
    assert soportes, "debería anotar al menos una rama interna"
    assert max(soportes) >= 90, f"con señal clara el soporte debe ser alto: {soportes}"
    assert "100" in arbol.newick()               # aparece en el Newick


def test_bootstrap_avisa_cuando_no_hay_senal():
    """Cuatro secuencias AL AZAR no tienen genealogía: el soporte debe ser bajo.

    Es la prueba que separa una herramienta honesta de una que siempre dibuja algo
    bonito. El árbol saldrá igual (siempre sale), pero el soporte debe delatarlo.
    """
    rng = np.random.default_rng(99)
    seqs = ["".join(rng.choice(list("ACGT"), size=400)) for _ in range(5)]
    nombres = list("VWXYZ")
    arbol = bootstrap_support(seqs, names=nombres, replicates=100, seed=3)
    soportes = []
    def rec(n):
        if n.support is not None:
            soportes.append(n.support)
        for h in n.children:
            rec(h)
    rec(arbol.root)
    assert soportes
    assert max(soportes) < 70, (
        f"sin señal real el soporte no debería ser alto: {soportes}")


def test_bootstrap_no_cuenta_particiones_triviales():
    """REGRESIÓN: una partición que aísla UNA hoja aparece en cualquier árbol.

    Contarla daría soporte 100 siempre y falsearía la confianza. Solo cuentan las
    que dejan ≥2 hojas a cada lado.
    """
    rng = np.random.default_rng(5)
    L = 600
    anc = "".join(rng.choice(list("ACGT"), size=L))
    seqs = [_mutar(anc, 0.2, rng) for _ in range(4)]
    arbol = build_tree(seqs, names=["A", "B", "C", "D"], method="upgma")
    for p in _particiones(arbol):
        assert 2 <= len(p) <= 2, f"partición no informativa: {sorted(p)}"


def test_bootstrap_reproducible(cuarteto):
    seqs, nombres = cuarteto
    a = bootstrap_support(seqs, names=nombres, replicates=50, seed=42).newick()
    b = bootstrap_support(seqs, names=nombres, replicates=50, seed=42).newick()
    assert a == b, "con la misma semilla el resultado debe ser idéntico"


def test_bootstrap_replicas_invalidas(cuarteto):
    seqs, nombres = cuarteto
    with pytest.raises(SequenceValueError):
        bootstrap_support(seqs, names=nombres, replicates=0)


# ── contraste contra el estándar (Biopython) ─────────────────────────────────
def _biparticiones_bio(arbol, hojas):
    ancla = min(hojas)
    salida = set()
    for c in arbol.get_nonterminals():
        abajo = {t.name for t in c.get_terminals()}
        if min(len(abajo), len(hojas) - len(abajo)) >= 2:
            salida.add(frozenset(abajo if ancla not in abajo else hojas - abajo))
    return salida


@pytest.mark.parametrize("n", [10, 20, 40])
def test_nj_identico_a_biopython(n):
    """Nuestro Neighbor-Joining debe dar EXACTAMENTE el árbol de Biopython.

    Es la validación externa: coincidir con una implementación independiente y
    veterana demuestra que el algoritmo está bien, no solo que no revienta.
    """
    Bio = pytest.importorskip("Bio.Phylo.TreeConstruction")
    from Bio.Phylo.TreeConstruction import DistanceMatrix as BioDM

    rng = np.random.default_rng(n)
    M = rng.random((n, n)) * 0.5 + 0.05
    D = (M + M.T) / 2
    np.fill_diagonal(D, 0.0)
    nombres = [f"t{i:02d}" for i in range(n)]

    nuestro = _particiones(neighbor_joining(
        DistanceMatrix(names=nombres, matrix=D, model="p")))
    suyo = _biparticiones_bio(
        Bio.DistanceTreeConstructor().nj(
            BioDM(list(nombres), [[float(D[i][j]) for j in range(i + 1)] for i in range(n)])),
        set(nombres))
    assert nuestro == suyo, "nuestro NJ se separó del de Biopython"


@pytest.mark.parametrize("n", [20, 40])
def test_el_upgma_de_biopython_es_en_realidad_wpgma(n):
    """HALLAZGO fijado: ``Bio.Phylo`` etiqueta como UPGMA lo que es WPGMA.

    Su código promedia ``(d(k,i)+d(k,j))/2``, sin ponderar por el tamaño del
    grupo. Eso es WPGMA. El UPGMA de la definición original (Sokal & Michener)
    pondera, y es el que implementa ``bioforge.phylo.upgma``.

    Este test deja constancia de la diferencia para que nadie la tome por un bug
    nuestro, y garantiza que ``wpgma`` sigue sirviendo para reproducir su salida.
    """
    Bio = pytest.importorskip("Bio.Phylo.TreeConstruction")
    from Bio.Phylo.TreeConstruction import DistanceMatrix as BioDM

    rng = np.random.default_rng(n)
    M = rng.random((n, n)) * 0.5 + 0.05
    D = (M + M.T) / 2
    np.fill_diagonal(D, 0.0)
    nombres = [f"t{i:02d}" for i in range(n)]
    hojas = set(nombres)
    dm = DistanceMatrix(names=nombres, matrix=D, model="p")

    def suyo():
        return _biparticiones_bio(
            Bio.DistanceTreeConstructor().upgma(
                BioDM(list(nombres),
                      [[float(D[i][j]) for j in range(i + 1)] for i in range(n)])),
            hojas)

    assert _particiones(wpgma(dm)) == suyo(), (
        "nuestro WPGMA debería reproducir exactamente el 'upgma' de Biopython")
    assert _particiones(upgma(dm)) != suyo(), (
        "si nuestro UPGMA coincide con el suyo, o lo han corregido o hemos "
        "dejado de ponderar por tamaño: revisar antes de tocar nada")
