"""Eje B — la quimica del destino y el veredicto calibrado."""
import pytest

from bioforge.escape.chemistry import (AMINOACIDOS, escape_score,
                                       percentil_en_sitio, score_sitio)
from bioforge.escape.verdict import (BASE_CUARTIL_SUPERIOR, CALIBRACION,
                                     MITADES, evaluar, evaluar_muchas,
                                     probabilidad_de_propagacion)


# --------------------------------------------------------------- la quimica
def test_el_score_solo_depende_del_destino():
    """La tesis del eje: manda el residuo al que se LLEGA, no de donde vienes."""
    assert escape_score("L", "R") == pytest.approx(escape_score("V", "R"))
    assert escape_score("A", "R") == pytest.approx(escape_score("W", "R"))


def test_arginina_por_encima_de_valina():
    """R es grande e hidrofilica; V pequena e hidrofoba. El orden es la senal."""
    assert escape_score("L", "R") > escape_score("L", "V")


def test_los_extremos_son_los_esperados():
    tabla = score_sitio()
    mejor = max(tabla, key=tabla.get)
    peor = min(tabla, key=tabla.get)
    assert mejor in "RKWY", f"el maximo deberia ser grande e hidrofilico, es {mejor}"
    assert peor in "AGCV", f"el minimo deberia ser pequeno e hidrofobo, es {peor}"


def test_el_score_esta_tipificado_dentro_del_sitio():
    """Es un orden LOCAL: la media de las alternativas es cero por construccion."""
    tabla = score_sitio()
    assert sum(tabla.values()) == pytest.approx(0.0, abs=1e-9)


def test_las_alternativas_cambian_el_orden():
    """Comparar contra 20 no es lo mismo que contra las que de verdad salen."""
    todos = percentil_en_sitio("E", "K")
    pocas = percentil_en_sitio("E", "K", ["K", "R", "W"])
    assert todos != pocas


def test_percentil_en_rango():
    for aa in AMINOACIDOS:
        assert 0.0 <= percentil_en_sitio("A", aa) <= 1.0


@pytest.mark.parametrize("wt,mut", [("Z", "K"), ("E", "J"), ("", "K"), ("E", "")])
def test_aminoacidos_invalidos_dan_error_claro(wt, mut):
    with pytest.raises(ValueError):
        escape_score(wt, mut)


def test_hacen_falta_alternativas_suficientes():
    with pytest.raises(ValueError):
        score_sitio(["K", "R"])


def test_el_mutante_debe_estar_entre_las_alternativas():
    with pytest.raises(ValueError):
        escape_score("E", "K", ["A", "V", "L"])


# ------------------------------------------------------------- el veredicto
def test_el_veredicto_es_monotono():
    """Un percentil mayor NUNCA puede dar una probabilidad menor.

    La calibracion cruda por cuartiles NO es monotona (q3 32.2% > q4 30.6%):
    por eso se reportan dos niveles y no cuatro. Este test fija esa decision.
    """
    vs = evaluar_muchas([f"E484{a}" for a in AMINOACIDOS if a != "E"])
    for anterior, siguiente in zip(vs, vs[1:]):
        assert anterior.percentil >= siguiente.percentil
        assert anterior.p_escape_alto >= siguiente.p_escape_alto


def test_solo_hay_dos_niveles_declarados():
    niveles = {evaluar(f"E484{a}").p_escape_alto
               for a in AMINOACIDOS if a != "E"}
    assert len(niveles) == 2, "el dato no da para mas de dos niveles"
    assert niveles == set(MITADES)


def test_la_calibracion_rodea_la_base():
    """Un nivel por encima del 25% base y otro por debajo: si no, no informa."""
    assert MITADES[0] < BASE_CUARTIL_SUPERIOR < MITADES[1]


def test_la_calibracion_cruda_queda_expuesta():
    """El crudo no monotono se publica; no se esconde detras del suavizado."""
    assert len(CALIBRACION) == 4
    assert CALIBRACION[2] > CALIBRACION[3], "asi salio medido, y asi se deja"


def test_el_enriquecimiento_es_modesto_y_se_dice():
    """No vender como fuerte lo que mueve del 25% al 31%."""
    v = evaluar("L452R")
    assert 1.1 < v.enriquecimiento < 1.5


def test_el_texto_avisa_de_lo_que_NO_dice():
    texto = str(evaluar("E484K"))
    assert "propagar" in texto.lower()


def test_se_niega_a_dar_probabilidad_de_propagacion():
    """La medicion dijo 1.10x frente a 1.00x del azar: no se ofrece el numero."""
    with pytest.raises(NotImplementedError) as e:
        probabilidad_de_propagacion("E484K")
    assert "1.10x" in str(e.value)


@pytest.mark.parametrize("mala", ["", "X", "484", "E484"])
def test_mutaciones_mal_escritas_dan_error(mala):
    with pytest.raises(ValueError):
        evaluar(mala)


def test_ordena_de_mayor_a_menor():
    vs = evaluar_muchas(["E484A", "E484K", "E484G", "E484R"])
    assert [v.mutation for v in vs][0] in ("E484R", "E484K")
    assert vs[-1].mutation in ("E484A", "E484G")
