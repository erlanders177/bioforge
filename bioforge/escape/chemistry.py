"""Eje B — escape a anticuerpos por la quimica del residuo DESTINO.

QUE ES ESTO
-----------
Dado un cambio de aminoacido, estima si esa sustitucion es de las que mas
estorban a un anticuerpo *entre las alternativas de su misma posicion*.

    score = z(hidrofilia del destino) + z(volumen del destino)

Las dos piezas son propiedades del residuo al que se LLEGA, no del cambio. Eso
es lo que lo separa del planteamiento habitual: EVEscape (Marks lab, Nature
2023) modela este termino como DISIMILITUD -cuanto te alejas del original- y
medido contra escape real esa formulacion rinde mucho menos.

MEDIDO (tools/bench_escape_multivirus.py), frente al termino de EVEscape:

    conjunto                  familia            rol          EVEscape   este
    SARS-CoV-2 RBD            Coronaviridae      desarrollo     +0.107  +0.172
    gripe H3N2 - sueros       Orthomyxoviridae   desarrollo     +0.045  +0.164
    gripe H3N2 - mAbs         Orthomyxoviridae   desarrollo     +0.070  +0.205
    VIH-1 BG505 - bnAbs       Retroviridae       desarrollo     +0.021  +0.135
    Zika proteina E           Flaviviridae       desarrollo     +0.103  +0.135
    VIH-1 - sueros HUMANOS    Retroviridae       RETENIDO       +0.053  +0.169
    VIH-1 - sueros CONEJO     Retroviridae       RETENIDO       +0.020  +0.068

Gana en 7/7, con intervalo de confianza limpio en 6/7, incluidos los dos
RETENIDOS (no intervinieron en ninguna decision de diseno). Datos: mapas de
escape por deep mutational scanning del laboratorio de Bloom.

LO QUE ESTO **NO** ES
---------------------
No predice que variante se impondra. Medido sobre 74.065 mutaciones reales de
33 clados: este score separa las que se propagaron de las que no en **1.10x**,
frente a **0.98x** del azar ppuro. Es decir, no informa nada del exito
epidemiologico -y el termino de EVEscape sale a 0.55x, peor que el azar-.
El motivo no es el score: es que propagarse = escape x fitness x
transmisibilidad x azar, y el escape es solo un factor.

Por eso este modulo **no expone ninguna probabilidad de propagacion**. Ver
``NO_PREDICE_PROPAGACION``.
"""
from typing import Optional, Sequence

__all__ = ["AMINOACIDOS", "NO_PREDICE_PROPAGACION", "escape_score",
           "score_sitio", "percentil_en_sitio"]

AMINOACIDOS = "ACDEFGHIKLMNPQRSTVWY"

# Kyte & Doolittle 1982 (hidropatia): negativo = hidrofilico
_HIDROPATIA = dict(zip(AMINOACIDOS,
                       (1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
                        1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3)))
# volumen del residuo en A^3
_VOLUMEN = dict(zip(AMINOACIDOS,
                    (88.6, 108.5, 111.1, 138.4, 189.9, 60.1, 153.2, 166.7,
                     168.6, 166.7, 162.9, 114.1, 112.7, 143.8, 173.4, 89.0,
                     116.1, 140.0, 227.8, 193.6)))

NO_PREDICE_PROPAGACION = (
    "Este eje mide ESCAPE A ANTICUERPOS medido en laboratorio, no exito "
    "epidemiologico. Sobre 74.065 mutaciones reales de 33 clados, separa las "
    "que se propagaron en 1.10x frente a 1.00x del azar: no informa. "
    "Propagarse = escape x fitness x transmisibilidad x azar, y aqui solo se "
    "mide el primer factor. Cualquier 'probabilidad de propagacion' calculada "
    "a partir de esto seria inventada."
)


def _z(valores: Sequence[float]) -> list[float]:
    """Tipificar sin NumPy: la entrada son <=20 numeros."""
    n = len(valores)
    media = sum(valores) / n
    var = sum((v - media) ** 2 for v in valores) / n
    if var <= 0:
        return [0.0] * n
    ds = var ** 0.5
    return [(v - media) / ds for v in valores]


def _validar(aa: str, que: str) -> str:
    aa = (aa or "").strip().upper()
    if len(aa) != 1 or aa not in AMINOACIDOS:
        raise ValueError(f"{que} no es un aminoacido valido: {aa!r}")
    return aa


def score_sitio(alternativas: Optional[Sequence[str]] = None) -> dict[str, float]:
    """Puntuacion de cada destino posible, tipificada DENTRO del sitio.

    El score solo tiene sentido comparando alternativas de la misma posicion:
    es un orden local, no una escala absoluta. Si no se dan alternativas se
    usan los 20 aminoacidos.

    >>> s = score_sitio()
    >>> s["R"] > s["V"]        # arginina: grande e hidrofilica
    True
    """
    alt = list(alternativas) if alternativas else list(AMINOACIDOS)
    alt = [_validar(a, "alternativa") for a in alt]
    if len(alt) < 3:
        raise ValueError("hacen falta al menos 3 alternativas para comparar")
    hid = _z([-_HIDROPATIA[a] for a in alt])
    vol = _z([_VOLUMEN[a] for a in alt])
    return {a: h + v for a, h, v in zip(alt, hid, vol)}


def escape_score(wt: str, mut: str,
                 alternativas: Optional[Sequence[str]] = None) -> float:
    """Puntuacion de escape de ``wt -> mut``, dentro de su sitio.

    ``wt`` no entra en el calculo -esa es justamente la tesis: manda el destino-
    pero se pide para validar la sustitucion y para que la llamada se lea como
    la mutacion que es.

    >>> round(escape_score("L", "R"), 2) > round(escape_score("L", "V"), 2)
    True
    """
    _validar(wt, "residuo original")
    mut = _validar(mut, "residuo mutante")
    tabla = score_sitio(alternativas)
    if mut not in tabla:
        raise ValueError(f"{mut} no esta entre las alternativas del sitio")
    return tabla[mut]


def percentil_en_sitio(wt: str, mut: str,
                       alternativas: Optional[Sequence[str]] = None) -> float:
    """Fraccion de alternativas del sitio a las que este cambio supera (0..1).

    Es la salida honesta del eje: un ORDEN dentro de la posicion. 1.0 significa
    'la que mas estorba de las posibles aqui', no 'escapara'.
    """
    tabla = score_sitio(alternativas)
    mut = _validar(mut, "residuo mutante")
    _validar(wt, "residuo original")
    if mut not in tabla:
        raise ValueError(f"{mut} no esta entre las alternativas del sitio")
    otros = [v for a, v in tabla.items() if a != mut]
    if not otros:
        return 0.5
    mio = tabla[mut]
    menores = sum(1 for v in otros if v < mio)
    iguales = sum(1 for v in otros if v == mio)
    return (menores + 0.5 * iguales) / len(otros)
