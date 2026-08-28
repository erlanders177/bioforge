"""El veredicto de escape: numeros CALIBRADOS, y lo que se niega a decir.

Un porcentaje solo vale si esta calibrado: de cada 100 mutaciones a las que se
les da un 31 %, unas 31 tienen que cumplirlo. Este modulo solo expone numeros
que se han medido contra datos reales, y se niega explicitamente a dar los que
no se sostienen.

LO QUE SI SE OFRECE, y de donde sale
------------------------------------
``p_escape_alto``: probabilidad de que la sustitucion este en el CUARTIL
SUPERIOR de escape de su propia posicion. Calibrado sobre los siete conjuntos
de escape medido (deep mutational scanning, laboratorio de Bloom; ~4.000
anticuerpos y sueros, cuatro familias de virus). Base: 25 % por definicion.

    cuartil de score      1       2       3       4
    % real de escape alto 20.8 %  22.5 %  32.2 %  30.6 %

Consistente en los 7 conjuntos por separado (razon extremo/extremo de 1.21x a
1.77x), incluidos los dos RETENIDOS.

La curva **no es monotona**: el cuartil 3 rinde igual o algo mas que el 4. Eso
no se maquilla: se lee como que **el dato no da para cuatro niveles**, y por eso
se reportan DOS (~22 % y ~31 %). El crudo queda expuesto en ``CALIBRACION``
para quien quiera comprobarlo.

Es una senal REAL pero MODESTA: mueve del 25 % base al 31 % o al 22 %. Se dice
con esas palabras, no con dos decimales que aparenten precision.

LO QUE SE NIEGA A DECIR
-----------------------
La probabilidad de que una variante se PROPAGUE. Medido y descartado: ver
``chemistry.NO_PREDICE_PROPAGACION``. ``probabilidad_de_propagacion()`` existe
solo para explicar por que no existe.
"""
from typing import NamedTuple, Optional, Sequence

__all__ = ["EscapeVerdict", "CALIBRACION", "BASE_CUARTIL_SUPERIOR",
           "MITADES",
           "evaluar", "evaluar_muchas", "probabilidad_de_propagacion"]

# Fraccion REAL en el cuartil superior de escape, por cuartil de score,
# medida sobre los 7 conjuntos juntos. Es el dato crudo, sin suavizar.
CALIBRACION: tuple[float, ...] = (0.208, 0.225, 0.322, 0.306)
BASE_CUARTIL_SUPERIOR = 0.25          # por definicion de "cuartil superior"

# La curva NO es monotona: el cuartil 3 (32.2 %) rinde igual o algo mas que el
# 4 (30.6 %). No es una anomalia a maquillar: significa que **el dato no da
# para cuatro niveles**. Dar cuatro seria fingir una resolucion que no existe,
# y produciria salidas absurdas (una mutacion en el percentil 89 puntuando por
# debajo de otra en el 74). Se reportan DOS niveles, que es lo que aguanta:
MITADES: tuple[float, float] = (
    (CALIBRACION[0] + CALIBRACION[1]) / 2,        # mitad inferior del score
    (CALIBRACION[2] + CALIBRACION[3]) / 2,        # mitad superior
)


class EscapeVerdict(NamedTuple):
    """Lo que se puede decir de una sustitucion, y con cuanta confianza."""

    mutation: str
    percentil: float                  # 0..1, orden DENTRO de su posicion
    p_escape_alto: float              # calibrado; comparar con base 0.25
    base: float
    lectura: str                      # "por encima" / "en la media" / "por debajo"

    @property
    def enriquecimiento(self) -> float:
        """Cuantas veces la base. 1.0 = no aporta nada."""
        return self.p_escape_alto / self.base

    def __str__(self) -> str:
        return (
            f"{self.mutation}\n"
            f"  orden en su posicion : percentil {self.percentil:.0%} "
            f"(supera al {self.percentil:.0%} de las alternativas)\n"
            f"  escape alto          : {self.p_escape_alto:.0%} "
            f"(base {self.base:.0%}, x{self.enriquecimiento:.2f}) -> {self.lectura}\n"
            f"  NO dice              : si se propagara. Eso no se puede medir "
            f"con esto."
        )


def _lectura(p: float) -> str:
    if p >= BASE_CUARTIL_SUPERIOR * 1.15:
        return "algo mas probable que la media de su sitio"
    if p <= BASE_CUARTIL_SUPERIOR * 0.9:
        return "algo menos probable que la media de su sitio"
    return "indistinguible de la media de su sitio"


def _parse(mutacion: str) -> tuple[str, str, str]:
    """'E484K' -> ('E', '484', 'K'). Acepta tambien 'E484K' sin posicion util."""
    t = (mutacion or "").strip().upper()
    if len(t) < 3:
        raise ValueError(f"mutacion no reconocida: {mutacion!r}")
    wt, mut, pos = t[0], t[-1], t[1:-1]
    if not pos:
        raise ValueError(f"mutacion sin posicion: {mutacion!r}")
    return wt, pos, mut


def evaluar(mutacion: str,
            alternativas: Optional[Sequence[str]] = None) -> EscapeVerdict:
    """Veredicto calibrado para una sustitucion, p. ej. ``evaluar("E484K")``.

    ``alternativas``: los aminoacidos realmente observables en esa posicion. Si
    no se dan, se comparan los 20. Darlas ajusta el orden al sitio real.

    >>> v = evaluar("L452R")
    >>> 0.0 <= v.percentil <= 1.0 and 0.0 < v.p_escape_alto < 1.0
    True
    """
    from .chemistry import percentil_en_sitio          # dentro: carga perezosa

    wt, _, mut = _parse(mutacion)
    pct = percentil_en_sitio(wt, mut, alternativas)
    p = MITADES[1 if pct >= 0.5 else 0]        # dos niveles: lo que aguanta
    return EscapeVerdict(mutacion.strip().upper(), pct, p,
                         BASE_CUARTIL_SUPERIOR, _lectura(p))


def evaluar_muchas(mutaciones: Sequence[str],
                   alternativas: Optional[Sequence[str]] = None
                   ) -> list[EscapeVerdict]:
    """Veredicto para varias, ordenadas de mayor a menor percentil."""
    out = [evaluar(m, alternativas) for m in mutaciones]
    return sorted(out, key=lambda v: -v.percentil)


def probabilidad_de_propagacion(*_args, **_kwargs):
    """No existe, y esta funcion esta aqui para explicar por que.

    Se midio: sobre 74.065 mutaciones reales de 33 clados de tres virus, este
    eje separa las que se propagaron de las que no en 1.10x, frente a 1.00x del
    azar. El termino equivalente de EVEscape sale a 0.55x, peor que el azar.
    Ninguno informa del exito epidemiologico.

    Si lo que se busca es la trayectoria REAL de una mutacion que ya circula,
    eso si son datos y no conjetura: ``bioforge.evolution.realitycheck``
    devuelve su frecuencia observada, separada de cualquier estimacion.
    """
    from .chemistry import NO_PREDICE_PROPAGACION

    raise NotImplementedError(
        NO_PREDICE_PROPAGACION +
        " Para la trayectoria REAL de una mutacion que ya circula, usa "
        "bioforge.evolution.realitycheck.RealityCheck, que separa lo OBSERVADO "
        "de lo ESTIMADO."
    )


# Alias para la API publica de `bioforge`, donde `evaluar` a secas seria ambiguo
# junto a las demas familias.
evaluar_escape = evaluar
evaluar_escapes = evaluar_muchas
__all__ += ["evaluar_escape", "evaluar_escapes"]
