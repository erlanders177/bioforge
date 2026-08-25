"""
bioforge/phylo/distance.py — de un alineamiento a una matriz de distancias.

Antes de dibujar un árbol evolutivo hay que responder a una pregunta más simple:
**¿cuánto se diferencian dos secuencias?** Contar las letras distintas parece
suficiente, pero engaña: si dos secuencias llevan mucho tiempo separadas, una
misma posición puede haber mutado **varias veces** y haber vuelto a la letra
original. Las diferencias observadas se «saturan» y subestiman el tiempo real.

Por eso se usan *modelos de corrección*, que estiman las mutaciones que hubo a
partir de las que se ven:

* ``p``        — proporción cruda de posiciones distintas. Sin corregir.
* ``jc``       — Jukes-Cantor (1969): todas las mutaciones son igual de probables.
                 ``d = −3/4 · ln(1 − 4p/3)``
* ``k2p``      — Kimura de 2 parámetros (1980): distingue **transiciones**
                 (A↔G, C↔T, químicamente parecidas y mucho más frecuentes) de
                 **transversiones** (el resto).
                 ``d = −½·ln(1−2P−Q) − ¼·ln(1−2Q)``
* ``poisson``  — para proteínas: ``d = −ln(1 − p)``.

Todo el cálculo es **vectorizado con productos de matrices**: la matriz de todas
las parejas sale de multiplicar la codificación *one-hot* del alineamiento por su
propia traspuesta. No hay un solo bucle por posición ni por pareja.
"""

from __future__ import annotations

from typing import Iterable, NamedTuple, Optional, Sequence

import numpy as np

from bioforge.core.biocore import SequenceValueError

# Alfabetos: ADN (4 símbolos) y proteína (20). Los huecos y símbolos ambiguos no
# son un estado más: son AUSENCIA de dato, y se excluyen del recuento.
_ADN = "ACGT"
_PROT = "ACDEFGHIKLMNPQRSTVWY"

_MAX_DIST = 5.0          # tope cuando el modelo se satura (log de un negativo)


def _lut(alfabeto: str) -> np.ndarray:
    """LUT ASCII → índice del símbolo; 255 = no válido (hueco, N, X…)."""
    lut = np.full(256, 255, dtype=np.uint8)
    for i, ch in enumerate(alfabeto):
        lut[ord(ch)] = i
        lut[ord(ch.lower())] = i
    return lut


_LUT_ADN = _lut(_ADN)
_LUT_PROT = _lut(_PROT)


class DistanceMatrix(NamedTuple):
    """Matriz de distancias por parejas.

    Attributes
    ----------
    names:
        Nombre de cada secuencia, en el orden de las filas/columnas.
    matrix:
        Matriz cuadrada ``(n, n)``, simétrica y con diagonal 0.
    model:
        Modelo de corrección aplicado (``p``, ``jc``, ``k2p``, ``poisson``).
    saturated:
        Cuántas parejas superaron el límite del modelo (demasiado divergentes
        para corregirlas): su distancia se recortó a un tope. Si es alto, el
        árbol resultante hay que mirarlo con desconfianza.
    """

    names: list[str]
    matrix: np.ndarray
    model: str
    saturated: int = 0

    def __len__(self) -> int:
        return len(self.names)

    def to_text(self, decimals: int = 4) -> str:
        """Formato PHYLIP de matriz de distancias (el que leen otras herramientas)."""
        n = len(self.names)
        filas = [f"{n}"]
        for i, nom in enumerate(self.names):
            vals = " ".join(f"{v:.{decimals}f}" for v in self.matrix[i])
            filas.append(f"{nom[:10]:<10} {vals}")
        return "\n".join(filas) + "\n"

    def __repr__(self) -> str:                       # pragma: no cover - cosmético
        return (f"DistanceMatrix({len(self.names)} secuencias, modelo {self.model!r}"
                + (f", {self.saturated} parejas saturadas" if self.saturated else "")
                + ")")


def _codificar(aligned: Sequence[str], proteina: bool):
    """Alineamiento → (códigos (n,L) uint8, válidos (n,L) bool)."""
    n = len(aligned)
    L = len(aligned[0])
    plano = "".join(aligned).encode("ascii")
    arr = np.frombuffer(plano, dtype=np.uint8).reshape(n, L)
    lut = _LUT_PROT if proteina else _LUT_ADN
    codes = lut[arr]
    return codes, codes != 255


def _pares_validos(validos: np.ndarray) -> np.ndarray:
    """(n,n): en cuántas columnas AMBAS secuencias tienen dato."""
    V = validos.astype(np.float32)
    return V @ V.T


def _iguales(codes: np.ndarray, validos: np.ndarray, S: int) -> np.ndarray:
    """(n,n): en cuántas columnas ambas son válidas y llevan el MISMO símbolo.

    Se resuelve con un único producto de matrices sobre la codificación one-hot:
    ``X[i, l·S + s] = 1`` si la secuencia *i* tiene el símbolo *s* en la columna
    *l*. Entonces ``(X · Xᵀ)[i,j]`` cuenta exactamente las coincidencias.
    """
    n, L = codes.shape
    X = np.zeros((n, L * S), dtype=np.float32)
    filas, cols = np.nonzero(validos)
    X[filas, cols * S + codes[filas, cols]] = 1.0
    return X @ X.T


def _contingencia(codes: np.ndarray, validos: np.ndarray, a: int, b: int) -> np.ndarray:
    """(n,n): columnas donde *i* tiene el símbolo ``a`` y *j* tiene el símbolo ``b``."""
    Xa = ((codes == a) & validos).astype(np.float32)
    Xb = ((codes == b) & validos).astype(np.float32)
    return Xa @ Xb.T


def _corregir(p: np.ndarray, modelo: str) -> tuple[np.ndarray, int]:
    """Aplica el modelo de sustitución a la proporción cruda de diferencias."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if modelo == "p":
            d = p.copy()
            saturadas = 0
        elif modelo == "jc":
            dentro = 1.0 - (4.0 / 3.0) * p           # el logaritmo exige que sea > 0
            saturadas = int(np.count_nonzero(dentro <= 0) // 2)
            d = np.where(dentro > 0, -0.75 * np.log(np.maximum(dentro, 1e-12)), _MAX_DIST)
        elif modelo == "poisson":
            dentro = 1.0 - p
            saturadas = int(np.count_nonzero(dentro <= 0) // 2)
            d = np.where(dentro > 0, -np.log(np.maximum(dentro, 1e-12)), _MAX_DIST)
        else:
            raise SequenceValueError(f"modelo desconocido: {modelo!r}")
    return np.clip(np.nan_to_num(d, nan=_MAX_DIST), 0.0, _MAX_DIST), saturadas


def distance_matrix(aligned: Iterable[str], *, model: str = "jc",
                    names: Optional[Sequence[str]] = None,
                    protein: Optional[bool] = None) -> DistanceMatrix:
    """Calcula la matriz de distancias de un alineamiento.

    Parameters
    ----------
    aligned:
        Secuencias **ya alineadas** (todas de la misma longitud, con ``-`` en los
        huecos), tal como las devuelve
        :func:`bioforge.align.msa.align_multiple` en ``.aligned``.
    model:
        ``"p"`` (crudo), ``"jc"`` (Jukes-Cantor, por defecto para ADN),
        ``"k2p"`` (Kimura 2 parámetros, solo ADN) o ``"poisson"`` (proteínas).
    names:
        Nombres de las secuencias. Por defecto ``seq1``, ``seq2``…
    protein:
        Fuerza el alfabeto. Por defecto se deduce del contenido.

    Returns
    -------
    DistanceMatrix

    Notes
    -----
    Las columnas donde alguna de las dos secuencias tiene un hueco o un símbolo
    ambiguo se **excluyen de esa pareja** (borrado por parejas). Es lo estándar, y
    lo honesto: no inventamos datos donde no los hay.
    """
    seqs = [s.upper() for s in aligned]
    if len(seqs) < 2:
        raise SequenceValueError(
            f"hacen falta al menos 2 secuencias para una matriz de distancias "
            f"(hay {len(seqs)}).")
    L = len(seqs[0])
    if any(len(s) != L for s in seqs):
        raise SequenceValueError(
            "las secuencias deben estar ALINEADAS (todas de la misma longitud). "
            "Usa align_multiple() antes de llamar aquí.")
    if L == 0:
        raise SequenceValueError("el alineamiento no tiene columnas.")

    if protein is None:                              # deducción simple y explícita
        cuerpo = set("".join(seqs)) - set("-.*XN")
        protein = bool(cuerpo - set(_ADN + "U"))
    S = len(_PROT) if protein else len(_ADN)

    if model == "k2p" and protein:
        raise SequenceValueError(
            "el modelo k2p distingue transiciones de transversiones, que solo "
            "existen en ADN. Usa 'poisson' o 'p' para proteínas.")

    codes, validos = _codificar(seqs, protein)
    comunes = _pares_validos(validos)
    if not np.any(comunes > 0):
        raise SequenceValueError(
            "ninguna pareja de secuencias comparte columnas con datos: "
            "¿el alineamiento es solo huecos?")

    nombres = list(names) if names is not None else [f"seq{i+1}" for i in range(len(seqs))]
    if len(nombres) != len(seqs):
        raise SequenceValueError(
            f"hay {len(seqs)} secuencias pero {len(nombres)} nombres.")

    seguro = np.maximum(comunes, 1.0)
    if model == "k2p":
        # transiciones: A(0)↔G(2) y C(1)↔T(3)
        trans = (_contingencia(codes, validos, 0, 2) + _contingencia(codes, validos, 2, 0)
                 + _contingencia(codes, validos, 1, 3) + _contingencia(codes, validos, 3, 1))
        iguales = _iguales(codes, validos, S)
        distintas = comunes - iguales
        transv = distintas - trans
        P, Q = trans / seguro, transv / seguro
        with np.errstate(divide="ignore", invalid="ignore"):
            a = 1.0 - 2.0 * P - Q
            b = 1.0 - 2.0 * Q
            valido = (a > 0) & (b > 0)
            saturadas = int(np.count_nonzero(~valido & (comunes > 0)) // 2)
            d = np.where(valido,
                         -0.5 * np.log(np.maximum(a, 1e-12))
                         - 0.25 * np.log(np.maximum(b, 1e-12)),
                         _MAX_DIST)
        d = np.clip(np.nan_to_num(d, nan=_MAX_DIST), 0.0, _MAX_DIST)
    else:
        iguales = _iguales(codes, validos, S)
        p = (comunes - iguales) / seguro
        d, saturadas = _corregir(p, model)

    d = np.where(comunes > 0, d, _MAX_DIST)          # sin solape → máxima distancia
    d = (d + d.T) / 2.0                              # simetría exacta
    np.fill_diagonal(d, 0.0)
    return DistanceMatrix(names=nombres, matrix=d.astype(np.float64),
                          model=model, saturated=int(saturadas))
