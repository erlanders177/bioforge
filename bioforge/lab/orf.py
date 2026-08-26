"""
bioforge/lab/orf.py — buscar genes candidatos (marcos abiertos de lectura).

Un **ORF** (*open reading frame*, marco abierto de lectura) es un tramo de ADN que
podría codificar una proteína: empieza en un codón de inicio (``ATG``), sigue de
tres en tres sin tropezarse con ningún codón de parada, y termina en uno
(``TAA``, ``TAG`` o ``TGA``).

Encontrarlos es el primer paso para responder *«¿qué genes hay aquí?»* ante una
secuencia desconocida — un genoma bacteriano recién ensamblado, un virus nuevo, un
plásmido que alguien te pasó sin anotar.

Los **seis marcos**
-------------------
El ADN tiene dos hebras, y en cada una se puede empezar a leer en tres puntos
distintos (posición 0, 1 o 2). Eso da **seis marcos de lectura** posibles, y un gen
puede estar en cualquiera de ellos. Buscar solo en uno es el error clásico.

Dos caminos, elegidos por tamaño
--------------------------------
El escaneo de cada marco tiene **dos implementaciones que dan lo mismo**, y se
escoge según el tamaño de la entrada. La razón está medida:

=================  =============  ==========  ====================
secuencia          Python puro    NumPy       quién gana
=================  =============  ==========  ====================
plásmido 5 kb          2.5 ms       2.4 ms    empate
virus 50 kb           30.4 ms      15.5 ms    NumPy en cálculo
bacteria 0.5 Mb      312.2 ms     152.9 ms    NumPy en cálculo
=================  =============  ==========  ====================

NumPy solo va **2× más rápido**, pero **cargarlo cuesta ~500 ms fijos**. El punto de
equilibrio ronda **1,5 Mb**: por debajo, Python puro es más rápido *de punta a
punta*, porque no paga esa carga. Como la mayor parte del trabajo real son
plásmidos y virus, el camino por defecto es el puro; NumPy entra solo en genomas
grandes, donde de verdad compensa.

Es el mismo patrón que el motor C frente al NumPy en el resto del proyecto: dos
caminos y una red de paridad que exige que den EXACTAMENTE lo mismo.
"""

from __future__ import annotations

import bisect
from typing import NamedTuple, Optional

from bioforge.core.errors import SequenceValueError

_COMPL = bytes.maketrans(b"ACGTacgtNn", b"TGCAtgcaNn")

# Tabla del código genético, indexada por 16·b₀ + 4·b₁ + b₂ (A=0, C=1, G=2, T=3)
_TABLA = "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF"
_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def _codigo(codon: str) -> int:
    a, b, c = (_IDX.get(x, -1) for x in codon)
    return -1 if -1 in (a, b, c) else a * 16 + b * 4 + c


_ATG = _codigo("ATG")
_STOPS = frozenset(_codigo(c) for c in ("TAA", "TAG", "TGA"))

# Por encima de este tamaño compensa pagar la carga de NumPy (ver el docstring).
UMBRAL_NUMPY = 1_500_000


def _revcomp(s: str) -> str:
    return s.encode("ascii").translate(_COMPL)[::-1].decode("ascii")


class ORF(NamedTuple):
    """Un marco abierto de lectura encontrado.

    Attributes
    ----------
    start, end:
        Coordenadas **0-based** sobre la secuencia original (hebra directa), con
        ``end`` exclusivo. Incluyen el codón de parada si lo hay.
    strand:
        ``"+"`` o ``"-"``.
    frame:
        1..3 en la hebra directa, −1..−3 en la inversa (convención del campo).
    length:
        Longitud en nucleótidos.
    protein:
        La proteína que codifica (sin el ``*`` final).
    has_stop:
        Si termina en codón de parada. Un ORF que llega al final de la secuencia
        sin parada puede estar **truncado**: el gen quizá continúa más allá.
    """

    start: int
    end: int
    strand: str
    frame: int
    length: int
    protein: str
    has_stop: bool

    @property
    def n_aa(self) -> int:
        return len(self.protein)

    def __repr__(self) -> str:                        # pragma: no cover - cosmético
        return (f"ORF({self.strand}{abs(self.frame)} {self.start}-{self.end}, "
                f"{self.length} nt, {self.n_aa} aa"
                + ("" if self.has_stop else ", truncado") + ")")


def escanear_python(s: str, marco: int) -> tuple[list[int], int]:
    """Códigos de codón de un marco, en Python puro. Devuelve (códigos, nº codones)."""
    # max(0,…): con una secuencia más corta que el desplazamiento del marco, la
    # división entera de Python da NEGATIVO. Lo cazó la red de paridad.
    n = max(0, (len(s) - marco) // 3)
    obten = _IDX.get
    salida = []
    for i in range(marco, marco + 3 * n, 3):          # bucle por CODÓN
        a = obten(s[i]); b = obten(s[i + 1]); c = obten(s[i + 2])
        salida.append(-1 if a is None or b is None or c is None
                      else a * 16 + b * 4 + c)
    return salida, n


def escanear_numpy(s: str, marco: int) -> tuple[list[int], int]:
    """Lo mismo, vectorizado. Solo se usa con secuencias muy grandes."""
    import numpy as np

    base = np.full(256, 255, dtype=np.uint8)
    for i, b in enumerate(b"ACGT"):
        base[b] = i
        base[b + 32] = i
    codes = base[np.frombuffer(s.encode("ascii"), dtype=np.uint8)][marco:]
    n = int(codes.size // 3)
    if n == 0:
        return [], 0
    tri = codes[:n * 3].reshape(n, 3).astype(np.int16)
    idx = tri[:, 0] * 16 + tri[:, 1] * 4 + tri[:, 2]
    idx[np.any(tri > 3, axis=1)] = -1                 # un codón con N nunca encaja
    return idx.tolist(), n


def _traducir(codigos) -> str:
    """Códigos de codón → proteína (``X`` donde había una base ambigua)."""
    return "".join(_TABLA[c] if c >= 0 else "X" for c in codigos)


def find_orfs(sequence: str, *, min_length: int = 90,
              require_start: bool = True,
              both_strands: bool = True,
              include_stop: bool = True) -> list[ORF]:
    """Encuentra los marcos abiertos de lectura de una secuencia de ADN.

    Parameters
    ----------
    sequence:
        ADN a analizar.
    min_length:
        Longitud mínima en **nucleótidos**. El valor por defecto (90 nt = 30
        aminoácidos) es el habitual: por debajo, casi todo lo que aparece es
        casualidad estadística, no genes.
    require_start:
        Si ``True`` (por defecto), el ORF debe empezar en ``ATG``. Si ``False``, se
        devuelve todo el tramo entre paradas — que es lo que hace ``getorf -find 0``
        de EMBOSS y sirve para buscar genes con inicios poco corrientes.
    both_strands:
        Buscar también en la hebra inversa (los seis marcos). Casi siempre sí.
    include_stop:
        Incluir el codón de parada en las coordenadas y la longitud.

    Returns
    -------
    list[ORF]
        Ordenados de mayor a menor: el ORF más largo suele ser el gen de interés.
    """
    seq = sequence.upper()
    if not seq:
        raise SequenceValueError("la secuencia está vacía.")
    if min_length < 3:
        raise SequenceValueError(f"min_length debe ser ≥3 nt (es {min_length}).")

    escanear = escanear_numpy if len(seq) >= UMBRAL_NUMPY else escanear_python
    salida: list[ORF] = []
    largo = len(seq)
    hebras = [("+", seq)] + ([("-", _revcomp(seq))] if both_strands else [])

    for hebra, s in hebras:                           # bucle por HEBRA (2)
        for marco in (0, 1, 2):                       # bucle por MARCO (3)
            codigos, n = escanear(s, marco)
            if n == 0:
                continue
            paradas = [i for i, v in enumerate(codigos) if v in _STOPS]
            inicios = ([i for i, v in enumerate(codigos) if v == _ATG]
                       if require_start else [])

            # tramos: desde justo después de una parada hasta la siguiente parada
            bordes = [-1] + paradas
            for b in range(len(bordes)):              # bucle por TRAMO (pocos)
                desde = bordes[b] + 1
                hay_parada = b < len(paradas)
                hasta = paradas[b] if hay_parada else n   # codón de parada (excl.)
                if hasta <= desde:
                    continue
                if require_start:
                    pos = bisect.bisect_left(inicios, desde)   # el primer ATG del tramo
                    if pos >= len(inicios) or inicios[pos] >= hasta:
                        continue
                    arranque = inicios[pos]
                else:
                    arranque = desde

                fin_codon = hasta + (1 if (hay_parada and include_stop) else 0)
                nt = (fin_codon - arranque) * 3
                if nt < min_length:
                    continue

                ini_local = marco + arranque * 3
                fin_local = marco + fin_codon * 3
                prot = _traducir(codigos[arranque:hasta])
                if hebra == "+":
                    ini, fin = ini_local, fin_local
                else:                                 # devolver coords sobre la directa
                    ini, fin = largo - fin_local, largo - ini_local
                salida.append(ORF(
                    start=ini, end=fin, strand=hebra,
                    frame=(marco + 1) * (1 if hebra == "+" else -1),
                    length=nt, protein=prot, has_stop=hay_parada))

    salida.sort(key=lambda o: (-o.length, o.start))
    return salida


def longest_orf(sequence: str, **kw) -> Optional[ORF]:
    """El ORF más largo — casi siempre, el gen que se busca."""
    orfs = find_orfs(sequence, **kw)
    return orfs[0] if orfs else None
