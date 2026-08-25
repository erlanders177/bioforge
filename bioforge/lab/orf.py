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

Cómo busca (regla de oro nº1)
-----------------------------
Cada codón se convierte en un número (``16·b₀ + 4·b₁ + b₂``), de modo que
«¿es un codón de parada?» pasa a ser una comparación entre enteros sobre todo el
array a la vez. Los ORFs salen de los tramos entre paradas consecutivas, calculados
con ``flatnonzero`` y ``searchsorted``. No hay un solo bucle por base ni por codón:
el único bucle recorre los **seis marcos**.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

from bioforge.core.biocore import SequenceValueError

_BASE = np.full(256, 255, dtype=np.uint8)
for _i, _b in enumerate(b"ACGT"):
    _BASE[_b] = _i
    _BASE[_b + 32] = _i

_COMPL = bytes.maketrans(b"ACGTacgtNn", b"TGCAtgcaNn")

# índices de codón: 16·b0 + 4·b1 + b2 con A=0, C=1, G=2, T=3
_ATG = 0 * 16 + 3 * 4 + 2          # 14
_STOPS = np.array([3 * 16 + 0 * 4 + 0,      # TAA = 48
                   3 * 16 + 0 * 4 + 2,      # TAG = 50
                   3 * 16 + 2 * 4 + 0],     # TGA = 56
                  dtype=np.int16)

_TABLA = ("KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF")


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


def _codones(codes: np.ndarray, marco: int) -> tuple[np.ndarray, int]:
    """Índices de codón del marco dado. Devuelve (array, nº de codones)."""
    resto = codes[marco:]
    n = resto.size // 3
    if n == 0:
        return np.empty(0, dtype=np.int16), 0
    tri = resto[:n * 3].reshape(n, 3).astype(np.int16)
    # una base inválida (N) contamina el codón: se marca como -1 y nunca encaja
    malo = np.any(tri > 3, axis=1)
    idx = tri[:, 0] * 16 + tri[:, 1] * 4 + tri[:, 2]
    idx[malo] = -1
    return idx, n


def _traducir(idx: np.ndarray) -> str:
    """Índices de codón → proteína, vectorizado con la tabla del código genético."""
    if idx.size == 0:
        return ""
    letras = np.frombuffer(_TABLA.encode("ascii"), dtype=np.uint8)
    salida = np.where(idx >= 0, letras[np.clip(idx, 0, 63)], ord("X"))
    return salida.astype(np.uint8).tobytes().decode("ascii")


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

    salida: list[ORF] = []
    largo = len(seq)
    hebras = [("+", seq)] + ([("-", _revcomp(seq))] if both_strands else [])

    for hebra, s in hebras:                           # bucle por HEBRA (2)
        codes = _BASE[np.frombuffer(s.encode("ascii"), dtype=np.uint8)]
        for marco in (0, 1, 2):                       # bucle por MARCO (3)
            idx, n = _codones(codes, marco)
            if n == 0:
                continue
            paradas = np.flatnonzero(np.isin(idx, _STOPS))
            inicios = np.flatnonzero(idx == _ATG) if require_start else None

            # tramos: desde justo después de una parada hasta la siguiente parada
            bordes = np.concatenate(([-1], paradas))
            for b in range(bordes.size):              # bucle por TRAMO (pocos)
                desde = int(bordes[b]) + 1
                hay_parada = b < paradas.size
                hasta = int(paradas[b]) if hay_parada else n   # codón de parada (excl.)
                if hasta <= desde:
                    continue
                if require_start:
                    # el primer ATG dentro del tramo
                    pos = np.searchsorted(inicios, desde)
                    if pos >= inicios.size or inicios[pos] >= hasta:
                        continue
                    arranque = int(inicios[pos])
                else:
                    arranque = desde

                fin_codon = hasta + (1 if (hay_parada and include_stop) else 0)
                nt = (fin_codon - arranque) * 3
                if nt < min_length:
                    continue

                ini_local = marco + arranque * 3
                fin_local = marco + fin_codon * 3
                prot = _traducir(idx[arranque:hasta])
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
