"""
bioforge/lab/restriction.py — enzimas de restricción: dónde corta el ADN.

Las **enzimas de restricción** son tijeras moleculares: cada una reconoce una
secuencia corta concreta y corta el ADN justo ahí. Son la herramienta básica de
la clonación —lo que permite recortar un gen de un sitio y pegarlo en otro— y lo
primero que hace cualquiera en un laboratorio de biología molecular al planear un
experimento es preguntarse *«¿qué enzimas cortan mi secuencia, y en cuántos
trozos?»*.

Qué hace este módulo
--------------------
* ``find_sites``   — dónde corta una enzima (o todas) en una secuencia.
* ``digest``       — los **fragmentos** que salen del corte, con sus tamaños.
* ``unique_cutters`` — qué enzimas cortan **una sola vez**: las valiosas para clonar.
* ``gel``          — simula un gel de electroforesis: los fragmentos ordenados por
  tamaño, que es como se ven en el laboratorio.

Cómo busca (regla de oro nº1)
-----------------------------
Los sitios de reconocimiento llevan **códigos ambiguos** de la IUPAC: ``N`` es
cualquier base, ``R`` es A o G, ``Y`` es C o T… Buscar eso con comparaciones de
texto obligaría a expandir cada patrón en decenas de variantes.

En su lugar, cada base se codifica como una **máscara de bits** (A=1, C=2, G=4,
T=8), de modo que un código ambiguo es simplemente la unión de sus bases: ``R`` =
A|G = 5, ``N`` = 15. Entonces «la base *x* encaja en el código *c*» es una sola
operación: ``x & c != 0``. Con ``sliding_window_view`` se comprueban **todas las
posiciones a la vez**, sin un solo bucle por base.

Sobre la tabla de enzimas
-------------------------
Se incluyen ~60 enzimas: las de uso corriente en un laboratorio (el catálogo que
de verdad se usa), no las 4000 de REBASE. Los datos de reconocimiento y corte
proceden de REBASE (Roberts et al., *Nucleic Acids Research*), la base de
referencia del campo. Es un subconjunto declarado, no una omisión.
"""

from __future__ import annotations

from typing import Iterable, NamedTuple, Optional, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from bioforge.core.biocore import SequenceValueError

# ── código IUPAC como máscaras de bits ───────────────────────────────────────
_BITS = {"A": 1, "C": 2, "G": 4, "T": 8, "U": 8,
         "R": 1 | 4, "Y": 2 | 8, "S": 2 | 4, "W": 1 | 8,
         "K": 4 | 8, "M": 1 | 2, "B": 2 | 4 | 8, "D": 1 | 4 | 8,
         "H": 1 | 2 | 8, "V": 1 | 2 | 4, "N": 15}

_LUT = np.zeros(256, dtype=np.uint8)
for _c, _v in _BITS.items():
    _LUT[ord(_c)] = _v
    _LUT[ord(_c.lower())] = _v

_COMPL = bytes.maketrans(b"ACGTRYSWKMBDHVNacgtryswkmbdhvn",
                         b"TGCAYRSWMKVHDBNtgcayrswmkvhdbn")


def _revcomp(s: str) -> str:
    return s.encode("ascii").translate(_COMPL)[::-1].decode("ascii")


class Enzyme(NamedTuple):
    """Una enzima de restricción.

    Attributes
    ----------
    name:
        Nombre comercial (EcoRI, BamHI…).
    site:
        Secuencia que reconoce, en código IUPAC.
    cut5:
        Bases del sitio que quedan a la izquierda del corte en la hebra **de
        arriba**. En ``G^AATTC`` (EcoRI) vale 1.
    cut3:
        Lo mismo en la hebra **de abajo**, contando desde el mismo extremo. En
        EcoRI vale 5.

    La diferencia ``cut3 - cut5`` es el **saliente** (*overhang*): positivo = extremo
    cohesivo 5', cero = extremo romo, negativo = cohesivo 3'. Es lo que determina
    si dos fragmentos se pueden pegar entre sí.
    """

    name: str
    site: str
    cut5: int
    cut3: int

    @property
    def is_palindromic(self) -> bool:
        return self.site == _revcomp(self.site)

    @property
    def overhang(self) -> int:
        return self.cut3 - self.cut5

    @property
    def ends(self) -> str:
        o = self.overhang
        return "romo" if o == 0 else ("cohesivo 5'" if o > 0 else "cohesivo 3'")

    def pretty_site(self) -> str:
        """El sitio con el corte marcado: ``G^AATTC``."""
        return self.site[:self.cut5] + "^" + self.site[self.cut5:]


# ── catálogo (datos de REBASE) ───────────────────────────────────────────────
# (nombre, sitio, cut5, cut3)
_CATALOGO: tuple[tuple[str, str, int, int], ...] = (
    ("EcoRI", "GAATTC", 1, 5), ("BamHI", "GGATCC", 1, 5),
    ("HindIII", "AAGCTT", 1, 5), ("NotI", "GCGGCCGC", 2, 6),
    ("XhoI", "CTCGAG", 1, 5), ("SalI", "GTCGAC", 1, 5),
    ("PstI", "CTGCAG", 5, 1), ("SmaI", "CCCGGG", 3, 3),
    ("KpnI", "GGTACC", 5, 1), ("SacI", "GAGCTC", 5, 1),
    ("XbaI", "TCTAGA", 1, 5), ("SpeI", "ACTAGT", 1, 5),
    ("NcoI", "CCATGG", 1, 5), ("NdeI", "CATATG", 2, 4),
    ("BglII", "AGATCT", 1, 5), ("EcoRV", "GATATC", 3, 3),
    ("HaeIII", "GGCC", 2, 2), ("AluI", "AGCT", 2, 2),
    ("TaqI", "TCGA", 1, 3), ("HpaII", "CCGG", 1, 3),
    ("MspI", "CCGG", 1, 3), ("Sau3AI", "GATC", 0, 4),
    ("MboI", "GATC", 0, 4), ("ApaI", "GGGCCC", 5, 1),
    ("ClaI", "ATCGAT", 2, 4), ("NheI", "GCTAGC", 1, 5),
    ("SphI", "GCATGC", 5, 1), ("StuI", "AGGCCT", 3, 3),
    ("AatII", "GACGTC", 5, 1), ("AvrII", "CCTAGG", 1, 5),
    ("EagI", "CGGCCG", 1, 5), ("DraI", "TTTAAA", 3, 3),
    ("ScaI", "AGTACT", 3, 3), ("SspI", "AATATT", 3, 3),
    ("PvuII", "CAGCTG", 3, 3), ("PvuI", "CGATCG", 4, 2),
    ("BsrGI", "TGTACA", 1, 5), ("AflII", "CTTAAG", 1, 5),
    ("MfeI", "CAATTG", 1, 5), ("NsiI", "ATGCAT", 5, 1),
    ("SacII", "CCGCGG", 4, 2), ("BssHII", "GCGCGC", 1, 5),
    ("MluI", "ACGCGT", 1, 5), ("AscI", "GGCGCGCC", 2, 6),
    ("PacI", "TTAATTAA", 5, 3), ("FseI", "GGCCGGCC", 6, 2),
    ("SbfI", "CCTGCAGG", 6, 2), ("SwaI", "ATTTAAAT", 4, 4),
    ("PmeI", "GTTTAAAC", 4, 4), ("SrfI", "GCCCGGGC", 4, 4),
    ("NruI", "TCGCGA", 3, 3), ("SnaBI", "TACGTA", 3, 3),
    ("XmaI", "CCCGGG", 1, 5), ("BspEI", "TCCGGA", 1, 5),
    ("AgeI", "ACCGGT", 1, 5), ("NgoMIV", "GCCGGC", 1, 5),
    ("HincII", "GTYRAC", 3, 3), ("HinfI", "GANTC", 1, 4),
    ("BstEII", "GGTNACC", 1, 6), ("DdeI", "CTNAG", 1, 4),
    ("NlaIII", "CATG", 4, 0), ("RsaI", "GTAC", 2, 2),
    ("HhaI", "GCGC", 3, 1), ("MseI", "TTAA", 1, 3),
)

ENZYMES: dict[str, Enzyme] = {n: Enzyme(n, s, c5, c3) for n, s, c5, c3 in _CATALOGO}


def get_enzyme(name: str) -> Enzyme:
    """Busca una enzima por nombre (sin distinguir mayúsculas)."""
    if name in ENZYMES:
        return ENZYMES[name]
    for n, e in ENZYMES.items():
        if n.lower() == name.lower():
            return e
    raise SequenceValueError(
        f"enzima desconocida: {name!r}. Hay {len(ENZYMES)} en el catálogo; "
        f"p. ej. {', '.join(list(ENZYMES)[:6])}…")


class Site(NamedTuple):
    """Un sitio de corte encontrado."""

    enzyme: str
    position: int            # posición del CORTE en la hebra de arriba (0-based)
    site_start: int          # dónde empieza el sitio reconocido (0-based)
    strand: str              # "+" o "-" (relevante solo si el sitio no es palindrómico)


def _buscar_patron(codes: np.ndarray, patron: str) -> np.ndarray:
    """Posiciones donde encaja ``patron`` (con códigos IUPAC). Vectorizado."""
    k = len(patron)
    if k == 0 or codes.size < k:
        return np.empty(0, dtype=np.int64)
    mask = _LUT[np.frombuffer(patron.encode("ascii"), dtype=np.uint8)]
    ventanas = sliding_window_view(codes, k)          # (n-k+1, k), sin copiar
    # encaja si TODAS las posiciones comparten al menos un bit con el código
    return np.flatnonzero(np.all((ventanas & mask) != 0, axis=1))


def find_sites(sequence: str, enzymes: Optional[Iterable[str]] = None, *,
               circular: bool = False) -> list[Site]:
    """Busca los sitios de corte de una o varias enzimas.

    Parameters
    ----------
    sequence:
        ADN a analizar.
    enzymes:
        Nombres de enzimas. Si es ``None``, se prueban **todas** las del catálogo.
    circular:
        Si el ADN es circular (un plásmido), también se buscan los sitios que
        cruzan el punto de unión entre el final y el principio.

    Returns
    -------
    list[Site]
        Ordenados por posición de corte.
    """
    seq = sequence.upper()
    if not seq:
        raise SequenceValueError("la secuencia está vacía.")
    nombres = list(enzymes) if enzymes is not None else list(ENZYMES)
    encontrados: list[Site] = []

    largo = len(seq)
    buscar_en = seq + seq[:30] if circular else seq   # 30 = sitio más largo con margen
    codes = _LUT[np.frombuffer(buscar_en.encode("ascii"), dtype=np.uint8)]

    for nombre in nombres:                            # bucle por ENZIMA (pocas)
        enz = get_enzyme(nombre)
        hebras = [("+", enz.site, enz.cut5)]
        if not enz.is_palindromic:
            # sitio no palindrómico: también corta leyendo la hebra de abajo
            hebras.append(("-", _revcomp(enz.site), len(enz.site) - enz.cut5))
        vistos = set()
        for hebra, patron, desplazamiento in hebras:
            for inicio in _buscar_patron(codes, patron).tolist():
                if inicio >= largo:                   # el envoltorio circular ya se contó
                    continue
                corte = (inicio + desplazamiento) % largo if circular \
                    else inicio + desplazamiento
                if not circular and corte > largo:
                    continue
                clave = (inicio % largo, hebra)
                if clave in vistos:
                    continue
                vistos.add(clave)
                encontrados.append(Site(enz.name, int(corte), int(inicio % largo), hebra))
    encontrados.sort(key=lambda s: (s.position, s.enzyme))
    return encontrados


class Fragment(NamedTuple):
    """Un trozo de ADN resultante del corte."""

    start: int
    end: int
    length: int
    sequence: str


class Digestion(NamedTuple):
    """Resultado de digerir una secuencia con una o varias enzimas."""

    enzymes: list[str]
    sites: list[Site]
    fragments: list[Fragment]
    circular: bool

    @property
    def n_cuts(self) -> int:
        return len(self.sites)

    def sizes(self) -> list[int]:
        """Tamaños de los fragmentos, de mayor a menor (como se leen en un gel)."""
        return sorted((f.length for f in self.fragments), reverse=True)

    def __repr__(self) -> str:                        # pragma: no cover - cosmético
        return (f"Digestion({'+'.join(self.enzymes)}: {self.n_cuts} cortes, "
                f"{len(self.fragments)} fragmentos)")


def digest(sequence: str, enzymes, *, circular: bool = False) -> Digestion:
    """Corta la secuencia con las enzimas dadas y devuelve los fragmentos.

    Con ADN **lineal**, *n* cortes dan *n+1* fragmentos; con ADN **circular**
    (un plásmido) dan exactamente *n*. Esa diferencia es justo la que se usa en el
    laboratorio para saber si un plásmido está cerrado o roto.
    """
    if isinstance(enzymes, str):
        enzymes = [enzymes]
    nombres = [get_enzyme(e).name for e in enzymes]
    seq = sequence.upper()
    sitios = find_sites(seq, nombres, circular=circular)
    cortes = sorted({s.position for s in sitios})

    fragmentos: list[Fragment] = []
    if not cortes:
        fragmentos.append(Fragment(0, len(seq), len(seq), seq))
    elif circular:
        for i, c in enumerate(cortes):                # bucle por CORTE (pocos)
            sig = cortes[(i + 1) % len(cortes)]
            if sig > c:
                trozo = seq[c:sig]
            else:                                     # el que cruza el origen
                trozo = seq[c:] + seq[:sig]
            fragmentos.append(Fragment(c, sig, len(trozo), trozo))
    else:
        bordes = [0] + cortes + [len(seq)]
        for a, b in zip(bordes[:-1], bordes[1:], strict=True):
            fragmentos.append(Fragment(a, b, b - a, seq[a:b]))
    return Digestion(enzymes=nombres, sites=sitios, fragments=fragmentos,
                     circular=circular)


def unique_cutters(sequence: str, *, circular: bool = False) -> list[str]:
    """Enzimas que cortan **exactamente una vez**: las útiles para clonar.

    Son las valiosas: permiten abrir la secuencia por un único punto conocido, sin
    romperla en trozos. Encontrarlas a mano es tedioso; es de las consultas que más
    se repiten en un laboratorio.
    """
    sitios = find_sites(sequence, circular=circular)
    cuenta: dict[str, int] = {}
    for s in sitios:
        cuenta[s.enzyme] = cuenta.get(s.enzyme, 0) + 1
    return sorted(n for n, c in cuenta.items() if c == 1)


def gel(digestion: Digestion, *, escalera: Sequence[int] = (
        10000, 8000, 6000, 5000, 4000, 3000, 2000, 1500, 1000, 750, 500, 250)) -> str:
    """Dibuja un gel de electroforesis en texto: los fragmentos por tamaño.

    En un gel real los fragmentos migran según su tamaño —los pequeños llegan más
    lejos— y se comparan contra una *escalera* de tamaños conocidos. Esto reproduce
    esa lectura, que es como se comprueba en el laboratorio si la digestión salió
    como se esperaba.
    """
    tam = digestion.sizes()
    if not tam:
        return "(sin fragmentos)"
    tope = max(max(tam), max(escalera))
    filas = ["  escalera        muestra", "  " + "-" * 30]
    for marca in escalera:                            # bucle por BANDA (pocas)
        aqui = [t for t in tam if abs(t - marca) <= marca * 0.12]
        filas.append(f"  {marca:>6} ──   " +
                     ("███  " + ", ".join(f"{t} pb" for t in aqui) if aqui else ""))
    sueltos = [t for t in tam
               if not any(abs(t - m) <= m * 0.12 for m in escalera)]
    if sueltos:
        filas.append("  " + "-" * 30)
        filas.append("  fuera de la escalera: " + ", ".join(f"{t} pb" for t in sueltos))
    filas.append(f"  total: {len(tam)} fragmentos, {sum(tam)} pb"
                 + (f" (máx {tope} pb)" if False else ""))
    return "\n".join(filas)
