"""
bioforge/lab/primers.py — cebadores, temperatura de fusión y PCR *in silico*.

La **PCR** es la reacción que copia millones de veces un trozo concreto de ADN, y
es la base de casi todo el trabajo de laboratorio: diagnosticar una infección,
clonar un gen, hacer un test de paternidad. Para que funcione hacen falta dos
**cebadores** (*primers*): dos fragmentos cortos que marcan dónde empieza y dónde
acaba el trozo a copiar.

Diseñarlos mal es la causa nº1 de que una PCR falle, y el parámetro crítico es la
**temperatura de fusión (Tm)**: la temperatura a la que la mitad del cebador está
pegado a su molde. Si los dos cebadores tienen Tm muy distintas, uno se pega y el
otro no, y no se amplifica nada.

Cómo se calcula la Tm
---------------------
Se usa el modelo de **vecino más próximo** (Allawi & SantaLucia, 1997), que es el
estándar del campo. La idea: la estabilidad de una doble hélice no depende solo de
cuántas G y C hay, sino de **qué bases están juntas** — una ``GC`` seguida de otra
``GC`` aporta más que las mismas bases separadas. Se suman las contribuciones
termodinámicas (ΔH, ΔS) de cada pareja consecutiva y se aplica:

    ``Tm = 1000·ΔH / (ΔS + R·ln(C/4)) − 273.15``

con corrección por sal (SantaLucia, 1998). La regla casera «2·(A+T) + 4·(G+C)»
también está disponible como ``tm_wallace``, pero solo sirve para cebadores muy
cortos: para cualquier cosa seria, el vecino más próximo es lo correcto.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import numpy as np

from bioforge.core.biocore import SequenceValueError

_COMPL = bytes.maketrans(b"ACGTacgt", b"TGCAtgca")

R_GAS = 1.987              # constante de los gases, cal/(mol·K)

# Parámetros de vecino más próximo (Allawi & SantaLucia, 1997): (ΔH kcal/mol, ΔS cal/mol·K)
_NN = {
    "AA": (-7.9, -22.2), "AT": (-7.2, -20.4), "TA": (-7.2, -21.3),
    "CA": (-8.5, -22.7), "GT": (-8.4, -22.4), "CT": (-7.8, -21.0),
    "GA": (-8.2, -22.2), "CG": (-10.6, -27.2), "GC": (-9.8, -24.4),
    "GG": (-8.0, -19.9),
}
_INIT_AT = (2.3, 4.1)      # iniciación con extremo A o T
_INIT_GC = (0.1, -2.8)     # iniciación con extremo G o C
_SIMETRIA = (0.0, -1.4)    # corrección si la secuencia es autocomplementaria


def _revcomp(s: str) -> str:
    return s.encode("ascii").translate(_COMPL)[::-1].decode("ascii")


def _par_nn(dinuc: str) -> tuple[float, float]:
    """Parámetros del dinucleótido, usando la simetría 5'XY3' ≡ 5'comp(Y)comp(X)3'."""
    if dinuc in _NN:
        return _NN[dinuc]
    equivalente = _revcomp(dinuc)
    if equivalente in _NN:
        return _NN[equivalente]
    raise SequenceValueError(f"dinucleótido no válido: {dinuc!r} (¿hay bases que no son ACGT?)")


def tm_nn(seq: str, *, conc_primer: float = 25.0, conc_molde: float = 25.0,
          na: float = 50.0) -> float:
    """Temperatura de fusión por **vecino más próximo** (el método correcto).

    Parameters
    ----------
    seq:
        El cebador (solo A, C, G, T).
    conc_primer, conc_molde:
        Concentraciones en **nanomolar**. Los valores por defecto (25 nM) son los
        habituales en PCR y coinciden con los de Biopython, para poder contrastar.
    na:
        Concentración de sodio en **milimolar** (50 mM es lo típico de un tampón
        de PCR). La sal estabiliza la hélice: más sal, más Tm.

    Returns
    -------
    float
        Temperatura en grados Celsius.
    """
    s = seq.upper().strip()
    if len(s) < 2:
        raise SequenceValueError(f"el cebador necesita al menos 2 bases (tiene {len(s)}).")
    if set(s) - set("ACGT"):
        raise SequenceValueError(
            "la Tm de vecino más próximo solo admite A, C, G y T "
            f"(hay: {sorted(set(s) - set('ACGT'))}).")

    dh, ds = 0.0, 0.0
    for i in range(len(s) - 1):                       # bucle por PAREJA (no por base suelta)
        h, sdd = _par_nn(s[i:i + 2])
        dh += h
        ds += sdd

    for extremo in (s[0], s[-1]):                     # iniciación en cada punta
        h, sdd = _INIT_GC if extremo in "GC" else _INIT_AT
        dh += h
        ds += sdd

    autocompl = s == _revcomp(s)
    if autocompl:
        dh += _SIMETRIA[0]
        ds += _SIMETRIA[1]

    # corrección por sal (SantaLucia 1998): entra en la entropía
    mon = na / 1000.0
    if mon > 0:
        ds += 0.368 * (len(s) - 1) * math.log(mon)

    # concentración efectiva
    if autocompl:
        k = conc_primer * 1e-9
    else:
        k = (conc_primer - conc_molde / 2.0) * 1e-9
    if k <= 0:
        raise SequenceValueError(
            "la concentración efectiva sale ≤0: revisa conc_primer y conc_molde.")

    return (1000.0 * dh) / (ds + R_GAS * math.log(k)) - 273.15


def tm_wallace(seq: str) -> float:
    """Regla casera ``2·(A+T) + 4·(G+C)``.

    Solo vale para cebadores muy cortos (<14 bases). Se incluye porque sigue
    apareciendo en protocolos antiguos, pero para diseñar de verdad usa
    :func:`tm_nn`: esta regla ignora en qué ORDEN están las bases, que es
    justo lo que determina la estabilidad.
    """
    s = seq.upper()
    return 2.0 * (s.count("A") + s.count("T")) + 4.0 * (s.count("G") + s.count("C"))


def gc_percent(seq: str) -> float:
    """Porcentaje de G+C, que gobierna la estabilidad general."""
    s = seq.upper()
    return 100.0 * (s.count("G") + s.count("C")) / max(len(s), 1)


class Primer(NamedTuple):
    """Un cebador propuesto, con lo que hace falta para juzgarlo."""

    sequence: str
    start: int               # 0-based sobre la secuencia original
    end: int
    strand: str              # "+" (directo) o "-" (inverso)
    tm: float
    gc: float
    warnings: list[str]

    @property
    def length(self) -> int:
        return len(self.sequence)

    def __repr__(self) -> str:                        # pragma: no cover - cosmético
        aviso = f" ⚠{len(self.warnings)}" if self.warnings else ""
        return (f"Primer({self.sequence}, {self.strand}, Tm {self.tm:.1f}°C, "
                f"GC {self.gc:.0f}%{aviso})")


def _avisos(s: str, tm: float, gc: float) -> list[str]:
    """Las pegas clásicas que hacen fallar una PCR."""
    av = []
    if not 40.0 <= gc <= 60.0:
        av.append(f"GC fuera del 40-60% ({gc:.0f}%)")
    if not 55.0 <= tm <= 65.0:
        av.append(f"Tm fuera del rango cómodo 55-65°C ({tm:.1f}°C)")
    if s[-1] not in "GC":
        av.append("no termina en G o C (la 'pinza GC' ayuda a anclar el extremo 3')")
    for base in "ACGT":
        if base * 4 in s:
            av.append(f"repetición de {base}×4 o más (puede deslizarse)")
            break
    if _revcomp(s[-5:]) in s:
        av.append("el extremo 3' se aparea consigo mismo (riesgo de dímero)")
    return av


def design_primers(sequence: str, *, target_tm: float = 60.0,
                   min_len: int = 18, max_len: int = 27,
                   amplicon_min: int = 100) -> Optional[tuple[Primer, Primer]]:
    """Propone una pareja de cebadores para amplificar la secuencia entera.

    Busca en los extremos el cebador cuya Tm más se acerque a ``target_tm``, y
    prefiere los que no tienen pegas. Es un diseñador **sencillo y honesto**: no
    pretende sustituir a Primer3, que además modela estructuras secundarias y
    dímeros con termodinámica completa.

    Returns
    -------
    tuple[Primer, Primer] | None
        (directo, inverso), o ``None`` si la secuencia es demasiado corta.
    """
    s = sequence.upper()
    if len(s) < amplicon_min:
        return None

    def mejor(desde_inicio: bool) -> Optional[Primer]:
        candidatos = []
        for L in range(min_len, max_len + 1):         # bucle por LONGITUD (pocas)
            if desde_inicio:
                sub, ini = s[:L], 0
            else:
                sub, ini = _revcomp(s[-L:]), len(s) - L
            if set(sub) - set("ACGT"):
                continue
            try:
                tm = tm_nn(sub)
            except SequenceValueError:
                continue
            gc = gc_percent(sub)
            av = _avisos(sub, tm, gc)
            candidatos.append(Primer(sub, ini, ini + L,
                                     "+" if desde_inicio else "-", tm, gc, av))
        if not candidatos:
            return None
        # primero los que no tienen pegas; dentro de eso, el más cercano al objetivo
        return min(candidatos, key=lambda p: (len(p.warnings), abs(p.tm - target_tm)))

    d, i = mejor(True), mejor(False)
    return (d, i) if d and i else None


class Amplicon(NamedTuple):
    """El producto que saldría de una PCR."""

    start: int
    end: int
    length: int
    sequence: str


def pcr(sequence: str, forward: str, reverse: str, *,
        max_mismatches: int = 0, circular: bool = False) -> list[Amplicon]:
    """PCR *in silico*: qué fragmento(s) amplificaría esa pareja de cebadores.

    Comprueba antes de ir al laboratorio si los cebadores pegan donde se espera —
    y, sobre todo, si pegan en **más sitios** de los previstos, que es lo que
    produce las bandas inesperadas en el gel.

    Parameters
    ----------
    max_mismatches:
        Cuántas bases se permiten diferentes al aparear. 0 = apareamiento perfecto.
    circular:
        Para plásmidos: permite productos que cruzan el origen.
    """
    s = sequence.upper()
    f, r = forward.upper(), reverse.upper()
    if not f or not r:
        raise SequenceValueError("hacen falta los dos cebadores.")
    diana = s + s[:max(len(f), len(r))] if circular else s

    def pegadas(patron: str) -> list[int]:
        """Posiciones donde el patrón aparea (vectorizado, con tolerancia a fallos)."""
        n, k = len(diana), len(patron)
        if k > n:
            return []
        a = np.frombuffer(diana.encode("ascii"), dtype=np.uint8)
        p = np.frombuffer(patron.encode("ascii"), dtype=np.uint8)
        from numpy.lib.stride_tricks import sliding_window_view
        ventanas = sliding_window_view(a, k)
        fallos = (ventanas != p).sum(axis=1)
        return np.flatnonzero(fallos <= max_mismatches).tolist()

    inicios = pegadas(f)                              # el directo, tal cual
    finales = pegadas(_revcomp(r))                    # el inverso, sobre la hebra de arriba

    salida: list[Amplicon] = []
    for a in inicios:                                 # bucle por SITIO (pocos)
        for b in finales:
            fin = b + len(r)
            if fin <= a:
                continue
            largo = fin - a
            if largo > len(s) * (2 if circular else 1):
                continue
            trozo = diana[a:fin]
            salida.append(Amplicon(a % len(s), fin % len(s) if circular else fin,
                                   largo, trozo))
    salida.sort(key=lambda x: x.length)
    return salida
