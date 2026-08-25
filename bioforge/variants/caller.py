"""
bioforge/variants/caller.py — de la evidencia apilada a las mutaciones (VCF).

Es el eslabón que cerraba la tubería: BioForge ya sabía **mapear** lecturas contra
un genoma, pero no decía **qué cambió**. Aquí se decide qué diferencias son
mutaciones reales y cuáles son ruido de secuenciación, y se escribe el resultado
en VCF, el formato estándar del campo.

El modelo estadístico (y por qué este)
--------------------------------------
Para cada posición se contrasta una hipótesis contra otra, con una **razón de
verosimilitudes** binomial:

* **H₀ — solo error.** Las lecturas discrepantes son fallos del secuenciador, que
  ocurren con probabilidad ``error_rate`` (por defecto 1 %, ~Q20).
* **H₁ — variante real.** La base alternativa está presente de verdad, con la
  frecuencia observada ``f̂ = k/n``.

  ``log₁₀ LR = k·log₁₀(f̂/ε) + (n−k)·log₁₀((1−f̂)/(1−ε))``

y ``QUAL = 10 · log₁₀ LR``, que es exactamente la escala Phred que usa el VCF: 20
significa 100 veces más probable que sea variante que error; 30, mil veces.

El coeficiente binomial se cancela al dividir, así que no hacen falta funciones
especiales: son logaritmos sobre vectores NumPy. Todo el cálculo de SNVs es
vectorizado sobre el genoma entero, sin un solo bucle por posición.

Qué NO hace (honestidad, regla nº9)
-----------------------------------
Es un llamador **haploide de una muestra**, pensado para virus, bacterias y
amplicones — no compite con GATK ni FreeBayes en genomas diploides. En concreto:

* no modela genotipos diploides (0/1, 1/1): asume una población haploide o clonal;
* no integra la calidad por base individual (usa una tasa de error única);
* no aplica filtros de sesgo de hebra ni de posición en la lectura.

**Los indels largos pueden salir partidos, y la causa está aguas arriba.** El
alineador de BioForge usa un modelo de hueco **lineal** (``GAP = −2`` por base, ver
``align/pairwise.py``): un hueco de 5 pb cuesta lo mismo entero que partido en 3+2,
así que nada empuja al alineador a mantenerlo junto. Medido: una deleción de 5 pb se
llama como 3 pb + 2 pb, y una inserción de 4 pb como 1 + 3. Las SNVs no se ven
afectadas (medido: 0 falsos positivos con 1 % de error y 40× de profundidad).

El arreglo correcto es un modelo de hueco **afín** (abrir caro, extender barato) en
el alineador, no un parche en el llamador: es un cambio del núcleo que afecta al
mapeo y a sus benchmarks, y se aborda aparte. Mientras tanto, el indel se detecta
—queda registrado como evento real— pero sus coordenadas exactas pueden estar
repartidas.
"""

from __future__ import annotations

from typing import Iterable, NamedTuple, Optional, Sequence

import numpy as np

from bioforge.core.biocore import SequenceValueError
from bioforge.variants.pileup import BASES, DEL, Pileup, _CODE

_QUAL_MAX = 5000.0


class Variant(NamedTuple):
    """Una mutación llamada sobre la referencia.

    Las coordenadas son **1-based** (como el VCF), no como los índices de Python.
    """

    contig: str
    pos: int                 # 1-based, la del VCF
    ref: str
    alt: str
    qual: float              # Phred: 10·log10(razón de verosimilitudes)
    depth: int               # lecturas que cubren la posición
    alt_count: int           # lecturas que apoyan la alternativa
    af: float                # fracción alélica = alt_count / depth
    kind: str                # "SNV" | "INS" | "DEL"

    def to_vcf(self) -> str:
        """Serializa como una línea de datos VCF."""
        info = f"DP={self.depth};AC={self.alt_count};AF={self.af:.4f};TYPE={self.kind}"
        return (f"{self.contig}\t{self.pos}\t.\t{self.ref}\t{self.alt}\t"
                f"{self.qual:.1f}\tPASS\t{info}")

    def __repr__(self) -> str:                       # pragma: no cover - cosmético
        return (f"Variant({self.contig}:{self.pos} {self.ref}>{self.alt} "
                f"{self.kind} AF={self.af:.2f} DP={self.depth} Q={self.qual:.0f})")


def _phred_lr(k: np.ndarray, n: np.ndarray, error_rate: float) -> np.ndarray:
    """Razón de verosimilitudes binomial en escala Phred, vectorizada.

    ``k`` observaciones alternativas de ``n`` totales. Devuelve ``10·log₁₀ LR``
    comparando «variante a frecuencia k/n» contra «solo error a tasa ε».
    """
    k = k.astype(np.float64)
    n = n.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(n > 0, k / np.maximum(n, 1.0), 0.0)
        # cada término se anula cuando su contador es 0 (evita 0·log 0 = nan)
        t1 = np.where(k > 0, k * np.log10(np.maximum(f, 1e-300) / error_rate), 0.0)
        t2 = np.where(n - k > 0,
                      (n - k) * np.log10(np.maximum(1.0 - f, 1e-300) / (1.0 - error_rate)),
                      0.0)
    lr = 10.0 * (t1 + t2)
    return np.clip(np.nan_to_num(lr, nan=0.0, posinf=_QUAL_MAX), 0.0, _QUAL_MAX)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Tramos contiguos de ``True`` en una máscara → [(inicio, fin_exclusivo)].

    Vectorizado con ``diff``/``flatnonzero``; el bucle final recorre TRAMOS (pocos).
    """
    if not mask.any():
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1) + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(mask.size)
    return list(zip(starts, ends, strict=True))


def call_variants(pile: Pileup, reference: str, *,
                  min_depth: int = 5,
                  min_af: float = 0.2,
                  min_qual: float = 20.0,
                  min_alt_count: int = 2,
                  error_rate: float = 0.01,
                  indels: bool = True) -> list[Variant]:
    """Llama variantes (SNVs e indels) a partir de la evidencia apilada.

    Parameters
    ----------
    pile:
        Salida de :func:`bioforge.variants.pileup`.
    reference:
        Secuencia del contig, como cadena.
    min_depth:
        Profundidad mínima para considerar una posición. Por debajo no hay
        evidencia suficiente y se calla, que es lo honesto.
    min_af:
        Fracción alélica mínima. 0.2 va bien para virus/bacterias; súbelo si
        esperas una población clonal, bájalo para detectar minoritarias.
    min_qual:
        Calidad Phred mínima de la razón de verosimilitudes.
    min_alt_count:
        Lecturas alternativas mínimas: evita llamar variantes por una sola lectura.
    error_rate:
        Tasa de error asumida del secuenciador (0.01 ≈ Q20). Súbela para nanoporo.
    indels:
        Si ``False``, solo llama sustituciones.

    Returns
    -------
    list[Variant]
        Ordenadas por posición.
    """
    if len(reference) < pile.counts.shape[0]:
        raise SequenceValueError(
            f"la referencia ({len(reference)} pb) es más corta que el pileup "
            f"({pile.counts.shape[0]} pb): ¿es el contig correcto?")
    if not 0.0 < error_rate < 1.0:
        raise SequenceValueError(f"error_rate debe estar en (0,1), es {error_rate}.")

    counts = pile.counts
    L = counts.shape[0]
    if L == 0:
        return []

    ref_arr = np.frombuffer(reference[:L].upper().encode("ascii"), dtype=np.uint8)
    ref_code = _CODE[ref_arr].astype(np.int64)

    acgt = counts[:, :4]
    prof = acgt.sum(axis=1) + counts[:, DEL]         # lecturas que cubren la posición
    idx = np.arange(L)

    # ── sustituciones (SNV): todo vectorizado sobre el genoma entero ──────────
    sin_ref = acgt.copy()
    valido_ref = ref_code < 4                        # posiciones con base ACGT en la ref
    sin_ref[idx[valido_ref], ref_code[valido_ref]] = 0
    alt_code = sin_ref.argmax(axis=1)
    alt_count = sin_ref[idx, alt_code]

    prof_acgt = acgt.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        af = np.where(prof_acgt > 0, alt_count / np.maximum(prof_acgt, 1), 0.0)
    qual = _phred_lr(alt_count, prof_acgt, error_rate)

    pasa = (valido_ref & (prof >= min_depth) & (alt_count >= min_alt_count)
            & (af >= min_af) & (qual >= min_qual))

    variantes: list[Variant] = []
    for p in np.flatnonzero(pasa).tolist():          # bucle por VARIANTE (pocas)
        variantes.append(Variant(
            contig=pile.contig, pos=p + 1,
            ref=chr(ref_arr[p]), alt=BASES[alt_code[p]],
            qual=float(qual[p]), depth=int(prof[p]),
            alt_count=int(alt_count[p]), af=float(af[p]), kind="SNV"))

    if not indels:
        return sorted(variantes, key=lambda v: v.pos)

    # ── deleciones: tramos contiguos con suficiente apoyo ─────────────────────
    del_count = counts[:, DEL]
    with np.errstate(divide="ignore", invalid="ignore"):
        del_af = np.where(prof > 0, del_count / np.maximum(prof, 1), 0.0)
    del_qual = _phred_lr(del_count, prof, error_rate)
    del_mask = ((prof >= min_depth) & (del_count >= min_alt_count)
                & (del_af >= min_af) & (del_qual >= min_qual))

    for ini, fin in _runs(del_mask):
        if ini == 0:                                 # el VCF necesita una base ancla previa
            continue
        ancla = reference[ini - 1]
        soporte = int(del_count[ini:fin].min())
        variantes.append(Variant(
            contig=pile.contig, pos=ini,             # 1-based de la base ancla
            ref=ancla + reference[ini:fin], alt=ancla,
            qual=float(del_qual[ini:fin].min()),
            depth=int(prof[ini:fin].min()),
            alt_count=soporte,
            af=float(del_af[ini:fin].min()), kind="DEL"))

    # ── inserciones: vienen aparte del pileup (no ocupan sitio en la ref) ─────
    for (r, seq), n_ins in pile.insertions.items():  # bucle por EVENTO (pocos)
        if r == 0 or r > L:
            continue
        cobertura = int(prof[min(r, L - 1)])
        if cobertura < min_depth or n_ins < min_alt_count:
            continue
        frac = n_ins / cobertura if cobertura else 0.0
        if frac < min_af:
            continue
        q = float(_phred_lr(np.array([n_ins]), np.array([cobertura]), error_rate)[0])
        if q < min_qual:
            continue
        ancla = reference[r - 1]
        variantes.append(Variant(
            contig=pile.contig, pos=r, ref=ancla, alt=ancla + seq,
            qual=q, depth=cobertura, alt_count=int(n_ins),
            af=float(frac), kind="INS"))

    return sorted(variantes, key=lambda v: (v.pos, v.kind))


def write_vcf(variants: Iterable[Variant],
              contigs: Optional[Sequence[tuple[str, int]]] = None,
              *, sample: str = "muestra", source: str = "BioForge") -> str:
    """Genera un VCF 4.2 completo (cabecera + registros) como texto.

    ``contigs`` es una lista de ``(nombre, longitud)`` para declarar en la cabecera;
    si se omite, se deducen los nombres de las propias variantes.
    """
    variants = list(variants)
    lineas = [
        "##fileformat=VCFv4.2",
        f"##source={source}",
        '##INFO=<ID=DP,Number=1,Type=Integer,Description="Profundidad total">',
        '##INFO=<ID=AC,Number=1,Type=Integer,Description="Lecturas que apoyan la alternativa">',
        '##INFO=<ID=AF,Number=1,Type=Float,Description="Fraccion alelica">',
        '##INFO=<ID=TYPE,Number=1,Type=String,Description="SNV, INS o DEL">',
    ]
    if contigs:
        lineas += [f"##contig=<ID={n},length={ln}>" for n, ln in contigs]
    else:
        for n in dict.fromkeys(v.contig for v in variants):
            lineas.append(f"##contig=<ID={n}>")
    lineas.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO")
    lineas += [v.to_vcf() for v in variants]
    return "\n".join(lineas) + "\n"
