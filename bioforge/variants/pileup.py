"""
bioforge/variants/pileup.py — apilar las lecturas sobre la referencia.

El *pileup* es el paso intermedio entre mapear y llamar variantes: para cada
posición del genoma, cuántas lecturas dijeron A, C, G, T, N o «aquí falta una
base». Es la evidencia cruda; decidir qué es una mutación real es cosa del
llamador (:mod:`bioforge.variants.caller`).

Se publica aparte porque tiene valor por sí mismo: la **profundidad** de cobertura
(cuántas lecturas cubren cada punto) contesta sola la pregunta más común de un
experimento de secuenciación — «¿he leído bastante?».

Cómo cumple la regla de oro nº1
-------------------------------
No hay bucles por símbolo. Se itera por **lectura** (registro) y, dentro de cada
una, por **tramo del CIGAR** (unos pocos por lectura, como hace ``_cigar`` en el
mapeador). Todo el trabajo por base es NumPy vectorizado, y el recuento final es
un único ``bincount`` sobre el lote acumulado.
"""

from __future__ import annotations

import re
from typing import Iterable, NamedTuple, Optional

import numpy as np

from bioforge.core.biocore import SequenceValueError

# Códigos de base usados en la matriz de recuentos.
A, C, G, T, N, DEL = 0, 1, 2, 3, 4, 5
N_CANALES = 6
BASES = "ACGT"

# LUT ASCII → código. Todo lo que no sea ACGT (mayús o minús) cae en N.
_CODE = np.full(256, N, dtype=np.uint8)
for _i, _b in enumerate(b"ACGT"):
    _CODE[_b] = _i
    _CODE[_b + 32] = _i                      # minúsculas

_CIGAR_RE = re.compile(r"(\d+)([MIDNSHP=X])")

# Cuántas posiciones acumular antes de volcar al contador. Acota la RAM: el
# pileup de un genoma entero no necesita tener todas las bases a la vez en una
# lista, y volcar por lotes mantiene el bincount vectorizado.
_LOTE = 1 << 20

_COMPL = bytes.maketrans(b"ACGTacgtNn", b"TGCAtgcaNn")


def _revcomp(s: str) -> str:
    return s.encode("ascii").translate(_COMPL)[::-1].decode("ascii")


class Pileup(NamedTuple):
    """Evidencia apilada de un contig.

    Attributes
    ----------
    contig:
        Nombre del cromosoma/contig.
    counts:
        Matriz ``(L, 6)`` de enteros: por cada posición de la referencia, cuántas
        lecturas apoyaron A, C, G, T, N y deleción (columnas ``A``…``DEL``).
    insertions:
        ``{(pos, secuencia_insertada): nº de lecturas}``. Las inserciones no caben
        en la matriz (no ocupan una posición de la referencia), así que van aparte.
    n_reads:
        Lecturas que se apilaron (tras filtrar por calidad de mapeo).
    n_skipped:
        Lecturas descartadas (sin CIGAR, o por debajo de ``min_mapq``).
    """

    contig: str
    counts: np.ndarray
    insertions: dict[tuple[int, str], int]
    n_reads: int
    n_skipped: int

    @property
    def depth(self) -> np.ndarray:
        """Profundidad por posición: cuántas lecturas cubren cada base."""
        return self.counts.sum(axis=1)

    @property
    def mean_depth(self) -> float:
        """Profundidad media a lo largo del contig."""
        return float(self.depth.mean()) if self.counts.shape[0] else 0.0

    def covered(self, min_depth: int = 1) -> float:
        """Fracción del contig cubierta por al menos ``min_depth`` lecturas."""
        if not self.counts.shape[0]:
            return 0.0
        return float((self.depth >= min_depth).mean())

    def __repr__(self) -> str:                       # pragma: no cover - cosmético
        return (f"Pileup({self.contig!r}, {self.counts.shape[0]} pb, "
                f"{self.n_reads} lecturas, prof. media {self.mean_depth:.1f}×)")


def _walk(cigar: str, ref_start: int, query_start: int):
    """Recorre el CIGAR → tramos alineados (pos. referencia, pos. lectura, largo).

    Devuelve ``(bloques_M, deleciones, inserciones)``:
      * ``bloques_M``  — lista de ``(ref_pos, read_pos, largo)`` de columnas alineadas.
      * ``deleciones`` — lista de ``(ref_pos, largo)``: la referencia tiene bases que
        la lectura no.
      * ``inserciones``— lista de ``(ref_pos, read_pos, largo)``: la lectura tiene
        bases de más.

    Entiende el **alfabeto CIGAR completo** del estándar SAM, no solo ``M/I/D``:

    ====  ==========================  ================================
    op    significado                 avanza
    ====  ==========================  ================================
    M     alineado (igual o distinto)  referencia y lectura
    = X   igual / distinto explícito   referencia y lectura
    I     inserción en la lectura      lectura
    S     recorte blando (soft clip)   lectura  (NO alineado: se salta)
    D     deleción                     referencia
    N     hueco largo (intrón)         referencia (no cuenta como deleción)
    H P   recorte duro / relleno       nada
    ====  ==========================  ================================

    Esto importa: un BAM de ``minimap2`` o ``bwa`` trae ``S`` constantemente. Si se
    ignorase, la posición dentro de la lectura quedaría desplazada y **todas** las
    bases se apilarían en el sitio equivocado, en silencio. Se descubrió al montar
    el contraste contra ``bcftools``.

    Es un bucle por TRAMO (unos pocos por lectura), no por base: permitido por la
    regla nº2, igual que el traceback del alineador.
    """
    bloques, deleciones, inserciones = [], [], []
    r, q = ref_start, query_start
    for largo, op in _CIGAR_RE.findall(cigar):
        largo = int(largo)
        if op in "M=X":                              # alineado: avanzan los dos
            bloques.append((r, q, largo))
            r += largo
            q += largo
        elif op == "I":                              # bases de más en la lectura
            inserciones.append((r, q, largo))
            q += largo
        elif op == "S":                              # recorte blando: ni se mira
            q += largo
        elif op == "D":                              # falta en la lectura
            deleciones.append((r, largo))
            r += largo
        elif op == "N":                              # hueco largo: NO es una deleción
            r += largo
        # "H" (recorte duro) y "P" (relleno) no consumen nada
    return bloques, deleciones, inserciones


def _izquierda_del(ref: str, r: int, largo: int) -> int:
    """Desplaza una deleción todo lo posible a la IZQUIERDA (normalización del VCF).

    En una repetición, ``AAAA`` con una base menos puede escribirse borrando
    cualquiera de las cuatro: son la MISMA variante escrita de formas distintas.
    El estándar (lo que hace ``bcftools norm``) es fijar siempre la más a la
    izquierda, para que la llamada sea **canónica y comparable** con la de otras
    herramientas y para que dos lecturas que colocaron el hueco en sitios
    distintos cuenten como una sola variante.

    Nota honesta: con el mapeador de BioForge, que es determinista, todas las
    lecturas suelen colocar ya el hueco igual, así que esto rara vez cambia el
    recuento; su valor es la interoperabilidad, no arreglar indels partidos
    (eso depende del modelo de hueco del alineador — ver ``caller``).

    Bucle por DESPLAZAMIENTO (unos pocos), no por base.
    """
    tope = 0
    while r > 0 and r + largo <= len(ref) and ref[r - 1] == ref[r + largo - 1]:
        r -= 1
        tope += 1
        if tope > 1000:                              # cinturón: repeticiones enormes
            break
    return r


def _izquierda_ins(ref: str, r: int, seq: str) -> tuple[int, str]:
    """Igual que :func:`_izquierda_del`, pero para inserciones.

    Al desplazar, la secuencia insertada ROTA: insertar ``GA`` antes de una ``A``
    equivale a insertar ``AG`` una posición antes.
    """
    tope = 0
    while r > 0 and seq and ref[r - 1] == seq[-1]:
        seq = seq[-1] + seq[:-1]
        r -= 1
        tope += 1
        if tope > 1000:
            break
    return r, seq


def pileup(reference, alignments: Iterable[tuple[str, object]], *,
           contig: Optional[str] = None, min_mapq: int = 0,
           oriented: bool = True) -> Pileup:
    """Apila lecturas mapeadas sobre la referencia.

    Parameters
    ----------
    reference:
        La secuencia del contig (``str``) o, si no se tiene a mano, solo su
        longitud (``int``). Con la secuencia completa los **indels se normalizan
        a la izquierda**, que es lo que evita que una misma inserción aparezca
        repetida en dos posiciones; pasando solo la longitud eso no es posible.
    alignments:
        Iterable de pares ``(lectura, Mapping)``, tal como los devuelve
        :meth:`bioforge.mapping.genomemap.GenomeAligner.map`. Se ignoran los
        mapeos sin CIGAR.
    contig:
        Nombre del contig a apilar. Si es ``None`` (por defecto) **no se filtra**:
        se toma el nombre del primer mapeo y se apila todo. Pásalo explícitamente
        solo cuando la referencia tenga VARIOS contigs y quieras uno concreto.

        (Antes el valor por defecto era ``"ref"`` y filtraba siempre, lo que
        descartaba en silencio el 100 % de las lecturas si el mapeador etiquetaba
        el contig con otro nombre — la profundidad salía 0× sin decir por qué.
        Ese fallo se detectó integrando la CLI, y por eso el defecto ya no filtra.)
    min_mapq:
        Descarta mapeos con calidad de mapeo inferior (0 = no filtrar).
    oriented:
        Si ``True`` (por defecto) las coordenadas ``query_start`` del mapeo se
        interpretan sobre la lectura ya orientada, que es lo que produce el
        mapeador de BioForge al alinear el complemento inverso en la hebra ``-``.

    Returns
    -------
    Pileup
        La evidencia apilada, lista para :func:`bioforge.variants.call_variants`.
    """
    ref_seq = reference.upper() if isinstance(reference, str) else None
    ref_len = len(reference) if ref_seq is not None else int(reference)
    if ref_len <= 0:
        raise SequenceValueError(
            f"la referencia debe tener longitud positiva, se recibió {ref_len}.")

    counts = np.zeros((ref_len, N_CANALES), dtype=np.int32)
    inserciones: dict[tuple[int, str], int] = {}
    n_reads = n_skipped = 0

    # lote acumulado: índices planos (pos*N_CANALES + código) → un bincount al volcar
    buffer: list[np.ndarray] = []
    pendientes = 0

    def volcar() -> None:
        nonlocal buffer, pendientes
        if not buffer:
            return
        plano = np.concatenate(buffer)
        counts.reshape(-1)[:] += np.bincount(
            plano, minlength=ref_len * N_CANALES).astype(np.int32)
        buffer = []
        pendientes = 0

    nombre = contig                                  # se fija con el primer mapeo
    for lectura, mp in alignments:
        cigar = getattr(mp, "cigar", None)
        if not cigar:
            n_skipped += 1
            continue
        suyo = getattr(mp, "target_name", None)
        if nombre is None:
            nombre = suyo or "ref"
        elif suyo is not None and suyo != nombre:    # solo filtra si se pidió contig
            n_skipped += 1
            continue
        if getattr(mp, "mapq", 0) < min_mapq:
            n_skipped += 1
            continue

        seq = lectura.upper()
        if oriented and getattr(mp, "strand", "+") == "-":
            seq = _revcomp(seq)
        codigos = _CODE[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]

        bloques, dels, ins = _walk(cigar, int(mp.target_start), int(mp.query_start))

        for r0, q0, largo in bloques:                # bucle por TRAMO, no por base
            # recorte a los límites reales (defensivo ante CIGAR inconsistente)
            largo = min(largo, ref_len - r0, codigos.size - q0)
            if largo <= 0:
                continue
            posiciones = np.arange(r0, r0 + largo, dtype=np.int64)
            bases = codigos[q0:q0 + largo].astype(np.int64)
            buffer.append(posiciones * N_CANALES + bases)
            pendientes += largo

        for r0, largo in dels:
            largo = min(largo, ref_len - r0)
            if largo <= 0:
                continue
            if ref_seq is not None:                  # normaliza: todas al mismo sitio
                r0 = _izquierda_del(ref_seq, r0, largo)
            posiciones = np.arange(r0, r0 + largo, dtype=np.int64)
            buffer.append(posiciones * N_CANALES + DEL)
            pendientes += largo

        for r0, q0, largo in ins:
            if not 0 <= r0 < ref_len or q0 + largo > len(seq):
                continue
            insertada = seq[q0:q0 + largo]
            if ref_seq is not None:
                r0, insertada = _izquierda_ins(ref_seq, r0, insertada)
            clave = (r0, insertada)
            inserciones[clave] = inserciones.get(clave, 0) + 1

        n_reads += 1
        if pendientes >= _LOTE:
            volcar()

    volcar()
    return Pileup(contig=nombre or (contig or "ref"), counts=counts,
                  insertions=inserciones, n_reads=n_reads, n_skipped=n_skipped)


def pileup_from_mappings(reference, reads: Iterable[str], mappings, *,
                         min_mapq: int = 0,
                         contig: Optional[str] = None) -> Pileup:
    """Atajo: apila el resultado de ``GenomeAligner.map_batch``.

    ``mappings`` es una lista de listas (los candidatos de cada lectura); se toma
    el mapeo **primario** de cada una, que es el primero.
    """
    pares = []
    for lectura, maps in zip(reads, mappings):
        if maps:
            pares.append((lectura, maps[0]))
    return pileup(reference, pares, contig=contig, min_mapq=min_mapq)
