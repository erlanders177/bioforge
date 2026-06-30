"""
smart_translator.py
═══════════════════════════════════════════════════════════════════════
Standard Genetic Code translator for the biocore 5-bit engine.

Integrates with biocore.py (A=0, C=1, G=2, T/U=3, AA codes 4–23,
STOP=24) without any Python-level loops in the hot path.

Pipeline (fully vectorised — zero Python loops at call time)
─────────────────────────────────────────────────────────────
  ①  Decode PackedSequence(NUC) → uint8 codes [0–3]
  ②  Locate first ATG/AUG via sliding_window_view        (Fail-Fast)
  ③  Extract ORF from ATG; reshape into (N, 3) codon matrix
  ④  Compute LUT index: idx = n₁×16 + n₂×4 + n₃         (base-4, vectorised)
  ⑤  CODON_LUT[indices]  →  amino-acid BioCode array      (single fancy-index)
  ⑥  Truncate at first STOP codon (BioCode.STOP = 24)
  ⑦  Raise UserWarning if result < 50 aa  (possible ncRNA / broken fragment)
  ⑧  BitPacker.pack(aa_codes) → new PackedSequence(PROTEIN)

CODON_LUT  (64 entries, Standard Genetic Code NCBI table #1)
─────────────────────────────────────────────────────────────
  Index = n₁×16 + n₂×4 + n₃  │  A=0  C=1  G=2  T/U=3

  Idx  Codon  AA  │  Idx  Codon  AA  │  Idx  Codon  AA  │  Idx  Codon  AA
  ──────────────── │  ──────────────── │  ──────────────── │  ────────────────
   0   AAA    K   │  16   CAA    Q   │  32   GAA    E   │  48   UAA   STP
   1   AAC    N   │  17   CAC    H   │  33   GAC    D   │  49   UAC    Y
   2   AAG    K   │  18   CAG    Q   │  34   GAG    E   │  50   UAG   STP
   3   AAU    N   │  19   CAU    H   │  35   GAU    D   │  51   UAU    Y
   4   ACA    T   │  20   CCA    P   │  36   GCA    A   │  52   UCA    S
   5   ACC    T   │  21   CCC    P   │  37   GCC    A   │  53   UCC    S
   6   ACG    T   │  22   CCG    P   │  38   GCG    A   │  54   UCG    S
   7   ACU    T   │  23   CCU    P   │  39   GCU    A   │  55   UCU    S
   8   AGA    R   │  24   CGA    R   │  40   GGA    G   │  56   UGA   STP
   9   AGC    S   │  25   CGC    R   │  41   GGC    G   │  57   UGC    C
  10   AGG    R   │  26   CGG    R   │  42   GGG    G   │  58   UGG    W
  11   AGU    S   │  27   CGU    R   │  43   GGU    G   │  59   UGU    C
  12   AUA    I   │  28   CUA    L   │  44   GUA    V   │  60   UUA    L
  13   AUC    I   │  29   CUC    L   │  45   GUC    V   │  61   UUC    F
  14   AUG    M   │  30   CUG    L   │  46   GUG    V   │  62   UUG    L
  15   AUU    I   │  31   CUU    L   │  47   GUU    V   │  63   UUU    F
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# ── biocore integration ────────────────────────────────────────────────────────
from .biocore import (
    BioCode,
    BitPacker,
    PackedSequence,
    SeqType,
    SequenceTypeError,
    SequenceValueError,
    TranslationError,
)

# ── motor C (opcional) ────────────────────────────────────────────────────────
try:
    from .engine._loader import C_AVAILABLE as _C_AVAILABLE
    from .engine._loader import c_find_atg as _c_find_atg
    from .engine._loader import c_translate as _c_translate
except ImportError:
    _C_AVAILABLE = False

__all__: list[str] = ["SmartTranslator"]


# ══════════════════════════════════════════════════════════════════════════════
# §1  CODON LUT  —  Standard Genetic Code, built once at module load
# ══════════════════════════════════════════════════════════════════════════════

def _build_codon_lut() -> np.ndarray:
    """
    Build the 64-entry Standard Genetic Code lookup array.

    Encoding: A=0, C=1, G=2, U=3 (T and U share the same slot as in biocore).
    LUT index = n₁×16 + n₂×4 + n₃  →  [0, 63].
    Stored values: BioCode.AA_* (4–23), BioCode.STOP (24), BioCode.UNK (31).

    This function runs **once** at module import time and is never called again.

    Returns
    -------
    np.ndarray, dtype=uint8, shape=(64,)
    """
    # RNA base → biocore nucleotide code
    _B: dict[str, int] = {"A": 0, "C": 1, "G": 2, "U": 3}

    # Standard Genetic Code  (NCBI table #1)
    # RNA codon string → BioCode integer value
    _TABLE: dict[str, int] = {
        # ── Phenylalanine  F = 8 ─────────────────────────────────────────────
        "UUU": BioCode.AA_F, "UUC": BioCode.AA_F,
        # ── Leucine  L = 13 ──────────────────────────────────────────────────
        "UUA": BioCode.AA_L, "UUG": BioCode.AA_L,
        "CUU": BioCode.AA_L, "CUC": BioCode.AA_L,
        "CUA": BioCode.AA_L, "CUG": BioCode.AA_L,
        # ── Isoleucine  I = 11 ───────────────────────────────────────────────
        "AUU": BioCode.AA_I, "AUC": BioCode.AA_I, "AUA": BioCode.AA_I,
        # ── Methionine / START  M = 14 ───────────────────────────────────────
        "AUG": BioCode.AA_M,
        # ── Valine  V = 21 ───────────────────────────────────────────────────
        "GUU": BioCode.AA_V, "GUC": BioCode.AA_V,
        "GUA": BioCode.AA_V, "GUG": BioCode.AA_V,
        # ── Serine  S = 19 ───────────────────────────────────────────────────
        "UCU": BioCode.AA_S, "UCC": BioCode.AA_S,
        "UCA": BioCode.AA_S, "UCG": BioCode.AA_S,
        "AGU": BioCode.AA_S, "AGC": BioCode.AA_S,
        # ── Proline  P = 16 ──────────────────────────────────────────────────
        "CCU": BioCode.AA_P, "CCC": BioCode.AA_P,
        "CCA": BioCode.AA_P, "CCG": BioCode.AA_P,
        # ── Threonine  T = 20 ────────────────────────────────────────────────
        "ACU": BioCode.AA_T, "ACC": BioCode.AA_T,
        "ACA": BioCode.AA_T, "ACG": BioCode.AA_T,
        # ── Alanine  A = 4 ───────────────────────────────────────────────────
        "GCU": BioCode.AA_A, "GCC": BioCode.AA_A,
        "GCA": BioCode.AA_A, "GCG": BioCode.AA_A,
        # ── Tyrosine  Y = 23 ─────────────────────────────────────────────────
        "UAU": BioCode.AA_Y, "UAC": BioCode.AA_Y,
        # ── STOP codons  (24) ────────────────────────────────────────────────
        "UAA": BioCode.STOP, "UAG": BioCode.STOP, "UGA": BioCode.STOP,
        # ── Histidine  H = 10 ────────────────────────────────────────────────
        "CAU": BioCode.AA_H, "CAC": BioCode.AA_H,
        # ── Glutamine  Q = 17 ────────────────────────────────────────────────
        "CAA": BioCode.AA_Q, "CAG": BioCode.AA_Q,
        # ── Asparagine  N = 15 ───────────────────────────────────────────────
        "AAU": BioCode.AA_N, "AAC": BioCode.AA_N,
        # ── Lysine  K = 12 ───────────────────────────────────────────────────
        "AAA": BioCode.AA_K, "AAG": BioCode.AA_K,
        # ── Aspartic acid  D = 6 ─────────────────────────────────────────────
        "GAU": BioCode.AA_D, "GAC": BioCode.AA_D,
        # ── Glutamic acid  E = 7 ─────────────────────────────────────────────
        "GAA": BioCode.AA_E, "GAG": BioCode.AA_E,
        # ── Cysteine  C = 5 ──────────────────────────────────────────────────
        "UGU": BioCode.AA_C, "UGC": BioCode.AA_C,
        # ── Tryptophan  W = 22 ───────────────────────────────────────────────
        "UGG": BioCode.AA_W,
        # ── Arginine  R = 18 ─────────────────────────────────────────────────
        "CGU": BioCode.AA_R, "CGC": BioCode.AA_R,
        "CGA": BioCode.AA_R, "CGG": BioCode.AA_R,
        "AGA": BioCode.AA_R, "AGG": BioCode.AA_R,
        # ── Glycine  G = 9 ───────────────────────────────────────────────────
        "GGU": BioCode.AA_G, "GGC": BioCode.AA_G,
        "GGA": BioCode.AA_G, "GGG": BioCode.AA_G,
    }

    assert len(_TABLE) == 64, f"Genetic code table must have 64 entries, got {len(_TABLE)}"

    lut = np.full(64, BioCode.UNK, dtype=np.uint8)
    for codon, aa_code in _TABLE.items():
        n1, n2, n3 = _B[codon[0]], _B[codon[1]], _B[codon[2]]
        lut[n1 * 16 + n2 * 4 + n3] = np.uint8(aa_code)

    return lut


# ══════════════════════════════════════════════════════════════════════════════
# §2  SMART TRANSLATOR
# ══════════════════════════════════════════════════════════════════════════════

class SmartTranslator:
    """
    Vectorised Standard Genetic Code translator for biocore PackedSequences.

    Accepts a ``PackedSequence(NUCLEOTIDE)``, locates the first ATG/AUG
    codon via a zero-copy sliding-window view, translates the ORF using
    a single numpy fancy-index lookup, truncates at the first STOP codon,
    and returns a ``PackedSequence(PROTEIN)`` — all without a single
    Python-level loop in the call path.

    Class constants
    ───────────────
    CODON_LUT     : np.ndarray, shape (64,) — Standard Genetic Code LUT.
    _START_CODON  : np.ndarray [0, 3, 2]   — ATG/AUG in 5-bit encoding.
    _CODON_WEIGHTS: np.ndarray [16, 4, 1]  — base-4 index weights (uint16).
    _MIN_AA_LEN   : int = 50               — biological plausibility floor.

    Quick start
    ───────────
    >>> from bioforge import SmartImporter, SeqType
    >>> nuc_seq = SmartImporter.from_string(fasta)[0]
    >>> prot_seq = SmartTranslator.translate(nuc_seq)
    >>> print(prot_seq.to_string())
    """

    # ── Class-level constants (allocated once at class definition) ─────────────

    # Standard Genetic Code LUT — 64 entries, computed at class load time.
    # Index formula: n₁×16 + n₂×4 + n₃  (A=0 C=1 G=2 T/U=3)
    # Values: BioCode.AA_* (4–23), BioCode.STOP (24), BioCode.UNK (31)
    CODON_LUT: np.ndarray = _build_codon_lut()

    # ATG / AUG start codon in biocore 5-bit encoding  (A=0, T/U=3, G=2)
    _START_CODON: np.ndarray = np.array([0, 3, 2], dtype=np.uint8)

    # Base-4 weights for codon index arithmetic.
    # uint16 prevents overflow when ambiguous/gap codes (values > 3) appear:
    #   worst case: UNK=31 → 31×16 + 31×4 + 31 = 651 → out of [0,63] → masked
    _CODON_WEIGHTS: np.ndarray = np.array([16, 4, 1], dtype=np.uint16)

    # Proteins shorter than this trigger a UserWarning
    _MIN_AA_LEN: int = 50

    # ── Public API ─────────────────────────────────────────────────────────────

    @classmethod
    def translate(
        cls,
        seq:        PackedSequence,
        warn_short: bool = True,
    ) -> PackedSequence:
        """
        Translate a nucleotide ``PackedSequence`` into a protein ``PackedSequence``.

        Translation starts at the first ATG/AUG codon and stops at the
        first in-frame STOP codon.  The STOP codon itself is excluded from
        the result.

        Parameters
        ----------
        seq : PackedSequence
            A nucleotide-type packed sequence (``SeqType.NUCLEOTIDE``).
        warn_short : bool, default True
            If ``True``, emit a ``UserWarning`` when the translated protein
            is shorter than 50 amino acids.

        Returns
        -------
        PackedSequence
            A protein-type packed sequence (``SeqType.PROTEIN``).
            Header is prefixed with ``[PROT | ORF@<start>]``.

        Raises
        ------
        TypeError
            If *seq* is not a ``NUCLEOTIDE`` sequence.
        ValueError
            If the sequence is too short, or no ATG/AUG codon is found,
            or the ORF yields no complete codon after the start.
        """

        # ── Guard clauses ─────────────────────────────────────────────────────
        if not isinstance(seq, PackedSequence):
            raise SequenceTypeError(
                f"seq debe ser PackedSequence, se recibió {type(seq).__name__!r}. "
                "Crea la secuencia con SmartImporter.from_string() o SmartImporter.from_file()."
            )
        if seq.seq_type != SeqType.NUCLEOTIDE:
            raise SequenceTypeError(
                f"Se esperaba SeqType.NUCLEOTIDE, se recibió {seq.seq_type.name}. "
                "SmartTranslator solo traduce ADN/ARN. "
                "Para alinear proteínas usa SequenceAligner directamente."
            )
        if seq.n_symbols < 3:
            raise SequenceValueError(
                f"Secuencia demasiado corta ({seq.n_symbols} nt): "
                "se necesitan al menos 3 nucleótidos para un codón."
            )

        # ① Unpack to flat uint8 nucleotide code array
        nuc_codes: np.ndarray = seq.decode()                       # shape (n,)

        # ② Vectorised ORF search — locate first ATG/AUG  (Fail-Fast)
        orf_start: int = cls._find_orf_start(nuc_codes)

        # ③ Extract ORF from start codon; discard any incomplete trailing codon
        orf: np.ndarray = nuc_codes[orf_start:]
        n_codons: int   = len(orf) // 3
        if n_codons == 0:
            raise TranslationError(
                f"El ORF en la posición {orf_start} no tiene ningún codón "
                "completo tras el ATG de inicio. "
                "La secuencia termina justo en el ATG sin residuos posteriores."
            )

        # ④+⑤ Traducir codones → AAs  (C si disponible, NumPy si no)
        if _C_AVAILABLE:
            aa_codes: np.ndarray = _c_translate(cls.CODON_LUT, orf, n_codons)
        else:
            codons: np.ndarray  = orf[: n_codons * 3].reshape(n_codons, 3)
            indices_u16: np.ndarray = (
                codons.astype(np.uint16) * cls._CODON_WEIGHTS
            ).sum(axis=1)
            valid_codon: np.ndarray = indices_u16 < np.uint16(64)
            safe_idx: np.ndarray    = np.where(
                valid_codon, indices_u16, np.uint16(0)
            ).astype(np.uint8)
            aa_codes = np.where(
                valid_codon,
                cls.CODON_LUT[safe_idx],
                np.uint8(BioCode.UNK),
            ).astype(np.uint8)

        # ⑥ Truncate at first in-frame STOP codon (BioCode.STOP = 24).
        stop_mask: np.ndarray = aa_codes == np.uint8(BioCode.STOP)
        if stop_mask.any():
            aa_codes = aa_codes[: int(np.argmax(stop_mask))]      # exclude STOP itself

        # ⑦ Biological plausibility check
        n_aa: int = int(len(aa_codes))
        if warn_short and n_aa < cls._MIN_AA_LEN:
            warnings.warn(
                f"Secuencia sospechosamente corta ({n_aa} aa < {cls._MIN_AA_LEN} aa). "
                "Podría ser un ARNt, micro-ARN o un fragmento roto.",
                UserWarning,
                stacklevel=2,
            )

        # ⑧ Pack and return as a new PROTEIN PackedSequence
        return PackedSequence(
            header    = f"[PROT | ORF@{orf_start}] {seq.header}",
            seq_type  = SeqType.PROTEIN,
            n_symbols = n_aa,
            data      = BitPacker.pack(aa_codes),
        )

    @classmethod
    def translate_all_frames(
        cls,
        seq:        PackedSequence,
        warn_short: bool = False,
    ) -> list[PackedSequence]:
        """
        Translate all 6 reading frames of a nucleotide sequence.

        The 6 frames are:
        +1, +2, +3 — forward strand, offsets 0, 1, 2.
        -1, -2, -3 — reverse complement strand, offsets 0, 1, 2.

        Only frames that contain at least one ATG codon are included.
        Frames with no ATG are silently skipped, so the result may have
        fewer than 6 entries (or be empty for non-coding sequences).

        Parameters
        ----------
        seq : PackedSequence
            A nucleotide-type packed sequence (``SeqType.NUCLEOTIDE``).
        warn_short : bool, default False
            If ``True``, emit a ``UserWarning`` for proteins shorter than
            50 amino acids.  Off by default because short ORFs are common
            in non-primary frames.

        Returns
        -------
        list[PackedSequence]
            Up to 6 protein PackedSequences, one per frame with an ORF.
            Each header: ``[PROT | frame +/-N | ORF@<pos>] <original>``.
            Empty list if no ORF is found in any frame.

        Raises
        ------
        SequenceTypeError
            If *seq* is not a ``NUCLEOTIDE`` sequence.
        SequenceValueError
            If the sequence is shorter than 3 nucleotides.
        """
        if not isinstance(seq, PackedSequence):
            raise SequenceTypeError(
                f"seq debe ser PackedSequence, se recibió {type(seq).__name__!r}. "
                "Crea la secuencia con SmartImporter.from_string() o SmartImporter.from_file()."
            )
        if seq.seq_type != SeqType.NUCLEOTIDE:
            raise SequenceTypeError(
                f"Se esperaba SeqType.NUCLEOTIDE, se recibió {seq.seq_type.name}. "
                "translate_all_frames() solo traduce ADN/ARN."
            )
        if seq.n_symbols < 3:
            raise SequenceValueError(
                f"Secuencia demasiado corta ({seq.n_symbols} nt): "
                "se necesitan al menos 3 nucleótidos para un codón."
            )

        rc_seq  = seq.reverse_complement()
        results: list[PackedSequence] = []

        for strand_codes, strand_sign in [
            (seq.decode(),    '+'),
            (rc_seq.decode(), '-'),
        ]:
            for offset in range(3):
                frame_codes = strand_codes[offset:]
                if len(frame_codes) < 3:
                    continue
                frame_label = f"{strand_sign}{offset + 1}"

                try:
                    orf_start = cls._find_orf_start(frame_codes)
                except (TranslationError, ValueError):
                    continue  # no ATG in this frame — skip silently

                orf      = frame_codes[orf_start:]
                n_codons = len(orf) // 3
                if n_codons == 0:
                    continue

                # Translate codons (C engine or NumPy)
                if _C_AVAILABLE:
                    aa_codes: np.ndarray = _c_translate(cls.CODON_LUT, orf, n_codons)
                else:
                    codons      = orf[: n_codons * 3].reshape(n_codons, 3)
                    idx_u16     = (codons.astype(np.uint16) * cls._CODON_WEIGHTS).sum(axis=1)
                    valid       = idx_u16 < np.uint16(64)
                    safe_idx    = np.where(valid, idx_u16, np.uint16(0)).astype(np.uint8)
                    aa_codes    = np.where(
                        valid, cls.CODON_LUT[safe_idx], np.uint8(BioCode.UNK)
                    ).astype(np.uint8)

                # Truncate at first in-frame STOP codon
                stop_mask = aa_codes == np.uint8(BioCode.STOP)
                if stop_mask.any():
                    aa_codes = aa_codes[: int(np.argmax(stop_mask))]

                n_aa = int(len(aa_codes))
                if n_aa == 0:
                    continue

                if warn_short and n_aa < cls._MIN_AA_LEN:
                    warnings.warn(
                        f"Frame {frame_label}: proteína corta "
                        f"({n_aa} aa < {cls._MIN_AA_LEN} aa).",
                        UserWarning,
                        stacklevel=2,
                    )

                abs_pos = offset + orf_start
                results.append(PackedSequence(
                    header    = f"[PROT | frame {frame_label} | ORF@{abs_pos}] {seq.header}",
                    seq_type  = SeqType.PROTEIN,
                    n_symbols = n_aa,
                    data      = BitPacker.pack(aa_codes),
                ))

        return results

    # ── Private helpers ────────────────────────────────────────────────────────

    @classmethod
    def _find_orf_start(cls, codes: np.ndarray) -> int:
        """
        Locate the first ATG/AUG codon using a vectorised sliding-window view.

        Uses ``numpy.lib.stride_tricks.sliding_window_view`` to create a
        zero-copy ``(n-2, 3)`` view over *codes*, then compares every
        3-mer against ``[0, 3, 2]`` (= A, T/U, G) in one broadcast op.
        ``np.argmax`` performs a C-level scan and returns the first hit index.

        Parameters
        ----------
        codes : np.ndarray, dtype uint8
            Decoded nucleotide code array (values 0–3 for ACGTU).

        Returns
        -------
        int
            Zero-based index of the first nucleotide of the ATG codon.

        Raises
        ------
        ValueError
            If no ATG/AUG codon is found (Fail-Fast).
        """
        if _C_AVAILABLE:
            pos = _c_find_atg(codes)
            if pos == -1:
                raise TranslationError(
                    "No se encontró ningún codón de inicio ATG/AUG en la secuencia. "
                    "Comprueba que sea una secuencia codificante (CDS) completa. "
                    "Si la auto-detección de tipo falla, importa con force_type=SeqType.NUCLEOTIDE."
                )
            return pos

        # fallback NumPy: sliding-window vectorizado
        windows: np.ndarray = sliding_window_view(codes, window_shape=3)
        hit_mask: np.ndarray = np.all(windows == cls._START_CODON, axis=1)
        first_hit: int = int(np.argmax(hit_mask))
        if not hit_mask[first_hit]:
            raise TranslationError(
                "No se encontró ningún codón de inicio ATG/AUG en la secuencia. "
                "Asegúrate de que sea una secuencia codificante (CDS) válida."
            )
        return first_hit


# ══════════════════════════════════════════════════════════════════════════════
# §3  DEMO / SELF-TEST   (python smart_translator.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import time
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from bioforge import SmartImporter, compute_stats

    W = 65
    print("═" * W)
    print("  smart_translator.py — Vectorised Genetic Code demo")
    print("═" * W)

    # ── Inspect LUT ────────────────────────────────────────────────────────────
    lut = SmartTranslator.CODON_LUT
    n_stop = int((lut == BioCode.STOP).sum())
    n_aa   = int(((lut >= 4) & (lut <= 23)).sum())
    print(f"\n  CODON_LUT  ({len(lut)} entries)")
    print(f"  ├─ Sense codons (AA 4–23) : {n_aa}")
    print(f"  ├─ STOP codons  (24)      : {n_stop}")
    print(f"  └─ Filled / UNK entries   : {n_aa + n_stop} / {64 - n_aa - n_stop}")

    # ── Test 1: HBB_HUMAN — long CDS with upstream noise ──────────────────────
    # First 48 codons of human haemoglobin beta-chain mRNA (NM_000518),
    # prefixed with 8 nt of 5′ UTR noise to exercise the ORF finder.
    HBB_FASTA = """\
>NM_000518.5|HBB_HUMAN|partial CDS with 5-UTR noise
NNNAACCCATGGTGCACCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGT
GGATGAAGTTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGGTTCTTTGAG
TCCTTTGGGGATCTGTCCACTCCTGATGCTGTTATGGGCAACCCTAAGGTGAAGGCTCATGGCAAGAAAG
TGCTCGGTGCCTTTAGTGATGGCCTGGCTCACCTGGACAACCTCAAGGGCACCTTTGCCACACTGAGTGA
GCTGCACTGTGACAAGCTGCACGTGGATCCTGAGAACTTCAGGCTCCTGGGCAACGTGCTGGTCTGTGTG
CTGGCCCATCACTTTGGCAAAGAATTCACCCCACCAGTGCAGGCTGCCTATCAGAAAGTGGTGGCTGGTGT
GGCTAATGCCCTGGCCCACAAGTATCACTAA
"""

    # Test 2: short synthetic ORF to trigger UserWarning
    SHORT_FASTA = """\
>synthetic|short_ORF|triggers_warning
ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGA
"""

    test_cases = [
        ("HBB_HUMAN (full CDS)",    HBB_FASTA,   False),
        ("Synthetic short ORF",     SHORT_FASTA,  True),
    ]

    for title, fasta, expect_warn in test_cases:
        print(f"\n  {'─'*55}")
        print(f"  {title}")
        print(f"  {'─'*55}")

        nuc_seq = SmartImporter.from_string(fasta, force_type=SeqType.NUCLEOTIDE)[0]
        print(f"  Input   : {repr(nuc_seq)}")

        import warnings as _w
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            t0 = time.perf_counter()
            prot_seq = SmartTranslator.translate(nuc_seq)
            elapsed_us = (time.perf_counter() - t0) * 1e6

        prot_str = prot_seq.to_string()
        stats    = compute_stats(prot_seq)

        print(f"  Output  : {repr(prot_seq)}")
        print(f"  Protein : {prot_str[:60]}{'…' if len(prot_str)>60 else ''}")
        print(f"  Elapsed : {elapsed_us:.1f} µs")
        print(f"  Mem save: {stats.compression_pct:.1f}%  "
              f"({stats.n_packed_bytes} B packed vs {stats.n_symbols} B naive)")

        if caught:
            print(f"  ⚠  UserWarning: {caught[0].message}")
            if expect_warn:
                print("  UserWarning raised as expected ✅")
        else:
            print("  No warning (sequence ≥ 50 aa) ✅")

        # Round-trip integrity
        aa_codes  = prot_seq.decode()
        repacked  = BitPacker.pack(aa_codes)
        rt_ok     = np.array_equal(repacked, prot_seq.data)
        print(f"  Round-trip: {'✅' if rt_ok else '❌'}  "
              f"({prot_seq.n_symbols} aa → {prot_seq.packed_bytes} B → {'OK' if rt_ok else 'FAIL'})")

    # ── Test 3: Error paths ────────────────────────────────────────────────────
    print(f"\n  {'─'*55}")
    print("  Error-path validation")
    print(f"  {'─'*55}")

    # TypeError: protein input
    prot_input = PackedSequence(
        header="fake_prot", seq_type=SeqType.PROTEIN, n_symbols=3,
        data=BitPacker.pack(np.array([4, 5, 6], dtype=np.uint8)),
    )
    try:
        SmartTranslator.translate(prot_input)
        print("  TypeError not raised ❌")
    except TypeError as exc:
        print(f"  TypeError ✅  → {exc}")

    # ValueError: no ATG
    no_atg_fasta = ">noatg\nCCCGGGTTTACCCACC\n"
    no_atg_seq = SmartImporter.from_string(no_atg_fasta, force_type=SeqType.NUCLEOTIDE)[0]
    try:
        SmartTranslator.translate(no_atg_seq)
        print("  ValueError (no ATG) not raised ❌")
    except ValueError as exc:
        print(f"  ValueError (no ATG) ✅  → {exc}")

    print(f"\n{'═' * W}\n")
