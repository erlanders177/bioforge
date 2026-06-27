"""
aligner.py
══════════════════════════════════════════════════════════════════════
Needleman-Wunsch pairwise sequence aligner — anti-diagonal wavefront.

Integrates with biocore.py / smart_translator.py (5-bit engine).

Vectorization strategy
──────────────────────
The NW recurrence H[i,j] = max(H[i-1,j-1]+s, H[i-1,j]+g, H[i,j-1]+g)
carries a cell-level data dependency that prevents full 2-D NumPy
vectorisation.  Cells on the same anti-diagonal (i+j = d) are however
mutually independent, enabling:

  • ONE outer Python loop over anti-diagonals  →  O(m+n) iterations
  • NumPy-vectorised computation inside each   →  up to min(m,n) ops

This reduces Python-level iterations from O(m·n) to O(m+n).

The traceback is inherently sequential (O(m+n)) — the data dependency
cannot be removed at that stage either.

Scoring (linear gap model)
──────────────────────────
  Match    : +2
  Mismatch : −1
  Gap      :  −2  (uniform open+extend)

Memory constraint
─────────────────
The full DP matrix is O(m·n) int32.  For sequences > 15 000 symbols
a UserWarning is emitted; banded alignment (not yet implemented)
would be the correct solution at chromosome scale.

Quick start
───────────
>>> from bioforge import SmartImporter, SeqType
>>> seqs  = SmartImporter.from_string(fasta_a + fasta_b,
...                                   force_type=SeqType.NUCLEOTIDE)
>>> result = SequenceAligner.align(seqs[0], seqs[1])
>>> print(format_alignment(result))
>>> for mut in result.mutations:
...     print(mut)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .biocore import BioCode, BitPacker, PackedSequence, SeqType
from .biocore import _NUC_DECODE, _AA_DECODE, _NUC_DECODE_ARR, _AA_DECODE_ARR
from .biocore import SequenceTypeError, SequenceValueError, AlignmentError

try:
    from .engine._loader import C_AVAILABLE
    from .engine._loader import c_nw_align   as _c_nw_align
    from .engine._loader import c_sw_align   as _c_sw_align
    from .engine._loader import c_nw_banded  as _c_nw_banded
except ImportError:
    C_AVAILABLE   = False
    _c_nw_align   = None  # type: ignore[assignment]
    _c_sw_align   = None  # type: ignore[assignment]
    _c_nw_banded  = None  # type: ignore[assignment]

# Bytes de decodificación para el motor C (32 bytes: BioCode -> ASCII)
_NUC_DECODE_BYTES: bytes = bytes(_NUC_DECODE_ARR)
_AA_DECODE_BYTES:  bytes = bytes(_AA_DECODE_ARR)


__all__: list[str] = [
    "Mutation",
    "AlignmentResult",
    "SequenceAligner",
    "format_alignment",
]


# ══════════════════════════════════════════════════════════════════════════════
# §1  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Mutation:
    """
    A single sequence variant between two aligned sequences.

    Attributes
    ──────────
    kind  : 'substitution' | 'deletion' | 'insertion'
            *deletion*  — seq_a has a symbol; seq_b has a gap.
            *insertion* — seq_a has a gap;    seq_b has a symbol.
    pos_a : 0-based index in seq_a (gap-insertion point for 'insertion').
    pos_b : 0-based index in seq_b (gap-insertion point for 'deletion').
    sym_a : symbol in seq_a  ('-' for insertions).
    sym_b : symbol in seq_b  ('-' for deletions).
    """
    kind:  str
    pos_a: int
    pos_b: int
    sym_a: str
    sym_b: str

    def __str__(self) -> str:
        if self.kind == 'substitution':
            return (
                f"SUB  a[{self.pos_a}]={self.sym_a!r} → "
                f"b[{self.pos_b}]={self.sym_b!r}"
            )
        if self.kind == 'deletion':
            return (
                f"DEL  a[{self.pos_a}]={self.sym_a!r}  "
                f"(deleción en seq_b, tras b[{self.pos_b}])"
            )
        return (
            f"INS  b[{self.pos_b}]={self.sym_b!r}  "
            f"(inserción en seq_b, gap en seq_a[{self.pos_a}])"
        )


@dataclass
class AlignmentResult:
    """Full result of a pairwise Needleman-Wunsch alignment."""
    score:        int
    identity:     float        # n_matches / aligned_length
    n_matches:    int
    n_mismatches: int
    n_gaps:       int          # total gap characters in the aligned region
    mutations:    list[Mutation]
    aligned_a:    str          # seq_a string with '-' at gap positions
    aligned_b:    str          # seq_b string with '-' at gap positions
    seq_type:     SeqType
    mode:         str          # 'global' or 'semi-global'


# ══════════════════════════════════════════════════════════════════════════════
# §2  SEQUENCE ALIGNER
# ══════════════════════════════════════════════════════════════════════════════

class SequenceAligner:
    """
    Needleman-Wunsch pairwise aligner with anti-diagonal (wavefront) strategy.

    Scoring-matrix fill : O(m+n) Python iterations × NumPy inner ops.
    Traceback           : O(m+n) sequential Python loop (unavoidable).

    Class constants
    ───────────────
    MATCH         :  np.int32  +2
    MISMATCH      :  np.int32  −1
    GAP           :  np.int32  −2  (linear gap model)
    _MAX_SAFE_LEN :  int  15 000   (DP matrix stays ≤ ~3.4 GB int32)
    """

    MATCH:    np.int32 = np.int32( 2)
    MISMATCH: np.int32 = np.int32(-1)
    GAP:      np.int32 = np.int32(-2)

    _MAX_SAFE_LEN: int = 15_000

    # ── Public API ─────────────────────────────────────────────────────────────

    @classmethod
    def align(
        cls,
        seq_a: PackedSequence,
        seq_b: PackedSequence,
        mode:  Literal['global', 'semi-global'] = 'global',
        band:  int | None = None,
    ) -> AlignmentResult:
        """
        Align seq_a (reference) against seq_b (query) — Needleman-Wunsch.

        Parameters
        ----------
        seq_a : PackedSequence — reference.
        seq_b : PackedSequence — query.
        mode  : 'global'      — penalises all terminal gaps.
                'semi-global' — free terminal gaps on the query side.
        band  : int, optional
            Half-width of the alignment band.  When given, only cells
            with ``|i - j| ≤ band`` are computed (banded NW).
            Reduces memory from O(m·n) to O(m·band) with the C engine.
            Required when sequences exceed ``_MAX_SAFE_LEN``.

        Returns
        -------
        AlignmentResult

        Raises
        ------
        TypeError  — seq_a.seq_type ≠ seq_b.seq_type.
        ValueError — either sequence is empty or band is too narrow.
        """
        if not isinstance(seq_a, PackedSequence):
            raise SequenceTypeError(
                f"seq_a debe ser PackedSequence, se recibió {type(seq_a).__name__!r}. "
                "Crea la secuencia con SmartImporter.from_string() o SmartImporter.from_file()."
            )
        if not isinstance(seq_b, PackedSequence):
            raise SequenceTypeError(
                f"seq_b debe ser PackedSequence, se recibió {type(seq_b).__name__!r}. "
                "Crea la secuencia con SmartImporter.from_string() o SmartImporter.from_file()."
            )
        if mode not in ('global', 'semi-global'):
            raise AlignmentError(
                f"mode debe ser 'global' o 'semi-global', se recibió {mode!r}."
            )
        if seq_a.seq_type != seq_b.seq_type:
            raise SequenceTypeError(
                f"Los tipos no coinciden: "
                f"{seq_a.seq_type.name} ≠ {seq_b.seq_type.name}. "
                "Ambas secuencias deben ser del mismo tipo (NUCLEOTIDE o PROTEIN)."
            )
        if seq_a.n_symbols == 0 or seq_b.n_symbols == 0:
            raise SequenceValueError("Las secuencias no pueden estar vacías.")

        m, n = seq_a.n_symbols, seq_b.n_symbols

        codes_a = seq_a.decode()   # (m,) uint8
        codes_b = seq_b.decode()   # (n,) uint8

        # ── Banded NW path ─────────────────────────────────────────────────────
        if band is not None:
            if band < 1:
                raise AlignmentError(f"band debe ser ≥ 1, se recibió {band}.")
            if C_AVAILABLE:
                return cls._align_banded_c(
                    codes_a, codes_b, m, n, band, mode, seq_a.seq_type
                )
            # NumPy fallback: banded fill sobre matriz completa
            if m > cls._MAX_SAFE_LEN or n > cls._MAX_SAFE_LEN:
                raise AlignmentError(
                    f"Secuencias > {cls._MAX_SAFE_LEN:,} síms con band= requieren "
                    "el motor C. Compila con: python bioforge/engine/build.py"
                )
            H = cls._fill_matrix_banded(codes_a, codes_b, m, n, band, mode)
            return cls._traceback(H, codes_a, codes_b, m, n, mode, seq_a.seq_type)

        # ── NW completo ────────────────────────────────────────────────────────
        if m > cls._MAX_SAFE_LEN or n > cls._MAX_SAFE_LEN:
            mem_mb = (m + 1) * (n + 1) * 4 / 1_000_000
            warnings.warn(
                f"Secuencia larga ({max(m, n):,} síms → "
                f"matriz DP ≈ {mem_mb:.0f} MB). "
                f"Usa band=<valor> para alineamiento por bandas.",
                UserWarning,
                stacklevel=2,
            )

        if C_AVAILABLE:
            return cls._align_c(codes_a, codes_b, m, n, mode, seq_a.seq_type)

        H = cls._fill_matrix(codes_a, codes_b, m, n, mode)
        return cls._traceback(H, codes_a, codes_b, m, n, mode, seq_a.seq_type)

    @classmethod
    def align_local(
        cls,
        seq_a: PackedSequence,
        seq_b: PackedSequence,
    ) -> AlignmentResult:
        """
        Smith-Waterman local alignment — finds the best-scoring subsequence.

        Unlike ``align()`` (global/semi-global NW), local alignment:
        • Does not penalise unaligned ends.
        • Finds the highest-scoring contiguous region.
        • Returns ``mode='local'`` in the result.

        Use when searching for a short motif inside a longer sequence,
        or when sequences share only a conserved domain.

        Parameters
        ----------
        seq_a : PackedSequence — reference (or the longer sequence).
        seq_b : PackedSequence — query (or the motif to search for).

        Returns
        -------
        AlignmentResult with ``mode='local'``.

        Raises
        ------
        SequenceTypeError — seq_a.seq_type ≠ seq_b.seq_type.
        SequenceValueError — either sequence is empty.
        """
        if not isinstance(seq_a, PackedSequence):
            raise SequenceTypeError(
                f"seq_a debe ser PackedSequence, se recibió {type(seq_a).__name__!r}."
            )
        if not isinstance(seq_b, PackedSequence):
            raise SequenceTypeError(
                f"seq_b debe ser PackedSequence, se recibió {type(seq_b).__name__!r}."
            )
        if seq_a.seq_type != seq_b.seq_type:
            raise SequenceTypeError(
                f"Los tipos no coinciden: "
                f"{seq_a.seq_type.name} ≠ {seq_b.seq_type.name}."
            )
        if seq_a.n_symbols == 0 or seq_b.n_symbols == 0:
            raise SequenceValueError("Las secuencias no pueden estar vacías.")

        m, n = seq_a.n_symbols, seq_b.n_symbols
        codes_a = seq_a.decode()
        codes_b = seq_b.decode()

        if C_AVAILABLE:
            return cls._align_sw_c(codes_a, codes_b, m, n, seq_a.seq_type)

        H = cls._fill_matrix_sw(codes_a, codes_b, m, n)
        return cls._traceback_sw(H, codes_a, codes_b, m, n, seq_a.seq_type)

    # ── Rutas C (motor nativo) ─────────────────────────────────────────────────

    @classmethod
    def _align_sw_c(
        cls,
        codes_a: np.ndarray,
        codes_b: np.ndarray,
        m: int,
        n: int,
        seq_type: SeqType,
    ) -> AlignmentResult:
        decode_bytes = (
            _NUC_DECODE_BYTES if seq_type == SeqType.NUCLEOTIDE else _AA_DECODE_BYTES
        )
        aligned_a, aligned_b, score, n_matches, n_mismatches, n_gaps = _c_sw_align(
            codes_a, codes_b, decode_bytes,
            int(cls.MATCH), int(cls.MISMATCH), int(cls.GAP),
        )
        mutations = cls._detect_mutations(aligned_a, aligned_b)
        aln_len   = n_matches + n_mismatches + n_gaps
        identity  = n_matches / aln_len if aln_len else 0.0
        return AlignmentResult(
            score=score, identity=identity,
            n_matches=n_matches, n_mismatches=n_mismatches, n_gaps=n_gaps,
            mutations=mutations, aligned_a=aligned_a, aligned_b=aligned_b,
            seq_type=seq_type, mode='local',
        )

    @classmethod
    def _align_banded_c(
        cls,
        codes_a: np.ndarray,
        codes_b: np.ndarray,
        m: int,
        n: int,
        band: int,
        mode: str,
        seq_type: SeqType,
    ) -> AlignmentResult:
        decode_bytes = (
            _NUC_DECODE_BYTES if seq_type == SeqType.NUCLEOTIDE else _AA_DECODE_BYTES
        )
        aligned_a, aligned_b, score, n_matches, n_mismatches, n_gaps = _c_nw_banded(
            codes_a, codes_b, decode_bytes,
            int(cls.MATCH), int(cls.MISMATCH), int(cls.GAP),
            band, mode,
        )
        mutations = cls._detect_mutations(aligned_a, aligned_b)
        aln_len   = n_matches + n_mismatches + n_gaps
        identity  = n_matches / aln_len if aln_len else 0.0
        return AlignmentResult(
            score=score, identity=identity,
            n_matches=n_matches, n_mismatches=n_mismatches, n_gaps=n_gaps,
            mutations=mutations, aligned_a=aligned_a, aligned_b=aligned_b,
            seq_type=seq_type, mode=mode,
        )

    @classmethod
    def _align_c(
        cls,
        codes_a: np.ndarray,
        codes_b: np.ndarray,
        m: int,
        n: int,
        mode: str,
        seq_type: SeqType,
    ) -> AlignmentResult:
        decode_bytes = (
            _NUC_DECODE_BYTES if seq_type == SeqType.NUCLEOTIDE else _AA_DECODE_BYTES
        )
        aligned_a, aligned_b, score, n_matches, n_mismatches, n_gaps = _c_nw_align(
            codes_a, codes_b, decode_bytes,
            int(cls.MATCH), int(cls.MISMATCH), int(cls.GAP),
            mode,
        )
        mutations = cls._detect_mutations(aligned_a, aligned_b)
        aln_len   = n_matches + n_mismatches + n_gaps
        identity  = n_matches / aln_len if aln_len else 0.0
        return AlignmentResult(
            score        = score,
            identity     = identity,
            n_matches    = n_matches,
            n_mismatches = n_mismatches,
            n_gaps       = n_gaps,
            mutations    = mutations,
            aligned_a    = aligned_a,
            aligned_b    = aligned_b,
            seq_type     = seq_type,
            mode         = mode,
        )

    @staticmethod
    def _detect_mutations(aligned_a: str, aligned_b: str) -> list[Mutation]:
        mutations: list[Mutation] = []
        pos_a = pos_b = 0
        for sa, sb in zip(aligned_a, aligned_b):
            if sa == '-':
                mutations.append(Mutation('insertion', pos_a, pos_b, '-', sb))
                pos_b += 1
            elif sb == '-':
                mutations.append(Mutation('deletion', pos_a, pos_b, sa, '-'))
                pos_a += 1
            else:
                if sa != sb:
                    mutations.append(Mutation('substitution', pos_a, pos_b, sa, sb))
                pos_a += 1
                pos_b += 1
        return mutations

    # ── Scoring matrix (anti-diagonal wavefront) ──────────────────────────────

    @classmethod
    def _fill_matrix(
        cls,
        codes_a: np.ndarray,
        codes_b: np.ndarray,
        m: int,
        n: int,
        mode: str,
    ) -> np.ndarray:
        """
        Fill the (m+1)×(n+1) NW matrix using the anti-diagonal wavefront.

        Anti-diagonal d = i + j.  For each d ≥ 2 the computable cells
        satisfy 1 ≤ i ≤ m  and  1 ≤ j = d−i ≤ n, i.e.:
            i ∈ [max(1, d−n) … min(m, d−1)]

        All cells in one anti-diagonal are independent and are computed
        in a single vectorised NumPy step.

        Returns
        -------
        np.ndarray, dtype int32, shape (m+1, n+1)
        """
        H = np.zeros((m + 1, n + 1), dtype=np.int32)

        if mode == 'global':
            H[0, :] = np.arange(n + 1, dtype=np.int32) * cls.GAP
            H[:, 0] = np.arange(m + 1, dtype=np.int32) * cls.GAP
        # semi-global: borders stay at 0 (free terminal gaps on the query)

        # Pre-allocate index buffers once outside the loop.
        # i_full[k] = k+1 → i_full[i_lo-1 : i_hi] is a zero-copy slice [i_lo…i_hi].
        # j_buf receives d − i_arr in-place, eliminating one allocation per diagonal.
        i_full = np.arange(1, max(m, n) + 1, dtype=np.int32)
        j_buf  = np.empty(min(m, n) + 1,     dtype=np.int32)

        for d in range(2, m + n + 1):
            i_lo = max(1, d - n)
            i_hi = min(m, d - 1)   # j = d−i ≥ 1  ⟹  i ≤ d−1
            if i_lo > i_hi:
                continue

            w     = i_hi - i_lo + 1
            i_arr = i_full[i_lo - 1: i_hi]          # zero-copy slice — no allocation
            np.subtract(d, i_arr, out=j_buf[:w])    # in-place: avoids j = d - i_arr copy
            j_arr = j_buf[:w]

            # ① Match/mismatch score per cell
            match_mask = codes_a[i_arr - 1] == codes_b[j_arr - 1]
            step = np.where(match_mask, cls.MATCH, cls.MISMATCH).astype(np.int32)

            # ② Three candidate moves
            diag = H[i_arr - 1, j_arr - 1] + step     # ↖ match/mismatch
            up   = H[i_arr - 1, j_arr    ] + cls.GAP  # ↑ gap in seq_b
            left = H[i_arr,     j_arr - 1] + cls.GAP  # ← gap in seq_a

            # ③ Cell = maximum of three
            H[i_arr, j_arr] = np.maximum(np.maximum(diag, up), left)

        return H

    @classmethod
    def _fill_matrix_banded(
        cls,
        codes_a: np.ndarray,
        codes_b: np.ndarray,
        m: int,
        n: int,
        band: int,
        mode: str,
    ) -> np.ndarray:
        """
        Fill the NW matrix restricted to the band |i-j| ≤ band.

        Out-of-band cells are initialised to NEG_INF so they can never
        be chosen as optimal predecessors.  The anti-diagonal wavefront
        is clipped to the band intersection on each diagonal.
        """
        NEG = np.int32(-10 ** 9)
        H = np.full((m + 1, n + 1), NEG, dtype=np.int32)

        H[0, 0] = np.int32(0)
        for j in range(1, min(n, band) + 1):
            H[0, j] = np.int32(0) if mode == 'semi-global' else np.int32(j) * cls.GAP
        for i in range(1, min(m, band) + 1):
            H[i, 0] = np.int32(i) * cls.GAP

        i_full = np.arange(1, max(m, n) + 1, dtype=np.int32)
        j_buf  = np.empty(min(m, n) + 1, dtype=np.int32)

        for d in range(2, m + n + 1):
            i_lo = max(1, d - n, (d - band + 1) // 2)
            i_hi = min(m, d - 1, (d + band)     // 2)
            if i_lo > i_hi:
                continue

            w     = i_hi - i_lo + 1
            i_arr = i_full[i_lo - 1: i_hi]
            np.subtract(d, i_arr, out=j_buf[:w])
            j_arr = j_buf[:w]

            dv = H[i_arr - 1, j_arr - 1]
            uv = H[i_arr - 1, j_arr]
            lv = H[i_arr,     j_arr - 1]

            match_mask = codes_a[i_arr - 1] == codes_b[j_arr - 1]
            step = np.where(match_mask, cls.MATCH, cls.MISMATCH).astype(np.int32)

            d_score = np.where(dv != NEG, dv + step, NEG)
            u_score = np.where(uv != NEG, uv + cls.GAP, NEG)
            l_score = np.where(lv != NEG, lv + cls.GAP, NEG)

            H[i_arr, j_arr] = np.maximum(np.maximum(d_score, u_score), l_score)

        return H

    @classmethod
    def _fill_matrix_sw(
        cls,
        codes_a: np.ndarray,
        codes_b: np.ndarray,
        m: int,
        n: int,
    ) -> np.ndarray:
        """
        Fill the Smith-Waterman scoring matrix.

        Identical to NW except cells are floored at 0.
        Uses the same anti-diagonal wavefront strategy (O(m+n) iterations).
        """
        H = np.zeros((m + 1, n + 1), dtype=np.int32)
        # Borders stay 0 (SW boundary condition)

        i_full = np.arange(1, max(m, n) + 1, dtype=np.int32)
        j_buf  = np.empty(min(m, n) + 1, dtype=np.int32)

        for d in range(2, m + n + 1):
            i_lo = max(1, d - n)
            i_hi = min(m, d - 1)
            if i_lo > i_hi:
                continue

            w     = i_hi - i_lo + 1
            i_arr = i_full[i_lo - 1: i_hi]
            np.subtract(d, i_arr, out=j_buf[:w])
            j_arr = j_buf[:w]

            match_mask = codes_a[i_arr - 1] == codes_b[j_arr - 1]
            step = np.where(match_mask, cls.MATCH, cls.MISMATCH).astype(np.int32)

            diag = H[i_arr - 1, j_arr - 1] + step
            up   = H[i_arr - 1, j_arr    ] + cls.GAP
            left = H[i_arr,     j_arr - 1] + cls.GAP

            # SW: floor at 0
            H[i_arr, j_arr] = np.maximum(
                np.int32(0), np.maximum(np.maximum(diag, up), left)
            )

        return H

    @classmethod
    def _traceback_sw(
        cls,
        H:       np.ndarray,
        codes_a: np.ndarray,
        codes_b: np.ndarray,
        m: int,
        n: int,
        seq_type: SeqType,
    ) -> AlignmentResult:
        """Traceback for Smith-Waterman: start at max(H), stop at 0."""
        decode = _NUC_DECODE if seq_type == SeqType.NUCLEOTIDE else _AA_DECODE

        max_idx   = np.argmax(H)
        score     = int(H.flat[max_idx])
        start_i, start_j = divmod(int(max_idx), n + 1)

        i, j = start_i, start_j
        aln_a:     list[str]      = []
        aln_b:     list[str]      = []
        mutations: list[Mutation] = []
        n_matches = n_mismatches = n_gaps = 0

        while i > 0 and j > 0 and H[i, j] > 0:
            step = cls.MATCH if codes_a[i-1] == codes_b[j-1] else cls.MISMATCH

            if H[i, j] == H[i-1, j-1] + step:
                sa = decode.get(int(codes_a[i-1]), '?')
                sb = decode.get(int(codes_b[j-1]), '?')
                aln_a.append(sa); aln_b.append(sb)
                if codes_a[i-1] == codes_b[j-1]:
                    n_matches += 1
                else:
                    n_mismatches += 1
                    mutations.append(Mutation('substitution', i - 1, j - 1, sa, sb))
                i -= 1; j -= 1

            elif H[i, j] == H[i-1, j] + cls.GAP:
                sa = decode.get(int(codes_a[i-1]), '?')
                aln_a.append(sa); aln_b.append('-')
                n_gaps += 1
                mutations.append(Mutation('deletion', i - 1, j, sa, '-'))
                i -= 1

            else:
                sb = decode.get(int(codes_b[j-1]), '?')
                aln_a.append('-'); aln_b.append(sb)
                n_gaps += 1
                mutations.append(Mutation('insertion', i, j - 1, '-', sb))
                j -= 1

        aln_a.reverse(); aln_b.reverse(); mutations.reverse()
        aln_len  = n_matches + n_mismatches + n_gaps
        identity = n_matches / aln_len if aln_len else 0.0

        return AlignmentResult(
            score=score, identity=identity,
            n_matches=n_matches, n_mismatches=n_mismatches, n_gaps=n_gaps,
            mutations=mutations,
            aligned_a=''.join(aln_a), aligned_b=''.join(aln_b),
            seq_type=seq_type, mode='local',
        )

    # ── Traceback ─────────────────────────────────────────────────────────────

    @classmethod
    def _traceback(
        cls,
        H:        np.ndarray,
        codes_a:  np.ndarray,
        codes_b:  np.ndarray,
        m: int,
        n: int,
        mode: str,
        seq_type: SeqType,
    ) -> AlignmentResult:
        """
        Reconstruct the optimal alignment by walking back through H.

        O(m+n) sequential loop — the step dependency makes vectorisation
        impossible here.  Each iteration is O(1) scalar work.
        """
        decode = _NUC_DECODE if seq_type == SeqType.NUCLEOTIDE else _AA_DECODE

        # ── Starting cell ──────────────────────────────────────────────────────
        if mode == 'global':
            i, j = m, n
        else:
            # Semi-global: pick the best score in the last row or last column.
            j_best_row = int(np.argmax(H[m, :]))
            i_best_col = int(np.argmax(H[:, n]))
            if H[m, j_best_row] >= H[i_best_col, n]:
                i, j = m, j_best_row
            else:
                i, j = i_best_col, n

        score = int(H[i, j])

        aln_a:     list[str]      = []
        aln_b:     list[str]      = []
        mutations: list[Mutation] = []
        n_matches = n_mismatches = n_gaps = 0

        # ── Traceback loop ─────────────────────────────────────────────────────
        while i > 0 or j > 0:

            if i > 0 and j > 0:
                step = cls.MATCH if codes_a[i-1] == codes_b[j-1] else cls.MISMATCH

                if H[i, j] == H[i-1, j-1] + step:
                    # Diagonal — match or mismatch
                    sa = decode.get(int(codes_a[i-1]), '?')
                    sb = decode.get(int(codes_b[j-1]), '?')
                    aln_a.append(sa)
                    aln_b.append(sb)
                    if codes_a[i-1] == codes_b[j-1]:
                        n_matches += 1
                    else:
                        n_mismatches += 1
                        mutations.append(
                            Mutation('substitution', i - 1, j - 1, sa, sb)
                        )
                    i -= 1; j -= 1

                elif H[i, j] == H[i-1, j] + cls.GAP:
                    # Up — gap in seq_b (deletion in query vs reference)
                    sa = decode.get(int(codes_a[i-1]), '?')
                    aln_a.append(sa)
                    aln_b.append('-')
                    n_gaps += 1
                    mutations.append(Mutation('deletion', i - 1, j, sa, '-'))
                    i -= 1

                else:
                    # Left — gap in seq_a (insertion in query vs reference)
                    sb = decode.get(int(codes_b[j-1]), '?')
                    aln_a.append('-')
                    aln_b.append(sb)
                    n_gaps += 1
                    mutations.append(Mutation('insertion', i, j - 1, '-', sb))
                    j -= 1

            elif i > 0:
                # Terminal: remaining seq_a consumed as gap in seq_b
                sa = decode.get(int(codes_a[i-1]), '?')
                aln_a.append(sa)
                aln_b.append('-')
                if mode == 'global':
                    n_gaps += 1
                    mutations.append(Mutation('deletion', i - 1, 0, sa, '-'))
                i -= 1

            else:
                # Terminal: remaining seq_b consumed as gap in seq_a
                sb = decode.get(int(codes_b[j-1]), '?')
                aln_a.append('-')
                aln_b.append(sb)
                if mode == 'global':
                    n_gaps += 1
                    mutations.append(Mutation('insertion', 0, j - 1, '-', sb))
                j -= 1

        # Traceback builds alignment end→start; reverse to restore order.
        aln_a.reverse()
        aln_b.reverse()
        mutations.reverse()

        aln_len  = n_matches + n_mismatches + n_gaps
        identity = n_matches / aln_len if aln_len else 0.0

        return AlignmentResult(
            score        = score,
            identity     = identity,
            n_matches    = n_matches,
            n_mismatches = n_mismatches,
            n_gaps       = n_gaps,
            mutations    = mutations,
            aligned_a    = ''.join(aln_a),
            aligned_b    = ''.join(aln_b),
            seq_type     = seq_type,
            mode         = mode,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §3  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def format_alignment(result: AlignmentResult, width: int = 60) -> str:
    """
    Return a human-readable block alignment string.

    Symbol legend: '|' match · 'X' mismatch · ' ' gap.
    Vectorised: builds the midline with NumPy, no Python character loop.
    """
    if width <= 0:
        raise AlignmentError(f"width debe ser > 0, se recibió {width}.")
    a, b = result.aligned_a, result.aligned_b
    if len(a) != len(b):
        raise AlignmentError(
            f"aligned_a y aligned_b tienen longitudes distintas: {len(a)} ≠ {len(b)}."
        )
    a_arr = np.frombuffer(a.encode(), dtype=np.uint8)
    b_arr = np.frombuffer(b.encode(), dtype=np.uint8)
    gap   = np.uint8(ord('-'))
    mid   = np.full(len(a_arr), ord(' '), dtype=np.uint8)
    match = (a_arr == b_arr) & (a_arr != gap)
    mism  = (a_arr != b_arr) & (a_arr != gap) & (b_arr != gap)
    mid[match] = ord('|')
    mid[mism]  = ord('X')
    midline = mid.tobytes().decode('ascii')
    lines: list[str] = []
    for s in range(0, len(a), width):
        e = s + width
        lines += [
            f"  A: {a[s:e]}",
            f"     {midline[s:e]}",
            f"  B: {b[s:e]}",
            "",
        ]
    return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# §4  DEMO / SELF-TEST   (python aligner.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import time
    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from bioforge import SmartImporter

    W = 65
    print("═" * W)
    print("  aligner.py — Needleman-Wunsch anti-diagonal wavefront demo")
    print("═" * W)

    # ── Test 1: secuencias idénticas (sanity check) ────────────────────────
    print("\n  ── Test 1: secuencias idénticas " + "─" * 32)
    _SEQ  = "ATGGTGCACCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGG"
    _FASTA = f">a\n{_SEQ}\n>b\n{_SEQ}\n"
    _seqs  = SmartImporter.from_string(_FASTA, force_type=SeqType.NUCLEOTIDE)
    r1     = SequenceAligner.align(_seqs[0], _seqs[1])
    assert r1.n_mismatches == 0 and r1.n_gaps == 0, "❌ deberían ser idénticas"
    assert r1.identity == 1.0
    assert r1.score == len(_SEQ) * int(SequenceAligner.MATCH)
    print(f"  Score={r1.score}  Identity={r1.identity:.0%}  "
          f"Mutaciones={len(r1.mutations)}  ✅")

    # ── Test 2: HBB normal vs sickle cell (única sustitución A→T, pos 19) ──
    print("\n  ── Test 2: HBB normal vs sickle cell (A19T) " + "─" * 20)
    _NORMAL = ">HBB_normal\nATGGTGCACCTGACTCCTGAGGAGAAGTCT\n"
    _SICKLE = ">HBB_sickle\nATGGTGCACCTGACTCCTGTGGAGAAGTCT\n"
    _n = SmartImporter.from_string(_NORMAL, force_type=SeqType.NUCLEOTIDE)[0]
    _s = SmartImporter.from_string(_SICKLE, force_type=SeqType.NUCLEOTIDE)[0]
    r2  = SequenceAligner.align(_n, _s)
    print(f"  Score={r2.score}  Identity={r2.identity:.1%}  "
          f"Mismatches={r2.n_mismatches}  Gaps={r2.n_gaps}")
    print(format_alignment(r2))
    assert len(r2.mutations) == 1
    assert r2.mutations[0].kind  == 'substitution'
    assert r2.mutations[0].pos_a == 19
    assert r2.mutations[0].sym_a == 'A' and r2.mutations[0].sym_b == 'T'
    print(f"  Mutación detectada: {r2.mutations[0]}  ✅")

    # ── Test 3: inserción de 3 bases (codón extra) ─────────────────────────
    print("\n  ── Test 3: inserción de 3 bases " + "─" * 33)
    _REF = ">ref\nATGGTGCACCTGACTGAA\n"          # 18 nt
    _INS = ">ins\nATGGTGCACCTGACTCCCGAA\n"        # 21 nt  (CCC en pos 15)
    _r   = SmartImporter.from_string(_REF, force_type=SeqType.NUCLEOTIDE)[0]
    _i   = SmartImporter.from_string(_INS, force_type=SeqType.NUCLEOTIDE)[0]
    r3   = SequenceAligner.align(_r, _i)
    print(f"  Score={r3.score}  Identity={r3.identity:.1%}  "
          f"Mismatches={r3.n_mismatches}  Gaps={r3.n_gaps}")
    print(format_alignment(r3))
    _ins_muts = [m for m in r3.mutations if m.kind == 'insertion']
    print(f"  Inserciones detectadas: {len(_ins_muts)} base(s)  "
          f"{'✅' if _ins_muts else '⚠  (ninguna)'}")

    # ── Test 4: alineamiento de proteínas (HBB α vs β) ────────────────────
    print("\n  ── Test 4: proteínas HBB-alpha vs HBB-beta " + "─" * 21)
    _HBB = ">HBB\nMVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLST\n"
    _HBA = ">HBA\nMVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLS\n"
    _hbb = SmartImporter.from_string(_HBB, force_type=SeqType.PROTEIN)[0]
    _hba = SmartImporter.from_string(_HBA, force_type=SeqType.PROTEIN)[0]
    r4   = SequenceAligner.align(_hbb, _hba)
    print(f"  Score={r4.score}  Identity={r4.identity:.1%}  "
          f"Matches={r4.n_matches}  Mismatches={r4.n_mismatches}  "
          f"Gaps={r4.n_gaps}")
    print(format_alignment(r4))
    print(f"  Mutaciones totales: {len(r4.mutations)}")
    for _m in r4.mutations[:5]:
        print(f"    {_m}")
    if len(r4.mutations) > 5:
        print(f"    … y {len(r4.mutations) - 5} más")

    # ── Test 5: rutas de error ─────────────────────────────────────────────
    print("\n  ── Test 5: rutas de error " + "─" * 39)
    _prot = PackedSequence(
        header="p", seq_type=SeqType.PROTEIN, n_symbols=4,
        data=BitPacker.pack(np.array([4, 5, 6, 7], dtype=np.uint8)),
    )
    _nuc = SmartImporter.from_string(">n\nACGT\n",
                                     force_type=SeqType.NUCLEOTIDE)[0]
    try:
        SequenceAligner.align(_prot, _nuc)
        print("  TypeError no lanzado ❌")
    except TypeError as exc:
        print(f"  TypeError ✅  → {exc}")

    _empty = PackedSequence(
        header="e", seq_type=SeqType.NUCLEOTIDE, n_symbols=1,
        data=BitPacker.pack(np.array([0], dtype=np.uint8)),
    )
    _empty2 = PackedSequence(
        header="e2", seq_type=SeqType.NUCLEOTIDE, n_symbols=1,
        data=BitPacker.pack(np.array([0], dtype=np.uint8)),
    )
    r_tiny = SequenceAligner.align(_empty, _empty2)
    assert r_tiny.score == int(SequenceAligner.MATCH)
    print(f"  Alineamiento 1×1 (A vs A): score={r_tiny.score}  ✅")

    # ── Test 6: benchmark — 1 000 × 1 000 nt con 1 % mutaciones ───────────
    print(f"\n  ── Test 6: benchmark 1 000 × 1 000 nt (1 % mutaciones) " + "─" * 8)
    _rng    = np.random.default_rng(42)
    _bases  = np.array([0, 1, 2, 3], dtype=np.uint8)
    _ca     = _rng.choice(_bases, size=1000)
    _cb     = _ca.copy()
    _mutpos = _rng.choice(1000, size=10, replace=False)
    for _p in _mutpos:
        _cb[_p] = (_cb[_p] + 1) % 4   # cyclic 1-step substitution

    _ps_a = PackedSequence(
        header="bench_a", seq_type=SeqType.NUCLEOTIDE,
        n_symbols=1000, data=BitPacker.pack(_ca),
    )
    _ps_b = PackedSequence(
        header="bench_b", seq_type=SeqType.NUCLEOTIDE,
        n_symbols=1000, data=BitPacker.pack(_cb),
    )

    _t0 = time.perf_counter()
    _rb  = SequenceAligner.align(_ps_a, _ps_b)
    _t1  = time.perf_counter()

    _elapsed_ms = (_t1 - _t0) * 1e3
    print(f"  Tiempo      : {_elapsed_ms:.1f} ms")
    print(f"  Score       : {_rb.score}")
    print(f"  Identity    : {_rb.identity:.2%}")
    print(f"  Mutaciones  : {len(_rb.mutations)}  (esperadas ≈ 10)")
    assert abs(len(_rb.mutations) - 10) <= 2, "Mutaciones fuera de rango esperado"
    print(f"  ✅  Benchmark superado")

    print(f"\n{'═' * W}\n")
