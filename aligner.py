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
>>> from biocore import SmartImporter, SeqType
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

from biocore import BioCode, BitPacker, PackedSequence, SeqType
from biocore import _NUC_DECODE, _AA_DECODE   # within-project private dicts


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
    ) -> AlignmentResult:
        """
        Align seq_a (reference) against seq_b (query).

        Mutations are reported relative to seq_a: a 'deletion' means
        seq_b is missing a symbol that seq_a has; an 'insertion' means
        seq_b carries an extra symbol absent from seq_a.

        Parameters
        ----------
        seq_a : PackedSequence — reference.
        seq_b : PackedSequence — query.
        mode  : 'global'      — penalises all terminal gaps (use for
                                sequences of similar length).
                'semi-global' — free terminal gaps on the query side;
                                use when seq_b is a fragment of seq_a.

        Returns
        -------
        AlignmentResult

        Raises
        ------
        TypeError  — seq_a.seq_type ≠ seq_b.seq_type.
        ValueError — either sequence is empty.
        """
        if seq_a.seq_type != seq_b.seq_type:
            raise TypeError(
                f"Los tipos no coinciden: "
                f"{seq_a.seq_type.name} ≠ {seq_b.seq_type.name}."
            )
        if seq_a.n_symbols == 0 or seq_b.n_symbols == 0:
            raise ValueError("Las secuencias no pueden estar vacías.")

        m, n = seq_a.n_symbols, seq_b.n_symbols
        if m > cls._MAX_SAFE_LEN or n > cls._MAX_SAFE_LEN:
            mem_mb = (m + 1) * (n + 1) * 4 / 1_000_000
            warnings.warn(
                f"Secuencia larga ({max(m, n):,} síms → "
                f"matriz DP ≈ {mem_mb:.0f} MB). "
                f"Considera alineamiento por bandas para secuencias "
                f"> {cls._MAX_SAFE_LEN:,} símbolos.",
                UserWarning,
                stacklevel=2,
            )

        codes_a = seq_a.decode()   # (m,) uint8
        codes_b = seq_b.decode()   # (n,) uint8

        H = cls._fill_matrix(codes_a, codes_b, m, n, mode)
        return cls._traceback(H, codes_a, codes_b, m, n, mode, seq_a.seq_type)

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

        for d in range(2, m + n + 1):
            i_lo = max(1, d - n)
            i_hi = min(m, d - 1)   # j = d−i ≥ 1  ⟹  i ≤ d−1
            if i_lo > i_hi:
                continue

            i_arr = np.arange(i_lo, i_hi + 1, dtype=np.int32)
            j_arr = d - i_arr                                    # shape (w,)

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

                if int(H[i, j]) == int(H[i-1, j-1]) + int(step):
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

                elif int(H[i, j]) == int(H[i-1, j]) + int(cls.GAP):
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
    """
    a, b   = result.aligned_a, result.aligned_b
    midline = ''.join(
        '|' if ca == cb and ca != '-' else
        'X' if ca != '-' and cb != '-' else
        ' '
        for ca, cb in zip(a, b)
    )
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
    from biocore import SmartImporter

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
