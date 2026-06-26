"""
biocore.py
══════════════════════════════════════════════════════════════════════
Core engine for a bioinformatics library built on a unified 5-bit
biological alphabet.

Design principles
─────────────────
• No biological sequence is ever stored as a Python ``str`` in memory.
• All sequences live as compact, bit-packed NumPy uint8 arrays.
• All encode/decode operations are fully vectorised — zero Python loops.
• Sequences are read-only (numpy write-lock) after construction.

Unified 5-bit encoding map  (32 possible states)
─────────────────────────────────────────────────
 State │ Binary  │ Symbol
───────┼─────────┼──────────────────────────────────────────────────
   0   │ 00000   │ Adenine           ─ nucleotide  A
   1   │ 00001   │ Cytosine          ─ nucleotide  C
   2   │ 00010   │ Guanine           ─ nucleotide  G
   3   │ 00011   │ Thymine / Uracil  ─ nucleotide  T / U (shared)
   4   │ 00100   │ Alanine           ─ amino acid  A
   5   │ 00101   │ Cysteine          ─ amino acid  C
   6   │ 00110   │ Aspartic acid     ─ amino acid  D
   7   │ 00111   │ Glutamic acid     ─ amino acid  E
   8   │ 01000   │ Phenylalanine     ─ amino acid  F
   9   │ 01001   │ Glycine           ─ amino acid  G
  10   │ 01010   │ Histidine         ─ amino acid  H
  11   │ 01011   │ Isoleucine        ─ amino acid  I
  12   │ 01100   │ Lysine            ─ amino acid  K
  13   │ 01101   │ Leucine           ─ amino acid  L
  14   │ 01110   │ Methionine        ─ amino acid  M
  15   │ 01111   │ Asparagine        ─ amino acid  N
  16   │ 10000   │ Proline           ─ amino acid  P
  17   │ 10001   │ Glutamine         ─ amino acid  Q
  18   │ 10010   │ Arginine          ─ amino acid  R
  19   │ 10011   │ Serine            ─ amino acid  S
  20   │ 10100   │ Threonine         ─ amino acid  T
  21   │ 10101   │ Valine            ─ amino acid  V
  22   │ 10110   │ Tryptophan        ─ amino acid  W
  23   │ 10111   │ Tyrosine          ─ amino acid  Y
  24   │ 11000   │ STOP codon / chain terminator  *
  25   │ 11001   │ Alignment gap     ─             -
 26–30 │   …     │ Reserved for future extension
  31   │ 11111   │ Unknown / ambiguous  (N in DNA, X in protein)
───────┴─────────┴──────────────────────────────────────────────────
Memory savings over plain ASCII: 5 bits/symbol → 37.5 % reduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator, Optional

import numpy as np

try:
    from engine._loader import C_AVAILABLE as _C_AVAILABLE
    from engine._loader import c_getitem5 as _c_getitem5
    from engine._loader import c_pack5    as _c_pack5
    from engine._loader import c_unpack5  as _c_unpack5
except ImportError:
    _C_AVAILABLE = False


__all__: list[str] = [
    "BioCode",
    "SeqType",
    "NUC_LUT",
    "AA_LUT",
    "BitPacker",
    "PackedSequence",
    "SmartImporter",
    "SequenceStats",
    "compute_stats",
]


# ══════════════════════════════════════════════════════════════════════════════
# §1  BIOLOGICAL ALPHABET  —  5-bit codes  (values 0 … 31)
# ══════════════════════════════════════════════════════════════════════════════

class BioCode(IntEnum):
    """
    Unified 5-bit code for every biological symbol supported by the library.

    Guaranteed range [0, 31].  Stored as ``np.uint8`` in packed arrays,
    where the upper 3 bits are always zero after unpacking.
    """

    # ── Nucleotides (states 0–3) ──────────────────────────────────────────────
    NUC_A  =  0   # Adenine
    NUC_C  =  1   # Cytosine
    NUC_G  =  2   # Guanine
    NUC_TU =  3   # Thymine (DNA) / Uracil (RNA) — unified slot

    # ── Amino acids (states 4–23), alphabetical IUPAC 1-letter order ──────────
    AA_A   =  4   # Alanine        (A)
    AA_C   =  5   # Cysteine       (C)
    AA_D   =  6   # Aspartic acid  (D)
    AA_E   =  7   # Glutamic acid  (E)
    AA_F   =  8   # Phenylalanine  (F)
    AA_G   =  9   # Glycine        (G)
    AA_H   = 10   # Histidine      (H)
    AA_I   = 11   # Isoleucine     (I)
    AA_K   = 12   # Lysine         (K)
    AA_L   = 13   # Leucine        (L)
    AA_M   = 14   # Methionine     (M)
    AA_N   = 15   # Asparagine     (N)
    AA_P   = 16   # Proline        (P)
    AA_Q   = 17   # Glutamine      (Q)
    AA_R   = 18   # Arginine       (R)
    AA_S   = 19   # Serine         (S)
    AA_T   = 20   # Threonine      (T)
    AA_V   = 21   # Valine         (V)
    AA_W   = 22   # Tryptophan     (W)
    AA_Y   = 23   # Tyrosine       (Y)

    # ── Special / terminal states (24–31) ────────────────────────────────────
    STOP   = 24   # Stop codon / chain terminator   *
    GAP    = 25   # Alignment gap                   -
    #        26–30 reserved (e.g. modified bases, selenocysteine, pyrrolysine)
    UNK    = 31   # Unknown / ambiguous  (N in DNA, X in protein)


class SeqType(IntEnum):
    """Biological sequence alphabet family."""
    NUCLEOTIDE = 0   # DNA or RNA
    PROTEIN    = 1   # amino-acid chain


# ══════════════════════════════════════════════════════════════════════════════
# §2  LOOKUP TABLES  —  constant-time  ASCII ↔ BioCode  translation
# ══════════════════════════════════════════════════════════════════════════════

def _build_encode_lut(
    char_map: dict[str, int],
    default:  int = BioCode.UNK,
) -> np.ndarray:
    """
    Build a 256-element uint8 lookup array indexed by ASCII ordinal.

    A single ``lut[raw_uint8_array]`` call translates an entire sequence
    in one vectorised numpy step — no Python-level loop required.

    Parameters
    ----------
    char_map : dict[str, int]
        Character → BioCode mapping; applied to both upper and lower case.
    default : int
        BioCode assigned to any character absent from *char_map*.

    Returns
    -------
    np.ndarray, dtype=uint8, shape=(256,)
    """
    lut = np.full(256, default, dtype=np.uint8)
    for ch, code in char_map.items():
        lut[ord(ch.upper())] = code
        lut[ord(ch.lower())] = code
    return lut


# ── Nucleotide encode table ────────────────────────────────────────────────────
NUC_LUT: np.ndarray = _build_encode_lut({
    "A": BioCode.NUC_A,
    "C": BioCode.NUC_C,
    "G": BioCode.NUC_G,
    "T": BioCode.NUC_TU,   # DNA thymine
    "U": BioCode.NUC_TU,   # RNA uracil → same slot as T
    "N": BioCode.UNK,       # IUPAC ambiguous nucleotide
    "-": BioCode.GAP,
    ".": BioCode.GAP,
})

# ── Amino-acid encode table ────────────────────────────────────────────────────
AA_LUT: np.ndarray = _build_encode_lut({
    "A": BioCode.AA_A,  "C": BioCode.AA_C,  "D": BioCode.AA_D,
    "E": BioCode.AA_E,  "F": BioCode.AA_F,  "G": BioCode.AA_G,
    "H": BioCode.AA_H,  "I": BioCode.AA_I,  "K": BioCode.AA_K,
    "L": BioCode.AA_L,  "M": BioCode.AA_M,  "N": BioCode.AA_N,
    "P": BioCode.AA_P,  "Q": BioCode.AA_Q,  "R": BioCode.AA_R,
    "S": BioCode.AA_S,  "T": BioCode.AA_T,  "V": BioCode.AA_V,
    "W": BioCode.AA_W,  "Y": BioCode.AA_Y,
    "*": BioCode.STOP,
    "-": BioCode.GAP,
    "X": BioCode.UNK,
})

# ── Decode maps  (BioCode int → canonical single-letter character) ─────────────
_NUC_DECODE: dict[int, str] = {
    BioCode.NUC_A:  "A", BioCode.NUC_C:  "C",
    BioCode.NUC_G:  "G", BioCode.NUC_TU: "T",
    BioCode.GAP:    "-", BioCode.UNK:    "N",
}
_AA_DECODE: dict[int, str] = {
    BioCode.AA_A: "A", BioCode.AA_C: "C", BioCode.AA_D: "D",
    BioCode.AA_E: "E", BioCode.AA_F: "F", BioCode.AA_G: "G",
    BioCode.AA_H: "H", BioCode.AA_I: "I", BioCode.AA_K: "K",
    BioCode.AA_L: "L", BioCode.AA_M: "M", BioCode.AA_N: "N",
    BioCode.AA_P: "P", BioCode.AA_Q: "Q", BioCode.AA_R: "R",
    BioCode.AA_S: "S", BioCode.AA_T: "T", BioCode.AA_V: "V",
    BioCode.AA_W: "W", BioCode.AA_Y: "Y", BioCode.STOP: "*",
    BioCode.GAP:  "-", BioCode.UNK:  "X",
}

# ── Protein-exclusive character LUT  (for vectorised type auto-detection) ───────
# Characters that appear in protein FASTA but NOT in the standard IUPAC
# nucleotide alphabet  {A C G T U R Y S W K M B D H V N - .}.
# Conservative set to minimise false positives on degenerate-nucleotide files.
_IS_PROTEIN_CHAR: np.ndarray = np.zeros(256, dtype=np.bool_)
for _ch in "EFILPQefilpq*":
    _IS_PROTEIN_CHAR[ord(_ch)] = True

# ── Vectorised decode LUTs  (BioCode index → ASCII byte) ────────────────────────
# Pre-built once at module load.  to_string() indexes into these arrays with a
# single fancy-index op, then calls .tobytes().decode('ascii') — no Python loop.
_NUC_DECODE_ARR: np.ndarray = np.full(32, ord('?'), dtype=np.uint8)
for _code, _char in _NUC_DECODE.items():
    _NUC_DECODE_ARR[int(_code)] = ord(_char)

_AA_DECODE_ARR: np.ndarray = np.full(32, ord('?'), dtype=np.uint8)
for _code, _char in _AA_DECODE.items():
    _AA_DECODE_ARR[int(_code)] = ord(_char)


# ══════════════════════════════════════════════════════════════════════════════
# §3  BIT PACKER  —  compact 5-bit ↔ uint8 array conversion
# ══════════════════════════════════════════════════════════════════════════════

class BitPacker:
    """
    Stateless utility for lossless 5-bit dense packing.

    **Bit layout** — MSB-first big-endian bit stream::

        sym₀[b₄b₃b₂b₁b₀]  sym₁[b₄b₃b₂b₁b₀]  sym₂ …
        ──────────────────────────────────────────────────
        byte₀[b₇b₆b₅b₄b₃b₂b₁b₀]   byte₁ …

    For 8 symbols (= 40 bits = 5 bytes) the stream is byte-aligned with
    zero padding overhead.  Any other length gets ≤ 7 zero-padding bits
    appended in the final byte, transparent to the caller.

    Both ``pack`` and ``unpack`` are fully vectorised — no Python loops.
    """

    # Pre-computed constants — allocated once at class definition time.
    _SHIFTS:  np.ndarray = np.array([4, 3, 2, 1, 0], dtype=np.uint8)  # MSB→LSB
    _WEIGHTS: np.ndarray = np.array([16, 8, 4, 2, 1], dtype=np.uint8) # reconstruction

    # ── Pack ──────────────────────────────────────────────────────────────────

    @staticmethod
    def pack(codes: np.ndarray) -> np.ndarray:
        """
        Pack 5-bit BioCode values into a compact uint8 byte array.

        Parameters
        ----------
        codes : np.ndarray, dtype uint8, values in [0, 31]
            Sequence of BioCode integers to compress.

        Returns
        -------
        np.ndarray, dtype uint8
            Packed byte array.  ``len`` = ⌈``len(codes)`` × 5 / 8⌉.
            Trailing bits in the last byte are zero-padded.

        Complexity
        ──────────
        Time  : O(n) — two vectorised numpy ops.
        Memory: O(n) — peak ≈ 5n bits for the intermediate bit matrix.
        """
        if _C_AVAILABLE:
            return _c_pack5(codes)

        # ①  Expand each 5-bit code → one row of a (n, 5) bit matrix, MSB first.
        bits: np.ndarray = (
            (codes[:, np.newaxis] >> BitPacker._SHIFTS) & np.uint8(1)
        )                                               # shape: (n, 5), dtype uint8

        # ②  Flatten to a 1-D bit stream; numpy packs every 8 bits → 1 byte.
        return np.packbits(bits.ravel())

    # ── Unpack ────────────────────────────────────────────────────────────────

    @staticmethod
    def unpack(packed: np.ndarray, n: int) -> np.ndarray:
        """
        Unpack a 5-bit packed byte array back to BioCode values.

        Parameters
        ----------
        packed : np.ndarray, dtype uint8
            Byte array as returned by :meth:`pack`.
        n : int
            Original symbol count (required to trim padding bits).

        Returns
        -------
        np.ndarray, dtype uint8, shape (n,), values in [0, 31]
        """
        if _C_AVAILABLE:
            return _c_unpack5(packed, n)

        # ①  Expand all bytes → individual bits; keep exactly n×5 of them.
        bits: np.ndarray = np.unpackbits(packed)[: n * 5].reshape(n, 5)

        # ②  Dot each row with [16, 8, 4, 2, 1] to restore the 5-bit integer.
        return (bits * BitPacker._WEIGHTS).sum(axis=1, dtype=np.uint8)

    @staticmethod
    def packed_size(n_symbols: int) -> int:
        """Minimum byte count needed to store ``n_symbols`` 5-bit codes."""
        return (n_symbols * 5 + 7) // 8


# ══════════════════════════════════════════════════════════════════════════════
# §4  PACKED SEQUENCE  —  immutable, memory-efficient sequence container
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(eq=False)
class PackedSequence:
    """
    Immutable container for a single 5-bit packed biological sequence.

    Biological data lives **only** inside ``data`` — a read-only uint8
    numpy array write-locked after construction.  The ``header`` field
    stores FASTA metadata only and is never derived from sequence data.

    Attributes
    ──────────
    header    : FASTA description line (``>`` prefix stripped).
    seq_type  : NUCLEOTIDE or PROTEIN.
    n_symbols : Original sequence length in biological symbols.
    data      : Read-only uint8 numpy array of 5-bit packed codes.
                ``len(data)`` == ⌈``n_symbols`` × 5 / 8⌉  bytes.
    """

    header:    str
    seq_type:  SeqType
    n_symbols: int
    data:      np.ndarray   # uint8, write-locked, 5-bit packed

    # ── Construction & validation ──────────────────────────────────────────────

    def __post_init__(self) -> None:
        """Normalise *data* to a write-locked uint8 array and validate length."""
        arr = np.asarray(self.data, dtype=np.uint8)
        arr.flags.writeable = False          # seal the buffer against mutations
        self.data = arr

        min_bytes = BitPacker.packed_size(self.n_symbols)
        if len(arr) < min_bytes:
            raise ValueError(
                f"data has {len(arr)} byte(s); "
                f"need ≥ {min_bytes} to hold {self.n_symbols} symbols."
            )

    # ── Equality & hashing ────────────────────────────────────────────────────

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PackedSequence):
            return NotImplemented
        return (
            self.seq_type  == other.seq_type
            and self.n_symbols == other.n_symbols
            and self.header    == other.header
            and np.array_equal(self.data, other.data)
        )

    def __hash__(self) -> int:
        # data.tobytes() is O(n) but guarantees content-correct hashing.
        return hash((self.seq_type, self.n_symbols,
                     self.header, self.data.tobytes()))

    # ── Representation ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        tag = "NUC" if self.seq_type == SeqType.NUCLEOTIDE else "PRO"
        return (
            f"PackedSequence(type={tag}, n={self.n_symbols:,}, "
            f"packed={self.packed_bytes:,} B, "
            f"saved={100 * (1 - self.memory_ratio):.1f}%, "
            f"header={self.header[:40]!r})"
        )

    # ── Sequence protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Number of biological symbols in the sequence."""
        return self.n_symbols

    def __getitem__(self, key: int | slice) -> int | PackedSequence:
        """
        Access one symbol (→ ``int`` BioCode) or a sub-sequence
        (→ new ``PackedSequence``).

        Single-index access is **O(1)** and reads only the 1–2 bytes
        containing the target 5-bit window without unpacking the whole array.
        Slice access decodes the required range and repacks.
        """
        if isinstance(key, int):
            idx = key + self.n_symbols if key < 0 else key
            if not 0 <= idx < self.n_symbols:
                raise IndexError(
                    f"index {key} out of range for sequence of length {self.n_symbols}"
                )
            return self._code_at(idx)

        if isinstance(key, slice):
            sub_codes = self.decode()[key]
            return PackedSequence(
                header    = self.header,
                seq_type  = self.seq_type,
                n_symbols = len(sub_codes),
                data      = BitPacker.pack(sub_codes),
            )

        raise TypeError(f"index must be int or slice, not {type(key).__name__}")

    def _code_at(self, idx: int) -> int:
        """Return the 5-bit BioCode at position *idx* sin desempaquetar todo."""
        if _C_AVAILABLE:
            return _c_getitem5(self.data, idx)

        bit_pos = idx * 5
        byte_i  = bit_pos >> 3
        bit_off = bit_pos & 7

        buf = np.zeros(2, dtype=np.uint8)
        buf[0] = self.data[byte_i]
        if byte_i + 1 < len(self.data):
            buf[1] = self.data[byte_i + 1]

        bits = np.unpackbits(buf)[bit_off: bit_off + 5]
        return int((bits * BitPacker._WEIGHTS).sum())

    # ── Storage properties ────────────────────────────────────────────────────

    @property
    def packed_bytes(self) -> int:
        """Byte count consumed by the ``data`` array."""
        return int(self.data.nbytes)

    @property
    def memory_ratio(self) -> float:
        """
        Bytes used per symbol relative to naive 8-bit (ASCII) storage.

        Ideal value for 5-bit packing: **0.625** (= 5 ÷ 8 = 37.5 % reduction).
        Marginally above 0.625 only for sequences shorter than 8 symbols
        due to byte-alignment padding.
        """
        return self.packed_bytes / self.n_symbols if self.n_symbols else 1.0

    # ── Decode / output ───────────────────────────────────────────────────────

    def decode(self) -> np.ndarray:
        """
        Unpack to a uint8 array of :class:`BioCode` values.

        Returns
        -------
        np.ndarray, dtype uint8, shape (n_symbols,), values in [0, 31]
        """
        return BitPacker.unpack(self.data, self.n_symbols)

    def to_string(self) -> str:
        """
        Decode to a human-readable single-letter string.

        **For output / FASTA export only.**
        Do *not* store the result as biological data — storing it defeats
        the library's memory-efficiency design.

        Returns
        -------
        str — canonical IUPAC single-letter sequence.
        """
        lut = (
            _NUC_DECODE_ARR
            if self.seq_type == SeqType.NUCLEOTIDE
            else _AA_DECODE_ARR
        )
        return lut[self.decode()].tobytes().decode("ascii")


# ══════════════════════════════════════════════════════════════════════════════
# §5  SMART IMPORTER  —  FASTA parser and 5-bit encoder
# ══════════════════════════════════════════════════════════════════════════════

class SmartImporter:
    """
    Parse FASTA text and immediately encode sequences into 5-bit packed
    :class:`PackedSequence` objects.

    The raw sequence ``str`` exists **only** inside :meth:`_encode` as a
    local variable released on return.  No biological sequence data
    persists as a Python ``str`` after that call.

    Sequence-type auto-detection
    ────────────────────────────
    The importer scans raw ASCII byte values for characters that are
    exclusive to protein sequences and absent from the IUPAC nucleotide
    alphabet  ``{A C G T U R Y S W K M B D H V N - .}``::

        E  F  I  L  P  Q  *   (and their lowercase equivalents)

    Any match → ``PROTEIN``; no match → ``NUCLEOTIDE``.
    Override per-call with the ``force_type`` parameter.

    Quick start
    ───────────
    >>> records = SmartImporter.from_string(fasta_text)
    >>> records = SmartImporter.from_file("genome.fa")
    >>> for rec in SmartImporter.from_file_chunked("large.fa"):
    ...     process(rec)
    """

    @classmethod
    def from_string(
        cls,
        fasta:      str,
        force_type: Optional[SeqType] = None,
    ) -> list[PackedSequence]:
        """
        Parse a complete FASTA string.

        Parameters
        ----------
        fasta : str
            FASTA-formatted text (one or more ``>``-delimited records).
        force_type : SeqType, optional
            Skip auto-detection and force this type for all records.

        Returns
        -------
        list[PackedSequence]
            One object per FASTA record, in input order.
        """
        return list(cls._iter_records(fasta, force_type))

    @classmethod
    def from_file(
        cls,
        path:       str,
        force_type: Optional[SeqType] = None,
    ) -> list[PackedSequence]:
        """
        Load and parse an entire FASTA file into memory.

        Parameters
        ----------
        path : str
            Path to the FASTA file (ASCII or UTF-8 with ASCII sequences).
        force_type : SeqType, optional
            Override auto-detection for all records.

        Returns
        -------
        list[PackedSequence]
        """
        with open(path, "r", encoding="ascii", errors="replace") as fh:
            return cls.from_string(fh.read(), force_type)

    @classmethod
    def from_file_chunked(
        cls,
        path:       str,
        force_type: Optional[SeqType] = None,
    ) -> Iterator[PackedSequence]:
        """
        Lazy generator for genome-scale files that do not fit in RAM.

        Only one record's raw lines are held in memory at a time.
        Each record is encoded and yielded before the next is read.

        Parameters
        ----------
        path : str
            Path to the FASTA file.
        force_type : SeqType, optional
            Override auto-detection.

        Yields
        ------
        PackedSequence — one per FASTA record, in file order.
        """
        header: Optional[str] = None
        chunks: list[str]     = []

        with open(path, "r", encoding="ascii", errors="replace") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if line.startswith(">"):
                    if header is not None and chunks:
                        yield cls._encode("".join(chunks), header, force_type)
                    header = line[1:]
                    chunks = []
                elif line and not line.startswith(";"):   # skip FASTA comments
                    chunks.append(line)

        if header is not None and chunks:
            yield cls._encode("".join(chunks), header, force_type)

    # ── Private helpers ───────────────────────────────────────────────────────

    @classmethod
    def _iter_records(
        cls,
        fasta:      str,
        force_type: Optional[SeqType],
    ) -> Iterator[PackedSequence]:
        """Yield one ``PackedSequence`` per record from a FASTA string."""
        header: Optional[str] = None
        chunks: list[str]     = []

        for line in fasta.splitlines():
            if line.startswith(">"):
                if header is not None and chunks:
                    yield cls._encode("".join(chunks), header, force_type)
                header = line[1:]
                chunks = []
            elif line and not line.startswith(";"):
                chunks.append(line.strip())

        if header is not None and chunks:
            yield cls._encode("".join(chunks), header, force_type)

    @staticmethod
    def _detect_type(ascii_bytes: np.ndarray) -> SeqType:
        """
        Classify nucleotide vs protein from raw ASCII byte values.

        ``_IS_PROTEIN_CHAR[ascii_bytes].any()`` performs the entire
        classification in one vectorised numpy call, with short-circuit
        semantics on the first protein-exclusive byte found.

        Parameters
        ----------
        ascii_bytes : np.ndarray, dtype uint8
            ASCII ordinals of the (uppercased) raw sequence.

        Returns
        -------
        SeqType
        """
        return (
            SeqType.PROTEIN
            if _IS_PROTEIN_CHAR[ascii_bytes].any()
            else SeqType.NUCLEOTIDE
        )

    @staticmethod
    def _encode(
        raw_seq:    str,
        header:     str,
        force_type: Optional[SeqType] = None,
    ) -> PackedSequence:
        """
        Core encoding pipeline — **the only function that ever holds a
        biological sequence as a ``str``.**  Four vectorised steps:

        ① ``str``  →  raw uint8 ASCII array        (``np.frombuffer``, near-zero copy)
        ② ASCII    →  5-bit BioCode array           (single LUT fancy-index)
        ③ BioCode  →  packed uint8 byte array       (``np.packbits``-based)
        ④ Wrap in :class:`PackedSequence` and return.

        *raw_seq* and all intermediate arrays are local and released on return.

        Parameters
        ----------
        raw_seq    : Raw sequence text (mixed case / gaps accepted).
        header     : FASTA description line without the ``>`` prefix.
        force_type : Override auto-detection.

        Returns
        -------
        PackedSequence
        """
        # ① String → ASCII ordinal array (frombuffer avoids a full copy).
        #    NUC_LUT and AA_LUT already map both upper and lower case, so
        #    .upper() would waste one full string copy — skip it.
        ascii_bytes: np.ndarray = np.frombuffer(
            raw_seq.encode("ascii", errors="replace"),
            dtype=np.uint8,
        )

        # ② Detect or apply sequence type
        seq_type: SeqType = (
            force_type
            if force_type is not None
            else SmartImporter._detect_type(ascii_bytes)
        )

        # ③ Translate ASCII ordinals → 5-bit BioCode (single LUT index op)
        lut:   np.ndarray = NUC_LUT if seq_type == SeqType.NUCLEOTIDE else AA_LUT
        codes: np.ndarray = lut[ascii_bytes]          # shape (n,), dtype uint8

        # ④ Compress 5-bit codes → packed byte array
        packed: np.ndarray = BitPacker.pack(codes)

        return PackedSequence(                         # raw_seq released here ✓
            header    = header,
            seq_type  = seq_type,
            n_symbols = int(len(codes)),
            data      = packed,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §6  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SequenceStats:
    """Composition and storage statistics for a :class:`PackedSequence`."""
    n_symbols:       int
    n_packed_bytes:  int
    compression_pct: float           # percent saved vs naive 8-bit ASCII
    composition:     dict[str, int]  # canonical IUPAC letter → symbol count


def compute_stats(seq: PackedSequence) -> SequenceStats:
    """
    Compute composition and storage statistics for a ``PackedSequence``.

    All counting is done via ``np.bincount`` — fully vectorised.

    Parameters
    ----------
    seq : PackedSequence

    Returns
    -------
    SequenceStats
    """
    codes      = seq.decode()                          # uint8, shape (n,)
    dec_map    = _NUC_DECODE if seq.seq_type == SeqType.NUCLEOTIDE else _AA_DECODE
    raw_counts = np.bincount(codes, minlength=32)      # counts per BioCode

    composition: dict[str, int] = {
        dec_map[code]: int(cnt)
        for code, cnt in enumerate(raw_counts)
        if cnt > 0 and code in dec_map
    }

    return SequenceStats(
        n_symbols       = seq.n_symbols,
        n_packed_bytes  = seq.packed_bytes,
        compression_pct = (1.0 - seq.memory_ratio) * 100.0,
        composition     = composition,
    )


# ══════════════════════════════════════════════════════════════════════════════
# §7  DEMO / SELF-TEST   (run with:  python biocore.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time

    _DEMO_FASTA = """\
>NC_000913.3 E. coli K-12 MG1655 — genomic fragment (DNA)
ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGA
CGCGTACAGGAAACAGCCAGCGATAAGTCCTGAATCAGCAAAAGCTTTTGCCCATCAGTTCAGTCA
>sp|P68871|HBB_HUMAN Hemoglobin subunit beta OS=Homo sapiens
MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGK
KVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQ
KVVAGVANALAHKYH*
>sp|P69905|HBA_HUMAN Hemoglobin subunit alpha OS=Homo sapiens
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADAT
LNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVS
TVLTSKYR*
"""

    W = 65
    print("═" * W)
    print("  biocore.py — Unified 5-bit bioinformatics library demo")
    print("═" * W)

    records = SmartImporter.from_string(_DEMO_FASTA)

    for rec in records:
        stats = compute_stats(rec)
        tag   = rec.seq_type.name
        print(f"\n  ── {tag} {'─' * (W - 6 - len(tag))}")
        print(f"  Header       : {rec.header[:55]}")
        print(f"  Symbols      : {stats.n_symbols:>10,}")
        print(f"  Packed       : {stats.n_packed_bytes:>10,} B")
        print(f"  Naive (ASCII): {stats.n_symbols:>10,} B")
        print(f"  Saved        : {stats.compression_pct:>9.1f} %")
        print(f"  Preview      : {rec.to_string()[:45]!r}")
        comp = "  ".join(f"{k}:{v}" for k, v in sorted(stats.composition.items()))
        print(f"  Composition  : {comp}")

        # ── Round-trip integrity check ─────────────────────────────────────
        codes    = rec.decode()
        repacked = BitPacker.pack(codes)
        assert np.array_equal(repacked, rec.data), "❌  Round-trip parity failure!"
        print(f"  Round-trip   : ✅  ({rec.n_symbols} sym → {rec.packed_bytes} B → OK)")

        # ── Per-symbol O(1) access check ───────────────────────────────────
        n_check = min(10, rec.n_symbols)
        for i in range(n_check):
            assert rec[i] == int(codes[i]), f"❌  rec[{i}] mismatch"
        print(f"  __getitem__  : ✅  (first {n_check} symbols checked, each O(1))")

    # ── Micro-benchmark ────────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print("  Micro-benchmark — 10 million random nucleotides")
    print(f"{'═' * W}")

    N    = 10_000_000
    rng  = np.random.default_rng(seed=42)
    bench_codes = rng.integers(0, 4, size=N, dtype=np.uint8)

    t0 = time.perf_counter()
    bench_packed = BitPacker.pack(bench_codes)
    t1 = time.perf_counter()
    bench_unpacked = BitPacker.unpack(bench_packed, N)
    t2 = time.perf_counter()

    assert np.array_equal(bench_unpacked, bench_codes), "❌  Benchmark round-trip failed"

    pack_ms   = (t1 - t0) * 1e3
    unpack_ms = (t2 - t1) * 1e3

    print(f"  Symbols      : {N:>12,}")
    print(f"  Pack time    : {pack_ms:>10.2f} ms   ({N / pack_ms / 1e3:>7.0f} M sym/s)")
    print(f"  Unpack time  : {unpack_ms:>10.2f} ms   ({N / unpack_ms / 1e3:>7.0f} M sym/s)")
    print(f"  Packed size  : {len(bench_packed):>12,} B  (naive: {N:,} B)")
    print(f"  Memory ratio : {len(bench_packed) / N:.4f}   (ideal 5-bit: 0.6250)")
    print()
