"""
biocore.py
══════════════════════════════════════════════════════════════════════
BioForge — high-performance bioinformatics engine built on a unified
5-bit biological alphabet.

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

import ctypes
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator, Optional

import numpy as np

try:
    from .engine._loader import C_AVAILABLE        as _C_AVAILABLE
    from .engine._loader import C_PARSER_AVAILABLE as _C_PARSER_AVAILABLE
    from .engine._loader import C_BATCH_AVAILABLE  as _C_BATCH_AVAILABLE
    from .engine._loader import c_getitem5         as _c_getitem5
    from .engine._loader import c_pack5            as _c_pack5
    from .engine._loader import c_unpack5          as _c_unpack5
    from .engine._loader import c_parser_open      as _c_parser_open
    from .engine._loader import c_parser_next      as _c_parser_next
    from .engine._loader import c_parser_next_fastq as _c_parser_next_fastq
    from .engine._loader import c_parser_next_batch as _c_parser_next_batch
    from .engine._loader import c_parser_close     as _c_parser_close
except ImportError:
    _C_AVAILABLE        = False
    _C_PARSER_AVAILABLE = False
    _C_BATCH_AVAILABLE  = False


__all__: list[str] = [
    # Excepciones — importar para capturar errores del motor
    "BioForgeError",
    "SequenceTypeError",
    "SequenceValueError",
    "TranslationError",
    "AlignmentError",
    # Núcleo
    "BioCode",
    "SeqType",
    "NUC_LUT",
    "AA_LUT",
    "BitPacker",
    "PackedSequence",
    "FastqRecord",
    "SmartImporter",
    "SequenceStats",
    "compute_stats",
]


# ══════════════════════════════════════════════════════════════════════════════
# §0  EXCEPTIONS  —  jerarquía de errores de BioForge
# ══════════════════════════════════════════════════════════════════════════════

class BioForgeError(Exception):
    """Base para todos los errores propios de BioForge.

    Úsala en bloques ``except`` para capturar cualquier error del motor
    sin interferir con el resto de Python::

        from bioforge import BioForgeError
        try:
            prot = SmartTranslator.translate(seq)
        except BioForgeError as e:
            print(f"Error de BioForge: {e}")

    Las subclases también heredan de ``TypeError`` o ``ValueError`` según
    corresponda, por lo que el código existente que ya atrapa esos tipos
    estándar sigue funcionando sin cambios.
    """


class SequenceTypeError(BioForgeError, TypeError):
    """Tipo incorrecto al llamar a una función del motor.

    Se lanza cuando:

    - Se pasa un ``str``, ``list`` u otro objeto donde se esperaba
      ``PackedSequence``.
    - Se mezclan tipos biológicos incompatibles (NUCLEOTIDE con PROTEIN).
    - El ``seq_type`` de un ``PackedSequence`` no es un valor ``SeqType``.
    """


class SequenceValueError(BioForgeError, ValueError):
    """Valor inválido en una secuencia o en sus metadatos.

    Se lanza cuando:

    - ``n_symbols`` es negativo.
    - El buffer ``packed`` es demasiado pequeño para ``n`` símbolos.
    - La secuencia está vacía donde se requiere contenido.
    - ``codes`` no es un array 1-D.
    """


class TranslationError(BioForgeError, ValueError):
    """Error durante la traducción ADN→Proteína.

    Se lanza cuando:

    - La secuencia no contiene ningún codón ATG/AUG.
    - El ORF no tiene ningún codón completo tras el ATG.
    - La secuencia es demasiado corta para contener un codón.
    """


class AlignmentError(BioForgeError, ValueError):
    """Error durante el alineamiento o en sus parámetros.

    Se lanza cuando:

    - El modo no es ``'global'`` ni ``'semi-global'``.
    - ``width`` es ≤ 0 en ``format_alignment``.
    - Las cadenas alineadas tienen longitudes incongruentes.
    """


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

# ── Watson-Crick complement LUT  (BioCode index → complement BioCode) ───────────
# A(0)↔T/U(3), C(1)↔G(2).  Gaps and UNK map to themselves.  All other codes
# (amino acids, reserved) map to UNK — reverse complement of protein is undefined.
_NUC_COMPLEMENT: np.ndarray = np.full(32, BioCode.UNK, dtype=np.uint8)
_NUC_COMPLEMENT[int(BioCode.NUC_A )] = np.uint8(BioCode.NUC_TU)
_NUC_COMPLEMENT[int(BioCode.NUC_C )] = np.uint8(BioCode.NUC_G)
_NUC_COMPLEMENT[int(BioCode.NUC_G )] = np.uint8(BioCode.NUC_C)
_NUC_COMPLEMENT[int(BioCode.NUC_TU)] = np.uint8(BioCode.NUC_A)
_NUC_COMPLEMENT[int(BioCode.UNK   )] = np.uint8(BioCode.UNK)
_NUC_COMPLEMENT[int(BioCode.GAP   )] = np.uint8(BioCode.GAP)


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
        codes = np.asarray(codes, dtype=np.uint8)
        if codes.ndim != 1:
            raise SequenceValueError(
                f"codes debe ser un array 1-D, se recibió shape {codes.shape}. "
                "Pasa un array plano, p.ej. np.array([0, 1, 2], dtype=np.uint8)."
            )
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
        if not isinstance(n, (int, np.integer)) or int(n) < 0:
            raise SequenceValueError(
                f"n debe ser un entero no negativo, se recibió {n!r}."
            )
        n = int(n)
        min_bytes = BitPacker.packed_size(n)
        if len(packed) < min_bytes:
            raise SequenceValueError(
                f"packed tiene {len(packed)} byte(s) pero se necesitan "
                f"al menos {min_bytes} para desempaquetar {n} símbolo(s). "
                "¿El buffer procede de BitPacker.pack() con los mismos datos?"
            )
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
        if not isinstance(self.n_symbols, (int, np.integer)) or int(self.n_symbols) < 0:
            raise SequenceValueError(
                f"n_symbols debe ser un entero no negativo, se recibió {self.n_symbols!r}."
            )
        if not isinstance(self.seq_type, SeqType):
            raise SequenceTypeError(
                f"seq_type debe ser SeqType.NUCLEOTIDE o SeqType.PROTEIN, "
                f"se recibió {type(self.seq_type).__name__!r}."
            )
        arr = np.asarray(self.data, dtype=np.uint8)
        arr.flags.writeable = False          # seal the buffer against mutations
        self.data = arr

        min_bytes = BitPacker.packed_size(self.n_symbols)
        if len(arr) < min_bytes:
            raise SequenceValueError(
                f"data tiene {len(arr)} byte(s) pero se necesitan "
                f"≥ {min_bytes} para almacenar {self.n_symbols} símbolo(s). "
                "Usa BitPacker.pack(codes) para generar el buffer correcto."
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

    def reverse_complement(self) -> PackedSequence:
        """
        Compute the reverse complement (5'→3' antisense strand).

        Watson-Crick pairing: A↔T/U, C↔G.  Unknown bases (N) and gaps (-)
        map to themselves.  Two fully vectorised NumPy operations: flip + LUT.

        Returns
        -------
        PackedSequence (NUCLEOTIDE) with header prefixed ``[RC]``.

        Raises
        ------
        SequenceTypeError
            If the sequence is not ``SeqType.NUCLEOTIDE``.
        """
        if self.seq_type != SeqType.NUCLEOTIDE:
            raise SequenceTypeError(
                f"reverse_complement() requiere SeqType.NUCLEOTIDE, "
                f"se recibió {self.seq_type.name}. "
                "Las proteínas no tienen complemento de Watson-Crick."
            )
        rc = _NUC_COMPLEMENT[self.decode()[::-1]]   # flip + complement, two vectorised ops
        return PackedSequence(
            header    = f"[RC] {self.header}",
            seq_type  = SeqType.NUCLEOTIDE,
            n_symbols = self.n_symbols,
            data      = BitPacker.pack(rc),
        )


# ══════════════════════════════════════════════════════════════════════════════
# §5  FASTQ RECORD  —  secuencia 5-bit + calidades Phred
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FastqRecord:
    """
    Un registro FASTQ: secuencia nucleotídica empaquetada + calidades Phred.

    Attributes
    ──────────
    sequence : PackedSequence (SeqType.NUCLEOTIDE, 5-bit packed)
    quality  : np.ndarray uint8, valores Phred 0–93 (ASCII-33 ya restado).
               Longitud idéntica a ``sequence.n_symbols``.

    Quick start
    ───────────
    >>> for rec in SmartImporter.stream_fastq("reads.fastq"):
    ...     if rec.passes_quality(20):
    ...         process(rec.sequence)
    """

    sequence: PackedSequence
    quality:  np.ndarray   # uint8, Phred 0–93

    @property
    def mean_quality(self) -> float:
        """Calidad Phred media de la lectura."""
        return float(self.quality.mean()) if len(self.quality) > 0 else 0.0

    def passes_quality(self, min_q: int) -> bool:
        """True si la calidad Phred media es ≥ min_q."""
        return self.mean_quality >= min_q

    def __repr__(self) -> str:
        return (
            f"FastqRecord(n={self.sequence.n_symbols:,}, "
            f"q_mean={self.mean_quality:.1f}, "
            f"header={self.sequence.header[:40]!r})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# §5b  COLUMNAR BATCHES  —  miles de registros como matrices, sin objeto/registro
# ══════════════════════════════════════════════════════════════════════════════
#
# La ruta rápida de v2.0.  En vez de fabricar un objeto Python por registro
# (≈5 µs cada uno = ~2 s para 200 000 lecturas), se conservan las matrices
# contiguas que el motor C ya produce y los análisis se hacen como operaciones
# NumPy sobre columnas enteras.  Los objetos individuales (PackedSequence /
# FastqRecord) se materializan **solo** cuando se piden con indexación.
#
# Caso ideal (Illumina): todas las lecturas miden lo mismo → las calidades son
# una matriz 2-D limpia (m × L) y filtrar es indexación booleana pura.


def _gather_headers(
    hdr_raw: bytes, hdr_off: np.ndarray, idx: np.ndarray
) -> "tuple[bytes, np.ndarray]":
    """Reconstruye el blob de cabeceras para los registros seleccionados."""
    parts: list[bytes] = []
    new_off = np.empty(len(idx) + 1, dtype=np.int32)
    new_off[0] = 0
    cur = 0
    for k, j in enumerate(idx):
        seg = hdr_raw[int(hdr_off[j]): int(hdr_off[j + 1])]   # incluye '\0'
        parts.append(seg)
        cur += len(seg)
        new_off[k + 1] = cur
    return b"".join(parts), new_off


# ── Núcleo vectorizado de GC y k-meros (compartido por ambos lotes) ──────────

_GC_W = np.array([16, 8, 4, 2, 1], dtype=np.uint8)   # pesos de bit MSB→LSB


def _decode_fixed_2d(packed: np.ndarray, m: int, L: int) -> np.ndarray:
    """Decodifica m registros de longitud fija L → matriz (m, L) de BioCode.

    Totalmente vectorizado: una sola ``unpackbits`` sobre toda la matriz de
    bytes empaquetados. Aprovecha que cada registro ocupa exactamente
    ``plen`` bytes cuando todas las longitudes coinciden.
    """
    plen = (L * 5 + 7) // 8
    packed2d = packed[: m * plen].reshape(m, plen)
    bits = np.unpackbits(packed2d, axis=1)[:, : L * 5].reshape(m, L, 5)
    return (bits * _GC_W).sum(axis=2, dtype=np.uint8)


def _batch_gc(packed, pack_off, n_syms) -> np.ndarray:
    """Fracción GC (0..1) de cada registro. Vectorizado si la longitud es fija."""
    m = int(n_syms.shape[0])
    if m == 0:
        return np.empty(0, dtype=np.float64)
    if bool(np.all(n_syms == n_syms[0])) and int(n_syms[0]) > 0:
        L = int(n_syms[0])
        codes = _decode_fixed_2d(packed, m, L)
        gc = ((codes == 1) | (codes == 2)).sum(axis=1)
        return gc / L
    # Longitud irregular: bucle por registro (cada uno se decodifica vectorizado).
    out = np.empty(m, dtype=np.float64)
    for i in range(m):
        n = int(n_syms[i])
        if n == 0:
            out[i] = 0.0
            continue
        c = BitPacker.unpack(packed[int(pack_off[i]): int(pack_off[i + 1])], n)
        out[i] = float(((c == 1) | (c == 2)).sum()) / n
    return out


def _batch_kmer_spectrum(packed, pack_off, n_syms, k: int) -> np.ndarray:
    """Espectro de k-meros del lote entero → array int64 de longitud 4**k.

    Cuenta todos los k-meros de todas las secuencias. Los k-meros con bases
    ambiguas (código > 3) se descartan. Vectorizado si la longitud es fija.
    """
    if k < 1:
        raise SequenceValueError(f"k debe ser >= 1, se recibió {k}.")
    n_kmers = 4 ** k
    out = np.zeros(n_kmers, dtype=np.int64)
    m = int(n_syms.shape[0])
    if m == 0:
        return out
    powers = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)
    sw = np.lib.stride_tricks.sliding_window_view

    if bool(np.all(n_syms == n_syms[0])) and int(n_syms[0]) >= k:
        L = int(n_syms[0])
        codes = _decode_fixed_2d(packed, m, L)              # (m, L)
        win = sw(codes, k, axis=1)                          # (m, L-k+1, k)
        valid = (win <= 3).all(axis=2)                      # (m, L-k+1)
        ids = (win.astype(np.int64) * powers).sum(axis=2)   # (m, L-k+1)
        return np.bincount(ids[valid].ravel(), minlength=n_kmers)[:n_kmers]

    for i in range(m):
        n = int(n_syms[i])
        if n < k:
            continue
        c = BitPacker.unpack(
            packed[int(pack_off[i]): int(pack_off[i + 1])], n).astype(np.int64)
        win = sw(c, k)                                      # (n-k+1, k)
        valid = (win <= 3).all(axis=1)
        ids = (win * powers).sum(axis=1)[valid]
        out += np.bincount(ids, minlength=n_kmers)[:n_kmers]
    return out


@dataclass
class SequenceBatch:
    """
    Lote columnar de secuencias FASTA.

    Guarda todas las secuencias de un lote como matrices contiguas.  El acceso
    a un registro concreto (``batch[i]``) materializa un :class:`PackedSequence`
    en ese momento; las operaciones de conjunto se hacen vectorizadas.

    No instancies esto a mano — lo produce :meth:`SmartImporter.stream_batches`.
    """

    _packed:   np.ndarray   # uint8, secuencias 5-bit concatenadas (byte-alineadas)
    _pack_off: np.ndarray   # int32, offsets de byte, len = m+1
    _n_syms:   np.ndarray   # int32, longitud de cada registro, len = m
    _types:    np.ndarray   # int32, tipo (0=nuc,1=prot), len = m
    _hdr_raw:  bytes        # blob de cabeceras null-terminadas
    _hdr_off:  np.ndarray   # int32, offsets de cabecera, len = m+1

    def __len__(self) -> int:
        return int(self._n_syms.shape[0])

    @property
    def n_symbols(self) -> np.ndarray:
        """Longitud de cada registro del lote (array int32)."""
        return self._n_syms

    def header(self, i: int) -> str:
        """Cabecera del registro ``i`` (decodificada bajo demanda)."""
        i = int(i)
        return self._hdr_raw[
            int(self._hdr_off[i]): int(self._hdr_off[i + 1]) - 1
        ].decode("ascii", errors="replace")

    def __getitem__(self, i: int) -> PackedSequence:
        n = len(self)
        i = int(i)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(f"índice {i} fuera de rango (lote de {n})")
        return PackedSequence(
            header    = self.header(i),
            seq_type  = SeqType(int(self._types[i])),
            n_symbols = int(self._n_syms[i]),
            data      = self._packed[
                int(self._pack_off[i]): int(self._pack_off[i + 1])
            ].copy(),
        )

    def __iter__(self) -> "Iterator[PackedSequence]":
        for i in range(len(self)):
            yield self[i]

    # ── Análisis vectorizado de composición (solo nucleótidos) ──────────────
    def _require_nucleotide(self, op: str) -> None:
        if len(self) and bool((self._types == 1).any()):
            raise SequenceTypeError(
                f"{op} solo aplica a secuencias nucleotídicas; el lote contiene "
                "proteínas. Filtra por tipo antes de llamarlo."
            )

    def gc_content(self) -> np.ndarray:
        """Fracción GC (0..1) de cada secuencia del lote (vectorizado)."""
        self._require_nucleotide("gc_content()")
        return _batch_gc(self._packed, self._pack_off, self._n_syms)

    def kmer_spectrum(self, k: int) -> np.ndarray:
        """Espectro de k-meros del lote → array int64 de longitud ``4**k``."""
        self._require_nucleotide("kmer_spectrum()")
        return _batch_kmer_spectrum(self._packed, self._pack_off, self._n_syms, k)

    def __repr__(self) -> str:
        return (f"SequenceBatch(m={len(self)}, "
                f"bases={int(self._n_syms.sum()):,})")


@dataclass
class ReadBatch:
    """
    Lote columnar de lecturas FASTQ — la vía rápida para control de calidad.

    Las calidades de todo el lote viven en una sola matriz.  Filtrar por calidad
    media es una operación NumPy sobre las ``m`` lecturas a la vez, sin fabricar
    un objeto por lectura.  ``batch[i]`` materializa un :class:`FastqRecord`
    solo cuando lo pides.

    Caso de longitud fija (Illumina): ``_fixed_len > 0`` y las calidades son una
    matriz 2-D ``(m, L)``.  Caso irregular (Nanopore): calidades concatenadas en
    1-D con ``_qual_off``.

    No instancies esto a mano — lo produce :meth:`SmartImporter.stream_fastq_batches`.
    """

    _packed:   np.ndarray
    _pack_off: np.ndarray
    _n_syms:   np.ndarray
    _types:    np.ndarray
    _hdr_raw:  bytes
    _hdr_off:  np.ndarray
    _qual:     np.ndarray            # 2-D (m,L) si fijo; 1-D concatenado si no
    _qual_off: "Optional[np.ndarray]"  # None si longitud fija
    _fixed_len: int                  # >0 = longitud fija; 0 = irregular

    def __len__(self) -> int:
        return int(self._n_syms.shape[0])

    @property
    def n_symbols(self) -> np.ndarray:
        return self._n_syms

    # ── Operaciones vectorizadas sobre TODO el lote ─────────────────────────
    def mean_quality(self) -> np.ndarray:
        """Calidad Phred media de cada lectura (array float, una op NumPy)."""
        if len(self) == 0:
            return np.empty(0, dtype=np.float64)
        if self._fixed_len:
            return self._qual.mean(axis=1)
        # Irregular: suma por segmentos con reduceat (vectorizado).
        starts = self._qual_off[:-1]
        sums   = np.add.reduceat(self._qual.astype(np.int64), starts)
        n      = self._n_syms.astype(np.int64)
        return np.where(n > 0, sums / np.maximum(n, 1), 0.0)

    def passes(self, min_q: float) -> np.ndarray:
        """Máscara booleana: True donde la calidad media ≥ ``min_q``."""
        return self.mean_quality() >= min_q

    def gc_content(self) -> np.ndarray:
        """Fracción GC (0..1) de cada lectura del lote (vectorizado)."""
        return _batch_gc(self._packed, self._pack_off, self._n_syms)

    def kmer_spectrum(self, k: int) -> np.ndarray:
        """Espectro de k-meros del lote → array int64 de longitud ``4**k``.

        Cuenta todos los k-meros de todas las lecturas (los que tienen bases
        ambiguas se descartan). Útil para perfiles de k-meros, corrección de
        errores o estimación de cobertura — sin crear objetos por lectura.
        """
        return _batch_kmer_spectrum(self._packed, self._pack_off,
                                    self._n_syms, k)

    def filter(self, mask: np.ndarray) -> "ReadBatch":
        """Devuelve un nuevo ReadBatch con solo las lecturas de ``mask``."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != len(self):
            raise SequenceValueError(
                f"la máscara tiene {mask.shape[0]} elementos pero el lote "
                f"tiene {len(self)} lecturas."
            )
        idx = np.flatnonzero(mask)
        new_n   = self._n_syms[idx].copy()
        new_t   = self._types[idx].copy()
        new_hdr, new_hoff = _gather_headers(self._hdr_raw, self._hdr_off, idx)

        if self._fixed_len:
            # Todo es 2-D regular → indexación pura, sin bucles.
            L    = self._fixed_len
            plen = (L * 5 + 7) // 8
            m    = len(self)
            packed2d  = self._packed[: m * plen].reshape(m, plen)
            new_packed = packed2d[idx].reshape(-1).copy()
            new_poff   = (np.arange(len(idx) + 1, dtype=np.int32) * plen)
            new_qual   = self._qual[idx].copy()
            return ReadBatch(new_packed, new_poff, new_n, new_t,
                             new_hdr, new_hoff, new_qual, None, L)

        # Irregular: reunir slices de los supervivientes (bucle por registro).
        pack_parts, qual_parts = [], []
        new_poff = np.empty(len(idx) + 1, dtype=np.int32)
        new_qoff = np.empty(len(idx) + 1, dtype=np.int32)
        new_poff[0] = new_qoff[0] = 0
        pcur = qcur = 0
        for k, j in enumerate(idx):
            ps = self._packed[int(self._pack_off[j]): int(self._pack_off[j + 1])]
            qs = self._qual[int(self._qual_off[j]): int(self._qual_off[j + 1])]
            pack_parts.append(ps); qual_parts.append(qs)
            pcur += ps.shape[0]; qcur += qs.shape[0]
            new_poff[k + 1] = pcur; new_qoff[k + 1] = qcur
        new_packed = (np.concatenate(pack_parts) if pack_parts
                      else np.empty(0, dtype=np.uint8))
        new_qual   = (np.concatenate(qual_parts) if qual_parts
                      else np.empty(0, dtype=np.uint8))
        return ReadBatch(new_packed, new_poff, new_n, new_t,
                         new_hdr, new_hoff, new_qual, new_qoff, 0)

    # ── Acceso por registro (materializa el objeto solo aquí) ───────────────
    def header(self, i: int) -> str:
        i = int(i)
        return self._hdr_raw[
            int(self._hdr_off[i]): int(self._hdr_off[i + 1]) - 1
        ].decode("ascii", errors="replace")

    def quality_of(self, i: int) -> np.ndarray:
        """Calidades Phred de la lectura ``i`` (copia uint8)."""
        i = int(i)
        if self._fixed_len:
            return self._qual[i].copy()
        return self._qual[
            int(self._qual_off[i]): int(self._qual_off[i + 1])
        ].copy()

    def __getitem__(self, i: int) -> FastqRecord:
        n = len(self)
        i = int(i)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(f"índice {i} fuera de rango (lote de {n})")
        seq = PackedSequence(
            header    = self.header(i),
            seq_type  = SeqType(int(self._types[i])),
            n_symbols = int(self._n_syms[i]),
            data      = self._packed[
                int(self._pack_off[i]): int(self._pack_off[i + 1])
            ].copy(),
        )
        return FastqRecord(sequence=seq, quality=self.quality_of(i))

    def __iter__(self) -> "Iterator[FastqRecord]":
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        kind = f"fixed L={self._fixed_len}" if self._fixed_len else "ragged"
        return (f"ReadBatch(m={len(self)}, "
                f"bases={int(self._n_syms.sum()):,}, {kind})")


# ══════════════════════════════════════════════════════════════════════════════
# §6  SMART IMPORTER  —  FASTA/FASTQ parser and 5-bit encoder
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

    # ── Streaming API (O(1) RAM, motor C con buffer 64 KB) ───────────────────

    _STREAM_HDR  = 4096          # bytes máximos para una cabecera FASTA/FASTQ
    _STREAM_SEQ  = 16 * 1024 * 1024   # 16 MB — cubre lecturas Nanopore largas

    # ── Modo por lotes (batch) — el camino rápido de v2.0 ────────────────────
    # Una sola llamada a C parsea miles de registros y los empaqueta a 5-bit
    # dentro de C. Elimina el peaje ctypes y el pack NumPy por registro.
    _BATCH_RECORDS = 8192               # registros por llamada
    _BATCH_HDR     = 2 * 1024 * 1024    # 2 MB para cabeceras concatenadas
    _BATCH_PACK    = 16 * 1024 * 1024   # 16 MB de secuencias 5-bit (≤25 Mbp/lote)

    @classmethod
    def stream(
        cls,
        path:       str,
        force_type: Optional[SeqType] = None,
    ) -> Iterator[PackedSequence]:
        """
        Generador de bajo consumo de RAM para archivos FASTA de cualquier tamaño.

        Cuando el motor C está disponible (compilado) usa el parser C con
        buffer de 64 KB y encoding 5-bit directo en C — sin strings Python
        en la ruta crítica.  Fallback automático a :meth:`from_file_chunked`
        si el motor C no está presente.

        Parameters
        ----------
        path : str
            Ruta al archivo FASTA.
        force_type : SeqType, optional
            Fuerza el tipo para todos los registros (omite auto-detección).

        Yields
        ------
        PackedSequence — un objeto por registro, en orden de fichero.

        Example
        -------
        >>> for seq in SmartImporter.stream("genome.fa"):
        ...     print(seq.n_symbols)
        """
        ft = -1
        if force_type == SeqType.NUCLEOTIDE: ft = 0
        elif force_type == SeqType.PROTEIN:  ft = 1

        if _C_BATCH_AVAILABLE:
            yield from cls._stream_batch(path, ft, fastq=False)
            return

        if not _C_PARSER_AVAILABLE:
            yield from cls.from_file_chunked(path, force_type)
            return

        hdr_buf   = ctypes.create_string_buffer(cls._STREAM_HDR)
        codes_buf = np.empty(cls._STREAM_SEQ, dtype=np.uint8)

        handle = _c_parser_open(path)
        if not handle:
            raise IOError(f"No se puede abrir el archivo: {path!r}")

        try:
            while True:
                ret, n, stype = _c_parser_next(handle, hdr_buf, codes_buf, ft)
                if ret <= 0:
                    break
                packed = BitPacker.pack(codes_buf[:n])
                header = hdr_buf.value.decode("ascii", errors="replace")
                yield PackedSequence(
                    header    = header,
                    seq_type  = SeqType(stype),
                    n_symbols = n,
                    data      = packed,
                )
        finally:
            _c_parser_close(handle)

    @classmethod
    def _stream_batch(cls, path: str, force_type: int, fastq: bool):
        """Núcleo del modo por lotes — compartido por stream() y stream_fastq().

        C parsea hasta ``_BATCH_RECORDS`` registros por llamada y empaqueta cada
        secuencia a 5-bit. Aquí solo cruzamos la frontera ~N/8192 veces y
        creamos los objetos Python a partir de buffers ya empaquetados.

        Yields PackedSequence (FASTA) o FastqRecord (FASTQ).
        """
        BR = cls._BATCH_RECORDS
        hdr_buf  = ctypes.create_string_buffer(cls._BATCH_HDR)
        pack_buf = np.empty(cls._BATCH_PACK, dtype=np.uint8)
        hdr_off  = np.empty(BR + 1, dtype=np.int32)
        pack_off = np.empty(BR + 1, dtype=np.int32)
        n_syms   = np.empty(BR, dtype=np.int32)
        types    = np.empty(BR, dtype=np.int32)
        if fastq:
            qual_buf = np.empty(cls._BATCH_PACK, dtype=np.uint8)
            qual_off = np.empty(BR + 1, dtype=np.int32)
        else:
            qual_buf = qual_off = None

        handle = _c_parser_open(path)
        if not handle:
            raise IOError(f"No se puede abrir el archivo: {path!r}")

        ps = PackedSequence  # alias local — menos lookups en el bucle
        try:
            while True:
                m = _c_parser_next_batch(
                    handle, BR, force_type,
                    hdr_buf, hdr_off, pack_buf, pack_off,
                    n_syms, types, qual_buf, qual_off,
                )
                if m == 0:
                    break
                if m < 0:
                    raise IOError(
                        f"Error del parser por lotes (código {m}) en {path!r}. "
                        "Código -2 = un registro supera el buffer de 16 MB; "
                        "usa una herramienta de lecturas ultra-largas."
                    )
                # Snapshot de los buffers como bytes/listas: una copia por lote,
                # no por registro.
                hraw = hdr_buf.raw
                hoff = hdr_off[: m + 1].tolist()
                poff = pack_off[: m + 1].tolist()
                nlst = n_syms[:m].tolist()
                tlst = types[:m].tolist()
                qoff = qual_off[: m + 1].tolist() if fastq else None

                for i in range(m):
                    header = hraw[hoff[i]: hoff[i + 1] - 1].decode(
                        "ascii", errors="replace")
                    seq = ps(
                        header    = header,
                        seq_type  = SeqType(tlst[i]),
                        n_symbols = nlst[i],
                        data      = pack_buf[poff[i]: poff[i + 1]].copy(),
                    )
                    if fastq:
                        yield FastqRecord(
                            sequence = seq,
                            quality  = qual_buf[qoff[i]: qoff[i + 1]].copy(),
                        )
                    else:
                        yield seq
        finally:
            _c_parser_close(handle)

    # ── Núcleo columnar (la vía rápida de v2.0) ──────────────────────────────
    @classmethod
    def _stream_columnar(cls, path: str, force_type: int, fastq: bool):
        """Entrega lotes como matrices contiguas (SequenceBatch / ReadBatch).

        Cero objetos por registro: se conservan las matrices que C ya produce.
        Una copia por lote (no por registro) las desacopla de los buffers
        reutilizados. Yields SequenceBatch (FASTA) o ReadBatch (FASTQ).
        """
        if not _C_BATCH_AVAILABLE:
            yield from cls._columnar_fallback(path, force_type, fastq)
            return

        BR = cls._BATCH_RECORDS
        hdr_buf  = ctypes.create_string_buffer(cls._BATCH_HDR)
        pack_buf = np.empty(cls._BATCH_PACK, dtype=np.uint8)
        hdr_off  = np.empty(BR + 1, dtype=np.int32)
        pack_off = np.empty(BR + 1, dtype=np.int32)
        n_syms   = np.empty(BR, dtype=np.int32)
        types    = np.empty(BR, dtype=np.int32)
        if fastq:
            qual_buf = np.empty(cls._BATCH_PACK, dtype=np.uint8)
            qual_off = np.empty(BR + 1, dtype=np.int32)
        else:
            qual_buf = qual_off = None

        handle = _c_parser_open(path)
        if not handle:
            raise IOError(f"No se puede abrir el archivo: {path!r}")

        try:
            while True:
                m = _c_parser_next_batch(
                    handle, BR, force_type,
                    hdr_buf, hdr_off, pack_buf, pack_off,
                    n_syms, types, qual_buf, qual_off,
                )
                if m == 0:
                    break
                if m < 0:
                    raise IOError(
                        f"Error del parser por lotes (código {m}) en {path!r}. "
                        "Código -2 = un registro supera el buffer de 16 MB."
                    )
                # Una copia por lote para desacoplar de los buffers reutilizados.
                pack_used = int(pack_off[m])
                hdr_used  = int(hdr_off[m])
                packed = pack_buf[:pack_used].copy()
                poff   = pack_off[: m + 1].copy()
                nsy    = n_syms[:m].copy()
                tps    = types[:m].copy()
                hraw   = bytes(hdr_buf.raw[:hdr_used])
                hoff   = hdr_off[: m + 1].copy()

                if fastq:
                    qual_used = int(qual_off[m])
                    fixed = (int(nsy[0]) if (m > 0 and nsy[0] > 0
                             and bool(np.all(nsy == nsy[0]))) else 0)
                    if fixed:
                        qual = qual_buf[:qual_used].copy().reshape(m, fixed)
                        qoff = None
                    else:
                        qual = qual_buf[:qual_used].copy()
                        qoff = qual_off[: m + 1].copy()
                    yield ReadBatch(packed, poff, nsy, tps,
                                    hraw, hoff, qual, qoff, fixed)
                else:
                    yield SequenceBatch(packed, poff, nsy, tps, hraw, hoff)
        finally:
            _c_parser_close(handle)

    @classmethod
    def _columnar_fallback(cls, path: str, force_type: int, fastq: bool):
        """Construye lotes columnares desde el generador por registro.

        Solo se usa si el motor C por lotes no está disponible (DLL antiguo o
        sin compilar). Más lento, pero produce idénticos SequenceBatch/ReadBatch.
        """
        BR = cls._BATCH_RECORDS
        ft = (SeqType.NUCLEOTIDE if force_type == 0 else
              SeqType.PROTEIN if force_type == 1 else None)
        gen = (cls.stream_fastq(path) if fastq else cls.stream(path, ft))
        buf: list = []
        for item in gen:
            buf.append(item)
            if len(buf) >= BR:
                yield cls._assemble_batch(buf, fastq)
                buf = []
        if buf:
            yield cls._assemble_batch(buf, fastq)

    @staticmethod
    def _assemble_batch(items: list, fastq: bool):
        """Ensambla una lista de PackedSequence/FastqRecord en un lote columnar."""
        m = len(items)
        seqs = [(it.sequence if fastq else it) for it in items]
        nsy  = np.array([s.n_symbols for s in seqs], dtype=np.int32)
        tps  = np.array([int(s.seq_type) for s in seqs], dtype=np.int32)

        pack_parts = [np.asarray(s.data, dtype=np.uint8) for s in seqs]
        poff = np.empty(m + 1, dtype=np.int32)
        poff[0] = 0
        acc = 0
        for k, p in enumerate(pack_parts):
            acc += p.shape[0]; poff[k + 1] = acc
        packed = (np.concatenate(pack_parts) if pack_parts
                  else np.empty(0, dtype=np.uint8))

        hdr_parts = [s.header.encode("ascii", "replace") + b"\0" for s in seqs]
        hoff = np.empty(m + 1, dtype=np.int32)
        hoff[0] = 0
        acc = 0
        for k, h in enumerate(hdr_parts):
            acc += len(h); hoff[k + 1] = acc
        hraw = b"".join(hdr_parts)

        if not fastq:
            return SequenceBatch(packed, poff, nsy, tps, hraw, hoff)

        quals = [np.asarray(it.quality, dtype=np.uint8) for it in items]
        fixed = (int(nsy[0]) if (m > 0 and nsy[0] > 0
                 and bool(np.all(nsy == nsy[0]))) else 0)
        if fixed:
            qual = (np.stack(quals) if quals
                    else np.empty((0, fixed), dtype=np.uint8))
            qoff = None
        else:
            qual = (np.concatenate(quals) if quals
                    else np.empty(0, dtype=np.uint8))
            qoff = np.empty(m + 1, dtype=np.int32)
            qoff[0] = 0
            acc = 0
            for k, q in enumerate(quals):
                acc += q.shape[0]; qoff[k + 1] = acc
        return ReadBatch(packed, poff, nsy, tps, hraw, hoff, qual, qoff, fixed)

    @classmethod
    def stream_batches(
        cls,
        path:       str,
        force_type: Optional[SeqType] = None,
    ) -> "Iterator[SequenceBatch]":
        """
        Lee un FASTA como lotes columnares :class:`SequenceBatch` (vía rápida).

        Cada lote agrupa hasta ``_BATCH_RECORDS`` registros como matrices
        contiguas, sin crear un objeto por registro. Ideal para barrer, contar
        o filtrar grandes colecciones. Para acceder a un registro concreto usa
        ``batch[i]``.

        Example
        -------
        >>> for batch in SmartImporter.stream_batches("genome.fa"):
        ...     print(len(batch), "secuencias,", int(batch.n_symbols.sum()), "bases")
        """
        ft = -1
        if force_type == SeqType.NUCLEOTIDE: ft = 0
        elif force_type == SeqType.PROTEIN:  ft = 1
        yield from cls._stream_columnar(path, ft, fastq=False)

    @classmethod
    def stream_fastq_batches(cls, path: str) -> "Iterator[ReadBatch]":
        """
        Lee un FASTQ como lotes columnares :class:`ReadBatch` — la vía rápida
        para control de calidad.

        Filtrar por calidad media se vuelve una operación NumPy sobre todo el
        lote, sin fabricar un objeto por lectura.

        Example
        -------
        >>> total = buenas = 0
        >>> for batch in SmartImporter.stream_fastq_batches("reads.fastq"):
        ...     mask = batch.passes(20)        # 1 op NumPy para miles de lecturas
        ...     total  += len(batch)
        ...     buenas += int(mask.sum())
        >>> print(f"{buenas}/{total} lecturas con calidad media ≥ 20")
        """
        yield from cls._stream_columnar(path, 0, fastq=True)

    @classmethod
    def stream_fastq(cls, path: str) -> Iterator[FastqRecord]:
        """
        Generador de bajo consumo de RAM para archivos FASTQ.

        Cada ``FastqRecord`` contiene la secuencia 5-bit empaquetada y las
        calidades Phred (valor entero 0–93, ya restado el offset ASCII de 33).

        Parameters
        ----------
        path : str
            Ruta al archivo FASTQ (no comprimido).

        Yields
        ------
        FastqRecord — uno por lectura, en orden de fichero.

        Example
        -------
        >>> for rec in SmartImporter.stream_fastq("reads.fastq"):
        ...     if rec.passes_quality(20):
        ...         prot = SmartTranslator.translate(rec.sequence)
        """
        if _C_BATCH_AVAILABLE:
            yield from cls._stream_batch(path, 0, fastq=True)
            return

        if not _C_PARSER_AVAILABLE:
            yield from cls._stream_fastq_python(path)
            return

        hdr_buf   = ctypes.create_string_buffer(cls._STREAM_HDR)
        codes_buf = np.empty(cls._STREAM_SEQ, dtype=np.uint8)
        qual_buf  = np.empty(cls._STREAM_SEQ, dtype=np.uint8)

        handle = _c_parser_open(path)
        if not handle:
            raise IOError(f"No se puede abrir el archivo: {path!r}")

        try:
            while True:
                ret, n, q = _c_parser_next_fastq(
                    handle, hdr_buf, codes_buf, qual_buf
                )
                if ret <= 0:
                    break
                packed = BitPacker.pack(codes_buf[:n])
                header = hdr_buf.value.decode("ascii", errors="replace")
                seq = PackedSequence(
                    header    = header,
                    seq_type  = SeqType.NUCLEOTIDE,
                    n_symbols = n,
                    data      = packed,
                )
                yield FastqRecord(
                    sequence = seq,
                    quality  = qual_buf[:q].copy(),
                )
        finally:
            _c_parser_close(handle)

    @classmethod
    def _stream_fastq_python(cls, path: str) -> Iterator[FastqRecord]:
        """Fallback Python puro para FASTQ (sin motor C)."""
        with open(path, "r", encoding="ascii", errors="replace") as fh:
            while True:
                line1 = fh.readline()
                if not line1:
                    break
                line1 = line1.strip()
                if not line1.startswith("@"):
                    continue
                header  = line1[1:]
                seq_raw = fh.readline().strip()
                fh.readline()          # línea '+comment'
                qual_raw = fh.readline().strip()
                if not seq_raw:
                    continue
                seq = cls._encode(seq_raw, header, None)
                q   = np.frombuffer(qual_raw.encode("ascii"), dtype=np.uint8)
                q   = np.clip(q.astype(np.int16) - 33, 0, 93).astype(np.uint8)
                yield FastqRecord(sequence=seq, quality=q)

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
    print("  BioForge — biocore.py — Unified 5-bit bioinformatics engine demo")
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
