"""
BioForge — high-performance bioinformatics engine for Edge Computing.

Quick start
-----------
>>> from bioforge import SmartImporter, SmartTranslator, SequenceAligner, SeqType
>>> seqs = SmartImporter.from_string(">gene\\nATGAAAGGGTAA\\n")
>>> prot = SmartTranslator.translate(seqs[0])
>>> prot.to_string()
'MKG'
"""

from .biocore import (
    BioForgeError,
    SequenceTypeError,
    SequenceValueError,
    TranslationError,
    AlignmentError,
    BioCode,
    SeqType,
    NUC_LUT,
    AA_LUT,
    BitPacker,
    PackedSequence,
    FastqRecord,
    SequenceBatch,
    ReadBatch,
    SmartImporter,
    SequenceStats,
    compute_stats,
)
from .smart_translator import SmartTranslator
from .aligner import SequenceAligner, format_alignment, Mutation, AlignmentResult
from .analyze import run, build_report, AnalysisResult

__version__ = "2.3.0"
__author__  = "Aarón Aranda Torrijos"

__all__ = [
    # Exceptions
    "BioForgeError",
    "SequenceTypeError",
    "SequenceValueError",
    "TranslationError",
    "AlignmentError",
    # Core types
    "BioCode",
    "SeqType",
    "NUC_LUT",
    "AA_LUT",
    "BitPacker",
    "PackedSequence",
    "FastqRecord",
    "SequenceBatch",
    "ReadBatch",
    "SmartImporter",
    "SequenceStats",
    "compute_stats",
    # Translator
    "SmartTranslator",
    # Aligner
    "SequenceAligner",
    "format_alignment",
    "Mutation",
    "AlignmentResult",
    # Pipeline
    "run",
    "build_report",
    "AnalysisResult",
]
