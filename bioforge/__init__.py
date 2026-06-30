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

from .aligner import AlignmentResult, Mutation, SequenceAligner, format_alignment
from .analyze import AnalysisResult, build_report, run
from .biocore import (
    AA_LUT,
    NUC_LUT,
    AlignmentError,
    BioCode,
    BioForgeError,
    BioForgeIOError,
    BitPacker,
    EngineError,
    FastqRecord,
    PackedSequence,
    ReadBatch,
    SeqType,
    SequenceBatch,
    SequenceStats,
    SequenceTypeError,
    SequenceValueError,
    SmartImporter,
    TranslationError,
    compute_stats,
)
from .smart_translator import SmartTranslator

__version__ = "2.3.0"
__author__  = "Aarón Aranda Torrijos"

__all__ = [
    # Exceptions
    "BioForgeError",
    "SequenceTypeError",
    "SequenceValueError",
    "TranslationError",
    "AlignmentError",
    "BioForgeIOError",
    "EngineError",
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
