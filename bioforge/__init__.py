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
from .evolution import (
    BacktestResult,
    CladePrediction,
    EscapeResult,
    EvolutionResult,
    FusionResult,
    GrowthResult,
    LineageSystem,
    MutationRanking,
    backtest_evolution,
    designate_lineages,
    escape_potential,
    escape_weights,
    estimate_growth,
    predict_clade,
    predict_evolution,
    predict_fusion,
    rank_mutations,
    site_mutability,
)
from .evalkit import Context, EvolutionBenchmark, Report
from .genomemap import GenomeAligner, Mapping
from .nanopore import (
    EventTable,
    SignalRead,
    basecall,
    detect_events,
    estimate_pore_model,
    normalize_signal,
    read_fast5,
    read_pod5,
    viterbi_basecall,
)
from .realitycheck import RealityCheck, Verdict
from .msa import MSAResult, align_multiple
from .smart_translator import SmartTranslator

__version__ = "9.1.0"
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
    # Genome mapper (v3.0)
    "GenomeAligner",
    "Mapping",
    # Multiple sequence alignment (v6.3)
    "align_multiple",
    "MSAResult",
    # Evolution predictor (Fase 1-5)
    "predict_evolution",
    "backtest_evolution",
    "estimate_growth",
    "escape_potential",
    "predict_fusion",
    "predict_clade",
    "site_mutability",
    # Juez honesto de predictores (L6, v8.0)
    "EvolutionBenchmark",
    "Context",
    "Report",
    # Filtro de realidad: ¿esta mutación sobreviviría? (L6, v8.0)
    "RealityCheck",
    "Verdict",
    # Nanoporo: señal cruda → bases, basecaller clásico NumPy puro (v9.0)
    "SignalRead",
    "EventTable",
    "read_pod5",
    "read_fast5",
    "normalize_signal",
    "detect_events",
    "estimate_pore_model",
    "viterbi_basecall",
    "basecall",
    # Linajes estables (Pango/autolin, v7.0)
    "designate_lineages",
    "escape_weights",
    "LineageSystem",
    # Ranking de mutaciones + modelo entrenado (v7.0)
    "rank_mutations",
    "MutationRanking",
    "EvolutionResult",
    "BacktestResult",
    "GrowthResult",
    "EscapeResult",
    "FusionResult",
    "CladePrediction",
    # Pipeline
    "run",
    "build_report",
    "AnalysisResult",
]
