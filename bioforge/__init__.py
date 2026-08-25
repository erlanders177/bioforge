"""
BioForge — high-performance bioinformatics engine for Edge Computing.

Quick start
-----------
>>> from bioforge import SmartImporter, SmartTranslator, SequenceAligner, SeqType
>>> seqs = SmartImporter.from_string(">gene\\nATGAAAGGGTAA\\n")
>>> prot = SmartTranslator.translate(seqs[0])
>>> prot.to_string()
'MKG'

Organización (por FUNCIÓN, no por capas)
----------------------------------------
``core`` almacenamiento 5-bit y lectura · ``sequence`` traducción ·
``align`` alineamiento · ``mapping`` mapeo de genomas · ``evolution``
predicción + juez + filtro · ``nanopore`` basecaller · ``io`` calidad/compresión ·
``engine`` motor C · ``app`` app de escritorio · ``cli`` comandos.

Carga PEREZOSA (v10.1)
----------------------
Importar ``bioforge`` **no** carga el motor entero: cada nombre trae su módulo
sólo cuando se usa (PEP 562). Si solo traduces ADN, el basecaller, el mapeador y
la evolución nunca se cargan — coherente con el ADN Edge del proyecto. La API
pública no cambia: ``from bioforge import SequenceAligner`` funciona igual.
"""

from typing import TYPE_CHECKING

__version__ = "10.1.0"
__author__  = "Aarón Aranda Torrijos"

# ── mapa nombre público -> módulo que lo define ───────────────────────────────
# Es la única fuente de verdad de la API pública: __all__ y la carga perezosa
# salen de aquí, así que no pueden desincronizarse.
_EXPORTS: dict[str, str] = {}


def _register(module: str, *names: str) -> None:
    for n in names:
        _EXPORTS[n] = module


_register(
    "bioforge.core.biocore",
    # errores
    "BioForgeError", "SequenceTypeError", "SequenceValueError", "TranslationError",
    "AlignmentError", "BioForgeIOError", "EngineError",
    # tipos base y lectura
    "BioCode", "SeqType", "NUC_LUT", "AA_LUT", "BitPacker", "PackedSequence",
    "FastqRecord", "SequenceBatch", "ReadBatch", "SmartImporter", "SequenceStats",
    "compute_stats",
)
_register("bioforge.sequence.translator", "SmartTranslator")
_register("bioforge.align.pairwise",
          "SequenceAligner", "format_alignment", "Mutation", "AlignmentResult")
_register("bioforge.align.msa", "align_multiple", "MSAResult")
_register("bioforge.mapping.genomemap", "GenomeAligner", "Mapping")
_register("bioforge.variants.pileup", "Pileup", "pileup", "pileup_from_mappings")
_register("bioforge.variants.caller", "Variant", "call_variants", "write_vcf")
_register("bioforge.phylo.distance", "DistanceMatrix", "distance_matrix")
_register("bioforge.phylo.tree", "Clade", "Tree", "neighbor_joining", "upgma",
          "build_tree", "bootstrap_support", "wpgma")
_register(
    "bioforge.evolution.predict",
    "predict_evolution", "backtest_evolution", "estimate_growth", "escape_potential",
    "predict_fusion", "predict_clade", "site_mutability", "rank_mutations",
    "designate_lineages", "escape_weights", "LineageSystem", "MutationRanking",
    "EvolutionResult", "BacktestResult", "GrowthResult", "EscapeResult",
    "FusionResult", "CladePrediction",
)
_register("bioforge.evolution.evalkit", "EvolutionBenchmark", "Context", "Report")
_register("bioforge.evolution.realitycheck", "RealityCheck", "Verdict")
_register(
    "bioforge.nanopore.basecaller",
    "SignalRead", "EventTable", "read_pod5", "read_fast5", "normalize_signal",
    "detect_events", "estimate_pore_model", "viterbi_basecall", "basecall",
)
_register("bioforge.cli.analyze", "run", "build_report", "AnalysisResult")

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """Resuelve un nombre público cargando SOLO su módulo (PEP 562)."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'bioforge' has no attribute {name!r}")
    from importlib import import_module
    value = getattr(import_module(module), name)
    globals()[name] = value      # se cachea: los accesos siguientes son directos
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})


# Para editores y comprobadores de tipos: los nombres existen aunque se carguen
# de forma perezosa en tiempo de ejecución.
if TYPE_CHECKING:                                   # pragma: no cover
    from .align.msa import MSAResult, align_multiple
    from .align.pairwise import (
        AlignmentResult,
        Mutation,
        SequenceAligner,
        format_alignment,
    )
    from .cli.analyze import AnalysisResult, build_report, run
    from .core.biocore import (
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
    from .evolution.evalkit import Context, EvolutionBenchmark, Report
    from .evolution.predict import (
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
    from .evolution.realitycheck import RealityCheck, Verdict
    from .mapping.genomemap import GenomeAligner, Mapping
    from .nanopore.basecaller import (
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
    from .sequence.translator import SmartTranslator
    from .phylo.distance import DistanceMatrix, distance_matrix
    from .phylo.tree import (
        Clade,
        Tree,
        bootstrap_support,
        build_tree,
        neighbor_joining,
        upgma,
        wpgma,
    )
    from .variants.caller import Variant, call_variants, write_vcf
    from .variants.pileup import Pileup, pileup, pileup_from_mappings
