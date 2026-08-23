"""
bioforge.evolution — predecir evolución, y JUZGAR esa predicción con honestidad.

Tres piezas que se usan juntas:

* ``predict``      — ranking de mutaciones, linajes estables y backtest honesto.
* ``evalkit``      — el JUEZ: somete cualquier predictor al listón trivial, al
  régimen de mutaciones NUEVAS, a IC bootstrap y al detector de fuga.
* ``realitycheck`` — el FILTRO: ¿esta mutación concreta tiene tracción real?
  (OBSERVADO = evidencia · ESTIMADO = conjetura).

``fetch`` descarga secuencias fechadas del NCBI y ``ai`` es el eje opcional ESM-2.
"""

from .evalkit import Context, EvolutionBenchmark, Report
from .predict import (
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
from .realitycheck import RealityCheck, Verdict

__all__ = [
    # predicción
    "predict_evolution", "backtest_evolution", "estimate_growth", "escape_potential",
    "predict_fusion", "predict_clade", "site_mutability", "rank_mutations",
    "designate_lineages", "escape_weights", "LineageSystem", "MutationRanking",
    "EvolutionResult", "BacktestResult", "GrowthResult", "EscapeResult",
    "FusionResult", "CladePrediction",
    # juez honesto
    "EvolutionBenchmark", "Context", "Report",
    # filtro de realidad
    "RealityCheck", "Verdict",
]


def __getattr__(name):
    """Compatibilidad: antes ``bioforge.evolution`` era un módulo plano.

    Reenvía a :mod:`bioforge.evolution.predict` cualquier nombre que no exporte el
    paquete (incluidos los internos), para no romper código que ya los usaba.
    """
    from . import predict as _predict
    try:
        return getattr(_predict, name)
    except AttributeError:
        raise AttributeError(
            f"module 'bioforge.evolution' has no attribute {name!r}") from None
