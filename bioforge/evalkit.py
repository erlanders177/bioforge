"""
evalkit.py
══════════════════════════════════════════════════════════════════════
El JUEZ HONESTO de predictores evolutivos — L6.

Un predictor que no se mide contra la baseline correcta es una opinión con decimales.
Este módulo coge **cualquier** función de puntuación y le aplica la batería de pruebas
que hace falta para no engañarse. Cada prueba nació de un autoengaño REAL, cazado
midiendo:

  1. ¿Le gana a la INGENUA ("mañana = hoy")?  → skill, no exactitud a secas.
     Predecir frecuencias empata con la ingenua a todo horizonte: sin este juez,
     cualquiera publica un 0.98 de exactitud que no aporta nada.
  2. ¿Le gana al MEJOR EJE TRIVIAL?  → el listón NO es 0.5.
     Nuestro propio AUC de 0.80 venía de "los sitios variables siguen variando":
     casi una tautología. Contra el azar parecía brillante; contra contar, no.
  3. ¿Y en el régimen DIFÍCIL?  → separa mutaciones ya circulantes de NUEVAS.
     Sobre las circulantes basta con contar; el mérito está en las que aún no existen.
  4. ¿La ventaja es ROBUSTA o suerte de muestreo?  → bootstrap con IC95%.
     Varios "wins" nuestros se evaporaron al poner intervalos.
  5. ¿Hay FUGA DE PREENTRENAMIENTO?  → se parte por épocas.
     ESM-2 caía −0.20 después de su fecha de corte: recordaba, no predecía. Los ejes
     sin preentrenar sirven de control: si cae solo el modelo, es memoria.
  6. ¿GENERALIZA a un organismo que no ha visto?  → validación cruzada entre virus.

Nada de esto exige GPU. Exige honestidad, que es más barata y más rara.

Uso
───
    from bioforge import EvolutionBenchmark

    bench = EvolutionBenchmark(sequences, dates)      # secuencias fechadas REALES
    print(bench.judge(mi_predictor))                  # informe honesto

``mi_predictor(ctx)`` recibe un :class:`Context` con lo conocido hasta el momento y
devuelve una puntuación por mutación candidata. Cuanto más alta, más probable que
esa mutación suba.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional, Sequence

import numpy as np

from .biocore import SequenceValueError
from .evolution import (
    _bin_ids,
    _conservation_table,
    _freqs,
    _mutability,
    _mutability_gate,
    _prepare,
)

__all__ = ["EvolutionBenchmark", "Context", "Report"]

_AA20 = np.frombuffer(b"ACDEFGHIKLMNPQRSTVWY", dtype=np.uint8)


class Context(NamedTuple):
    """Lo que un predictor puede mirar: SOLO el pasado (garantizado leak-free).

    freq      : (bins, símbolos, L) frecuencia alélica por bin temporal, hasta ahora.
    symbols   : (símbolos,) alfabeto, en el orden de la dimensión 1 de ``freq``.
    sites     : (n,) columna de cada mutación candidata.
    alleles   : (n,) índice de alelo (en ``symbols``) de cada candidata.
    horizon   : a cuántos bins vista se pregunta.
    """
    freq: np.ndarray
    symbols: np.ndarray
    sites: np.ndarray
    alleles: np.ndarray
    horizon: int

    @property
    def current(self) -> np.ndarray:
        """Frecuencia actual de cada candidata (la baseline de "contar")."""
        return self.freq[-1][self.alleles, self.sites]

    @property
    def is_novel(self) -> np.ndarray:
        """True en las candidatas que NUNCA se han visto (el régimen difícil)."""
        return self.freq.max(axis=0)[self.alleles, self.sites] == 0


class Report(NamedTuple):
    """Veredicto de un predictor. ``verdict`` resume en una línea."""
    auc: float
    auc_novel: float
    skill_vs_naive: float
    ci95: tuple
    best_trivial: float
    best_trivial_name: str
    beats_trivial: bool
    leakage: Optional[float]
    n_candidates: int
    n_folds: int
    verdict: str

    def __str__(self) -> str:
        L = [
            "── VEREDICTO ─────────────────────────────────────────────",
            f"  AUC global            {self.auc:.3f}",
            f"  AUC en NUEVAS         {self.auc_novel:.3f}   (el régimen difícil)",
            f"  listón trivial        {self.best_trivial:.3f}   ({self.best_trivial_name})",
            f"  IC95% del AUC         [{self.ci95[0]:.3f}, {self.ci95[1]:.3f}]",
        ]
        if self.leakage is not None:
            L.append(f"  fuga temporal         {self.leakage:+.3f}   "
                     f"{'⚠ SOSPECHA' if self.leakage < -0.05 else 'sin señal'}")
        L += [f"  ({self.n_candidates:,} candidatas · {self.n_folds} cortes)",
              "", f"  → {self.verdict}"]
        return "\n".join(L)


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    """AUC de Mann-Whitney con empates promediados (sin scipy). 0.5 = azar."""
    npos = int(label.sum())
    nneg = int(label.size - npos)
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    ranks = np.empty(score.size, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1)
    s = score[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[label].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


class EvolutionBenchmark:
    """Banco de pruebas honesto para predictores de evolución.

    Se construye con secuencias REALES fechadas (proteína o nucleótido alineables) y
    juzga cualquier función de puntuación con la batería completa. El backtest es
    rodante y leak-free: en cada corte el predictor solo ve bins anteriores.
    """

    def __init__(self, sequences: Sequence[str], dates: Sequence[float], *,
                 n_bins: Optional[int] = None, align: bool = True,
                 rise: float = 0.05, min_freq_rise: float = 0.5):
        if len(sequences) < 10:
            raise SequenceValueError(
                f"hacen falta ≥10 secuencias fechadas para un backtest con sentido "
                f"(se recibieron {len(sequences)}).")
        arr, t, symbols = _prepare(sequences, dates, align)
        self.bins, self.nb = _bin_ids(t, n_bins)
        if self.nb < 3:
            raise SequenceValueError(
                f"hacen falta ≥3 bins temporales para un backtest rodante "
                f"(hay {self.nb}). Usa fechas más finas o n_bins.")
        self.freq = _freqs(arr, self.bins, self.nb, symbols)
        self.symbols = symbols
        self.times = np.unique(t)
        self._rise = rise
        self._minor = min_freq_rise
        self._real = np.isin(symbols, _AA20)

    # ── candidatas y verdad de un corte ──────────────────────────────────────
    def _fold(self, k: int, horizon: int):
        cur = self.freq[:k][-1]
        target = self.freq[k + horizon - 1]
        cand = (cur < self._minor) & self._real[:, None]
        si, li = np.nonzero(cand)
        rose = ((target - cur) >= self._rise)[si, li]
        ctx = Context(freq=self.freq[:k], symbols=self.symbols,
                      sites=li, alleles=si, horizon=horizon)
        return ctx, rose

    def _trivial_axes(self, ctx: Context) -> "dict[str, np.ndarray]":
        """Los ejes que cualquiera tiene gratis — el listón de verdad."""
        cur = ctx.freq[-1]
        cons = _conservation_table(self.symbols)[:, cur.argmax(axis=0)]
        mut = _mutability_gate(_mutability(ctx.freq))
        return {
            "frecuencia actual": ctx.current,
            "mutabilidad del sitio": mut[ctx.sites],
            "conservación": cons[ctx.alleles, ctx.sites],
        }

    # ── el juicio ────────────────────────────────────────────────────────────
    def judge(self, predictor: Callable[[Context], np.ndarray], *,
              horizon: int = 1, n_boot: int = 30,
              leak_cutoff: Optional[float] = None,
              seed: int = 0) -> Report:
        """Somete ``predictor`` a la batería completa y devuelve el veredicto.

        ``leak_cutoff``: si el predictor usa un modelo preentrenado, pasa aquí su
        fecha de corte. Se compara su rendimiento antes y después: si cae SOLO él
        (los ejes triviales sirven de control), estaba recordando, no prediciendo.
        """
        rng = np.random.default_rng(seed)
        sc_all, y_all, novel_all, era_all = [], [], [], []
        triv_all: "dict[str, list]" = {}
        folds = 0
        for k in range(2, self.nb - horizon + 1):
            ctx, rose = self._fold(k, horizon)
            if ctx.sites.size == 0 or rose.sum() < 3 or (~rose).sum() < 3:
                continue
            s = np.asarray(predictor(ctx), dtype=np.float64).ravel()
            if s.size != ctx.sites.size:
                raise SequenceValueError(
                    f"el predictor devolvió {s.size} puntuaciones para "
                    f"{ctx.sites.size} candidatas.")
            sc_all.append(s); y_all.append(rose); novel_all.append(ctx.is_novel)
            era_all.append(np.full(s.size, self.times[k + horizon - 1]))
            for name, v in self._trivial_axes(ctx).items():
                triv_all.setdefault(name, []).append(v)
            folds += 1
        if folds == 0:
            raise SequenceValueError(
                "ningún corte evaluable: no hay suficientes mutaciones que suban. "
                "Prueba más secuencias, más bins o un `rise` menor.")

        score = np.concatenate(sc_all)
        y = np.concatenate(y_all).astype(bool)
        novel = np.concatenate(novel_all)
        era = np.concatenate(era_all)
        auc = _auc(score, y)
        auc_novel = _auc(score[novel], y[novel]) if 3 <= int(y[novel].sum()) else float("nan")

        # listón trivial: el MEJOR eje gratis, no el azar
        best_name, best_val = "azar", 0.5
        for name, parts in triv_all.items():
            a = _auc(np.concatenate(parts), y)
            if not np.isnan(a) and max(a, 1 - a) > best_val:
                best_val, best_name = max(a, 1 - a), name

        # skill sobre la ingenua (persistir la frecuencia actual)
        naive = _auc(np.concatenate(triv_all["frecuencia actual"]), y)
        skill = (auc - naive) / (1.0 - naive) if naive < 1.0 else 0.0

        # bootstrap: ¿robusto o suerte?
        boots = []
        n = score.size
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            if y[idx].sum() < 3 or (~y[idx]).sum() < 3:
                continue
            boots.append(_auc(score[idx], y[idx]))
        ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) \
            if len(boots) >= 5 else (float("nan"), float("nan"))

        # fuga de preentrenamiento
        leak = None
        if leak_cutoff is not None:
            pre, post = era <= leak_cutoff, era > leak_cutoff
            if y[pre].sum() >= 3 and y[post].sum() >= 3:
                a_pre, a_post = _auc(score[pre], y[pre]), _auc(score[post], y[post])
                ctrl = np.concatenate(triv_all[best_name])
                c_pre, c_post = _auc(ctrl[pre], y[pre]), _auc(ctrl[post], y[post])
                # caída del modelo DESCONTANDO la del control (la época puede ser peor)
                leak = (a_post - a_pre) - (c_post - c_pre)

        beats = bool(auc > best_val + 0.005)
        if np.isnan(ci[0]) or ci[0] <= 0.5:
            verdict = "NO demostrado: el IC95% toca el azar."
        elif not beats:
            verdict = (f"NO APORTA: no supera al eje trivial '{best_name}' "
                       f"({best_val:.3f}). Ese es el listón, no el 0.5.")
        elif leak is not None and leak < -0.05:
            verdict = (f"SOSPECHA DE FUGA: cae {leak:+.3f} tras el corte, más que el "
                       f"control. Puede estar recordando, no prediciendo.")
        elif not np.isnan(auc_novel) and auc_novel < 0.55:
            verdict = ("APORTA solo en lo fácil: en mutaciones NUEVAS (donde contar no "
                       "sirve) se queda en el azar.")
        else:
            verdict = f"APORTA de verdad: supera el listón trivial y aguanta en NUEVAS."
        return Report(auc=auc, auc_novel=auc_novel, skill_vs_naive=skill, ci95=ci,
                      best_trivial=best_val, best_trivial_name=best_name,
                      beats_trivial=beats, leakage=leak,
                      n_candidates=int(score.size), n_folds=folds, verdict=verdict)

    def cross_validate(self, predictor, others: "dict[str, EvolutionBenchmark]",
                       **kw) -> "dict[str, Report]":
        """Juzga el mismo predictor en OTROS organismos — el examen de generalización.

        Un predictor que solo funciona en el virus con el que se ajustó no es un
        predictor, es un ajuste. Aquí se ve enseguida.
        """
        return {name: b.judge(predictor, **kw) for name, b in others.items()}
