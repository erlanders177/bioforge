"""
bioforge/realitycheck.py — ¿esta mutación sobreviviría AHÍ FUERA?

El filtro de realidad. Las herramientas del campo (ESM-2, EVEscape, un ensayo DMS
de laboratorio) responden a *"¿esta mutación podría escapar / desestabilizar?"* y lo
hacen bien. Pero contestan **en el tubo de ensayo o en el ordenador**, y la mayoría
de las mutaciones "preocupantes" nunca llegan a nada en la población real: rompen
otra cosa, o simplemente no tienen ocasión.

Este módulo responde a la pregunta SIGUIENTE, que es la que decide si algo importa:

    de todas esas candidatas, ¿cuáles tienen tracción REAL en el mundo?

No compite con esas herramientas: se enchufa detrás de cualquiera de ellas.

    rc = RealityCheck(secuencias_reales, fechas)
    for v in rc.filter(lista_de_otra_herramienta):
        print(v)

Dos niveles de respuesta, SIEMPRE etiquetados por separado — porque mezclar un dato
con una estimación es la forma más fácil de mentir sin querer:

  OBSERVADO  la mutación ya existe en el registro histórico. Traemos su trayectoria
             real (frecuencia actual, pico, tendencia, desde cuándo). Es EVIDENCIA.
  ESTIMADO   nunca vista. Solo podemos estimarla con el modelo entrenado, a partir
             de lo mutable que es ese sitio y lo tolerable que es el cambio. Es una
             CONJETURA, y se dice.

Y una tercera honestidad, la que más cuesta: cada nivel se mide por separado con el
juez de ``evalkit`` y **el veredicto viaja con su propia fiabilidad**. Si el nivel
ESTIMADO acierta poco, sale escrito en el propio veredicto en vez de esconderse en
un README. Ver ``RealityCheck.reliability``.

Las probabilidades están CALIBRADAS contra el histórico: un 0.70 significa que, de
las mutaciones que sacaron ~0.70 en el pasado, aproximadamente el 70% despegó. Un
logit crudo de la red no significaría nada.
"""

from typing import NamedTuple, Optional, Sequence

import numpy as np

from ..core.biocore import SequenceValueError
from .evalkit import _auc
from .predict import (
    _AA20,
    _bin_ids,
    _conservation_table,
    _freqs,
    _mutability,
    _mutability_gate,
    _prepare,
    score_mutations,
)

__all__ = ["RealityCheck", "Verdict"]

_OBS = "OBSERVADO"
_EST = "ESTIMADO"


def _parse_mutation(text: str) -> tuple[Optional[str], int, str]:
    """``"N121K"`` → ``("N", 121, "K")``; ``"121K"`` → ``(None, 121, "K")``.

    Acepta el formato estándar del campo (1-based). El aminoácido original es
    opcional: si se da, se comprueba contra la referencia y se avisa si no cuadra.
    """
    s = "".join(str(text).split()).upper()
    if not s:
        raise SequenceValueError("mutación vacía")
    alt = s[-1]
    if not ("A" <= alt <= "Z" or alt == "*"):
        raise SequenceValueError(f"mutación '{text}': falta el aminoácido nuevo al final")
    body = s[:-1]
    wt = None
    if body and not body[0].isdigit():
        wt, body = body[0], body[1:]
    if not body.isdigit():
        raise SequenceValueError(
            f"mutación '{text}': formato esperado 'N121K' o '121K' (posición 1-based)")
    return wt, int(body), alt


class Verdict(NamedTuple):
    """El dictamen sobre UNA mutación. ``str(v)`` lo imprime en claro."""

    mutation: str
    site: int                    # 1-based, como en el campo
    wildtype: str                # aminoácido de referencia hoy
    alt: str                     # aminoácido propuesto
    tier: str                    # OBSERVADO (evidencia) | ESTIMADO (conjetura)
    probability: float           # calibrada contra el histórico
    label: str                   # veredicto en una línea
    freq_now: float              # frecuencia actual (0.0 si nunca vista)
    freq_peak: float             # máximo histórico alcanzado
    trend: float                 # cambio en el último tramo
    first_seen: Optional[float]  # fecha del primer avistamiento
    reliability: float           # AUC medido de ESTE nivel (no del global)
    note: str

    def __str__(self) -> str:
        L = [f"{self.mutation}  ({self.tier})",
             f"  probabilidad de despegar   {self.probability:.2f}",
             f"  → {self.label}"]
        if self.tier == _OBS:
            L.append(f"  histórico: ahora {self.freq_now:.1%} · pico "
                     f"{self.freq_peak:.1%} · tendencia {self.trend:+.1%}"
                     + (f" · visto desde {self.first_seen:.0f}"
                        if self.first_seen is not None else ""))
        L.append(f"  fiabilidad de este nivel: AUC {self.reliability:.3f}"
                 f"{'  ⚠ flojo' if self.reliability < 0.6 else ''}")
        if self.note:
            L.append(f"  nota: {self.note}")
        return "\n".join(L)


class RealityCheck:
    """Filtro de realidad: juzga MUTACIONES concretas, vengan de donde vengan.

    Se construye una vez con secuencias reales fechadas (el registro contra el que
    se contrasta) y luego responde a cuantas mutaciones se le pregunten.

    >>> rc = RealityCheck(sequences, dates)          # doctest: +SKIP
    >>> print(rc.check("N121K"))                     # doctest: +SKIP
    >>> supervivientes = rc.filter(candidatas_de_otra_herramienta)   # doctest: +SKIP

    ``horizon`` es a cuántos tramos vista se pregunta. ``traction`` es el umbral de
    "sobrevivir de verdad": una mutación cuenta como superviviente si en el horizonte
    ALCANZA O MANTIENE esa frecuencia (0.5 = llega a dominante, por defecto).

    Por qué ``traction`` y no "subió un 5%": una mutación ya instalada al 98% no
    puede subir más, pero es la superviviente más clara que existe. Medir "¿sube?"
    la marcaría como fracaso justo en el caso más real (efecto techo). La pregunta
    correcta para un filtro de supervivencia es "¿está / llega a estar ahí?", no
    "¿crece?". Esto distingue este módulo de ``evalkit``, que sí mide predicción de
    ASCENSO — otra pregunta, otra herramienta.
    """

    def __init__(self, sequences: Sequence[str], dates: Sequence[float], *,
                 n_bins: Optional[int] = None, align: bool = True,
                 horizon: int = 1, traction: float = 0.5) -> None:
        if len(sequences) != len(dates):
            raise SequenceValueError(
                f"secuencias ({len(sequences)}) y fechas ({len(dates)}) no cuadran")
        if len(sequences) < 20:
            raise SequenceValueError(
                "hacen falta al menos 20 secuencias fechadas para contrastar")
        if horizon < 1:
            raise SequenceValueError("horizon debe ser >= 1")

        arr, t, symbols = _prepare(sequences, dates, align=align)
        times = np.asarray(t, dtype=np.float64)
        bins, nb = _bin_ids(times, n_bins)
        if nb < 3:
            raise SequenceValueError(
                f"solo {nb} tramos temporales; hacen falta 3 para ver una trayectoria")

        self.freq = _freqs(arr, bins, nb, symbols)        # (tramos, símbolos, L)
        self.symbols = symbols
        self.nb, self.horizon, self.traction = nb, horizon, traction
        self.L = int(self.freq.shape[2])
        self.times = np.array([times[bins == b].mean() if np.any(bins == b) else np.nan
                               for b in range(nb)])

        self._mut = _mutability_gate(_mutability(self.freq))
        self._cons = _conservation_table(symbols)
        self._root = self.freq[-1].argmax(axis=0)          # referencia actual por sitio
        self._sym_ix = {chr(int(c)): i for i, c in enumerate(symbols)}
        self._ever = self.freq.max(axis=0) > 0             # (símbolos, L) ¿existió?

        self._calibrate()

    # ── calibración + fiabilidad, contra el propio histórico ─────────────────
    def _grid(self, k: int):
        """Features y verdad de TODAS las candidatas en el corte ``k`` (vectorizado)."""
        past, cur, prev = self.freq[:k], self.freq[k - 1], self.freq[k - 2]
        target = self.freq[k + self.horizon - 1]
        alle = np.nonzero(np.isin(self.symbols, _AA20))[0]
        ai = np.repeat(alle, self.L)
        si = np.tile(np.arange(self.L), alle.size)

        root = cur.argmax(axis=0)
        feats = np.column_stack([
            cur[ai, si],                                   # frecuencia
            self._cons[ai, root[si]],                      # conservación
            _mutability_gate(_mutability(past))[si],       # mutabilidad
            (cur - prev)[ai, si],                          # crecimiento
            np.full(ai.size, float(self.horizon)),         # horizonte
        ])
        # "sobrevive" = alcanza/mantiene presencia real en el horizonte (no "sube")
        y = (target[ai, si] >= self.traction).astype(np.float64)
        seen = (past.max(axis=0) > 0)[ai, si]
        return feats, y, seen

    def _calibrate(self) -> None:
        """Convierte el logit crudo en probabilidad REAL, por nivel y por separado.

        Un 0.70 debe significar "el 70% de las que puntuaron así despegaron". Se
        aprende del histórico con bins por cuantiles, y de paso se mide el AUC de
        cada nivel: la fiabilidad viaja pegada a cada veredicto.
        """
        F, Y, S = [], [], []
        for k in range(2, self.nb - self.horizon + 1):
            f, y, s = self._grid(k)
            F.append(f); Y.append(y); S.append(s)
        feats = np.concatenate(F); y = np.concatenate(Y); seen = np.concatenate(S)
        score = score_mutations(feats)

        self._curve, self.reliability = {}, {}
        for tier, mask in ((_OBS, seen), (_EST, ~seen)):
            sc, yy = score[mask], y[mask]
            if sc.size < 50 or yy.sum() < 3 or yy.sum() == yy.size:
                self._curve[tier] = None                   # sin base para calibrar
                self.reliability[tier] = float("nan")
                continue
            edges = np.unique(np.quantile(sc, np.linspace(0, 1, 11)))
            if edges.size < 3:
                self._curve[tier] = None
                self.reliability[tier] = float("nan")
                continue
            ix = np.clip(np.searchsorted(edges, sc, side="right") - 1, 0, edges.size - 2)
            cnt = np.bincount(ix, minlength=edges.size - 1).astype(np.float64)
            hit = np.bincount(ix, weights=yy, minlength=edges.size - 1)
            # suavizado de Laplace: un bin con 2 casos no puede decir "100%"
            rate = (hit + 1.0) / (cnt + 2.0)
            self._curve[tier] = (edges, rate)
            self.reliability[tier] = _auc(sc, yy > 0)

    def _probability(self, score: float, tier: str) -> float:
        curve = self._curve.get(tier)
        if curve is None:
            return float("nan")
        edges, rate = curve
        mid = 0.5 * (edges[:-1] + edges[1:])
        return float(np.interp(score, mid, rate))

    # ── la pregunta ──────────────────────────────────────────────────────────
    def check(self, mutation: str) -> Verdict:
        """Juzga UNA mutación en formato ``"N121K"`` o ``"121K"`` (posición 1-based)."""
        return self.check_many([mutation])[0]

    def check_many(self, mutations: Sequence[str]) -> list[Verdict]:
        """Juzga una lista entera. El trabajo pesado va vectorizado en una pasada.

        Resiliente por diseño: una entrada mal formada (formato imposible) NO tumba
        el lote — se devuelve como ``NO EVALUABLE`` con la causa en la nota. Es un
        filtro que recibe listas de OTRAS herramientas; que una línea sucia hunda las
        49 buenas sería inaceptable.
        """
        n = len(mutations)
        if n == 0:
            return []
        parsed: list[Optional[tuple]] = []
        parse_err: list[str] = []
        for m in mutations:                                     # bucle por registro
            try:
                parsed.append(_parse_mutation(m))
                parse_err.append("")
            except SequenceValueError as e:
                parsed.append(None)
                parse_err.append(str(e))
        bad_parse = np.array([pp is None for pp in parsed])
        safe = [pp if pp is not None else (None, 0, "?") for pp in parsed]

        pos = np.array([p for _, p, _ in safe])
        bad_pos = ((pos < 1) | (pos > self.L)) & ~bad_parse
        si = np.clip(pos, 1, self.L) - 1
        ai = np.array([self._sym_ix.get(a, -1) for _, _, a in safe])
        bad_aa = (ai < 0) & ~bad_parse
        ai_safe = np.where(ai < 0, 0, ai)

        cur, prev = self.freq[-1], self.freq[-2]
        freq_now = cur[ai_safe, si]
        freq_peak = self.freq.max(axis=0)[ai_safe, si]
        trend = (cur - prev)[ai_safe, si]
        feats = np.column_stack([
            freq_now,
            self._cons[ai_safe, self._root[si]],
            self._mut[si],
            trend,
            np.full(n, float(self.horizon)),
        ])
        score = score_mutations(feats)
        ever = self._ever[ai_safe, si]

        # primer avistamiento (vectorizado sobre los tramos)
        present = self.freq[:, ai_safe, si] > 0                 # (tramos, n)
        any_seen = present.any(axis=0)
        first_ix = present.argmax(axis=0)

        out = []
        for i, ((wt, p, alt), raw) in enumerate(zip(safe, mutations)):
            ref = chr(int(self.symbols[self._root[si[i]]]))
            note = ""
            if bad_parse[i]:
                note = parse_err[i]
            elif bad_pos[i]:
                note = (f"posición {p} fuera del alineamiento (1-{self.L}); "
                        "no se puede juzgar")
            elif bad_aa[i]:
                note = f"aminoácido '{alt}' no está en el alfabeto de estos datos"
            elif wt is not None and wt != ref:
                note = (f"ojo: das '{wt}' como original pero hoy la referencia en "
                        f"{p} es '{ref}' — ¿otra numeración?")

            if bad_parse[i] or bad_pos[i] or bad_aa[i]:
                out.append(Verdict(str(raw), int(p), ref if not bad_parse[i] else "?",
                                   alt, _EST, float("nan"), "NO EVALUABLE",
                                   0.0, 0.0, 0.0, None, float("nan"), note))
                continue

            tier = _OBS if ever[i] else _EST
            prob = self._probability(float(score[i]), tier)
            rel = self.reliability.get(tier, float("nan"))
            first = float(self.times[first_ix[i]]) if any_seen[i] else None

            if np.isnan(prob):
                label = "SIN BASE: no hay histórico suficiente para calibrar este nivel"
            elif tier == _OBS and freq_now[i] >= self.traction:
                label = ("YA ESTABLECIDA: no es una predicción, ya sobrevive "
                         f"({freq_now[i]:.0%})")
            elif prob >= 0.5:
                label = ("PLAUSIBLE: tracción real en la población"
                         if tier == _OBS else
                         "PLAUSIBLE (conjetura): sitio activo y cambio tolerable")
            elif prob >= 0.2:
                label = ("INCIERTA: señal débil, ni descartar ni apostar"
                         if tier == _OBS else
                         "INCIERTA (conjetura): nunca vista, señal débil")
            else:
                label = ("IMPROBABLE: lleva tiempo sin ir a ninguna parte"
                         if tier == _OBS else
                         "IMPROBABLE (conjetura): sitio quieto o cambio mal tolerado")
            if tier == _EST and not note:
                note = "nunca observada en estos datos: es una estimación, no evidencia"

            out.append(Verdict(str(raw), p, ref, alt, tier, prob, label,
                               float(freq_now[i]), float(freq_peak[i]),
                               float(trend[i]), first, float(rel), note))
        return out

    def filter(self, mutations: Sequence[str], *,
               min_probability: float = 0.5) -> list[Verdict]:
        """Criba la lista de otra herramienta y devuelve solo lo que tiene tracción.

        El caso de uso real: alguien corre EVEscape / ESM-2 / un ensayo DMS, saca 50
        mutaciones "preocupantes", y quiere saber cuáles importan de verdad ahí fuera.
        Ordenadas de más a menos probable.
        """
        keep = [v for v in self.check_many(mutations)
                if not np.isnan(v.probability) and v.probability >= min_probability]
        return sorted(keep, key=lambda v: -v.probability)

    def summary(self) -> str:
        """Resumen honesto de cuánto se puede fiar uno de cada nivel."""
        L = ["── FIABILIDAD DEL FILTRO (medida en este histórico) ──"]
        for tier in (_OBS, _EST):
            a = self.reliability.get(tier, float("nan"))
            if np.isnan(a):
                L.append(f"  {tier:10s} sin base suficiente para medir")
            else:
                L.append(f"  {tier:10s} AUC {a:.3f}"
                         + ("   ⚠ poco mejor que el azar" if a < 0.6 else ""))
        L.append(f"  ({self.nb} tramos · {self.L} sitios · horizonte {self.horizon})")
        return "\n".join(L)
