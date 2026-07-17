"""
evocli.py
══════════════════════════════════════════════════════════════════════
CLI del predictor de evolución (L5) — ``bioforge-evolution``.

Envuelve las funciones ya probadas de ``evolution.py`` para que se usen sin escribir
Python. Tres subcomandos, cada uno una capacidad HONESTA del predictor:

  rank      ordena qué MUTACIONES subirán (el producto estrella; estilo EVEscape).
            La ingenua "mañana = hoy" no juega aquí: no ordena nada.
  backtest  mide la predicción contra esa ingenua y reporta el SKILL. Es el juez:
            si no le gana, no aporta, y lo dice.
  lineages  designa linajes ESTABLES (Pango/autolin sin árbol) y su jerarquía.

Entrada: un FASTA con la FECHA en la cabecera (año, o AAAA-MM). Se extrae con el mismo
parser que la descarga de NCBI. ``rank`` necesita PROTEÍNA (ejes físico-químicos);
``--translate`` traduce nucleótido antes.

Uso
───
  bioforge-evolution rank cepas.fasta --top 20
  bioforge-evolution backtest cepas.fasta
  bioforge-evolution lineages cepas.fasta
  python -m bioforge.evocli rank cepas.fasta --translate
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .biocore import (
    BioForgeError,
    BioForgeIOError,
    SeqType,
    SequenceValueError,
    SmartImporter,
)
from .evolution import (
    _assign_lineages,
    _own,
    _prepare,
    backtest_evolution,
    designate_lineages,
    rank_mutations,
)
from .fetch import _decimal_year, _parse_fasta
from .smart_translator import SmartTranslator

_YEAR = re.compile(r"(19|20)\d{2}")
_NUC = set("ACGTUN-")


def _year_from_header(header: str) -> Optional[float]:
    """Año decimal desde la cabecera. Prueba tokens (por |, espacio, /) con el parser
    de NCBI y prefiere el MÁS PRECISO (con mes) sobre un año pelado — así una cabecera
    con '.../2021|2021-03' resuelve a marzo, no a enero. Si ninguno cuadra, busca un
    año de 4 cifras suelto."""
    cands = [y for tok in re.split(r"[|\s/]+", header)
             if (y := _decimal_year(tok)) is not None]
    if cands:
        precisos = [y for y in cands if y != int(y)]    # los que llevan mes
        return precisos[0] if precisos else cands[0]
    m = _YEAR.search(header)
    return float(m.group(0)) if m else None


def _read_dated_fasta(path: str) -> tuple[list[str], list[float]]:
    """FASTA → (secuencias, años). Descarta registros sin fecha reconocible."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError as e:
        raise BioForgeIOError(f"archivo no encontrado: {path}") from e
    except OSError as e:                              # permisos, es-un-directorio…
        raise BioForgeIOError(f"no se pudo leer {path}: {e}") from e
    seqs, times, sin_fecha = [], [], 0
    for header, seq in _parse_fasta(text):
        y = _year_from_header(header)
        if y is None:
            sin_fecha += 1
            continue
        seqs.append(seq.upper())
        times.append(y)
    if sin_fecha:
        print(f"  ({sin_fecha} secuencias sin fecha en la cabecera — descartadas)",
              file=sys.stderr)
    if len(seqs) < 2:
        raise SequenceValueError(
            "hacen falta ≥2 secuencias con fecha en la cabecera (año o AAAA-MM).")
    return seqs, times


def _translate_all(seqs: list[str]) -> list[str]:
    """Traduce nucleótido→proteína con nuestro traductor; descarta lo intraducible."""
    out = []
    for s in seqs:
        try:
            packed = SmartImporter.from_string(f">s\n{s}\n",
                                               force_type=SeqType.NUCLEOTIDE)[0]
            out.append(SmartTranslator.translate(packed, warn_short=False).to_string())
        except BioForgeError:
            out.append(None)
    return out


def _maybe_translate(seqs: list[str], times: list[float], translate: bool):
    """Traduce si se pidió; alinea filas y descarta las que no traducen."""
    if not translate:
        return seqs, times
    prot = _translate_all(seqs)
    keep = [(p, t) for p, t in zip(prot, times) if p is not None]
    if len(keep) < 2:
        raise SequenceValueError("ninguna secuencia se pudo traducir a proteína.")
    return [p for p, _ in keep], [t for _, t in keep]


def _cli_rank(args) -> int:
    seqs, times = _read_dated_fasta(args.fasta)
    seqs, times = _maybe_translate(seqs, times, args.translate)
    r = rank_mutations(seqs, times, novel_only=args.novel, horizon=args.horizon,
                       method=args.method)
    if not r.ranked:
        print("No hay mutaciones candidatas (¿secuencias sin variación?).")
        return 0
    modo = "solo mutaciones NUEVAS" if args.novel else "todas las candidatas"
    combinador = "modelo entrenado" if args.method == "model" else "fusión a mano"
    print(f"MUTACIONES ordenadas por probabilidad de ascender ({modo}, "
          f"horizonte {args.horizon}, {combinador})")
    print(f"{'rango':>5}  {'sitio':>6} {'alelo':>5}  {'score':>8}  nueva")
    for i, (site, al, sc) in enumerate(r.ranked[:args.top], 1):
        marca = "sí" if r.novel.get((site, al)) else ""
        print(f"{i:>5}  {site + 1:>6} {al:>5}  {sc:>8.3f}  {marca:>5}")
    print("\nHonesto: ordena mutaciones (AUC ~0.7-0.9 en gripe HA, held-out). NO "
          "predice frecuencias\nexactas (eso empata con 'mañana = hoy'). "
          "Ver docs para los límites por régimen.")
    return 0


def _cli_backtest(args) -> int:
    seqs, times = _read_dated_fasta(args.fasta)
    r = backtest_evolution(seqs, times, method=args.method)
    print(f"BACKTEST — método '{r.method}' vs ingenua ('mañana = hoy')")
    print(f"  exactitud método : {r.method_accuracy:.4f}")
    print(f"  exactitud ingenua: {r.naive_accuracy:.4f}")
    print(f"  SKILL            : {r.skill:+.4f}  ({r.n_evaluations} cortes)")
    veredicto = ("le gana a la ingenua" if r.skill > 0 else
                 "NO le gana a la ingenua (no aporta)" if r.skill < 0 else
                 "empata con la ingenua")
    print(f"  → {veredicto}.")
    return 0


def _cli_lineages(args) -> int:
    seqs, times = _read_dated_fasta(args.fasta)
    arr, _, symbols = _prepare(seqs, times, align=True)
    sysd = designate_lineages(arr, symbols, min_size=args.min_size)
    labels = _assign_lineages(arr, sysd)
    counts = np.bincount(labels, minlength=sysd.n)
    print(f"LINAJES ESTABLES designados: {sysd.n} (estilo Pango/autolin, sin árbol)")
    for i in range(sysd.n):
        os_, oa = _own(sysd, i)
        muts = ", ".join(f"{s + 1}{chr(int(a))}" for s, a in zip(os_, oa)) or "(raíz)"
        padre = "" if sysd.parents[i] < 0 else f" ⊂ L{sysd.parents[i]}"
        print(f"  L{i}{padre}: {counts[i]:>4} seqs · definitorias: {muts}")
    return 0


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="bioforge-evolution",
        description="Predictor de evolución: ordena mutaciones, backtest honesto, "
                    "linajes estables. La fecha va en la cabecera FASTA (año o AAAA-MM).")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("rank", help="ordena qué mutaciones subirán")
    pr.add_argument("fasta", help="FASTA con fecha en la cabecera")
    pr.add_argument("--top", type=int, default=25, help="cuántas mostrar (25)")
    pr.add_argument("--novel", action="store_true", help="solo mutaciones nunca vistas")
    pr.add_argument("--horizon", type=int, default=1, help="periodos vista (1)")
    pr.add_argument("--translate", action="store_true", help="traducir nucleótido antes")
    pr.add_argument("--method", choices=("model", "manual"), default="model",
                    help="modelo entrenado (por defecto) o fusión a mano")
    pr.set_defaults(func=_cli_rank)

    pb = sub.add_parser("backtest", help="mide el skill vs 'mañana = hoy'")
    pb.add_argument("fasta", help="FASTA con fecha en la cabecera")
    pb.add_argument("--method", default="trend", help="método a evaluar (trend)")
    pb.set_defaults(func=_cli_backtest)

    pl = sub.add_parser("lineages", help="designa linajes estables (Pango/autolin)")
    pl.add_argument("fasta", help="FASTA con fecha en la cabecera")
    pl.add_argument("--min-size", type=int, default=10, dest="min_size",
                    help="tamaño mínimo de linaje (10)")
    pl.set_defaults(func=_cli_lineages)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    args = _parse_args(argv)
    try:
        return args.func(args)
    except BioForgeError as exc:                      # incluye BioForgeIOError (ficheros)
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
