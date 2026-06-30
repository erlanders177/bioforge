"""
qcreport.py
══════════════════════════════════════════════════════════════════════
Fast FASTQ quality report — la versión rápida y ligera de FastQC.

Una sola pasada sobre el archivo usando la API columnar de BioForge
(``SmartImporter.stream_fastq_batches``): sin crear un objeto por lectura,
RAM constante, y el grueso del trabajo en operaciones NumPy sobre el lote.

Métricas
────────
  • Resumen: nº lecturas, bases, longitud (min/media/max), GC global,
    calidad media global, % de lecturas con Q media ≥ 20 y ≥ 30.
  • Histograma de calidad media por lectura.
  • Histograma de %GC por lectura.
  • Calidad media por posición (el gráfico estrella de FastQC).
  • Composición por base (A/C/G/T/N) por posición.

Uso
───
  python -m bioforge.qcreport reads.fastq.gz
  python -m bioforge.qcreport reads.fastq --output informe.txt
  bioforge-qc reads.fastq.gz        (si está instalado el paquete)
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .biocore import SmartImporter, SequenceValueError

# Posiciones máximas que se siguen en los gráficos por-posición. Las lecturas
# más largas (p.ej. Nanopore) solo contribuyen sus primeras _MAXPOS bases a
# esos gráficos; las métricas escalares usan la lectura completa.
_MAXPOS = 1000


# ══════════════════════════════════════════════════════════════════════════════
# §1  RESULTADO
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QCReport:
    path:          str
    n_reads:       int
    total_bases:   int
    min_len:       int
    max_len:       int
    gc_overall:    float          # fracción 0..1
    mean_q_overall: float
    pct_q20:       float          # % de lecturas con calidad media ≥ 20
    pct_q30:       float
    meanq_hist:    np.ndarray     # bincount de int(calidad media por lectura)
    gc_hist:       np.ndarray     # bincount de int(%GC por lectura), 0..100
    pos_q_mean:    np.ndarray     # calidad media por posición (len = pos seguidas)
    base_frac:     np.ndarray     # (5, pos): fracción A/C/G/T/N por posición
    elapsed_s:     float

    @property
    def mean_len(self) -> float:
        return self.total_bases / self.n_reads if self.n_reads else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# §2  CÁLCULO (una sola pasada, columnar)
# ══════════════════════════════════════════════════════════════════════════════

def run(path: str) -> QCReport:
    """Calcula el informe de calidad de un FASTQ (.gz o plano)."""
    t0 = time.perf_counter()

    n_reads = 0
    total_bases = 0
    min_len = None
    max_len = 0
    q_sum_total = 0.0          # suma de todas las calidades → media global
    gc_bases = 0.0             # total de bases G+C → GC global
    pass20 = pass30 = 0

    meanq_hist = np.zeros(64, dtype=np.int64)       # Phred 0..63
    gc_hist    = np.zeros(101, dtype=np.int64)      # %GC 0..100

    pos_q_sum  = np.zeros(_MAXPOS, dtype=np.float64)
    pos_count  = np.zeros(_MAXPOS, dtype=np.int64)
    base_counts = np.zeros((5, _MAXPOS), dtype=np.int64)   # A,C,G,T,N

    for batch in SmartImporter.stream_fastq_batches(path):
        m = len(batch)
        if m == 0:
            continue
        nsy = np.asarray(batch.n_symbols)
        n_reads     += m
        total_bases += int(nsy.sum())
        bmin = int(nsy.min()); bmax = int(nsy.max())
        min_len = bmin if min_len is None else min(min_len, bmin)
        max_len = max(max_len, bmax)

        # ── Por lectura (vectorizado) ──────────────────────────────────────
        mq = batch.mean_quality()                  # float por lectura
        gc = batch.gc_content()                    # fracción por lectura
        q_sum_total += float((mq * nsy).sum())
        gc_bases    += float((gc * nsy).sum())
        pass20 += int((mq >= 20).sum())
        pass30 += int((mq >= 30).sum())
        meanq_hist += np.bincount(
            np.clip(mq, 0, 63).astype(np.int64), minlength=64)[:64]
        gc_hist += np.bincount(
            np.clip(gc * 100.0, 0, 100).astype(np.int64), minlength=101)[:101]

        # ── Por posición ───────────────────────────────────────────────────
        qm = batch.quality_matrix()                # (m,L) o None
        c2d = batch.decoded_2d()                    # (m,L) o None
        if qm is not None and c2d is not None:
            cols = min(qm.shape[1], _MAXPOS)
            pos_q_sum[:cols] += qm[:, :cols].sum(axis=0)
            pos_count[:cols] += m
            sub = c2d[:, :cols]
            for b in range(4):
                base_counts[b, :cols] += (sub == b).sum(axis=0)
            base_counts[4, :cols] += (sub > 3).sum(axis=0)
        else:
            # Longitud irregular: acumular por lectura (caso menos común).
            for i in range(m):
                qi = batch.quality_of(i)
                ci = batch[i].sequence.decode()
                c = min(qi.shape[0], _MAXPOS)
                pos_q_sum[:c] += qi[:c]
                pos_count[:c] += 1
                sub = ci[:c]
                for b in range(4):
                    base_counts[b, :c] += (sub == b)
                base_counts[4, :c] += (sub > 3)

    elapsed = time.perf_counter() - t0
    if n_reads == 0:
        raise SequenceValueError(
            f"No se encontraron lecturas en {path!r}. "
            "¿Es un archivo FASTQ (.fastq / .fastq.gz)?"
        )

    eff = int(np.count_nonzero(pos_count))          # nº de posiciones con datos
    with np.errstate(invalid="ignore", divide="ignore"):
        pos_q_mean = np.where(
            pos_count[:eff] > 0, pos_q_sum[:eff] / pos_count[:eff], 0.0)
        col_tot = base_counts[:, :eff].sum(axis=0)
        base_frac = np.where(col_tot > 0, base_counts[:, :eff] / col_tot, 0.0)

    return QCReport(
        path=path, n_reads=n_reads, total_bases=total_bases,
        min_len=int(min_len), max_len=int(max_len),
        gc_overall=(gc_bases / total_bases if total_bases else 0.0),
        mean_q_overall=(q_sum_total / total_bases if total_bases else 0.0),
        pct_q20=100.0 * pass20 / n_reads,
        pct_q30=100.0 * pass30 / n_reads,
        meanq_hist=meanq_hist, gc_hist=gc_hist,
        pos_q_mean=pos_q_mean, base_frac=base_frac,
        elapsed_s=elapsed,
    )


# ══════════════════════════════════════════════════════════════════════════════
# §3  INFORME DE TEXTO
# ══════════════════════════════════════════════════════════════════════════════

_SPARK = " .:-=+*#%@"   # 10 niveles de densidad ASCII


def _sparkline(values: np.ndarray, vmin: float, vmax: float) -> str:
    """Convierte un array en una línea de caracteres de densidad."""
    if values.size == 0:
        return ""
    span = vmax - vmin
    if span <= 0:
        return _SPARK[-1] * values.size
    idx = np.clip(((values - vmin) / span) * (len(_SPARK) - 1), 0,
                  len(_SPARK) - 1).astype(int)
    return "".join(_SPARK[i] for i in idx)


def _hist_bars(hist: np.ndarray, lo: int, hi: int, step: int,
               unit: str, width: int = 40) -> list[str]:
    """Histograma agrupado en cubos [lo, hi] de tamaño step, con barras."""
    edges = list(range(lo, hi + 1, step))
    buckets = []
    for e in edges:
        buckets.append(int(hist[e: e + step].sum()))
    peak = max(buckets) if buckets else 0
    out = []
    for e, c in zip(edges, buckets):
        bar = "█" * int(round(width * c / peak)) if peak else ""
        out.append(f"    {e:>3}-{min(e+step-1, hi):<3} {unit} | {bar} {c:,}")
    return out


def build_report(r: QCReport, width: int = 60) -> str:
    W = 64
    dbl = "═" * W
    sep = "─" * W
    L: list[str] = []

    def add(*t: str) -> None:
        L.extend(t)

    add(dbl, "  INFORME DE CALIDAD FASTQ  (BioForge)", dbl, "")
    add(f"  Archivo        : {Path(r.path).name}")
    add(f"  Lecturas       : {r.n_reads:,}")
    add(f"  Bases totales  : {r.total_bases:,}")
    if r.min_len == r.max_len:
        add(f"  Longitud       : {r.min_len} bp (fija)")
    else:
        add(f"  Longitud       : {r.min_len}–{r.max_len} bp "
            f"(media {r.mean_len:.0f})")
    add(f"  GC global      : {r.gc_overall * 100:.1f}%")
    add(f"  Calidad media  : Q{r.mean_q_overall:.1f}")
    add(f"  Lecturas Q≥20  : {r.pct_q20:.1f}%")
    add(f"  Lecturas Q≥30  : {r.pct_q30:.1f}%")
    add(f"  Procesado en   : {r.elapsed_s * 1000:.0f} ms "
        f"({r.total_bases / r.elapsed_s / 1e6:.0f} M bases/s)", "")

    add(sep, "  CALIDAD MEDIA POR LECTURA", sep)
    add(*_hist_bars(r.meanq_hist, 0, 45, 5, "Q"), "")

    add(sep, "  CONTENIDO GC POR LECTURA", sep)
    add(*_hist_bars(r.gc_hist, 0, 100, 10, "%"), "")

    add(sep, "  CALIDAD MEDIA POR POSICIÓN", sep)
    pos = r.pos_q_mean
    if pos.size:
        # Submuestrear a 'width' columnas para la sparkline.
        if pos.size > width:
            idx = np.linspace(0, pos.size - 1, width).astype(int)
            line = pos[idx]
        else:
            line = pos
        add(f"    pos 1{' ' * (max(0, len(_sparkline(line, 0, 42)) - 6))}{pos.size}")
        add(f"    Q42 |{_sparkline(line, 0, 42)}|")
        add(f"    min Q{pos.min():.0f}  ·  media Q{pos.mean():.0f}  ·  "
            f"max Q{pos.max():.0f}")
        zona = "buena (Q≥28 en toda la lectura)" if pos.min() >= 28 else \
               "cae al final (típico)" if pos[-1] < pos[0] else "irregular"
        add(f"    Diagnóstico: {zona}", "")

    add(sep, "  COMPOSICIÓN POR BASE (global)", sep)
    names = "ACGTN"
    tot = r.base_frac.sum(axis=1)
    tot = tot / tot.sum() if tot.sum() else tot
    for i, name in enumerate(names):
        bar = "█" * int(round(30 * tot[i]))
        add(f"    {name} | {bar} {tot[i] * 100:.1f}%")
    add("", dbl)
    return "\n".join(L)


# ══════════════════════════════════════════════════════════════════════════════
# §4  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="bioforge-qc",
        description="Informe rápido de calidad de un archivo FASTQ (.gz o plano).",
    )
    p.add_argument("fastq", help="Archivo FASTQ (.fastq o .fastq.gz)")
    p.add_argument("--output", "-o", metavar="FILE",
                   help="Guardar el informe en un archivo (si no, a pantalla).")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    args = _parse_args(argv)
    try:
        report = run(args.fastq)
    except FileNotFoundError:
        print(f"Archivo no encontrado: {args.fastq}", file=sys.stderr)
        return 1
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    text = build_report(report)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Informe guardado en: {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
