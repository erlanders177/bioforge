"""
tests/test_nanopore.py — núcleo del basecaller de nanoporo (NumPy puro).

Prueban que el ALGORITMO es correcto sobre señal cuya verdad conocemos (simulada
desde un pore model conocido). NO afirman precisión sobre señal real — esa cifra
exige datos reales de Oxford Nanopore y se medirá aparte antes de tocar v9.0.
"""

import numpy as np
import pytest

from bioforge.nanopore import (
    SignalRead,
    detect_events,
    normalize_signal,
    random_pore_model,
    simulate_signal,
)


# ── normalización ─────────────────────────────────────────────────────────────

def test_normaliza_centra_en_cero():
    rng = np.random.default_rng(0)
    x = rng.normal(100.0, 5.0, 5000)          # corriente cruda desplazada y escalada
    z = normalize_signal(x)
    assert abs(np.median(z)) < 1e-9           # mediana → 0
    assert 0.8 < np.median(np.abs(z)) * 1.4826 < 1.2   # MAD ≈ 1


def test_normaliza_senal_plana_no_revienta():
    z = normalize_signal(np.full(100, 7.0))   # MAD = 0 → no dividir por cero
    assert np.all(np.isfinite(z))


def test_normaliza_vacia():
    assert normalize_signal(np.array([])).size == 0


# ── detección de eventos ──────────────────────────────────────────────────────

def test_detecta_escalones_conocidos():
    # tres niveles claros de 40 muestras cada uno → 3 eventos
    sig = np.concatenate([np.full(40, -2.0), np.full(40, 0.0), np.full(40, 2.0)])
    ev = detect_events(sig, threshold=0.5)
    assert len(ev) == 3
    assert np.allclose(np.sort(ev.means), [-2.0, 0.0, 2.0], atol=1e-6)
    # los eventos cubren la señal entera, sin huecos ni solapes
    assert ev.starts[0] == 0
    assert int(ev.starts[-1] + ev.lengths[-1]) == sig.size


def test_eventos_cubren_toda_la_senal():
    rng = np.random.default_rng(1)
    sig = normalize_signal(rng.normal(0, 1, 500))
    ev = detect_events(sig)
    assert ev.starts[0] == 0
    reconstruido = int(ev.starts[-1] + ev.lengths[-1])
    assert reconstruido == sig.size
    assert np.all(ev.lengths > 0)


def test_senal_muy_corta_un_evento():
    ev = detect_events(np.array([1.0, 2.0]))
    assert len(ev) == 1


# ── simulador honesto ─────────────────────────────────────────────────────────

def test_simula_longitud_coherente():
    pm = random_pore_model(3, seed=0)
    read = simulate_signal("ACGTACGTACGT", pm, k=3, dwell=8, noise=0.0, seed=1)
    assert isinstance(read, SignalRead)
    # 12 bases, k=3 → 10 k-meros; sin ruido, la señal son 10 niveles repetidos
    assert read.n_samples > 10
    niveles_unicos = np.unique(np.round(read.signal, 6))
    assert niveles_unicos.size <= 10          # a lo sumo un nivel por k-mero


def test_simulador_exige_k_bases():
    pm = random_pore_model(5, seed=0)
    with pytest.raises(ValueError):
        simulate_signal("AC", pm, k=5)


def test_eventos_recuperan_pasos_distinguibles():
    """De punta a punta con niveles CONTROLADOS y separados.

    Con un pore model aleatorio, k-meros vecinos pueden tener corrientes casi
    iguales y entonces NO hay frontera detectable (degeneración física, no un fallo).
    Para probar el detector de forma justa, forzamos niveles bien separados: una
    secuencia alterna dos di-meros (AC/CA) puestos a -2 y +2 → cada paso es un salto
    real, y el detector debe encontrarlos casi todos."""
    pm = np.zeros(16)                          # k=2 → 16 di-meros
    pm[_di("AC")] = -2.0
    pm[_di("CA")] = +2.0
    seq = "ACACACACACAC"                       # 12 bases → 11 pasos alternos
    read = simulate_signal(seq, pm, k=2, dwell=10, noise=0.05, seed=3)
    z = normalize_signal(read.signal)
    ev = detect_events(z, threshold=1.0, min_length=3)
    n_pasos = len(seq) - 2 + 1                  # 11 transiciones, todas distinguibles
    assert abs(len(ev) - n_pasos) <= 2
    # y los niveles recuperados son los dos que metimos (tras normalizar, ±algo)
    assert np.ptp(ev.means) > 1.0              # hay contraste alto/bajo claro


def _di(dimer: str) -> int:
    """Índice base-4 de un di-mero (A=0 C=1 G=2 T=3), para armar pore models de test."""
    code = {"A": 0, "C": 1, "G": 2, "T": 3}
    return code[dimer[0]] * 4 + code[dimer[1]]
