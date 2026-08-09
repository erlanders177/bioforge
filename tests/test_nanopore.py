"""
tests/test_nanopore.py — núcleo del basecaller de nanoporo (NumPy puro).

Prueban que el ALGORITMO es correcto sobre señal cuya verdad conocemos (simulada
desde un pore model conocido). NO afirman precisión sobre señal real — esa cifra
exige datos reales de Oxford Nanopore y se medirá aparte antes de tocar v9.0.
"""

import sys

import numpy as np
import pytest

from bioforge.nanopore import (
    SignalRead,
    detect_events,
    estimate_pore_model,
    kmer_indices,
    kmer_levels,
    normalize_signal,
    random_pore_model,
    simulate_signal,
    viterbi_decode,
)


def _identity(a: str, b: str) -> float:
    """Fracción de bases coincidentes en la misma posición (0..1)."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(x == y for x, y in zip(a[:n], b[:n])) / max(len(a), len(b))


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


# ── Viterbi: el decodificador (matemáticas, no IA) ────────────────────────────

def test_viterbi_recupera_secuencia_sin_ruido():
    """Con niveles ideales (sin ruido) debe recuperar la secuencia EXACTA."""
    k = 3
    pm = random_pore_model(k, seed=5)
    seq = "ACGTAGCTAGCATCGATCGTACGATCGATG"    # sin homopolímeros largos
    levels = kmer_levels(seq, pm, k)           # la 'verdad' de nivel
    out = viterbi_decode(levels, pm, k, sigma=0.5)
    assert len(out) == len(seq)                # T eventos → T+k-1 bases
    assert _identity(out, seq) == 1.0          # reconstrucción perfecta


def test_viterbi_longitud_correcta():
    k = 3
    pm = random_pore_model(k, seed=6)
    levels = kmer_levels("ACGTACGTACGT", pm, k)   # 12 bases → 10 niveles
    out = viterbi_decode(levels, pm, k)
    assert len(out) == 10 + k - 1                 # eventos + k - 1

def test_viterbi_aguanta_ruido_moderado():
    """Con ruido moderado en los niveles, la mayoría de bases siguen bien."""
    k = 3
    pm = random_pore_model(k, seed=7)
    seq = "ACGTAGCTAGCATCGATCGTACGATCGATGCATGCAT"
    levels = kmer_levels(seq, pm, k)
    rng = np.random.default_rng(0)
    ruidoso = levels + rng.normal(0, 0.25, levels.size)
    out = viterbi_decode(ruidoso, pm, k, sigma=0.4)
    assert _identity(out, seq) > 0.85          # el contexto corrige casi todo

def test_viterbi_valida_tamano_del_pore_model():
    with pytest.raises(ValueError):
        viterbi_decode(np.zeros(5), random_pore_model(3), k=2)   # 4**2≠4**3

def test_viterbi_vacio():
    assert viterbi_decode(np.array([]), random_pore_model(3), k=3) == ""


# ── Estimar nuestro propio pore model + el CÍRCULO COMPLETO ───────────────────

def test_estima_el_pore_model_desde_etiquetas():
    """Recupera la tabla del fabricante desde datos etiquetados (con ruido)."""
    k = 3
    verdad = random_pore_model(k, seed=8)
    seq = "ACGTAGCTAGCATCGATCGTACGATCGATG" * 3          # cubre muchos k-meros
    idx = kmer_indices(seq, k)
    rng = np.random.default_rng(0)
    niveles = verdad[idx] + rng.normal(0, 0.1, idx.size)  # 'medidos' con ruido
    est = estimate_pore_model(niveles, idx, k)
    vistos = np.bincount(idx, minlength=4 ** k) > 0
    assert np.allclose(est[vistos], verdad[vistos], atol=0.15)   # recupera el modelo


def test_kmeros_no_vistos_no_rompen():
    k = 3
    idx = kmer_indices("ACGTACGT", k)               # solo unos pocos k-meros
    est = estimate_pore_model(np.zeros(idx.size), idx, k)
    assert est.shape[0] == 4 ** k and np.all(np.isfinite(est))   # sin NaN ni huecos


def test_circulo_completo_estimar_y_decodificar():
    """El pipeline autocontenido en su forma más pura: sin conocer la tabla del
    fabricante, la ESTIMAMOS de datos etiquetados y luego decodificamos señal nueva.
    Sobre datos sintéticos con verdad conocida, debe cerrar el círculo con acierto
    alto. (Sobre señal REAL el número será menor: eso se mide aparte.)"""
    k = 3
    verdad = random_pore_model(k, seed=9)
    # 1) ENTRENAR: estimar el modelo de una secuencia etiquetada
    train = "ACGTAGCTAGCATCGATCGTACGATCGATGCATGCATTAGC" * 2
    tidx = kmer_indices(train, k)
    rng = np.random.default_rng(1)
    tlev = verdad[tidx] + rng.normal(0, 0.15, tidx.size)
    modelo = estimate_pore_model(tlev, tidx, k)
    # 2) DECODIFICAR: una secuencia NUEVA, con el modelo estimado
    test = "TAGCATCGATCGTACGATCGATGCATGGCTAGCATCGA"
    tl = verdad[kmer_indices(test, k)] + rng.normal(0, 0.15, len(test) - k + 1)
    out = viterbi_decode(tl, modelo, k, sigma=0.4)
    assert _identity(out, test) > 0.9              # el círculo se cierra en sintético


# ── Lector POD5 (dependencia opcional 'pod5') ─────────────────────────────────

def test_read_pod5_sin_libreria_mensaje_claro(monkeypatch):
    """Sin la librería opcional, debe fallar con un mensaje accionable, no críptico."""
    import bioforge.nanopore as nano
    monkeypatch.setitem(sys.modules, "pod5", None)      # simula 'pod5' ausente
    with pytest.raises(ImportError, match="bioforge\\[nanopore\\]"):
        list(nano.read_pod5("cualquiera.pod5"))


def test_read_pod5_ida_y_vuelta(tmp_path):
    """Con la librería presente: escribe un POD5 mínimo y compruébalo con NUESTRO
    lector (se salta si 'pod5' no está instalado — es un extra opcional)."""
    pod5 = pytest.importorskip("pod5")
    import datetime
    import uuid

    from pod5.pod5_types import (
        Calibration,
        EndReason,
        EndReasonEnum,
        Pore,
        Read,
        RunInfo,
    )
    from bioforge.nanopore import read_pod5

    sig = (np.sin(np.arange(1500) / 20) * 100 + 300).astype(np.int16)
    t0 = datetime.datetime(2020, 1, 1)
    run = RunInfo(acquisition_id="a", acquisition_start_time=t0, adc_max=4095,
                  adc_min=-4096, context_tags={}, experiment_name="e",
                  flow_cell_id="f", flow_cell_product_code="p", protocol_name="n",
                  protocol_run_id="r", protocol_start_time=t0, sample_id="s",
                  sample_rate=4000, sequencing_kit="k", sequencer_position="x",
                  sequencer_position_type="t", software="w", system_name="sn",
                  system_type="st", tracking_id={})
    rd = Read(read_id=uuid.uuid4(), pore=Pore(channel=1, well=1, pore_type="p"),
              calibration=Calibration(offset=3.0, scale=0.2), read_number=1,
              start_sample=0, median_before=100.0,
              end_reason=EndReason(reason=EndReasonEnum.SIGNAL_POSITIVE, forced=False),
              run_info=run, signal=sig)
    p = tmp_path / "mini.pod5"
    with pod5.Writer(str(p)) as w:
        w.add_read(rd)

    got = list(read_pod5(str(p)))
    assert len(got) == 1
    r0 = got[0]
    assert r0.n_samples == sig.size
    assert np.array_equal(np.asarray(r0.signal), sig)   # señal intacta
    assert r0.sample_rate == 4000.0
    # calibración preservada (POD5 guarda scale en float32 → comparación aproximada)
    assert r0.scale == pytest.approx(0.2, abs=1e-6)
    assert r0.offset == pytest.approx(3.0, abs=1e-6)
