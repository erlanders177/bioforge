"""
bioforge/nanopore.py — señal de nanoporo → bases, DESDE CERO (v9.0, en construcción).

Los aparatos tipo Oxford Nanopore (MinION…) hacen pasar el ADN por un poro y miden
la CORRIENTE IÓNICA que lo atraviesa. El resultado no es una onda periódica: es una
señal de NIVELES (una escalera). En la parte estrecha del poro caben ~k bases a la
vez (un k-mero), así que cada escalón de corriente depende de k bases juntas, no de
una. Eso hace el problema inverso (corriente → bases) AMBIGUO por física —muchos
k-meros dan corrientes parecidas— y por eso no existe "una fórmula por base": se
resuelve con DECODIFICACIÓN estadística (Viterbi), no con álgebra.

Este módulo lo construye por la vía CLÁSICA (pre-IA), fiel a la filosofía BioForge:
NumPy puro, sin torch, ligero, y —cuando toque— con motor C opcional. Escalera:

  Nivel 2  leer la señal cruda           (POD5/FAST5 → array)  [dep opcional]
  Nivel 3  normalizar + detectar eventos (aquí, NumPy puro)
  Nivel 4  pore model + Viterbi → bases  (aquí, NumPy puro)

Backend enchufable: si el usuario tiene Dorado/Guppy (el basecaller oficial, de red
neuronal, ~99%), se usa ESE (máxima calidad). El nuestro es el respaldo honesto que
corre sin GPU y sin instalar nada — con un techo de precisión menor, medido sin
trampa contra una referencia conocida.

⚠️ NADA aquí afirma precisión todavía: el simulador de este módulo sirve para probar
que el ALGORITMO es correcto (recupera lo que metió un pore model conocido), NO para
declarar cuánto acierta en señal real. Esa cifra sale solo con datos reales de
Oxford Nanopore + su referencia, y es la que dará luz verde a v9.0.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

__all__ = [
    "SignalRead",
    "EventTable",
    "read_pod5",
    "read_fast5",
    "normalize_signal",
    "detect_events",
    "simulate_signal",
    "random_pore_model",
    "kmer_levels",
    "kmer_indices",
    "estimate_pore_model",
    "viterbi_decode",
    "viterbi_basecall",
    "basecall",
]

# Bases canónicas ↔ código 0..3 (mismo criterio que el resto del motor: A C G T).
_BASES = "ACGT"
_BASE_CODE = {b: i for i, b in enumerate(_BASES)}


class SignalRead(NamedTuple):
    """Una lectura de señal cruda de nanoporo.

    signal      : corriente cruda (1-D). int16 tal como sale del aparato, o pA.
    read_id     : identificador de la lectura (UUID en POD5/FAST5).
    sample_rate : muestras por segundo (Hz). Para convertir índices a tiempo.
    offset,scale: calibración a picoamperios: pA = scale * (signal + offset).
                  Si el archivo no la trae, quedan en 0.0 / 1.0 (señal ya en pA).
    """
    signal: np.ndarray
    read_id: str = ""
    sample_rate: float = 4000.0
    offset: float = 0.0
    scale: float = 1.0

    @property
    def n_samples(self) -> int:
        return int(self.signal.size)

    def to_picoamperes(self) -> np.ndarray:
        """Corriente en pA aplicando la calibración del aparato (vectorizado)."""
        return self.scale * (self.signal.astype(np.float64) + self.offset)


def read_pod5(path: str):
    """Lee un archivo POD5 (formato moderno de Oxford Nanopore) → iterador de
    ``SignalRead``, en streaming (una lectura cada vez, RAM acotada).

    Leer POD5 es FONTANERÍA (un contenedor Apache Arrow), no ciencia: reimplementarlo
    a mano no aportaría nada y sería un pozo sin fondo. Por eso se apoya en la librería
    oficial ``pod5`` como DEPENDENCIA OPCIONAL —igual que el eje ESM-2 con torch—. El
    núcleo de BioForge sigue siendo solo-NumPy; esto solo se activa si el usuario va a
    tocar señal cruda e instala el extra:  ``pip install "bioforge[nanopore]"``.

    Falla con un mensaje accionable si la librería no está, en vez de un ImportError
    críptico. Toda la CIENCIA de después (normalizar, eventos, pore model, Viterbi) es
    nuestra y en NumPy puro.
    """
    try:
        import pod5
    except ImportError as exc:                       # dep opcional ausente
        raise ImportError(
            "leer POD5 necesita la librería 'pod5' (fontanería del formato). "
            "Instala el extra opcional:  pip install \"bioforge[nanopore]\"") from exc

    with pod5.Reader(str(path)) as reader:
        for rd in reader.reads():                    # bucle por LECTURA, no por muestra
            cal = rd.calibration
            yield SignalRead(
                signal=np.asarray(rd.signal),        # corriente cruda int16
                read_id=str(rd.read_id),
                sample_rate=float(rd.run_info.sample_rate),
                offset=float(cal.offset),            # pA = scale·(señal + offset)
                scale=float(cal.scale),
            )


def read_fast5(path: str):
    """Lee un archivo FAST5 (formato antiguo de Oxford Nanopore) → ``SignalRead``.

    FAST5 es HDF5; se apoya en ``h5py`` como DEPENDENCIA OPCIONAL (``bioforge[nanopore]``),
    igual que POD5 en ``pod5`` — es fontanería del formato, no ciencia. Maneja los dos
    sabores: single-read (un read por archivo, `/Raw/Reads/Read_N`) y multi-read
    (`/read_<uuid>`). Calibración a pA: ``pA = (señal + offset)·range/digitisation``.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "leer FAST5 necesita la librería 'h5py' (fontanería del formato). "
            "Instala el extra opcional:  pip install \"bioforge[nanopore]\"") from exc

    def _emit(sig, rid, ch):
        digit = float(ch["digitisation"]); rng = float(ch["range"])
        return SignalRead(signal=np.asarray(sig), read_id=str(rid),
                          sample_rate=float(ch["sampling_rate"]),
                          offset=float(ch["offset"]), scale=rng / digit)

    with h5py.File(str(path), "r") as f:
        if "Raw" in f:                                   # single-read (antiguo)
            ch = f["UniqueGlobalKey/channel_id"].attrs
            for rk in f["Raw/Reads"]:                    # normalmente uno
                g = f["Raw/Reads/" + rk]
                rid = g.attrs["read_id"]
                yield _emit(g["Signal"][:], rid.decode() if isinstance(rid, bytes)
                            else rid, ch)
        else:                                            # multi-read
            for name in f:                               # bucle por read (no por muestra)
                if not name.startswith("read_"):
                    continue
                g = f[name]
                ch = g["channel_id"].attrs
                rid = name[5:]
                yield _emit(g["Raw/Signal"][:], rid, ch)


class EventTable(NamedTuple):
    """Eventos detectados: tramos de corriente aproximadamente constante.

    Cada evento es un escalón de la señal (idealmente, un k-mero en el poro). Todo
    en arrays paralelos —columnar, sin objetos por evento— para que el trabajo por
    símbolo siga siendo NumPy sobre el lote, no un bucle Python.

    starts  : (n,) índice de muestra donde empieza cada evento.
    lengths : (n,) nº de muestras del evento.
    means   : (n,) corriente media (señal NORMALIZADA) del evento.
    stdvs   : (n,) desviación típica dentro del evento.
    """
    starts: np.ndarray
    lengths: np.ndarray
    means: np.ndarray
    stdvs: np.ndarray

    def __len__(self) -> int:
        return int(self.means.size)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Normalización robusta (mediana / MAD) — el estándar en nanoporo.

    Cada poro y cada lectura tienen una escala y un offset distintos; sin normalizar,
    los niveles de corriente no son comparables entre lecturas ni con el pore model.
    Se usa mediana y MAD (no media y desviación) porque son robustas a los picos y
    bloqueos de la señal. MAD·1.4826 ≈ σ para ruido gaussiano.

    Todo vectorizado; O(n log n) por las dos medianas.
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    if x.size == 0:
        return x
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = mad * 1.4826
    if scale < 1e-9:                          # señal plana: evita dividir por ~0
        scale = 1.0
    return (x - med) / scale


def detect_events(signal: np.ndarray, *, window: int = 3,
                  threshold: float = 1.0, min_length: int = 2) -> EventTable:
    """Segmenta la señal NORMALIZADA en eventos (escalones de corriente).

    Método clásico del campo: un estadístico tipo t deslizante que compara la media
    de la ventana de delante con la de detrás en cada punto; donde ese salto supera
    ``threshold`` hay una FRONTERA (un cambio de k-mero en el poro). Es la misma idea
    que usaban los primeros basecallers de MinION.

    El estadístico se calcula vectorizado con sumas acumuladas (O(n)). La elección de
    fronteras recorre solo los picos candidatos (no muestra a muestra) —análogo a los
    bucles por registro permitidos—, y la agregación por evento vuelve a ser NumPy.

    window     : nº de muestras a cada lado para comparar medias.
    threshold  : salto mínimo (en unidades normalizadas) para marcar frontera.
    min_length : eventos más cortos se funden con el vecino (ruido).
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    n = x.size
    if n < 2 * window + 1:                     # demasiado corta para segmentar
        m = float(x.mean()) if n else 0.0
        s = float(x.std()) if n else 0.0
        return EventTable(np.array([0]), np.array([n]),
                          np.array([m]), np.array([s]))

    # media de ventana deslizante vía suma acumulada (prefix sums)
    csum = np.concatenate([[0.0], np.cumsum(x)])
    # media de [i-window, i)  y  [i, i+window)  para cada frontera candidata i
    i = np.arange(window, n - window + 1)
    before = (csum[i] - csum[i - window]) / window
    after = (csum[i + window] - csum[i]) / window
    tstat = np.abs(after - before)             # salto de nivel en cada i

    # picos locales del salto por encima del umbral = fronteras candidatas
    is_peak = np.zeros(tstat.size, dtype=bool)
    is_peak[1:-1] = (tstat[1:-1] >= tstat[:-2]) & (tstat[1:-1] > tstat[2:]) \
        & (tstat[1:-1] >= threshold)
    cand = i[is_peak]

    # imponer separación mínima entre fronteras (recorre solo picos, no muestras)
    bounds = [0]
    for b in cand.tolist():
        if b - bounds[-1] >= min_length:
            bounds.append(int(b))
    if n - bounds[-1] < min_length and len(bounds) > 1:
        bounds.pop()
    bounds.append(n)
    bounds_arr = np.array(bounds)

    starts = bounds_arr[:-1]
    lengths = np.diff(bounds_arr)
    # media y std por evento con reduceat (vectorizado sobre los tramos)
    means = np.add.reduceat(x, starts) / lengths
    sq = np.add.reduceat(x * x, starts) / lengths
    stdvs = np.sqrt(np.maximum(sq - means * means, 0.0))
    return EventTable(starts, lengths, means, stdvs)


def simulate_signal(sequence: str, pore_model: np.ndarray, k: int, *,
                    dwell: int = 8, noise: float = 0.15,
                    sample_rate: float = 4000.0, seed: int = 0) -> SignalRead:
    """SIMULADOR — genera señal a partir de una secuencia y un pore model CONOCIDO.

    SOLO para probar que el algoritmo es correcto (si el decodificador recupera esta
    secuencia, el Viterbi funciona) y para demos. NO produce señal 'realista' ni sirve
    para afirmar precisión: eso exige señal real. Aquí la verdad es conocida a drede.

    sequence   : cadena ACGT.
    pore_model : (4**k,) corriente media por k-mero (índice base-4, A=0..T=3).
    k          : tamaño del k-mero del poro.
    dwell      : muestras medias por k-mero (la enzima va a tirones → se aleatoriza).
    noise      : desviación del ruido gaussiano añadido a cada muestra.
    """
    codes = np.array([_BASE_CODE[b] for b in sequence.upper() if b in _BASE_CODE])
    if codes.size < k:
        raise ValueError(f"la secuencia debe tener al menos k={k} bases")
    # k-meros deslizantes → índice base-4 (Horner vectorizado)
    win = np.lib.stride_tricks.sliding_window_view(codes, k)
    powers = (4 ** np.arange(k - 1, -1, -1))
    kmer_idx = win @ powers                       # (n_kmers,)
    levels = pore_model[kmer_idx]                 # corriente ideal por paso

    rng = np.random.default_rng(seed)
    # dwell variable por k-mero (Poisson-ish, mínimo 1) → velocidad irregular
    dwells = np.maximum(1, rng.poisson(dwell, size=levels.size))
    clean = np.repeat(levels, dwells)
    signal = clean + rng.normal(0.0, noise, size=clean.size)
    return SignalRead(signal=signal.astype(np.float64),
                      read_id="sim", sample_rate=sample_rate,
                      offset=0.0, scale=1.0)


def random_pore_model(k: int, *, seed: int = 0,
                      low: float = -2.0, high: float = 2.0) -> np.ndarray:
    """Pore model SINTÉTICO (4**k niveles) para pruebas del algoritmo.

    El pore model REAL se estima de señal etiquetada (o lo aporta el fabricante);
    este es solo un modelo conocido para validar el decodificador de punta a punta.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=4 ** k)


def kmer_levels(sequence: str, pore_model: np.ndarray, k: int) -> np.ndarray:
    """Corriente ideal por k-mero de una secuencia (los niveles que 'vería' el poro).

    Útil para validar el decodificador sin el ruido de la detección de eventos: es la
    verdad de nivel que Viterbi debería recuperar. Vectorizado (Horner base-4)."""
    codes = np.array([_BASE_CODE[b] for b in sequence.upper() if b in _BASE_CODE])
    if codes.size < k:
        raise ValueError(f"la secuencia debe tener al menos k={k} bases")
    win = np.lib.stride_tricks.sliding_window_view(codes, k)
    powers = 4 ** np.arange(k - 1, -1, -1)
    return pore_model[win @ powers]


def kmer_indices(sequence: str, k: int) -> np.ndarray:
    """Índices base-4 de los k-meros deslizantes de una secuencia (A=0..T=3)."""
    codes = np.array([_BASE_CODE[b] for b in sequence.upper() if b in _BASE_CODE])
    if codes.size < k:
        raise ValueError(f"la secuencia debe tener al menos k={k} bases")
    win = np.lib.stride_tricks.sliding_window_view(codes, k)
    return win @ (4 ** np.arange(k - 1, -1, -1))


def estimate_pore_model(event_means: np.ndarray, kmer_idx: np.ndarray, k: int, *,
                        min_count: int = 1) -> np.ndarray:
    """Estima el pore model (corriente media por k-mero) desde señal ETIQUETADA.

    Esta es la pieza que nos hace INDEPENDIENTES: en vez de copiar la tabla del
    fabricante, la aprendemos de datos —eventos ya asignados a su k-mero (por una
    alineación a una referencia conocida)—. Y como se estima en el MISMO espacio
    normalizado que la señal, resuelve de raíz el desajuste de escala señal↔modelo.

    Es un promedio por grupo, vectorizado con ``bincount`` (sin bucles por evento):
    para cada k-mero, la media de las corrientes observadas cuando estaba en el poro.
    Los k-meros no vistos (o con < ``min_count`` ejemplos) caen a la media global —
    un respaldo neutro, nunca un hueco que rompa el decodificador.

    event_means : (E,) corriente media de cada evento (señal normalizada).
    kmer_idx    : (E,) índice base-4 del k-mero de cada evento (la etiqueta).
    """
    M = 4 ** k
    m = np.asarray(event_means, dtype=np.float64).ravel()
    idx = np.asarray(kmer_idx, dtype=np.intp).ravel()
    sums = np.bincount(idx, weights=m, minlength=M)
    counts = np.bincount(idx, minlength=M)
    model = np.full(M, m.mean() if m.size else 0.0)     # respaldo: media global
    seen = counts >= min_count
    model[seen] = sums[seen] / counts[seen]
    return model


def viterbi_decode(event_means: np.ndarray, pore_model: np.ndarray, k: int, *,
                   sigma: float = 1.0) -> str:
    """Decodifica una secuencia de niveles de corriente en bases — VITERBI, NumPy puro.

    El estado oculto es el k-mero que hay en el poro (4**k estados). Al avanzar el ADN
    una base, el k-mero se desplaza: sale la base 5' más vieja y entra una nueva por el
    3'. Eso restringe las transiciones —de cada k-mero solo se puede ir a 4 (los que
    comparten k−1 bases)— que es justo lo que hace el problema resoluble pese a la
    ambigüedad de un nivel suelto. La emisión es gaussiana: qué probable es ver la
    corriente observada si en el poro estuviera el k-mero s (media = pore_model[s]).

    Viterbi elige el camino de k-meros de máxima verosimilitud. El bucle es sobre
    EVENTOS (dependencia de datos secuencial, inevitable —como el traceback del
    alineador—); todo el trabajo por estado va vectorizado sobre los 4**k a la vez.

    Modelo v1 "solo-avance": un evento = un paso de k-mero. No modela todavía
    'quedarse' (mismo k-mero varios eventos) ni 'saltar'; por eso los homopolímeros
    (repeticiones) no se pueden contar —limitación FÍSICA, no del código—. Se
    documentará y se medirá con honestidad.

    Devuelve la cadena de bases (longitud = nº de eventos + k − 1).
    Memoria O(T·4**k): mantener k pequeño (real: 5-6) es parte del diseño edge.
    """
    m = np.asarray(event_means, dtype=np.float64).ravel()
    M = 4 ** k
    if pore_model.shape[0] != M:
        raise ValueError(f"pore_model debe tener 4**k = {M} niveles, "
                         f"tiene {pore_model.shape[0]}")
    T = m.size
    if T == 0:
        return ""

    # log-emisión gaussiana (constantes fuera; solo importa el orden) → (T, M)
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    emit = -inv2s2 * (m[:, None] - pore_model[None, :]) ** 2

    # los 4 predecesores de cada estado t: t//4 + j·(M/4), j=0..3 (base que salió)
    step = M // 4
    preds = (np.arange(M) // 4)[:, None] + (np.arange(4) * step)[None, :]   # (M,4)
    ar = np.arange(M)

    V = emit[0].copy()                          # prior uniforme (constante, se ignora)
    back = np.empty((T, M), dtype=np.intp)
    back[0] = ar
    for i in range(1, T):                       # bucle sobre eventos (data-dependencia)
        cand = V[preds]                         # (M,4): mejor prev por predecesor
        best = cand.argmax(axis=1)
        V = emit[i] + cand[ar, best]
        back[i] = preds[ar, best]

    # backtrack del camino de k-meros
    path = np.empty(T, dtype=np.intp)
    path[-1] = int(V.argmax())
    for i in range(T - 1, 0, -1):
        path[i - 1] = back[i, path[i]]

    # k-meros → bases: el primero aporta k bases; cada siguiente, su última base
    first = int(path[0])
    first_bases = [(first // (4 ** (k - 1 - j))) % 4 for j in range(k)]
    rest = (path[1:] % 4).tolist()
    codes = first_bases + rest
    return "".join(_BASES[c] for c in codes)


def basecall(signal, pore_model: np.ndarray, k: int, *,
             sigma: float = 0.25, event_threshold: float = 0.12,
             min_event_len: int = 2, refit: bool = True, p_stay: float = 0.50,
             p_step: float = 0.45, p_skip: float = 0.05) -> str:
    """Señal cruda → bases, de una vez: normaliza → SOBRE-segmenta → Viterbi stay/skip.

    El entry point de alto nivel del basecaller clásico. ``signal`` puede ser un array
    de corriente o un ``SignalRead``. Los valores por defecto salieron de barrer sobre
    señal R9.4 REAL capturada (E. coli): sobre-segmentar y dejar que los STAY reabsorban.

    Escala por-read por MOMENTOS: lleva las medias de evento Y el pore model cada uno a
    media 0 / desviación 1. Esto —igualar la distribución de NIVELES, no la del raw—
    fue clave: en señal real capturada subió el acierto de ~53% (median/MAD del raw) a
    ~70%. Cada poro/lectura tiene su escala; ajustar por-read es lo que hacen los pro.

    ``refit`` (v9.1): un segundo pase de REFINADO. El primer basecall dice qué k-mero
    toca cada evento; con eso se reajusta la escala a los niveles REALES de esos k-meros
    (no a la distribución global del modelo), corrigiendo el sesgo de composición del
    read. Junto con ``p_stay=0.5`` (el 0.42 penalizaba de más los STAY de la
    sobre-segmentación) subió el acierto en señal R9.4 real de ~70% a ~75%.

    ``pore_model`` es la tabla k-mero→corriente (en cualquier escala; se normaliza aquí).
    Úsala estimada por nosotros (``estimate_pore_model``) o la oficial de ONT. Recuerda:
    es la vía CLÁSICA (R9), no compite en precisión con Dorado; su valor es correr sin
    IA, sin GPU y sin instalar nada.
    """
    sig = signal.signal if isinstance(signal, SignalRead) else signal
    sig = np.asarray(sig, dtype=np.float64).ravel()
    if sig.size == 0:
        return ""
    z = normalize_signal(sig)
    ev = detect_events(z, threshold=event_threshold, min_length=min_event_len)
    # escala por-read por momentos: medias de evento y modelo a media 0 / desv 1
    em_z = (ev.means - ev.means.mean()) / (ev.means.std() or 1.0)
    model_z = (pore_model - pore_model.mean()) / (pore_model.std() or 1.0)
    kw = dict(sigma=sigma, p_stay=p_stay, p_step=p_step, p_skip=p_skip)

    # el refit necesita datos finitos (señal corrupta con inf/nan → sáltalo, no revientes)
    if refit and em_z.size > 20 and np.all(np.isfinite(em_z)):
        # 1er pase → k-mero por evento; reajusta la escala a ESOS niveles (mínimos
        # cuadrados robustos, descartando el 20% de eventos con mayor residuo).
        _, path = viterbi_basecall(em_z, model_z, k, return_path=True, **kw)
        exp = model_z[path]
        a, b = np.polyfit(exp, em_z, 1)
        keep = np.abs(em_z - (a * exp + b)) <= np.quantile(np.abs(em_z - (a * exp + b)), 0.8)
        a, b = np.polyfit(exp[keep], em_z[keep], 1)
        if abs(a) > 1e-6:
            em_z = (em_z - b) / a

    return viterbi_basecall(em_z, model_z, k, **kw)


def viterbi_basecall(event_means: np.ndarray, pore_model: np.ndarray, k: int, *,
                     sigma=1.0, p_stay: float = 0.35, p_step: float = 0.60,
                     p_skip: float = 0.05, return_path: bool = False):
    """Basecalling robusto con estados STAY / STEP / SKIP — el HMM clásico completo.

    ``viterbi_decode`` asume un evento = un paso de k-mero, así que cualquier error de
    segmentación (fundir o partir un escalón) rompe el resultado — era nuestro cuello
    de botella (~65%). Aquí el modelo de transición absorbe esos errores, que es como
    lo resolvían los basecallers clásicos de verdad:

      STAY  el mismo k-mero emite VARIOS eventos (el ADN se demora)  → 0 bases nuevas
      STEP  avanza una base (caso normal)                            → 1 base nueva
      SKIP  la enzima saltó un k-mero (translocación rápida)         → 2 bases nuevas

    Con esto conviene SOBRE-segmentar (más eventos que k-meros): los STAY reabsorben
    el exceso, que es más seguro que quedarse corto. Sigue siendo Viterbi (matemáticas,
    no IA): bucle sobre eventos, trabajo por estado vectorizado sobre los 4**k.

    Memoria O(T·4**k) para el backtrack (dos matrices). Mantener k moderado (real 6)
    y sobre-segmentar con mesura tiene sentido en un portátil.
    """
    m = np.asarray(event_means, dtype=np.float64).ravel()
    M = 4 ** k
    if pore_model.shape[0] != M:
        raise ValueError(f"pore_model debe tener 4**k = {M} niveles")
    if k < 2:
        raise ValueError("viterbi_basecall requiere k >= 2 (para SKIP)")
    T = m.size
    if T == 0:
        return ("", np.empty(0, dtype=np.intp)) if return_path else ""

    # log-emisión gaussiana. sigma puede ser ESCALAR o un array por k-mero (4**k):
    # con ruido por-k-mero (la columna level_stdv del pore model) el término −log σ
    # deja de ser constante y hay que incluirlo para comparar estados de forma justa.
    sig = np.asarray(sigma, dtype=np.float64)
    if sig.ndim == 0:
        emit = -(m[:, None] - pore_model[None, :]) ** 2 / (2.0 * sig * sig)
    else:
        emit = (-((m[:, None] - pore_model[None, :]) / sig[None, :]) ** 2 / 2.0
                - np.log(sig)[None, :])
    ar = np.arange(M)
    step_pred = (ar // 4)[:, None] + (np.arange(4) * (M // 4))[None, :]     # (M,4)
    skip_pred = (ar // 16)[:, None] + (np.arange(16) * (M // 16))[None, :]  # (M,16)
    l_stay, l_step, l_skip = np.log(p_stay), np.log(p_step), np.log(p_skip)

    V = emit[0].copy()
    back = np.empty((T, M), dtype=np.intp)
    move = np.empty((T, M), dtype=np.int8)                         # 0 stay,1 step,2 skip
    back[0] = ar
    move[0] = 1
    for i in range(1, T):                                          # bucle sobre eventos
        sc = V[step_pred] + l_step                                 # (M,4)
        sbest = sc.argmax(axis=1)
        step_v = sc[ar, sbest]
        kc = V[skip_pred] + l_skip                                 # (M,16)
        kbest = kc.argmax(axis=1)
        skip_v = kc[ar, kbest]
        stay_v = V + l_stay
        cand = np.stack([stay_v, step_v, skip_v])                  # (3,M)
        mv = cand.argmax(axis=0)
        V = emit[i] + cand[mv, ar]
        pred = np.where(mv == 0, ar,
                        np.where(mv == 1, step_pred[ar, sbest], skip_pred[ar, kbest]))
        back[i] = pred
        move[i] = mv

    # backtrack: camino de k-meros + tipo de movimiento en cada paso
    path = np.empty(T, dtype=np.intp)
    moves = np.empty(T, dtype=np.int8)
    path[-1] = int(V.argmax())
    for i in range(T - 1, 0, -1):
        moves[i] = move[i, path[i]]
        path[i - 1] = back[i, path[i]]

    # reconstruir bases según el movimiento (stay=0, step=1, skip=2 bases nuevas)
    first = int(path[0])
    codes = [(first // (4 ** (k - 1 - j))) % 4 for j in range(k)]
    for i in range(1, T):
        s = int(path[i])
        if moves[i] == 1:
            codes.append(s % 4)
        elif moves[i] == 2:
            codes.append((s // 4) % 4)
            codes.append(s % 4)
    bases = "".join(_BASES[c] for c in codes)
    if return_path:
        return bases, path            # path[i] = k-mero asignado al evento i
    return bases
