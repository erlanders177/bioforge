"""
viability.py — eje B del predictor: VIABILIDAD / gramaticalidad con ESM-2.

Un modelo de lenguaje de proteínas (ESM-2), entrenado con millones de secuencias, sabe
qué aminoácido es "natural" en cada contexto. Dos usos:
  - `grammaticality_profile`: cuán natural ve el modelo cada posición (interpretabilidad).
  - `viability_scores`: para una mutación candidata, P(residuo | contexto) ∈ [0, 1]
    — el eje B que se enchufa en `predict_fusion(..., viability=...)`.

Modelo por defecto: **ESM-2 8M** (`esm2_t6_8M_UR50D`) — el más ligero, corre en CPU
(la evidencia de 2025: 8M destilado llega a 88% AUC → bajo recurso ≠ baja precisión).
Se carga UNA vez y se cachea. Extra opcional: `pip install bioforge[ai]`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Sequence

import numpy as np

from ..biocore import EngineError, SequenceValueError

_DEFAULT_MODEL = "facebook/esm2_t6_8M_UR50D"


@lru_cache(maxsize=2)
def _load(model_name: str):
    """Carga (perezosa y cacheada) el tokenizer + modelo ESM-2. Degrada con gracia."""
    try:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError as e:                     # sin el extra 'ai'
        raise EngineError(
            "el eje B (viabilidad con ESM-2) requiere el extra opcional: "
            "'pip install bioforge[ai]' (torch + transformers).") from e
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()
    return tok, model, torch


def _probs(sequence: str, model_name: str) -> tuple[np.ndarray, object]:
    """Matriz (T, vocab) de probabilidades por posición para ``sequence`` (wildtype
    marginal). Devuelve (probs, tokenizer) — el índice de secuencia i va en la fila i+1
    (ESM antepone <cls>)."""
    tok, model, torch = _load(model_name)
    seq = sequence.upper().replace("-", "")
    if not seq:
        raise SequenceValueError("secuencia vacía para ESM-2.")
    with torch.no_grad():
        enc = tok(seq, return_tensors="pt")
        logits = model(**enc).logits[0]          # (T, vocab)
        probs = logits.softmax(dim=-1).cpu().numpy()
    return probs, tok


def viability_scores(sequence: str, changes: dict[int, str], *,
                     model_name: str = _DEFAULT_MODEL) -> dict[int, float]:
    """Viabilidad de cada mutación candidata según ESM-2, en [0, 1].

    ``sequence`` : proteína de referencia (contexto), sin huecos.
    ``changes``  : {posición_0based → aminoácido_candidato}.
    Devuelve {posición → P(aminoácido | contexto)} — alto = el modelo lo ve viable.
    Se enchufa directamente en ``predict_fusion(..., viability=...)`` (eje B).
    """
    probs, tok = _probs(sequence, model_name)
    L = probs.shape[0] - 2                        # descuenta <cls>/<eos>
    out: dict[int, float] = {}
    for pos, res in changes.items():
        if not (0 <= pos < L):
            raise SequenceValueError(
                f"posición {pos} fuera de rango (proteína de {L} residuos).")
        tid = tok.convert_tokens_to_ids(res.upper())
        out[pos] = float(probs[pos + 1, tid])     # +1 por el <cls> inicial
    return out


def viability_matrix(sequence: str, alphabet: Sequence[str], *,
                     model_name: str = _DEFAULT_MODEL) -> np.ndarray:
    """Matriz (len(alphabet), len(sequence)) con P(aminoácido | contexto) según ESM-2.

    Todas las mutaciones posibles de un tirón, en **una sola pasada** del modelo (los
    logits ya traen la distribución completa por posición: pedirlas de una en una sería
    tirar el 95% del cómputo). Es el eje B para ``rank_mutations(viability=...)``, que
    ordena MUTACIONES y por tanto necesita el alfabeto entero en cada sitio.

    Alto = ESM-2 considera ese residuo verosímil ahí = la proteína probablemente lo
    tolera. Es la medida buena de lo que ``conservation`` aproxima con dos tablas de
    aminoácidos.
    """
    probs, tok = _probs(sequence, model_name)
    L = probs.shape[0] - 2                        # descuenta <cls>/<eos>
    ids = [tok.convert_tokens_to_ids(a.upper()) for a in alphabet]
    return probs[1:L + 1, ids].T.astype(np.float64)      # (S, L)


def grammaticality_profile(sequence: str, *,
                           model_name: str = _DEFAULT_MODEL) -> np.ndarray:
    """Gramaticalidad por posición: P(residuo_presente | contexto) para cada sitio.

    Vector (L,) en [0, 1]. Bajo = el modelo ve esa posición "rara" (candidata a cambio
    o error); alto = muy conservada/natural. Útil para interpretar y priorizar."""
    probs, tok = _probs(sequence, model_name)
    seq = sequence.upper().replace("-", "")
    ids = np.array([tok.convert_tokens_to_ids(c) for c in seq])
    rows = np.arange(1, len(seq) + 1)             # +1 por el <cls>
    return probs[rows, ids]
