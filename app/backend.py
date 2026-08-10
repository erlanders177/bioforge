"""
app/backend.py — el PUENTE entre la interfaz web y el motor BioForge.

La app de escritorio es "la otra cara" del mismo motor: una capa FINA de interfaz
sobre el paquete ``bioforge`` (que ya está probado, 525 tests). Aquí vive la lógica
que la interfaz (HTML/JS) invoca; la ventana en sí (PyWebview) es solo un lanzador.

Todo es LOCAL y SIN SERVIDOR: los datos —el ADN del usuario— nunca salen de la
máquina. Es la promesa "ADN Edge".

Diseño clave: cada método
  · recibe tipos simples (str/int) y devuelve DICCIONARIOS serializables a JSON,
  · NUNCA lanza hacia la interfaz: cualquier fallo se captura y se devuelve como
    ``{"error": "..."}`` para que la app lo muestre con elegancia, no que reviente.
Así el backend se prueba entero SIN necesidad de abrir ninguna ventana.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from bioforge import (
    BioForgeError,
    SeqType,
    SequenceAligner,
    SmartImporter,
    SmartTranslator,
    compute_stats,
)


def _guard(fn: Callable) -> Callable:
    """Envuelve un método para que un error del motor viaje como dato, no como crash."""
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except BioForgeError as e:
            return {"error": f"{type(e).__name__}: {e}"}
        except Exception as e:                       # noqa: BLE001 — la interfaz no debe caer
            return {"error": f"error inesperado: {type(e).__name__}: {e}"}
    wrapper.__name__ = fn.__name__
    return wrapper


class Api:
    """API que PyWebview expone a la interfaz como ``window.pywebview.api``.

    Mantiene el ESTADO de la sesión (el archivo cargado y sus secuencias) entre
    llamadas, para que la interfaz no tenga que reenviar los datos cada vez.
    """

    def __init__(self) -> None:
        self.records: list = []          # PackedSequence cargadas
        self.filename: str = ""

    # ── carga de archivos ────────────────────────────────────────────────────
    @_guard
    def open_file(self, path: str) -> dict[str, Any]:
        """Carga un FASTA (y FASTQ básico) y devuelve un resumen del conjunto."""
        if not path or not os.path.exists(path):
            return {"error": "no existe el archivo"}
        records = list(SmartImporter.from_file(path))
        if not records:
            return {"error": "el archivo no contiene secuencias legibles"}
        self.records = records
        self.filename = os.path.basename(path)
        return self.summary()

    @_guard
    def summary(self) -> dict[str, Any]:
        """Resumen del conjunto cargado: cuántas, tipos, longitudes."""
        if not self.records:
            return {"loaded": False}
        lengths = [r.n_symbols for r in self.records]
        n_nuc = sum(1 for r in self.records if r.seq_type == SeqType.NUCLEOTIDE)
        return {
            "loaded": True,
            "filename": self.filename,
            "count": len(self.records),
            "nucleotide": n_nuc,
            "protein": len(self.records) - n_nuc,
            "total_symbols": int(sum(lengths)),
            "min_len": int(min(lengths)),
            "max_len": int(max(lengths)),
            "mean_len": round(sum(lengths) / len(lengths), 1),
        }

    @_guard
    def records_page(self, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        """Una página de la lista de secuencias (para no volcar miles de golpe)."""
        offset, limit = int(offset), int(limit)
        page = self.records[offset:offset + limit]
        return {
            "total": len(self.records),
            "offset": offset,
            "items": [{
                "index": offset + i,
                "header": r.header[:80],
                "type": "ADN" if r.seq_type == SeqType.NUCLEOTIDE else "proteína",
                "length": r.n_symbols,
                "preview": r.to_string()[:60],
            } for i, r in enumerate(page)],
        }

    # ── operaciones sobre una secuencia ──────────────────────────────────────
    @_guard
    def sequence_detail(self, index: int) -> dict[str, Any]:
        """Detalle de una secuencia: composición y estadísticas."""
        r = self._get(index)
        st = compute_stats(r)
        return {
            "index": int(index),
            "header": r.header,
            "type": "ADN" if r.seq_type == SeqType.NUCLEOTIDE else "proteína",
            "length": r.n_symbols,
            "sequence": r.to_string(),
            "composition": st.composition,
            "compression_pct": round(st.compression_pct, 1),
        }

    @_guard
    def translate(self, index: int) -> dict[str, Any]:
        """Traduce una secuencia de ADN a proteína (primer ORF desde ATG)."""
        r = self._get(index)
        if r.seq_type != SeqType.NUCLEOTIDE:
            return {"error": "esta secuencia ya es una proteína"}
        prot = SmartTranslator.translate(r)
        return {
            "index": int(index),
            "protein": prot.to_string(),
            "length": prot.n_symbols,
        }

    @_guard
    def align(self, i: int, j: int) -> dict[str, Any]:
        """Alinea dos secuencias cargadas y devuelve identidad + mutaciones."""
        a, b = self._get(i), self._get(j)
        res = SequenceAligner.align(a, b, mode="global", band="auto",
                                    detect_mutations=True)
        matches = sum(x == y for x, y in zip(res.aligned_a, res.aligned_b))
        cols = len(res.aligned_a) or 1
        muts = [{"pos_a": m.pos_a, "pos_b": m.pos_b, "ref": m.sym_a,
                 "alt": m.sym_b, "kind": m.kind} for m in (res.mutations or [])]
        return {
            "i": int(i), "j": int(j),
            "score": int(res.score),
            "identity": round(100.0 * matches / cols, 1),
            "length": cols,
            "aligned_a": res.aligned_a,
            "aligned_b": res.aligned_b,
            "mutations": muts,
            "n_mutations": len(muts),
        }

    # ── util ─────────────────────────────────────────────────────────────────
    def _get(self, index: int):
        idx = int(index)
        if not 0 <= idx < len(self.records):
            raise IndexError(f"índice {idx} fuera de rango (hay {len(self.records)})")
        return self.records[idx]

    def ping(self) -> dict[str, Any]:
        """Prueba de vida del puente interfaz↔Python (lo usa el arranque)."""
        return {"ok": True, "engine": "BioForge"}
