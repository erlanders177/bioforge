"""
bioforge/app/backend.py — el PUENTE entre la interfaz web y el motor BioForge.

La app de escritorio es "la otra cara" del mismo motor: una capa FINA de interfaz
sobre el paquete ``bioforge`` (que ya está probado, 546 tests). Aquí vive la lógica
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
import sys
from typing import Any, Callable

import numpy as np


def app_dir() -> str:
    """Carpeta base de recursos — funciona desde el código Y desde el .exe empaquetado.

    PyInstaller extrae los datos a ``sys._MEIPASS``; desde el paquete instalado es la
    carpeta ``bioforge/app`` (donde viven ``index.html`` y ``data/``).
    """
    return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))

from bioforge import (
    BioForgeError,
    RealityCheck,
    SeqType,
    SequenceAligner,
    SmartImporter,
    SmartTranslator,
    compute_stats,
    rank_mutations,
)
from bioforge.nanopore import basecall as _basecall
from bioforge.nanopore import read_fast5 as _read_fast5
from bioforge.nanopore import read_pod5 as _read_pod5
from bioforge.io.qcreport import run as _qc_run

# pore model R9.4 6-mer de ONT, empaquetado con la app (ver app/data/README.md).
_MODEL_PATH = os.path.join(app_dir(), "data", "r9.4_6mer.model")
_PORE_MODEL = None


def _pore_model():
    """Carga (una vez) la tabla k-mero→corriente para el basecaller."""
    global _PORE_MODEL
    if _PORE_MODEL is None:
        code = {"A": 0, "C": 1, "G": 2, "T": 3}
        mean = np.zeros(4096)
        with open(_MODEL_PATH) as f:
            for ln in f:
                if ln.startswith("kmer"):
                    continue
                parts = ln.split()
                idx = 0
                for ch in parts[0]:
                    idx = idx * 4 + code[ch]
                mean[idx] = float(parts[1])
        _PORE_MODEL = mean
    return _PORE_MODEL


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


def _read_records(path: str):
    """Lee un FASTA/FASTQ del disco -> (records, qualities)."""
    if os.path.splitext(path)[1].lower() in (".fastq", ".fq"):
        recs = list(SmartImporter.stream_fastq(path))
        return [r.sequence for r in recs], [r.quality for r in recs]
    return list(SmartImporter.from_file(path)), []


def _meta_of(records: list, qualities: list) -> dict[str, Any]:
    """Resumen BARATO de un archivo: se guarda para no tener que releerlo."""
    lengths = [r.n_symbols for r in records]
    n_nuc = sum(1 for r in records if r.seq_type == SeqType.NUCLEOTIDE)
    return {
        "count": len(records),
        "nucleotide": n_nuc,
        "protein": len(records) - n_nuc,
        "total_symbols": int(sum(lengths)),
        "min_len": int(min(lengths)),
        "max_len": int(max(lengths)),
        "mean_len": round(sum(lengths) / len(lengths), 1),
        "has_quality": bool(qualities),
    }


class Api:
    """API que PyWebview expone a la interfaz como ``window.pywebview.api``.

    Mantiene VARIOS archivos abiertos a la vez (como pestañas de genomas) y uno
    ACTIVO. Las operaciones (traducir, alinear…) trabajan sobre el archivo activo.

    RAM PLANA (v10.1): solo el archivo ACTIVO tiene sus secuencias materializadas.
    De los demás se guarda su ficha (nombre, ruta y resumen ya calculado), y sus
    datos se sueltan; al volver a uno se relee del disco. Así puedes tener 500
    archivos abiertos y la memoria no crece con ellos — que es la promesa Edge.
    Los conjuntos SIN archivo en disco (p.ej. un basecall añadido desde Nanoporo)
    nunca se sueltan, porque no habría de dónde recuperarlos.
    """

    def __init__(self) -> None:
        # cada dataset: {filename, path, records|None, qualities, meta}
        self.datasets: list[dict] = []
        self.active: int = -1            # índice del archivo activo (-1 = ninguno)
        self.signals: list = []          # SignalRead de nanoporo cargadas
        self.signal_filename: str = ""

    # ── memoria: materializar el activo, soltar los demás ────────────────────
    def _materialize(self, i: int) -> list:
        """Devuelve los registros del archivo *i*, releyéndolos si se soltaron."""
        ds = self.datasets[i]
        if ds["records"] is None:
            ds["records"], ds["qualities"] = _read_records(ds["path"])
        return ds["records"]

    def _release_inactive(self) -> None:
        """Suelta los datos de todo archivo que no sea el activo (si es releíble)."""
        for j, ds in enumerate(self.datasets):
            if j != self.active and ds["path"] and ds["records"] is not None:
                ds["records"] = None
                ds["qualities"] = []

    # ── carga y gestión de MÚLTIPLES archivos ────────────────────────────────
    @_guard
    def open_file(self, path: str) -> dict[str, Any]:
        """AÑADE un archivo (FASTA o FASTQ) a los abiertos y lo deja activo."""
        if not path or not os.path.exists(path):
            return {"error": "no existe el archivo"}
        records, qualities = _read_records(path)
        if not records:
            return {"error": "el archivo no contiene secuencias legibles"}
        self.datasets.append({"filename": os.path.basename(path), "path": path,
                              "records": records, "qualities": qualities,
                              "meta": _meta_of(records, qualities)})
        self.active = len(self.datasets) - 1
        self._release_inactive()
        return self.workspace()

    @_guard
    def workspace(self) -> dict[str, Any]:
        """Estado del área de trabajo: los archivos abiertos y cuál está activo."""
        return {
            "files": [self._file_entry(i) for i in range(len(self.datasets))],
            "active": self.active,
            "n_files": len(self.datasets),
        }

    @_guard
    def select_file(self, index: int) -> dict[str, Any]:
        """Cambia el archivo activo (lo relee si hacía falta) y devuelve su resumen."""
        if not 0 <= int(index) < len(self.datasets):
            return {"error": "archivo no válido"}
        self.active = int(index)
        self._materialize(self.active)
        self._release_inactive()
        return self.summary()

    @_guard
    def close_file(self, index: int) -> dict[str, Any]:
        """Cierra un archivo abierto; reajusta cuál queda activo."""
        i = int(index)
        if not 0 <= i < len(self.datasets):
            return {"error": "archivo no válido"}
        self.datasets.pop(i)
        if not self.datasets:
            self.active = -1
        elif self.active >= i:                       # el activo se movió o se cerró
            self.active = max(0, self.active - 1)
        return self.workspace()

    def _file_entry(self, i: int) -> dict[str, Any]:
        # usa la FICHA guardada: listar las pestañas no obliga a cargar los archivos
        m = self.datasets[i]["meta"]
        return {"index": i, "filename": self.datasets[i]["filename"],
                "count": m["count"], "nucleotide": m["nucleotide"],
                "protein": m["protein"], "active": i == self.active}

    @_guard
    def summary(self) -> dict[str, Any]:
        """Resumen del archivo ACTIVO: cuántas secuencias, tipos, longitudes."""
        if self.active < 0 or not self.datasets:
            return {"loaded": False, "n_files": len(self.datasets)}
        ds = self.datasets[self.active]
        return {                                     # todo sale de la ficha guardada
            "loaded": True,
            "filename": ds["filename"],
            "active_index": self.active,
            "n_files": len(self.datasets),
            **ds["meta"],
        }

    @_guard
    def records_page(self, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        """Una página de la lista de secuencias del archivo activo (paginado)."""
        offset, limit = int(offset), int(limit)
        records = self._records()
        page = records[offset:offset + limit]
        return {
            "total": len(records),
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
        """Traduce una secuencia de ADN a proteína (primer ORF desde ATG).

        Devuelve además la lectura codón→aminoácido de los primeros tripletes, para
        poder EXPLICAR en la app cómo el ADN se convierte en proteína.
        """
        r = self._get(index)
        if r.seq_type != SeqType.NUCLEOTIDE:
            return {"error": "esta secuencia ya es una proteína (no hay nada que "
                             "traducir; la traducción va de ADN a proteína)"}
        prot = SmartTranslator.translate(r)
        protein = prot.to_string()
        dna = r.to_string().upper()
        orf = dna.find("ATG")                        # inicio del marco de lectura
        codons = []
        if orf >= 0:
            for i, aa in enumerate(protein):
                c = dna[orf + 3 * i: orf + 3 * i + 3]
                if len(c) < 3:
                    break
                codons.append({"codon": c, "aa": aa})
        return {
            "index": int(index),
            "protein": protein,
            "length": prot.n_symbols,
            "orf_start": int(orf),
            "codons": codons[:60],
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

    # ── informe de calidad (FASTQ) ───────────────────────────────────────────
    @_guard
    def qc_report(self) -> dict[str, Any]:
        """Informe de calidad estilo FastQC del FASTQ activo (una pasada columnar)."""
        if self.active < 0 or not self.datasets:
            return {"error": "no hay ningún archivo activo"}
        ds = self.datasets[self.active]
        if not ds["meta"]["has_quality"]:            # la ficha lo sabe sin releer
            return {"error": "el informe de calidad es para FASTQ (con calidades); "
                             "este archivo no las tiene."}
        r = _qc_run(ds["path"])
        return {
            "filename": ds["filename"],
            "n_reads": int(r.n_reads),
            "total_bases": int(r.total_bases),
            "min_len": int(r.min_len),
            "max_len": int(r.max_len),
            "mean_len": round(r.mean_len, 1),
            "gc_overall": round(r.gc_overall * 100, 1),
            "mean_q": round(r.mean_q_overall, 1),
            "pct_q20": round(r.pct_q20, 1),
            "pct_q30": round(r.pct_q30, 1),
            "pos_q_mean": [round(float(x), 2) for x in r.pos_q_mean],   # calidad/posición
            "meanq_hist": [int(x) for x in r.meanq_hist],               # hist. calidad media
            "gc_hist": [int(x) for x in r.gc_hist],                     # hist. %GC
            "base_frac": [[round(float(x), 4) for x in row]             # composición/posición
                          for row in r.base_frac.tolist()],
        }

    @_guard
    def add_sequence(self, header: str, sequence: str) -> dict[str, Any]:
        """Añade una secuencia (p.ej. un basecall) como NUEVO archivo, usable en todo.

        Cierra el círculo: lo que sale de Nanoporo (o de una traducción) pasa a ser un
        genoma más, disponible en Secuencias, Alinear y demás. Se deja activo.
        """
        seq = "".join(str(sequence).split()).upper()
        if not seq:
            return {"error": "secuencia vacía"}
        name = (header or "secuencia").strip()[:60]
        rec = SmartImporter.from_string(f">{name}\n{seq}\n",
                                        force_type=SeqType.NUCLEOTIDE)[0]
        # sin archivo en disco -> se queda siempre en memoria (no se puede releer)
        self.datasets.append({"filename": name, "path": "",
                              "records": [rec], "qualities": [],
                              "meta": _meta_of([rec], [])})
        self.active = len(self.datasets) - 1
        self._release_inactive()
        return self.workspace()

    @_guard
    def open_example(self) -> dict[str, Any]:
        """Carga un FASTA de EJEMPLO (en memoria) para probar la app sin buscar archivos.

        Tres genes cortos de ADN: dos casi idénticos —para que **Alinear** muestre una
        mutación— y un tercero distinto. Todos traducibles a proteína. Al no depender de
        ningún archivo en disco, funciona igual desde el código y desde el .exe.
        """
        a = ("ATGGCACGTAAAGGCTTTGATGAACTGCATAGCTGGATTCGCGGTCTGGCAGATGAACTGAAAGGCTTTCCG"
             "GATTGGCATAGCGATGAACTGCATAGCTGGATTCGTAAAGGCCTGGCAGATGAAATTAAAGGCTTTGAACTG"
             "CATAGCTGGAAATAA")
        a_var = ("ATGGCACGTAAAGGCTTTGATGAACTGCATAGCTGGATTAGCGGTCTGGCAGATGAACTGAAAGGCTTTCCG"
                 "GATTGGCATAGCGATGAACTGCATAGCTGGATTCGTAAAGGCCTGGCAGATGAAATTAAAGGCTTTGAACTG"
                 "CATAGCTGGAAATAA")
        b = ("ATGAAACCGGGTTTTGATCTGCATTGGGCAAGCCTGAAAGATTTTCGCGGCCTGGCAGAAGGCCATTGGCTG"
             "GATGAACTGAAAGCAGGCTTTCCGTGGCATAGCGAAATTCGCAAAGGCCTGGATGAACTGCATAGCTGGAAA"
             "GGCTTTGATTAA")
        fasta = (f">gen_ejemplo_A demostracion\n{a}\n"
                 f">gen_ejemplo_A_variante una mutacion\n{a_var}\n"
                 f">gen_ejemplo_B otro gen\n{b}\n")
        records = SmartImporter.from_string(fasta)
        if not records:
            return {"error": "no se pudo construir el ejemplo"}
        self.datasets.append({"filename": "ejemplo_adn.fasta", "path": "",
                              "records": records, "qualities": [],
                              "meta": _meta_of(records, [])})
        self.active = len(self.datasets) - 1
        self._release_inactive()
        return self.workspace()

    # ── evolución: predecir qué mutaciones subirán ────────────────────────────
    def _protein_series(self):
        """Comprueba que el archivo activo sirve para evolución (proteínas fechadas)."""
        records = self._records()
        if records and records[0].seq_type != SeqType.PROTEIN:
            return None, ("la predicción de evolución trabaja sobre PROTEÍNAS a lo "
                          "largo del tiempo. Traduce tus genes o carga "
                          "ejemplos/evolucion_proteina.fasta.")
        return records, None

    @_guard
    def predict_mutations(self, top: int = 15) -> dict[str, Any]:
        """Ordena qué mutaciones tienen más probabilidad de SUBIR (rank_mutations)."""
        records, err = self._protein_series()
        if err:
            return {"error": err}
        if len(records) < 10:
            return {"error": "hacen falta muchas secuencias de la misma proteína a lo "
                             "largo del tiempo (mínimo ~10)."}
        seqs = [r.to_string() for r in records]
        res = rank_mutations(seqs, list(range(len(seqs))), method="model")
        muts = [{"site": int(s) + 1, "allele": a, "score": round(float(sc), 3),
                 "novel": bool(res.novel.get((s, a), False))}
                for s, a, sc in res.ranked[:int(top)]]
        return {"n_sequences": len(seqs), "mutations": muts}

    @_guard
    def check_mutation(self, mutation: str) -> dict[str, Any]:
        """Filtro de realidad: ¿esta mutación tiene tracción real? (RealityCheck)."""
        records, err = self._protein_series()
        if err:
            return {"error": err}
        if len(records) < 20:
            return {"error": "el filtro de realidad necesita un histórico amplio "
                             "(≥20 secuencias fechadas de la misma proteína)."}
        seqs = [r.to_string() for r in records]
        rc = RealityCheck(seqs, list(range(len(seqs))))
        v = rc.check(mutation)
        num = lambda x: round(float(x), 3) if x == x else None   # None si es NaN
        return {
            "mutation": v.mutation, "tier": v.tier, "label": v.label,
            "probability": num(v.probability),
            "freq_now": num(v.freq_now), "freq_peak": num(v.freq_peak),
            "reliability": num(v.reliability), "note": v.note,
        }

    # ── nanoporo: señal cruda → bases ─────────────────────────────────────────
    @_guard
    def open_signal(self, path: str) -> dict[str, Any]:
        """Carga un archivo de señal de nanoporo (POD5/FAST5) y lista sus lecturas."""
        if not path or not os.path.exists(path):
            return {"error": "no existe el archivo"}
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pod5":
            try:
                import pod5  # noqa: F401
            except ImportError:
                return {"error": "para leer POD5 falta la librería 'pod5'. "
                                 "Instálala con:  pip install pod5"}
            sigs = list(_read_pod5(path))
        elif ext in (".fast5", ".h5"):
            try:
                import h5py  # noqa: F401
            except ImportError:
                return {"error": "el lector de nanoporo aún se está instalando en "
                                 "segundo plano. Espera 1–2 minutos y reinténtalo."}
            sigs = list(_read_fast5(path))
        else:
            return {"error": "formato de señal no reconocido (usa .pod5 o .fast5)"}
        if not sigs:
            return {"error": "el archivo no contiene lecturas de señal"}
        self.signals = sigs
        self.signal_filename = os.path.basename(path)
        return {
            "filename": self.signal_filename,
            "n": len(sigs),
            "reads": [{"index": i, "read_id": s.read_id,
                       "samples": int(s.n_samples),
                       "sample_rate": float(s.sample_rate)}
                      for i, s in enumerate(sigs[:300])],
        }

    @_guard
    def basecall_read(self, index: int) -> dict[str, Any]:
        """Convierte la señal de una lectura en bases con el basecaller clásico."""
        idx = int(index)
        if not 0 <= idx < len(self.signals):
            return {"error": "lectura no válida"}
        s = self.signals[idx]
        pa = s.to_picoamperes()
        # se acota a un tramo para que responda en segundos (~1000 bases); se recorta
        # el líder/adaptador del inicio, como en el basecaller.
        chunk = pa[1000:11000] if pa.size > 12000 else pa
        bases = _basecall(chunk, _pore_model(), 6)
        # squiggle submuestreado para el gráfico (los primeros ~4000 muestras)
        head = pa[:4000]
        step = max(1, head.size // 600)
        return {
            "index": idx,
            "read_id": s.read_id,
            "n_samples": int(s.n_samples),
            "bases": bases,
            "n_bases": len(bases),
            "signal": [round(float(x), 1) for x in head[::step]],
        }

    # ── util ─────────────────────────────────────────────────────────────────
    def _records(self) -> list:
        if self.active < 0 or not self.datasets:
            raise IndexError("no hay ningún archivo activo")
        return self._materialize(self.active)        # lo relee si se había soltado

    def _get(self, index: int):
        records = self._records()
        idx = int(index)
        if not 0 <= idx < len(records):
            raise IndexError(f"índice {idx} fuera de rango (hay {len(records)})")
        return records[idx]

    def ping(self) -> dict[str, Any]:
        """Prueba de vida del puente interfaz↔Python (lo usa el arranque)."""
        return {"ok": True, "engine": "BioForge"}
