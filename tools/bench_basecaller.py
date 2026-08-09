"""
tools/bench_basecaller.py — el basecaller clásico de BioForge sobre señal R9.4 REAL.

Reproduce, desde cero y con datos públicos, el número honesto del basecaller: cuánto
acierta nuestro decodificador Viterbi (NumPy puro, sin IA) sobre señal de nanoporo
CAPTURADA de verdad. Nada simulado en la medición.

Descarga (una vez, a un directorio temporal):
  · el pore model R9.4 6-mer OFICIAL de Oxford Nanopore (github nanoporetech/kmer_models),
  · un puñado de reads R9.4 REALES de E. coli (github hasindu2008/f5c, el clásico
    dataset del tutorial de nanopolish) con sus basecalls de producción (Guppy).

Luego basecalleamos cada read con nuestro pipeline y medimos la identidad (por
alineamiento LOCAL, con nuestro propio alineador) contra el basecall de producción.

Uso:  python tools/bench_basecaller.py [n_reads]

Requiere el extra:  pip install "bioforge[nanopore]"  (h5py, para leer FAST5).
Números de referencia medidos: media ~70% (mediana ~71%, n=36) — en el rango de los
basecallers clásicos históricos (nanocall), lejísimos aún del ~99% neuronal de Dorado,
que es justo la lección honesta: el método clásico es de la era R9 y sin IA tiene techo.
"""

import io
import json
import os
import sys
import urllib.request

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from bioforge import SeqType, SequenceAligner, SmartImporter  # noqa: E402
from bioforge.nanopore import basecall, read_fast5  # noqa: E402

CACHE = os.path.join(os.path.expanduser("~"), ".cache", "bioforge", "basecaller_bench")
MODEL_URL = ("https://raw.githubusercontent.com/nanoporetech/kmer_models/master/"
             "legacy/legacy_r9.4_180mv_450bps_6mer/template_median68pA.model")
F5_API = "https://api.github.com/repos/hasindu2008/f5c/git/trees/HEAD?recursive=1"
F5_RAW = "https://media.githubusercontent.com/media/hasindu2008/f5c/master/"
F5_RAW2 = "https://raw.githubusercontent.com/hasindu2008/f5c/master/"
DIR = "test/ecoli_2kb_region/"


def _get(url, binary=True):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read() if binary else r.read().decode()


def _fetch(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    data = _get(url)
    if data[:40].lstrip().startswith(b"version https://git-lfs"):   # puntero LFS
        data = _get(url.replace(F5_RAW2, F5_RAW))
    with open(dest, "wb") as f:
        f.write(data)
    return dest


def load_model(path):
    B = {"A": 0, "C": 1, "G": 2, "T": 3}
    mean = np.zeros(4096)
    for ln in open(path):
        if ln.startswith("kmer"):
            continue
        p = ln.split()
        idx = 0
        for ch in p[0]:
            idx = idx * 4 + B[ch]
        mean[idx] = float(p[1])
    return mean


def revcomp(s):
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def local_identity(a, b):
    """Identidad LOCAL (sobre la región solapada) con nuestro propio alineador."""
    if not a or not b:
        return 0.0
    pa = SmartImporter.from_string(f">a\n{a}\n", force_type=SeqType.NUCLEOTIDE)[0]
    pb = SmartImporter.from_string(f">b\n{b}\n", force_type=SeqType.NUCLEOTIDE)[0]
    r = SequenceAligner.align_local(pa, pb)
    if not r.aligned_a:
        return 0.0
    return sum(x == y for x, y in zip(r.aligned_a, r.aligned_b)) / len(r.aligned_a)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    print("═" * 62)
    print("  BioForge — basecaller clásico sobre señal R9.4 REAL")
    print("═" * 62)

    print("Descargando (o cacheando) modelo R9.4 + reads reales…")
    model = load_model(_fetch(MODEL_URL, os.path.join(CACHE, "r9.4_6mer.model")))
    _fetch(F5_RAW2 + DIR + "reads.fasta", os.path.join(CACHE, "reads.fasta"))

    # basecalls oficiales (Guppy) por read_id
    offi, cur, buf = {}, None, []
    for ln in open(os.path.join(CACHE, "reads.fasta")):
        if ln.startswith(">"):
            if cur:
                offi[cur] = "".join(buf)
            cur, buf = ln[1:].split()[0].strip(), []
        else:
            buf.append(ln.strip())
    if cur:
        offi[cur] = "".join(buf)

    # nombres de los FAST5 (vía API de GitHub), y bajar n
    tree = json.loads(_get(F5_API, binary=False))["tree"]
    names = [e["path"] for e in tree
             if e["path"].startswith(DIR + "fast5_files/") and e["path"].endswith(".fast5")]
    names = sorted(names)[:n]

    ids = []
    for i, key in enumerate(names, 1):
        dest = os.path.join(CACHE, os.path.basename(key))
        _fetch(F5_RAW2 + key, dest)
        read = next(iter(read_fast5(dest)), None)
        if read is None or read.read_id not in offi:
            continue
        pa = read.to_picoamperes()[2000:18000]      # recorta el líder/adaptador
        ours = basecall(pa, model, 6)
        truth = offi[read.read_id]
        idn = max(local_identity(ours, truth), local_identity(revcomp(ours), truth))
        ids.append(idn)
        print(f"  [{i:2d}/{len(names)}] read {read.read_id[:8]}… identidad {idn:.1%}")

    a = np.array(ids)
    print("─" * 62)
    print(f"  IDENTIDAD sobre señal R9.4 REAL: media {a.mean():.1%} · "
          f"mediana {np.median(a):.1%} · rango {a.min():.0%}-{a.max():.0%} · n={a.size}")
    print(f"  (azar ≈ 25% · clásico histórico ~68-85% · Dorado neuronal ~95-99%)")
    print("═" * 62)


if __name__ == "__main__":
    main()
