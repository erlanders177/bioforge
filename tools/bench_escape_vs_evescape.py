"""Eje B — el escape a anticuerpos, medido contra verdad de campo REAL.

QUE PONE A PRUEBA
-----------------
EVEscape (Marks lab, Nature 2023) modela el escape como el producto de tres
terminos:  viabilidad  x  accesibilidad a anticuerpos  x  DISIMILITUD quimica.
El tercero mide *cuanto te alejas* del residuo original. Este script pone a
prueba ese supuesto contra un modelo alternativo: que lo que manda no es la
distancia recorrida, sino **el residuo al que llegas**.

Y antes de eso responde a la pregunta que sostiene el diseno de BioForge:
son ESCAPE y VIABILIDAD dos ejes de verdad separados? Si correlacionaran
fuerte, un solo eje bastaria y separarlos en dos sobraria.

DATOS (no se redistribuyen; se descargan)
-----------------------------------------
* Escape: mapas de escape a anticuerpos del laboratorio de Bloom,
  https://github.com/jbloomlab/SARS2_RBD_Ab_escape_maps  (GPL-3; datos de
  estudios publicados, ver processed_data/studies.csv). ~3000 anticuerpos y
  sueros x ~2150 mutaciones del RBD de SARS-CoV-2, por deep mutational scanning.
* Viabilidad: DMS de union a ACE2 y de expresion (Starr et al. 2020), tal como
  los distribuye el benchmark EVEREST/ProteinGym.

DISCIPLINA
----------
* El nivel de SITIO se excluye por TAUTOLOGICO: que los sitios antigenicos son
  los variables se sabe desde los anos 80. Todo se mide DENTRO de cada sitio.
* El liston no es cero: es la mejor senal quimica gratis (el cambio de carga).
* La prueba de diferencia es bootstrap PAREADO por sitio, no comparar dos
  intervalos de confianza que se solapan.
* Se reporta con DOS agregaciones distintas del escape, porque el resultado es
  sensible a ello y ocultarlo seria vender humo.
"""
import collections
import csv
import math
import os
import sys
import urllib.request

import numpy as np

DATOS = os.path.join(os.environ.get("TEMP", "."), "bioforge_escape")
LFS = ("https://media.githubusercontent.com/media/jbloomlab/"
       "SARS2_RBD_Ab_escape_maps/main/processed_data/escape_data_mutation.csv")
EVEREST = os.path.join(os.environ.get("TEMP", "."), "everest_benchmark")

_AA = "ACDEFGHIKLMNPQRSTVWY"
# Kyte & Doolittle 1982 (hidropatia) y carga neta a pH 7
HID = dict(zip(_AA, [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
                     1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]))
CAR = {a: 0.0 for a in _AA}
CAR.update({"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1})

SENALES = {
    "destino hidrofilico": lambda w, m: -HID[m],                 # la alternativa
    "|dcarga|": lambda w, m: abs(CAR[m] - CAR[w]),               # el liston
    "|dhidrofobia|": lambda w, m: abs(HID[m] - HID[w]),          # marco EVEscape
}


def rank(v):
    return np.argsort(np.argsort(np.asarray(v, float))).astype(float)


def spearman(a, b):
    ra, rb = rank(a) - rank(a).mean(), rank(b) - rank(b).mean()
    d = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return float((ra * rb).sum() / d) if d else float("nan")


def descargar():
    os.makedirs(DATOS, exist_ok=True)
    dst = os.path.join(DATOS, "escape_mut.csv")
    if not os.path.exists(dst):
        print("bajando los mapas de escape del laboratorio de Bloom (~140 MB)...")
        urllib.request.urlretrieve(LFS, dst)
    return dst


def leer_escape(path):
    """Dos agregaciones a proposito: por medida y por clase de epitopo."""
    med = collections.defaultdict(lambda: [0.0, 0])
    grp = collections.defaultdict(lambda: [0.0, 0])
    conds = set()
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):                      # bucle por MEDIDA
            k = (int(r["site"]), r["mutation"])
            v = float(r["mut_escape"])
            a = med[k]
            a[0] += v
            a[1] += 1
            b = grp[(r["group"], k)]
            b[0] += v
            b[1] += 1
            conds.add(r["condition"])
    n_med = sum(n for _, n in med.values())
    agg = collections.defaultdict(list)
    for (_, k), (s, n) in grp.items():
        agg[k].append(s / n)
    return ({k: s / n for k, (s, n) in med.items()},
            {k: float(np.mean(v)) for k, v in agg.items()},
            len(conds), n_med)


def leer_dms(nombre):
    ruta = os.path.join(EVEREST, nombre)
    if not os.path.exists(ruta):
        return None
    out = {}
    with open(ruta, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            m = r["mutant"]
            if ":" in m or m[0] not in _AA or m[-1] not in _AA:
                continue
            out[(int(m[1:-1]), m[-1])] = (m[0], float(r["DMS_score"]))
    return out


def rhos_por_sitio(esc, wt, fn, minimo=8):
    """Correlacion DENTRO de cada sitio: el efecto del sitio queda eliminado."""
    por = collections.defaultdict(list)
    for (s, aa) in esc:
        if s in wt and aa in _AA:
            por[s].append(aa)
    out = {}
    for s, aas in por.items():
        if len(aas) < minimo:
            continue
        x = [fn(wt[s], a) for a in aas]
        if len(set(x)) < 3:
            continue
        r = spearman(x, [esc[(s, a)] for a in aas])
        if not math.isnan(r):
            out[s] = r
    return out


def main():
    union = leer_dms("SARS2_RBD_Starr_binding_dms.csv")
    expr = leer_dms("SARS2_RBD_Starr_expression_dms.csv")
    if union is None or expr is None:
        print(f"Faltan los DMS de viabilidad en {EVEREST}.")
        print("Se obtienen del benchmark EVEREST/ProteinGym (SARS2_RBD_Starr_*).")
        return 1

    esc_med, esc_cls, n_ab, n_med = leer_escape(descargar())
    wt = {s: w for (s, _), (w, _) in union.items()}

    print("=" * 78)
    print("EJE B - escape a anticuerpos MEDIDO (mapas del laboratorio de Bloom)")
    print("=" * 78)
    print(f"anticuerpos y sueros: {n_ab:,}   medidas: {n_med:,}   "
          f"mutaciones: {len(esc_med):,}\n")

    # ---- 1. dos ejes, o uno? -----------------------------------------------
    ks = [k for k in esc_med if k in union and k in expr]
    e = [esc_med[k] for k in ks]
    u = [union[k][1] for k in ks]
    x = [expr[k][1] for k in ks]
    print("-" * 78)
    print("1) Son ESCAPE y VIABILIDAD ejes separados?")
    print("-" * 78)
    print(f"  escape vs union a ACE2   rho = {spearman(e, u):+.4f}")
    print(f"  escape vs expresion      rho = {spearman(e, x):+.4f}")
    print(f"  union vs expresion       rho = {spearman(u, x):+.4f}"
          f"   <- CONTROL: dos medidas de viabilidad SI se parecen")

    # ---- 2. cuanto del escape es trivial ------------------------------------
    por_sitio = collections.defaultdict(list)
    for k in ks:
        por_sitio[k[0]].append(esc_med[k])
    medias = {s: float(np.mean(v)) for s, v in por_sitio.items()}
    ev = np.array(e)
    resid = np.array([esc_med[k] - medias[k[0]] for k in ks])
    dentro = float(resid.var() / ev.var())
    print(f"\n  varianza del escape explicada SOLO por el sitio: {1 - dentro:.1%}"
          f"  (tautologico, se descarta)")
    print(f"  queda DENTRO del sitio: {dentro:.1%}  <- lo no trivial")

    # ---- 3. destino frente a disimilitud ------------------------------------
    rng = np.random.default_rng(0)
    for etiqueta, esc in (("promediando MEDIDAS", esc_med),
                          ("promediando CLASES de epitopo", esc_cls)):
        print()
        print("-" * 78)
        print(f"2) DENTRO de cada sitio - agregacion: {etiqueta}")
        print("-" * 78)
        R = {k: rhos_por_sitio(esc, wt, f) for k, f in SENALES.items()}
        for k, d in R.items():
            print(f"  {k:<24}{np.mean(list(d.values())):>+9.4f}   ({len(d)} sitios)")
        print("\n  bootstrap PAREADO de la diferencia (2000 remuestreos):")
        for a, b in (("destino hidrofilico", "|dcarga|"),
                     ("destino hidrofilico", "|dhidrofobia|")):
            com = sorted(set(R[a]) & set(R[b]))
            d = np.array([R[a][s] - R[b][s] for s in com])
            bs = np.array([d[rng.integers(0, len(d), len(d))].mean()
                           for _ in range(2000)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f"    {a} - {b:<16} d={d.mean():+.4f}"
                  f"  IC95% [{lo:+.4f}, {hi:+.4f}]"
                  f"  -> {'SIGNIFICATIVA' if lo > 0 else 'no concluyente'}")

    print()
    print("Lectura honesta: 'destino' bate a 'disimilitud' con las DOS")
    print("agregaciones; NO bate de forma fiable al cambio de carga (depende de")
    print("como se agregue), asi que eso no se afirma.")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
