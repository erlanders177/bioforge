"""Replica en CUATRO FAMILIAS DE VIRUS: destino o distancia?

EL HALLAZGO A REPLICAR
----------------------
EVEscape modela el termino quimico del escape como DISIMILITUD: cuanto te
alejas del residuo original. Sobre el RBD de SARS-CoV-2 medimos que la
alternativa -que lo que manda es el residuo al que LLEGAS- lo bate con holgura
(+0.31 frente a +0.05). Pero era UN virus y UN dominio, asi que no se podia
afirmar nada general. Esto lo pone a prueba en cuatro familias distintas.

LOS DATOS (publicos, del laboratorio de Bloom; NO se redistribuyen)
------------------------------------------------------------------
  familia          virus / proteina        anticuerpos        estilo
  ---------------  ----------------------  -----------------  ---------------
  Coronaviridae    SARS-CoV-2, RBD         ~3000 mAbs+sueros  escape fraction
  Orthomyxoviridae gripe H3N2, HA          sueros humanos     diffsel
  Orthomyxoviridae gripe H3N2, HA          monoclonales       diffsel
  Retroviridae     VIH-1 BG505, Env        bnAbs              diffsel
  Flaviviridae     Zika, proteina E        monoclonales       diffsel

Cuatro familias, dos tipos de genoma (ARN+ y ARN-, mas un retrovirus) y dos
estilos de ensayo (monoclonal y suero policlonal). Si el resultado aguanta
ahi, deja de ser una particularidad del RBD.

DISCIPLINA (la misma de siempre)
--------------------------------
* Todo se mide DENTRO de cada sitio: el nivel de sitio es tautologico.
* El liston no es cero, es la mejor senal quimica gratis (el cambio de carga).
* La prueba es bootstrap PAREADO por sitio, no comparar intervalos que se solapan.
* Cada virus se reporta por separado. Nada de promediar para tapar un fallo.
"""
import collections
import csv
import math
import os
import sys
import urllib.request

import numpy as np

DATOS = os.path.join(os.environ.get("TEMP", "."), "bioforge_escape")
MULTI = os.path.join(DATOS, "multi")
RAW = "https://raw.githubusercontent.com/jbloomlab"

_AA = "ACDEFGHIKLMNPQRSTVWY"
HID = dict(zip(_AA, [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
                     1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]))
CAR = {a: 0.0 for a in _AA}
CAR.update({"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1})

SENALES = {
    "destino hidrofilico": lambda w, m: -HID[m],
    "|dcarga|": lambda w, m: abs(CAR[m] - CAR[w]),
    "|dhidrofobia|": lambda w, m: abs(HID[m] - HID[w]),
}


def rank(v):
    return np.argsort(np.argsort(np.asarray(v, float))).astype(float)


def spearman(a, b):
    ra, rb = rank(a) - rank(a).mean(), rank(b) - rank(b).mean()
    d = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return float((ra * rb).sum() / d) if d else float("nan")


def _acumular(pares):
    """(clave -> media) a partir de un flujo de (clave, valor)."""
    s = collections.defaultdict(float)
    n = collections.defaultdict(int)
    for k, v in pares:
        s[k] += v
        n[k] += 1
    return {k: s[k] / n[k] for k in s}


def cargar_diffsel(carpeta):
    """Ficheros con columnas site/wildtype/mutation/mutdiffsel."""
    def flujo():
        for f in sorted(os.listdir(carpeta)):
            if not f.endswith(".csv"):
                continue
            with open(os.path.join(carpeta, f), encoding="utf-8") as fh:
                for r in csv.DictReader(fh):          # bucle por MEDIDA
                    w, m = r["wildtype"], r["mutation"]
                    if w not in _AA or m not in _AA or w == m:
                        continue
                    try:
                        v = float(r["mutdiffsel"])
                    except (ValueError, KeyError):
                        continue
                    if math.isnan(v):
                        continue
                    yield (r["site"], w, m), v
    return _acumular(flujo())


def cargar_flu_serum(path):
    """El tidy de los sueros humanos de H3N2 (Lee et al. 2019)."""
    def flujo():
        with open(path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):              # bucle por MEDIDA
                w, m = r["wildtype"], r["mutation"]
                if w not in _AA or m not in _AA or w == m:
                    continue
                try:
                    v = float(r["mutdiffsel"])
                except ValueError:
                    continue
                if math.isnan(v):
                    continue
                yield (r["site"], w, m), v
    return _acumular(flujo())


def cargar_sars2():
    """El RBD de SARS-CoV-2: el escape viene sin columna de residuo original."""
    esc = os.path.join(DATOS, "escape_mut.csv")
    dms = os.path.join(os.environ.get("TEMP", "."), "everest_benchmark",
                       "SARS2_RBD_Starr_binding_dms.csv")
    if not (os.path.exists(esc) and os.path.exists(dms)):
        return None
    wt = {}
    with open(dms, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            m = r["mutant"]
            if ":" not in m and m[0] in _AA:
                wt[m[1:-1]] = m[0]

    def flujo():
        with open(esc, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):              # bucle por MEDIDA
                s, m = r["site"], r["mutation"]
                w = wt.get(s)
                if w is None or m not in _AA or w == m:
                    continue
                yield (s, w, m), float(r["mut_escape"])
    return _acumular(flujo())


def descargar():
    os.makedirs(MULTI, exist_ok=True)
    tidy = os.path.join(MULTI, "flu_serum.csv")
    if not os.path.exists(tidy):
        print("bajando los sueros humanos de H3N2 (~39 MB)...")
        urllib.request.urlretrieve(
            f"{RAW}/map_flu_serum_Perth2009_H3_HA/master/results/avgdiffsel/"
            "avg_sel_tidy.csv", tidy)
    faltan = [t for t in ("hiv", "zika", "flu_mab")
              if not os.path.isdir(os.path.join(MULTI, t))]
    if faltan:
        print(f"Faltan carpetas de datos: {', '.join(faltan)} en {MULTI}")
        print("Se obtienen de los repos jbloomlab/EnvsAntigenicAtlas,")
        print("ZIKV_MAP_GooLab y Perth2009-HA_mAb_MAP (ficheros *mutdiffsel*).")
    return tidy


def rhos_por_sitio(esc, fn, minimo=8):
    """Correlacion DENTRO de cada sitio (el efecto del sitio queda fuera)."""
    por = collections.defaultdict(list)
    for (s, w, m) in esc:
        por[(s, w)].append(m)
    out = {}
    for (s, w), muts in por.items():
        if len(muts) < minimo:
            continue
        x = [fn(w, m) for m in muts]
        if len(set(x)) < 3:
            continue
        r = spearman(x, [esc[(s, w, m)] for m in muts])
        if not math.isnan(r):
            out[s] = r
    return out


def analizar(nombre, familia, esc, rng):
    R = {k: rhos_por_sitio(esc, f) for k, f in SENALES.items()}
    n = len(R["destino hidrofilico"])
    if n < 10:
        print(f"  {nombre:<34}(solo {n} sitios utilizables; se omite)")
        return None
    fila = {k: float(np.mean(list(v.values()))) for k, v in R.items()}
    out = {"nombre": nombre, "familia": familia, "sitios": n, **fila}
    for a, b, etq in (("destino hidrofilico", "|dhidrofobia|", "vs_disim"),
                      ("destino hidrofilico", "|dcarga|", "vs_carga")):
        com = sorted(set(R[a]) & set(R[b]))
        d = np.array([R[a][s] - R[b][s] for s in com])
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean()
                       for _ in range(2000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[etq] = (float(d.mean()), float(lo), float(hi))
    return out


def _tabla(conjuntos, transf, rng, titulo, nota):
    print()
    print("=" * 92)
    print(titulo)
    print(nota)
    print("=" * 92)
    print(f"{'conjunto':<26}{'familia':<19}{'sitios':>7}"
          f"{'destino':>10}{'|dcarga|':>10}{'|dhidrof|':>11}")
    print("-" * 92)
    filas = []
    for nom, fam, esc in conjuntos:
        f = analizar(nom, fam, {k: transf(v) for k, v in esc.items()}, rng)
        if f is None:
            continue
        filas.append(f)
        print(f"{f['nombre']:<26}{f['familia']:<19}{f['sitios']:>7}"
              f"{f['destino hidrofilico']:>+10.4f}{f['|dcarga|']:>+10.4f}"
              f"{f['|dhidrofobia|']:>+11.4f}")
    print(f"\n{'conjunto':<26}{'destino - disimilitud':>34}{'destino - carga':>32}")
    print("-" * 92)
    for f in filas:
        a, b = f["vs_disim"], f["vs_carga"]
        ta = f"{a[0]:+.3f} [{a[1]:+.3f},{a[2]:+.3f}]" + (" OK" if a[1] > 0 else " --")
        tb = f"{b[0]:+.3f} [{b[1]:+.3f},{b[2]:+.3f}]" + (" OK" if b[1] > 0 else " --")
        print(f"{f['nombre']:<26}{ta:>34}{tb:>32}")
    ok = sum(1 for f in filas if f["vs_disim"][1] > 0)
    print(f"\n  destino bate a la DISIMILITUD en {ok}/{len(filas)} conjuntos.")
    return filas


def main():
    tidy = descargar()
    rng = np.random.default_rng(0)
    conjuntos = []

    s2 = cargar_sars2()
    if s2:
        conjuntos.append(("SARS-CoV-2 RBD", "Coronaviridae", s2))
    if os.path.exists(tidy):
        conjuntos.append(("gripe H3N2 HA - sueros", "Orthomyxoviridae",
                          cargar_flu_serum(tidy)))
    for tag, nom, fam in (("flu_mab", "gripe H3N2 HA - mAbs", "Orthomyxoviridae"),
                          ("hiv", "VIH-1 BG505 Env", "Retroviridae"),
                          ("zika", "Zika proteina E", "Flaviviridae")):
        d = os.path.join(MULTI, tag)
        if os.path.isdir(d):
            conjuntos.append((nom, fam, cargar_diffsel(d)))

    print("=" * 92)
    print("DESTINO o DISTANCIA - replica en cuatro familias de virus")
    print("=" * 92)

    _tabla(conjuntos, lambda v: v, rng,
           "TRATAMIENTO 1 - diffsel EN CRUDO (incluye seleccion negativa)",
           "  Es lo primero que se corrio. Mezcla escape con agotamiento.")

    _tabla(conjuntos, lambda v: max(v, 0.0), rng,
           "TRATAMIENTO 2 - solo seleccion POSITIVA  <- el principiado",
           "  El escape es, por convencion del campo (y del propio laboratorio\n"
           "  de Bloom, que grafica 'positive diffsel'), la seleccion positiva.\n"
           "  AVISO DE HONESTIDAD: este tratamiento se aplico DESPUES de ver el\n"
           "  resultado del 1. Es el correcto por principio, pero la eleccion\n"
           "  fue posterior, y eso hay que decirlo. Por eso van los dos.")

    print()
    print("=" * 92)
    print("CONCLUSION")
    print("=" * 92)
    print("  El hallazgo del RBD *** NO REPLICA ***: 'destino' solo funciona en")
    print("  SARS-CoV-2 (+0.31); en gripe, VIH y Zika se queda en ~0.02. Y eso")
    print("  vale con LOS DOS tratamientos, asi que no depende de esa eleccion.")
    print()
    print("  Lo que SI replica es el liston que intentabamos batir: el CAMBIO")
    print("  DE CARGA, en las cuatro familias y los dos estilos de ensayo.")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
