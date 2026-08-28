"""DIAGNOSTICO: por que 'destino' funciona en SARS-CoV-2 y en nada mas.

No basta con saber que no replica: hay que saber DONDE se pierde. Este script
descompone el fallo en tres preguntas, cada una con su experimento.

PASO 1 - Es un fallo UNIFORME o CONCENTRADO?
   Si en gripe/VIH/Zika la senal fuera buena en unos sitios y mala en otros,
   habria algo que rescatar. Si esta centrada en cero en todos, no.

PASO 2 - Es RUIDO DE MEDIDA?  <- la hipotesis principal
   SARS-CoV-2 promedia ~3000 anticuerpos; Zika, 5. Promediar muchos reduce el
   ruido de la ETIQUETA, y menos ruido sube cualquier correlacion. Si al
   degradar SARS-CoV-2 a 5 anticuerpos su senal cae al nivel de los demas,
   entonces la diferencia nunca fue biologica: era el tamano de la muestra.
   Se mide tambien el liston (carga) en la misma curva: si la carga aguanta la
   degradacion y el destino no, eso SI seria una diferencia real entre senales.

PASO 3 - Es el FILTRADO de mutaciones inviables?
   El escape de SARS-CoV-2 se mide solo sobre variantes que pliegan y unen
   ACE2. 'diffsel' no filtra nada: las mutaciones que rompen la proteina entran
   como ruido. Se comprueba filtrando la gripe con su propio DMS de fitness.
"""
import collections
import csv
import math
import os
import sys

import numpy as np

DATOS = os.path.join(os.environ.get("TEMP", "."), "bioforge_escape")
MULTI = os.path.join(DATOS, "multi")
EVEREST = os.path.join(os.environ.get("TEMP", "."), "everest_benchmark")

_AA = "ACDEFGHIKLMNPQRSTVWY"
IDX = {a: i for i, a in enumerate(_AA)}
HID = dict(zip(_AA, [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
                     1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]))
CAR = {a: 0.0 for a in _AA}
CAR.update({"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1})
SENALES = {"destino": lambda w, m: -HID[m],
           "carga": lambda w, m: abs(CAR[m] - CAR[w]),
           "disimilitud": lambda w, m: abs(HID[m] - HID[w])}


def rank(v):
    return np.argsort(np.argsort(np.asarray(v, float))).astype(float)


def spearman(a, b):
    ra, rb = rank(a) - rank(a).mean(), rank(b) - rank(b).mean()
    d = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return float((ra * rb).sum() / d) if d else float("nan")


def rhos(esc, fn, minimo=8):
    """Un rho por sitio; el efecto del sitio queda eliminado."""
    por = collections.defaultdict(list)
    for (s, w, m) in esc:
        por[(s, w)].append(m)
    out = []
    for (s, w), muts in por.items():
        if len(muts) < minimo:
            continue
        x = [fn(w, m) for m in muts]
        if len(set(x)) < 3:
            continue
        r = spearman(x, [esc[(s, w, m)] for m in muts])
        if not math.isnan(r):
            out.append(r)
    return np.array(out)


# ----------------------------------------------------------------- cargadores
def sars2_crudo():
    """Devuelve arrays por MEDIDA, para poder submuestrear anticuerpos."""
    wt = {}
    with open(os.path.join(EVEREST, "SARS2_RBD_Starr_binding_dms.csv"),
             encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            m = r["mutant"]
            if ":" not in m and m[0] in _AA:
                wt[m[1:-1]] = m[0]
    cond, sitio, mut, val = [], [], [], []
    ids = {}
    with open(os.path.join(DATOS, "escape_mut.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):                    # bucle por MEDIDA
            s, m = r["site"], r["mutation"]
            w = wt.get(s)
            if w is None or m not in _AA or w == m:
                continue
            c = r["condition"]
            if c not in ids:
                ids[c] = len(ids)
            cond.append(ids[c])
            sitio.append(int(s))
            mut.append(IDX[m])
            val.append(float(r["mut_escape"]))
    return (np.array(cond), np.array(sitio), np.array(mut),
            np.array(val), wt, len(ids))


def diffsel(carpeta, positivo=True):
    s = collections.defaultdict(float)
    n = collections.defaultdict(int)
    for f in sorted(os.listdir(carpeta)):
        if not f.endswith(".csv"):
            continue
        with open(os.path.join(carpeta, f), encoding="utf-8") as fh:
            for r in csv.DictReader(fh):                # bucle por MEDIDA
                w, m = r["wildtype"], r["mutation"]
                if w not in _AA or m not in _AA or w == m:
                    continue
                try:
                    v = float(r["mutdiffsel"])
                except (ValueError, KeyError):
                    continue
                if math.isnan(v):
                    continue
                k = (r["site"], w, m)
                s[k] += max(v, 0.0) if positivo else v
                n[k] += 1
    return {k: s[k] / n[k] for k in s}


def flu_serum(positivo=True):
    s = collections.defaultdict(float)
    n = collections.defaultdict(int)
    with open(os.path.join(MULTI, "flu_serum.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):                    # bucle por MEDIDA
            w, m = r["wildtype"], r["mutation"]
            if w not in _AA or m not in _AA or w == m:
                continue
            try:
                v = float(r["mutdiffsel"])
            except ValueError:
                continue
            if math.isnan(v):
                continue
            k = (r["site"], w, m)
            s[k] += max(v, 0.0) if positivo else v
            n[k] += 1
    return {k: s[k] / n[k] for k in s}


# --------------------------------------------------------------------- pasos
def paso1(conjuntos):
    print("=" * 82)
    print("PASO 1 - El fallo, ¿es uniforme o hay sitios que se salvan?")
    print("=" * 82)
    print("  Si la senal fuera buena en un subconjunto de sitios, habria algo")
    print("  que rescatar. Si esta centrada en cero, no hay nada dentro.\n")
    print(f"  {'conjunto':<24}{'sitios':>7}{'media':>9}{'mediana':>9}"
          f"{'>0':>8}{'>+0.3':>8}{'<-0.3':>8}")
    print("  " + "-" * 72)
    for nom, esc in conjuntos:
        r = rhos(esc, SENALES["destino"])
        print(f"  {nom:<24}{len(r):>7}{r.mean():>+9.3f}{np.median(r):>+9.3f}"
              f"{f'{(r>0).mean():.0%}':>8}{f'{(r>0.3).mean():.0%}':>8}"
              f"{f'{(r<-0.3).mean():.0%}':>8}")


def paso2(rng):
    print()
    print("=" * 82)
    print("PASO 2 - ¿Es ruido de medida? Degradando SARS-CoV-2 a proposito")
    print("=" * 82)
    cond, sitio, mut, val, wt, n_cond = sars2_crudo()
    print(f"  {n_cond} anticuerpos y sueros, {len(val):,} medidas.")
    print("  Se promedian solo N anticuerpos al azar y se remide (20 repeticiones).\n")
    sitios_u = np.unique(sitio)
    pos = {s: i for i, s in enumerate(sitios_u)}
    si = np.array([pos[s] for s in sitio])
    clave = si * 20 + mut
    nclv = len(sitios_u) * 20
    print(f"  {'N anticuerpos':>14}{'destino':>12}{'carga':>12}{'disimilitud':>14}")
    print("  " + "-" * 54)
    for N in (5, 10, 25, 50, 100, 300, 1000, n_cond):
        acc = {k: [] for k in SENALES}
        reps = 1 if N == n_cond else 20
        for _ in range(reps):
            sel = (np.ones(len(cond), bool) if N == n_cond
                   else np.isin(cond, rng.choice(n_cond, N, replace=False)))
            suma = np.bincount(clave[sel], weights=val[sel], minlength=nclv)
            cuenta = np.bincount(clave[sel], minlength=nclv)
            hay = cuenta > 0
            esc = {}
            for k in np.where(hay)[0]:
                s = int(sitios_u[k // 20])
                m = _AA[k % 20]
                w = wt.get(str(s))
                if w and w != m:
                    esc[(s, w, m)] = suma[k] / cuenta[k]
            for nm, fn in SENALES.items():
                r = rhos(esc, fn)
                if len(r):
                    acc[nm].append(r.mean())
        print(f"  {N:>14}{np.mean(acc['destino']):>+12.4f}"
              f"{np.mean(acc['carga']):>+12.4f}"
              f"{np.mean(acc['disimilitud']):>+14.4f}")
    print("\n  Si 'destino' cae a ~0.02 con pocos anticuerpos, la diferencia con")
    print("  gripe/VIH/Zika era TAMANO DE MUESTRA, no biologia.")


def paso3():
    print()
    print("=" * 82)
    print("PASO 3 - ¿Es el filtrado de mutaciones inviables?")
    print("=" * 82)
    dms = os.path.join(EVEREST, "IAV_H3_HA_Lee_dms.csv")
    if not os.path.exists(dms):
        print("  (falta IAV_H3_HA_Lee_dms.csv del benchmark EVEREST)")
        return
    fit = {}
    with open(dms, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            m = r["mutant"]
            if ":" in m or m[0] not in _AA or m[-1] not in _AA:
                continue
            fit[(m[1:-1], m[-1])] = float(r["DMS_score"])
    print(f"  DMS de fitness de la misma HA (Perth/2009): {len(fit):,} mutaciones.")
    print("  El escape de SARS-CoV-2 solo se mide sobre variantes que PLIEGAN;")
    print("  'diffsel' no filtra. Se filtra la gripe igual y se remide.\n")
    esc = flu_serum()
    corte = np.percentile(list(fit.values()), 50)
    viables = {k: v for k, v in esc.items() if fit.get((k[0], k[2]), corte) >= corte}
    print(f"  {'version':<34}{'sitios':>8}{'destino':>10}{'carga':>10}")
    print("  " + "-" * 62)
    for nom, e in (("gripe H3N2, todo", esc),
                   ("gripe H3N2, solo VIABLES", viables)):
        rd = rhos(e, SENALES["destino"])
        rc = rhos(e, SENALES["carga"])
        print(f"  {nom:<34}{len(rd):>8}{rd.mean():>+10.4f}{rc.mean():>+10.4f}")
    print(f"\n  cobertura del DMS sobre las mutaciones medidas: "
          f"{sum(1 for k in esc if (k[0],k[2]) in fit)/len(esc):.0%}")


def main():
    rng = np.random.default_rng(0)
    conjuntos = []
    if os.path.exists(os.path.join(DATOS, "escape_mut.csv")):
        cond, sitio, mut, val, wt, _ = sars2_crudo()
        s = collections.defaultdict(float)
        n = collections.defaultdict(int)
        for c, si, mi, v in zip(cond, sitio, mut, val):
            k = (int(si), wt[str(si)], _AA[mi])
            s[k] += v
            n[k] += 1
        conjuntos.append(("SARS-CoV-2 RBD", {k: s[k] / n[k] for k in s}))
    if os.path.exists(os.path.join(MULTI, "flu_serum.csv")):
        conjuntos.append(("gripe H3N2 sueros", flu_serum()))
    for tag, nom in (("flu_mab", "gripe H3N2 mAbs"), ("hiv", "VIH-1 Env"),
                     ("zika", "Zika E")):
        d = os.path.join(MULTI, tag)
        if os.path.isdir(d):
            conjuntos.append((nom, diffsel(d)))
    paso1(conjuntos)
    paso2(rng)
    paso3()
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
