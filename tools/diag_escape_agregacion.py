"""Cuanto de lo que medimos depende de COMO agregamos? (auditoria propia)

Se detecto una inconsistencia entre dos scripts nuestros: uno promediaba las
medidas y luego recortaba la seleccion negativa; el otro recortaba primero y
promediaba despues. No son la misma operacion, y daban signos opuestos para el
cambio de carga en gripe (+0.267 frente a -0.090).

Antes de afirmar nada mas, se miden LAS CUATRO agregaciones sobre los mismos
datos. Si una senal cambia de signo segun la agregacion, esa senal no se
puede vender, por bonito que sea el numero que salga en una de ellas.

  T1  media del diffsel CRUDO                (incluye seleccion negativa)
  T2  recortar la MEDIA a cero               (lo que hacia el benchmark)
  T3  recortar cada MEDIDA y luego promediar (lo que hacia el diagnostico)
  T4  RANGOS dentro del sitio por condicion, promediados   <- el robusto

T4 es el mas defendible para un analisis basado en rangos: cada anticuerpo
ordena las mutaciones de un sitio, y se promedian ordenes. No depende de la
escala de la metrica, ni de recortes, ni de que un anticuerpo tenga valores
mas grandes que otro. Es el arbitro.
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


def medidas_sars2():
    """(condicion, sitio, wt, mut, valor) para el RBD de SARS-CoV-2."""
    wt = {}
    with open(os.path.join(EVEREST, "SARS2_RBD_Starr_binding_dms.csv"),
             encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            m = r["mutant"]
            if ":" not in m and m[0] in _AA:
                wt[m[1:-1]] = m[0]
    with open(os.path.join(DATOS, "escape_mut.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):                     # bucle por MEDIDA
            s, m = r["site"], r["mutation"]
            w = wt.get(s)
            if w and m in _AA and w != m:
                yield r["condition"], s, w, m, float(r["mut_escape"])


def medidas_diffsel(carpeta):
    for f in sorted(os.listdir(carpeta)):
        if not f.endswith(".csv"):
            continue
        with open(os.path.join(carpeta, f), encoding="utf-8") as fh:
            for r in csv.DictReader(fh):                 # bucle por MEDIDA
                w, m = r["wildtype"], r["mutation"]
                if w not in _AA or m not in _AA or w == m:
                    continue
                try:
                    v = float(r["mutdiffsel"])
                except (ValueError, KeyError):
                    continue
                if not math.isnan(v):
                    yield f, r["site"], w, m, v


def medidas_flu_serum():
    with open(os.path.join(MULTI, "flu_serum.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):                     # bucle por MEDIDA
            w, m = r["wildtype"], r["mutation"]
            if w not in _AA or m not in _AA or w == m:
                continue
            try:
                v = float(r["mutdiffsel"])
            except ValueError:
                continue
            if not math.isnan(v):
                yield r["serum"], r["site"], w, m, v


def agregar(medidas, modo):
    """Devuelve {(sitio, wt, mut): valor} segun el modo de agregacion."""
    if modo == "T4":
        # rangos DENTRO del sitio, por condicion, promediados
        porcond = collections.defaultdict(list)
        for c, s, w, m, v in medidas:
            porcond[(c, s, w)].append((m, v))
        acc = collections.defaultdict(list)
        for (c, s, w), lista in porcond.items():
            if len(lista) < 3:
                continue
            vals = [v for _, v in lista]
            rr = rank(vals) / (len(vals) - 1)            # 0..1, escala-libre
            for (m, _), q in zip(lista, rr):
                acc[(s, w, m)].append(float(q))
        return {k: float(np.mean(v)) for k, v in acc.items()}

    s_ = collections.defaultdict(float)
    n_ = collections.defaultdict(int)
    for c, s, w, m, v in medidas:
        k = (s, w, m)
        s_[k] += max(v, 0.0) if modo == "T3" else v
        n_[k] += 1
    out = {k: s_[k] / n_[k] for k in s_}
    if modo == "T2":
        out = {k: max(v, 0.0) for k, v in out.items()}
    return out


def rho_medio(esc, fn, minimo=8):
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
    return (float(np.mean(out)), len(out)) if out else (float("nan"), 0)


def main():
    fuentes = []
    if os.path.exists(os.path.join(DATOS, "escape_mut.csv")):
        fuentes.append(("SARS-CoV-2 RBD", lambda: medidas_sars2()))
    if os.path.exists(os.path.join(MULTI, "flu_serum.csv")):
        fuentes.append(("gripe H3N2 sueros", lambda: medidas_flu_serum()))
    for tag, nom in (("flu_mab", "gripe H3N2 mAbs"), ("hiv", "VIH-1 Env"),
                     ("zika", "Zika E")):
        d = os.path.join(MULTI, tag)
        if os.path.isdir(d):
            fuentes.append((nom, (lambda dd: lambda: medidas_diffsel(dd))(d)))

    print("=" * 86)
    print("AUDITORIA - ¿cuanto depende el resultado de COMO se agrega?")
    print("=" * 86)
    print("  T1 media cruda | T2 recortar la media | T3 recortar y promediar")
    print("  T4 rangos por condicion  <- el robusto, el arbitro\n")

    for nom, gen in fuentes:
        print(f"  {nom}")
        print(f"    {'senal':<14}{'T1':>10}{'T2':>10}{'T3':>10}{'T4':>10}"
              f"{'  ¿estable?':>14}")
        print("    " + "-" * 68)
        tablas = {t: agregar(gen(), t) for t in ("T1", "T2", "T3", "T4")}
        for sn, fn in SENALES.items():
            vals = []
            for t in ("T1", "T2", "T3", "T4"):
                r, _ = rho_medio(tablas[t], fn)
                vals.append(r)
            signos = {math.copysign(1, v) for v in vals if not math.isnan(v)}
            estable = "si" if len(signos) == 1 else "NO (cambia de signo)"
            print(f"    {sn:<14}" + "".join(f"{v:>+10.4f}" for v in vals)
                  + f"{estable:>14}")
        n = rho_medio(tablas["T4"], SENALES["destino"])[1]
        print(f"    (sitios: {n})\n")

    print("=" * 86)
    print("REGLA: una senal que cambia de signo segun la agregacion NO se afirma.")
    print("El veredicto se lee en T4, que no depende de escalas ni de recortes.")
    print("=" * 86)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
