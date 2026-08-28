"""El termino quimico del escape: DESTINO, no distancia. Siete conjuntos.

QUE SE AFIRMA
-------------
EVEscape (Marks lab, Nature 2023) modela el termino quimico del escape como
DISIMILITUD: cuanto se aleja el residuo mutante del original. Aqui se mide una
alternativa -que lo que manda son las propiedades del residuo al que se LLEGA-
y se comprueba que la bate en las cuatro familias de virus disponibles.

    score = z(hidrofilia del destino) + z(volumen del destino)

Ambas piezas son propiedades del DESTINO, no del cambio. Y son casi
independientes entre si (cada una aguanta al descontar la otra), por eso sumarlas
gana a cualquiera por separado.

COMO SE LLEGO AQUI (y los dos errores propios que hubo que corregir)
--------------------------------------------------------------------
1. Primera medida, solo sobre el RBD de SARS-CoV-2: la hidrofilia del destino
   daba +0.31 frente a +0.05 de la disimilitud. Parecia enorme.
2. Al replicar en gripe, VIH y Zika, NO replicaba. Se publico el negativo.
3. La auditoria (tools/diag_escape_agregacion.py) destapo un error PROPIO: dos
   scripts nuestros agregaban distinto -uno promediaba y recortaba, el otro
   recortaba y promediaba- y daban signos opuestos. En SARS-CoV-2 daba igual
   (su metrica ya es >=0), en los demas cambiaba el signo de TODO. Comparar
   unos con otros bajo esas agregaciones era comparar cosas incomparables.
4. Con la unica agregacion coherente con un analisis de rangos -T4: rangos
   dentro del sitio por anticuerpo, promediados-, la hidrofilia del destino es
   positiva en 5/5, pero pequena fuera de SARS-CoV-2.
5. El diagnostico de POR QUE (tools/diag_escape_por_que_pierde.py) mostro que
   gran parte de la diferencia es TAMANO DE MUESTRA: degradando SARS-CoV-2 de
   3051 anticuerpos a 5, su senal cae de +0.31 a +0.14, hacia los valores de
   los demas conjuntos, que tienen entre 5 y 50.
6. Buscando en el mismo diagnostico aparecio el VOLUMEN del destino, positivo
   en 5/5 y mas fuerte que la hidrofilia en gripe. Sumadas, ganan en todos.

El volumen se eligio DESPUES de ver los datos: eso es un descubrimiento, no una
validacion. Por eso el modelo se CONGELA y se aplica a dos conjuntos retenidos
que no intervinieron en ninguna decision.

DATOS (publicos, laboratorio de Bloom; NO se redistribuyen)
-----------------------------------------------------------
  desarrollo   SARS-CoV-2 RBD          Coronaviridae     ~3000 mAbs y sueros
               gripe H3N2 HA           Orthomyxoviridae  sueros humanos
               gripe H3N2 HA           Orthomyxoviridae  monoclonales
               VIH-1 BG505 Env         Retroviridae      bnAbs
               Zika proteina E         Flaviviridae      monoclonales
  RETENIDOS    VIH-1 Env               Retroviridae      sueros HUMANOS
               VIH-1 Env               Retroviridae      sueros de CONEJO

DISCIPLINA
----------
* Todo se mide DENTRO de cada sitio: el nivel de sitio es tautologico (que los
  sitios antigenicos son los variables se sabe desde los anos 80).
* La agregacion es T4 (rangos), la unica coherente con un analisis de rangos.
* La prueba es bootstrap PAREADO por sitio.
* Cada conjunto se reporta por separado. Nada de promediar para tapar un fallo.
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
# Kyte & Doolittle 1982 (hidropatia)
HID = dict(zip(_AA, [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
                     1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]))
# volumen del residuo (A^3)
VOL = dict(zip(_AA, [88.6, 108.5, 111.1, 138.4, 189.9, 60.1, 153.2, 166.7,
                     168.6, 166.7, 162.9, 114.1, 112.7, 143.8, 173.4, 89.0,
                     116.1, 140.0, 227.8, 193.6]))

# ---- el modelo CONGELADO y su rival ----------------------------------------
COMBO = [lambda w, m: -HID[m], lambda w, m: VOL[m]]      # destino
EVESCAPE = [lambda w, m: abs(HID[m] - HID[w])]           # disimilitud
PARTES = {"hidrofilia destino": [COMBO[0]], "volumen destino": [COMBO[1]]}


def rank(v):
    return np.argsort(np.argsort(np.asarray(v, float))).astype(float)


def spearman(a, b):
    ra, rb = rank(a) - rank(a).mean(), rank(b) - rank(b).mean()
    d = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return float((ra * rb).sum() / d) if d else float("nan")


def z(v):
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / s if s else v * 0


# ------------------------------------------------------------------- lectura
def medidas_sars2():
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


def agregar_rangos(medidas):
    """T4: cada anticuerpo ORDENA las mutaciones de un sitio; se promedian ordenes.

    Es la unica agregacion coherente con un analisis basado en rangos: no
    depende de la escala de la metrica, ni de recortes, ni de que un anticuerpo
    tenga valores mas grandes que otro. Es lo que permite comparar el escape
    fraction de SARS-CoV-2 con el diffsel de los demas.
    """
    porcond = collections.defaultdict(list)
    for c, s, w, m, v in medidas:
        porcond[(c, s, w)].append((m, v))
    acc = collections.defaultdict(list)
    for (c, s, w), lista in porcond.items():
        if len(lista) < 3:
            continue
        q = rank([v for _, v in lista]) / (len(lista) - 1)      # 0..1
        for (m, _), qq in zip(lista, q):
            acc[(s, w, m)].append(float(qq))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def rhos(esc, partes, minimo=8):
    """Un rho por sitio; el efecto del sitio queda eliminado."""
    por = collections.defaultdict(list)
    for (s, w, m) in esc:
        por[(s, w)].append(m)
    out = {}
    for (s, w), muts in por.items():
        if len(muts) < minimo:
            continue
        x = np.sum([z([f(w, m) for m in muts]) for f in partes], axis=0)
        if len(set(x.tolist())) < 3:
            continue
        r = spearman(x, [esc[(s, w, m)] for m in muts])
        if not math.isnan(r):
            out[(s, w)] = r
    return out


def comparar(esc, rng):
    A, B = rhos(esc, COMBO), rhos(esc, EVESCAPE)
    com = sorted(set(A) & set(B))
    if len(com) < 10:
        return None
    d = np.array([A[k] - B[k] for k in com])
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    part = {k: float(np.mean(list(rhos(esc, p).values())))
            for k, p in PARTES.items()}
    return {"n": len(com), "evescape": float(np.mean([B[k] for k in com])),
            "combo": float(np.mean([A[k] for k in com])),
            "d": float(d.mean()), "lo": float(lo), "hi": float(hi), **part}


def fuentes():
    out = []
    if os.path.exists(os.path.join(DATOS, "escape_mut.csv")):
        out.append(("SARS-CoV-2 RBD", "Coronaviridae", "desarrollo",
                    medidas_sars2))
    if os.path.exists(os.path.join(MULTI, "flu_serum.csv")):
        out.append(("gripe H3N2 - sueros", "Orthomyxoviridae", "desarrollo",
                    medidas_flu_serum))
    tabla = (("flu_mab", "gripe H3N2 - mAbs", "Orthomyxoviridae", "desarrollo"),
             ("hiv", "VIH-1 BG505 - bnAbs", "Retroviridae", "desarrollo"),
             ("zika", "Zika proteina E", "Flaviviridae", "desarrollo"),
             ("hiv_sera", "VIH-1 - sueros HUMANOS", "Retroviridae", "RETENIDO"),
             ("hiv_conejo", "VIH-1 - sueros CONEJO", "Retroviridae", "RETENIDO"))
    for tag, nom, fam, rol in tabla:
        d = os.path.join(MULTI, tag)
        if os.path.isdir(d):
            out.append((nom, fam, rol, (lambda dd: lambda: medidas_diffsel(dd))(d)))
    return out


def main():
    fs = fuentes()
    if not fs:
        print(f"No hay datos en {MULTI}. Los descarga "
              "tools/bench_escape_vs_evescape.py y los repos jbloomlab.")
        return 1
    rng = np.random.default_rng(0)
    print("=" * 100)
    print("EL TERMINO QUIMICO DEL ESCAPE: destino, no distancia")
    print("=" * 100)
    print("  score = z(hidrofilia del destino) + z(volumen del destino)")
    print("  rival = |delta hidrofobia|, el termino quimico de EVEscape")
    print("  agregacion T4 (rangos por anticuerpo); todo medido DENTRO del sitio\n")
    print(f"{'conjunto':<24}{'familia':<19}{'rol':<12}{'n':>5}"
          f"{'EVEscape':>10}{'COMBO':>9}{'   diferencia (IC95%)':>28}")
    print("-" * 100)
    gana = tot = ret_ok = ret = 0
    for nom, fam, rol, gen in fs:
        r = comparar(agregar_rangos(gen()), rng)
        if r is None:
            continue
        marca = "SI" if r["lo"] > 0 else ("NO" if r["hi"] < 0 else "--")
        dif = "{:+.4f} [{:+.4f},{:+.4f}] {}".format(r["d"], r["lo"], r["hi"], marca)
        print(f"{nom:<24}{fam:<19}{rol:<12}{r['n']:>5}"
              f"{r['evescape']:>+10.4f}{r['combo']:>+9.4f}{dif:>28}")
        tot += 1
        gana += r["lo"] > 0
        if rol == "RETENIDO":
            ret += 1
            ret_ok += r["lo"] > 0
    print("-" * 100)
    print(f"El COMBO bate al termino de EVEscape con IC limpio en {gana}/{tot}"
          f" conjuntos, incluidos {ret_ok}/{ret} de los RETENIDOS.")
    print("(SI = el IC95% no toca el cero; -- = no concluyente; NO = pierde)")
    print("\nLimites declarados: los dos retenidos son Env de VIH, asi que validan")
    print("generalizacion entre REPERTORIOS de anticuerpos, no entre virus. Y los")
    print("tamanos de efecto son modestos (0.07-0.21): esto ordena mejor que el")
    print("termino quimico de EVEscape, no resuelve el escape.")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
