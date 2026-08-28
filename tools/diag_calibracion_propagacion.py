"""¿Podemos dar un "probabilidad de propagarse = Y%" honesto?

LA PREGUNTA
-----------
Una herramienta que dice "esta variante tiene un 37% de propagarse" solo vale si
ese 37% esta CALIBRADO: de cada 100 mutaciones a las que les da 37%, deben
propagarse unas 37. Si la curva de calibracion sale plana, el numero es
decorativo y NO se puede mostrar, por mucho que el usuario lo pida.

Ya sabemos que el escape de laboratorio no ordena las mutaciones que triunfaron
(percentil 0.44 sobre 22 mutaciones reales de SARS-CoV-2, por debajo del azar).
Aqui se comprueba en grande, sobre 33 clados reales.

EL CONFUSOR QUE HAY QUE MATAR PRIMERO
-------------------------------------
Una mutacion vista el primer mes de la ventana acumula recuento todo el periodo;
una vista el ultimo, no. Medido antes: 'first_seen' solo correlaciona 0.45 con
el recuento. Por eso "propagarse" se define DENTRO del mismo mes de aparicion:
se marca como propagada la que entra en el decil superior de recuento ENTRE LAS
QUE APARECIERON A LA VEZ. Asi el reloj deja de contar.

QUE SE MIDE
-----------
Curva de calibracion: se ordenan las mutaciones por score, se parten en deciles
y se mira que fraccion se propago de verdad en cada decil. Si el primero y el
ultimo dan lo mismo, el score no informa y el porcentaje no se puede ofrecer.
"""
import collections
import csv
import datetime as dt
import math
import os
import sys

import numpy as np

FC = os.path.join(os.environ.get("TEMP", "."), "everest_benchmark", "forecast")
_AA = "ACDEFGHIKLMNPQRSTVWY"
HID = dict(zip(_AA, [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
                     1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]))
VOL = dict(zip(_AA, [88.6, 108.5, 111.1, 138.4, 189.9, 60.1, 153.2, 166.7,
                     168.6, 166.7, 162.9, 114.1, 112.7, 143.8, 173.4, 89.0,
                     116.1, 140.0, 227.8, 193.6]))


def z(v):
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / s if s else v * 0


def cargar(path):
    """(mutacion, recuento, mes de aparicion) de un clado."""
    out = []
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return out
    col = "count" if "count" in rows[0] else "test_count"
    for r in rows:                                       # bucle por MUTACION
        m = r["mutation"]
        if len(m) < 3 or m[0] not in _AA or m[-1] not in _AA:
            continue
        try:
            int(m[1:-1])
            d = dt.date.fromisoformat(r["first_seen"][:10])
            c = int(r[col])
        except (ValueError, KeyError):
            continue
        out.append((m, c, d.year * 12 + d.month))
    return out


def marcar_propagadas(filas, decil=0.9):
    """Propagada = decil superior de recuento ENTRE LAS APARECIDAS EL MISMO MES.

    Asi se elimina la ventaja de haber aparecido antes, que es el confusor.
    """
    pormes = collections.defaultdict(list)
    for i, (_, c, mes) in enumerate(filas):
        pormes[mes].append((c, i))
    y = np.zeros(len(filas), bool)
    for mes, lista in pormes.items():
        if len(lista) < 20:                              # muestra insuficiente
            continue
        corte = np.quantile([c for c, _ in lista], decil)
        for c, i in lista:
            if c > corte:
                y[i] = True
    return y


def calibracion(score, y, n=10):
    """Fraccion realmente propagada en cada decil de score."""
    orden = np.argsort(score)
    trozos = np.array_split(orden, n)
    return [(float(score[t].mean()), float(y[t].mean()), len(t)) for t in trozos]


def main():
    if not os.path.isdir(FC):
        print(f"Faltan los clados de pronostico en {FC}.")
        return 1
    rng = np.random.default_rng(0)

    muts, cnt, mes, virus = [], [], [], []
    for v in sorted(os.listdir(FC)):
        for f in sorted(os.listdir(os.path.join(FC, v))):
            filas = cargar(os.path.join(FC, v, f))
            if len(filas) < 200:
                continue
            y = marcar_propagadas(filas)
            for (m, c, mm), yy in zip(filas, y):
                muts.append(m)
                cnt.append(yy)
                mes.append(mm)
                virus.append(v)
    y = np.array(cnt, bool)
    print("=" * 80)
    print("¿SE PUEDE DAR UN 'PROBABILIDAD DE PROPAGARSE = Y%' CALIBRADO?")
    print("=" * 80)
    print(f"  {len(muts):,} mutaciones de 33 clados reales "
          f"({len(set(virus))} virus).")
    print(f"  'Propagada' = decil superior de recuento ENTRE LAS APARECIDAS EL")
    print(f"  MISMO MES (asi el reloj no cuenta): {y.mean():.1%} lo son.\n")

    combo = z([-HID[m[-1]] for m in muts]) + z([VOL[m[-1]] for m in muts])
    señales = {
        "nuestro eje B (destino)": combo,
        "disimilitud (EVEscape)": z([abs(HID[m[-1]] - HID[m[0]]) for m in muts]),
        "azar (control)": rng.random(len(muts)),
    }
    for nom, sc in señales.items():
        print(f"  {nom}")
        print(f"    {'decil':<8}{'score medio':>14}{'% propagadas':>15}{'n':>9}")
        print("    " + "-" * 46)
        cal = calibracion(np.asarray(sc, float), y)
        for i, (s, p, n) in enumerate(cal, 1):
            print(f"    {i:<8}{s:>+14.3f}{p:>14.1%}{n:>9,}")
        lo, hi = cal[0][1], cal[-1][1]
        print(f"    ultimo decil / primero: {hi/lo if lo else float('nan'):.2f}x"
              f"   (1.00x = el score NO informa)\n")

    print("=" * 80)
    print("REGLA: si la curva es plana, el porcentaje NO se muestra. Un numero")
    print("inventado con dos decimales es peor que no dar numero.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
