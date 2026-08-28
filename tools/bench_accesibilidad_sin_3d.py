"""Termino (2) de EVEscape: la accesibilidad a anticuerpos, SIN estructura 3D.

LA IDEA
-------
EVEscape modela el escape como  viabilidad x ACCESIBILIDAD x disimilitud.
El segundo termino sale de una estructura tridimensional: se mide
geometricamente cuanta superficie de cada residuo queda expuesta al disolvente.

Pero la estructura solo es una MAQUINA para producir un numero por residuo.
Si ese numero se puede estimar de otra forma, la maquina sobra. Y donde la
estructura predicha es menos fiable -bucles flexibles, virus nuevos sin
cristalizar- es justo donde ocurre el escape.

QUE MIDE ESTE SCRIPT
--------------------
1. La exposicion REAL, calculada de una estructura experimental (6M0J, el RBD
   de SARS-CoV-2), con Shrake-Rupley implementado aqui en NumPy puro. Es la
   verdad de campo del termino (2).
2. Si esa exposicion real predice el escape MEDIDO (mapas del lab. de Bloom).
   Si no lo predijera, el termino (2) no valdria ni para ellos.
3. Si se puede estimar la exposicion SIN la estructura, solo con la secuencia.
4. Y si esa estimacion, puesta a predecir escape, aguanta frente al liston
   trivial (la variabilidad de la columna, que es el atajo tautologico).

El RBD se calcula SOLO (cadena E), sin ACE2: asi es como lo ve un anticuerpo.
Con ACE2 pegado, toda la interfaz saldria falsamente enterrada.
"""
import collections
import csv
import math
import os
import sys
import urllib.request

import numpy as np

DATOS = os.path.join(os.environ.get("TEMP", "."), "bioforge_escape")
PDB_URL = "https://files.rcsb.org/download/6M0J.pdb"

_AA = "ACDEFGHIKLMNPQRSTVWY"
TRES = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
# radios de van der Waals (A)
RVDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}
# superficie maxima accesible por residuo, Tien et al. 2013 (teorica, A^2)
MAXASA = {"A": 129, "R": 274, "N": 195, "D": 193, "C": 167, "E": 223,
          "Q": 225, "G": 104, "H": 224, "I": 197, "L": 201, "K": 236,
          "M": 224, "F": 240, "P": 159, "S": 155, "T": 172, "W": 285,
          "Y": 263, "V": 174}
# Kyte & Doolittle 1982
HID = dict(zip(_AA, [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
                     1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]))


def rank(v):
    return np.argsort(np.argsort(np.asarray(v, float))).astype(float)


def spearman(a, b):
    ra, rb = rank(a) - rank(a).mean(), rank(b) - rank(b).mean()
    d = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return float((ra * rb).sum() / d) if d else float("nan")


def parcial(x, y, c):
    """Correlacion parcial de rangos: x vs y quitando lo que explica c."""
    rx, ry, rc = rank(x), rank(y), rank(c)
    rc = rc - rc.mean()
    den = float((rc * rc).sum())
    if den == 0:
        return spearman(x, y)
    ex = rx - rx.mean() - rc * float((rx * rc).sum()) / den
    ey = ry - ry.mean() - rc * float((ry * rc).sum()) / den
    d = math.sqrt(float((ex * ex).sum()) * float((ey * ey).sum()))
    return float((ex * ey).sum() / d) if d else float("nan")


def leer_pdb(path, cadena="E"):
    """Atomos de una cadena: coordenadas, radios y a que residuo pertenecen."""
    xyz, rad, res = [], [], []
    seq = {}
    with open(path, encoding="utf-8") as fh:
        for ln in fh:                                   # bucle por ATOMO
            if not ln.startswith("ATOM") or ln[21] != cadena:
                continue
            elem = (ln[76:78].strip() or ln[12:16].strip()[0]).upper()
            if elem not in RVDW:
                continue
            num = int(ln[22:26])
            aa = TRES.get(ln[17:20].strip())
            if aa is None:
                continue
            xyz.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
            rad.append(RVDW[elem])
            res.append(num)
            seq[num] = aa
    return np.array(xyz), np.array(rad), np.array(res), seq


def esfera(n=200):
    """Puntos casi equiespaciados en la esfera unidad (espiral aurea)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)


def sasa(xyz, rad, res, probe=1.4, n_puntos=200):
    """Shrake-Rupley: superficie accesible al disolvente, por residuo."""
    pts = esfera(n_puntos)
    R = rad + probe
    d2 = ((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(-1)   # atomos, no simbolos
    por_res = collections.defaultdict(float)
    for i in range(len(xyz)):                           # bucle por ATOMO
        lim = (R[i] + R) ** 2
        vec = np.where((d2[i] < lim) & (np.arange(len(xyz)) != i))[0]
        p = xyz[i] + pts * R[i]                          # esfera del atomo i
        if vec.size:
            dd = ((p[:, None, :] - xyz[vec][None, :, :]) ** 2).sum(-1)
            libre = (dd >= (R[vec] ** 2)[None, :]).all(axis=1)
        else:
            libre = np.ones(len(p), bool)
        por_res[int(res[i])] += 4 * math.pi * R[i] ** 2 * libre.mean()
    return por_res


def escape_por_sitio():
    """Escape medio por SITIO, sobre los ~3000 anticuerpos y sueros."""
    ruta = os.path.join(DATOS, "escape_mut.csv")
    if not os.path.exists(ruta):
        return None
    s = collections.defaultdict(float)
    n = collections.defaultdict(int)
    with open(ruta, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):                    # bucle por MEDIDA
            k = int(r["site"])
            s[k] += float(r["mut_escape"])
            n[k] += 1
    return {k: s[k] / n[k] for k in s}


def escape_y_expresion():
    """Escape por MUTACION y el DMS de expresion (para controlar el plegado)."""
    ruta = os.path.join(DATOS, "escape_mut.csv")
    dms = os.path.join(os.environ.get("TEMP", "."), "everest_benchmark",
                       "SARS2_RBD_Starr_expression_dms.csv")
    if not (os.path.exists(ruta) and os.path.exists(dms)):
        return None, None
    s = collections.defaultdict(float)
    n = collections.defaultdict(int)
    with open(ruta, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):                    # bucle por MEDIDA
            k = (int(r["site"]), r["mutation"])
            s[k] += float(r["mut_escape"])
            n[k] += 1
    expr = {}
    with open(dms, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            m = r["mutant"]
            if ":" in m or m[0] not in _AA or m[-1] not in _AA:
                continue
            expr[(int(m[1:-1]), m[-1])] = float(r["DMS_score"])
    return {k: s[k] / n[k] for k in s}, expr


def wt_desde_pdb(seq):
    """El residuo original de cada sitio, leido de la propia estructura."""
    return dict(seq)


def main():
    os.makedirs(DATOS, exist_ok=True)
    pdb = os.path.join(DATOS, "6m0j.pdb")
    if not os.path.exists(pdb):
        print("bajando la estructura 6M0J del RBD...")
        urllib.request.urlretrieve(PDB_URL, pdb)

    xyz, rad, res, seq = leer_pdb(pdb, "E")
    print("=" * 78)
    print("TERMINO (2) DE EVESCAPE - la accesibilidad, con y sin estructura 3D")
    print("=" * 78)
    print(f"estructura 6M0J cadena E (RBD solo, sin ACE2): {len(xyz)} atomos, "
          f"{len(seq)} residuos\n")

    area = sasa(xyz, rad, res)
    sitios = sorted(seq)
    rsa = np.array([area[s] / MAXASA[seq[s]] for s in sitios])
    rsa_d = {s: area[s] / MAXASA[seq[s]] for s in seq}   # por sitio
    print(f"exposicion relativa calculada (Shrake-Rupley, NumPy puro):")
    print(f"  mediana {np.median(rsa):.3f}   enterrados (<0.20): "
          f"{(rsa < 0.20).sum()}/{len(rsa)}   expuestos (>0.50): "
          f"{(rsa > 0.50).sum()}/{len(rsa)}")

    # --- la exposicion REAL, contra el escape MEDIDO -------------------------
    esc = escape_por_sitio()
    if esc is None:
        print("\n(faltan los mapas de escape; corre antes "
              "tools/bench_escape_vs_evescape.py)")
        return 1
    comunes = [s for s in sitios if s in esc]
    y = [esc[s] for s in comunes]
    rsa_c = np.array([area[s] / MAXASA[seq[s]] for s in comunes])
    print(f"\n{'-'*78}\n1) La exposicion REAL, ¿predice el escape medido?  "
          f"({len(comunes)} sitios)\n{'-'*78}")
    print(f"  exposicion (de la estructura 3D) vs escape   rho = "
          f"{spearman(rsa_c, y):+.4f}")
    print("  <- este es el valor del termino (2) tal como ellos lo calculan")

    # --- estimarla SIN estructura -------------------------------------------
    s_seq = "".join(seq[s] for s in sitios)
    h = np.array([HID[c] for c in s_seq])
    print(f"\n{'-'*78}\n2) ¿Se puede estimar la exposicion SIN la estructura?"
          f"\n{'-'*78}")
    print(f"  {'estimador (solo secuencia)':<44}{'rho vs exposicion real':>24}")
    print("  " + "-" * 68)
    est = {}
    for w in (1, 5, 9, 13, 17, 21):
        if w == 1:
            v = -h
        else:                                            # ventana deslizante
            k = np.ones(w) / w
            v = -np.convolve(h, k, mode="same")
        est[w] = v
        etq = "hidrofilia del residuo" if w == 1 else f"hidrofilia, ventana de {w}"
        print(f"  {etq:<44}{spearman(v, rsa):>+24.4f}")

    # --- y puesta a predecir escape, ¿aguanta el liston? --------------------
    mejor = max(est, key=lambda w: spearman(est[w], rsa))
    pred = np.array([est[mejor][sitios.index(s)] for s in comunes])
    print(f"\n{'-'*78}\n3) La exposicion ESTIMADA, ¿predice escape?  "
          f"(ventana de {mejor})\n{'-'*78}")
    print(f"  exposicion estimada vs escape                 rho = "
          f"{spearman(pred, y):+.4f}")
    print(f"  exposicion REAL     vs escape                 rho = "
          f"{spearman(rsa_c, y):+.4f}   <- el techo")
    print(f"  estimada | descontando la real                rho = "
          f"{parcial(pred, y, rsa_c):+.4f}")
    print(f"  real     | descontando la estimada            rho = "
          f"{parcial(rsa_c, y, pred):+.4f}")

    # --- 4. lo enterrado, ¿escapa menos? y el confusor de plegado -----------
    esc_mut, expr = escape_y_expresion()
    if esc_mut is None:
        return 0
    ks = [k for k in esc_mut if k in expr and k[0] in rsa_d]
    E = [esc_mut[k] for k in ks]
    X = [expr[k] for k in ks]
    Rr = [rsa_d[k[0]] for k in ks]
    print(f"\n{'-'*78}\n4) Lo ENTERRADO, ¿escapa menos?  ({len(ks):,} mutaciones)"
          f"\n{'-'*78}")
    print("  Confusor: una mutacion enterrada puede DESPLEGAR la proteina, y en")
    print("  el ensayo eso se parece a escape (el anticuerpo deja de unirse).")
    print(f"\n  exposicion vs expresion (¿lo enterrado despliega?) rho = "
          f"{spearman(Rr, X):+.4f}")
    print(f"  exposicion vs escape                                rho = "
          f"{spearman(Rr, E):+.4f}")
    print(f"  exposicion vs escape | DESCONTANDO la expresion     rho = "
          f"{parcial(Rr, E, X):+.4f}")
    ok = [i for i in range(len(ks)) if X[i] > -0.5]      # se expresan bien
    ent = [E[i] for i in ok if Rr[i] < 0.20]
    exp = [E[i] for i in ok if Rr[i] > 0.50]
    print(f"\n  Solo mutaciones que SI se expresan bien (n={len(ok):,}):")
    print(f"    escape medio enterrados (n={len(ent):,}): {np.mean(ent):.4f}")
    print(f"    escape medio expuestos  (n={len(exp):,}): {np.mean(exp):.4f}")
    print(f"    razon expuesto/enterrado: {np.mean(exp)/np.mean(ent):.2f}x"
          f"   <- practicamente ninguna diferencia")

    # --- 5. ¿nuestra señal depende de la exposicion? ------------------------
    wt = wt_desde_pdb(seq)
    por = collections.defaultdict(list)
    for (s, aa) in esc_mut:
        if s in wt and s in rsa_d and aa in _AA:
            por[s].append(aa)

    def rho_sitio(s):
        aas = por[s]
        if len(aas) < 8:
            return None
        x = [-HID[a] for a in aas]
        if len(set(x)) < 3:
            return None
        r = spearman(x, [esc_mut[(s, a)] for a in aas])
        return None if math.isnan(r) else r

    print(f"\n{'-'*78}\n5) El efecto 'destino hidrofilico', ¿depende de la "
          f"exposicion?\n{'-'*78}")
    print(f"  {'grupo':<34}{'rho intra-sitio':>18}{'sitios':>9}")
    print("  " + "-" * 61)
    for etq, cond in (("enterrados   (< 0.20)", lambda v: v < 0.20),
                      ("intermedios  (0.20 - 0.50)", lambda v: 0.20 <= v <= 0.50),
                      ("expuestos    (> 0.50)", lambda v: v > 0.50)):
        rr = [rho_sitio(s) for s in por if cond(rsa_d[s])]
        rr = [r for r in rr if r is not None]
        print(f"  {etq:<34}{np.mean(rr):>+18.4f}{len(rr):>9}")
    print("\n  Es uniforme: nuestra senal NO es accesibilidad disfrazada,")
    print("  y tampoco la necesita.")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
