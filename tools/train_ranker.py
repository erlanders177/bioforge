"""
tools/train_ranker.py
══════════════════════════════════════════════════════════════════════
Entrena el rankeador de mutaciones — la "IA" del predictor.

POR QUÉ UNA LOGÍSTICA Y NO UNA RED: el problema medido no es de capacidad, es de
PESOS. Tenemos 4 señales reales y las combinamos mal — dos fusiones a mano, las dos
PEORES que el mejor eje solo (0.760 y 0.734 vs 0.796). Para aprender 6 pesos con
~500.000 ejemplos, una red es un cañón que además no deja ver dónde disparó. Con 6
parámetros y 500k ejemplos no se puede sobreajustar, y los pesos SE LEEN: si sale
crecimiento≈0, el modelo confirma solo lo que medimos a mano. Eso es ciencia; un
.pt de 3M de parámetros no lo es.
Escalera (solo se sube con el número del anterior delante):
    logística → + interacciones → boosting / MLP diminuto

EL LISTÓN NO ES 0.5: es 0.796 (la mutabilidad sola). Si el modelo no bate al mejor
eje individual, NO APORTA y se dice. Igual que el listón del ranker no era el azar
sino contar (0.669).

ESM-2 NO entra por defecto: está contaminado hasta ~2021 (ver evaluate_ranking.py,
test de fuga: -0.199 mientras los controles siguen planos). Entrenar con él enseñaría
al modelo a fiarse de un feature tramposo y a estrellarse en el futuro real.

DOS EXÁMENES, y el segundo es el que importa:
  1. TEMPORAL   — entrenar en <T, validar en >T (leak-free).
  2. ENTRE VIRUS — entrenar en unos virus, probar en OTRO. El examen del
     agnosticismo, que ni evofr (COVID) ni Łuksza (gripe) hacen.

Salida: pesos en un .npz de unos KB → la inferencia es X@w+b en NumPy puro, sin
dependencias, en el núcleo (NO en bioforge/ai/, que es la zona de deps pesadas).
"""

import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from bioforge.evolution import (  # noqa: E402
    _bin_ids,
    _conservation_table,
    _freqs,
    _loglinear_fit,
    _mutability,
    _mutability_gate,
    _prepare,
)
from bioforge.fetch import fetch_dated_precise  # noqa: E402

_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "bioforge", "ranker")

sys.path.insert(0, "tools")
from evaluate_ranking import _auc, _to_protein  # noqa: E402

ORGANISMS = {
    "H3N2": "Influenza A virus[Organism] AND H3N2 AND hemagglutinin[Title] "
            "AND 1650:1780[SLEN] AND {year}",
    "H1N1": "Influenza A virus[Organism] AND H1N1 AND hemagglutinin[Title] "
            "AND 1650:1780[SLEN] AND {year}",
    "gripeB": "Influenza B virus[Organism] AND hemagglutinin[Title] "
              "AND 1700:1800[SLEN] AND {year}",
}
YEARS = range(2011, 2024)
PER_YEAR = 150
HORIZONS = (1, 2, 4)               # el horizonte entra como FEATURE: un solo modelo
RISE = 0.05                        # que sabe a qué distancia le preguntas → no puede
MINOR = 0.5                        # aprender a persistir
FEATURES = ["frecuencia", "conservacion", "mutabilidad", "crecimiento", "horizonte"]
#                                              ↑ crecimiento = DETECTOR DE MENTIRAS:
#      medido que en solitario es ruido (AUC ~0.42). ABLACIÓN (v8.0, proxy logístico,
#      examen temporal held-out): quitarla mueve el AUC H3N2 +0.000 / H1N1 −0.006 /
#      gripeB +0.007 → neutra (±0.007, nivel de ruido), ni ayuda ni estorba a la
#      generalización. Se mantiene: cambiarla obligaría a re-medir todo sin ganancia.
#      Que sea neutra (no dañina) confirma que el modelo NO se apoya en ella para
#      generalizar. Reejecutar la ablación con el MLP queda para v8.1.


def build_dataset(name, term, per_year=PER_YEAR, cache=True):
    """(X, y, t) — una fila por (mutación candidata, fold, horizonte). Leak-free:
    todo feature sale de bins < k; la etiqueta vive en k+h-1.

    Se cachea el dataset ya construido: lo caro no es entrenar (segundos) sino el MSA
    de ~1500 proteínas de 566 aa. Vamos a iterar sobre features y modelos muchas
    veces; realinear lo mismo cada vez es tirar minutos por iteración."""
    path = os.path.join(_CACHE, f"{name}_{per_year}_{'-'.join(map(str, HORIZONS))}.npz")
    if cache and os.path.exists(path):
        d = np.load(path)
        return d["X"], d["y"], d["t"]
    data = fetch_dated_precise(term, YEARS, per_year=per_year)
    prot = _to_protein([s for s, _ in data])
    keep = [i for i, p in enumerate(prot) if p is not None]
    arr, t, symbols = _prepare([prot[i] for i in keep],
                               [data[i][1] for i in keep], align=True)
    bins, nb = _bin_ids(t, None)
    freq = _freqs(arr, bins, nb, symbols)
    bin_time = np.unique(t)
    cons_tab = _conservation_table(symbols)

    X, y, ts = [], [], []
    for h in HORIZONS:
        for k in range(2, nb - h + 1):
            ftr = freq[:k]
            last = ftr[-1]
            target = freq[k + h - 1]
            if target.sum() == 0:
                continue
            _, slope = _loglinear_fit(ftr, weighted=False)
            conserv = cons_tab[:, last.argmax(axis=0)]
            mut = np.broadcast_to(_mutability_gate(_mutability(ftr)), last.shape)
            cand = last < MINOR
            si, li = np.nonzero(cand)
            X.append(np.column_stack([last[si, li], conserv[si, li], mut[si, li],
                                      slope[si, li], np.full(si.size, float(h))]))
            y.append(((target - last) >= RISE)[si, li])
            ts.append(np.full(si.size, bin_time[k + h - 1]))
    X, y, t = (np.concatenate(X), np.concatenate(y).astype(np.float64),
               np.concatenate(ts))
    if cache:
        os.makedirs(_CACHE, exist_ok=True)
        np.savez_compressed(path, X=X, y=y, t=t)
    return X, y, t


def fit_logistic(X, y, *, l2=1e-3, iters=25):
    """Logística por Newton (IRLS) — NumPy puro, cero deps. Convexa → óptimo único.

    Newton usa la curvatura (Hessiana), así que converge en ~10 iteraciones en vez de
    las miles del descenso de gradiente (que tardaba 7 min sobre 600k filas). La
    Hessiana es (k+1)×(k+1) — con ~20 features, invertirla es instantáneo. Se
    estandariza para que los pesos sean COMPARABLES (legibles como ciencia)."""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = np.column_stack([(X - mu) / sd, np.ones(X.shape[0])]).astype(np.float64)
    y = y.astype(np.float64)
    theta = np.zeros(Z.shape[1])
    reg = l2 * np.eye(Z.shape[1])
    reg[-1, -1] = 0.0                                    # no regularizar el sesgo
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Z @ theta)))
        W = np.clip(p * (1.0 - p), 1e-6, None)          # pesos IRLS
        g = Z.T @ (p - y) + l2 * theta                  # gradiente
        H = (Z * W[:, None]).T @ Z + reg                # Hessiana (k+1)×(k+1)
        step = np.linalg.solve(H, g)
        theta -= step
        if np.abs(step).max() < 1e-8:                   # convergido → parar
            break
    return theta[:-1], float(theta[-1]), mu, sd


def score(X, w, b, mu, sd):
    return (X - mu) / sd @ w + b


# ── MLP en NumPy PURO (sin torch): el modelo final se entrena y se sirve sin deps ──
#
# Torch fue el andamio para EXPLORAR (tamaños, semillas). Fijada la arquitectura
# (2×64, la del punto dulce medido), el entrenador definitivo es NumPy: forward +
# backprop (regla de la cadena) + Adam, ~40 líneas. Así el proyecto entero vuelve a
# cero dependencias, como el engine.dll: exploramos con lo cómodo, entregamos lo
# autocontenido.

def _he(shape, rng):
    return rng.standard_normal(shape) * np.sqrt(2.0 / shape[0])   # init de He (ReLU)


def fit_mlp(X, y, *, hidden=64, epochs=400, lr=2e-3, wd=1e-4, seed=0):
    """MLP de 2 capas ocultas (ReLU) por Adam, NumPy puro. Devuelve el dict de pesos
    estandarizados listos para la inferencia (mismos que score_mlp espera)."""
    rng = np.random.default_rng(seed)
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    Z = ((X - mu) / sd).astype(np.float64)
    yv = y.astype(np.float64)[:, None]
    pos_w = float((y == 0).sum() / max((y == 1).sum(), 1))     # pesar la clase rara
    P = [_he((Z.shape[1], hidden), rng), np.zeros(hidden),
         _he((hidden, hidden), rng), np.zeros(hidden),
         _he((hidden, 1), rng), np.zeros(1)]
    M = [np.zeros_like(p) for p in P]                          # momentos de Adam
    V = [np.zeros_like(p) for p in P]
    b1, b2, eps = 0.9, 0.999, 1e-8
    n = Z.shape[0]
    for step in range(1, epochs + 1):
        h1 = np.maximum(Z @ P[0] + P[1], 0.0)                 # forward
        h2 = np.maximum(h1 @ P[2] + P[3], 0.0)
        logit = h2 @ P[4] + P[5]
        p = 1.0 / (1.0 + np.exp(-logit))
        w = np.where(yv > 0, pos_w, 1.0)                      # BCE con peso de clase
        dlogit = w * (p - yv) / n                             # backprop
        g4 = h2.T @ dlogit;               g5 = dlogit.sum(0)
        dh2 = (dlogit @ P[4].T) * (h2 > 0)
        g2 = h1.T @ dh2;                  g3 = dh2.sum(0)
        dh1 = (dh2 @ P[2].T) * (h1 > 0)
        g0 = Z.T @ dh1;                   g1 = dh1.sum(0)
        grads = [g0, g1, g2, g3, g4, g5]
        for i, g in enumerate(grads):
            g = g + wd * P[i]                                 # weight decay
            M[i] = b1 * M[i] + (1 - b1) * g
            V[i] = b2 * V[i] + (1 - b2) * g * g
            mhat = M[i] / (1 - b1 ** step)
            vhat = V[i] / (1 - b2 ** step)
            P[i] -= lr * mhat / (np.sqrt(vhat) + eps)
    return {"W1": P[0], "b1": P[1], "W2": P[2], "b2": P[3],
            "W3": P[4], "b3": P[5], "mu": mu, "sd": sd}


def score_mlp(X, P):
    """Inferencia del MLP: forward puro en NumPy (h→relu→h→relu→logit)."""
    Z = (X - P["mu"]) / P["sd"]
    h1 = np.maximum(Z @ P["W1"] + P["b1"], 0.0)
    h2 = np.maximum(h1 @ P["W2"] + P["b2"], 0.0)
    return (h2 @ P["W3"] + P["b3"]).ravel()


def pairs(n):
    return [(i, j) for i in range(n) for j in range(i, n)]


def expand(X):
    """Añade los productos de pares — las interacciones a mano (medido: cons*muta es
    el 2º peso). Deja que la logística use no-linealidad sin dejar de ser legible."""
    cols = [X]
    for i, j in pairs(X.shape[1]):
        cols.append((X[:, i] * X[:, j])[:, None])
    return np.hstack(cols)


def subsample(X, y, neg_per_pos=60, seed=0):
    """Todos los positivos + una submuestra de negativos. El 99,8% de las filas son
    negativas; para el AUC de un ranker no hacen falta los 4,7M — mantener ~60 negativos
    por positivo conserva el orden y entrena en segundos (evita el swap de float64)."""
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    take = rng.choice(neg, min(len(neg), len(pos) * neg_per_pos), replace=False)
    idx = rng.permutation(np.concatenate([pos, take]))
    return X[idx].astype(np.float32), y[idx].astype(np.float32)


def train_and_save(path="bioforge/data/ranker_weights.npz", hidden=64):
    """Entrena el modelo distribuido (MLP 2×64, NumPy puro) sobre los 3 virus y
    versiona los pesos (.npz). Tamaño 64 = el punto dulce medido (barrido de neuronas:
    el techo se aplana ahí y 128 empieza a sobreajustar en H3N2)."""
    Xs, ys = [], []
    for v, term in ORGANISMS.items():
        X, y, _ = build_dataset(v, term)
        Xs.append(X)
        ys.append(y)
    X, y = np.vstack(Xs), np.concatenate(ys)
    Xsub, ysub = subsample(X, y)                  # todos los positivos + 60x negativos
    P = fit_mlp(Xsub, ysub, hidden=hidden)
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, features=np.array(FEATURES), hidden=hidden, **P)
    a = _auc(score_mlp(Xsub, P), ysub.astype(bool))
    n_params = sum(P[k].size for k in ("W1", "b1", "W2", "b2", "W3", "b3"))
    print(f"entrenado (MLP 2x{hidden}): {X.shape[0]:,} ejemplos → submuestra "
          f"{Xsub.shape[0]:,} · {n_params:,} params · AUC {a:.4f} · {path}")
    _print_detectors(P)
    return P


def _print_detectors(P, top=4):
    """Los DETECTORES de la primera capa: qué combinación de las 5 features nombradas
    dispara cada neurona oculta, ponderada por su peso de salida. Interpretabilidad
    parcial del MLP (lo que se puede leer aunque los valores bailen entre semillas)."""
    contrib = np.abs(P["W2"]).sum(1) if P["W2"].ndim == 2 else np.abs(P["W2"])
    orden = np.argsort(-contrib)[:top]
    print(f"  detectores más influyentes (capa 1, sobre {FEATURES}):")
    for k in orden:
        w = P["W1"][:, k]
        terms = " ".join(f"{f[:4]}{w[i]:+.2f}" for i, f in enumerate(FEATURES))
        print(f"    neurona {k:>2}: {terms}")


def report(tag, s, y, base):
    a = _auc(s, y.astype(bool))
    marca = "✓ APORTA" if a > base + 0.005 else "✗ NO APORTA (no bate al mejor eje)"
    print(f"    {tag:28s} AUC = {a:.3f}   (listón {base:.3f})  {marca}")
    return a


def _fit(X, y):
    """Ajusta el modelo distribuido (con interacciones) sobre una submuestra."""
    Xs, ys = subsample(X, y)
    return fit_logistic(expand(Xs), ys)


def main():
    print("RANKEADOR DE MUTACIONES — logística + interacciones (Newton, NumPy puro)")
    print("El listón NO es 0.5: es el MEJOR EJE INDIVIDUAL. Si no lo bate, no aporta.\n")
    ds = {}
    for name, term in ORGANISMS.items():
        X, y, t = build_dataset(name, term)
        ds[name] = (X, y, t)
        print(f"  {name}: {X.shape[0]:,} ejemplos · {int(y.sum()):,} suben "
              f"({y.mean():.1%})")
    print()

    # ── EXAMEN 1: TEMPORAL (entrenar en el pasado, validar en el futuro) ─────
    print("[EXAMEN 1] TEMPORAL — entrenar <2020, validar >2020")
    for name, (X, y, t) in ds.items():
        tr, te = t < 2020, t >= 2020
        if te.sum() < 100 or y[te].sum() < 5:
            continue
        w, b, mu, sd = _fit(X[tr], y[tr])
        base = max(_auc(X[te][:, i], y[te].astype(bool)) for i in range(X.shape[1]))
        for i, f in enumerate(FEATURES):        # cada eje solo, para ver el listón
            print(f"    {'  eje ' + f:28s} AUC = {_auc(X[te][:, i], y[te].astype(bool)):.3f}")
        report(f"  {name} MODELO", score(expand(X[te]), w, b, mu, sd), y[te], base)
        print()

    # ── EXAMEN 2: ENTRE VIRUS (el del agnosticismo) ──────────────────────────
    print("[EXAMEN 2] ENTRE VIRUS — entrenar en dos, predecir en el TERCERO")
    print("(ni evofr —COVID— ni Łuksza —gripe— cruzan de organismo)")
    for held in ds:
        tr_names = [n for n in ds if n != held]
        Xtr = np.concatenate([ds[n][0] for n in tr_names])
        ytr = np.concatenate([ds[n][1] for n in tr_names])
        Xte, yte, _ = ds[held]
        w, b, mu, sd = _fit(Xtr, ytr)
        base = max(_auc(Xte[:, i], yte.astype(bool)) for i in range(Xte.shape[1]))
        report(f"  {'+'.join(tr_names)} → {held}",
               score(expand(Xte), w, b, mu, sd), yte, base)
    print()

    train_and_save()               # regenera los pesos versionados que usa el paquete


if __name__ == "__main__":
    main()
