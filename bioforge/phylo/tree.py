"""
bioforge/phylo/tree.py — árboles evolutivos a partir de una matriz de distancias.

Un **árbol filogenético** resume quién se parece más a quién y en qué orden se
fueron separando los linajes. Aquí se construye por métodos de *distancia*, que
son los que caben en un portátil:

* **Neighbor-Joining** (Saitou & Nei, 1987) — el estándar de la familia. No
  supone que todo evolucione al mismo ritmo, así que aguanta linajes que mutan
  más deprisa que otros. Produce un árbol **sin raíz**.
* **UPGMA** — agrupamiento por media. Más simple, pero **sí** supone ritmo
  constante (*reloj molecular*); si esa suposición falla, el árbol sale torcido.
  Se incluye porque es el que se explica en clase y sirve de contraste.

Y, sobre todo, **soporte por bootstrap**: reconstruir el árbol cientos de veces
remuestreando columnas del alineamiento y contar cuántas veces reaparece cada
agrupación. Un árbol sin soporte es una opinión; con soporte, es una medida. Una
rama con 95 % es sólida; con 40 %, los datos no la sostienen y no hay que
contarla como resultado.

Reglas de oro
-------------
No hay bucles por símbolo: el trabajo por columna vive en ``distance.py``
(vectorizado con matmuls). Aquí los bucles son **por nodo** (n−3 uniones) y **por
réplica** de bootstrap, igual que el encadenado del mapeador itera por ancla.
"""

from __future__ import annotations

from typing import Iterable, NamedTuple, Optional, Sequence

import numpy as np

from bioforge.core.biocore import SequenceValueError
from bioforge.phylo.distance import DistanceMatrix, distance_matrix


class Clade:
    """Un nodo del árbol: una hoja con nombre, o un grupo con hijos."""

    __slots__ = ("name", "length", "children", "support")

    def __init__(self, name: str = "", length: float = 0.0,
                 children: Optional[list["Clade"]] = None,
                 support: Optional[float] = None) -> None:
        self.name = name
        self.length = float(length)
        self.children: list[Clade] = children if children is not None else []
        self.support = support

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def leaves(self) -> list[str]:
        """Nombres de todas las hojas por debajo de este nodo."""
        if self.is_leaf:
            return [self.name]
        out: list[str] = []
        pila = [self]
        while pila:                                  # recorrido iterativo: sin límite de recursión
            nodo = pila.pop()
            if nodo.is_leaf:
                out.append(nodo.name)
            else:
                pila.extend(nodo.children)
        return out

    def __repr__(self) -> str:                       # pragma: no cover - cosmético
        if self.is_leaf:
            return f"Clade({self.name!r}, {self.length:.4f})"
        return f"Clade({len(self.children)} hijos, {len(self.leaves())} hojas)"


class Tree(NamedTuple):
    """Un árbol evolutivo construido.

    Attributes
    ----------
    root:
        Nodo raíz. En Neighbor-Joining el árbol es **sin raíz**: la raíz es solo
        un punto de dibujo (una trifurcación), no un ancestro real.
    method:
        ``"nj"`` o ``"upgma"``.
    rooted:
        Si la raíz tiene significado biológico (UPGMA sí, NJ no).
    n_leaves:
        Número de secuencias.
    """

    root: Clade
    method: str
    rooted: bool
    n_leaves: int

    def leaves(self) -> list[str]:
        return self.root.leaves()

    def newick(self, *, decimals: int = 6, support: bool = True) -> str:
        """Serializa en formato **Newick**, el estándar del campo.

        Lo leen MEGA, FigTree, iTOL, Dendroscope, ete3, Biopython… así que el
        árbol se puede abrir en cualquier visor del mundo.
        """
        def render(nodo: Clade, raiz: bool = False) -> str:
            if nodo.is_leaf:
                nom = nodo.name.replace(" ", "_").replace(":", "_") \
                               .replace(",", "_").replace("(", "_").replace(")", "_")
                return f"{nom}:{nodo.length:.{decimals}f}"
            dentro = ",".join(render(h) for h in nodo.children)
            etiqueta = ""
            if support and nodo.support is not None:
                etiqueta = f"{nodo.support:.0f}"
            if raiz:
                return f"({dentro}){etiqueta}"
            return f"({dentro}){etiqueta}:{nodo.length:.{decimals}f}"

        return render(self.root, raiz=True) + ";"

    def to_dict(self) -> dict:
        """Estructura anidada simple (para dibujar el árbol en la interfaz)."""
        def conv(nodo: Clade) -> dict:
            d: dict = {"name": nodo.name, "length": round(nodo.length, 6)}
            if nodo.support is not None:
                d["support"] = round(nodo.support, 1)
            if nodo.children:
                d["children"] = [conv(h) for h in nodo.children]
            return d
        return conv(self.root)

    def __repr__(self) -> str:                       # pragma: no cover - cosmético
        return (f"Tree({self.method.upper()}, {self.n_leaves} hojas, "
                f"{'con raíz' if self.rooted else 'sin raíz'})")


def _matriz_valida(dm: DistanceMatrix) -> tuple[np.ndarray, list[str]]:
    D = np.asarray(dm.matrix, dtype=np.float64).copy()
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise SequenceValueError("la matriz de distancias debe ser cuadrada.")
    if D.shape[0] != len(dm.names):
        raise SequenceValueError(
            f"la matriz es {D.shape[0]}×{D.shape[0]} pero hay {len(dm.names)} nombres.")
    if D.shape[0] < 3:
        raise SequenceValueError(
            f"hacen falta al menos 3 secuencias para construir un árbol "
            f"(hay {D.shape[0]}).")
    return D, list(dm.names)


def _sin(D: np.ndarray, i: int, j: int, nueva: np.ndarray) -> np.ndarray:
    """Quita las filas/columnas i,j y añade una nueva con distancias ``nueva``."""
    keep = [k for k in range(D.shape[0]) if k not in (i, j)]
    D2 = D[np.ix_(keep, keep)]
    col = nueva[keep]
    D2 = np.vstack([np.hstack([D2, col[:, None]]),
                    np.append(col, 0.0)[None, :]])
    return D2


def neighbor_joining(dm: DistanceMatrix) -> Tree:
    """Construye el árbol por **Neighbor-Joining**.

    En cada paso se une la pareja que minimiza

        ``Q[i,j] = (n−2)·d[i,j] − Σd[i] − Σd[j]``

    que no es simplemente «los dos más cercanos»: descuenta lo lejos que está cada
    uno **del resto**, y por eso no se deja engañar por un linaje que evoluciona
    rápido (el error clásico de agrupar los de rama larga por parecerse entre sí).
    """
    D, nombres = _matriz_valida(dm)
    nodos = [Clade(name=n) for n in nombres]
    n = len(nodos)

    while n > 3:                                     # bucle por UNIÓN (nodo), no por símbolo
        r = D.sum(axis=1)
        Q = (n - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        i, j = np.unravel_index(int(np.argmin(Q)), Q.shape)
        i, j = (int(i), int(j)) if i < j else (int(j), int(i))

        dij = D[i, j]
        li = 0.5 * dij + (r[i] - r[j]) / (2.0 * (n - 2))
        lj = dij - li
        # las ramas negativas no tienen sentido biológico: es un artefacto conocido
        # de NJ. El convenio estándar es llevarlas a 0 y pasar el exceso a la hermana.
        if li < 0:
            lj, li = lj + li, 0.0
        if lj < 0:
            li, lj = li + lj, 0.0
        nodos[i].length, nodos[j].length = li, lj

        nueva = 0.5 * (D[i, :] + D[j, :] - dij)
        padre = Clade(children=[nodos[i], nodos[j]])
        D = _sin(D, i, j, nueva)
        nodos = [nodos[k] for k in range(len(nodos)) if k not in (i, j)] + [padre]
        n -= 1

    # los tres últimos forman una TRIFURCACIÓN: es el árbol sin raíz de verdad
    a, b, c = nodos
    dab, dac, dbc = D[0, 1], D[0, 2], D[1, 2]
    a.length = max(0.0, (dab + dac - dbc) / 2.0)
    b.length = max(0.0, (dab + dbc - dac) / 2.0)
    c.length = max(0.0, (dac + dbc - dab) / 2.0)
    raiz = Clade(children=[a, b, c])
    return Tree(root=raiz, method="nj", rooted=False, n_leaves=len(dm.names))


def _agrupar(dm: DistanceMatrix, ponderado: bool, metodo: str) -> Tree:
    """Núcleo común de UPGMA y WPGMA: solo cambia cómo se promedian las distancias."""
    D, nombres = _matriz_valida(dm)
    nodos = [Clade(name=n) for n in nombres]
    alturas = [0.0] * len(nodos)
    tam = [1] * len(nodos)

    while len(nodos) > 1:                            # bucle por UNIÓN
        M = D.copy()
        np.fill_diagonal(M, np.inf)
        i, j = np.unravel_index(int(np.argmin(M)), M.shape)
        i, j = (int(i), int(j)) if i < j else (int(j), int(i))

        altura = D[i, j] / 2.0
        nodos[i].length = max(0.0, altura - alturas[i])
        nodos[j].length = max(0.0, altura - alturas[j])
        padre = Clade(children=[nodos[i], nodos[j]])

        if ponderado:      # UPGMA: media sobre TODAS las parejas de taxones
            nueva = (tam[i] * D[i, :] + tam[j] * D[j, :]) / (tam[i] + tam[j])
        else:              # WPGMA: media simple entre los dos grupos
            nueva = (D[i, :] + D[j, :]) / 2.0

        D = _sin(D, i, j, nueva)
        keep = [k for k in range(len(nodos)) if k not in (i, j)]
        nodos = [nodos[k] for k in keep] + [padre]
        alturas = [alturas[k] for k in keep] + [altura]
        tam = [tam[k] for k in keep] + [tam[i] + tam[j]]

    return Tree(root=nodos[0], method=metodo, rooted=True, n_leaves=len(dm.names))


def upgma(dm: DistanceMatrix) -> Tree:
    """Construye el árbol por **UPGMA** (con raíz, supone reloj molecular).

    La distancia entre dos grupos es la media aritmética sobre **todas las parejas
    de taxones** que los cruzan, lo que equivale a ponderar por el tamaño de cada
    grupo::

        d(k, i∪j) = (nᵢ·d(k,i) + nⱼ·d(k,j)) / (nᵢ + nⱼ)

    Es la definición original (Sokal & Michener, 1958).

    Supone **reloj molecular**: que todos los linajes acumulan mutaciones al mismo
    ritmo. Cuando eso se cumple, la raíz que produce es informativa; cuando no, el
    árbol puede agrupar mal — ver ``neighbor_joining``, que no necesita esa
    suposición.

    .. note::
       **Ojo al comparar con otras herramientas.** El ``upgma()`` de Biopython
       (``Bio.Phylo.TreeConstruction``) promedia con ``(d(k,i)+d(k,j))/2``, sin
       ponderar: eso es **WPGMA**, no UPGMA. Medido en
       ``tools/bench_vs_biopython_phylo.py``: con matrices sin empates, nuestro
       ``wpgma`` reproduce su resultado en 6/6 casos y nuestro ``upgma`` difiere a
       partir de ~20 taxones. Si necesitas reproducir su salida exacta, usa
       :func:`wpgma`.
    """
    return _agrupar(dm, ponderado=True, metodo="upgma")


def wpgma(dm: DistanceMatrix) -> Tree:
    """Construye el árbol por **WPGMA** (media simple entre grupos).

        ``d(k, i∪j) = (d(k,i) + d(k,j)) / 2``

    Da el mismo peso a cada **grupo** en vez de a cada taxón, así que un grupo
    pequeño pesa tanto como uno grande. Se incluye por dos razones: es un método
    legítimo y publicado, y es lo que calculan varias herramientas que lo etiquetan
    como «UPGMA» (Biopython entre ellas), así que hace falta para reproducir sus
    resultados.
    """
    return _agrupar(dm, ponderado=False, metodo="wpgma")


def build_tree(aligned: Iterable[str], *, names: Optional[Sequence[str]] = None,
               method: str = "nj", model: str = "jc") -> Tree:
    """Atajo cómodo: alineamiento → matriz de distancias → árbol."""
    dm = distance_matrix(aligned, model=model, names=names)
    if method == "nj":
        return neighbor_joining(dm)
    if method == "upgma":
        return upgma(dm)
    if method == "wpgma":
        return wpgma(dm)
    raise SequenceValueError(
        f"método desconocido: {method!r} (usa 'nj', 'upgma' o 'wpgma')")


# ── soporte por bootstrap: cuánta confianza merece cada rama ──────────────────
def _particiones(tree: Tree) -> set[frozenset[str]]:
    """Las BIPARTICIONES del árbol: qué hojas quedan a un lado de cada rama interna.

    Dos árboles son iguales si separan las hojas igual, independientemente de cómo
    se dibujen o dónde se ponga la raíz. Por eso se comparan las particiones y no
    la forma. Se guarda siempre el lado que NO contiene a la primera hoja
    (alfabéticamente), para que la representación sea única.

    Solo cuentan las particiones **informativas**: las que dejan al menos 2 hojas a
    CADA lado. Una que aísla una sola hoja es trivial —aparece en cualquier árbol—
    y contarla inflaría artificialmente el soporte del bootstrap hasta el 100 %.
    """
    todas = set(tree.leaves())
    if not todas:
        return set()
    ancla = min(todas)
    salida: set[frozenset[str]] = set()

    def recorrer(nodo: Clade) -> set[str]:
        if nodo.is_leaf:
            return {nodo.name}
        acum: set[str] = set()
        for h in nodo.children:
            acum |= recorrer(h)
        # informativa solo si deja >=2 hojas a cada lado
        if min(len(acum), len(todas) - len(acum)) >= 2:
            lado = acum if ancla not in acum else (todas - acum)
            salida.add(frozenset(lado))
        return acum

    recorrer(tree.root)
    return salida


def bootstrap_support(aligned: Sequence[str], *, names: Optional[Sequence[str]] = None,
                      method: str = "nj", model: str = "jc",
                      replicates: int = 100,
                      seed: Optional[int] = None) -> Tree:
    """Construye el árbol y **anota cada rama con su soporte** (0-100).

    El procedimiento (Felsenstein, 1985) es el estándar del campo: se remuestrean
    las columnas del alineamiento **con reemplazo** ``replicates`` veces, se
    reconstruye el árbol en cada réplica y se cuenta en qué porcentaje reaparece
    cada agrupación del árbol original.

    Interpretación honesta: **≥95 % = sólido**, 70-95 % = razonable, **<70 % = los
    datos no lo sostienen**, y esa rama no debería presentarse como un resultado.

    Parameters
    ----------
    aligned:
        Secuencias ya alineadas.
    replicates:
        Número de réplicas (100 es lo habitual; 1000 para publicar).
    seed:
        Semilla, para que el resultado sea reproducible.

    Returns
    -------
    Tree
        El árbol del alineamiento real, con ``support`` puesto en cada nodo interno.
    """
    seqs = [s.upper() for s in aligned]
    if replicates < 1:
        raise SequenceValueError(f"replicates debe ser ≥1 (es {replicates}).")
    L = len(seqs[0])
    if any(len(s) != L for s in seqs):
        raise SequenceValueError("las secuencias deben estar alineadas (misma longitud).")

    arbol = build_tree(seqs, names=names, method=method, model=model)
    objetivo = _particiones(arbol)
    if not objetivo:
        return arbol

    votos: dict[frozenset[str], int] = {p: 0 for p in objetivo}
    rng = np.random.default_rng(seed)
    # matriz de caracteres: remuestrear columnas es indexar, sin bucles por símbolo
    M = np.array([list(s) for s in seqs], dtype="<U1")

    for _ in range(replicates):                      # bucle por RÉPLICA
        cols = rng.integers(0, L, size=L)
        remuestra = ["".join(fila) for fila in M[:, cols]]
        try:
            rep = build_tree(remuestra, names=names, method=method, model=model)
        except SequenceValueError:                   # réplica degenerada: se descarta
            continue
        for p in _particiones(rep):
            if p in votos:
                votos[p] += 1

    todas = set(arbol.leaves())
    ancla = min(todas) if todas else ""

    def anotar(nodo: Clade) -> set[str]:
        if nodo.is_leaf:
            return {nodo.name}
        acum: set[str] = set()
        for h in nodo.children:
            acum |= anotar(h)
        if min(len(acum), len(todas) - len(acum)) >= 2:
            lado = acum if ancla not in acum else (todas - acum)
            nodo.support = 100.0 * votos.get(frozenset(lado), 0) / replicates
        return acum

    anotar(arbol.root)
    return arbol
