"""
tests/test_isolation.py — el GUARDIÁN de la Regla #10: cada herramienta se usa
sola, sin activar a las demás.

Por qué existe este archivo
---------------------------
La carga perezosa (v10.1) es una promesa fácil de romper **en silencio**: basta un
``from .nanopore import basecall`` mal colocado en un ``__init__.py`` para que
``import bioforge`` vuelva a cargar el motor entero, y ningún test se quejaría.
Con 10 herramientas eso se nota; con 30, no. Este archivo convierte la regla
escrita en un contrato ejecutable.

Es un test TRANSVERSAL (cruza todas las funciones), por eso vive en la raíz de
``tests/`` y no en una carpeta espejo.

Cómo mide
---------
No se puede "descargar" un módulo de forma fiable dentro del mismo proceso, así que
cada caso arranca un intérprete LIMPIO, importa una sola cosa y reporta qué quedó
cargado. Es lento (un proceso por herramienta), pero es la única medición honesta.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from functools import lru_cache

import pytest

# Dependencias que NUNCA debe arrastrar un import público: o son opcionales
# (extras 'ai', 'nanopore', 'app') o son pesadas. Si alguna aparece, es que una
# herramienta la está importando con ansia en vez de dentro de la función.
PESADAS = ("torch", "transformers", "h5py", "pod5", "webview", "scipy", "pandas")


@lru_cache(maxsize=None)
def _huella(codigo: str) -> tuple[frozenset[str], frozenset[str]]:
    """Ejecuta ``codigo`` en un intérprete limpio y devuelve lo que quedó cargado.

    Returns (submódulos de bioforge, dependencias pesadas). Se cachea para no pagar
    un proceso nuevo por cada aserción sobre la misma herramienta.
    """
    prog = textwrap.dedent(f"""
        import sys
        {codigo}
        subs = sorted(m for m in sys.modules if m.startswith("bioforge."))
        pes = sorted(d for d in {PESADAS!r} if d in sys.modules)
        print("|".join(subs))
        print("|".join(pes))
    """)
    res = subprocess.run([sys.executable, "-c", prog], capture_output=True,
                         text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        pytest.fail(f"el import falló:\n{codigo}\n{res.stderr[-800:]}")
    lineas = res.stdout.strip().split("\n")
    subs = {s for s in lineas[0].split("|") if s}
    pes = {p for p in (lineas[1] if len(lineas) > 1 else "").split("|") if p}
    # los privados (bioforge.engine._loader) no cuentan como "herramienta cargada"
    return frozenset(s for s in subs if "._" not in s), frozenset(pes)


def _familias(submodulos: frozenset[str]) -> set[str]:
    """De {'bioforge.align.msa', ...} saca las FAMILIAS: {'align', ...}."""
    return {s.split(".")[1] for s in submodulos if s.count(".") >= 1}


# ── 1. la promesa mayor: importar el paquete no carga NADA ───────────────────
def test_import_bioforge_no_carga_nada():
    """``import bioforge`` debe ser gratis: ni un submódulo, ni siquiera NumPy.

    Es la diferencia entre arrancar en ~5 ms o en ~75 ms, y la que permite que la
    caja crezca sin que abrirla cueste más cada vez.
    """
    subs, pesadas = _huella("import bioforge")
    assert subs == frozenset(), f"import bioforge cargó submódulos: {sorted(subs)}"
    assert pesadas == frozenset(), f"import bioforge cargó dependencias: {sorted(pesadas)}"


def test_import_bioforge_no_carga_numpy():
    """Ni NumPy. Sin esto, 'importar la caja' ya cuesta cientos de milisegundos."""
    prog = "import sys; import bioforge; print('numpy' in sys.modules)"
    res = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True)
    assert res.stdout.strip() == "False", "import bioforge arrastró NumPy"


# ── 2. cada herramienta, sola: no debe despertar a sus vecinas ───────────────
# (herramienta, símbolo público, familias que NO puede tocar)
HERRAMIENTAS = [
    ("traducir",   "SmartTranslator", {"nanopore", "evolution", "mapping", "app", "io"}),
    ("alinear",    "SequenceAligner", {"nanopore", "evolution", "mapping", "app", "io"}),
    ("nanoporo",   "basecall",        {"evolution", "mapping", "app", "io", "cli"}),
    ("evolución",  "rank_mutations",  {"nanopore", "mapping", "app", "io"}),
    ("mapeo",      "GenomeAligner",   {"nanopore", "evolution", "app", "io"}),
    # el llamador de variantes NO depende del mapeador: consume cualquier objeto
    # con los atributos de un Mapping (pato). Por eso 'mapping' está prohibido.
    ("variantes",  "call_variants",   {"nanopore", "evolution", "mapping", "app", "io"}),
]


@pytest.mark.parametrize("nombre,simbolo,prohibidas", HERRAMIENTAS,
                         ids=[h[0] for h in HERRAMIENTAS])
def test_herramienta_no_activa_a_las_demas(nombre, simbolo, prohibidas):
    """Pedir una herramienta no puede cargar las familias ajenas.

    Este es el corazón de la Regla #10: usar el traductor no debe encender el
    nanoporo, ni la evolución, ni el mapeador.
    """
    subs, _ = _huella(f"from bioforge import {simbolo}")
    cargadas = _familias(subs)
    intrusas = cargadas & prohibidas
    assert not intrusas, (
        f"'{nombre}' ({simbolo}) activó familias que no necesita: {sorted(intrusas)}\n"
        f"cargado: {sorted(cargadas)}\n"
        "Causa habitual: un import con ansia en el __init__.py de un subpaquete.")


@pytest.mark.parametrize("nombre,simbolo,_", HERRAMIENTAS,
                         ids=[h[0] for h in HERRAMIENTAS])
def test_herramienta_no_arrastra_dependencias_pesadas(nombre, simbolo, _):
    """Ninguna herramienta puede arrastrar un extra opcional ni una dep pesada.

    Los extras ('ai' → torch, 'nanopore' → pod5/h5py, 'app' → pywebview) se importan
    DENTRO de la función que los usa, nunca al cargar el módulo. Si esto falla, un
    usuario que solo quiere traducir ADN acabaría cargando PyTorch.
    """
    _, pesadas = _huella(f"from bioforge import {simbolo}")
    assert pesadas == frozenset(), (
        f"'{nombre}' ({simbolo}) arrastró {sorted(pesadas)}. "
        "Muévelo a un import perezoso dentro de la función que lo necesita.")


# ── 3. pereza DENTRO de una familia (el listón nuevo) ────────────────────────
def test_familia_variants_es_perezosa_por_dentro():
    """Pedir el pileup no debe cargar el llamador, ni al revés.

    ``bioforge.variants`` es la primera familia con carga perezosa interna (su
    ``__init__`` resuelve por PEP 562, igual que el paquete raíz). Quien solo
    quiera medir profundidad de cobertura no tiene por qué cargar la estadística
    de llamada de variantes. Es el patrón a copiar cuando una familia crezca.
    """
    subs, _ = _huella("from bioforge import pileup")
    assert "bioforge.variants.pileup" in subs
    assert "bioforge.variants.caller" not in subs, (
        "pedir el pileup cargó el llamador: el __init__ de variants dejó de ser perezoso")


# ── 4. el nanoporo, caso ejemplar: aislamiento total ─────────────────────────
def test_nanoporo_es_autonomo():
    """El basecaller no necesita ni el core: es la referencia de aislamiento.

    Se fija como prueba viva de que una herramienta PUEDE ser completamente
    autónoma, para que las que se añadan tengan un listón que copiar.
    """
    subs, _ = _huella("from bioforge import basecall")
    assert _familias(subs) == {"nanopore"}, (
        f"nanoporo dejó de ser autónomo: cargó {sorted(_familias(subs))}")
