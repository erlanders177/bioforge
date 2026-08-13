# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec — empaqueta la app de escritorio BioForge en un ejecutable local.

Construir (desde la raíz del repo):   pyinstaller BioForge.spec --noconfirm
Resultado:  dist/BioForge/BioForge.exe  (carpeta autocontenida; doble clic, sin Python)

Empaqueta: la interfaz (index.html), el pore model de nanoporo, el motor C
(engine.dll) y los pesos del ranker entrenado, además de NumPy, pywebview y h5py.
Las rutas son relativas a la carpeta de este .spec (la raíz del repo).
"""

from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []

# pywebview arrastra sus backends nativos (WebView2 / winforms) → recogerlos todos
for _pkg in ("webview",):
    _d, _b, _h = collect_all(_pkg)
    datas += _d
    binaries += _b
    hiddenimports += _h

# recursos propios (rutas relativas a la raíz del repo, donde está este .spec)
datas += [
    ("app/index.html", "."),                              # la interfaz web local
    ("app/data/r9.4_6mer.model", "data"),                 # pore model (nanoporo)
    ("bioforge/engine/engine.dll", "bioforge/engine"),    # motor C
    ("bioforge/data/ranker_weights.npz", "bioforge/data"),  # ranker entrenado
]
hiddenimports += ["h5py", "bioforge", "backend"]

a = Analysis(
    ["app/main.py"],
    pathex=["app", "."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="BioForge",
    console=False,           # app de ventana: sin consola negra (build final)
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False,
    name="BioForge",
)
