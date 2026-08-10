"""
app/main.py — lanzador de la app de escritorio BioForge (PyWebview).

Una ventana NATIVA y LOCAL que muestra ``index.html`` y le da acceso al motor a
través de ``Api`` (app/backend.py). Sin servidor, sin red: el ADN del usuario nunca
sale de su máquina.

Ejecutar:  python app/main.py     (requiere:  pip install pywebview)
Empaquetar a .exe:  ver app/build.py (PyInstaller) — para que un no-programador
solo tenga que hacer doble clic, sin instalar nada.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import webview  # noqa: E402

from backend import Api  # noqa: E402

HERE = Path(__file__).resolve().parent
_FILTER = ("Secuencias (*.fasta;*.fa;*.fna;*.fastq;*.fq;*.txt)",
           "Señal nanoporo (*.pod5;*.fast5)", "Todos los archivos (*.*)")


class DesktopApi(Api):
    """Api del backend + lo específico de la ventana (diálogo de archivos)."""

    def __init__(self) -> None:
        super().__init__()
        self.window = None                           # lo inyecta main() al crear la ventana

    def pick_and_open(self) -> dict:
        """Abre el diálogo nativo de archivos y carga lo elegido."""
        try:
            paths = self.window.create_file_dialog(
                webview.FileDialog.OPEN, allow_multiple=False, file_types=_FILTER)
        except Exception as e:                       # noqa: BLE001
            return {"error": f"no se pudo abrir el diálogo: {e}"}
        if not paths:
            return {"cancelled": True}
        return self.open_file(paths[0])


def main() -> None:
    api = DesktopApi()
    window = webview.create_window(
        "BioForge — motor bioinformático local",
        url=str(HERE / "index.html"),
        js_api=api,
        width=1120, height=740, min_size=(860, 580),
    )
    api.window = window
    webview.start()                                  # bloquea hasta cerrar la ventana


if __name__ == "__main__":
    main()
