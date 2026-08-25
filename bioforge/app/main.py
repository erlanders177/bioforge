"""
bioforge/app/main.py — lanzador de la app de escritorio BioForge (PyWebview).

Una ventana NATIVA y LOCAL que muestra ``index.html`` y le da acceso al motor a
través de ``Api`` (bioforge.app.backend). Sin servidor, sin red: el ADN del usuario
nunca sale de su máquina.

Ejecutar desde el paquete:   bioforge-app        (requiere:  pip install "bioforge[app]")
                    o bien:   python -m bioforge.app.main
Empaquetar a .exe:  BioForge.spec en la raíz del repo (PyInstaller) — para que un
no-programador solo tenga que hacer doble clic, sin instalar nada.
"""

from pathlib import Path

from bioforge.app.backend import Api, app_dir

HERE = Path(app_dir())                     # carpeta de recursos (código o .exe)

_FILTER = ("Secuencias (*.fasta;*.fa;*.fna;*.fastq;*.fq;*.txt)",
           "Señal nanoporo (*.pod5;*.fast5)", "Todos los archivos (*.*)")


def _require_webview():
    """Importa pywebview, con un mensaje claro si falta el extra 'app'."""
    try:
        import webview
        return webview
    except ImportError:
        raise SystemExit(
            "La app de escritorio de BioForge necesita pywebview.\n"
            'Instálalo con:  pip install "bioforge[app]"\n'
            "(o usa el ejecutable .exe, que no necesita instalar nada).")


class DesktopApi(Api):
    """Api del backend + lo específico de la ventana (diálogo de archivos)."""

    def __init__(self) -> None:
        super().__init__()
        self.window = None                           # lo inyecta main() al crear la ventana

    def pick_and_open(self) -> dict:
        """Abre el diálogo nativo de archivos y carga lo elegido."""
        import webview
        try:
            paths = self.window.create_file_dialog(
                webview.FileDialog.OPEN, allow_multiple=False, file_types=_FILTER)
        except Exception as e:                       # noqa: BLE001
            return {"error": f"no se pudo abrir el diálogo: {e}"}
        if not paths:
            return {"cancelled": True}
        return self.open_file(paths[0])

    def save_vcf(self) -> dict:
        """Diálogo nativo para guardar el VCF del último análisis de variantes."""
        datos = self.vcf_text()
        if "error" in datos:
            return datos
        import webview
        try:
            destino = self.window.create_file_dialog(
                webview.FileDialog.SAVE, save_filename="variantes.vcf",
                file_types=("Archivo VCF (*.vcf)", "Todos los archivos (*.*)"))
        except Exception as e:                       # noqa: BLE001
            return {"error": f"no se pudo abrir el diálogo: {e}"}
        if not destino:
            return {"cancelled": True}
        ruta = destino if isinstance(destino, str) else destino[0]
        try:
            with open(ruta, "w", encoding="utf-8") as fh:
                fh.write(datos["vcf"])
        except OSError as e:
            return {"error": f"no se pudo escribir el archivo: {e}"}
        return {"saved": ruta}

    def pick_and_open_signal(self) -> dict:
        """Diálogo de archivos para señal de nanoporo (POD5/FAST5)."""
        import webview
        try:
            paths = self.window.create_file_dialog(
                webview.FileDialog.OPEN, allow_multiple=False,
                file_types=("Señal nanoporo (*.pod5;*.fast5)", "Todos los archivos (*.*)"))
        except Exception as e:                       # noqa: BLE001
            return {"error": f"no se pudo abrir el diálogo: {e}"}
        if not paths:
            return {"cancelled": True}
        return self.open_signal(paths[0])


def main() -> None:
    """Punto de entrada de la app (comando ``bioforge-app`` y ``python -m``)."""
    webview = _require_webview()
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
