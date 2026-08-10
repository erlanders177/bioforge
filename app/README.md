# BioForge — app de escritorio (v10.0, en construcción)

La **otra cara** del motor: una ventana local para usar BioForge **con clics**, sin
escribir código. Todo corre en tu ordenador — **sin servidor, sin conexión**. Tus
datos (tu ADN) nunca salen de la máquina.

> "Un motor, dos caras." La app es una capa fina de interfaz sobre el paquete
> `bioforge`, que ya está probado (525 tests). El motor no se toca.

## Probarla (desde el código)

```bash
pip install pywebview          # o: pip install "bioforge[app]"
python app/main.py
```

Se abre una ventana. Pulsa **Abrir archivo…**, elige un FASTA/FASTQ, y explora.

## Qué hace ya

- Cargar FASTA/FASTQ y ver un resumen (cuántas secuencias, tipos, longitudes).
- Listar las secuencias y ver el detalle de cada una (composición).
- Traducir ADN → proteína.
- Alinear dos secuencias y ver identidad + mutaciones, coloreadas.

## En construcción

- Informe de calidad FASTQ (estilo FastQC).
- Nanoporo: cargar señal POD5/FAST5 → bases (basecaller clásico).
- Evolución: rankear mutaciones, juez de predictores, filtro de realidad.
- Empaquetado a `.exe` (PyInstaller) para doble clic, sin instalar nada.

## Arquitectura

```
app/
  backend.py   la clase Api: métodos que la interfaz llama (puro Python, TESTEABLE
               sin abrir ventana — devuelve diccionarios, nunca lanza a la interfaz).
  main.py      lanzador PyWebview: crea la ventana y le inyecta la Api. Aquí vive lo
               específico de la ventana (el diálogo de archivos).
  index.html   la interfaz: una sola página HTML/CSS/JS vanilla (sin frameworks ni
               build), tema oscuro. Llama a Python con window.pywebview.api.*
```

Los tests del puente están en `tests/test_app_backend.py` (no necesitan pantalla).
