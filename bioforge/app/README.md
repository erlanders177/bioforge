# BioForge — app de escritorio (v10.0)

La **otra cara** del motor: una ventana local para usar BioForge **con clics**, sin
escribir código. Todo corre en tu ordenador — **sin servidor, sin conexión**. Tus
datos (tu ADN) nunca salen de la máquina.

> "Un motor, dos caras." La app es una capa fina de interfaz sobre el paquete
> `bioforge`, que ya está probado (585 tests). El motor no se toca.

## Abrirla

Dos formas, mismo código y misma versión:

- **Sin Python (doble clic):** descarga el `.exe` desde la
  [última Release](https://github.com/erlanders177/bioforge/releases), descomprime y
  abre `BioForge.exe`. Autocontenido — nada que instalar.
- **Desde el paquete (para quien programa):**
  ```bash
  pip install "bioforge[app]"
  bioforge-app
  ```
  (o, desde el repo:  `python -m bioforge.app.main`)

Se abre una ventana. Pulsa **Abrir archivo…** (o **Probar con un ejemplo**) y explora.

## Qué hace (5 pestañas)

- **Secuencias** — cargar FASTA/FASTQ, ver el resumen y el detalle de cada secuencia,
  traducir ADN → proteína (codón a codón, coloreado por tipo de aminoácido).
- **Calidad** — informe de calidad FASTQ estilo FastQC (calidad por posición, GC,
  Phred), con gráficos SVG.
- **Alinear** — comparar dos secuencias: identidad + mutaciones coloreadas.
- **Nanoporo** — cargar señal POD5/FAST5 → bases con nuestro basecaller clásico, y
  "usar en las otras pestañas".
- **Evolución** — rankear qué mutaciones podrían subir y filtrar una concreta con
  RealityCheck (OBSERVADO vs ESTIMADO).

Cada pantalla trae una explicación "para todos". Varios genomas abiertos a la vez
(pestañas) e integración entre pantallas.

## Arquitectura

```
bioforge/app/
  backend.py   la clase Api: métodos que la interfaz llama (puro Python, TESTEABLE
               sin abrir ventana — devuelve diccionarios, nunca lanza a la interfaz).
  main.py      lanzador PyWebview: crea la ventana y le inyecta la Api. Aquí vive lo
               específico de la ventana (el diálogo de archivos nativo).
  index.html   la interfaz: una sola página HTML/CSS/JS vanilla (sin frameworks ni
               build), tema oscuro, gráficos SVG inline. Llama a Python con
               window.pywebview.api.*
  data/        recursos de la interfaz: pore model (nanoporo) + icon.ico (doble hélice).
```

El `.exe` se construye con `BioForge.spec` (PyInstaller, en la raíz del repo) y lo
adjunta solo a cada Release el workflow `.github/workflows/build-app.yml`.

Los tests del puente están en `tests/test_app_backend.py` (no necesitan pantalla).
