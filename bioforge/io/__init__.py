"""
bioforge.io — entrada/salida de datos: informes de calidad y compresión.

``qcreport`` (informe FASTQ estilo FastQC, columnar) y ``bgzf`` (gzip por bloques,
descomprimible en paralelo). La lectura de secuencias en sí vive en
:mod:`bioforge.core` (``SmartImporter``).
"""
