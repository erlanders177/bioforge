"""
bioforge/core/errors.py — la jerarquía de errores, SIN dependencias.

Todo fallo del motor hereda de :class:`BioForgeError`, para que un solo
``except BioForgeError`` lo capture todo (regla de oro nº8). Cada subclase hereda
además del builtin estándar adecuado, así el código que ya atrapa ``ValueError`` u
``OSError`` sigue funcionando sin cambios.

Por qué vive aparte de ``biocore``
----------------------------------
Estas clases no necesitan NumPy ni nada: son siete ``class`` y sus docstrings.
Tenerlas dentro de ``biocore`` obligaba a **cargar NumPy entero** a cualquier
herramienta ligera que solo quisiera lanzar un error decente — media herramienta de
laboratorio pagaba ~500 ms por eso. Separarlas deja que las herramientas de texto
puro (enzimas de restricción, cebadores) sean instantáneas sin renunciar a la
jerarquía común.

``biocore`` las reexporta, así que ``from bioforge.core.biocore import
SequenceValueError`` sigue funcionando igual que antes.
"""

from __future__ import annotations


class BioForgeError(Exception):
    """Base para todos los errores propios de BioForge.

    Úsala en bloques ``except`` para capturar cualquier error del motor
    sin interferir con el resto de Python::

        from bioforge import BioForgeError
        try:
            prot = SmartTranslator.translate(seq)
        except BioForgeError as e:
            print(f"Error de BioForge: {e}")

    Las subclases también heredan de ``TypeError`` o ``ValueError`` según
    corresponda, por lo que el código existente que ya atrapa esos tipos
    estándar sigue funcionando sin cambios.
    """


class SequenceTypeError(BioForgeError, TypeError):
    """Tipo incorrecto al llamar a una función del motor.

    Se lanza cuando:

    - Se pasa un ``str``, ``list`` u otro objeto donde se esperaba
      ``PackedSequence``.
    - Se mezclan tipos biológicos incompatibles (NUCLEOTIDE con PROTEIN).
    - El ``seq_type`` de un ``PackedSequence`` no es un valor ``SeqType``.
    """


class SequenceValueError(BioForgeError, ValueError):
    """Valor inválido en una secuencia o en sus metadatos.

    Se lanza cuando:

    - ``n_symbols`` es negativo.
    - El buffer ``packed`` es demasiado pequeño para ``n`` símbolos.
    - La secuencia está vacía donde se requiere contenido.
    - ``codes`` no es un array 1-D.
    """


class TranslationError(BioForgeError, ValueError):
    """Error durante la traducción ADN→Proteína.

    Se lanza cuando:

    - La secuencia no contiene ningún codón ATG/AUG.
    - El ORF no tiene ningún codón completo tras el ATG.
    - La secuencia es demasiado corta para contener un codón.
    """


class AlignmentError(BioForgeError, ValueError):
    """Error durante el alineamiento o en sus parámetros.

    Se lanza cuando:

    - El modo no es ``'global'`` ni ``'semi-global'``.
    - ``width`` es ≤ 0 en ``format_alignment``.
    - Las cadenas alineadas tienen longitudes incongruentes.
    """


class BioForgeIOError(BioForgeError, OSError):
    """No se pudo abrir o leer un archivo de secuencias.

    Hereda de ``OSError`` (= ``IOError``), por lo que el código que ya atrapa
    errores de E/S sigue funcionando, y además se captura con ``BioForgeError``.
    """


class EngineError(BioForgeError, RuntimeError):
    """Fallo del motor de ingesta: parser, descompresión o conversión BGZF.

    Se lanza cuando:

    - El parser por lotes o paralelo devuelve un código de error (buffer
      desbordado, ventana demasiado densa, registro gigante).
    - La (des)compresión BGZF/libdeflate falla.
    - Se pide la vía rápida ``.gz`` sin libdeflate compilado.

    Hereda de ``RuntimeError`` y se captura con ``BioForgeError``.
    """


# ══════════════════════════════════════════════════════════════════════════════
# §1  BIOLOGICAL ALPHABET  —  5-bit codes  (values 0 … 31)
# ══════════════════════════════════════════════════════════════════════════════
