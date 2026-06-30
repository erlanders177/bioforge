"""
tests/test_errors.py
Sistema de errores unificado: todo fallo del motor (parser, E/S, ingesta,
(de)compresión) es capturable con un único ``except BioForgeError``, y además
hereda del builtin estándar correspondiente (compatibilidad hacia atrás).
"""

import pytest

from bioforge import (
    BioForgeError, SequenceTypeError, SequenceValueError,
    TranslationError, AlignmentError, BioForgeIOError, EngineError,
    SmartImporter, qcreport,
)
from bioforge import bgzf


def test_hierarchy():
    # Todas las excepciones del motor heredan de BioForgeError…
    for exc in (SequenceTypeError, SequenceValueError, TranslationError,
                AlignmentError, BioForgeIOError, EngineError):
        assert issubclass(exc, BioForgeError)
    # …y del builtin estándar adecuado (para `except` existentes).
    assert issubclass(SequenceTypeError, TypeError)
    assert issubclass(SequenceValueError, ValueError)
    assert issubclass(BioForgeIOError, OSError)        # OSError == IOError
    assert issubclass(EngineError, RuntimeError)


def test_missing_file_raises_bioforge_io(tmp_path):
    missing = str(tmp_path / "no_existe.fastq")
    # Capturable como BioForgeError (promesa del proyecto)…
    with pytest.raises(BioForgeError):
        list(SmartImporter.stream_fastq(missing))
    # …y como OSError (compatibilidad).
    with pytest.raises(OSError):
        list(SmartImporter.stream(missing))


def test_missing_file_batches(tmp_path):
    missing = str(tmp_path / "no_existe.fasta")
    with pytest.raises(BioForgeError):
        list(SmartImporter.stream_batches(missing))


def test_qc_empty_file_is_bioforge(tmp_path):
    p = tmp_path / "vacio.fastq"
    p.write_text("", encoding="ascii")
    with pytest.raises(BioForgeError):
        qcreport.run(str(p))
    # sigue siendo un ValueError
    with pytest.raises(ValueError):
        qcreport.run(str(p))


def test_bgzf_overwrite_guard(tmp_path):
    src = tmp_path / "x.fastq"
    src.write_text("@r\nACGT\n+\nIIII\n", encoding="ascii")
    # out == in debe rechazarse (no sobrescribir el original)
    with pytest.raises(ValueError):
        bgzf.compress_file(str(src), str(src))
