"""
bioforge.sequence — transformaciones de secuencia.

Traducción ADN→proteína vectorizada, 6 marcos de lectura y complemento reverso.
"""
from .translator import SmartTranslator          # noqa: F401
__all__ = ["SmartTranslator"]
