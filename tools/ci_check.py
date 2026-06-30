"""
tools/ci_check.py — Verificación de humo para los wheels en CI.

cibuildwheel instala el wheel recién construido en un entorno limpio y ejecuta
este script en CADA plataforma (Windows / Linux / macOS). Si el motor C no
cargó, el wheel no sirve para competir con las herramientas profesionales, así
que aquí fallamos a propósito (`assert`) para enterarnos en CI, no el usuario.
"""

import sys

import bioforge
from bioforge.engine import _loader as L

print(f"BioForge {bioforge.__version__}  ·  Python {sys.version.split()[0]}  "
      f"·  {sys.platform}")
print(f"  C_AVAILABLE           = {L.C_AVAILABLE}")
print(f"  C_PARALLEL_AVAILABLE  = {getattr(L, 'C_PARALLEL_AVAILABLE', '?')}")
print(f"  C_LIBDEFLATE_AVAILABLE= {getattr(L, 'C_LIBDEFLATE_AVAILABLE', '?')}")

# Prueba funcional mínima: importar, traducir, comprobar resultado.
seqs = bioforge.SmartImporter.from_string(">g\nATGAAAGGGTAA\n")
prot = bioforge.SmartTranslator.translate(seqs[0])
assert prot.to_string() == "MKG", f"traducción inesperada: {prot.to_string()!r}"
print("  traducción ATGAAAGGGTAA -> MKG  OK")

# El objetivo de los wheels nativos: motor C en TODAS las plataformas.
assert L.C_AVAILABLE, (
    "El motor C NO cargó en este wheel — habría caído al fallback NumPy. "
    "Revisa la compilación de esta plataforma en el log de cibuildwheel."
)
print("  Motor C activo. Wheel válido.")
