"""
tools/stress_test.py
Benchmark de rendimiento con ~30 millones de bases sintéticas.

Mide:
  - Tiempo de empaquetado (encoding + BitPacker.pack)
  - RAM consumida por la secuencia empaquetada
  - Tiempo de traducción con SmartTranslator

Uso:
    python tools/stress_test.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
from biocore import SeqType, BitPacker, PackedSequence
from smart_translator import SmartTranslator


def stress_test():
    print("=" * 50)
    print("TEST DE ESTRES EXTREMO (MOTOR 5-BIT)")
    print("=" * 50)

    secuencia_masiva = "ATGGCCCTG" * 3_333_333
    print(f"\nGenerando genoma sintetico de {len(secuencia_masiva):,} letras...")

    # Empaquetado
    t0 = time.perf_counter()

    texto_bytes = np.frombuffer(secuencia_masiva.encode('ascii'), dtype=np.uint8)
    lut = np.zeros(256, dtype=np.uint8)
    lut[ord('A')] = 0; lut[ord('C')] = 1; lut[ord('G')] = 2
    lut[ord('T')] = 3; lut[ord('U')] = 3
    codigos_nuc = lut[texto_bytes]

    adn_empaquetado = PackedSequence(
        header="Cromosoma_Titan",
        seq_type=SeqType.NUCLEOTIDE,
        n_symbols=len(codigos_nuc),
        data=BitPacker.pack(codigos_nuc),
    )
    t1 = time.perf_counter()

    print(f"\nEmpaquetado : {(t1 - t0) * 1000:.2f} ms")
    print(f"RAM usada   : {adn_empaquetado.packed_bytes / 1024 / 1024:.2f} MB")
    print(f"Ahorro      : {(1 - adn_empaquetado.memory_ratio) * 100:.1f}% vs ASCII")

    # Traduccion
    print("\nArrancando motor de traduccion NumPy...")
    t2 = time.perf_counter()
    proteina = SmartTranslator.translate(adn_empaquetado)
    t3 = time.perf_counter()

    print(f"Traduccion  : {(t3 - t2) * 1000:.2f} ms")
    print(f"Aminoacidos : {proteina.n_symbols:,}")
    print("\nTEST SUPERADO.")


if __name__ == "__main__":
    stress_test()
