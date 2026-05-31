"""
tools/visor.py
Frontend interactivo para introducir una secuencia de ADN manualmente
y ver la traducción codón a codón en pantalla.

Los loops están PERMITIDOS aquí — este archivo es display, no motor.

Uso:
    python tools/visor.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from biocore import BioCode, BitPacker, PackedSequence, SeqType
from smart_translator import SmartTranslator


NUC_CHARS = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
AA_NAMES = {
    4: "Alanina", 5: "Cisteína", 6: "Ácido Aspártico", 7: "Ácido Glutámico",
    8: "Fenilalanina", 9: "Glicina", 10: "Histidina", 11: "Isoleucina",
    12: "Lisina", 13: "Leucina", 14: "Metionina [INICIO]", 15: "Asparagina",
    16: "Prolina", 17: "Glutamina", 18: "Arginina", 19: "Serina",
    20: "Treonina", 21: "Valina", 22: "Triptófano", 23: "Tirosina",
    24: "[STOP]"
}


def decodificar_array_nuc(arr):
    return "".join(NUC_CHARS.get(n, '?') for n in arr)


def analizar_mi_adn(secuencia_texto):
    print("\n" + "="*50)
    print("SECUENCIADOR Y TRADUCTOR MANUAL (MOTOR 5-BIT)")
    print("="*50)

    secuencia_limpia = secuencia_texto.upper().replace(" ", "").replace("\n", "")

    texto_bytes = np.frombuffer(secuencia_limpia.encode('ascii'), dtype=np.uint8)
    lut = np.zeros(256, dtype=np.uint8)
    lut[ord('A')] = 0; lut[ord('C')] = 1; lut[ord('G')] = 2
    lut[ord('T')] = 3; lut[ord('U')] = 3

    codigos_nuc  = lut[texto_bytes]
    adn_empaquetado = PackedSequence(
        header="Mi_ADN_Manual",
        seq_type=SeqType.NUCLEOTIDE,
        n_symbols=len(codigos_nuc),
        data=BitPacker.pack(codigos_nuc),
    )

    print(f"\n--- DIAGNOSTICO ---")
    print(f"Longitud : {adn_empaquetado.n_symbols} bases")
    print(f"Empaquetado en {adn_empaquetado.packed_bytes} bytes")

    try:
        proteina_empaquetada = SmartTranslator.translate(adn_empaquetado)
        codigos_amino        = proteina_empaquetada.decode()
        nuc_desempaquetados  = adn_empaquetado.decode()

        ventanas   = np.lib.stride_tricks.sliding_window_view(nuc_desempaquetados, 3)
        es_atg     = np.all(ventanas == [0, 3, 2], axis=1)
        inicio_orf = int(np.argmax(es_atg))

        print("\n--- TRADUCCION PASO A PASO ---")
        for i, amino_code in enumerate(codigos_amino):
            pos_adn     = inicio_orf + (i * 3)
            codon_nums  = nuc_desempaquetados[pos_adn: pos_adn + 3]
            codon_texto = decodificar_array_nuc(codon_nums)
            nombre      = AA_NAMES.get(int(amino_code), f"Desconocido ({amino_code})")
            print(f"Codon {i+1:>3}: {codon_texto} -> {nombre}")
            if amino_code == 24:
                break

        print("\nTraduccion completada.")

    except Exception as exc:
        print(f"\nERROR: {exc}")


if __name__ == "__main__":
    mi_secuencia = """
    CCCGGTACGTCGATCGTAGCTAGCTAGCTGCTCGATCGATCGATCGA
    """
    analizar_mi_adn(mi_secuencia)
