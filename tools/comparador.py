"""
tools/comparador.py
Herramienta interactiva para comparar dos secuencias genéticas.

Pega o escribe dos secuencias directamente en la terminal.
El programa detecta el tipo (ADN o proteína), traduce si hace falta,
alinea a nivel de aminoácido y muestra las diferencias.

Uso:
    python tools/comparador.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biocore import SeqType, BitPacker, PackedSequence
from biocore import NUC_LUT, AA_LUT
import numpy as np
from smart_translator import SmartTranslator
from aligner import SequenceAligner, format_alignment
from analyze import run as _run_analysis, build_report, _AA_NAMES, _change_type


# ── Colores ANSI para la terminal ──────────────────────────────────────────────
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

def _g(t): return f"{_GREEN}{t}{_RESET}"
def _r(t): return f"{_RED}{t}{_RESET}"
def _y(t): return f"{_YELLOW}{t}{_RESET}"
def _c(t): return f"{_CYAN}{t}{_RESET}"
def _b(t): return f"{_BOLD}{t}{_RESET}"


# ── Helpers de entrada ─────────────────────────────────────────────────────────

def _leer_secuencia(numero: int) -> tuple[str, str]:
    """Pide al usuario que pegue una secuencia. Devuelve (nombre, secuencia)."""
    print(f"\n{_b(f'  SECUENCIA {numero}')}")
    print(f"  Nombre (opcional, Enter para omitir): ", end="")
    nombre = input().strip() or f"Secuencia_{numero}"

    print(f"  Pega la secuencia (ADN o proteína, una o varias líneas).")
    print(f"  Cuando termines escribe FIN en una línea sola y pulsa Enter:")
    print()

    lineas = []
    while True:
        linea = input().strip()
        if linea.upper() == "FIN":
            break
        if linea and not linea.startswith(">"):
            lineas.append(linea.upper())

    secuencia = "".join(lineas).replace(" ", "").replace("\n", "")
    return nombre, secuencia


def _detectar_tipo(secuencia: str) -> SeqType:
    """Detecta si la secuencia es ADN o proteína."""
    proteina_chars = set("EFILPQefilpq*")
    if any(c in proteina_chars for c in secuencia):
        return SeqType.PROTEIN
    return SeqType.NUCLEOTIDE


def _hacer_packed(nombre: str, secuencia: str, tipo: SeqType) -> PackedSequence:
    """Convierte una secuencia de texto a PackedSequence."""
    raw = np.frombuffer(secuencia.encode("ascii", errors="replace"), dtype=np.uint8)
    lut = NUC_LUT if tipo == SeqType.NUCLEOTIDE else AA_LUT
    codes = lut[raw]
    return PackedSequence(
        header    = nombre,
        seq_type  = tipo,
        n_symbols = len(codes),
        data      = BitPacker.pack(codes),
    )


# ── Visualización de diferencias ───────────────────────────────────────────────

def _mostrar_resultado(nombre_a: str, nombre_b: str,
                       seq_a: PackedSequence, seq_b: PackedSequence,
                       fue_adn: bool) -> None:
    """Muestra el análisis completo con colores en la terminal."""
    W = 60
    sep = "─" * W
    dbl = "═" * W

    print(f"\n{_b(dbl)}")
    print(f"{_b('  RESULTADO DEL ANÁLISIS')}")
    print(f"{_b(dbl)}\n")

    # Tipo de entrada
    tipo_txt = "ADN (traducido a proteína)" if fue_adn else "Proteína"
    print(f"  Referencia : {_c(nombre_a)}")
    print(f"  Comparada  : {_c(nombre_b)}")
    print(f"  Tipo       : {tipo_txt}\n")

    # Traducir si es ADN
    if fue_adn:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prot_a = SmartTranslator.translate(seq_a)
            prot_b = SmartTranslator.translate(seq_b)
        print(f"  {_b('Proteína referencia:')} {prot_a.to_string()}")
        print(f"  {_b('Proteína comparada: ')} {prot_b.to_string()}\n")
    else:
        prot_a = seq_a
        prot_b = seq_b

    # Alinear
    aln = SequenceAligner.align(prot_a, prot_b)

    # Resumen numérico
    print(f"  {sep}")
    print(f"  RESUMEN")
    print(f"  {sep}")
    identidad_color = _g if aln.identity >= 0.95 else (_y if aln.identity >= 0.80 else _r)
    print(f"  Identidad      : {identidad_color(f'{aln.identity:.1%}')}"
          f"  ({aln.n_matches}/{aln.n_matches + aln.n_mismatches + aln.n_gaps} posiciones)")
    print(f"  Coincidencias  : {_g(str(aln.n_matches))}")
    print(f"  Sustituciones  : {_r(str(aln.n_mismatches)) if aln.n_mismatches else _g('0')}")
    print(f"  Indels (aa)    : {_y(str(aln.n_gaps)) if aln.n_gaps else _g('0')}\n")

    # Alineamiento visual
    print(f"  {sep}")
    print(f"  ALINEAMIENTO PROTEICO")
    print(f"  {sep}")
    print(format_alignment(aln, width=56))

    # Mutaciones
    subs = [m for m in aln.mutations if m.kind == "substitution"]
    dels = [m for m in aln.mutations if m.kind == "deletion"]
    ins  = [m for m in aln.mutations if m.kind == "insertion"]
    total = len(aln.mutations)

    print(f"  {sep}")
    print(f"  MUTACIONES DE AMINOÁCIDO  ({_b(str(total))} encontradas)")
    print(f"  {sep}")

    if total == 0:
        print(f"  {_g('Sin mutaciones. Las proteínas son funcionalmente idénticas.')}\n")
    else:
        if subs:
            print(f"\n  Sustituciones ({len(subs)}):")
            for m in subs:
                ref_n  = _AA_NAMES.get(m.sym_a, m.sym_a)
                qry_n  = _AA_NAMES.get(m.sym_b, m.sym_b)
                tipo   = _change_type(m.sym_a, m.sym_b)
                tag    = _y("conservativa") if tipo == "conservative" else _r("RADICAL")
                print(f"    posición {m.pos_a + 1:>4}  "
                      f"{_r(m.sym_a)} → {_g(m.sym_b)}  "
                      f"[{ref_n} → {qry_n}]  {tag}")

        if dels:
            print(f"\n  Deleciones en la comparada ({len(dels)} aa):")
            for m in dels:
                print(f"    posición {m.pos_a + 1:>4}  "
                      f"{_r(m.sym_a)} [{_AA_NAMES.get(m.sym_a, m.sym_a)}] ausente")

        if ins:
            print(f"\n  Inserciones en la comparada ({len(ins)} aa):")
            for m in ins:
                print(f"    posición {m.pos_b + 1:>4}  "
                      f"{_g(m.sym_b)} [{_AA_NAMES.get(m.sym_b, m.sym_b)}] extra")

        # Interpretación
        radical = [m for m in subs if _change_type(m.sym_a, m.sym_b) == "radical"]
        print(f"\n  {sep}")
        print(f"  INTERPRETACIÓN")
        print(f"  {sep}")
        if radical:
            print(f"  {_r(f'{len(radical)} sustitución(es) RADICAL(ES) detectada(s).')}")
            print(f"  Cambian las propiedades fisicoquímicas del aminoácido.")
            print(f"  Impacto funcional probable: alto.")
        elif subs:
            print(f"  {_y('Todas las sustituciones son conservativas.')}")
            print(f"  Las propiedades de los aminoácidos son similares.")
            print(f"  Impacto funcional probable: bajo.")
        if dels or ins:
            print(f"  {_r(f'{len(dels)+len(ins)} posición(es) con indel.')}")
            print(f"  Los indels suelen tener impacto funcional significativo.")

    print(f"\n{_b(dbl)}\n")


# ══════════════════════════════════════════════════════════════════════════════
# PROGRAMA PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    print(f"\n{_b('  ╔══════════════════════════════════════════╗')}")
    print(f"{_b('  ║   COMPARADOR DE SECUENCIAS GENÉTICAS    ║')}")
    print(f"{_b('  ║        Motor 5-bit · NumPy              ║')}")
    print(f"{_b('  ╚══════════════════════════════════════════╝')}")
    print()
    print("  Compara dos secuencias a nivel de aminoácido.")
    print("  Acepta ADN o proteína. Las mutaciones sinónimas se ignoran.")
    print("  Escribe FIN cuando termines de pegar cada secuencia.\n")

    while True:
        # Leer secuencias
        nombre_a, seq_a_txt = _leer_secuencia(1)
        if not seq_a_txt:
            print(f"  {_r('Secuencia vacía. Inténtalo de nuevo.')}")
            continue

        nombre_b, seq_b_txt = _leer_secuencia(2)
        if not seq_b_txt:
            print(f"  {_r('Secuencia vacía. Inténtalo de nuevo.')}")
            continue

        # Detectar tipo (usa la referencia)
        tipo = _detectar_tipo(seq_a_txt)
        tipo_b = _detectar_tipo(seq_b_txt)
        if tipo != tipo_b:
            print(f"\n  {_r('Advertencia: una secuencia parece ADN y la otra proteína.')}")
            print(f"  Se usará el tipo de la secuencia 1: {tipo.name}\n")

        # Construir PackedSequences
        try:
            packed_a = _hacer_packed(nombre_a, seq_a_txt, tipo)
            packed_b = _hacer_packed(nombre_b, seq_b_txt, tipo)
        except Exception as e:
            print(f"  {_r(f'Error al procesar las secuencias: {e}')}")
            continue

        # Mostrar resultado
        fue_adn = tipo == SeqType.NUCLEOTIDE
        try:
            _mostrar_resultado(nombre_a, nombre_b, packed_a, packed_b, fue_adn)
        except Exception as e:
            print(f"  {_r(f'Error en el análisis: {e}')}")

        # ¿Otra comparación?
        print("  ¿Quieres comparar otras secuencias? (s/n): ", end="")
        resp = input().strip().lower()
        if resp not in ("s", "si", "sí", "y", "yes"):
            print("\n  Hasta luego.\n")
            break


if __name__ == "__main__":
    main()
