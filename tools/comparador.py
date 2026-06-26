"""
tools/comparador.py
Herramienta interactiva para comparar dos secuencias genéticas.

Pega o escribe dos secuencias directamente en la terminal.
Elige si quieres ver las diferencias a nivel de nucleótido, aminoácido o ambos.

Uso:
    python tools/comparador.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from bioforge import SeqType, BitPacker, PackedSequence, NUC_LUT, AA_LUT
from bioforge import SmartTranslator, SequenceAligner, format_alignment
from bioforge.analyze import _AA_NAMES, _NUC_NAMES, _change_type


# ── Colores ANSI ───────────────────────────────────────────────────────────────
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
    print(f"\n{_b(f'  SECUENCIA {numero}')}")
    print(f"  Nombre (opcional, Enter para omitir): ", end="")
    nombre = input().strip() or f"Secuencia_{numero}"
    print(f"  Pega la secuencia (ADN o proteína, una o varias líneas).")
    print(f"  Escribe FIN cuando termines:")
    print()
    lineas = []
    while True:
        linea = input().strip()
        if linea.upper() == "FIN":
            break
        if linea and not linea.startswith(">"):
            lineas.append(linea.upper())
    return nombre, "".join(lineas).replace(" ", "")


def _elegir_modo(es_adn: bool) -> str:
    print(f"\n{_b('  NIVEL DE ANÁLISIS')}")
    if es_adn:
        print("  [1] Nucleótido  — ver qué bases cambiaron (A, C, G, T)")
        print("  [2] Aminoácido  — ver qué aminoácidos cambiaron (ignora cambios sinónimos)")
        print("  [3] Ambos       — nucleótido + impacto en proteína (recomendado)")
        print()
        print("  Elige [1/2/3]: ", end="")
        resp = input().strip()
        return {"1": "dna", "2": "protein", "3": "both"}.get(resp, "both")
    else:
        print("  La entrada es proteína — solo análisis a nivel de aminoácido.")
        return "protein"


def _detectar_tipo(secuencia: str) -> SeqType:
    if any(c in "EFILPQefilpq*" for c in secuencia):
        return SeqType.PROTEIN
    return SeqType.NUCLEOTIDE


def _hacer_packed(nombre: str, secuencia: str, tipo: SeqType) -> PackedSequence:
    raw   = np.frombuffer(secuencia.encode("ascii", errors="replace"), dtype=np.uint8)
    lut   = NUC_LUT if tipo == SeqType.NUCLEOTIDE else AA_LUT
    codes = lut[raw]
    return PackedSequence(
        header=nombre, seq_type=tipo,
        n_symbols=len(codes), data=BitPacker.pack(codes),
    )


# ── Sección de nucleótidos ─────────────────────────────────────────────────────

def _mostrar_dna(nombre_a: str, nombre_b: str,
                 seq_a: PackedSequence, seq_b: PackedSequence) -> None:
    W = 60
    sep = "─" * W
    aln = SequenceAligner.align(seq_a, seq_b)

    print(f"\n  {_b(sep)}")
    print(f"  {_b('NIVEL NUCLEÓTIDO')}")
    print(f"  {sep}")
    id_color = _g if aln.identity >= 0.95 else (_y if aln.identity >= 0.80 else _r)
    print(f"  Identidad    : {id_color(f'{aln.identity:.1%}')}  "
          f"({aln.n_matches}/{aln.n_matches + aln.n_mismatches + aln.n_gaps} posiciones)")
    print(f"  Cambios de base  : {_r(str(aln.n_mismatches)) if aln.n_mismatches else _g('0')}")
    print(f"  Indels (nt)      : {_y(str(aln.n_gaps)) if aln.n_gaps else _g('0')}\n")

    print(format_alignment(aln, width=56))

    subs = [m for m in aln.mutations if m.kind == "substitution"]
    dels = [m for m in aln.mutations if m.kind == "deletion"]
    ins  = [m for m in aln.mutations if m.kind == "insertion"]

    print(f"  {sep}")
    print(f"  MUTACIONES NUCLEOTÍDICAS  ({_b(str(len(aln.mutations)))} encontradas)")
    print(f"  {sep}")

    if not aln.mutations:
        print(f"  {_g('Sin mutaciones nucleotídicas.')}\n")
        return

    if subs:
        print(f"\n  Sustituciones ({len(subs)}):")
        for m in subs:
            rn = _NUC_NAMES.get(m.sym_a, m.sym_a)
            qn = _NUC_NAMES.get(m.sym_b, m.sym_b)
            print(f"    posición {m.pos_a + 1:>5}  {_r(m.sym_a)} → {_g(m.sym_b)}  [{rn} → {qn}]")
    if dels:
        print(f"\n  Deleciones ({len(dels)} nt):")
        for m in dels:
            print(f"    posición {m.pos_a + 1:>5}  {_r(m.sym_a)} [{_NUC_NAMES.get(m.sym_a,'')}] ausente")
    if ins:
        print(f"\n  Inserciones ({len(ins)} nt):")
        for m in ins:
            print(f"    posición {m.pos_b + 1:>5}  {_g(m.sym_b)} [{_NUC_NAMES.get(m.sym_b,'')}] insertado")
    print()


# ── Sección de aminoácidos ─────────────────────────────────────────────────────

def _mostrar_protein(nombre_a: str, nombre_b: str,
                     prot_a: PackedSequence, prot_b: PackedSequence,
                     fue_adn: bool) -> None:
    W   = 60
    sep = "─" * W
    aln = SequenceAligner.align(prot_a, prot_b)

    print(f"\n  {_b(sep)}")
    print(f"  {_b('NIVEL AMINOÁCIDO')}")
    print(f"  {sep}")

    if fue_adn:
        print(f"  {_b('Proteína referencia:')} {prot_a.to_string()[:55]}")
        print(f"  {_b('Proteína comparada: ')} {prot_b.to_string()[:55]}\n")

    id_color = _g if aln.identity >= 0.95 else (_y if aln.identity >= 0.80 else _r)
    print(f"  Identidad        : {id_color(f'{aln.identity:.1%}')}  "
          f"({aln.n_matches}/{aln.n_matches + aln.n_mismatches + aln.n_gaps} posiciones)")
    print(f"  Sustituciones AA : {_r(str(aln.n_mismatches)) if aln.n_mismatches else _g('0')}")
    print(f"  Indels (aa)      : {_y(str(aln.n_gaps)) if aln.n_gaps else _g('0')}\n")

    print(format_alignment(aln, width=56))

    subs = [m for m in aln.mutations if m.kind == "substitution"]
    dels = [m for m in aln.mutations if m.kind == "deletion"]
    ins  = [m for m in aln.mutations if m.kind == "insertion"]
    total = len(aln.mutations)

    print(f"  {sep}")
    print(f"  MUTACIONES DE AMINOÁCIDO  ({_b(str(total))} encontradas)")
    print(f"  {sep}")

    if total == 0:
        print(f"  {_g('Sin mutaciones de aminoácido. Proteínas funcionalmente idénticas.')}\n")
        return

    if subs:
        print(f"\n  Sustituciones ({len(subs)}):")
        for m in subs:
            rn     = _AA_NAMES.get(m.sym_a, m.sym_a)
            qn     = _AA_NAMES.get(m.sym_b, m.sym_b)
            tipo   = _change_type(m.sym_a, m.sym_b)
            tag    = _y("conservativa") if tipo == "conservative" else _r("RADICAL")
            print(f"    posición {m.pos_a + 1:>4}  "
                  f"{_r(m.sym_a)} → {_g(m.sym_b)}  [{rn} → {qn}]  {tag}")
    if dels:
        print(f"\n  Deleciones en comparada ({len(dels)} aa):")
        for m in dels:
            print(f"    posición {m.pos_a + 1:>4}  {_r(m.sym_a)} [{_AA_NAMES.get(m.sym_a,'')}] ausente")
    if ins:
        print(f"\n  Inserciones en comparada ({len(ins)} aa):")
        for m in ins:
            print(f"    posición {m.pos_b + 1:>4}  {_g(m.sym_b)} [{_AA_NAMES.get(m.sym_b,'')}] extra")

    radical = [m for m in subs if _change_type(m.sym_a, m.sym_b) == "radical"]
    print(f"\n  {sep}  INTERPRETACIÓN  {sep[:20]}")
    if radical:
        print(f"  {_r(f'{len(radical)} sustitución(es) RADICAL(ES).')} Impacto funcional alto probable.")
    elif subs:
        print(f"  {_y('Sustituciones conservativas.')} Impacto funcional bajo probable.")
    if dels or ins:
        print(f"  {_r(f'{len(dels)+len(ins)} posición(es) con indel.')} Impacto significativo probable.")
    print()


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
    print("  Acepta ADN o proteína. Escribe FIN al acabar de pegar cada secuencia.\n")

    while True:
        nombre_a, seq_a_txt = _leer_secuencia(1)
        if not seq_a_txt:
            print(f"  {_r('Secuencia vacía.')}"); continue

        nombre_b, seq_b_txt = _leer_secuencia(2)
        if not seq_b_txt:
            print(f"  {_r('Secuencia vacía.')}"); continue

        tipo   = _detectar_tipo(seq_a_txt)
        tipo_b = _detectar_tipo(seq_b_txt)
        if tipo != tipo_b:
            print(f"\n  {_y('Aviso: tipos distintos detectados. Se usará el tipo de la secuencia 1.')}")

        try:
            packed_a = _hacer_packed(nombre_a, seq_a_txt, tipo)
            packed_b = _hacer_packed(nombre_b, seq_b_txt, tipo)
        except Exception as e:
            print(f"  {_r(f'Error al procesar: {e}')}")
            continue

        fue_adn = tipo == SeqType.NUCLEOTIDE
        modo    = _elegir_modo(fue_adn)

        W   = 60
        dbl = "═" * W
        print(f"\n{_b(dbl)}")
        print(f"{_b('  RESULTADO DEL ANÁLISIS')}")
        print(f"{_b(dbl)}")
        print(f"\n  Referencia : {_c(nombre_a)}")
        print(f"  Comparada  : {_c(nombre_b)}")
        print(f"  Tipo       : {'ADN' if fue_adn else 'Proteína'}"
              f"   |   Nivel: {modo}\n")

        try:
            # DNA level
            if fue_adn and modo in ("dna", "both"):
                _mostrar_dna(nombre_a, nombre_b, packed_a, packed_b)

            # Protein level
            if modo in ("protein", "both"):
                if fue_adn:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        prot_a = SmartTranslator.translate(packed_a)
                        prot_b = SmartTranslator.translate(packed_b)
                else:
                    prot_a, prot_b = packed_a, packed_b
                _mostrar_protein(nombre_a, nombre_b, prot_a, prot_b, fue_adn)

        except Exception as e:
            print(f"  {_r(f'Error en el análisis: {e}')}")

        print(f"{_b(dbl)}\n")

        print("  ¿Comparar otras secuencias? (s/n): ", end="")
        if input().strip().lower() not in ("s", "si", "sí", "y", "yes"):
            print("\n  Hasta luego.\n")
            break


if __name__ == "__main__":
    main()
