"""
tools/integrity_check.py — CERTIFICADO DE INTEGRIDAD del motor BioForge.

Dos usos, un mismo principio: los datos que entran deben salir intactos.

  python tools/integrity_check.py                 # certifica el MOTOR (invariantes)
  python tools/integrity_check.py mis_datos.fasta # certifica un ARCHIVO real

El primer modo corre la batería de invariantes que garantizan que ningún nivel
corrompe en silencio (el fallo que nos costó una versión: proteínas empaquetadas
como ADN → todo residuo raro convertido en 'N'). El segundo importa cada registro
de tu archivo y comprueba que se re-codifica idéntico, avisando de cualquier pérdida.

Pensado para ser LEGIBLE por alguien que no programa: dice PASA/FALLA y por qué.
En la Fase 2 (app de escritorio) este mismo chequeo será un botón "verificar datos".
"""

import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")            # consola Windows (cp1252)
sys.path.insert(0, ".")

from bioforge.biocore import (  # noqa: E402
    AA_LUT,
    NUC_LUT,
    BitPacker,
    SeqType,
    SequenceValueError,
    SmartImporter,
)
from bioforge.msa import _infer_type, align_multiple  # noqa: E402
from bioforge.smart_translator import SmartTranslator  # noqa: E402

_DNA = "ACGT"
_PROT = "ACDEFGHIKLMNPQRSTVWY"
OK, NO = "  ✓", "  ✗"


def _imp(seq, t):
    return SmartImporter.from_string(f">x\n{seq}\n", force_type=t)[0]


# ══ Batería de invariantes del motor ══════════════════════════════════════════

def _check(name, cond, detail=""):
    return (name, bool(cond), detail)


def internal_checks(seed=0):
    """Devuelve [(nombre, pasa, detalle)] — cada uno un invariante de no-corrupción."""
    rng = np.random.default_rng(seed)
    out = []

    # 1. pack/unpack de códigos aleatorios (la capa de bits)
    codes = rng.integers(0, 32, 5000, dtype=np.uint8)
    back = BitPacker.unpack(BitPacker.pack(codes), codes.size)
    out.append(_check("pack/unpack 5-bit (ida y vuelta)", np.array_equal(codes, back)))

    # 1b. pack sobre un array con stride (el bug real de v6: slices con paso)
    strided = codes[::3]
    back2 = BitPacker.unpack(BitPacker.pack(strided), strided.size)
    out.append(_check("pack con stride (no lee memoria lineal)",
                      np.array_equal(strided, back2)))

    # 2. codificar/decodificar ADN y proteína aleatorios
    for name, alpha, t in (("ADN", _DNA, SeqType.NUCLEOTIDE),
                           ("proteína", _PROT, SeqType.PROTEIN)):
        losses = 0
        for _ in range(200):
            s = "".join(rng.choice(list(alpha), rng.integers(1, 300)))
            if _imp(s, t).to_string() != s:
                losses += 1
        out.append(_check(f"encode/decode {name} (200 aleatorias)", losses == 0,
                          f"{losses} con pérdida" if losses else ""))

    # 3. el guard: proteína forzada como ADN debe FALLAR (no corromper)
    caught = False
    try:
        _imp("MKLPQEFILPQWYVHNDST" * 3, SeqType.NUCLEOTIDE)
    except SequenceValueError:
        caught = True
    out.append(_check("guard rechaza tipo equivocado", caught,
                      "" if caught else "¡corrompió en silencio!"))

    # 3b. el guard NO molesta a ADN con N reales dispersas
    ndna = "ATGCNNATGCGTANCGTAGCTAGC" + "ACGT" * 20
    ok_ndna = _imp(ndna, SeqType.NUCLEOTIDE).to_string() == ndna
    out.append(_check("guard tolera ADN con N legítimas", ok_ndna))

    # 4. el MSA no altera secuencias, en ambos alfabetos
    for name, base, alpha in (("ADN", "ATGCGTACGTAGCTAGCATCGATCG", _DNA),
                              ("proteína", "MKTIIALSYIFCLVFAQKLPGNDNST", _PROT)):
        seqs = []
        for _ in range(6):
            sl = list(base)
            for p in rng.choice(len(sl), 3, replace=False):
                sl[p] = rng.choice(list(alpha))
            seqs.append("".join(sl))
        aligned = align_multiple(seqs).aligned
        intact = all(a.replace("-", "") == o for a, o in zip(aligned, seqs))
        out.append(_check(f"MSA no corrompe {name}", intact))

    # 4b. _infer_type clasifica bien el conjunto
    ok_infer = (_infer_type(["ACGT", "ACGT"]) is SeqType.NUCLEOTIDE
                and _infer_type(["MKLW", "MKLF"]) is SeqType.PROTEIN)
    out.append(_check("inferencia de tipo del conjunto", ok_infer))

    # 5. tubería completa: ADN → traducir → proteína intacta
    nuc = _imp("ATG" + "".join(rng.choice(list(_DNA), 300)), SeqType.NUCLEOTIDE)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prot = SmartTranslator.translate(nuc).to_string()
    ok_pipe = (not prot) or _imp(prot, SeqType.PROTEIN).to_string() == prot
    out.append(_check("tubería import→traducir sin pérdida", ok_pipe))

    return out


# ══ Chequeo de un archivo real del usuario ════════════════════════════════════

def check_file(path):
    """Importa cada registro y verifica que se re-codifica idéntico. Reporta
    tipos, longitudes y cualquier registro con pérdida (posible dato corrupto)."""
    print(f"\nVerificando archivo: {path}")
    n, lossy, by_type = 0, [], {"NUCLEOTIDE": 0, "PROTEIN": 0}
    try:
        for rec in SmartImporter.stream(path):
            n += 1
            by_type[rec.seq_type.name] = by_type.get(rec.seq_type.name, 0) + 1
            txt = rec.to_string()
            # re-empaquetar con su MISMO tipo debe dar exactamente lo mismo
            try:
                again = _imp(txt, rec.seq_type).to_string()
                if again != txt:
                    lossy.append(rec.header[:50])
            except SequenceValueError:
                lossy.append(rec.header[:50] + " (rechazado al re-codificar)")
            if n >= 100_000:
                print("  (parado a 100.000 registros)")
                break
    except FileNotFoundError:
        print(f"{NO} no existe el archivo: {path}")
        return False
    except Exception as e:                              # noqa: BLE001
        print(f"{NO} error leyendo el archivo: {type(e).__name__}: {e}")
        return False

    print(f"  registros: {n:,}   "
          f"ADN: {by_type.get('NUCLEOTIDE', 0):,}   "
          f"proteína: {by_type.get('PROTEIN', 0):,}")
    if lossy:
        print(f"{NO} {len(lossy)} registro(s) con PÉRDIDA (posible corrupción):")
        for h in lossy[:10]:
            print(f"       - {h}")
        return False
    print(f"{OK} todos los registros se re-codifican intactos")
    return True


# ══ Main ══════════════════════════════════════════════════════════════════════

def main():
    print("═" * 62)
    print("  BioForge — CERTIFICADO DE INTEGRIDAD")
    print("═" * 62)

    checks = internal_checks()
    print("\nInvariantes del motor:")
    passed = 0
    for name, ok, detail in checks:
        mark = OK if ok else NO
        line = f"{mark} {name}"
        if detail:
            line += f"   [{detail}]"
        print(line)
        passed += ok

    engine_ok = passed == len(checks)
    print(f"\n  {passed}/{len(checks)} invariantes " +
          ("PASAN — motor íntegro ✓" if engine_ok else "— ¡HAY FALLOS! ✗"))

    file_ok = True
    if len(sys.argv) > 1:
        file_ok = check_file(sys.argv[1])

    print("═" * 62)
    ok = engine_ok and file_ok
    print("  RESULTADO:", "TODO ÍNTEGRO ✓" if ok else "REVISAR ✗")
    print("═" * 62)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
