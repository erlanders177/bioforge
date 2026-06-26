"""
analyze.py
══════════════════════════════════════════════════════════════════════
Mutation analysis pipeline — nucleotide and/or amino acid level.

Three analysis modes:
  --mode dna     : compare at nucleotide level only (A/C/G/T mutations)
  --mode protein  : compare at amino acid level only (translate if DNA input)
  --mode both     : full report — nucleotide mutations + amino acid impact
                    (default when input is DNA)

Silent (synonymous) DNA changes that do NOT alter the amino acid are
labelled as such in 'both' mode and excluded from 'protein' mode entirely.

Pipeline
────────
  1. Load both FASTA files            (SmartImporter)
  2. [dna / both]  align at DNA level (SequenceAligner on nucleotides)
  3. [protein/both] translate + align at protein level (SmartTranslator →
                    SequenceAligner on amino acids)
  4. Build and output report

Usage
─────
  python analyze.py reference.fa query.fa
  python analyze.py reference.fa query.fa --mode dna
  python analyze.py reference.fa query.fa --mode protein
  python analyze.py reference.fa query.fa --mode both --output report.md
  python analyze.py --help
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from .biocore import SeqType, SmartImporter, PackedSequence
from .biocore import BioForgeError, SequenceTypeError, SequenceValueError
from .smart_translator import SmartTranslator
from .aligner import AlignmentResult, SequenceAligner, format_alignment


# ── Amino acid full names ──────────────────────────────────────────────────────
_AA_NAMES: dict[str, str] = {
    "A": "Ala", "C": "Cys", "D": "Asp", "E": "Glu", "F": "Phe",
    "G": "Gly", "H": "His", "I": "Ile", "K": "Lys", "L": "Leu",
    "M": "Met", "N": "Asn", "P": "Pro", "Q": "Gln", "R": "Arg",
    "S": "Ser", "T": "Thr", "V": "Val", "W": "Trp", "Y": "Tyr",
    "*": "Stop", "-": "Gap", "X": "Unk",
}

_NUC_NAMES: dict[str, str] = {
    "A": "Adenina", "C": "Citosina", "G": "Guanina",
    "T": "Timina",  "U": "Uracilo", "N": "Ambigua", "-": "Gap",
}

# Conservative substitution groups (physicochemical similarity)
_CONSERVATIVE: list[frozenset[str]] = [
    frozenset("ST"), frozenset("DE"), frozenset("KR"), frozenset("NQ"),
    frozenset("LIVM"), frozenset("FYW"), frozenset("AG"),
]


def _change_type(a: str, b: str) -> str:
    for group in _CONSERVATIVE:
        if a in group and b in group:
            return "conservative"
    return "radical"


# ══════════════════════════════════════════════════════════════════════════════
# §1  RESULT DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AnalysisResult:
    ref_header:    str
    query_header:  str
    ref_seq:       PackedSequence          # original input sequence
    qry_seq:       PackedSequence          # original input sequence
    dna_alignment: Optional[AlignmentResult]   # None if mode=protein or input=protein
    ref_protein:   Optional[PackedSequence]    # None if mode=dna
    qry_protein:   Optional[PackedSequence]    # None if mode=dna
    aa_alignment:  Optional[AlignmentResult]   # None if mode=dna
    mode:          str
    was_dna:       bool


# ══════════════════════════════════════════════════════════════════════════════
# §2  PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run(
    ref_path:  str,
    query_path: str,
    mode:      Literal["dna", "protein", "both"] = "both",
) -> AnalysisResult:
    """
    Execute the full analysis pipeline.

    Parameters
    ----------
    ref_path   : path to reference FASTA file
    query_path : path to query FASTA file
    mode       : 'dna' | 'protein' | 'both'
                 If input is protein, 'dna' and 'both' fall back to 'protein'.

    Returns
    -------
    AnalysisResult
    """
    if mode not in ("dna", "protein", "both"):
        raise ValueError(
            f"mode debe ser 'dna', 'protein' o 'both', se recibió {mode!r}."
        )
    ref_seqs   = SmartImporter.from_file(ref_path)
    query_seqs = SmartImporter.from_file(query_path)

    if not ref_seqs:
        raise SequenceValueError(
            f"No se encontraron secuencias en: {ref_path}. "
            "Comprueba que el archivo FASTA tenga al menos un registro con '>'."
        )
    if not query_seqs:
        raise SequenceValueError(
            f"No se encontraron secuencias en: {query_path}. "
            "Comprueba que el archivo FASTA tenga al menos un registro con '>'."
        )

    ref_seq   = ref_seqs[0]
    query_seq = query_seqs[0]

    if ref_seq.seq_type != query_seq.seq_type:
        raise SequenceTypeError(
            f"Tipos incompatibles: referencia es {ref_seq.seq_type.name} "
            f"pero query es {query_seq.seq_type.name}. "
            "Ambos archivos deben contener el mismo tipo de secuencia."
        )

    was_dna = ref_seq.seq_type == SeqType.NUCLEOTIDE

    # Si la entrada es proteína no se puede hacer análisis de nucleótidos
    effective_mode = mode
    if not was_dna and mode in ("dna", "both"):
        effective_mode = "protein"

    # ── DNA alignment ──────────────────────────────────────────────────────────
    dna_aln: Optional[AlignmentResult] = None
    if was_dna and effective_mode in ("dna", "both"):
        dna_aln = SequenceAligner.align(ref_seq, query_seq)

    # ── Protein alignment ──────────────────────────────────────────────────────
    ref_prot = qry_prot = aa_aln = None
    if effective_mode in ("protein", "both"):
        if was_dna:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                ref_prot = SmartTranslator.translate(ref_seq)
                qry_prot = SmartTranslator.translate(query_seq)
        else:
            ref_prot = ref_seq
            qry_prot = query_seq
        aa_aln = SequenceAligner.align(ref_prot, qry_prot)

    return AnalysisResult(
        ref_header    = ref_seq.header,
        query_header  = query_seq.header,
        ref_seq       = ref_seq,
        qry_seq       = query_seq,
        dna_alignment = dna_aln,
        ref_protein   = ref_prot,
        qry_protein   = qry_prot,
        aa_alignment  = aa_aln,
        mode          = effective_mode,
        was_dna       = was_dna,
    )


# ══════════════════════════════════════════════════════════════════════════════
# §3  REPORT FORMATTER
# ══════════════════════════════════════════════════════════════════════════════

def build_report(result: AnalysisResult) -> str:
    W   = 68
    sep = "─" * W
    dbl = "═" * W
    lines: list[str] = []

    def add(*text: str) -> None:
        lines.extend(text)

    # ── Header ─────────────────────────────────────────────────────────────────
    add(dbl, "  MUTATION ANALYSIS REPORT", dbl, "")
    add(f"  Reference : {result.ref_header[:55]}")
    add(f"  Query     : {result.query_header[:55]}")
    tipo = "DNA" if result.was_dna else "Protein"
    add(f"  Input     : {tipo}   |   Mode: {result.mode}", "")

    # ── DNA section ────────────────────────────────────────────────────────────
    if result.dna_alignment is not None:
        aln = result.dna_alignment
        add(sep, "  DNA ALIGNMENT", sep)
        add(f"  Reference length : {result.ref_seq.n_symbols:>6} nt")
        add(f"  Query length     : {result.qry_seq.n_symbols:>6} nt")
        add(f"  Identity         : {aln.identity:>6.1%}  "
            f"({aln.n_matches}/{aln.n_matches + aln.n_mismatches + aln.n_gaps} positions)")
        add(f"  Matches          : {aln.n_matches:>6}")
        add(f"  Substitutions    : {aln.n_mismatches:>6}")
        add(f"  Indels (nt)      : {aln.n_gaps:>6}", "")

        add(format_alignment(aln, width=60))

        nuc_muts = aln.mutations
        add(sep, f"  NUCLEOTIDE MUTATIONS  ({len(nuc_muts)} total)", sep)
        if not nuc_muts:
            add("  No nucleotide mutations.", "")
        else:
            subs = [m for m in nuc_muts if m.kind == "substitution"]
            dels = [m for m in nuc_muts if m.kind == "deletion"]
            ins  = [m for m in nuc_muts if m.kind == "insertion"]
            if subs:
                add(f"  Substitutions ({len(subs)}):")
                for m in subs:
                    rn = _NUC_NAMES.get(m.sym_a, m.sym_a)
                    qn = _NUC_NAMES.get(m.sym_b, m.sym_b)
                    # Mark synonymous if both modes active (protein section will clarify)
                    add(f"    pos {m.pos_a + 1:>5}  {m.sym_a} → {m.sym_b}  [{rn} → {qn}]")
                add("")
            if dels:
                add(f"  Deletions in query ({len(dels)} nt):")
                for m in dels:
                    add(f"    pos {m.pos_a + 1:>5}  {m.sym_a} [{_NUC_NAMES.get(m.sym_a, m.sym_a)}] missing")
                add("")
            if ins:
                add(f"  Insertions in query ({len(ins)} nt):")
                for m in ins:
                    add(f"    pos {m.pos_b + 1:>5}  {m.sym_b} [{_NUC_NAMES.get(m.sym_b, m.sym_b)}] inserted")
                add("")

    # ── Protein section ────────────────────────────────────────────────────────
    if result.aa_alignment is not None:
        aln  = result.aa_alignment
        add(sep, "  PROTEIN ALIGNMENT", sep)
        add(f"  Reference length : {result.ref_protein.n_symbols:>6} aa")
        add(f"  Query length     : {result.qry_protein.n_symbols:>6} aa")
        add(f"  Identity         : {aln.identity:>6.1%}  "
            f"({aln.n_matches}/{aln.n_matches + aln.n_mismatches + aln.n_gaps} positions)")
        add(f"  Matches          : {aln.n_matches:>6}")
        add(f"  Substitutions    : {aln.n_mismatches:>6}")
        add(f"  Indels (aa)      : {aln.n_gaps:>6}", "")

        if result.was_dna:
            add(f"  Reference protein : {result.ref_protein.to_string()[:60]}")
            add(f"  Query protein     : {result.qry_protein.to_string()[:60]}", "")

        add(format_alignment(aln, width=60))

        subs = [m for m in aln.mutations if m.kind == "substitution"]
        dels = [m for m in aln.mutations if m.kind == "deletion"]
        ins  = [m for m in aln.mutations if m.kind == "insertion"]
        total = len(aln.mutations)

        add(sep, f"  AMINO ACID MUTATIONS  ({total} total)", sep)
        if total == 0:
            add("  No amino acid mutations. Sequences are functionally identical.", "")
            if result.dna_alignment and result.dna_alignment.n_mismatches > 0:
                add("  Note: DNA differences exist but are all synonymous (silent).", "")
        else:
            if subs:
                add(f"  Substitutions ({len(subs)}):")
                for m in subs:
                    rn     = _AA_NAMES.get(m.sym_a, m.sym_a)
                    qn     = _AA_NAMES.get(m.sym_b, m.sym_b)
                    impact = _change_type(m.sym_a, m.sym_b)
                    tag    = "(conservative)" if impact == "conservative" else "(RADICAL)"
                    add(f"    pos {m.pos_a + 1:>4}  {m.sym_a} → {m.sym_b}  "
                        f"[{rn} → {qn}]  {tag}")
                add("")
            if dels:
                add(f"  Deletions in query ({len(dels)} aa):")
                for m in dels:
                    add(f"    pos {m.pos_a + 1:>4}  {m.sym_a} [{_AA_NAMES.get(m.sym_a, m.sym_a)}] missing")
                add("")
            if ins:
                add(f"  Insertions in query ({len(ins)} aa):")
                for m in ins:
                    add(f"    pos {m.pos_b + 1:>4}  {m.sym_b} [{_AA_NAMES.get(m.sym_b, m.sym_b)}] inserted")
                add("")

            # Interpretation
            radical = [m for m in subs if _change_type(m.sym_a, m.sym_b) == "radical"]
            add(sep, "  INTERPRETATION", sep)
            if radical:
                add(f"  {len(radical)} radical substitution(s) — likely affect protein function.")
            elif subs:
                add("  All substitutions conservative — functional impact likely low.")
            if dels or ins:
                add(f"  {len(dels)+len(ins)} indel position(s) — often high functional impact.")
            add("")

    add(dbl)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# §4  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="analyze",
        description=textwrap.dedent("""\
            Mutation analysis at nucleotide and/or amino acid level.
            Modes: dna | protein | both (default)
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("reference", help="Reference FASTA file")
    parser.add_argument("query",     help="Query FASTA file")
    parser.add_argument(
        "--mode", "-m",
        choices=["dna", "protein", "both"],
        default="both",
        help="Analysis level: dna, protein, or both (default: both)",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Save report to file (.md or .txt). Prints to screen if omitted.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    args = _parse_args(argv)

    try:
        result = run(args.reference, args.query, mode=args.mode)
    except BioForgeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"Archivo no encontrado: {exc.filename}", file=sys.stderr)
        return 1
    except PermissionError as exc:
        print(f"Sin permiso para leer: {exc.filename}", file=sys.stderr)
        return 1
    except MemoryError:
        print(
            "Error: memoria insuficiente para el alineamiento. "
            "Las secuencias son demasiado largas para el modo global.",
            file=sys.stderr,
        )
        return 1
    except OSError as exc:
        print(f"Error de E/S: {exc}", file=sys.stderr)
        return 1

    report = build_report(result)

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Report saved to: {args.output}")
    else:
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
