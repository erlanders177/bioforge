"""
analyze.py
══════════════════════════════════════════════════════════════════════
Mutation analysis pipeline — amino acid level.

Compares two biological sequences and reports only the mutations that
actually change an amino acid. Silent (synonymous) DNA changes are
ignored entirely — they have no functional impact.

Pipeline
────────
  1. Load both FASTA files        (SmartImporter)
  2. Translate to protein          (SmartTranslator, if input is DNA)
  3. Align at protein level        (SequenceAligner)
  4. Report amino acid mutations   (substitutions, insertions, deletions)

Usage
─────
  python analyze.py reference.fa query.fa
  python analyze.py reference.fa query.fa --output report.md
  python analyze.py reference.fa query.fa --output report.txt
  python analyze.py --help
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from biocore import SeqType, SmartImporter, PackedSequence
from smart_translator import SmartTranslator
from aligner import AlignmentResult, SequenceAligner, format_alignment


# ── Amino acid full names ──────────────────────────────────────────────────────
_AA_NAMES: dict[str, str] = {
    "A": "Ala", "C": "Cys", "D": "Asp", "E": "Glu", "F": "Phe",
    "G": "Gly", "H": "His", "I": "Ile", "K": "Lys", "L": "Leu",
    "M": "Met", "N": "Asn", "P": "Pro", "Q": "Gln", "R": "Arg",
    "S": "Ser", "T": "Thr", "V": "Val", "W": "Trp", "Y": "Tyr",
    "*": "Stop", "-": "Gap", "X": "Unk",
}

# Conservative substitutions rarely affect protein function.
# Grouped by physicochemical similarity (charge, polarity, size).
_CONSERVATIVE: list[frozenset[str]] = [
    frozenset("ST"),        # small hydroxyl
    frozenset("DE"),        # acidic
    frozenset("KR"),        # basic
    frozenset("NQ"),        # amide
    frozenset("LIVM"),      # aliphatic hydrophobic
    frozenset("FYW"),       # aromatic
    frozenset("AG"),        # tiny
]


def _change_type(a: str, b: str) -> str:
    """Classify an amino acid substitution as conservative or radical."""
    for group in _CONSERVATIVE:
        if a in group and b in group:
            return "conservative"
    return "radical"


# ══════════════════════════════════════════════════════════════════════════════
# §1  RESULT DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AnalysisResult:
    ref_header:   str
    query_header: str
    ref_protein:  PackedSequence
    qry_protein:  PackedSequence
    alignment:    AlignmentResult
    was_dna:      bool   # True if input was nucleotide (translated internally)


# ══════════════════════════════════════════════════════════════════════════════
# §2  PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run(ref_path: str, query_path: str) -> AnalysisResult:
    """
    Execute the full analysis pipeline.

    Parameters
    ----------
    ref_path   : path to reference FASTA file
    query_path : path to query FASTA file

    Returns
    -------
    AnalysisResult
    """
    # ── Load ──────────────────────────────────────────────────────────────────
    ref_seqs   = SmartImporter.from_file(ref_path)
    query_seqs = SmartImporter.from_file(query_path)

    if not ref_seqs:
        raise ValueError(f"No se encontraron secuencias en: {ref_path}")
    if not query_seqs:
        raise ValueError(f"No se encontraron secuencias en: {query_path}")

    ref_seq   = ref_seqs[0]
    query_seq = query_seqs[0]

    if ref_seq.seq_type != query_seq.seq_type:
        raise TypeError(
            f"Tipos incompatibles: referencia es {ref_seq.seq_type.name}, "
            f"query es {query_seq.seq_type.name}. "
            "Ambos archivos deben contener el mismo tipo de secuencia."
        )

    was_dna = ref_seq.seq_type == SeqType.NUCLEOTIDE

    # ── Translate if DNA ───────────────────────────────────────────────────────
    if was_dna:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ref_prot   = SmartTranslator.translate(ref_seq)
            query_prot = SmartTranslator.translate(query_seq)
    else:
        ref_prot   = ref_seq
        query_prot = query_seq

    # ── Align at protein level ─────────────────────────────────────────────────
    alignment = SequenceAligner.align(ref_prot, query_prot)

    return AnalysisResult(
        ref_header   = ref_seq.header,
        query_header = query_seq.header,
        ref_protein  = ref_prot,
        qry_protein  = query_prot,
        alignment    = alignment,
        was_dna      = was_dna,
    )


# ══════════════════════════════════════════════════════════════════════════════
# §3  REPORT FORMATTER
# ══════════════════════════════════════════════════════════════════════════════

def build_report(result: AnalysisResult) -> str:
    """Format an AnalysisResult into a human-readable report string."""
    aln  = result.alignment
    W    = 68
    sep  = "─" * W
    dbl  = "═" * W
    lines: list[str] = []

    def add(*text: str) -> None:
        lines.extend(text)

    # ── Header ─────────────────────────────────────────────────────────────────
    add(
        dbl,
        "  MUTATION ANALYSIS REPORT",
        dbl,
        "",
        f"  Reference : {result.ref_header[:55]}",
        f"  Query     : {result.query_header[:55]}",
        f"  Input     : {'DNA (translated internally)' if result.was_dna else 'Protein'}",
        "",
    )

    # ── Protein summary ────────────────────────────────────────────────────────
    add(
        sep,
        "  PROTEIN ALIGNMENT SUMMARY",
        sep,
        f"  Reference length : {result.ref_protein.n_symbols:>6} aa",
        f"  Query length     : {result.qry_protein.n_symbols:>6} aa",
        f"  Identity         : {aln.identity:>6.1%}  "
        f"({aln.n_matches}/{aln.n_matches + aln.n_mismatches + aln.n_gaps} positions)",
        f"  Matches          : {aln.n_matches:>6}",
        f"  Substitutions    : {aln.n_mismatches:>6}",
        f"  Indels (aa)      : {aln.n_gaps:>6}",
        "",
    )

    # ── Alignment visual ───────────────────────────────────────────────────────
    add(sep, "  PROTEIN ALIGNMENT", sep)
    add(format_alignment(aln, width=60))

    # ── Mutations ──────────────────────────────────────────────────────────────
    subs = [m for m in aln.mutations if m.kind == "substitution"]
    dels = [m for m in aln.mutations if m.kind == "deletion"]
    ins  = [m for m in aln.mutations if m.kind == "insertion"]

    total = len(aln.mutations)
    add(sep, f"  AMINO ACID MUTATIONS  ({total} total)", sep)

    if total == 0:
        add("  No mutations found. Sequences are functionally identical.", "")
    else:
        # Substitutions
        if subs:
            add(f"  Substitutions ({len(subs)}):")
            for m in subs:
                ref_name   = _AA_NAMES.get(m.sym_a, m.sym_a)
                qry_name   = _AA_NAMES.get(m.sym_b, m.sym_b)
                change     = _change_type(m.sym_a, m.sym_b)
                impact_tag = "(conservative)" if change == "conservative" else "(RADICAL)"
                add(
                    f"    pos {m.pos_a + 1:>4}  "
                    f"{m.sym_a} → {m.sym_b}  "
                    f"[{ref_name} → {qry_name}]  "
                    f"{impact_tag}"
                )
            add("")

        # Deletions
        if dels:
            add(f"  Deletions in query ({len(dels)} aa):")
            for m in dels:
                aa_name = _AA_NAMES.get(m.sym_a, m.sym_a)
                add(f"    pos {m.pos_a + 1:>4}  {m.sym_a} [{aa_name}] missing in query")
            add("")

        # Insertions
        if ins:
            add(f"  Insertions in query ({len(ins)} aa):")
            for m in ins:
                aa_name = _AA_NAMES.get(m.sym_b, m.sym_b)
                add(f"    pos {m.pos_b + 1:>4}  {m.sym_b} [{aa_name}] not present in reference")
            add("")

    # ── Interpretation ─────────────────────────────────────────────────────────
    if total > 0:
        radical_subs = [m for m in subs if _change_type(m.sym_a, m.sym_b) == "radical"]
        add(sep, "  INTERPRETATION", sep)
        if radical_subs:
            add(
                f"  {len(radical_subs)} radical substitution(s) detected — "
                "these changes alter the physicochemical",
                "  properties of the amino acid and are likely to affect protein function.",
            )
        elif subs:
            add(
                "  All substitutions are conservative — the physicochemical",
                "  properties of the amino acids are similar. Functional impact is likely low.",
            )
        if dels or ins:
            add(
                f"  {len(dels) + len(ins)} indel position(s) detected — "
                "insertions/deletions shift the reading",
                "  frame or alter protein length, often with significant functional impact.",
            )
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
            Mutation analysis at the amino acid level.
            Silent DNA changes (same amino acid) are ignored.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("reference", help="Reference FASTA file (.fa / .fasta)")
    parser.add_argument("query",     help="Query FASTA file (.fa / .fasta)")
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
        result = run(args.reference, args.query)
    except (ValueError, TypeError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
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
