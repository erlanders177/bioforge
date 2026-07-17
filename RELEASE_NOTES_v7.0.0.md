# BioForge v7.0.0 — The Evolution Front

**The largest release in the project's history.** An entire new level (L5), a trained
model, and a command-line tool — genome-agnostic, measured, and with honesty baked in.

---

## What's new

### Predict which mutations will rise — `bioforge-evolution`

Given dated sequences of a gene under selection (a flu HA across seasons, a gene under
pressure), BioForge now ranks the mutations most likely to rise next:

```bash
bioforge-evolution rank strains.fasta --top 20      # rank candidate mutations
bioforge-evolution rank strains.fasta --novel       # only never-seen mutations
bioforge-evolution backtest strains.fasta           # is it better than "tomorrow = today"?
bioforge-evolution lineages strains.fasta           # designate stable lineages
```

### Stable lineages without a phylogenetic tree

`designate_lineages` builds Pango/autolin-style stable lineages using a Genotype
Representation Index computed from two matrix multiplies over the MSA — no IQ-TREE, no
cluster. It runs on a laptop where phylogenetic methods need a server.

### A trained model that runs in pure NumPy

The mutation ranker is a small neural net (MLP 2×64). PyTorch was only the scaffolding —
the shipped model is a 39 KB `.npz` and three matrix multiplies. No PyTorch, no GPU. It
beats a plain linear model on all six held-out tests, and helps most when generalizing
to an influenza type it has never seen (cross-virus).

### Optional ESM-2 axis — with its leakage measured, not hidden

`pip install bioforge[ai]` adds a protein-language-model viability axis. Its pretraining
leakage is documented (AUC drops ~0.20 on data after its training cutoff), and it is off
by default.

---

## Requirements

- **Python 3.10 or newer**
- **NumPy 1.24 or newer** — the only required dependency
- **Windows, Linux or macOS** — the C engine ships pre-compiled; if it is unavailable on
  your platform, BioForge falls back to a transparent NumPy implementation
- **Optional** (`bioforge[ai]`): PyTorch 2.0+ and Transformers 4.30+, only for the ESM-2
  viability axis. The core tool and the trained ranker need none of this.

## Install

```bash
pip install --upgrade bioforge          # core
pip install --upgrade "bioforge[ai]"    # + optional ESM-2 viability axis
```

## Input format

The evolution tools read a FASTA where each record carries a **date in its header** — a
year, or `YYYY-MM` for month resolution:

```
>A/Sydney/5/2021|2021-03
MKTIIALSYIFCLVFA...
>strain_2019
MKTIIALSYIFCLVFA...
```

`rank` works on protein; pass `--translate` to translate nucleotide input first. Records
without a recognizable date are skipped.

---

## The honest part (this is a feature)

None of this is scientifically novel — DERIVE, EVEscape and Hie et al. already exist and
are better, with more resources. BioForge's value is not beating the state of the art. It
is being the integrated, accessible, honest box that runs on humble hardware and tells
you its own uncertainty instead of selling hype.

- Predicting exact *frequencies* is a dead end — it ties the naive "tomorrow = today"
  baseline at every horizon tested. So the tool ranks mutations instead, which is what
  the field actually measures.
- The physico-chemical "escape" axis turned out inverted — in flu HA the substitutions
  that rise are *conservative*. It measures viability, not escape. Replicated across
  H3N2, H1N1 and B.
- Every limit in this release is measured with confidence intervals and written into the
  README.

---

## By the numbers

- 454 tests passing
- Zero dependencies at inference (NumPy only; the C engine and trained ranker ship pre-compiled)
- Mutation ranking: cross-virus AUC approximately 0.77–0.95 on flu HA
- Trained model: 39 KB, three matrix multiplies, laptop-runnable

---

## Documentation and changelog

- Full documentation: [README](https://github.com/erlanders177/bioforge/blob/main/README.md)
- Full changelog: [`v6.3.0...v7.0.0`](https://github.com/erlanders177/bioforge/compare/v6.3.0...v7.0.0)
