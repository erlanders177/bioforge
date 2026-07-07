# References & inspiration

BioForge implements published, peer-reviewed algorithms **from scratch**. It
does **not** include, copy, or link any third-party source code — it builds on
the *ideas* described in the scientific literature. Algorithms and mathematical
methods are not copyrightable; reimplementing a published method (with proper
citation) is standard scientific practice. This page records the works that
inspired each level, in gratitude and for academic honesty.

## Level 4 — Genome mapper (`minimizers.py`, `refindex.py`, `genomemap.py`)

The seed-chain-align design mirrors the approach popularised by **minimap2**.

- **Li, H. (2018).** Minimap2: pairwise alignment for nucleotide sequences.
  *Bioinformatics*, 34(18), 3094–3100.
  https://doi.org/10.1093/bioinformatics/bty191
  Source (MIT): https://github.com/lh3/minimap2
  - Inspired: the seed → chain → align pipeline; the chaining dynamic program
    `f(i) = max_j { f(j) + α(j,i) − β(j,i) }` with gap cost
    `γ(l) = 0.01·w̄·|l| + 0.5·log₂|l|`; frequent-minimizer filtering (`max_occ`);
    the PAF output format.

- **Roberts, M., Hayes, W., Hunt, B. R., Mount, S. M., & Yorke, J. A. (2004).**
  Reducing storage requirements for biological sequence comparison.
  *Bioinformatics*, 20(18), 3363–3369.
  https://doi.org/10.1093/bioinformatics/bth408
  - Inspired: the (w, k) minimizer sampling scheme.

- **Schleimer, S., Wilkerson, D. S., & Aiken, A. (2003).** Winnowing: local
  algorithms for document fingerprinting. *SIGMOD.* — the "winnowing" idea
  underlying minimizers.

## Level 3 — Pairwise alignment (`aligner.py`)

- **Needleman, S. B., & Wunsch, C. D. (1970).** A general method applicable to
  the search for similarities in the amino acid sequence of two proteins.
  *J. Mol. Biol.*, 48(3), 443–453. — global alignment.
- **Smith, T. F., & Waterman, M. S. (1981).** Identification of common molecular
  subsequences. *J. Mol. Biol.*, 147(1), 195–197. — local alignment.

## Compression / IO

- **BGZF / bgzip** — the block-gzip format from the SAM/BAM ecosystem
  (Li et al., *The Sequence Alignment/Map format and SAMtools*, Bioinformatics
  2009), reimplemented in `bgzf.py` for parallel-decompressible `.gz`.
- **libdeflate** — Eric Biggers' fast DEFLATE library (used optionally by the
  C engine for ~2× gzip decompression).

---

BioForge is an independent project and is **not affiliated with or endorsed by**
the authors or maintainers of any of the above works.
