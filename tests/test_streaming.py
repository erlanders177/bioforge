"""
tests/test_streaming.py
Pruebas del parser por lotes y de la API columnar de v2.0:
  - SmartImporter.stream() / stream_fastq()           (un objeto por registro)
  - SmartImporter.stream_batches() / stream_fastq_batches()  (columnar)
  - FastqRecord, SequenceBatch, ReadBatch

Cubren correctitud frente a la carga clásica (from_file), longitud fija e
irregular, calidades Phred exactas, filtrado, y rutas de error.

Ejecutar:
    pytest tests/test_streaming.py -v
"""

import gzip
import os
import random

import numpy as np
import pytest

from bioforge import (
    SmartImporter, SeqType, FastqRecord, SequenceBatch, ReadBatch,
    SequenceTypeError,
)

QCHARS = "".join(chr(33 + q) for q in range(40))   # Phred 0..39
BASES = "ACGT"
_BASE_ID = {"A": 0, "C": 1, "G": 2, "T": 3}


# ── Helpers para escribir archivos de prueba ────────────────────────────────

def _write_fasta(path, records, width=60):
    with open(path, "w", newline="\n") as f:
        for header, seq in records:
            f.write(f">{header}\n")
            for j in range(0, len(seq), width):
                f.write(seq[j:j + width] + "\n")


def _write_fastq(path, records):
    with open(path, "w", newline="\n") as f:
        for header, seq, qual in records:
            qstr = "".join(QCHARS[q] for q in qual)
            f.write(f"@{header}\n{seq}\n+\n{qstr}\n")


def _rand_seq(n, rng):
    return "".join(rng.choice(BASES) for _ in range(n))


# ════════════════════════════════════════════════════════════════════════════
# stream()  —  FASTA por registro
# ════════════════════════════════════════════════════════════════════════════

def test_stream_fasta_matches_from_file(tmp_path):
    rng = random.Random(1)
    recs = [(f"seq{i} desc", _rand_seq(rng.randint(10, 400), rng))
            for i in range(300)]
    p = tmp_path / "a.fasta"
    _write_fasta(p, recs)

    ref = SmartImporter.from_file(str(p))
    got = list(SmartImporter.stream(str(p)))

    assert len(got) == len(ref) == len(recs)
    for r, g, (h, s) in zip(ref, got, recs):
        assert g.header == h
        assert g.n_symbols == len(s)
        assert g.to_string() == s == r.to_string()


def test_stream_fasta_multiline_and_blank_lines(tmp_path):
    p = tmp_path / "ml.fasta"
    p.write_text(">x\nACGT\nACGT\n\n>y\nGGGG\n", encoding="ascii")
    got = list(SmartImporter.stream(str(p)))
    assert [g.header for g in got] == ["x", "y"]
    assert got[0].to_string() == "ACGTACGT"
    assert got[1].to_string() == "GGGG"


def test_stream_force_type_protein(tmp_path):
    # Secuencia sin residuos exclusivos de proteína → se forzaría a ADN sin force
    p = tmp_path / "prot.fasta"
    _write_fasta(p, [("p1", "ACDGHKMNRSTVWY")])
    got = list(SmartImporter.stream(str(p), force_type=SeqType.PROTEIN))
    assert got[0].seq_type == SeqType.PROTEIN


# ════════════════════════════════════════════════════════════════════════════
# stream_fastq()  —  FASTQ por registro
# ════════════════════════════════════════════════════════════════════════════

def test_stream_fastq_sequence_and_quality(tmp_path):
    rng = random.Random(2)
    recs = []
    for i in range(200):
        L = rng.randint(20, 300)
        recs.append((f"r{i}", _rand_seq(L, rng),
                     [rng.randint(0, 39) for _ in range(L)]))
    p = tmp_path / "a.fastq"
    _write_fastq(p, recs)

    got = list(SmartImporter.stream_fastq(str(p)))
    assert len(got) == len(recs)
    for (h, s, q), rec in zip(recs, got):
        assert isinstance(rec, FastqRecord)
        assert rec.sequence.to_string() == s
        assert list(rec.quality) == q
        assert rec.sequence.seq_type == SeqType.NUCLEOTIDE


def test_fastqrecord_quality_helpers():
    from bioforge import PackedSequence, BitPacker
    codes = np.array([0, 1, 2, 3], dtype=np.uint8)
    seq = PackedSequence("h", SeqType.NUCLEOTIDE, 4, BitPacker.pack(codes))
    rec = FastqRecord(sequence=seq, quality=np.array([10, 20, 30, 40], np.uint8))
    assert rec.mean_quality == pytest.approx(25.0)
    assert rec.passes_quality(20) is True
    assert rec.passes_quality(30) is False


def test_fastq_phred_offset_decoding(tmp_path):
    # '!' = Phred 0, 'I' = Phred 40
    p = tmp_path / "q.fastq"
    p.write_text("@r\nACGT\n+\n!I!I\n", encoding="ascii")
    rec = next(SmartImporter.stream_fastq(str(p)))
    assert list(rec.quality) == [0, 40, 0, 40]


# ════════════════════════════════════════════════════════════════════════════
# stream_batches()  —  FASTA columnar
# ════════════════════════════════════════════════════════════════════════════

def test_sequence_batch_matches_stream(tmp_path):
    rng = random.Random(3)
    recs = [(f"g{i}", _rand_seq(rng.randint(20, 300), rng)) for i in range(500)]
    p = tmp_path / "b.fasta"
    _write_fasta(p, recs)

    flat = []
    nbatch = 0
    for batch in SmartImporter.stream_batches(str(p)):
        assert isinstance(batch, SequenceBatch)
        nbatch += 1
        for i in range(len(batch)):
            ps = batch[i]
            flat.append((ps.header, ps.to_string()))
    assert flat == recs
    assert nbatch >= 1


def test_sequence_batch_indexing_and_iter(tmp_path):
    recs = [("a", "ACGT"), ("b", "GGGGCCCC"), ("c", "TTTT")]
    p = tmp_path / "c.fasta"
    _write_fasta(p, recs)
    batch = next(SmartImporter.stream_batches(str(p)))
    assert len(batch) == 3
    assert batch.header(1) == "b"
    assert batch[1].to_string() == "GGGGCCCC"
    assert batch[-1].to_string() == "TTTT"
    assert [s.to_string() for s in batch] == ["ACGT", "GGGGCCCC", "TTTT"]
    with pytest.raises(IndexError):
        _ = batch[99]


# ════════════════════════════════════════════════════════════════════════════
# stream_fastq_batches()  —  FASTQ columnar
# ════════════════════════════════════════════════════════════════════════════

def test_read_batch_fixed_length_is_2d(tmp_path):
    rng = random.Random(4)
    recs = [(f"r{i}", _rand_seq(150, rng),
             [rng.randint(0, 39) for _ in range(150)]) for i in range(400)]
    p = tmp_path / "fix.fastq"
    _write_fastq(p, recs)

    batches = list(SmartImporter.stream_fastq_batches(str(p)))
    assert all(isinstance(b, ReadBatch) for b in batches)
    # longitud fija → camino 2-D
    assert batches[0]._fixed_len == 150
    assert batches[0]._qual.ndim == 2


def test_read_batch_mean_quality_matches_per_record(tmp_path):
    rng = random.Random(5)
    recs = []
    for i in range(600):
        L = 100
        q = [rng.randint(0, 39) for _ in range(L)]
        recs.append((f"r{i}", _rand_seq(L, rng), q))
    p = tmp_path / "m.fastq"
    _write_fastq(p, recs)

    ref = np.array([np.mean(q) for _, _, q in recs])
    got = np.concatenate(
        [b.mean_quality() for b in SmartImporter.stream_fastq_batches(str(p))])
    assert np.allclose(got, ref)


def test_read_batch_ragged_mean_quality(tmp_path):
    rng = random.Random(6)
    recs = []
    for i in range(600):
        L = rng.randint(50, 400)
        q = [rng.randint(0, 39) for _ in range(L)]
        recs.append((f"v{i}", _rand_seq(L, rng), q))
    p = tmp_path / "v.fastq"
    _write_fastq(p, recs)

    seen_ragged = False
    ref = np.array([np.mean(q) for _, _, q in recs])
    means = []
    for b in SmartImporter.stream_fastq_batches(str(p)):
        if b._fixed_len == 0:
            seen_ragged = True
        means.append(b.mean_quality())
    assert seen_ragged
    assert np.allclose(np.concatenate(means), ref)


def test_read_batch_passes_matches_threshold(tmp_path):
    rng = random.Random(7)
    recs = [(f"r{i}", _rand_seq(120, rng),
             [rng.randint(0, 39) for _ in range(120)]) for i in range(500)]
    p = tmp_path / "pass.fastq"
    _write_fastq(p, recs)

    ref_means = np.array([np.mean(q) for _, _, q in recs])
    for thr in (10, 20, 30):
        mask = np.concatenate(
            [b.passes(thr) for b in SmartImporter.stream_fastq_batches(str(p))])
        assert np.array_equal(mask, ref_means >= thr)


def test_read_batch_filter_fixed(tmp_path):
    rng = random.Random(8)
    recs = [(f"r{i}", _rand_seq(80, rng),
             [rng.randint(0, 39) for _ in range(80)]) for i in range(700)]
    p = tmp_path / "ff.fastq"
    _write_fastq(p, recs)

    kept_total = 0
    for b in SmartImporter.stream_fastq_batches(str(p)):
        kept = b.filter(b.passes(20))
        assert isinstance(kept, ReadBatch)
        if len(kept):
            assert bool((kept.mean_quality() >= 20).all())
            # la primera superviviente conserva su secuencia/calidad
            idx = np.flatnonzero(b.passes(20))
            assert kept[0].sequence.to_string() == b[int(idx[0])].sequence.to_string()
            assert list(kept[0].quality) == list(b[int(idx[0])].quality)
        kept_total += len(kept)
    ref = int((np.array([np.mean(q) for _, _, q in recs]) >= 20).sum())
    assert kept_total == ref


def test_read_batch_filter_ragged(tmp_path):
    rng = random.Random(9)
    recs = [(f"v{i}", _rand_seq(rng.randint(40, 200), rng), None)
            for i in range(400)]
    recs = [(h, s, [rng.randint(0, 39) for _ in range(len(s))]) for h, s, _ in recs]
    p = tmp_path / "fr.fastq"
    _write_fastq(p, recs)

    for b in SmartImporter.stream_fastq_batches(str(p)):
        assert b._fixed_len == 0
        mask = b.passes(25)
        kept = b.filter(mask)
        idx = np.flatnonzero(mask)
        assert len(kept) == len(idx)
        for k, j in enumerate(idx):
            assert kept[k].sequence.to_string() == b[int(j)].sequence.to_string()
            assert list(kept[k].quality) == list(b[int(j)].quality)


def test_read_batch_materialize_record(tmp_path):
    rng = random.Random(10)
    recs = [(f"r{i}", _rand_seq(60, rng),
             [rng.randint(0, 39) for _ in range(60)]) for i in range(50)]
    p = tmp_path / "mat.fastq"
    _write_fastq(p, recs)
    batch = next(SmartImporter.stream_fastq_batches(str(p)))
    rec = batch[5]
    assert isinstance(rec, FastqRecord)
    assert rec.sequence.to_string() == recs[5][1]
    assert list(rec.quality) == recs[5][2]
    assert batch.header(5) == "r5"


def test_filter_mask_wrong_size_raises(tmp_path):
    p = tmp_path / "w.fastq"
    _write_fastq(p, [("r", "ACGT", [10, 10, 10, 10])])
    batch = next(SmartImporter.stream_fastq_batches(str(p)))
    from bioforge import SequenceValueError
    with pytest.raises(SequenceValueError):
        batch.filter(np.array([True, False]))


# ════════════════════════════════════════════════════════════════════════════
# Coherencia columnar  ↔  por registro
# ════════════════════════════════════════════════════════════════════════════

def test_columnar_equals_per_record_fastq(tmp_path):
    rng = random.Random(12)
    recs = [(f"r{i}", _rand_seq(150, rng),
             [rng.randint(0, 39) for _ in range(150)]) for i in range(1000)]
    p = tmp_path / "eq.fastq"
    _write_fastq(p, recs)

    per_record = list(SmartImporter.stream_fastq(str(p)))
    columnar = []
    for b in SmartImporter.stream_fastq_batches(str(p)):
        columnar.extend(b[i] for i in range(len(b)))

    assert len(per_record) == len(columnar) == len(recs)
    for a, c in zip(per_record, columnar):
        assert a.sequence.to_string() == c.sequence.to_string()
        assert list(a.quality) == list(c.quality)
        assert a.sequence.header == c.sequence.header


def test_empty_file(tmp_path):
    p = tmp_path / "empty.fasta"
    p.write_text("", encoding="ascii")
    assert list(SmartImporter.stream(str(p))) == []
    assert list(SmartImporter.stream_batches(str(p))) == []


# ════════════════════════════════════════════════════════════════════════════
# GC content  y  k-mer spectrum  (columnar, vectorizado)
# ════════════════════════════════════════════════════════════════════════════

def _naive_gc(s):
    return (s.count("C") + s.count("G")) / len(s) if s else 0.0


def _naive_kmers(seqs, k):
    out = np.zeros(4 ** k, dtype=np.int64)
    pw = [4 ** (k - 1 - i) for i in range(k)]
    for s in seqs:
        for j in range(len(s) - k + 1):
            win = s[j:j + k]
            if any(b not in _BASE_ID for b in win):
                continue
            out[sum(_BASE_ID[win[i]] * pw[i] for i in range(k))] += 1
    return out


@pytest.mark.parametrize("fixed", [True, False])
def test_gc_content_matches_naive(tmp_path, fixed):
    rng = random.Random(20 + int(fixed))
    recs = []
    for i in range(300):
        L = 120 if fixed else rng.randint(40, 200)
        recs.append((f"r{i}", _rand_seq(L, rng),
                     [rng.randint(0, 39) for _ in range(L)]))
    p = tmp_path / "gc.fastq"
    _write_fastq(p, recs)

    ref = np.array([_naive_gc(s) for _, s, _ in recs])
    got = np.concatenate(
        [b.gc_content() for b in SmartImporter.stream_fastq_batches(str(p))])
    assert np.allclose(got, ref)


@pytest.mark.parametrize("fixed,k", [(True, 3), (True, 4), (False, 3), (False, 5)])
def test_kmer_spectrum_matches_naive(tmp_path, fixed, k):
    rng = random.Random(40 + k + int(fixed))
    recs = []
    for i in range(200):
        L = 130 if fixed else rng.randint(40, 160)
        recs.append((f"r{i}", _rand_seq(L, rng),
                     [rng.randint(0, 39) for _ in range(L)]))
    p = tmp_path / "km.fastq"
    _write_fastq(p, recs)

    ref = _naive_kmers([s for _, s, _ in recs], k)
    got = np.zeros(4 ** k, dtype=np.int64)
    for b in SmartImporter.stream_fastq_batches(str(p)):
        got += b.kmer_spectrum(k)
    assert np.array_equal(got, ref)


def test_kmer_spectrum_skips_ambiguous(tmp_path):
    # ACGNACGT → di-meros válidos: AC, CG, AC, CG, GT = 5 (los que tocan N se omiten)
    p = tmp_path / "amb.fastq"
    p.write_text("@r\nACGNACGT\n+\nIIIIIIII\n", encoding="ascii")
    spec = None
    for b in SmartImporter.stream_fastq_batches(str(p)):
        spec = b.kmer_spectrum(2)
    assert spec.sum() == 5
    assert spec[_BASE_ID["A"] * 4 + _BASE_ID["C"]] == 2   # AC ×2


def test_sequence_batch_gc_and_kmers(tmp_path):
    rng = random.Random(7)
    recs = [(f"g{i}", _rand_seq(rng.randint(30, 120), rng)) for i in range(200)]
    p = tmp_path / "g.fasta"
    _write_fasta(p, recs)
    ref_gc = np.array([_naive_gc(s) for _, s in recs])
    ref_km = _naive_kmers([s for _, s in recs], 3)
    gc, km = [], np.zeros(4 ** 3, dtype=np.int64)
    for b in SmartImporter.stream_batches(str(p)):
        gc.append(b.gc_content())
        km += b.kmer_spectrum(3)
    assert np.allclose(np.concatenate(gc), ref_gc)
    assert np.array_equal(km, ref_km)


def test_gc_on_protein_batch_raises(tmp_path):
    p = tmp_path / "prot.fasta"
    _write_fasta(p, [("p1", "MKLPQEFILV")])   # tiene residuos exclusivos de proteína
    batch = next(SmartImporter.stream_batches(str(p)))
    assert batch[0].seq_type == SeqType.PROTEIN
    with pytest.raises(SequenceTypeError):
        batch.gc_content()
    with pytest.raises(SequenceTypeError):
        batch.kmer_spectrum(3)


# ════════════════════════════════════════════════════════════════════════════
# Soporte .gz  (descompresión transparente en C)
# ════════════════════════════════════════════════════════════════════════════

def _write_fastq_gz(path, records):
    with gzip.open(path, "wt", newline="\n") as f:
        for header, seq, qual in records:
            qstr = "".join(QCHARS[q] for q in qual)
            f.write(f"@{header}\n{seq}\n+\n{qstr}\n")


def test_fastq_gz_equals_plain(tmp_path):
    rng = random.Random(50)
    recs = [(f"r{i}", _rand_seq(rng.randint(30, 200), rng),
             [rng.randint(0, 39) for _ in range(rng.randint(1, 3))])
            for i in range(200)]
    # regenerar calidades con longitud correcta
    recs = [(h, s, [rng.randint(0, 39) for _ in range(len(s))]) for h, s, _ in recs]

    plain = tmp_path / "a.fastq"
    gzf = tmp_path / "a.fastq.gz"
    _write_fastq(plain, recs)
    _write_fastq_gz(gzf, recs)

    def collect(path):
        return [(r.sequence.header, r.sequence.to_string(), list(r.quality))
                for r in SmartImporter.stream_fastq(str(path))]

    a = collect(plain)
    b = collect(gzf)
    assert a == b
    assert len(a) == len(recs)


def test_fasta_gz(tmp_path):
    rng = random.Random(51)
    recs = [(f"g{i}", _rand_seq(rng.randint(20, 100), rng)) for i in range(150)]
    gzf = tmp_path / "x.fasta.gz"
    with gzip.open(gzf, "wt", newline="\n") as f:
        for h, s in recs:
            f.write(f">{h}\n{s}\n")
    got = [(ps.header, ps.to_string()) for ps in SmartImporter.stream(str(gzf))]
    assert got == recs


def test_fastq_gz_columnar(tmp_path):
    rng = random.Random(52)
    recs = [(f"r{i}", _rand_seq(150, rng),
             [rng.randint(0, 39) for _ in range(150)]) for i in range(500)]
    gzf = tmp_path / "c.fastq.gz"
    _write_fastq_gz(gzf, recs)
    total = 0
    for b in SmartImporter.stream_fastq_batches(str(gzf)):
        total += len(b)
    assert total == len(recs)


# ════════════════════════════════════════════════════════════════════════════
# Registros vacíos — deben SALTARSE, nunca truncar el archivo
# (regresión: un registro vacío devolvía 0 en C = indistinguible de EOF)
# ════════════════════════════════════════════════════════════════════════════

def test_fasta_empty_record_in_middle_is_skipped(tmp_path):
    p = tmp_path / "m.fasta"
    p.write_text(">a\nACGT\n>vacio\n>c\nGGGG\n", encoding="ascii")
    got = [(s.header, s.to_string()) for s in SmartImporter.stream(str(p))]
    assert got == [("a", "ACGT"), ("c", "GGGG")]


def test_fasta_empty_first_record_does_not_truncate(tmp_path):
    p = tmp_path / "f.fasta"
    p.write_text(">vacio\n>a\nACGT\n>c\nGGGG\n", encoding="ascii")
    got = [s.header for s in SmartImporter.stream(str(p))]
    assert got == ["a", "c"]            # antes del fix devolvía []


def test_fastq_empty_record_in_middle_is_skipped(tmp_path):
    p = tmp_path / "m.fastq"
    p.write_text("@r1\nACGT\n+\nIIII\n@r2\n\n+\n\n@r3\nGGGG\n+\nIIII\n",
                 encoding="ascii")
    got = [r.sequence.header for r in SmartImporter.stream_fastq(str(p))]
    assert got == ["r1", "r3"]


def test_fastq_empty_first_record_does_not_truncate(tmp_path):
    p = tmp_path / "f.fastq"
    p.write_text("@vacio\n\n+\n\n@r2\nACGT\n+\nIIII\n@r3\nGGGG\n+\nIIII\n",
                 encoding="ascii")
    got = [r.sequence.header for r in SmartImporter.stream_fastq(str(p))]
    assert got == ["r2", "r3"]


def test_columnar_empty_first_record(tmp_path):
    p = tmp_path / "c.fastq"
    p.write_text("@vacio\n\n+\n\n@r2\nACGT\n+\nIIII\n", encoding="ascii")
    total = sum(len(b) for b in SmartImporter.stream_fastq_batches(str(p)))
    assert total == 1


def test_malformed_fastq_quality_length_no_crash(tmp_path):
    # Calidad más corta que la secuencia: no debe romper el reshape 2-D.
    p = tmp_path / "bad.fastq"
    p.write_text("@r1\nACGT\n+\nIIII\n@r2\nACGT\n+\nIII\n@r3\nACGT\n+\nIIII\n",
                 encoding="ascii")
    means = []
    for b in SmartImporter.stream_fastq_batches(str(p)):
        means.append(b.mean_quality())   # no debe lanzar ValueError
    assert np.concatenate(means).shape[0] == 3
