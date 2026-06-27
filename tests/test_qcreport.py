"""
tests/test_qcreport.py
Pruebas del informe de calidad FASTQ (bioforge/qcreport.py).

Verifican las métricas contra valores calculados a mano y la coherencia
entre FASTQ plano, .gz y longitud irregular.
"""

import gzip
import random

import numpy as np
import pytest

from bioforge import qcreport

QCHARS = "".join(chr(33 + q) for q in range(64))   # Phred 0..63
BASES = "ACGT"


def _write(path, records, gz=False):
    opener = (lambda p: gzip.open(p, "wt", newline="\n")) if gz else \
             (lambda p: open(p, "w", newline="\n"))
    with opener(path) as f:
        for h, s, q in records:
            f.write(f"@{h}\n{s}\n+\n{''.join(QCHARS[x] for x in q)}\n")


def test_basic_metrics(tmp_path):
    # 3 lecturas conocidas, longitud fija 4
    recs = [
        ("r1", "ACGT", [40, 40, 40, 40]),   # GC 50%, Q40
        ("r2", "GGCC", [20, 20, 20, 20]),   # GC 100%, Q20
        ("r3", "ATAT", [30, 30, 30, 30]),   # GC 0%,  Q30
    ]
    p = tmp_path / "a.fastq"
    _write(p, recs)
    r = qcreport.run(str(p))

    assert r.n_reads == 3
    assert r.total_bases == 12
    assert r.min_len == 4 and r.max_len == 4
    # GC global = (2 + 4 + 0) / 12
    assert r.gc_overall == pytest.approx(6 / 12)
    # Q global = (40*4 + 20*4 + 30*4) / 12 = 30
    assert r.mean_q_overall == pytest.approx(30.0)
    assert r.pct_q20 == pytest.approx(100.0)     # todas ≥ 20
    assert r.pct_q30 == pytest.approx(2 / 3 * 100)  # r1 y r3


def test_per_position_quality(tmp_path):
    # Calidad que decae: posición 0 alta, última baja
    recs = [("r%d" % i, "ACGT", [40, 30, 20, 10]) for i in range(10)]
    p = tmp_path / "pq.fastq"
    _write(p, recs)
    r = qcreport.run(str(p))
    assert r.pos_q_mean.shape[0] == 4
    assert np.allclose(r.pos_q_mean, [40, 30, 20, 10])


def test_base_composition(tmp_path):
    recs = [("r%d" % i, "AACC", [30, 30, 30, 30]) for i in range(5)]
    p = tmp_path / "bc.fastq"
    _write(p, recs)
    r = qcreport.run(str(p))
    # posición 0 y 1 = A (100%), 2 y 3 = C (100%)
    assert r.base_frac[0, 0] == pytest.approx(1.0)   # A en pos 0
    assert r.base_frac[1, 2] == pytest.approx(1.0)   # C en pos 2


def test_gz_equals_plain(tmp_path):
    rng = random.Random(3)
    recs = [("r%d" % i, "".join(rng.choice(BASES) for _ in range(50)),
             [rng.randint(0, 39) for _ in range(50)]) for i in range(200)]
    plain = tmp_path / "x.fastq"
    gzf = tmp_path / "x.fastq.gz"
    _write(plain, recs)
    _write(gzf, recs, gz=True)
    a = qcreport.run(str(plain))
    b = qcreport.run(str(gzf))
    assert a.n_reads == b.n_reads
    assert a.total_bases == b.total_bases
    assert a.gc_overall == pytest.approx(b.gc_overall)
    assert a.mean_q_overall == pytest.approx(b.mean_q_overall)
    assert np.allclose(a.pos_q_mean, b.pos_q_mean)


def test_ragged_lengths(tmp_path):
    rng = random.Random(4)
    recs = [("r%d" % i, "".join(rng.choice(BASES) for _ in range(rng.randint(20, 80))),
             None) for i in range(150)]
    recs = [(h, s, [rng.randint(0, 39) for _ in range(len(s))]) for h, s, _ in recs]
    p = tmp_path / "rag.fastq"
    _write(p, recs)
    r = qcreport.run(str(p))
    assert r.n_reads == 150
    assert r.min_len != r.max_len
    # media de calidad global coherente con referencia
    ref = sum(sum(q) for _, _, q in recs) / sum(len(s) for _, s, _ in recs)
    assert r.mean_q_overall == pytest.approx(ref)


def test_report_text_renders(tmp_path):
    recs = [("r%d" % i, "ACGTACGT", [30] * 8) for i in range(20)]
    p = tmp_path / "t.fastq"
    _write(p, recs)
    r = qcreport.run(str(p))
    text = qcreport.build_report(r)
    assert "INFORME DE CALIDAD FASTQ" in text
    assert "Lecturas" in text
    assert "20" in text


def test_cli(tmp_path, capsys):
    recs = [("r%d" % i, "ACGT", [30, 30, 30, 30]) for i in range(10)]
    p = tmp_path / "cli.fastq"
    _write(p, recs)
    rc = qcreport.main([str(p)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "INFORME DE CALIDAD" in out


def test_cli_output_file(tmp_path):
    recs = [("r%d" % i, "ACGT", [30, 30, 30, 30]) for i in range(10)]
    p = tmp_path / "cli.fastq"
    out = tmp_path / "rep.txt"
    _write(p, recs)
    rc = qcreport.main([str(p), "-o", str(out)])
    assert rc == 0
    assert out.exists() and "INFORME DE CALIDAD" in out.read_text(encoding="utf-8")


def test_empty_file_errors(tmp_path):
    p = tmp_path / "empty.fastq"
    p.write_text("", encoding="ascii")
    with pytest.raises(ValueError):
        qcreport.run(str(p))
