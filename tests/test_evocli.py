"""
tests/test_evocli.py — CLI del predictor de evolución (bioforge-evolution).

Comprueba que los tres subcomandos leen un FASTA fechado y devuelven 0, que la fecha
se extrae de la cabecera, y que los errores de uso se reportan con gracia.
"""

import numpy as np
import pytest

from bioforge.evocli import _read_dated_fasta, _year_from_header, main


def _write_dated_fasta(path, protein=True, n_year=12, years=range(2015, 2021)):
    """Barrido sintético: en el sitio 10, K→R sube con el tiempo. Fecha en cabecera."""
    base = list("MKTIIALSYIFCLVFADRICIGYHANNSTEQVDTIMEKNVTV") if protein \
        else list("ATGAAAACCATTATTGCTTTGAGCTACATTTTCTGTCTGGTTTTCGCT")
    rng = np.random.default_rng(0)
    lines = []
    for yr in years:
        frac = (yr - min(years)) / max(len(years) - 1, 1)
        for j in range(n_year):
            s = base.copy()
            s[10] = "R" if (protein and j < frac * n_year) else s[10]
            lines.append(f">strain_{j}/{yr}|{yr}")
            lines.append("".join(s))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_year_from_header_varios_formatos():
    assert _year_from_header("A/Sydney/5/2021|2021-03") == pytest.approx(2021 + 2 / 12)
    assert _year_from_header("strain_2019") == 2019.0
    assert _year_from_header(">algo/2008)") == 2008.0
    assert _year_from_header("sin fecha aqui") is None


def test_lectura_descarta_sin_fecha(tmp_path):
    p = tmp_path / "x.fasta"
    p.write_text(">con_fecha|2020\nMKTII\n>sin_fecha\nMKTII\n>otra|2021\nMKTIR\n",
                 encoding="utf-8")
    seqs, times = _read_dated_fasta(str(p))
    assert len(seqs) == 2 and times == [2020.0, 2021.0]


def test_lectura_falla_si_menos_de_dos(tmp_path):
    p = tmp_path / "x.fasta"
    p.write_text(">solo_una|2020\nMKTII\n", encoding="utf-8")
    with pytest.raises(Exception):
        _read_dated_fasta(str(p))


def test_cli_rank_devuelve_cero(tmp_path, capsys):
    p = _write_dated_fasta(tmp_path / "s.fasta")
    assert main(["rank", str(p), "--top", "5"]) == 0
    out = capsys.readouterr().out
    assert "MUTACIONES ordenadas" in out
    assert "Honesto:" in out                    # el descargo honesto siempre sale


def test_cli_backtest_devuelve_cero(tmp_path, capsys):
    p = _write_dated_fasta(tmp_path / "s.fasta")
    assert main(["backtest", str(p)]) == 0
    assert "SKILL" in capsys.readouterr().out


def test_cli_lineages_devuelve_cero(tmp_path, capsys):
    p = _write_dated_fasta(tmp_path / "s.fasta")
    assert main(["lineages", str(p), "--min-size", "5"]) == 0
    assert "LINAJES ESTABLES" in capsys.readouterr().out


def test_cli_archivo_inexistente(capsys):
    assert main(["rank", "no_existe_12345.fasta"]) == 1
    assert "no encontrado" in capsys.readouterr().err.lower()


def test_cli_rank_translate_desde_nucleotido(tmp_path, capsys):
    # entrada nucleótido + --translate → no debe fallar
    p = _write_dated_fasta(tmp_path / "n.fasta", protein=False)
    assert main(["rank", str(p), "--translate", "--top", "3"]) == 0
