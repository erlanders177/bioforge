"""
fetch.py
══════════════════════════════════════════════════════════════════════
Pegamento a bases de datos — descarga de secuencias de NCBI (Entrez E-utilities).

Idea robada a Biopython (su subpaquete de bases de datos) pero **sin dependencias**:
solo la stdlib (urllib). Es el prerrequisito del predictor de evolución, que necesita
secuencias reales **con fecha**; y una utilidad general útil por sí misma.

Cortesía con NCBI: se identifica con `tool`/`email` y se espacian las peticiones
(NCBI permite ~3/s sin clave). Para uso intensivo, consigue una API key y pásala.

CACHÉ EN DISCO (``cache=True``, por defecto): una consulta ya descargada no se vuelve
a pedir. No es solo comodidad — es cortesía real con NCBI (evaluar un predictor implica
repetir la misma consulta decenas de veces) y hace REPRODUCIBLE la medición: los
mismos datos exactos en cada ejecución, aunque NCBI crezca. Borra ``~/.cache/bioforge``
para refrescar.

Todo error de red se envuelve en ``BioForgeIOError`` (jerarquía unificada del motor).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from typing import Iterable, Optional

from .biocore import BioForgeIOError

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
# Año de cepa en la cabecera de gripe. Gripe A: ".../2019(H3N2)"; gripe B: ".../2008)".
# El año va tras el último '/' y antes de '(' o ')'. Sirve para A y B.
_YEAR_RE = re.compile(r"/(\d{4})\s*[()]")
_DEFAULT_EMAIL = "bioforge@users.noreply.github.com"
_CHUNK = 100                        # IDs por lote en efetch (evita URLs largas, error 414)
_RETRIES = 4                        # NCBI corta conexiones a medias: hay que reintentar
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "bioforge", "fetch")


def _cache_load(key: str) -> Optional[list]:
    """Consulta cacheada → [(secuencia, tiempo)], o None si no está."""
    path = os.path.join(_CACHE_DIR, key + ".json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return [(s, t) for s, t in json.load(fh)]
    except (OSError, ValueError):
        return None                                  # sin caché o corrupta → se rebaja


def _cache_save(key: str, data: list) -> None:
    """Guarda la consulta. Si el disco falla, no es fatal: se sigue sin caché."""
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        tmp = os.path.join(_CACHE_DIR, key + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp, os.path.join(_CACHE_DIR, key + ".json"))   # atómico
    except OSError:
        pass


def _cache_key(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:20]


def _efetch(ids, rettype, db, email, api_key, timeout):
    """efetch POR LOTES → concatena el texto. Evita el error 414 (URL demasiado larga)
    cuando hay muchos IDs; espacia los lotes por cortesía con NCBI."""
    out = []
    for i in range(0, len(ids), _CHUNK):
        out.append(_get("efetch.fcgi",
                        {"db": db, "id": ",".join(ids[i:i + _CHUNK]),
                         "rettype": rettype, "retmode": "text"},
                        email=email, api_key=api_key, timeout=timeout))
        if i + _CHUNK < len(ids):
            time.sleep(0.34)
    return "".join(out)


def _get(endpoint: str, params: dict, *, email: str, api_key: Optional[str],
         timeout: float, retries: int = _RETRIES) -> str:
    """Una consulta a NCBI, con REINTENTOS y espera creciente.

    NCBI corta conexiones a medias (``IncompleteRead``), da 429/500 bajo carga y a
    veces simplemente tarda: son fallos TRANSITORIOS, no errores del usuario. Sin
    reintento, un hipo de red tira abajo una evaluación de media hora. Se espera
    1s, 2s, 4s… (backoff exponencial, buen ciudadano) y solo se rinde al final."""
    q = {**params, "tool": "bioforge", "email": email}
    if api_key:
        q["api_key"] = api_key
    url = _EUTILS + endpoint + "?" + urllib.parse.urlencode(q)
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:                            # red, HTTP, timeout…
            last = e
            if attempt < retries - 1:
                time.sleep(2.0 ** attempt)
    raise BioForgeIOError(
        f"fallo consultando NCBI ({endpoint}) tras {retries} intentos: {last}") from last


def esearch(term: str, *, db: str = "nuccore", retmax: int = 100,
            email: str = _DEFAULT_EMAIL, api_key: Optional[str] = None,
            timeout: float = 60.0) -> list[str]:
    """Busca en NCBI y devuelve la lista de IDs que casan con ``term``."""
    raw = _get("esearch.fcgi",
               {"db": db, "term": term, "retmax": retmax, "retmode": "json"},
               email=email, api_key=api_key, timeout=timeout)
    try:
        return json.loads(raw)["esearchresult"]["idlist"]
    except (KeyError, json.JSONDecodeError) as e:
        raise BioForgeIOError(f"respuesta esearch inesperada de NCBI: {e}") from e


def efetch_fasta(ids: Iterable[str], *, db: str = "nuccore",
                 email: str = _DEFAULT_EMAIL, api_key: Optional[str] = None,
                 timeout: float = 120.0) -> list[tuple[str, str]]:
    """Descarga las secuencias de ``ids`` en FASTA. Devuelve [(cabecera, secuencia)]."""
    ids = list(ids)
    if not ids:
        return []
    return _parse_fasta(_efetch(ids, "fasta", db, email, api_key, timeout))


def _parse_fasta(text: str) -> list[tuple[str, str]]:
    recs: list[tuple[str, str]] = []
    hdr: Optional[str] = None
    seq: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if hdr is not None:
                recs.append((hdr, "".join(seq)))
            hdr, seq = line[1:], []
        elif hdr is not None:
            seq.append(line.strip())
    if hdr is not None:
        recs.append((hdr, "".join(seq)))
    return recs


_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def _decimal_year(s: str) -> Optional[float]:
    """Fecha de colecta → año decimal (resolución de MES). Maneja los formatos de
    GenBank: '2019-03-17', '2019-03', '2019', '17-Mar-2019', 'Mar-2019'."""
    s = s.strip()
    m = re.match(r"(\d{4})(?:-(\d{1,2}))?(?:-\d{1,2})?$", s)     # ISO
    if m:
        return int(m.group(1)) + (int(m.group(2) or 1) - 1) / 12.0
    m = re.match(r"(?:\d{1,2}-)?([A-Za-z]{3})-(\d{4})$", s)      # DD-Mon-YYYY
    if m:
        mo = _MONTHS.get(m.group(1).lower())
        if mo:
            return int(m.group(2)) + (mo - 1) / 12.0
    return None


def _parse_gb(text: str) -> list[tuple[str, str]]:
    """GenBank flat → [(secuencia, fecha_colecta)]. Extrae la secuencia de ORIGIN."""
    out: list[tuple[str, str]] = []
    for rec in text.split("//\n"):
        if "ORIGIN" not in rec:
            continue
        m = re.search(r'/collection_date="([^"]+)"', rec)
        if not m:
            continue
        seq = re.sub(r"[^A-Za-z]", "", rec.split("ORIGIN", 1)[1])
        if seq:
            out.append((seq.upper(), m.group(1)))
    return out


def fetch_dated_precise(term_template: str, years: Iterable[int], *, per_year: int = 100,
                        quarter: bool = True, db: str = "nuccore",
                        email: str = _DEFAULT_EMAIL, api_key: Optional[str] = None,
                        pause: float = 0.4, progress: bool = False,
                        cache: bool = True) -> list[tuple[str, float]]:
    """Como ``fetch_dated`` pero con **fecha de colecta real** (resolución de mes) del
    registro GenBank → permite bins finos (la palanca B). ``quarter=True`` redondea a
    trimestre (año.00/.25/.50/.75). Devuelve ``[(secuencia, año_decimal)]``.

    Con ``cache=True`` cada AÑO se cachea por separado: ampliar el rango de años solo
    descarga los nuevos, y repetir una evaluación no vuelve a tocar NCBI."""
    out: list[tuple[str, float]] = []
    for y in years:
        key = _cache_key("precise", term_template, y, per_year, quarter, db)
        hit = _cache_load(key) if cache else None
        if hit is not None:
            out.extend((s, float(t)) for s, t in hit)
            if progress:
                print(f"  {y}: {len(hit)} secuencias (caché)", flush=True)
            continue
        ids = esearch(term_template.format(year=y), db=db, retmax=per_year,
                      email=email, api_key=api_key)
        if not ids:
            continue
        raw = _efetch(ids, "gb", db, email, api_key, 180.0)
        year_out: list[tuple[str, float]] = []
        for seq, date in _parse_gb(raw):
            dy = _decimal_year(date)
            if dy is None:
                continue
            if quarter:
                dy = int(dy) + (int((dy % 1) * 4)) / 4.0        # → trimestre
            year_out.append((seq, dy))
        if cache:
            _cache_save(key, year_out)
        out.extend(year_out)
        if progress:
            print(f"  {y}: {len(year_out)} secuencias con fecha", flush=True)
        time.sleep(pause)
    return out


def fetch_dated(term_template: str, years: Iterable[int], *, per_year: int = 30,
                db: str = "nuccore", email: str = _DEFAULT_EMAIL,
                api_key: Optional[str] = None, pause: float = 0.4,
                progress: bool = False) -> list[tuple[str, int]]:
    """Descarga secuencias **fechadas** para el predictor de evolución.

    Para cada año de ``years`` lanza una búsqueda con ``term_template`` (que debe
    contener ``{year}``), descarga las secuencias, y extrae el año de la cabecera
    (nombre de cepa de gripe). Solo conserva las cuyo año parseado coincide con el
    consultado — así el binning temporal es limpio. Devuelve ``[(secuencia, año)]``.

    Ejemplo de ``term_template`` (gripe H3N2, HA completa):
        "Influenza A virus[Organism] AND H3N2 AND hemagglutinin[Title] "
        "AND 1650:1780[SLEN] AND {year}"
    """
    out: list[tuple[str, int]] = []
    for y in years:
        ids = esearch(term_template.format(year=y), db=db, retmax=per_year,
                      email=email, api_key=api_key)
        recs = efetch_fasta(ids, db=db, email=email, api_key=api_key)
        kept = 0
        for hdr, seq in recs:
            m = _YEAR_RE.search(hdr)
            if m and int(m.group(1)) == y and seq:
                out.append((seq.upper(), y))
                kept += 1
        if progress:
            print(f"  {y}: {kept}/{len(recs)} secuencias con año válido", flush=True)
        time.sleep(pause)                                # cortesía con NCBI
    return out
