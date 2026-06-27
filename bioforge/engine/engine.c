/*
 * engine/engine.c — Motor C de alto rendimiento
 *
 * Incluye:
 *   - Empaquetado/desempaquetado 5-bit (compatible con NumPy packbits)
 *   - Alineamiento Needleman-Wunsch (global + semi-global)
 *
 * Compilar (Windows/MinGW):
 *   gcc -O3 -march=native -fopenmp -shared -o engine.dll engine.c
 * Compilar (Linux/Mac):
 *   gcc -O3 -march=native -fopenmp -shared -fPIC -o engine.so engine.c
 */

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT __attribute__((visibility("default")))
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ── Lectura de archivos: gzip transparente si se compila con -DBIO_USE_ZLIB ──
   zlib gzopen/gzread leen tanto archivos planos como .gz (autodetección del
   magic gzip), así que un único código sirve para ambos formatos. Si zlib no
   está disponible, se compila sin -DBIO_USE_ZLIB y se usa stdio normal. */
#ifdef BIO_USE_ZLIB
  #include <zlib.h>
  typedef gzFile BIO_FILE;
  #define BIO_OPEN(path)     gzopen((path), "rb")
  #define BIO_READ(f, b, n)  gzread((f), (b), (unsigned)(n))
  #define BIO_CLOSE(f)       gzclose(f)
#else
  typedef FILE* BIO_FILE;
  #define BIO_OPEN(path)     fopen((path), "rb")
  #define BIO_READ(f, b, n)  ((int)fread((b), 1, (size_t)(n), (f)))
  #define BIO_CLOSE(f)       fclose(f)
#endif

/* ═══════════════════════════════════════════════════════════════════════════
   5-BIT PACK / UNPACK / GETITEM
   Formato compatible con np.packbits(bitorder='big') +
   np.unpackbits(bitorder='little', count=5)
   ═══════════════════════════════════════════════════════════════════════════ */

/*
 * Formato compatible con np.packbits(bits, bitorder='big') donde bits se
 * extraen MSB-first: code=5 (0b00101) → bits [0,0,1,0,1] en el stream.
 * Cada código ocupa 5 bits consecutivos en orden MSB→LSB.
 */

/* Extrae el código en la posición i. O(1) — lee 1-2 bytes. */
EXPORT uint8_t bio_getitem5(const uint8_t* packed, int32_t i) {
    uint32_t bit_start = (uint32_t)i * 5;
    uint32_t byte0     = bit_start >> 3;   /* / 8 */
    uint32_t shift0    = bit_start & 7;    /* % 8 */

    /* Cargar hasta 2 bytes en un word de 16 bits */
    uint16_t word = (uint16_t)packed[byte0] << 8;
    if (shift0 > 3) word |= packed[byte0 + 1];

    /* Los 5 bits del código quedan en las posiciones [15-shift0 .. 11-shift0] */
    return (uint8_t)((word >> (11u - shift0)) & 0x1Fu);
}

/* Desempaqueta n códigos hacia out[n]. */
EXPORT void bio_unpack5(const uint8_t* packed, int32_t n, uint8_t* out) {
    for (int32_t i = 0; i < n; i++)
        out[i] = bio_getitem5(packed, i);
}

/* Empaqueta n códigos de 5 bits en out (tamaño >= ceil(n*5/8)+1). */
EXPORT void bio_pack5(const uint8_t* codes, int32_t n, uint8_t* out) {
    int32_t out_len = (n * 5 + 7) / 8;
    memset(out, 0, (size_t)out_len + 1);

    for (int32_t i = 0; i < n; i++) {
        uint32_t bit_start = (uint32_t)i * 5;
        uint32_t byte0     = bit_start >> 3;
        uint32_t shift0    = bit_start & 7;

        /* Posicionar los 5 bits MSB-first en el word de 16 bits */
        uint16_t w = (uint16_t)(codes[i] & 0x1Fu) << (11u - shift0);
        out[byte0] |= (uint8_t)(w >> 8);
        if (shift0 > 3) out[byte0 + 1] |= (uint8_t)(w & 0xFFu);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
   TRADUCCION GENETICA
   ═══════════════════════════════════════════════════════════════════════════ */

/*
 * Localiza el primer codon ATG (A=0, T/U=3, G=2) en el array de codigos.
 * Devuelve la posicion base-0 del primer nucleotido, o -1 si no se encuentra.
 */
EXPORT int32_t bio_find_atg(const uint8_t* codes, int32_t n) {
    int32_t limit = n - 2;
    for (int32_t i = 0; i < limit; i++) {
        if (codes[i] == 0u && codes[i + 1] == 3u && codes[i + 2] == 2u)
            return i;
    }
    return -1;
}

/*
 * Traduce n_codons codones usando el LUT de 64 entradas del codigo genetico.
 *
 *   codon_lut  : 64 bytes, indice = n1*16 + n2*4 + n3 (A=0 C=1 G=2 T/U=3)
 *   nuc_codes  : array plano de n_codons*3 valores uint8 [0-3]
 *   out        : array de salida, n_codons bytes (BioCode de AA o UNK=31)
 *
 * Codones con indices fuera de [0,63] (bases ambiguas > 3) se mapean a UNK.
 * El bucle interno es trivialmente auto-vectorizable por GCC -O3.
 */
EXPORT void bio_translate(
    const uint8_t* codon_lut,
    const uint8_t* nuc_codes,
    int32_t        n_codons,
    uint8_t*       out
) {
    for (int32_t i = 0; i < n_codons; i++) {
        int32_t idx = (int32_t)nuc_codes[i * 3]      * 16
                    + (int32_t)nuc_codes[i * 3 + 1]  *  4
                    + (int32_t)nuc_codes[i * 3 + 2];
        out[i] = (idx < 64) ? codon_lut[idx] : (uint8_t)31u;  /* 31 = UNK */
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
   NEEDLEMAN-WUNSCH
   ═══════════════════════════════════════════════════════════════════════════ */

#define TB_DIAG ((uint8_t)0)
#define TB_UP   ((uint8_t)1)
#define TB_LEFT ((uint8_t)2)

/*
 * Núcleo interno: fill de la matriz NW + traceback.
 * semiglobal=0 → global; semiglobal=1 → semi-global (extremos de seq_b libres).
 * Devuelve longitud del alineamiento, -1 en error de memoria.
 */
static int32_t _nw_core(
    const uint8_t* codes_a, int32_t m,
    const uint8_t* codes_b, int32_t n,
    const char*    decode,          /* LUT 32 bytes: BioCode -> carácter ASCII */
    int32_t match_sc, int32_t mismatch_sc, int32_t gap_sc,
    int32_t semiglobal,
    char* out_a, char* out_b,       /* buffers de salida, tamaño >= m+n+1 */
    int32_t* p_score,
    int32_t* p_matches, int32_t* p_mismatches, int32_t* p_gaps
) {
    const int64_t cols  = (int64_t)n + 1;
    const int64_t total = ((int64_t)m + 1) * cols;

    int32_t* H  = (int32_t*)malloc((size_t)total * sizeof(int32_t));
    uint8_t* tb = (uint8_t*)malloc((size_t)total);
    if (!H || !tb) { free(H); free(tb); return -1; }

    /* Inicializar primera fila y columna */
    H[0] = 0; tb[0] = TB_DIAG;
    for (int32_t i = 1; i <= m; i++) {
        H[(int64_t)i * cols] = i * gap_sc;
        tb[(int64_t)i * cols] = TB_UP;
    }
    for (int32_t j = 1; j <= n; j++) {
        H[j] = semiglobal ? 0 : j * gap_sc;   /* semi-global: primera fila a 0 */
        tb[j] = TB_LEFT;
    }

    /* Fill de la matriz — bucle interno auto-vectorizable con -O3 */
    for (int32_t i = 1; i <= m; i++) {
        const uint8_t ca = codes_a[i - 1];
        const int32_t* pr = H + (int64_t)(i - 1) * cols;   /* fila anterior */
              int32_t* cr = H + (int64_t) i        * cols;  /* fila actual   */
              uint8_t* tr = tb + (int64_t)i        * cols;

        for (int32_t j = 1; j <= n; j++) {
            int32_t s    = (ca == codes_b[j - 1]) ? match_sc : mismatch_sc;
            int32_t diag = pr[j - 1] + s;
            int32_t up   = pr[j]     + gap_sc;
            int32_t left = cr[j - 1] + gap_sc;

            uint8_t dir; int32_t best;
            if      (diag >= up && diag >= left) { best = diag; dir = TB_DIAG; }
            else if (up   >= left)               { best = up;   dir = TB_UP;   }
            else                                 { best = left;  dir = TB_LEFT; }

            cr[j] = best; tr[j] = dir;
        }
    }

    /* Determinar punto de inicio del traceback */
    int32_t end_i = m, end_j = n;
    if (semiglobal) {
        /* Buscar el mejor score en la ultima fila Y en la ultima columna.
         * semi-global: gaps terminales del query (seq_b) son gratuitos.
         * La ultima fila (i=m) contiene el final del query alineado contra
         * distintas posiciones de la referencia (seq_a).
         * La ultima columna (j=n) contiene el final de la referencia. */
        int32_t best_sc = INT32_MIN;

        /* Escanear ultima fila (i=m, j de 0 a n) */
        for (int32_t j = 0; j <= n; j++) {
            int32_t sc = H[(int64_t)m * cols + j];
            if (sc > best_sc) { best_sc = sc; end_i = m; end_j = j; }
        }
        /* Escanear ultima columna (j=n, i de 0 a m) */
        for (int32_t i = 0; i <= m; i++) {
            int32_t sc = H[(int64_t)i * cols + n];
            if (sc > best_sc) { best_sc = sc; end_i = i; end_j = n; }
        }
        *p_score = best_sc;
    } else {
        end_j    = n;
        *p_score = H[(int64_t)m * cols + n];
    }

    /* Buffers temporales para traceback (se invierten al final) */
    char* ta  = (char*)malloc((size_t)(m + n + 1));
    char* tb2 = (char*)malloc((size_t)(m + n + 1));
    if (!ta || !tb2) { free(H); free(tb); free(ta); free(tb2); return -1; }

    int32_t pos = 0, nm = 0, nmi = 0, ng = 0;
    int32_t i = end_i, j = end_j;

    /* Semi-global: gaps terminales gratuitos en seq_a (end_i<m) o seq_b (end_j<n) */
    if (semiglobal) {
        for (int32_t k = m; k > end_i; k--) {
            ta[pos]  = decode[(uint8_t)codes_a[k - 1]];
            tb2[pos] = '-';
            ng++; pos++;
        }
        for (int32_t k = n; k > end_j; k--) {
            ta[pos]  = '-';
            tb2[pos] = decode[(uint8_t)codes_b[k - 1]];
            ng++; pos++;
        }
    }

    /* Traceback */
    while (i > 0 || j > 0) {
        uint8_t dir = tb[(int64_t)i * cols + j];
        if (dir == TB_DIAG) {
            uint8_t ca = codes_a[i - 1], cb = codes_b[j - 1];
            ta[pos]  = decode[ca];
            tb2[pos] = decode[cb];
            if (ca == cb) nm++; else nmi++;
            i--; j--;
        } else if (dir == TB_UP) {
            ta[pos]  = decode[(uint8_t)codes_a[i - 1]];
            tb2[pos] = '-';
            ng++; i--;
        } else {
            ta[pos]  = '-';
            tb2[pos] = decode[(uint8_t)codes_b[j - 1]];
            ng++; j--;
        }
        pos++;
    }

    /* Invertir en los buffers de salida */
    for (int32_t k = 0; k < pos; k++) {
        out_a[k] = ta[pos - 1 - k];
        out_b[k] = tb2[pos - 1 - k];
    }
    out_a[pos] = '\0';
    out_b[pos] = '\0';

    *p_matches    = nm;
    *p_mismatches = nmi;
    *p_gaps       = ng;

    free(H); free(tb); free(ta); free(tb2);
    return pos;
}

EXPORT int32_t nw_global(
    const uint8_t* ca, int32_t m,
    const uint8_t* cb, int32_t n,
    const char* decode,
    int32_t match, int32_t mismatch, int32_t gap,
    char* out_a, char* out_b,
    int32_t* score, int32_t* matches, int32_t* mismatches, int32_t* gaps
) {
    return _nw_core(ca, m, cb, n, decode, match, mismatch, gap, 0,
                    out_a, out_b, score, matches, mismatches, gaps);
}

EXPORT int32_t nw_semiglobal(
    const uint8_t* ca, int32_t m,
    const uint8_t* cb, int32_t n,
    const char* decode,
    int32_t match, int32_t mismatch, int32_t gap,
    char* out_a, char* out_b,
    int32_t* score, int32_t* matches, int32_t* mismatches, int32_t* gaps
) {
    return _nw_core(ca, m, cb, n, decode, match, mismatch, gap, 1,
                    out_a, out_b, score, matches, mismatches, gaps);
}

/* ═══════════════════════════════════════════════════════════════════════════
   SMITH-WATERMAN  (alineamiento local)
   H[i,j] = max(0, diag+s, up+gap, left+gap)
   Traceback desde el maximo de H hasta H[i][j]==0.
   ═══════════════════════════════════════════════════════════════════════════ */

static int32_t _sw_core(
    const uint8_t* codes_a, int32_t m,
    const uint8_t* codes_b, int32_t n,
    const char*    decode,
    int32_t match_sc, int32_t mismatch_sc, int32_t gap_sc,
    char* out_a, char* out_b,
    int32_t* p_score,
    int32_t* p_matches, int32_t* p_mismatches, int32_t* p_gaps
) {
    const int64_t cols  = (int64_t)n + 1;
    const int64_t total = ((int64_t)m + 1) * cols;

    /* calloc: todas las celdas a 0 (condicion de borde SW) */
    int32_t* H  = (int32_t*)calloc((size_t)total, sizeof(int32_t));
    uint8_t* tb = (uint8_t*)calloc((size_t)total, 1);
    if (!H || !tb) { free(H); free(tb); return -1; }

    int32_t max_score = 0, max_i = 0, max_j = 0;

    for (int32_t i = 1; i <= m; i++) {
        const uint8_t  ca = codes_a[i - 1];
        const int32_t* pr = H  + (int64_t)(i - 1) * cols;
              int32_t* cr = H  + (int64_t) i       * cols;
              uint8_t* tr = tb + (int64_t) i       * cols;

        for (int32_t j = 1; j <= n; j++) {
            int32_t s    = (ca == codes_b[j - 1]) ? match_sc : mismatch_sc;
            int32_t diag = pr[j - 1] + s;
            int32_t up   = pr[j]     + gap_sc;
            int32_t left = cr[j - 1] + gap_sc;

            /* SW: piso en 0 */
            int32_t best = 0; uint8_t dir = TB_DIAG;
            if (diag > best) { best = diag; dir = TB_DIAG; }
            if (up   > best) { best = up;   dir = TB_UP;   }
            if (left > best) { best = left; dir = TB_LEFT; }

            cr[j] = best; tr[j] = dir;
            if (best > max_score) { max_score = best; max_i = i; max_j = j; }
        }
    }
    *p_score = max_score;

    char* ta  = (char*)malloc((size_t)(m + n + 1));
    char* tb2 = (char*)malloc((size_t)(m + n + 1));
    if (!ta || !tb2) { free(H); free(tb); free(ta); free(tb2); return -1; }

    int32_t pos = 0, nm = 0, nmi = 0, ng = 0;
    int32_t i = max_i, j = max_j;

    /* Traceback: parar cuando H[i][j] == 0 */
    while (i > 0 && j > 0 && H[(int64_t)i * cols + j] > 0) {
        uint8_t dir = tb[(int64_t)i * cols + j];
        if (dir == TB_DIAG) {
            uint8_t ca = codes_a[i - 1], cb = codes_b[j - 1];
            ta[pos]  = decode[ca]; tb2[pos] = decode[cb];
            if (ca == cb) nm++; else nmi++;
            i--; j--;
        } else if (dir == TB_UP) {
            ta[pos] = decode[codes_a[i - 1]]; tb2[pos] = '-';
            ng++; i--;
        } else {
            ta[pos] = '-'; tb2[pos] = decode[codes_b[j - 1]];
            ng++; j--;
        }
        pos++;
    }

    for (int32_t k = 0; k < pos; k++) {
        out_a[k] = ta[pos - 1 - k];
        out_b[k] = tb2[pos - 1 - k];
    }
    out_a[pos] = '\0'; out_b[pos] = '\0';

    *p_matches = nm; *p_mismatches = nmi; *p_gaps = ng;
    free(H); free(tb); free(ta); free(tb2);
    return pos;
}

EXPORT int32_t sw_align(
    const uint8_t* ca, int32_t m,
    const uint8_t* cb, int32_t n,
    const char* decode,
    int32_t match, int32_t mismatch, int32_t gap,
    char* out_a, char* out_b,
    int32_t* score, int32_t* matches, int32_t* mismatches, int32_t* gaps
) {
    return _sw_core(ca, m, cb, n, decode, match, mismatch, gap,
                    out_a, out_b, score, matches, mismatches, gaps);
}

/* ═══════════════════════════════════════════════════════════════════════════
   NEEDLEMAN-WUNSCH CON BANDA  (banded NW)
   Solo se calculan celdas con |i-j| <= band.
   Almacenamiento en banda: H_band[i][k], k = j - i + band, j = k + i - band.
   Memoria: O(m * band) frente a O(m*n) del NW completo.
   ═══════════════════════════════════════════════════════════════════════════ */

#define _BH(i,k)  H_band[(int64_t)(i) * W + (k)]
#define _BTB(i,k) tb_band[(int64_t)(i) * W + (k)]

static int32_t _nw_banded_core(
    const uint8_t* codes_a, int32_t m,
    const uint8_t* codes_b, int32_t n,
    const char*    decode,
    int32_t match_sc, int32_t mismatch_sc, int32_t gap_sc,
    int32_t band, int32_t semiglobal,
    char* out_a, char* out_b,
    int32_t* p_score,
    int32_t* p_matches, int32_t* p_mismatches, int32_t* p_gaps
) {
    const int32_t W     = 2 * band + 1;
    const int64_t total = ((int64_t)m + 1) * W;
    const int32_t NEG   = INT32_MIN / 2;

    int32_t* H_band  = (int32_t*)malloc((size_t)total * sizeof(int32_t));
    uint8_t* tb_band = (uint8_t*)malloc((size_t)total);
    if (!H_band || !tb_band) { free(H_band); free(tb_band); return -1; }

    for (int64_t x = 0; x < total; x++) H_band[x] = NEG;
    memset(tb_band, TB_DIAG, (size_t)total);

    /* Fila 0: j en [0, min(n,band)], k = j + band */
    _BH(0, band) = 0;
    for (int32_t j = 1; j <= (band < n ? band : n); j++) {
        int32_t k = j + band;
        if (k >= W) break;
        _BH(0, k)  = semiglobal ? 0 : j * gap_sc;
        _BTB(0, k) = TB_LEFT;
    }
    /* Columna 0: i en [1, min(m,band)], k = band - i */
    for (int32_t i = 1; i <= (band < m ? band : m); i++) {
        int32_t k = band - i;
        if (k < 0) break;
        _BH(i, k)  = i * gap_sc;
        _BTB(i, k) = TB_UP;
    }

    /* Fill */
    for (int32_t i = 1; i <= m; i++) {
        int32_t j_lo = i - band; if (j_lo < 1) j_lo = 1;
        int32_t j_hi = i + band; if (j_hi > n) j_hi = n;

        for (int32_t j = j_lo; j <= j_hi; j++) {
            int32_t k = j - i + band;
            int32_t s = (codes_a[i-1] == codes_b[j-1]) ? match_sc : mismatch_sc;

            int32_t dv = _BH(i-1, k);
            int32_t uv = (k+1 < W) ? _BH(i-1, k+1) : NEG;
            int32_t lv = (k-1 >= 0) ? _BH(i,   k-1) : NEG;

            int32_t d  = (dv != NEG) ? dv + s       : NEG;
            int32_t u  = (uv != NEG) ? uv + gap_sc  : NEG;
            int32_t l  = (lv != NEG) ? lv + gap_sc  : NEG;

            uint8_t dir; int32_t best;
            if      (d >= u && d >= l && d != NEG) { best = d; dir = TB_DIAG; }
            else if (u >= l && u != NEG)           { best = u; dir = TB_UP;   }
            else if (l != NEG)                     { best = l; dir = TB_LEFT; }
            else                                   { best = NEG; dir = TB_DIAG; }

            _BH(i, k) = best; _BTB(i, k) = dir;
        }
    }

    /* Score y punto de inicio del traceback */
    int32_t end_i = m, end_j = n;
    if (semiglobal) {
        int32_t best_sc = NEG;
        int32_t i_lo = (n - band > 0) ? n - band : 0;
        int32_t i_hi = (n + band < m) ? n + band : m;
        for (int32_t i2 = i_lo; i2 <= i_hi; i2++) {
            int32_t k2 = n - i2 + band;
            if (k2 < 0 || k2 >= W) continue;
            int32_t v = _BH(i2, k2);
            if (v > best_sc) { best_sc = v; end_i = i2; }
        }
        end_j = n;
        *p_score = best_sc;
    } else {
        int32_t k_end = n - m + band;
        *p_score = (k_end >= 0 && k_end < W) ? _BH(m, k_end) : NEG;
    }

    char* ta  = (char*)malloc((size_t)(m + n + 1));
    char* tb2 = (char*)malloc((size_t)(m + n + 1));
    if (!ta || !tb2) { free(H_band); free(tb_band); free(ta); free(tb2); return -1; }

    int32_t pos = 0, nm = 0, nmi = 0, ng = 0;
    int32_t i = end_i, j = end_j;

    if (semiglobal) {
        for (int32_t ki = m; ki > end_i; ki--) {
            ta[pos] = decode[codes_a[ki-1]]; tb2[pos] = '-';
            ng++; pos++;
        }
    }

    while (i > 0 || j > 0) {
        int32_t k = j - i + band;
        uint8_t dir = (k >= 0 && k < W) ? _BTB(i, k) : TB_DIAG;

        if (i > 0 && j > 0 && dir == TB_DIAG) {
            uint8_t ca = codes_a[i-1], cb = codes_b[j-1];
            ta[pos] = decode[ca]; tb2[pos] = decode[cb];
            if (ca == cb) nm++; else nmi++;
            i--; j--;
        } else if (j == 0 || (i > 0 && dir == TB_UP)) {
            ta[pos] = decode[codes_a[i-1]]; tb2[pos] = '-';
            ng++; i--;
        } else {
            ta[pos] = '-'; tb2[pos] = decode[codes_b[j-1]];
            ng++; j--;
        }
        pos++;
    }

    for (int32_t k = 0; k < pos; k++) {
        out_a[k] = ta[pos-1-k]; out_b[k] = tb2[pos-1-k];
    }
    out_a[pos] = '\0'; out_b[pos] = '\0';

    *p_matches = nm; *p_mismatches = nmi; *p_gaps = ng;
    free(H_band); free(tb_band); free(ta); free(tb2);
    return pos;
}

#undef _BH
#undef _BTB

EXPORT int32_t nw_banded(
    const uint8_t* ca, int32_t m,
    const uint8_t* cb, int32_t n,
    const char* decode,
    int32_t match, int32_t mismatch, int32_t gap, int32_t band,
    char* out_a, char* out_b,
    int32_t* score, int32_t* matches, int32_t* mismatches, int32_t* gaps
) {
    return _nw_banded_core(ca, m, cb, n, decode, match, mismatch, gap, band, 0,
                           out_a, out_b, score, matches, mismatches, gaps);
}

EXPORT int32_t nw_banded_semiglobal(
    const uint8_t* ca, int32_t m,
    const uint8_t* cb, int32_t n,
    const char* decode,
    int32_t match, int32_t mismatch, int32_t gap, int32_t band,
    char* out_a, char* out_b,
    int32_t* score, int32_t* matches, int32_t* mismatches, int32_t* gaps
) {
    return _nw_banded_core(ca, m, cb, n, decode, match, mismatch, gap, band, 1,
                           out_a, out_b, score, matches, mismatches, gaps);
}

/* ═══════════════════════════════════════════════════════════════════════════
   STREAMING PARSER  (FASTA + FASTQ)
   ─────────────────────────────────────────────────────────────────────────
   Buffer de 64 KB. memchr() usa SIMD de la libc (AVX2/SSE2/NEON) para
   encontrar saltos de línea sin escribir intrínsecos manuales.
   La secuencia se codifica a BioCode 5-bit directamente en C, sin pasar
   nunca por strings Python — esto elimina el cuello de botella principal
   de Biopython y del SmartImporter.stream() en Python puro.

   API:
     bio_parser_open(path)           → handle opaco  (NULL si falla)
     bio_parser_next(handle, …)      → 1 OK | 0 EOF | -1 error | -2 overflow
     bio_parser_close(handle)
   ═══════════════════════════════════════════════════════════════════════════ */

#define PBUF 65536   /* 64 KB de buffer de lectura */

#define PSCR_HDR 4096          /* cabecera máxima en modo batch */
#define PSCR_CAP (1 << 24)     /* 16 MB scratch por secuencia (modo batch) */

typedef struct {
    BIO_FILE fp;
    uint8_t buf[PBUF];
    int32_t beg, end;
    int     eof;
    int     fmt;    /* 1 = FASTA, 2 = FASTQ */
    int     first;  /* 1 = no se ha leído aún el primer registro */
    uint8_t nuc_lut[256];
    uint8_t aa_lut[256];
    uint8_t is_prot[256];

    /* ── scratch para el parser por lotes (bio_parser_next_batch) ───────── */
    char     scr_hdr[PSCR_HDR];
    uint8_t* scr_codes;     /* malloc PSCR_CAP, NULL hasta el primer batch */
    uint8_t* scr_qual;      /* malloc PSCR_CAP, NULL hasta el primer batch */
    int32_t  scr_n, scr_q, scr_type;
    int      has_pending;   /* 1 = hay un registro en scratch sin emitir */
} BioParser;

/* ── LUTs idénticas a las de Python (mismo esquema BioCode 5-bit) ───────── */
static void _init_luts(BioParser* p) {
    memset(p->nuc_lut, 31, 256);
    p->nuc_lut['A'] = p->nuc_lut['a'] =  0;
    p->nuc_lut['C'] = p->nuc_lut['c'] =  1;
    p->nuc_lut['G'] = p->nuc_lut['g'] =  2;
    p->nuc_lut['T'] = p->nuc_lut['t'] =  3;
    p->nuc_lut['U'] = p->nuc_lut['u'] =  3;
    p->nuc_lut['N'] = p->nuc_lut['n'] = 31;
    p->nuc_lut['-'] = p->nuc_lut['.'] = 25;

    memset(p->aa_lut, 31, 256);
    p->aa_lut['A'] = p->aa_lut['a'] =  4;
    p->aa_lut['C'] = p->aa_lut['c'] =  5;
    p->aa_lut['D'] = p->aa_lut['d'] =  6;
    p->aa_lut['E'] = p->aa_lut['e'] =  7;
    p->aa_lut['F'] = p->aa_lut['f'] =  8;
    p->aa_lut['G'] = p->aa_lut['g'] =  9;
    p->aa_lut['H'] = p->aa_lut['h'] = 10;
    p->aa_lut['I'] = p->aa_lut['i'] = 11;
    p->aa_lut['K'] = p->aa_lut['k'] = 12;
    p->aa_lut['L'] = p->aa_lut['l'] = 13;
    p->aa_lut['M'] = p->aa_lut['m'] = 14;
    p->aa_lut['N'] = p->aa_lut['n'] = 15;
    p->aa_lut['P'] = p->aa_lut['p'] = 16;
    p->aa_lut['Q'] = p->aa_lut['q'] = 17;
    p->aa_lut['R'] = p->aa_lut['r'] = 18;
    p->aa_lut['S'] = p->aa_lut['s'] = 19;
    p->aa_lut['T'] = p->aa_lut['t'] = 20;
    p->aa_lut['V'] = p->aa_lut['v'] = 21;
    p->aa_lut['W'] = p->aa_lut['w'] = 22;
    p->aa_lut['Y'] = p->aa_lut['y'] = 23;
    p->aa_lut['*'] = 24;
    p->aa_lut['-'] = 25;
    p->aa_lut['X'] = p->aa_lut['x'] = 31;

    memset(p->is_prot, 0, 256);
    p->is_prot['E'] = p->is_prot['e'] = 1;
    p->is_prot['F'] = p->is_prot['f'] = 1;
    p->is_prot['I'] = p->is_prot['i'] = 1;
    p->is_prot['L'] = p->is_prot['l'] = 1;
    p->is_prot['P'] = p->is_prot['p'] = 1;
    p->is_prot['Q'] = p->is_prot['q'] = 1;
    p->is_prot['*'] = 1;
}

/* ── Rellena el buffer conservando bytes no consumidos ─────────────────── */
static void _refill_buf(BioParser* p) {
    if (p->eof) return;
    int32_t rem = p->end - p->beg;
    if (rem > 0) memmove(p->buf, p->buf + p->beg, (size_t)rem);
    p->beg = 0; p->end = rem;
    int32_t n = (int32_t)BIO_READ(p->fp, p->buf + rem, PBUF - rem);
    if (n <= 0) { p->eof = 1; n = 0; }   /* 0 = EOF, -1 = error de descompresión */
    p->end += n;
}

/* ── Primer byte disponible sin consumirlo, o -1 en EOF ─────────────────── */
static int _peek(BioParser* p) {
    if (p->beg >= p->end && !p->eof) _refill_buf(p);
    return (p->beg < p->end) ? (int)p->buf[p->beg] : -1;
}

/* ── Busca 'ch' usando memchr (SIMD en libc moderna), rellenando si hace falta ─ */
static uint8_t* _find_ch(BioParser* p, uint8_t ch) {
    for (;;) {
        uint8_t* f = (uint8_t*)memchr(
            p->buf + p->beg, (int)ch, (size_t)(p->end - p->beg));
        if (f) return f;
        if (p->eof) return NULL;
        _refill_buf(p);
        if (p->end == 0) return NULL;
    }
}

/* ── Lee texto hasta '\n', escribe en out[] si out!=NULL.
      Elimina '\r'. Retorna longitud leída (sin '\n'), o -1 si max es 0. ─── */
static int32_t _readline(BioParser* p, char* out, int32_t max) {
    if (out && max <= 0) out = NULL;
    int32_t total = 0;
    for (;;) {
        if (p->beg >= p->end) {
            _refill_buf(p);
            if (p->beg >= p->end) {
                if (out) out[total] = '\0';
                return total;
            }
        }
        uint8_t* nl = (uint8_t*)memchr(
            p->buf + p->beg, '\n', (size_t)(p->end - p->beg));
        int32_t seg_end = nl ? (int32_t)(nl - p->buf) : p->end;
        int32_t seg_len = seg_end - p->beg;
        if (seg_len > 0 && p->buf[seg_end - 1] == '\r') seg_len--;
        if (out) {
            int32_t copy = seg_len;
            if (total + copy >= max) copy = max - 1 - total;
            if (copy > 0) memcpy(out + total, p->buf + p->beg, (size_t)copy);
        }
        total += seg_len;
        p->beg = nl ? seg_end + 1 : p->end;
        if (nl) { if (out) out[total < max ? total : max - 1] = '\0'; return total; }
    }
}

/* ── Lee bytes crudos hasta '\n' en out[]; elimina '\r'.
      Retorna conteo, -1 si overflow. ─────────────────────────────────────── */
static int32_t _readbytes(BioParser* p, uint8_t* out, int32_t max) {
    int32_t total = 0;
    for (;;) {
        if (p->beg >= p->end) {
            _refill_buf(p);
            if (p->beg >= p->end) return total;
        }
        uint8_t* nl = (uint8_t*)memchr(
            p->buf + p->beg, '\n', (size_t)(p->end - p->beg));
        int32_t seg_end = nl ? (int32_t)(nl - p->buf) : p->end;
        int32_t seg_len = seg_end - p->beg;
        if (seg_len > 0 && p->buf[seg_end - 1] == '\r') seg_len--;
        if (total + seg_len > max) return -1;
        if (out && seg_len > 0)
            memcpy(out + total, p->buf + p->beg, (size_t)seg_len);
        total += seg_len;
        p->beg = nl ? seg_end + 1 : p->end;
        if (nl) return total;
    }
}

/* ── Codifica in-place raw_bytes → BioCode usando lut ─────────────────── */
static void _encode_inplace(const uint8_t* lut, uint8_t* data, int32_t n) {
    for (int32_t i = 0; i < n; i++) data[i] = lut[data[i]];
}

/* ── Detecta si hay caracteres exclusivos de proteína ─────────────────── */
static int _has_prot_chars(const BioParser* p, const uint8_t* raw, int32_t n) {
    for (int32_t i = 0; i < n; i++)
        if (p->is_prot[raw[i]]) return 1;
    return 0;
}

/* ─────────────────── API pública ─────────────────────────────────────── */

EXPORT void* bio_parser_open(const char* path) {
    BioParser* p = (BioParser*)calloc(1, sizeof(BioParser));
    if (!p) return NULL;
    p->fp = BIO_OPEN(path);
    if (!p->fp) { free(p); return NULL; }
    _init_luts(p);
    p->first = 1;
    /* Detectar formato: leer primer carácter no-espacio */
    _refill_buf(p);
    while (p->beg < p->end) {
        uint8_t c = p->buf[p->beg];
        if (c == '>') { p->fmt = 1; break; }
        if (c == '@') { p->fmt = 2; break; }
        if (c != '\n' && c != '\r' && c != ' ') break;
        p->beg++;
    }
    if (!p->fmt) p->fmt = 1;   /* por defecto FASTA */
    return (void*)p;
}

/*
 * bio_parser_next — lee el siguiente registro FASTA o FASTQ.
 *
 *   handle      : devuelto por bio_parser_open
 *   hdr/hdr_max : buffer para la cabecera sin '>'/'@' (null-terminado)
 *   codes/codes_max/n_out : buffer de BioCode 5-bit (sin empaquetar), símbolos escritos
 *   force_type  : -1 auto | 0 nucleótido | 1 proteína
 *   type_out    : tipo detectado (0=nuc, 1=prot)
 *   qual/qual_out : calidades Phred crudas (ASCII-33); NULL para FASTA
 *
 * Retorna: 1=registro leído | 0=EOF | -1=error | -2=overflow de buffer
 */
static int32_t _parse_one(
    BioParser* p,
    char*    hdr,      int32_t hdr_max,
    uint8_t* codes,    int32_t codes_max,  int32_t* n_out,
    int32_t  force_type,
    int32_t* type_out,
    uint8_t* qual,     int32_t* qual_out
) {
    if (!p) return -1;
    *n_out = 0;
    if (type_out) *type_out = 0;
    if (qual_out) *qual_out = 0;

    /* Bucle externo: los registros vacíos se SALTAN (no detienen la lectura).
       Devolver 0 solo en el fin de archivo real — nunca por un registro vacío,
       que de otro modo truncaría el resto del fichero. */
    for (;;) {
    if (p->fmt == 1) {
        /* ── FASTA ─────────────────────────────────────────────────────── */
        uint8_t* gt = _find_ch(p, '>');
        if (!gt) return 0;
        p->beg = (int32_t)(gt - p->buf) + 1;   /* saltar '>' */

        _readline(p, hdr, hdr_max);

        int32_t total = 0;
        for (;;) {
            int c = _peek(p);
            if (c < 0 || c == '>')   break;             /* EOF / siguiente registro */
            if (c == '\n' || c == '\r') { p->beg++; continue; }  /* línea vacía */
            if (c == ';') { _readline(p, NULL, 0); continue; }    /* comentario */
            int32_t n = _readbytes(p, codes + total, codes_max - total);
            if (n < 0) { *n_out = total; return -2; }
            total += n;
        }

        if (total > 0) {
            int type = (force_type >= 0) ? force_type
                     : (_has_prot_chars(p, codes, total) ? 1 : 0);
            _encode_inplace(type ? p->aa_lut : p->nuc_lut, codes, total);
            if (type_out) *type_out = type;
            *n_out = total;
            return 1;
        }
        if (_peek(p) < 0) return 0;   /* fin de archivo real */
        continue;                      /* registro vacío: saltar al siguiente */

    } else {
        /* ── FASTQ ─────────────────────────────────────────────────────── */
        if (p->first) {
            /* Sincronizar al primer '@' */
            uint8_t* at = _find_ch(p, '@');
            if (!at) return 0;
            p->beg  = (int32_t)(at - p->buf) + 1;
            p->first = 0;
        } else {
            /* Registros posteriores: el buffer apunta justo al '@' */
            if (p->beg >= p->end) _refill_buf(p);
            if (p->beg >= p->end) return 0;
            if (p->buf[p->beg] == '@') p->beg++;
        }

        /* Línea 1: cabecera */
        _readline(p, hdr, hdr_max);

        /* Línea 2: secuencia */
        int32_t n = _readbytes(p, codes, codes_max);
        if (n < 0) return -2;

        /* Línea 3: '+comment' — consumir y descartar */
        _readline(p, NULL, 0);

        /* Línea 4: calidades */
        int32_t q = 0;
        if (qual && qual_out) {
            q = _readbytes(p, qual, codes_max);
            if (q < 0) q = 0;
            for (int32_t i = 0; i < q; i++)
                qual[i] = (uint8_t)((qual[i] >= 33u) ? qual[i] - 33u : 0u);
        } else {
            _readline(p, NULL, 0);
        }

        if (n > 0) {
            int type = (force_type == 1) ? 1 : 0;
            _encode_inplace(type ? p->aa_lut : p->nuc_lut, codes, n);
            if (type_out) *type_out = type;
            if (qual_out) *qual_out = q;
            *n_out = n;
            return 1;
        }
        if (_peek(p) < 0) return 0;   /* fin de archivo real */
        continue;                      /* lectura vacía: saltar a la siguiente */
    }
    }   /* for (;;) */
}

/* Envoltorio público de un solo registro (compatibilidad v2.0). */
EXPORT int32_t bio_parser_next(
    void*    handle,
    char*    hdr,      int32_t hdr_max,
    uint8_t* codes,    int32_t codes_max,  int32_t* n_out,
    int32_t  force_type,
    int32_t* type_out,
    uint8_t* qual,     int32_t* qual_out
) {
    return _parse_one((BioParser*)handle, hdr, hdr_max, codes, codes_max,
                      n_out, force_type, type_out, qual, qual_out);
}

/*
 * bio_parser_next_batch — parsea hasta max_records registros en UNA llamada.
 *
 * Empaqueta cada secuencia a BioCode 5-bit dentro de C (bio_pack5), de modo
 * que Python no cruza la frontera ni llama a pack() por registro. Esto elimina
 * los dos cuellos de botella medidos: el peaje ctypes y el pack NumPy por
 * registro.
 *
 * Buffers de salida (asignados una vez en Python y reutilizados):
 *   hdr_buf  [hdr_buf_max]      cabeceras concatenadas, null-terminadas
 *   hdr_off  [max_records+1]    offset de inicio de cada cabecera en hdr_buf
 *   pack_buf [pack_buf_max]     secuencias 5-bit concatenadas, byte-alineadas
 *   pack_off [max_records+1]    offset de byte de inicio de cada secuencia
 *   n_syms   [max_records]      nº de símbolos por registro (para desempaquetar)
 *   types    [max_records]      tipo (0=nuc, 1=prot) por registro
 *   qual_buf [qual_buf_max]     calidades Phred concatenadas (NULL en FASTA)
 *   qual_off [max_records+1]    offset de inicio de calidades por registro
 *
 * Si un registro no cabe en lo que queda de los buffers, se guarda en scratch
 * (has_pending) y se emite en la siguiente llamada.
 *
 * Retorna: nº de registros parseados (>=0) | -1 error | -2 registro demasiado
 *          grande para el buffer entero.
 */
EXPORT int32_t bio_parser_next_batch(
    void*    handle,    int32_t max_records,  int32_t force_type,
    char*    hdr_buf,   int32_t hdr_buf_max,  int32_t* hdr_off,
    uint8_t* pack_buf,  int32_t pack_buf_max, int32_t* pack_off,
    int32_t* n_syms,    int32_t* types,
    uint8_t* qual_buf,  int32_t qual_buf_max, int32_t* qual_off
) {
    BioParser* p = (BioParser*)handle;
    if (!p) return -1;

    /* Reserva perezosa del scratch en el primer batch. */
    if (!p->scr_codes) {
        p->scr_codes = (uint8_t*)malloc(PSCR_CAP);
        p->scr_qual  = (uint8_t*)malloc(PSCR_CAP);
        if (!p->scr_codes || !p->scr_qual) return -1;
    }

    int32_t count = 0, hdr_used = 0, pack_used = 0, qual_used = 0;
    hdr_off[0] = 0; pack_off[0] = 0;
    if (qual_off) qual_off[0] = 0;

    while (count < max_records) {
        int32_t n, q = 0, type;

        if (p->has_pending) {
            n = p->scr_n; q = p->scr_q; type = p->scr_type;
        } else {
            int32_t r = _parse_one(
                p, p->scr_hdr, PSCR_HDR,
                p->scr_codes, PSCR_CAP, &n,
                force_type, &type,
                qual_buf ? p->scr_qual : NULL, &q);
            if (r == 0) break;      /* EOF */
            if (r < 0) return r;    /* error / registro mayor que el scratch */
        }

        int32_t hlen = (int32_t)strlen(p->scr_hdr);
        int32_t plen = (n * 5 + 7) / 8;

        /* ¿Cabe en lo que queda de los buffers de salida? (+1 slack de pack5) */
        if (hdr_used  + hlen + 1 > hdr_buf_max  ||
            pack_used + plen + 1 > pack_buf_max ||
            (qual_buf && qual_used + q > qual_buf_max)) {
            if (count == 0) return -2;          /* no cabe ni en buffer vacío */
            p->scr_n = n; p->scr_q = q; p->scr_type = type;
            p->has_pending = 1;
            break;
        }
        p->has_pending = 0;

        memcpy(hdr_buf + hdr_used, p->scr_hdr, (size_t)hlen);
        hdr_buf[hdr_used + hlen] = '\0';
        hdr_used += hlen + 1;
        hdr_off[count + 1] = hdr_used;

        bio_pack5(p->scr_codes, n, pack_buf + pack_used);
        pack_used += plen;
        pack_off[count + 1] = pack_used;

        n_syms[count] = n;
        types[count]  = type;

        if (qual_buf) {
            if (q > 0) memcpy(qual_buf + qual_used, p->scr_qual, (size_t)q);
            qual_used += q;
            qual_off[count + 1] = qual_used;
        }
        count++;
    }
    return count;
}

EXPORT void bio_parser_close(void* handle) {
    BioParser* p = (BioParser*)handle;
    if (p) {
        if (p->fp)        BIO_CLOSE(p->fp);
        if (p->scr_codes) free(p->scr_codes);
        if (p->scr_qual)  free(p->scr_qual);
        free(p);
    }
}
