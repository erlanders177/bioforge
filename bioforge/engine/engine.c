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

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

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
