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
    int32_t end_i = m;
    if (semiglobal) {
        /* Mejor score en la última columna */
        int32_t best_sc = H[(int64_t)m * cols + n];
        for (int32_t i = 1; i < m; i++) {
            int32_t sc = H[(int64_t)i * cols + n];
            if (sc > best_sc) { best_sc = sc; end_i = i; }
        }
        *p_score = best_sc;
    } else {
        *p_score = H[(int64_t)m * cols + n];
    }

    /* Buffers temporales para traceback (se invierten al final) */
    char* ta  = (char*)malloc((size_t)(m + n + 1));
    char* tb2 = (char*)malloc((size_t)(m + n + 1));
    if (!ta || !tb2) { free(H); free(tb); free(ta); free(tb2); return -1; }

    int32_t pos = 0, nm = 0, nmi = 0, ng = 0;
    int32_t i = end_i, j = n;

    /* Semi-global: añadir gaps finales de seq_a si end_i < m */
    if (semiglobal) {
        for (int32_t k = m; k > end_i; k--) {
            ta[pos]  = decode[(uint8_t)codes_a[k - 1]];
            tb2[pos] = '-';
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
