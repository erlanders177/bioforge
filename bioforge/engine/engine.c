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
#include <math.h>

/* SIMD (v6.0): la extensión banded se vectoriza con AVX2 si está disponible
   (compilado con -march=native o -mavx2). Si no, cae al kernel escalar. */
#ifdef __AVX2__
  #include <immintrin.h>
#endif

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

/* OpenMP para el parser paralelo en memoria. Sin -fopenmp, se compila en serie. */
#ifdef _OPENMP
  #include <omp.h>
#else
  static int omp_get_max_threads(void) { return 1; }
  static int omp_get_thread_num(void)  { return 0; }
#endif

/* libdeflate: descompresión gzip ~2x más rápida que zlib (SIMD). Opcional. */
#ifdef BIO_USE_LIBDEFLATE
  #include <libdeflate.h>
  EXPORT int bio_has_libdeflate(void) { return 1; }
  /* Descomprime gzip cbuf[0..clen) en obuf (cap ocap). Soporta multi-miembro.
     Devuelve nº de bytes descomprimidos, o -1 si no caben / error. */
  EXPORT int64_t bio_gzip_decompress(const uint8_t* cbuf, int64_t clen,
                                     uint8_t* obuf, int64_t ocap) {
      struct libdeflate_decompressor* d = libdeflate_alloc_decompressor();
      if (!d) return -1;
      int64_t in_pos = 0, out_pos = 0;
      int ok = 1;
      while (in_pos < clen) {
          size_t ain = 0, aout = 0;
          enum libdeflate_result r = libdeflate_gzip_decompress_ex(
              d, cbuf + in_pos, (size_t)(clen - in_pos),
              obuf + out_pos, (size_t)(ocap - out_pos), &ain, &aout);
          if (r != LIBDEFLATE_SUCCESS) { ok = 0; break; }
          in_pos  += (int64_t)ain;
          out_pos += (int64_t)aout;
          if (ain == 0) break;   /* sin avance: evitar bucle infinito */
      }
      libdeflate_free_decompressor(d);
      return ok ? out_pos : -1;
  }

  /* ── BGZF: gzip por bloques independientes → descompresión PARALELA ──────
     Un BGZF es un .gz válido (cada bloque es un miembro gzip con un subcampo
     extra 'BC' que da BSIZE). Los bloques se descomprimen en paralelo. */

  /* ¿Es BGZF? (primer bloque con subcampo extra 'BC'). */
  EXPORT int bio_is_bgzf(const uint8_t* c, int64_t n) {
      if (n < 18) return 0;
      if (c[0] != 0x1f || c[1] != 0x8b || c[2] != 8 || !(c[3] & 4)) return 0;
      int xlen = c[10] | (c[11] << 8);
      int64_t p = 12, e = 12 + (int64_t)xlen;
      if (e > n) return 0;
      while (p + 4 <= e) {
          int slen = c[p + 2] | (c[p + 3] << 8);
          if (c[p] == 66 && c[p + 1] == 67) return 1;   /* 'B','C' */
          p += 4 + slen;
      }
      return 0;
  }

  /* Tamaño del bloque BGZF en 'o' (BSIZE+1), o -1 si malformado. */
  static int64_t _bgzf_block_size(const uint8_t* c, int64_t clen, int64_t o) {
      if (o + 18 > clen || c[o] != 0x1f || c[o + 1] != 0x8b) return -1;
      int xlen = c[o + 10] | (c[o + 11] << 8);
      int64_t p = o + 12, e = o + 12 + xlen;
      while (p + 4 <= e) {
          int slen = c[p + 2] | (c[p + 3] << 8);
          if (c[p] == 66 && c[p + 1] == 67)
              return (int64_t)(c[p + 4] | (c[p + 5] << 8)) + 1;
          p += 4 + slen;
      }
      return -1;
  }

  /* Tamaño total descomprimido de un BGZF (suma de ISIZE), o -1 si malformado. */
  EXPORT int64_t bio_bgzf_usize(const uint8_t* c, int64_t clen) {
      int64_t o = 0, utot = 0;
      while (o < clen) {
          int64_t bs = _bgzf_block_size(c, clen, o);
          if (bs < 0 || o + bs > clen) return -1;
          int32_t isize; memcpy(&isize, c + o + bs - 4, 4);
          utot += isize;
          o += bs;
      }
      return utot;
  }

  /* Descomprime un BGZF en paralelo. Devuelve bytes descomprimidos, o -1. */
  EXPORT int64_t bio_bgzf_decompress_parallel(
      const uint8_t* c, int64_t clen, uint8_t* obuf, int64_t ocap,
      int n_threads)
  {
      /* Pass 0: contar bloques no vacíos y total descomprimido. */
      int64_t o = 0, nb = 0, utot = 0;
      while (o < clen) {
          int64_t bs = _bgzf_block_size(c, clen, o);
          if (bs < 0 || o + bs > clen) return -1;
          int32_t isize; memcpy(&isize, c + o + bs - 4, 4);
          if (isize > 0) { nb++; utot += isize; }
          o += bs;
      }
      if (utot > ocap) return -1;
      if (nb == 0) return 0;

      int64_t* coff = (int64_t*)malloc((size_t)nb * sizeof(int64_t));
      int32_t* csz  = (int32_t*)malloc((size_t)nb * sizeof(int32_t));
      int64_t* uoff = (int64_t*)malloc((size_t)nb * sizeof(int64_t));
      int32_t* usz  = (int32_t*)malloc((size_t)nb * sizeof(int32_t));
      if (!coff || !csz || !uoff || !usz) {
          free(coff); free(csz); free(uoff); free(usz); return -1;
      }

      /* Pass 1: registrar offsets de cada bloque. */
      o = 0; int64_t i = 0, uo = 0;
      while (o < clen) {
          int64_t bs = _bgzf_block_size(c, clen, o);
          int32_t isize; memcpy(&isize, c + o + bs - 4, 4);
          if (isize > 0) {
              coff[i] = o; csz[i] = (int32_t)bs;
              uoff[i] = uo; usz[i] = isize;
              uo += isize; i++;
          }
          o += bs;
      }

      int NT = n_threads;
      if (NT < 1) NT = 1;
      if (NT > omp_get_max_threads()) NT = omp_get_max_threads();
      if (NT > 64) NT = 64;

      /* Pass 2: descomprimir bloques en paralelo (1 descompresor por hilo). */
      int err = 0;
      #pragma omp parallel num_threads(NT)
      {
          /* El descompresor de libdeflate NO es seguro entre hilos: uno por hilo.
             El bucle de trabajo va FUERA del if para que TODOS los hilos del
             equipo encuentren el mismo 'omp for' (si no, deadlock en su barrera). */
          struct libdeflate_decompressor* d = libdeflate_alloc_decompressor();
          if (!d) err = 1;
          #pragma omp for schedule(static)
          for (int64_t k = 0; k < nb; k++) {
              if (!d) continue;
              size_t actual = 0;
              enum libdeflate_result r = libdeflate_gzip_decompress(
                  d, c + coff[k], (size_t)csz[k],
                  obuf + uoff[k], (size_t)usz[k], &actual);
              if (r != LIBDEFLATE_SUCCESS || (int32_t)actual != usz[k])
                  err = 1;
          }
          if (d) libdeflate_free_decompressor(d);
      }
      free(coff); free(csz); free(uoff); free(usz);
      return err ? -1 : utot;
  }

  /* Marcador EOF estándar de BGZF (bloque vacío de 28 bytes). */
  static const uint8_t _BGZF_EOF[28] = {
      0x1f,0x8b,0x08,0x04,0,0,0,0,0,0xff,6,0,0x42,0x43,
      2,0,0x1b,0,3,0,0,0,0,0,0,0,0,0
  };

  /* Comprime ``in`` a BGZF en paralelo (bloques de 64 KB independientes).
     Devuelve el tamaño comprimido, o -1 si no cabe / error. */
  EXPORT int64_t bio_bgzf_compress(const uint8_t* in, int64_t in_len,
                                   uint8_t* out, int64_t out_cap,
                                   int level, int n_threads) {
      const int64_t CHUNK = 0xff00;   /* 65280: máx. descomprimido por bloque */
      int64_t nb = (in_len + CHUNK - 1) / CHUNK;
      int NT = n_threads;
      if (NT < 1) NT = 1;
      if (NT > omp_get_max_threads()) NT = omp_get_max_threads();
      if (NT > 64) NT = 64;

      uint8_t** bufs = NULL; int32_t* lens = NULL; int err = 0;
      if (nb > 0) {
          bufs = (uint8_t**)calloc((size_t)nb, sizeof(uint8_t*));
          lens = (int32_t*)malloc((size_t)nb * sizeof(int32_t));
          if (!bufs || !lens) { free(bufs); free(lens); return -1; }

          #pragma omp parallel num_threads(NT)
          {
              /* Un compresor por hilo. El 'omp for' va FUERA del if para que
                 todos los hilos lo encuentren (si no, deadlock en la barrera). */
              struct libdeflate_compressor* comp =
                  libdeflate_alloc_compressor(level);
              if (!comp) err = 1;
              #pragma omp for schedule(static)
              for (int64_t i = 0; i < nb; i++) {
                  if (!comp) continue;
                  int64_t off = i * CHUNK;
                  int32_t csz = (int32_t)((in_len - off < CHUNK)
                                          ? in_len - off : CHUNK);
                  size_t bound = libdeflate_deflate_compress_bound(comp, csz);
                  uint8_t* blk = (uint8_t*)malloc(18 + bound + 8);
                  if (!blk) { err = 1; continue; }
                  size_t dlen = libdeflate_deflate_compress(
                      comp, in + off, csz, blk + 18, bound);
                  if (dlen == 0) { err = 1; free(blk); continue; }
                  int64_t total = 18 + (int64_t)dlen + 8;
                  int bsize = (int)total - 1;
                  blk[0]=0x1f; blk[1]=0x8b; blk[2]=0x08; blk[3]=0x04;
                  blk[4]=blk[5]=blk[6]=blk[7]=0;          /* mtime */
                  blk[8]=0; blk[9]=0xff;                  /* xfl, os */
                  blk[10]=6; blk[11]=0;                   /* xlen */
                  blk[12]=0x42; blk[13]=0x43;             /* 'B','C' */
                  blk[14]=2; blk[15]=0;                   /* slen */
                  blk[16]=(uint8_t)(bsize & 0xff);
                  blk[17]=(uint8_t)((bsize >> 8) & 0xff);
                  uint32_t crc = libdeflate_crc32(0, in + off, csz);
                  memcpy(blk + 18 + dlen, &crc, 4);
                  uint32_t isize = (uint32_t)csz;
                  memcpy(blk + 18 + dlen + 4, &isize, 4);
                  bufs[i] = blk; lens[i] = (int32_t)total;
              }
              if (comp) libdeflate_free_compressor(comp);
          }
      }

      int64_t pos = 0;
      if (!err) {
          for (int64_t i = 0; i < nb; i++) {
              if (!bufs[i]) { err = 1; break; }
              if (pos + lens[i] > out_cap) { err = 1; break; }
              memcpy(out + pos, bufs[i], (size_t)lens[i]);
              pos += lens[i];
          }
          if (!err && pos + 28 <= out_cap) {
              memcpy(out + pos, _BGZF_EOF, 28); pos += 28;
          } else if (!err) {
              err = 1;
          }
      }
      for (int64_t i = 0; i < nb; i++) free(bufs ? bufs[i] : NULL);
      free(bufs); free(lens);
      return err ? -1 : pos;
  }
#else
  EXPORT int bio_has_libdeflate(void) { return 0; }
  EXPORT int64_t bio_gzip_decompress(const uint8_t* cbuf, int64_t clen,
                                     uint8_t* obuf, int64_t ocap) {
      (void)cbuf; (void)clen; (void)obuf; (void)ocap; return -1;
  }
  EXPORT int bio_is_bgzf(const uint8_t* c, int64_t n) {
      (void)c; (void)n; return 0;
  }
  EXPORT int64_t bio_bgzf_usize(const uint8_t* c, int64_t clen) {
      (void)c; (void)clen; return -1;
  }
  EXPORT int64_t bio_bgzf_decompress_parallel(
      const uint8_t* c, int64_t clen, uint8_t* obuf, int64_t ocap, int nt) {
      (void)c; (void)clen; (void)obuf; (void)ocap; (void)nt; return -1;
  }
  EXPORT int64_t bio_bgzf_compress(const uint8_t* in, int64_t in_len,
                                   uint8_t* out, int64_t out_cap,
                                   int level, int nt) {
      (void)in;(void)in_len;(void)out;(void)out_cap;(void)level;(void)nt;
      return -1;
  }
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

/* Desempaqueta n códigos hacia out[n].
   Autónomo y seguro en los límites: nunca lee más allá de packed[plen-1], de
   modo que el llamante NO necesita un byte extra de relleno. Esto elimina una
   copia completa del array en cada decode() (ruta caliente: alineador, traductor,
   GC/k-meros). El bucle es trivialmente auto-vectorizable por GCC -O3. */
EXPORT void bio_unpack5(const uint8_t* packed, int32_t n, uint8_t* out) {
    int32_t plen = (n * 5 + 7) / 8;
    for (int32_t i = 0; i < n; i++) {
        uint32_t bit_start = (uint32_t)i * 5u;
        uint32_t byte0  = bit_start >> 3;
        uint32_t shift0 = bit_start & 7u;
        uint16_t word   = (uint16_t)packed[byte0] << 8;
        uint32_t b1     = byte0 + 1u;
        if (shift0 > 3u && b1 < (uint32_t)plen) word |= packed[b1];
        out[i] = (uint8_t)((word >> (11u - shift0)) & 0x1Fu);
    }
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
   BANDED NW EN ORDEN ANTIDIAGONAL  (base para el SIMD de v6.0)
   ─────────────────────────────────────────────────────────────────────────
   Mismo DP banded GLOBAL que _nw_banded_core, pero recorriendo las celdas por
   antidiagonales t = i+j (independientes entre sí: diag←t-2, up/left←t-1). Esa
   independencia es lo que luego permite procesar una antidiagonal entera con
   SIMD. Aquí es aún ESCALAR — su única misión es producir salida BIT-IDÉNTICA a
   _nw_banded_core (verificado con parity) para que el paso SIMD solo cambie el
   bucle interno, no la lógica.

   Almacenamiento: la matriz de traceback tb_band[(m+1)*W] es idéntica a la del
   core (indexada por (i,k), k=j-i+band) → el traceback se reutiliza tal cual.
   Para H solo se guardan TRES antidiagonales rotatorias indexadas por i (no la
   matriz entera) → menos memoria. Empate diag>up>left, idéntico al core.  */
static int32_t _ceil_div2(int32_t a) {          /* ceil(a/2) con a posible < 0 */
    return (a >= 0) ? (a + 1) / 2 : -((-a) / 2);
}

static int32_t _nw_banded_diag_global(
    const uint8_t* A, int32_t m,
    const uint8_t* B, int32_t n,
    const char*    decode,
    int32_t match_sc, int32_t mismatch_sc, int32_t gap_sc,
    int32_t band,
    char* out_a, char* out_b,
    int32_t* p_score, int32_t* p_matches, int32_t* p_mismatches, int32_t* p_gaps
) {
    const int32_t W   = 2 * band + 1;
    const int32_t NEG = INT32_MIN / 2;

    uint8_t* tb_band = (uint8_t*)malloc((size_t)(m + 1) * W);
    if (!tb_band) return -1;
    memset(tb_band, TB_DIAG, (size_t)(m + 1) * W);

    /* 3 antidiagonales de H indexadas por i (con celda -1 y m+1 de guarda). */
    int32_t* Hp2 = (int32_t*)malloc((size_t)(m + 3) * sizeof(int32_t));
    int32_t* Hp1 = (int32_t*)malloc((size_t)(m + 3) * sizeof(int32_t));
    int32_t* Hc  = (int32_t*)malloc((size_t)(m + 3) * sizeof(int32_t));
    if (!Hp2 || !Hp1 || !Hc) { free(tb_band); free(Hp2); free(Hp1); free(Hc); return -1; }
    for (int32_t i = 0; i < m + 3; i++) { Hp2[i] = NEG; Hp1[i] = NEG; Hc[i] = NEG; }

    #define HH(arr, i) (arr)[(i) + 1]          /* celda fila i (offset +1) */

    const int32_t tmax = m + n;
    int32_t score = NEG;
    for (int32_t t = 0; t <= tmax; t++) {
        int32_t i_lo = 0;
        if (t - n > i_lo) i_lo = t - n;
        int32_t cb = _ceil_div2(t - band);
        if (cb > i_lo) i_lo = cb;
        int32_t i_hi = m;
        if (t < i_hi) i_hi = t;
        int32_t fb = (t + band) / 2;
        if (fb < i_hi) i_hi = fb;

        for (int32_t i = i_lo; i <= i_hi; i++) {
            int32_t j = t - i;
            int32_t k = j - i + band;          /* columna en tb_band */
            int32_t best; uint8_t dir;
            if (i == 0 && j == 0) {
                best = 0; dir = TB_DIAG;
            } else if (i == 0) {               /* fila 0: gaps a la izquierda */
                best = j * gap_sc; dir = TB_LEFT;
            } else if (j == 0) {               /* columna 0: gaps arriba */
                best = i * gap_sc; dir = TB_UP;
            } else {
                int32_t s  = (A[i-1] == B[j-1]) ? match_sc : mismatch_sc;
                int32_t dv = HH(Hp2, i - 1);   /* diag (i-1,j-1) en t-2 */
                int32_t uv = HH(Hp1, i - 1);   /* up   (i-1,j)   en t-1 */
                int32_t lv = HH(Hp1, i);       /* left (i,j-1)   en t-1 */
                int32_t d = (dv != NEG) ? dv + s      : NEG;
                int32_t u = (uv != NEG) ? uv + gap_sc : NEG;
                int32_t l = (lv != NEG) ? lv + gap_sc : NEG;
                if      (d >= u && d >= l && d != NEG) { best = d; dir = TB_DIAG; }
                else if (u >= l && u != NEG)           { best = u; dir = TB_UP;   }
                else if (l != NEG)                     { best = l; dir = TB_LEFT; }
                else                                   { best = NEG; dir = TB_DIAG; }
            }
            HH(Hc, i) = best;
            tb_band[(int64_t)i * W + k] = dir;
        }
        /* guardas fuera de la ventana (acotadas al rango válido [-1, m+1]:
           con banda < |m-n| la ventana puede quedar vacía y más allá del array). */
        if (i_lo - 1 >= -1 && i_lo - 1 <= m + 1) HH(Hc, i_lo - 1) = NEG;
        if (i_hi + 1 >= -1 && i_hi + 1 <= m + 1) HH(Hc, i_hi + 1) = NEG;
        if (t == tmax) score = HH(Hc, m);

        int32_t* tmp = Hp2; Hp2 = Hp1; Hp1 = Hc; Hc = tmp;   /* rotar */
    }
    #undef HH
    free(Hp2); free(Hp1); free(Hc);

    /* Igual que el core: si el endpoint (m,n) queda fuera de banda, score=NEG. */
    int32_t k_end = n - m + band;
    if (!(k_end >= 0 && k_end < W)) score = NEG;
    *p_score = score;

    /* ── Traceback (idéntico al de _nw_banded_core, modo global) ─────────── */
    char* ta  = (char*)malloc((size_t)(m + n + 1));
    char* tb2 = (char*)malloc((size_t)(m + n + 1));
    if (!ta || !tb2) { free(tb_band); free(ta); free(tb2); return -1; }

    int32_t pos = 0, nm = 0, nmi = 0, ng = 0;
    int32_t i = m, j = n;
    while (i > 0 || j > 0) {
        int32_t k = j - i + band;
        uint8_t dir = (k >= 0 && k < W) ? tb_band[(int64_t)i * W + k] : TB_DIAG;
        if (i > 0 && j > 0 && dir == TB_DIAG) {
            uint8_t ca = A[i-1], cbb = B[j-1];
            ta[pos] = decode[ca]; tb2[pos] = decode[cbb];
            if (ca == cbb) nm++; else nmi++;
            i--; j--;
        } else if (j == 0 || (i > 0 && dir == TB_UP)) {
            ta[pos] = decode[A[i-1]]; tb2[pos] = '-';
            ng++; i--;
        } else {
            ta[pos] = '-'; tb2[pos] = decode[B[j-1]];
            ng++; j--;
        }
        pos++;
    }
    for (int32_t x = 0; x < pos; x++) { out_a[x] = ta[pos-1-x]; out_b[x] = tb2[pos-1-x]; }
    out_a[pos] = '\0'; out_b[pos] = '\0';

    *p_matches = nm; *p_mismatches = nmi; *p_gaps = ng;
    free(tb_band); free(ta); free(tb2);
    return pos;
}

/* Export de prueba (global) — para verificar paridad con nw_banded. */
EXPORT int32_t nw_banded_diag(
    const uint8_t* ca, int32_t m,
    const uint8_t* cb, int32_t n,
    const char* decode,
    int32_t match, int32_t mismatch, int32_t gap, int32_t band,
    char* out_a, char* out_b,
    int32_t* score, int32_t* matches, int32_t* mismatches, int32_t* gaps
) {
    return _nw_banded_diag_global(ca, m, cb, n, decode, match, mismatch, gap, band,
                                  out_a, out_b, score, matches, mismatches, gaps);
}

/* ═══════════════════════════════════════════════════════════════════════════
   BANDED NW ANTIDIAGONAL — VERSIÓN SIMD (AVX2, 8× int32)  · v6.0 incr.2
   ─────────────────────────────────────────────────────────────────────────
   Mismo DP que _nw_banded_diag_global, pero procesa 8 celdas interiores de la
   antidiagonal por instrucción. Las celdas de borde (fila 0 / columna 0) y el
   resto (<8) van en escalar con las MISMAS fórmulas → salida bit-idéntica.
   Cargas contiguas (Hp1/Hp2 indexadas por i; A por i); B se carga y se invierte
   con permutevar. Escrituras de dirección dispersas (stride W-2) en escalar.
   Sin __AVX2__ → cae al escalar verificado.  */
static int32_t _nw_banded_diag_simd(
    const uint8_t* A, int32_t m,
    const uint8_t* B, int32_t n,
    const char*    decode,
    int32_t match_sc, int32_t mismatch_sc, int32_t gap_sc,
    int32_t band,
    char* out_a, char* out_b,
    int32_t* p_score, int32_t* p_matches, int32_t* p_mismatches, int32_t* p_gaps
) {
#ifndef __AVX2__
    return _nw_banded_diag_global(A, m, B, n, decode, match_sc, mismatch_sc,
                                  gap_sc, band, out_a, out_b,
                                  p_score, p_matches, p_mismatches, p_gaps);
#else
    const int32_t W   = 2 * band + 1;
    const int32_t NEG = INT32_MIN / 2;

    uint8_t* tb_band = (uint8_t*)malloc((size_t)(m + 1) * W);
    if (!tb_band) return -1;
    memset(tb_band, TB_DIAG, (size_t)(m + 1) * W);

    int32_t* Hp2 = (int32_t*)malloc((size_t)(m + 3) * sizeof(int32_t));
    int32_t* Hp1 = (int32_t*)malloc((size_t)(m + 3) * sizeof(int32_t));
    int32_t* Hc  = (int32_t*)malloc((size_t)(m + 3) * sizeof(int32_t));
    if (!Hp2 || !Hp1 || !Hc) { free(tb_band); free(Hp2); free(Hp1); free(Hc); return -1; }
    for (int32_t i = 0; i < m + 3; i++) { Hp2[i] = NEG; Hp1[i] = NEG; Hc[i] = NEG; }

    /* celda escalar (i,j=t-i); Hp2[i]=diag, Hp1[i]=up, Hp1[i+1]=left, Hc[i+1]=out */
    #define CELL(i) do {                                                        \
        int32_t _i = (i), _j = t - _i, _k = _j - _i + band;                     \
        int32_t _best; uint8_t _dir;                                            \
        if (_i == 0 && _j == 0) { _best = 0; _dir = TB_DIAG; }                  \
        else if (_i == 0) { _best = _j * gap_sc; _dir = TB_LEFT; }              \
        else if (_j == 0) { _best = _i * gap_sc; _dir = TB_UP; }                \
        else {                                                                  \
            int32_t _s  = (A[_i-1] == B[_j-1]) ? match_sc : mismatch_sc;        \
            int32_t _dv = Hp2[_i], _uv = Hp1[_i], _lv = Hp1[_i+1];              \
            int32_t _d = (_dv != NEG) ? _dv + _s     : NEG;                     \
            int32_t _u = (_uv != NEG) ? _uv + gap_sc : NEG;                     \
            int32_t _l = (_lv != NEG) ? _lv + gap_sc : NEG;                     \
            if      (_d >= _u && _d >= _l && _d != NEG) { _best=_d; _dir=TB_DIAG; } \
            else if (_u >= _l && _u != NEG)             { _best=_u; _dir=TB_UP;   } \
            else if (_l != NEG)                         { _best=_l; _dir=TB_LEFT; } \
            else                                        { _best=NEG; _dir=TB_DIAG; } \
        }                                                                       \
        Hc[_i+1] = _best; tb_band[(int64_t)_i * W + _k] = _dir;                 \
    } while (0)

    const __m256i vNEG   = _mm256_set1_epi32(NEG);
    const __m256i vmatch = _mm256_set1_epi32(match_sc);
    const __m256i vmis   = _mm256_set1_epi32(mismatch_sc);
    const __m256i vgap   = _mm256_set1_epi32(gap_sc);
    const __m256i vone   = _mm256_set1_epi32(-1);       /* all-ones */
    const __m256i v1     = _mm256_set1_epi32(1);
    const __m256i v2     = _mm256_set1_epi32(2);
    const __m256i revidx = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);

    const int32_t tmax = m + n;
    int32_t score = NEG;
    for (int32_t t = 0; t <= tmax; t++) {
        int32_t i_lo = 0;
        if (t - n > i_lo) i_lo = t - n;
        int32_t cbd = _ceil_div2(t - band);
        if (cbd > i_lo) i_lo = cbd;
        int32_t i_hi = m;
        if (t < i_hi) i_hi = t;
        int32_t fbd = (t + band) / 2;
        if (fbd < i_hi) i_hi = fbd;

        int32_t ii_lo = i_lo < 1 ? 1 : i_lo;      /* interior: i>=1 y j>=1 */
        int32_t ii_hi = i_hi > t - 1 ? t - 1 : i_hi;

        /* bordes bajos (i=0) y cualquier i < ii_lo → escalar */
        for (int32_t i = i_lo; i < ii_lo; i++) CELL(i);

        /* interior: bloques completos de 8 con AVX2 */
        int32_t i = ii_lo;
        for (; i + 8 <= ii_hi + 1; i += 8) {
            __m128i a8 = _mm_loadl_epi64((const __m128i*)(A + i - 1));
            __m256i A32 = _mm256_cvtepu8_epi32(a8);
            __m128i b8 = _mm_loadl_epi64((const __m128i*)(B + (t - i - 8)));
            __m256i B32 = _mm256_permutevar8x32_epi32(_mm256_cvtepu8_epi32(b8), revidx);
            __m256i eq  = _mm256_cmpeq_epi32(A32, B32);
            __m256i s   = _mm256_blendv_epi8(vmis, vmatch, eq);

            __m256i dv = _mm256_loadu_si256((const __m256i*)(Hp2 + i));
            __m256i uv = _mm256_loadu_si256((const __m256i*)(Hp1 + i));
            __m256i lv = _mm256_loadu_si256((const __m256i*)(Hp1 + i + 1));

            __m256i vald = _mm256_andnot_si256(_mm256_cmpeq_epi32(dv, vNEG), vone);
            __m256i valu = _mm256_andnot_si256(_mm256_cmpeq_epi32(uv, vNEG), vone);
            __m256i vall = _mm256_andnot_si256(_mm256_cmpeq_epi32(lv, vNEG), vone);

            __m256i d = _mm256_blendv_epi8(vNEG, _mm256_add_epi32(dv, s),   vald);
            __m256i u = _mm256_blendv_epi8(vNEG, _mm256_add_epi32(uv, vgap), valu);
            __m256i l = _mm256_blendv_epi8(vNEG, _mm256_add_epi32(lv, vgap), vall);

            /* d>=u ⟺ !(u>d) ; d>=l ⟺ !(l>d) ; u>=l ⟺ !(l>u) */
            __m256i d_ge_u = _mm256_andnot_si256(_mm256_cmpgt_epi32(u, d), vone);
            __m256i d_ge_l = _mm256_andnot_si256(_mm256_cmpgt_epi32(l, d), vone);
            __m256i u_ge_l = _mm256_andnot_si256(_mm256_cmpgt_epi32(l, u), vone);

            __m256i cdiag = _mm256_and_si256(vald, _mm256_and_si256(d_ge_u, d_ge_l));
            __m256i ndiag = _mm256_andnot_si256(cdiag, vone);
            __m256i cup   = _mm256_and_si256(ndiag, _mm256_and_si256(valu, u_ge_l));
            __m256i nup   = _mm256_andnot_si256(cup, vone);
            __m256i clft  = _mm256_and_si256(ndiag, _mm256_and_si256(nup, vall));

            __m256i best = vNEG;
            best = _mm256_blendv_epi8(best, l, clft);
            best = _mm256_blendv_epi8(best, u, cup);
            best = _mm256_blendv_epi8(best, d, cdiag);
            _mm256_storeu_si256((__m256i*)(Hc + i + 1), best);

            __m256i dir = _mm256_setzero_si256();      /* DIAG=0 por defecto */
            dir = _mm256_blendv_epi8(dir, v2, clft);
            dir = _mm256_blendv_epi8(dir, v1, cup);
            int32_t dbuf[8];
            _mm256_storeu_si256((__m256i*)dbuf, dir);
            for (int lane = 0; lane < 8; lane++) {
                int32_t ii = i + lane, jj = t - ii, kk = jj - ii + band;
                tb_band[(int64_t)ii * W + kk] = (uint8_t)dbuf[lane];
            }
        }
        for (; i <= ii_hi; i++) CELL(i);           /* resto interior escalar */

        /* bordes altos (i=t → j=0) y cualquier i > ii_hi → escalar */
        for (int32_t ib = ii_hi + 1; ib <= i_hi; ib++) CELL(ib);

        if (i_lo - 1 >= -1 && i_lo - 1 <= m + 1) Hc[(i_lo - 1) + 1] = NEG;
        if (i_hi + 1 >= -1 && i_hi + 1 <= m + 1) Hc[(i_hi + 1) + 1] = NEG;
        if (t == tmax) score = Hc[m + 1];

        int32_t* tmp = Hp2; Hp2 = Hp1; Hp1 = Hc; Hc = tmp;
    }
    #undef CELL
    free(Hp2); free(Hp1); free(Hc);

    int32_t k_end = n - m + band;
    if (!(k_end >= 0 && k_end < W)) score = NEG;
    *p_score = score;

    /* Traceback (idéntico al global) */
    char* ta  = (char*)malloc((size_t)(m + n + 1));
    char* tb2 = (char*)malloc((size_t)(m + n + 1));
    if (!ta || !tb2) { free(tb_band); free(ta); free(tb2); return -1; }
    int32_t pos = 0, nm = 0, nmi = 0, ng = 0;
    int32_t i = m, j = n;
    while (i > 0 || j > 0) {
        int32_t k = j - i + band;
        uint8_t dir = (k >= 0 && k < W) ? tb_band[(int64_t)i * W + k] : TB_DIAG;
        if (i > 0 && j > 0 && dir == TB_DIAG) {
            uint8_t ca = A[i-1], cbb = B[j-1];
            ta[pos] = decode[ca]; tb2[pos] = decode[cbb];
            if (ca == cbb) nm++; else nmi++;
            i--; j--;
        } else if (j == 0 || (i > 0 && dir == TB_UP)) {
            ta[pos] = decode[A[i-1]]; tb2[pos] = '-'; ng++; i--;
        } else {
            ta[pos] = '-'; tb2[pos] = decode[B[j-1]]; ng++; j--;
        }
        pos++;
    }
    for (int32_t x = 0; x < pos; x++) { out_a[x] = ta[pos-1-x]; out_b[x] = tb2[pos-1-x]; }
    out_a[pos] = '\0'; out_b[pos] = '\0';
    *p_matches = nm; *p_mismatches = nmi; *p_gaps = ng;
    free(tb_band); free(ta); free(tb2);
    return pos;
#endif
}

/* Export de prueba de la versión SIMD. */
EXPORT int32_t nw_banded_diag_simd(
    const uint8_t* ca, int32_t m,
    const uint8_t* cb, int32_t n,
    const char* decode,
    int32_t match, int32_t mismatch, int32_t gap, int32_t band,
    char* out_a, char* out_b,
    int32_t* score, int32_t* matches, int32_t* mismatches, int32_t* gaps
) {
    return _nw_banded_diag_simd(ca, m, cb, n, decode, match, mismatch, gap, band,
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

/* ── LUTs idénticas a las de Python (mismo esquema BioCode 5-bit) ─────────
   Standalone para que el parser paralelo construya LUTs en la pila y las
   comparta entre hilos (solo lectura). */
static void _build_luts(uint8_t* nuc_lut, uint8_t* aa_lut, uint8_t* is_prot) {
    memset(nuc_lut, 31, 256);
    nuc_lut['A'] = nuc_lut['a'] =  0;
    nuc_lut['C'] = nuc_lut['c'] =  1;
    nuc_lut['G'] = nuc_lut['g'] =  2;
    nuc_lut['T'] = nuc_lut['t'] =  3;
    nuc_lut['U'] = nuc_lut['u'] =  3;
    nuc_lut['N'] = nuc_lut['n'] = 31;
    nuc_lut['-'] = nuc_lut['.'] = 25;

    memset(aa_lut, 31, 256);
    aa_lut['A'] = aa_lut['a'] =  4;
    aa_lut['C'] = aa_lut['c'] =  5;
    aa_lut['D'] = aa_lut['d'] =  6;
    aa_lut['E'] = aa_lut['e'] =  7;
    aa_lut['F'] = aa_lut['f'] =  8;
    aa_lut['G'] = aa_lut['g'] =  9;
    aa_lut['H'] = aa_lut['h'] = 10;
    aa_lut['I'] = aa_lut['i'] = 11;
    aa_lut['K'] = aa_lut['k'] = 12;
    aa_lut['L'] = aa_lut['l'] = 13;
    aa_lut['M'] = aa_lut['m'] = 14;
    aa_lut['N'] = aa_lut['n'] = 15;
    aa_lut['P'] = aa_lut['p'] = 16;
    aa_lut['Q'] = aa_lut['q'] = 17;
    aa_lut['R'] = aa_lut['r'] = 18;
    aa_lut['S'] = aa_lut['s'] = 19;
    aa_lut['T'] = aa_lut['t'] = 20;
    aa_lut['V'] = aa_lut['v'] = 21;
    aa_lut['W'] = aa_lut['w'] = 22;
    aa_lut['Y'] = aa_lut['y'] = 23;
    aa_lut['*'] = 24;
    aa_lut['-'] = 25;
    aa_lut['X'] = aa_lut['x'] = 31;

    memset(is_prot, 0, 256);
    is_prot['E'] = is_prot['e'] = 1;
    is_prot['F'] = is_prot['f'] = 1;
    is_prot['I'] = is_prot['i'] = 1;
    is_prot['L'] = is_prot['l'] = 1;
    is_prot['P'] = is_prot['p'] = 1;
    is_prot['Q'] = is_prot['q'] = 1;
    is_prot['*'] = 1;
}

static void _init_luts(BioParser* p) {
    _build_luts(p->nuc_lut, p->aa_lut, p->is_prot);
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

/* ═══════════════════════════════════════════════════════════════════════════
   PARSER PARALELO EN MEMORIA  (OpenMP)
   ─────────────────────────────────────────────────────────────────────────
   Parsea un bloque de memoria que contiene SOLO registros completos (el
   llamante en Python garantiza que la ventana termina en un registro completo).
   El bloque se trocea en N rangos alineados a límites de registro; cada hilo
   parsea su rango en buffers propios y luego se fusiona en orden. Salida
   columnar idéntica a bio_parser_next_batch.
   ═══════════════════════════════════════════════════════════════════════════ */

typedef struct { int64_t start, len; } Seg;

/* Lee una línea desde *pos (sin '\n', sin '\r'); avanza *pos tras el '\n'. */
static Seg _mem_line(const uint8_t* D, int64_t L, int64_t* pos) {
    int64_t p = *pos;
    Seg s; s.start = p; s.len = 0;
    if (p >= L) { *pos = p; return s; }
    const uint8_t* nl = (const uint8_t*)memchr(D + p, '\n', (size_t)(L - p));
    int64_t e = nl ? (int64_t)(nl - D) : L;
    int64_t len = e - p;
    if (len > 0 && D[e - 1] == '\r') len--;
    s.len = len;
    *pos = nl ? e + 1 : L;
    return s;
}

/* Siguiente inicio de registro FASTA ('>' a principio de línea) en [from, L). */
static int64_t _next_fasta_start(const uint8_t* D, int64_t L, int64_t from) {
    int64_t p = from;
    while (p < L) {
        const uint8_t* nl = (const uint8_t*)memchr(D + p, '\n', (size_t)(L - p));
        if (!nl) return L;
        int64_t q = (int64_t)(nl - D) + 1;
        if (q < L && D[q] == '>') return q;
        p = q;
    }
    return L;
}

/* Siguiente inicio de registro FASTQ en [from, L). Verifica la estructura de
   4 líneas (la 3ª empieza por '+') para no confundir un '@' de calidad. */
static int64_t _next_fastq_start(const uint8_t* D, int64_t L, int64_t from) {
    int64_t p = from;
    while (p < L) {
        const uint8_t* nl = (const uint8_t*)memchr(D + p, '\n', (size_t)(L - p));
        if (!nl) return L;
        int64_t q = (int64_t)(nl - D) + 1;
        if (q < L && D[q] == '@') {
            int64_t r = q; int ok = 1;
            for (int k = 0; k < 2; k++) {           /* avanzar 2 líneas */
                const uint8_t* n2 =
                    (const uint8_t*)memchr(D + r, '\n', (size_t)(L - r));
                if (!n2) { ok = 0; break; }
                r = (int64_t)(n2 - D) + 1;
            }
            if (ok && r < L && D[r] == '+') return q;
        }
        p = q;
    }
    return L;
}

/* Buffers de salida por hilo. */
typedef struct {
    uint8_t* pack; int64_t pack_len, pack_cap;
    uint8_t* qual; int64_t qual_len, qual_cap;
    char*    hdr;  int64_t hdr_len,  hdr_cap;
    uint8_t* code; int64_t code_cap;          /* scratch para codificar 1 seq */
    int32_t* nsy;  int32_t  nrec;  int64_t rec_cap;
    int32_t* typ;
    int32_t* hln;                              /* longitud de cabecera (incl '\0') */
    int      oom;                              /* 1 si falló una reserva */
} TOut;

static int _ensure(void** buf, int64_t* cap, int64_t need, int64_t esz) {
    if (*cap >= need) return 1;
    int64_t nc = (*cap < 1024) ? 1024 : *cap;
    while (nc < need) nc *= 2;
    void* nb = realloc(*buf, (size_t)(nc * esz));
    if (!nb) return 0;
    *buf = nb; *cap = nc; return 1;
}

/* Crece nsy/typ/hln a la vez (comparten rec_cap). */
static int _ensure_rec(TOut* t) {
    if (t->rec_cap > t->nrec) return 1;
    int64_t nc = (t->rec_cap < 1024) ? 1024 : t->rec_cap * 2;
    int32_t* a = (int32_t*)realloc(t->nsy, (size_t)nc * sizeof(int32_t));
    if (a) t->nsy = a;
    int32_t* b = (int32_t*)realloc(t->typ, (size_t)nc * sizeof(int32_t));
    if (b) t->typ = b;
    int32_t* c = (int32_t*)realloc(t->hln, (size_t)nc * sizeof(int32_t));
    if (c) t->hln = c;
    if (!a || !b || !c) return 0;
    t->rec_cap = nc;
    return 1;
}

/* ── API pública ─────────────────────────────────────────────────────────── */
/*
 * bio_parse_mem_parallel — parsea D[0..len) (solo registros completos) en
 * paralelo. Devuelve nº de registros (>=0) | -1 error | -2 overflow de buffer.
 */
EXPORT int32_t bio_parse_mem_parallel(
    const uint8_t* D, int64_t len,
    int32_t fmt, int32_t n_threads, int32_t force_type,
    char*    hdr_buf,  int32_t  hdr_buf_max, int32_t* hdr_off,
    uint8_t* pack_buf, int64_t  pack_buf_max, int32_t* pack_off,
    int32_t* n_syms,   int32_t* types,
    uint8_t* qual_buf, int64_t  qual_buf_max, int32_t* qual_off,
    int32_t  max_records
) {
    if (!D || len <= 0) return 0;

    uint8_t nuc_lut[256], aa_lut[256], is_prot[256];
    _build_luts(nuc_lut, aa_lut, is_prot);

    int maxt = omp_get_max_threads();
    int NT = n_threads;
    if (NT < 1) NT = 1;
    if (NT > maxt) NT = maxt;
    if (NT > 64)  NT = 64;

    /* Límites de los rangos, alineados a inicio de registro. */
    int64_t bound[65];
    bound[0] = 0;
    for (int t = 1; t < NT; t++) {
        int64_t guess = (int64_t)((double)len * t / NT);
        bound[t] = (fmt == 1) ? _next_fasta_start(D, len, guess)
                              : _next_fastq_start(D, len, guess);
    }
    bound[NT] = len;
    /* Colapsar rangos vacíos o desordenados. */
    for (int t = 1; t <= NT; t++)
        if (bound[t] < bound[t - 1]) bound[t] = bound[t - 1];

    TOut* outs = (TOut*)calloc((size_t)NT, sizeof(TOut));
    if (!outs) return -1;

    #pragma omp parallel num_threads(NT)
    {
        int tid = omp_get_thread_num();
        TOut* t = &outs[tid];
        int64_t lo = bound[tid], hi = bound[tid + 1];
        int64_t pos = lo;

        while (pos < hi) {
            int64_t rec_start = pos;
            int32_t n = 0, type = 0, q = 0;
            const char* hdr_ptr = NULL; int32_t hlen = 0;
            const uint8_t* qsrc = NULL;

            if (fmt == 1) {
                /* FASTA */
                Seg h = _mem_line(D, len, &pos);          /* '>...' */
                hdr_ptr = (const char*)(D + h.start + 1);
                hlen = (int32_t)(h.len > 0 ? h.len - 1 : 0);
                while (pos < len && D[pos] != '>') {
                    Seg sg = _mem_line(D, len, &pos);
                    if (!_ensure((void**)&t->code, &t->code_cap,
                                 (int64_t)n + sg.len, 1)) { t->oom = 1; break; }
                    for (int64_t k = 0; k < sg.len; k++)
                        t->code[n++] = D[sg.start + k];   /* crudo; se codifica abajo */
                }
                if (t->oom) break;
                type = (force_type >= 0) ? force_type : 0;
                if (force_type < 0) {
                    for (int32_t k = 0; k < n; k++)
                        if (is_prot[t->code[k]]) { type = 1; break; }
                }
                const uint8_t* lut = type ? aa_lut : nuc_lut;
                for (int32_t k = 0; k < n; k++) t->code[k] = lut[t->code[k]];
            } else {
                /* FASTQ: 4 líneas */
                Seg h  = _mem_line(D, len, &pos);
                Seg sq = _mem_line(D, len, &pos);
                Seg pl = _mem_line(D, len, &pos); (void)pl;
                Seg ql = _mem_line(D, len, &pos);
                hdr_ptr = (const char*)(D + h.start + 1);
                hlen = (int32_t)(h.len > 0 ? h.len - 1 : 0);
                n = (int32_t)sq.len;
                if (!_ensure((void**)&t->code, &t->code_cap, n, 1)) { t->oom = 1; break; }
                type = (force_type == 1) ? 1 : 0;
                const uint8_t* lut = type ? aa_lut : nuc_lut;
                for (int32_t k = 0; k < n; k++)
                    t->code[k] = lut[D[sq.start + k]];
                q = (int32_t)ql.len;
                qsrc = D + ql.start;
            }

            if (n <= 0) continue;   /* registro vacío: saltar (no contar) */

            /* Reservar y volcar al buffer del hilo. La calidad se escribe con
               longitud == n (se rellena con 0 o se trunca) para mantener el
               invariante calidad==secuencia incluso en FASTQ malformado. */
            int64_t plen = (int64_t)(n * 5 + 7) / 8;
            if (!_ensure((void**)&t->pack, &t->pack_cap, t->pack_len + plen + 1, 1) ||
                !_ensure((void**)&t->hdr,  &t->hdr_cap,  t->hdr_len + hlen + 1, 1) ||
                !_ensure_rec(t) ||
                (fmt == 2 &&
                 !_ensure((void**)&t->qual, &t->qual_cap, t->qual_len + n, 1))) {
                t->oom = 1; break;
            }

            bio_pack5(t->code, n, t->pack + t->pack_len);
            t->pack_len += plen;
            if (hlen > 0) memcpy(t->hdr + t->hdr_len, hdr_ptr, (size_t)hlen);
            t->hdr[t->hdr_len + hlen] = '\0';
            t->hdr_len += hlen + 1;
            if (fmt == 2) {
                for (int32_t k = 0; k < n; k++) {
                    uint8_t c = (k < q) ? qsrc[k] : 33u;
                    t->qual[t->qual_len + k] =
                        (uint8_t)((c >= 33u) ? c - 33u : 0u);
                }
                t->qual_len += n;
            }
            t->nsy[t->nrec] = n;
            t->typ[t->nrec] = type;
            t->hln[t->nrec] = hlen + 1;
            t->nrec++;

            if (rec_start >= hi) break;
        }
    }

    /* ── Fusión serial en orden de hilo ──────────────────────────────────── */
    int rc = 0;
    int32_t total_rec = 0;
    int64_t total_pack = 0, total_qual = 0, total_hdr = 0;
    for (int t = 0; t < NT; t++) {
        if (outs[t].oom) { rc = -1; }
        total_rec  += outs[t].nrec;
        total_pack += outs[t].pack_len;
        total_qual += outs[t].qual_len;
        total_hdr  += outs[t].hdr_len;
    }
    if (rc == 0) {
        if (total_rec > max_records ||
            total_pack + 1 > pack_buf_max ||
            total_hdr > hdr_buf_max ||
            (qual_buf && total_qual > qual_buf_max)) {
            rc = -2;
        }
    }
    if (rc == 0) {
        int32_t ri = 0; int64_t po = 0, qo = 0, ho = 0;
        pack_off[0] = 0; hdr_off[0] = 0;
        if (qual_off) qual_off[0] = 0;
        for (int t = 0; t < NT; t++) {
            TOut* o = &outs[t];
            if (o->pack_len) memcpy(pack_buf + po, o->pack, (size_t)o->pack_len);
            if (o->hdr_len)  memcpy(hdr_buf + ho, o->hdr, (size_t)o->hdr_len);
            if (qual_buf && o->qual_len)
                memcpy(qual_buf + qo, o->qual, (size_t)o->qual_len);
            int64_t lpo = po, lqo = qo, lho = ho;
            for (int32_t j = 0; j < o->nrec; j++) {
                int32_t n = o->nsy[j];
                n_syms[ri] = n;
                types[ri]  = o->typ[j];
                lpo += (int64_t)(n * 5 + 7) / 8; pack_off[ri + 1] = (int32_t)lpo;
                if (qual_off) { lqo += n; qual_off[ri + 1] = (int32_t)lqo; }
                lho += o->hln[j]; hdr_off[ri + 1] = (int32_t)lho;
                ri++;
            }
            po += o->pack_len; qo += o->qual_len; ho += o->hdr_len;
        }
    }

    for (int t = 0; t < NT; t++) {
        free(outs[t].pack); free(outs[t].qual); free(outs[t].hdr);
        free(outs[t].code); free(outs[t].nsy); free(outs[t].typ); free(outs[t].hln);
    }
    free(outs);
    return (rc != 0) ? rc : total_rec;
}

/* ── Chaining DP (alineador v3) ───────────────────────────────────────────────
   Rellena f[] (mejor score de cadena que TERMINA en i) y prev[] (predecesor)
   para anclas ordenadas ascendentemente por (x, y). El backtracking y la
   supresión de solapamientos se hacen en Python (baratos, O(n)).

   x, y     : posiciones de las anclas (ref y coord. de chaining), int64.
   n        : nº de anclas.  k : peso del ancla (longitud del k-mer).
   max_gap  : distancia máxima entre anclas encadenables.
   window   : nº de predecesores examinados por ancla.
   gap_w    : peso lineal de la penalización por hueco.

   Réplica exacta del DP vectorizado de genomemap.py (misma fórmula y mismo
   desempate: al iterar j de i-1 hacia abajo con `>` estricto, en empate gana
   el predecesor MÁS CERCANO). Así el resultado C == fallback NumPy. */
EXPORT void bio_chain_dp(const int64_t* x, const int64_t* y, int32_t n,
                         int32_t k, int64_t max_gap, int32_t window,
                         double gap_w, double* f, int32_t* prev) {
    for (int32_t i = 0; i < n; i++) {
        double  best = (double)k;
        int32_t bp   = -1;
        int32_t lo   = i - window;
        if (lo < 0) lo = 0;
        for (int32_t j = i - 1; j >= lo; j--) {
            int64_t dx = x[i] - x[j];
            if (dx > max_gap) break;              /* ordenado → resto aún más lejos */
            int64_t dy = y[i] - y[j];
            if (dy <= 0 || dx <= 0 || dy > max_gap) continue;
            int64_t gap = dx - dy;
            if (gap < 0) gap = -gap;
            if (gap > max_gap) continue;
            int64_t mn = (dx < dy) ? dx : dy;
            double match = (double)((mn < (int64_t)k) ? mn : (int64_t)k);
            /* coste de hueco estilo minimap2: gap_w·|l| + 0.5·log2|l| (l≠0) */
            double cost  = gap_w * (double)gap;
            if (gap > 0) cost += 0.5 * log2((double)gap);
            double sc = f[j] + match - cost;
            if (sc > best) { best = sc; bp = j; }
        }
        f[i]    = best;
        prev[i] = bp;
    }
}

/* ── Minimizers (w,k) canónicos en C (alineador v3) ───────────────────────────
   Réplica exacta del cálculo vectorizado de minimizers.py:
     - k-mer directo/inverso por hash RODANTE (idéntico al Horner de NumPy),
     - hash invertible _hash64 (mismo esquema), canónico = menor(hf,hr),
     - mínimo por ventana deslizante de w k-mers (argmin más a la IZQUIERDA),
     - dedup de posiciones consecutivas (== np.unique sobre pos no-decrecientes).
   Escala a genomas donde sliding_window_view de NumPy se ahoga.

   codes: uint8 2-bit (>=4 = base inválida). Salidas (tamaño ≥ n-k+1):
   out_hash/out_pos/out_strand. Devuelve el nº de minimizers, o -1 si malloc. */
static uint64_t _hash64_c(uint64_t x, uint64_t mask) {
    x = x & mask;
    x = (~x + (x << 21)) & mask;
    x = x ^ (x >> 24);
    x = ((x + (x << 3)) + (x << 8)) & mask;
    x = x ^ (x >> 14);
    x = ((x + (x << 2)) + (x << 4)) & mask;
    x = x ^ (x >> 28);
    x = (x + (x << 31)) & mask;
    return x;
}

EXPORT int64_t bio_minimizers(const uint8_t* codes, int64_t n, int32_t k, int32_t w,
                              uint64_t* out_hash, int64_t* out_pos,
                              uint8_t* out_strand) {
    int64_t nk = n - (int64_t)k + 1;
    if (nk <= 0) return 0;

    uint64_t* h  = (uint64_t*)malloc((size_t)nk * sizeof(uint64_t));
    uint8_t*  st = (uint8_t*) malloc((size_t)nk * sizeof(uint8_t));
    if (!h || !st) { free(h); free(st); return -1; }

    const uint64_t SENT  = ~0ULL;
    uint64_t mask  = (k >= 32) ? ~0ULL : ((1ULL << (2 * k)) - 1);
    uint64_t shift = (uint64_t)(2 * (k - 1));
    uint64_t fwd = 0, rev = 0;
    int32_t  l = 0;                          /* bases válidas consecutivas */

    for (int64_t i = 0; i < n; i++) {
        uint8_t c = codes[i];
        if (c < 4) {
            uint64_t cc = c;
            fwd = ((fwd << 2) | cc) & mask;
            rev = (rev >> 2) | ((3ULL - cc) << shift);
            l++;
        } else {
            l = 0; fwd = 0; rev = 0;
        }
        int64_t p = i - (int64_t)k + 1;       /* k-mer que termina en i */
        if (p >= 0) {
            if (l >= k) {
                uint64_t hf = _hash64_c(fwd, mask);
                uint64_t hr = _hash64_c(rev, mask);
                if (hr < hf) { h[p] = hr; st[p] = 1; }
                else         { h[p] = hf; st[p] = 0; }
            } else {
                h[p] = SENT; st[p] = 0;
            }
        }
    }

    int64_t count = 0, last = -1;
    if (nk < w) {                             /* una sola ventana */
        int64_t bi = 0;
        for (int64_t j = 1; j < nk; j++) if (h[j] < h[bi]) bi = j;
        if (h[bi] != SENT) {
            out_hash[0] = h[bi]; out_pos[0] = bi; out_strand[0] = st[bi];
            count = 1;
        }
    } else {
        for (int64_t i = 0; i + w <= nk; i++) {
            int64_t bi = i;                   /* argmin más a la izquierda */
            for (int64_t j = i + 1; j < i + w; j++) if (h[j] < h[bi]) bi = j;
            if (bi != last && h[bi] != SENT) {
                out_hash[count] = h[bi]; out_pos[count] = bi;
                out_strand[count] = st[bi];
                count++; last = bi;
            }
        }
    }
    free(h); free(st);
    return count;
}

/* ════════════════════════════════════════════════════════════════════════════
   ÍNDICE OPACO DE LA REFERENCIA  (mapeador v5 — pipeline entero en C)
   ─────────────────────────────────────────────────────────────────────────
   Fase C1. Construye UNA sola vez, en C, la estructura que el mapeo consulta
   miles de veces: tabla de minimizers ordenada por hash (para búsqueda binaria),
   copia de los codes de la referencia (para la extensión banded) y las fronteras
   de los contigs. El handle es opaco: Python solo guarda el void* y lo pasa a
   bio_map_read / bio_map_batch (fases siguientes). Así el índice "vive" en C y
   no se re-serializa por consulta — el cuello de botella del pipeline en Python.

   Réplica exacta de ReferenceIndex: mismos minimizers (bio_minimizers), mismo
   filtro max_occ (descarta hashes con más de max_occ apariciones). El orden
   entre hashes iguales no afecta al resultado (el lookup reúne TODAS las
   coincidencias de cada hash; el chaining reordena por (x,y) después).
   ════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    uint64_t h;
    int64_t  pos;
    uint8_t  strand;
} MinEntry;

static int _cmp_min(const void* a, const void* b) {
    uint64_t ha = ((const MinEntry*)a)->h;
    uint64_t hb = ((const MinEntry*)b)->h;
    return (ha < hb) ? -1 : (ha > hb) ? 1 : 0;
}

typedef struct {
    int32_t   k, w;
    int64_t   ref_len;
    int64_t   n_min;          /* nº de minimizers tras filtrar max_occ */
    uint64_t* hashes;         /* ordenados ascendentemente */
    int64_t*  positions;
    uint8_t*  strands;
    uint8_t*  ref_codes;      /* copia de los codes 2-bit (>=4 = N); para extensión */
    int32_t   n_contigs;
    int64_t*  ctg_starts;     /* inicio global de cada contig */
    int64_t*  ctg_lengths;
} BioIndex;

EXPORT void bio_index_free(void* handle);   /* fwd: bio_index_build lo usa al fallar */

/* Construye el índice. Devuelve un handle opaco (void*), o NULL si falla malloc.
   ref_codes: codes 2-bit de la referencia concatenada (n bases).
   max_occ<=0 → sin filtro; en otro caso descarta hashes con count > max_occ. */
EXPORT void* bio_index_build(const uint8_t* ref_codes, int64_t n,
                             int32_t k, int32_t w, int32_t max_occ,
                             const int64_t* ctg_starts, const int64_t* ctg_lengths,
                             int32_t n_contigs) {
    BioIndex* ix = (BioIndex*)calloc(1, sizeof(BioIndex));
    if (!ix) return NULL;
    ix->k = k; ix->w = w; ix->ref_len = n;

    int64_t nk = n - (int64_t)k + 1;
    if (nk < 0) nk = 0;
    size_t alloc_nk = (size_t)(nk ? nk : 1);

    uint64_t* rh = (uint64_t*)malloc(alloc_nk * sizeof(uint64_t));
    int64_t*  rp = (int64_t*) malloc(alloc_nk * sizeof(int64_t));
    uint8_t*  rs = (uint8_t*) malloc(alloc_nk);
    if (!rh || !rp || !rs) { free(rh); free(rp); free(rs); free(ix); return NULL; }

    int64_t m = bio_minimizers(ref_codes, n, k, w, rh, rp, rs);
    if (m < 0) { free(rh); free(rp); free(rs); free(ix); return NULL; }

    MinEntry* e = (MinEntry*)malloc((size_t)(m ? m : 1) * sizeof(MinEntry));
    if (!e) { free(rh); free(rp); free(rs); free(ix); return NULL; }
    for (int64_t i = 0; i < m; i++) {
        e[i].h = rh[i]; e[i].pos = rp[i]; e[i].strand = rs[i];
    }
    free(rh); free(rp); free(rs);
    qsort(e, (size_t)m, sizeof(MinEntry), _cmp_min);

    /* Contar cuántos sobreviven al filtro max_occ (runs de hash igual). */
    int64_t kept = 0;
    for (int64_t i = 0; i < m; ) {
        int64_t j = i + 1;
        while (j < m && e[j].h == e[i].h) j++;
        int64_t cnt = j - i;
        if (max_occ <= 0 || cnt <= (int64_t)max_occ) kept += cnt;
        i = j;
    }

    ix->hashes    = (uint64_t*)malloc((size_t)(kept ? kept : 1) * sizeof(uint64_t));
    ix->positions = (int64_t*) malloc((size_t)(kept ? kept : 1) * sizeof(int64_t));
    ix->strands   = (uint8_t*) malloc((size_t)(kept ? kept : 1));
    if (!ix->hashes || !ix->positions || !ix->strands) {
        free(e); bio_index_free(ix); return NULL;
    }
    int64_t o = 0;
    for (int64_t i = 0; i < m; ) {
        int64_t j = i + 1;
        while (j < m && e[j].h == e[i].h) j++;
        int64_t cnt = j - i;
        if (max_occ <= 0 || cnt <= (int64_t)max_occ) {
            for (int64_t t = i; t < j; t++) {
                ix->hashes[o] = e[t].h; ix->positions[o] = e[t].pos;
                ix->strands[o] = e[t].strand; o++;
            }
        }
        i = j;
    }
    ix->n_min = kept;
    free(e);

    /* Copia de los codes (C es dueño; liberados en bio_index_free) + contigs. */
    ix->ref_codes = (uint8_t*)malloc((size_t)(n ? n : 1));
    if (!ix->ref_codes) { bio_index_free(ix); return NULL; }
    if (n > 0) memcpy(ix->ref_codes, ref_codes, (size_t)n);

    ix->n_contigs   = n_contigs;
    ix->ctg_starts  = (int64_t*)malloc((size_t)(n_contigs ? n_contigs : 1) * sizeof(int64_t));
    ix->ctg_lengths = (int64_t*)malloc((size_t)(n_contigs ? n_contigs : 1) * sizeof(int64_t));
    if (!ix->ctg_starts || !ix->ctg_lengths) { bio_index_free(ix); return NULL; }
    if (n_contigs > 0) {
        memcpy(ix->ctg_starts,  ctg_starts,  (size_t)n_contigs * sizeof(int64_t));
        memcpy(ix->ctg_lengths, ctg_lengths, (size_t)n_contigs * sizeof(int64_t));
    }
    return ix;
}

EXPORT void bio_index_free(void* handle) {
    if (!handle) return;
    BioIndex* ix = (BioIndex*)handle;
    free(ix->hashes); free(ix->positions); free(ix->strands);
    free(ix->ref_codes); free(ix->ctg_starts); free(ix->ctg_lengths);
    free(ix);
}

/* nº de minimizers del índice (para verificar paridad con ReferenceIndex). */
EXPORT int64_t bio_index_n_minimizers(void* handle) {
    return handle ? ((BioIndex*)handle)->n_min : -1;
}

/* ════════════════════════════════════════════════════════════════════════════
   MAP_READ — pipeline entero de mapeo en C  (fases C2 + C3)
   ─────────────────────────────────────────────────────────────────────────
   Réplica fiel de GenomeAligner.map (genomemap.py): seeding → chaining (DP +
   backtrack + supresión) → extensión banded del read completo → Mapping.
   Todo en una sola llamada C, sin marshalling por consulta: la vía a competir
   en velocidad. Parámetros y desempates calcados del pipeline Python para que
   los resultados sean idénticos (verificado con parity tests).

   Constantes del mapeador (idénticas a genomemap.py):
     MAX_GAP=5000, WINDOW=64, GAP_COEF=0.01, MIN_ANCHORS=2.
   Puntuación de la extensión (idéntica a SequenceAligner): +2/-1/-2, global.
   ════════════════════════════════════════════════════════════════════════════ */

#define MAP_MAX_GAP     5000
#define MAP_WINDOW      64
#define MAP_GAP_COEF    0.01
#define MAP_MIN_ANCHORS 2
#define MAP_MATCH        2
#define MAP_MISMATCH   (-1)
#define MAP_GAP        (-2)

/* struct de salida (layout replicado en ctypes; doubles/int64 primero). */
typedef struct {
    double  identity;
    double  chain_score;
    int64_t target_start;     /* coords GLOBALES; Python las localiza al contig */
    int64_t target_end;
    int32_t query_start, query_end;
    int32_t strand;           /* 0=+  1=- */
    int32_t num_matches, block_len, mapq;
    int32_t contig;           /* índice de contig */
    int32_t cigar_off, cigar_len;
} MapOut;

typedef struct { int64_t x, y, rpos, qpos; } CAnc;
typedef struct { double f; int32_t i; } FIdx;
typedef struct {
    double  score;
    int32_t strand, n_anchors;
    int64_t ref_start, ref_end, read_start, read_end, d;
} CChain;

static int _cmp_anc(const void* a, const void* b) {
    const CAnc* pa = (const CAnc*)a; const CAnc* pb = (const CAnc*)b;
    if (pa->x != pb->x) return pa->x < pb->x ? -1 : 1;
    if (pa->y != pb->y) return pa->y < pb->y ? -1 : 1;
    return 0;
}
static int _cmp_fidx(const void* a, const void* b) {
    const FIdx* pa = (const FIdx*)a; const FIdx* pb = (const FIdx*)b;
    if (pa->f != pb->f) return pa->f > pb->f ? -1 : 1;    /* score desc */
    return pa->i < pb->i ? -1 : (pa->i > pb->i ? 1 : 0);  /* desempate: idx asc */
}
static int _cmp_cchain(const void* a, const void* b) {
    const CChain* pa = (const CChain*)a; const CChain* pb = (const CChain*)b;
    if (pa->score != pb->score) return pa->score > pb->score ? -1 : 1;
    if (pa->ref_start != pb->ref_start) return pa->ref_start < pb->ref_start ? -1 : 1;
    return 0;
}

static int64_t _lower_bound(const uint64_t* a, int64_t n, uint64_t key) {
    int64_t lo = 0, hi = n;
    while (lo < hi) { int64_t mid = (lo + hi) >> 1;
        if (a[mid] < key) lo = mid + 1; else hi = mid; }
    return lo;
}
static int64_t _upper_bound(const uint64_t* a, int64_t n, uint64_t key) {
    int64_t lo = 0, hi = n;
    while (lo < hi) { int64_t mid = (lo + hi) >> 1;
        if (a[mid] <= key) lo = mid + 1; else hi = mid; }
    return lo;
}

static int _cmp_i64(const void* a, const void* b) {
    int64_t x = *(const int64_t*)a, y = *(const int64_t*)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}
/* mediana como np.median + int(): media de los dos centrales si n par,
   truncada hacia cero (== cast a int64). tmp se ordena in situ. */
static int64_t _median_trunc(int64_t* tmp, int32_t n) {
    qsort(tmp, (size_t)n, sizeof(int64_t), _cmp_i64);
    if (n & 1) return tmp[n / 2];
    double m = ((double)tmp[n / 2 - 1] + (double)tmp[n / 2]) / 2.0;
    return (int64_t)m;                       /* trunca hacia cero, como int() */
}

/* Chaining de UNA hebra: rellena chains[] (desde *nc) y devuelve nº añadido.
   rpos/qpos: anclas de la hebra (sin ordenar). Réplica de _chain_one+chain. */
static int _chain_strand(const int64_t* rpos, const int64_t* qpos, int32_t na,
                         int32_t strand, int32_t Lr, int32_t k, double min_score,
                         CChain* chains, int32_t* nc) {
    if (na == 0) return 0;
    CAnc* anc = (CAnc*)malloc((size_t)na * sizeof(CAnc));
    int64_t* xs = (int64_t*)malloc((size_t)na * sizeof(int64_t));
    int64_t* ys = (int64_t*)malloc((size_t)na * sizeof(int64_t));
    double*  f  = (double*) malloc((size_t)na * sizeof(double));
    int32_t* pv = (int32_t*)malloc((size_t)na * sizeof(int32_t));
    FIdx*    fi = (FIdx*)   malloc((size_t)na * sizeof(FIdx));
    uint8_t* used = (uint8_t*)calloc((size_t)na, 1);
    int32_t* path = (int32_t*)malloc((size_t)na * sizeof(int32_t));
    int64_t* med  = (int64_t*)malloc((size_t)na * sizeof(int64_t));
    if (!anc || !xs || !ys || !f || !pv || !fi || !used || !path || !med) {
        free(anc); free(xs); free(ys); free(f); free(pv);
        free(fi); free(used); free(path); free(med); return -1;
    }
    for (int32_t i = 0; i < na; i++) {
        anc[i].rpos = rpos[i]; anc[i].qpos = qpos[i];
        anc[i].x = rpos[i];
        anc[i].y = (strand == 0) ? qpos[i] : (int64_t)(Lr - k) - qpos[i];
    }
    qsort(anc, (size_t)na, sizeof(CAnc), _cmp_anc);   /* lexsort((y,x)) */
    for (int32_t i = 0; i < na; i++) { xs[i] = anc[i].x; ys[i] = anc[i].y; }

    double gap_w = MAP_GAP_COEF * (double)k;
    bio_chain_dp(xs, ys, na, k, MAP_MAX_GAP, MAP_WINDOW, gap_w, f, pv);

    for (int32_t i = 0; i < na; i++) { fi[i].f = f[i]; fi[i].i = i; }
    qsort(fi, (size_t)na, sizeof(FIdx), _cmp_fidx);   /* argsort(-f) */

    int added = 0;
    for (int32_t s = 0; s < na; s++) {
        if (fi[s].f < min_score) break;               /* f desc → resto también */
        int32_t start = fi[s].i;
        if (used[start]) continue;
        int32_t plen = 0, i = start;
        while (i != -1 && !used[i]) { used[i] = 1; path[plen++] = i; i = pv[i]; }
        double frag = fi[s].f - ((i != -1) ? f[i] : 0.0);
        if (plen < MAP_MIN_ANCHORS || frag < min_score) continue;

        /* spans + diagonal (median), sobre las anclas del camino */
        int64_t rmin = anc[path[0]].rpos, rmax = rmin;
        int64_t qmin = anc[path[0]].qpos, qmax = qmin;
        for (int32_t t = 0; t < plen; t++) {
            int64_t rp = anc[path[t]].rpos, qp = anc[path[t]].qpos;
            if (rp < rmin) rmin = rp; if (rp > rmax) rmax = rp;
            if (qp < qmin) qmin = qp; if (qp > qmax) qmax = qp;
            med[t] = (strand == 0) ? (rp - qp) : (rp + qp);
        }
        int64_t d = _median_trunc(med, plen);
        if (strand != 0) d -= (int64_t)(Lr - k);

        CChain* c = &chains[*nc];
        c->score = frag; c->strand = strand; c->n_anchors = plen;
        c->ref_start = rmin; c->ref_end = rmax + k;
        c->read_start = qmin; c->read_end = qmax + k;
        c->d = d;
        (*nc)++; added++;
    }
    free(anc); free(xs); free(ys); free(f); free(pv);
    free(fi); free(used); free(path); free(med);
    return added;
}

/* Nucleótido decode para la extensión (0..3=ACGT, 4+=N). */
static const char _MAP_DECODE[32] =
    "ACGTNNNNNNNNNNNNNNNNNNNNNNNNNNNN";

static int32_t _map_one(BioIndex* ix, const uint8_t* read_codes, int32_t Lr,
                        int32_t max_hits, double min_score,
                        MapOut* out, int32_t max_out,
                        char* cigar_buf, int32_t cigar_cap) {
    if (!ix || Lr < ix->k) return 0;
    int32_t k = ix->k, w = ix->w;

    /* ── SEEDING: minimizers del read → lookup → anclas ──────────────────── */
    int64_t nk = (int64_t)Lr - k + 1;
    uint64_t* qh = (uint64_t*)malloc((size_t)nk * sizeof(uint64_t));
    int64_t*  qp = (int64_t*) malloc((size_t)nk * sizeof(int64_t));
    uint8_t*  qs = (uint8_t*) malloc((size_t)nk);
    if (!qh || !qp || !qs) { free(qh); free(qp); free(qs); return -1; }
    int64_t nm = bio_minimizers(read_codes, Lr, k, w, qh, qp, qs);
    if (nm < 0) { free(qh); free(qp); free(qs); return -1; }

    /* nº total de coincidencias (para dimensionar los arrays de anclas) */
    int64_t ntot = 0;
    for (int64_t j = 0; j < nm; j++) {
        int64_t lo = _lower_bound(ix->hashes, ix->n_min, qh[j]);
        int64_t hi = _upper_bound(ix->hashes, ix->n_min, qh[j]);
        ntot += hi - lo;
    }
    /* anclas separadas por hebra relativa */
    int64_t* r0 = (int64_t*)malloc((size_t)(ntot ? ntot : 1) * sizeof(int64_t));
    int64_t* q0 = (int64_t*)malloc((size_t)(ntot ? ntot : 1) * sizeof(int64_t));
    int64_t* r1 = (int64_t*)malloc((size_t)(ntot ? ntot : 1) * sizeof(int64_t));
    int64_t* q1 = (int64_t*)malloc((size_t)(ntot ? ntot : 1) * sizeof(int64_t));
    if (!r0 || !q0 || !r1 || !q1) {
        free(qh); free(qp); free(qs); free(r0); free(q0); free(r1); free(q1);
        return -1;
    }
    int64_t n0 = 0, n1 = 0;
    for (int64_t j = 0; j < nm; j++) {
        int64_t lo = _lower_bound(ix->hashes, ix->n_min, qh[j]);
        int64_t hi = _upper_bound(ix->hashes, ix->n_min, qh[j]);
        for (int64_t t = lo; t < hi; t++) {
            uint8_t rel = (uint8_t)(qs[j] ^ ix->strands[t]);
            if (rel == 0) { r0[n0] = ix->positions[t]; q0[n0] = qp[j]; n0++; }
            else          { r1[n1] = ix->positions[t]; q1[n1] = qp[j]; n1++; }
        }
    }
    free(qh); free(qp); free(qs);

    /* ── CHAINING (ambas hebras) ─────────────────────────────────────────── */
    int64_t cap = n0 + n1 + 1;
    CChain* chains = (CChain*)malloc((size_t)cap * sizeof(CChain));
    if (!chains) { free(r0); free(q0); free(r1); free(q1); return -1; }
    int32_t nc = 0;
    int a0 = _chain_strand(r0, q0, (int32_t)n0, 0, Lr, k, min_score, chains, &nc);
    int a1 = _chain_strand(r1, q1, (int32_t)n1, 1, Lr, k, min_score, chains, &nc);
    free(r0); free(q0); free(r1); free(q1);
    if (a0 < 0 || a1 < 0) { free(chains); return -1; }

    qsort(chains, (size_t)nc, sizeof(CChain), _cmp_cchain);   /* score desc */

    /* supresión de solapamientos (>50% en ref, misma hebra) */
    CChain* kept = (CChain*)malloc((size_t)(nc ? nc : 1) * sizeof(CChain));
    if (!kept) { free(chains); return -1; }
    int32_t nkept = 0;
    for (int32_t c = 0; c < nc; c++) {
        int redundant = 0;
        for (int32_t kk = 0; kk < nkept; kk++) {
            if (chains[c].strand != kept[kk].strand) continue;
            int64_t inter = (chains[c].ref_end < kept[kk].ref_end ? chains[c].ref_end : kept[kk].ref_end)
                          - (chains[c].ref_start > kept[kk].ref_start ? chains[c].ref_start : kept[kk].ref_start);
            int64_t la = chains[c].ref_end - chains[c].ref_start;
            int64_t lb = kept[kk].ref_end - kept[kk].ref_start;
            int64_t shorter = la < lb ? la : lb;
            if (shorter > 0 && (double)inter > 0.5 * (double)shorter) { redundant = 1; break; }
        }
        if (!redundant) kept[nkept++] = chains[c];
    }
    free(chains);
    if (nkept > max_hits) nkept = max_hits;

    /* ── EXTENSIÓN del read completo (banded NW) por cadena ──────────────── */
    uint8_t* oriented = (uint8_t*)malloc((size_t)Lr);      /* read orientado */
    char* aa = (char*)malloc((size_t)(2 * Lr + 4 * MAP_WINDOW + 1024));
    char* bb = (char*)malloc((size_t)(2 * Lr + 4 * MAP_WINDOW + 1024));
    if (!oriented || !aa || !bb) { free(kept); free(oriented); free(aa); free(bb); return -1; }

    int32_t nout = 0, cig_used = 0;
    for (int32_t c = 0; c < nkept && nout < max_out; c++) {
        CChain* ch = &kept[c];
        /* contig de la cadena → límites de la ventana */
        int64_t ci = _upper_bound((const uint64_t*)ix->ctg_starts, ix->n_contigs,
                                  (uint64_t)ch->ref_start) - 1;
        if (ci < 0) ci = 0;
        int64_t cstart = ix->ctg_starts[ci];
        int64_t cend   = cstart + ix->ctg_lengths[ci];

        int64_t d = ch->d;
        int64_t rs = d > cstart ? d : cstart;
        int64_t re = (d + Lr < cend) ? d + Lr : cend;
        if (re - rs < 1) continue;

        /* read orientado (revcomp si hebra inversa) */
        if (ch->strand == 0) {
            for (int32_t i = 0; i < Lr; i++) oriented[i] = read_codes[i];
        } else {
            for (int32_t i = 0; i < Lr; i++) {
                uint8_t cc = read_codes[Lr - 1 - i];
                oriented[i] = (cc < 4) ? (uint8_t)(3 - cc) : 4;
            }
        }
        int32_t m_len = (int32_t)(re - rs);
        int32_t n_len = (int32_t)(re - rs);         /* read_sub = oriented[rs-d : re-d] */
        const uint8_t* ref_sub  = ix->ref_codes + rs;
        const uint8_t* read_sub = oriented + (rs - d);

        int32_t diff = m_len - n_len; if (diff < 0) diff = -diff;
        int32_t band = diff + 64; if (band > 512) band = 512;

        int32_t score, mt, mm, gp;
        /* Extensión: kernel banded SIMD (AVX2, 6× sobre el escalar); bit-idéntico
           al core (verificado) y con fallback escalar si no hay AVX2. */
        int32_t pos = _nw_banded_diag_simd(ref_sub, m_len, read_sub, n_len, _MAP_DECODE,
                                           MAP_MATCH, MAP_MISMATCH, MAP_GAP, band,
                                           aa, bb, &score, &mt, &mm, &gp);
        if (pos < 0) continue;                      /* extensión fallida → sin mapeo */
        int32_t block = mt + mm + gp;
        double identity = block ? (double)mt / (double)block : 0.0;

        /* CIGAR run-length (M / I=hueco en ref / D=hueco en read) */
        int32_t coff = cig_used;
        int32_t t = 0;
        while (t < pos) {
            char op = (aa[t] == '-') ? 'I' : ((bb[t] == '-') ? 'D' : 'M');
            int32_t run = 1;
            while (t + run < pos) {
                char o2 = (aa[t+run] == '-') ? 'I' : ((bb[t+run] == '-') ? 'D' : 'M');
                if (o2 != op) break;
                run++;
            }
            if (cig_used > cigar_cap - 16) { cig_used = coff; break; }  /* sin sitio */
            cig_used += sprintf(cigar_buf + cig_used, "%d%c", run, op);
            t += run;
        }

        /* coords de query en el read ORIGINAL (hacia delante) */
        int32_t q_start, q_end;
        if (ch->strand == 0) { q_start = (int32_t)(rs - d); q_end = (int32_t)(re - d); }
        else { q_start = (int32_t)(Lr - (re - d)); q_end = (int32_t)(Lr - (rs - d)); }

        /* mapq (idéntico a _mapq): usa el ranking en kept[:max_hits] */
        int32_t mapq;
        if (c > 0) mapq = 5;
        else if (nkept < 2) mapq = 60;
        else {
            double ratio = kept[0].score ? kept[1].score / kept[0].score : 0.0;
            double v = rint(60.0 * (1.0 - ratio));
            if (v < 0) v = 0; if (v > 60) v = 60;
            mapq = (int32_t)v;
        }

        MapOut* o = &out[nout++];
        o->identity = identity; o->chain_score = ch->score;
        o->target_start = rs; o->target_end = re;
        o->query_start = q_start; o->query_end = q_end;
        o->strand = ch->strand;
        o->num_matches = mt; o->block_len = block; o->mapq = mapq;
        o->contig = (int32_t)ci;
        o->cigar_off = coff; o->cigar_len = cig_used - coff;
    }
    free(kept); free(oriented); free(aa); free(bb);
    return nout;
}

/* Mapea UN read (cubierta delgada sobre _map_one). */
EXPORT int32_t bio_map_read(void* handle, const uint8_t* read_codes, int32_t Lr,
                            int32_t max_hits, double min_score,
                            MapOut* out, int32_t max_out,
                            char* cigar_buf, int32_t cigar_cap) {
    BioIndex* ix = (BioIndex*)handle;
    if (!ix) return 0;
    return _map_one(ix, read_codes, Lr, max_hits, min_score,
                    out, max_out, cigar_buf, cigar_cap);
}

/* ── MAP_BATCH — mapea muchos reads en paralelo (OpenMP, sin GIL) ─────────────
   reads_concat : codes 2-bit de todos los reads concatenados.
   read_off     : n_reads+1 offsets → read i = reads_concat[read_off[i]..[i+1]).
   out          : n_reads*max_hits MapOut (read i escribe en out[i*max_hits..]).
   counts       : n_reads → nº de mapeos de cada read.
   cigar_buf    : buffer compartido; cig_off[i] delimita la región del read i.
                  Los cigar_off de salida quedan ABSOLUTOS (índices en cigar_buf).
   Cada read es independiente → paralelismo perfecto. Devuelve 0, o <0 si error. */
EXPORT int32_t bio_map_batch(void* handle, const uint8_t* reads_concat,
                             const int64_t* read_off, int32_t n_reads,
                             int32_t max_hits, double min_score,
                             MapOut* out, int32_t* counts,
                             char* cigar_buf, const int64_t* cig_off,
                             int32_t n_threads) {
    BioIndex* ix = (BioIndex*)handle;
    if (!ix) return -1;
    int32_t err = 0;
#ifdef _OPENMP
    /* n_threads>0 → ese nº; <=0 → todos los núcleos. Se fija SIEMPRE para no
       heredar un contador reducido de una llamada previa (bug de escalado). */
    omp_set_num_threads(n_threads > 0 ? n_threads : omp_get_num_procs());
    #pragma omp parallel for schedule(dynamic, 16)
#endif
    for (int32_t i = 0; i < n_reads; i++) {
        int32_t Lr = (int32_t)(read_off[i+1] - read_off[i]);
        int32_t cap = (int32_t)(cig_off[i+1] - cig_off[i]);
        MapOut* oi = out + (int64_t)i * max_hits;
        int32_t cnt = _map_one(ix, reads_concat + read_off[i], Lr, max_hits,
                               min_score, oi, max_hits,
                               cigar_buf + cig_off[i], cap);
        if (cnt < 0) { err = 1; counts[i] = 0; continue; }
        for (int32_t j = 0; j < cnt; j++) oi[j].cigar_off += (int32_t)cig_off[i];
        counts[i] = cnt;
    }
    return err ? -1 : 0;
}
