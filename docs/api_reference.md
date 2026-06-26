# BioForge — API Reference

## biocore.py

### SmartImporter

```python
from bioforge import SmartImporter, SeqType

# Desde string FASTA
records = SmartImporter.from_string(fasta_text)
records = SmartImporter.from_string(fasta_text, force_type=SeqType.PROTEIN)

# Desde archivo (carga todo en RAM)
records = SmartImporter.from_file("secuencia.fa")

# Desde archivo grande (modo streaming, un registro a la vez)
for rec in SmartImporter.from_file_chunked("genoma.fa"):
    procesar(rec)
```

**Nota:** La auto-detección de tipo busca E/F/I/L/P/Q/* para identificar
proteínas. Si tu proteína no contiene ninguno de esos residuos, usa
`force_type=SeqType.PROTEIN`.

---

### PackedSequence

```python
seq = records[0]

# Propiedades
seq.header          # str — cabecera FASTA sin '>'
seq.seq_type        # SeqType.NUCLEOTIDE o SeqType.PROTEIN
seq.n_symbols       # int — número de bases o aminoácidos
seq.packed_bytes    # int — bytes ocupados en RAM
seq.memory_ratio    # float — ratio vs ASCII (ideal: 0.625 = 37.5% de ahorro)

# Acceso a datos
seq.decode()        # → np.ndarray uint8, valores BioCode [0–31]
seq.to_string()     # → str IUPAC (solo para display/export FASTA)

# Indexación
seq[10]             # → int BioCode en posición 10 (O(1), sin desempaquetar todo)
seq[5:20]           # → nuevo PackedSequence con el sub-rango

# Igualdad
seq_a == seq_b      # compara tipo, longitud, cabecera y datos

# Reverse complement (v1.1) — solo para SeqType.NUCLEOTIDE
rc = seq.reverse_complement()
# Aplica emparejamiento Watson-Crick (A↔T, C↔G) e invierte.
# El header del resultado lleva prefijo "[RC]".
# Lanza SequenceTypeError si seq es proteína.
rc2 = rc.reverse_complement()  # RC(RC(x)) == x
```

---

### BitPacker

```python
from bioforge import BitPacker
import numpy as np

codes  = np.array([0, 1, 2, 3], dtype=np.uint8)  # A C G T
packed = BitPacker.pack(codes)                      # → bytes compactos
back   = BitPacker.unpack(packed, len(codes))       # → restaurar

BitPacker.packed_size(1_000_000)   # → 625000 bytes para 1M símbolos
```

---

### BioCode

```python
from bioforge import BioCode

# Nucleótidos
BioCode.NUC_A   # 0
BioCode.NUC_C   # 1
BioCode.NUC_G   # 2
BioCode.NUC_TU  # 3  (T y U comparten slot)

# Aminoácidos (selección)
BioCode.AA_M    # 14  Metionina = codón de inicio ATG
BioCode.AA_A    # 4   Alanina
BioCode.AA_G    # 9   Glicina

# Especiales
BioCode.STOP    # 24  codón de parada
BioCode.GAP     # 25  guión de alineamiento
BioCode.UNK     # 31  desconocido / ambiguo (N en ADN, X en proteína)
```

---

### compute_stats

```python
from bioforge import compute_stats

stats = compute_stats(seq)
stats.n_symbols         # longitud de la secuencia
stats.n_packed_bytes    # bytes empaquetados
stats.compression_pct   # porcentaje de ahorro vs ASCII (≈ 37.5 %)
stats.composition       # dict: {'A': 1234, 'C': 987, 'G': 1100, 'T': 1050}
```

---

## smart_translator.py

### SmartTranslator

```python
from bioforge import SmartTranslator

# Entrada: PackedSequence de tipo NUCLEOTIDE
# Salida:  PackedSequence de tipo PROTEIN (desde el primer ATG hasta STOP)

protein = SmartTranslator.translate(nuc_seq)
protein = SmartTranslator.translate(nuc_seq, warn_short=False)  # suprimir aviso < 50 aa

protein.to_string()   # secuencia de aminoácidos como str
protein.header        # "[PROT | ORF@<pos>] <header_original>"

# Errores posibles:
# SequenceTypeError  — si la entrada no es SeqType.NUCLEOTIDE
# TranslationError   — si la secuencia es < 3 nt, o no hay ATG, o no hay codón completo tras ATG
```

---

### SmartTranslator.translate_all_frames  *(v1.1)*

Traduce los 6 marcos de lectura (3 directos + 3 en reverse complement).
Devuelve una lista con una `PackedSequence(PROTEIN)` por cada marco que
contenga al menos un ATG. Los marcos sin ATG se omiten silenciosamente.

```python
from bioforge import SmartTranslator, SmartImporter, SeqType

seq = SmartImporter.from_string(">gen\nATGAAAGGGTAA\n",
                                force_type=SeqType.NUCLEOTIDE)[0]

proteinas = SmartTranslator.translate_all_frames(seq)
# proteinas es una list[PackedSequence]

for prot in proteinas:
    print(prot.header)      # "[PROT | frame +1 | ORF@0] gen"
    print(prot.to_string()) # "MKG"

# El header indica:
#   • strand: + (directo) o - (reverse complement)
#   • offset: 1, 2 o 3
#   • ORF@N: posición 0-based del ATG en la secuencia original

# Suprimir aviso de proteína corta (< 50 aa):
proteinas = SmartTranslator.translate_all_frames(seq, warn_short=False)

# Errores posibles:
# SequenceTypeError — si la entrada no es SeqType.NUCLEOTIDE
```

---

## aligner.py

### SequenceAligner

```python
from bioforge import SequenceAligner, format_alignment

# Ambas secuencias deben ser del mismo SeqType
# seq_a = referencia, seq_b = query (las mutaciones se reportan respecto a seq_a)

result = SequenceAligner.align(seq_a, seq_b)                     # global (recomendado para mutaciones)
result = SequenceAligner.align(seq_a, seq_b, mode='semi-global') # libre en extremos del query

# Banded NW (v1.1) — para secuencias largas con desviación acotada del diagonal
result = SequenceAligner.align(seq_a, seq_b, band=50)            # solo rellena ±50 celdas del diagonal
# band=N reduce memoria a O(m·N) y tiempo a O(m·N).
# Requiere que las secuencias no difieran en más de N inserciones/deleciones.
# Con el motor C: usa almacenamiento verdaderamente compacto.
# Sin motor C: usa la matriz completa con NEG_INF fuera de la banda (mismo resultado, más RAM).

# Métricas
result.score           # int   — puntuación NW
result.identity        # float — posiciones coincidentes / longitud alineada (0.0–1.0)
result.n_matches       # int   — posiciones idénticas
result.n_mismatches    # int   — sustituciones
result.n_gaps          # int   — caracteres gap totales en la región alineada

# Strings alineados (con '-' en posiciones de gap)
result.aligned_a       # str
result.aligned_b       # str

# Lista de mutaciones (en orden de posición)
for mut in result.mutations:
    print(mut)
    # Salida posible:
    # "SUB  a[19]='A' → b[19]='T'"
    # "DEL  a[5]='G'  (deleción en seq_b, tras b[5])"
    # "INS  b[12]='C' (inserción en seq_b, gap en seq_a[12])"

# Acceso directo a una mutación
mut.kind   # 'substitution' | 'deletion' | 'insertion'
mut.pos_a  # posición 0-based en seq_a
mut.pos_b  # posición 0-based en seq_b
mut.sym_a  # símbolo en seq_a ('-' para inserciones)
mut.sym_b  # símbolo en seq_b ('-' para deleciones)

# Visualización del alineamiento
print(format_alignment(result))           # bloques de 60 chars
print(format_alignment(result, width=80)) # bloques de 80 chars
# Formato:
#   A: ATGGTGCACCTGACTCCTGAGGAGAAGTCT
#      |||||||||||||||||||X||||||||||
#   B: ATGGTGCACCTGACTCCTGTGGAGAAGTCT
```

### SequenceAligner.align_local  *(v1.1)*

Alineamiento local Smith-Waterman. Encuentra la sub-región de mayor similitud
entre dos secuencias, ignorando flancos no homólogos.

```python
from bioforge import SequenceAligner

# Entrada: dos PackedSequence del mismo SeqType
result = SequenceAligner.align_local(seq_a, seq_b)

result.mode        # 'local'
result.score       # int — puntuación SW (siempre ≥ 0)
result.identity    # float — identidad en la región local alineada
result.aligned_a   # str — región alineada de seq_a (con gaps)
result.aligned_b   # str — región alineada de seq_b (con gaps)
result.mutations   # list[Mutation] — diferencias dentro de la región local

# Cuándo usar SW vs NW:
#   NW (align)       → comparar alelos o CDS completos (mutación puntual, indels)
#   SW (align_local) → encontrar un dominio/motivo dentro de una secuencia larga
#                      o comparar dos secuencias con regiones no homólogas en los extremos
```

### Errores posibles

```python
# SequenceTypeError — los dos tipos de secuencia no coinciden
SequenceAligner.align(prot_seq, nuc_seq)    # → SequenceTypeError
SequenceAligner.align_local(prot_seq, nuc_seq)  # → SequenceTypeError

# UserWarning — secuencias largas (> 15 000 símbolos) sin band=
# La matriz DP supera ~3.4 GB. Usar band=N para secuencias grandes.

# AlignmentError — band= con valor ≤ 0
SequenceAligner.align(seq_a, seq_b, band=0)  # → AlignmentError
```

---

## Flujo completo de ejemplo

```python
from bioforge import SmartImporter, SmartTranslator, SequenceAligner, format_alignment, SeqType

# 1. Importar dos alelos
fasta = """
>alelo_normal
ATGGTGCACCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCC
>alelo_mutante
ATGGTGCACCTGACTCCTGTGGAGAAGTCTGCCGTTACTGCC
"""
alelos = SmartImporter.from_string(fasta, force_type=SeqType.NUCLEOTIDE)
normal, mutante = alelos[0], alelos[1]

# 2. Traducir ambos
prot_normal  = SmartTranslator.translate(normal)
prot_mutante = SmartTranslator.translate(mutante)

# 3. Comparar a nivel de ADN
result_nuc = SequenceAligner.align(normal, mutante)
print(f"Identidad ADN: {result_nuc.identity:.1%}")
print(format_alignment(result_nuc))
for mut in result_nuc.mutations:
    print(mut)

# 4. Comparar a nivel de proteína
result_prot = SequenceAligner.align(prot_normal, prot_mutante)
print(f"Identidad proteína: {result_prot.identity:.1%}")
for mut in result_prot.mutations:
    print(mut)
```
