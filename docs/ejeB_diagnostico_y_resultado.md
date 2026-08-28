# Diagnóstico paso por paso — y el resultado que sí aguanta

**Reproducible con:**
`tools/bench_escape_multivirus.py` (el resultado) ·
`tools/diag_escape_agregacion.py` (la auditoría) ·
`tools/diag_escape_por_que_pierde.py` (el porqué)

**Sustituye a:** [`ejeB_replica_multivirus.md`](ejeB_replica_multivirus.md), cuya
conclusión («lo que replica es el cambio de carga») era **incorrecta** — venía de
un error propio de agregación que la auditoría destapó.

---

## Por qué existe este documento

En un solo día, sobre los mismos datos, se afirmaron tres cosas distintas:

1. *«El destino bate a la distancia»* — sobre un virus.
2. *«No replica; lo que replica es la carga»* — sobre cinco.
3. Lo que sigue.

Las dos primeras eran precipitadas. La tercera está **congelada y validada en
conjuntos retenidos**. El historial se deja entero a propósito.

---

## Paso 1 — ¿el fallo era uniforme o había sitios que se salvaran?

Distribución del rho por sitio, no solo la media:

| conjunto | sitios | media | mediana | >0 | >+0,3 |
|---|---|---|---|---|---|
| SARS-CoV-2 RBD | 120 | +0,308 | +0,303 | **82 %** | **52 %** |
| gripe H3N2 sueros | 566 | −0,050 | −0,054 | 43 % | 11 % |
| VIH-1 Env | 670 | −0,073 | −0,070 | 41 % | 9 % |

En SARS-CoV-2 la señal es **uniforme y fuerte**, no la arrastran cuatro sitios.
En los demás estaba centrada en cero: no había un subconjunto que rescatar.

---

## Paso 2 — ¿era ruido de medida? (degradando SARS-CoV-2 a propósito)

SARS-CoV-2 promedia **3.051** anticuerpos; Zika, **5**. Promediar reduce el ruido
de la etiqueta, y menos ruido sube cualquier correlación. Se degradó SARS-CoV-2
a propósito, 20 repeticiones por punto:

| anticuerpos promediados | destino | carga | disimilitud |
|---|---|---|---|
| 5 | +0,143 | +0,203 | +0,027 |
| 25 | +0,132 | +0,173 | +0,044 |
| 100 | +0,183 | +0,204 | +0,048 |
| 1.000 | +0,297 | +0,295 | +0,050 |
| **3.051** | **+0,308** | +0,297 | +0,049 |

> **Gran parte de la diferencia entre SARS-CoV-2 y los demás era tamaño de
> muestra, no biología.** Con 5 anticuerpos, la propia señal de SARS-CoV-2 cae a
> +0,14 — camino de los +0,06…+0,10 que dan los conjuntos con 5-50 anticuerpos.

Y una diferencia real entre señales: la **disimilitud no mejora nunca** (+0,03 →
+0,05 con 600 veces más datos). No es que le falten datos: es que no está ahí.

---

## Paso 3 — la auditoría: un error PROPIO

Dos scripts nuestros agregaban distinto y nadie lo había notado:

| | operación |
|---|---|
| T1 | media del `diffsel` crudo |
| T2 | promediar y **luego** recortar lo negativo |
| T3 | recortar cada medida y **luego** promediar |
| T4 | **rangos** dentro del sitio por anticuerpo, promediados |

Resultado de la auditoría:

| conjunto | ¿estable ante la agregación? |
|---|---|
| **SARS-CoV-2 RBD** | **sí** — T1=T2=T3 idénticos (su métrica ya es ≥0) |
| gripe, VIH, Zika | **no** — *todas* las señales cambian de signo |

El cambio de carga daba **+0,267 con T2 y −0,090 con T3** en los mismos datos.
La afirmación «la carga replica en 5/5» era un artefacto de T2.

**Y de aquí sale el argumento decisivo, que no depende de los resultados:** en
SARS-CoV-2 las tres primeras agregaciones son *idénticas*, y en los demás
cambian el signo de todo. Compararlos bajo T1/T2/T3 es comparar cosas
incomparables. **Solo T4 los pone en igualdad**, y además es la única coherente
con un análisis basado en rangos. Todo lo que sigue usa T4.

---

## Paso 4 — lo que apareció buscando el porqué

Probando propiedades del destino apareció el **volumen**, positivo en 5/5 y en
gripe más fuerte que la hidrofilia. Y son **casi independientes**:

| conjunto | volumen solo | volumen \| sin hidrofilia | hidrofilia \| sin volumen |
|---|---|---|---|
| SARS-CoV-2 RBD | +0,062 | +0,048 | +0,192 |
| gripe H3N2 sueros | +0,124 | +0,124 | +0,052 |
| gripe H3N2 mAbs | +0,148 | +0,146 | +0,074 |
| VIH-1 Env | +0,055 | +0,049 | +0,099 |
| Zika E | +0,072 | +0,066 | +0,089 |

Cada una sobrevive al descontar la otra → sumarlas debe ganar. Y gana.

---

## El resultado

> **score = z(hidrofilia del destino) + z(volumen del destino)**
> frente al término químico de EVEscape, **\|Δhidrofobia\|**.

| conjunto | familia | rol | EVEscape | **COMBO** | diferencia (IC95 %) |
|---|---|---|---|---|---|
| SARS-CoV-2 RBD | *Coronaviridae* | desarrollo | +0,107 | **+0,172** | +0,066 [−0,004, +0,134] |
| gripe H3N2 · sueros | *Orthomyxoviridae* | desarrollo | +0,045 | **+0,164** | +0,119 [+0,093, +0,144] ✔ |
| gripe H3N2 · mAbs | *Orthomyxoviridae* | desarrollo | +0,070 | **+0,205** | +0,134 [+0,107, +0,161] ✔ |
| VIH-1 BG505 · bnAbs | *Retroviridae* | desarrollo | +0,021 | **+0,135** | +0,114 [+0,086, +0,141] ✔ |
| Zika proteína E | *Flaviviridae* | desarrollo | +0,103 | **+0,135** | +0,032 [+0,005, +0,061] ✔ |
| VIH-1 · sueros **humanos** | *Retroviridae* | **RETENIDO** | +0,053 | **+0,169** | +0,116 [+0,088, +0,142] ✔ |
| VIH-1 · sueros **de conejo** | *Retroviridae* | **RETENIDO** | +0,020 | **+0,068** | +0,048 [+0,022, +0,074] ✔ |

**Gana en 7/7, con intervalo limpio en 6/7, incluidos los 2/2 retenidos.**

El volumen se eligió *después* de ver los datos de desarrollo. Por eso el modelo
se **congeló** y se aplicó tal cual a dos conjuntos que no intervinieron en
ninguna decisión. Ahí también gana.

---

## Límites, declarados

- Los dos retenidos son **Env de VIH**: validan generalización entre
  **repertorios de anticuerpos** (bnAbs → sueros humanos → sueros de conejo),
  **no entre virus**. Un retenido de otra familia sigue pendiente.
- Los tamaños de efecto son **modestos** (0,07–0,21). Esto **ordena mejor** que
  el término químico de EVEscape; no resuelve el escape.
- Se compara **un término** de EVEscape, no EVEscape entero. Su método completo
  multiplica tres factores y no se ha reproducido aquí.
- El cambio de **carga** queda **retirado**: era artefacto de agregación.

---

## Qué queda en pie de toda la investigación

| resultado | estado |
|---|---|
| escape ⊥ viabilidad (−0,147 / +0,131, control +0,463) | en pie |
| la accesibilidad 3D aporta ~0,07 dentro del RBD | en pie, con su límite |
| **destino (hidrofilia + volumen) > disimilitud, 7/7** | **en pie, con retenidos** |
| el cambio de carga replica | **RETIRADO** — artefacto de agregación |
| la rotación temporal aporta | descartado (intento 1) |
