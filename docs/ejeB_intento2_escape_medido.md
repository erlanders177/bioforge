# Eje B, intento 2 — contra escape MEDIDO (resultado positivo, pero acotado)

**Referencia:** [`preregistro_viabilidad.md`](preregistro_viabilidad.md) ·
[`ejeB_intento1_negativo.md`](ejeB_intento1_negativo.md)
**Reproducible con:** `python tools/bench_escape_vs_evescape.py`

> El intento 1 falló y el diagnóstico fue: *se buscaba señal de escape en datos que
> no miden escape*. Este documento registra el intento 2 con la etiqueta correcta.

---

## La etiqueta correcta

Mapas de escape a anticuerpos del **laboratorio de Bloom**
([`jbloomlab/SARS2_RBD_Ab_escape_maps`](https://github.com/jbloomlab/SARS2_RBD_Ab_escape_maps),
GPL-3, agregados de 13 estudios publicados):

| | |
|---|---|
| anticuerpos y sueros | **3.051** |
| medidas de escape | **2.388.385** |
| mutaciones del RBD | **2.157** |

Es contra este tipo de dato que EVEscape se valida. **No se redistribuye**: el
benchmark lo descarga y mide contra él.

---

## Un error que se cazó por el camino

Al medir la diversidad del RBD, cortar las secuencias por posición **sin alinear**
daba «150 de 201 sitios variables, entropía 0,50». Implausible para 2020. La
comprobación contra hechos conocidos lo destapó: el sitio 417 salía **D** cuando
tiene que ser **K**, y el 501 salía **H** cuando tiene que ser **N**. Había
deleciones que desplazaban toda la numeración. Alineado, K417/E484/N501 aparecen
correctos.

**Lección:** anclar siempre en un hecho externo verificable antes de creerse una
cifra. La entropía falsa habría fabricado señal de la nada.

Segundo hallazgo del mismo tipo: esas secuencias **no son de 2020** (N501Y en
173/247 → Ómicron/Beta/Gamma). Usarlas para predecir escape sería **circular**: esos
sitios varían en la población *precisamente porque* escaparon. Se descartó esa vía.

---

## Pregunta 1 — ¿son ESCAPE y VIABILIDAD dos ejes de verdad separados?

Es la pregunta que sostiene el diseño de la herramienta. Si correlacionaran fuerte,
con un eje bastaba.

| | rho |
|---|---|
| escape vs **unión a ACE2** | **−0,147** |
| escape vs **expresión** | **+0,131** |
| unión vs expresión | **+0,463** ← *control* |

El control es lo que hace válida la lectura: dos medidas de **viabilidad** sí
correlacionan a 0,46, así que el método detecta correlación cuando la hay. El escape
se queda en ~0,14 **y con signos opuestos** frente a las dos medidas de viabilidad —
es decir, no sigue a la viabilidad en absoluto.

> **Los dos ejes son casi ortogonales. La arquitectura de dos ejes queda validada
> con datos, no por intuición.**

Y explica limpiamente el fracaso del intento 1: si escape ⊥ viabilidad, un benchmark
lleno de ensayos de fitness **no podía** revelar señal de escape.

---

## Cuánto del problema es trivial

| | |
|---|---|
| varianza del escape explicada **solo por el sitio** | **56,3 %** |
| queda **dentro** del sitio | **43,7 %** |

Que los sitios antigénicos son los variables se sabe desde los años 80: el nivel de
sitio se **descarta por tautológico**. Todo lo que sigue se mide **dentro de cada
sitio**, con el efecto del sitio eliminado.

---

## Pregunta 2 — ¿destino o distancia?

EVEscape modela el término químico como **DISIMILITUD**: cuánto te alejas del
residuo original. Se puso a prueba la alternativa: que lo que manda es **el residuo
al que llegas**.

| señal | promediando medidas | promediando clases |
|---|---|---|
| **destino hidrofílico** | **+0,3083** | **+0,1823** |
| \|Δcarga\| (listón) | +0,2370 | +0,1474 |
| \|Δhidrofobia\| (marco EVEscape) | +0,0485 | +0,0461 |

Bootstrap **pareado** de la diferencia (la prueba correcta; comparar intervalos que
se solapan no vale):

| comparación | medidas | clases |
|---|---|---|
| destino − disimilitud | +0,2598 **[+0,176, +0,345]** | +0,1362 **[+0,060, +0,210]** |
| destino − Δcarga | +0,0816 [+0,031, +0,132] | +0,0432 **[−0,007, +0,093]** |

Y el desconfundido, que es lo que da sentido al resultado:

| | sola | descontando la otra | conserva |
|---|---|---|---|
| destino hidrofílico | +0,3083 | **+0,2404** | 78 % |
| \|Δcarga\| | +0,2370 | **+0,0816** | 34 % |

La carga era en buena medida **el reflejo** de la hidrofilia, no al revés.

### Lo que SÍ se afirma

> **Dentro de un sitio, el escape se predice mejor por el residuo al que se LLEGA
> que por lo lejos que se va.** Destino hidrofílico bate a la disimilitud con las
> **dos** agregaciones y con intervalos que no tocan el cero (+0,26 y +0,14).

Tiene sentido físico: los epítopos están en la superficie expuesta al disolvente.
Un residuo hidrofílico proyecta hacia fuera y cambia lo que el anticuerpo toca; uno
hidrofóbico tiende a plegarse hacia dentro.

### Lo que NO se afirma

- **No** que bata al cambio de carga. Con una agregación sale significativo y con la
  otra no (IC [−0,007, +0,093]). Depende de cómo se agregue → **no se vende**.
- El efecto **por clase de epítopo por separado es pequeño** (+0,01 a +0,15). El
  0,31 vale para el escape promediado sobre un repertorio amplio de anticuerpos,
  que es la magnitud relevante para inmunidad poblacional, pero **es otra cosa** que
  el escape frente a un anticuerpo concreto, y hay que decirlo así.
- Es **un virus y un dominio** (RBD de SARS-CoV-2). El pre-registro exige ≥3 virus:
  **este criterio no está cumplido todavía.**

---

## Siguiente paso

Repetir contra escape medido en **otro virus** — mapas de escape a suero de H3N2
(Lee et al. 2019) y de Env de VIH (Dingens et al.), ambos del mismo laboratorio y
también públicos. Si «destino > disimilitud» aguanta ahí, deja de ser una
particularidad del RBD y pasa a ser un resultado sobre el escape en general.

---

# Anexo — el término (2), la accesibilidad: ¿hace falta el 3D?

**Reproducible con:** `python tools/bench_accesibilidad_sin_3d.py`

La pregunta llegó del usuario, y era la correcta: *ellos usan 3D, pero para tener
el 3D primero hay que montarlo bien; ¿y si vemos cómo lo hacen y lo adaptamos?*

La versión afilada: **la estructura solo es una máquina para producir un número por
residuo — cuán expuesto está.** Si ese número se obtiene de otra forma, la máquina
sobra. Así que antes de intentar sustituirla, se midió **cuánto vale ese número**.

## Cómo se midió

Estructura experimental real **6M0J** (RBD de SARS-CoV-2), cadena E **sola, sin
ACE2** — así lo ve un anticuerpo; con ACE2 pegado la interfaz saldría falsamente
enterrada. Exposición por **Shrake-Rupley implementado aquí en NumPy puro**
(1.542 átomos, 194 residuos), normalizada por Tien et al. 2013.

**Validado contra hechos conocidos** antes de usarlo: F486 y T500 (interfaz con
ACE2) salen expuestos (0,73); el núcleo enterrado (<0,05) son 40 residuos y son
V/I/L/A/C/F/Y — hidrofóbicos, como debe ser.

## El resultado

| | rho vs escape medido |
|---|---|
| exposición **de la estructura real** | **+0,065** |
| exposición estimada solo de secuencia | +0,076 |

Y el contraste directo, restringido a mutaciones que **sí se expresan bien** (para
descartar que una mutación enterrada simplemente despliegue la proteína, que en el
ensayo se parece a escape):

| | escape medio |
|---|---|
| residuos **enterrados** (n=271) | 0,1563 |
| residuos **expuestos** (n=505) | 0,1771 |
| razón | **1,13×** |

> **Los residuos escondidos dentro de la proteína escapan casi igual que los de la
> superficie.** Con la estructura experimental en la mano —el mejor caso posible,
> ni siquiera una predicha— la accesibilidad no ordena el escape dentro del RBD.

Lectura biológica: el escape no exige que el anticuerpo *toque* el residuo mutado.
Una mutación en el núcleo reorganiza el dominio y deforma el epítopo desde dentro.

## Qué significa para el proyecto

**No hacía falta rendirse, pero tampoco insistir: la puerta que intentábamos abrir
no lleva a ninguna parte.** No podemos igualar a EVEscape por *no tener* el término
(2) — ese término, en este dominio, aporta ~0,07.

Ahorra semanas de trabajo en predecir accesibilidad sin estructura, y lo dice con
números en vez de con intuición.

Se comprobó además si nuestra señal era accesibilidad disfrazada. No lo es: el
efecto «destino hidrofílico» es **uniforme** con la exposición (+0,279 enterrados,
+0,355 intermedios, +0,270 expuestos).

## Límites de esta conclusión — importantes

- **Es dentro del RBD**, un dominio mayoritariamente expuesto. Sobre una proteína
  entera, la accesibilidad sí separaría dominios enteros (núcleo vs superficie) y
  pesaría más. **No se puede comprobar aquí** porque los mapas de Bloom son solo
  del RBD. La afirmación se limita a ese ámbito.
- Se midió la **accesibilidad al disolvente**, que es la forma estándar de
  operacionalizar el término, **no la fórmula exacta** de EVEscape (que usa un
  número de contactos ponderado y un término de orientación).
- Un término puede valer poco por separado y aportar dentro del producto de tres.
