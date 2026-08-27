# Eje B, intento 1 — resultado NEGATIVO (y por qué el negativo importa)

**Fecha:** el commit que introduce este archivo.
**Referencia:** [`preregistro_viabilidad.md`](preregistro_viabilidad.md), sección 6
(criterios de fracaso declarados de antemano).

> Este documento registra un intento fallido **completo**, con sus números. Se
> escribe con el mismo cuidado que si hubiera salido bien, porque el pre-registro
> se comprometió a ello.

---

## Qué se intentó

El eje B pregunta: *¿es esta variante nueva para el sistema inmune?* EVEscape lo
resuelve con **accesibilidad a anticuerpos sacada de estructuras 3D** — justo lo que
este proyecto renuncia a usar.

**Hipótesis de innovación:** la accesibilidad se puede sustituir por una señal
**temporal**. Un sitio al que llegan los anticuerpos está bajo presión inmune, y un
sitio bajo presión inmune **cambia de residuo dominante una y otra vez**. Eso se
mide con secuencias fechadas, que los rivales no tienen (sus MSA son de homólogos
sin fecha). Se llamó **ROTACIÓN**.

Se probaron tres señales sobre el conjunto de pronóstico de EVEREST (33 clados
reales de H1N1, H3N2, VIH y SARS-CoV-2):

| señal | idea |
|---|---|
| **rotación** | cuántas veces cambia el residuo dominante del sitio, año a año |
| **novedad** | ¿ha visto la población ese residuo en los últimos años? |
| **rotación × novedad** | producto, como hace EVEscape con sus tres términos |

---

## El confusor que hubo que eliminar primero

La etiqueta del benchmark es ``count``: cuántas veces se vio la mutación. Pero **una
mutación que aparece el primer mes acumula cuenta toda la ventana** y una que aparece
el último, no.

**Medido: ``first_seen`` solo correlaciona 0.45 de media** (0.20-0.65 según el clado)
con la cuenta. Un modelo que reportara 0.5 sin descontar eso estaría midiendo un
reloj, no biología. Y como el usuario traerá una variante que **quizá aún no existe**,
``first_seen`` ni siquiera es una característica disponible: es un confusor.

Por eso todo se mide con **correlación parcial de rangos**, eliminando ``first_seen``
de ambos lados. Control: el azar da ~0.00 con esa métrica.

---

## Resultados

Correlación parcial con la propagación real (media sobre los clados):

| señal | H1N1 (9 clados) | H3N2 (15 clados) |
|---|---|---|
| rotación | **+0.098** | **+0.157** |
| novedad | **−0.278** | **−0.231** |
| rotación × novedad | +0.092 | +0.120 |

Las dos señales son **consistentes en los 24 clados**, sin excepciones.

### La novedad salió INVERTIDA — y ya había pasado antes

Se predijo: *lo que el sistema inmune no ha visto → escapa → se propaga*. La medida
dice lo contrario, y con fuerza. Interpretación: «novedad» no mide escape, mide **si
el destino es viable** — un residuo que ya circula con éxito es un destino que
funciona; uno raro fracasa.

Es **exactamente el mismo patrón** que la disimilitud fisicoquímica del trabajo
previo, que también salió invertida y también resultó medir viabilidad. Dos veces el
mismo error confirma un principio: **casi cualquier señal derivada de secuencias
acaba midiendo viabilidad si no se controla contra ella.**

### La prueba decisiva: ¿aporta algo sobre el eje A?

| virus | eje A solo | rotación sola | eje A + rotación | ¿aporta? |
|---|---|---|---|---|
| H1N1 | +0.1235 | +0.0981 | +0.1236 | **no** (+0.0001) |
| H3N2 | +0.2339 | +0.1568 | +0.2264 | **no** (empeora) |

**La rotación no aporta nada.** Los sitios que rotan son los que toleran variación,
y eso ya estaba dentro del eje A. La hipótesis de la accesibilidad temporal **no se
sostiene** con estos datos.

---

## Por qué falló: el diagnóstico importa más que el fallo

Se revisó qué mide realmente el benchmark. De los **45 conjuntos DMS**:

| tipo de ensayo | cuántos |
|---|---|
| fitness | 31 |
| stability | 5 |
| expression | 5 |
| binding | 2 |
| abundance | 1 |
| activity | 1 |
| **escape a anticuerpos** | **0** |

**No hay un solo ensayo de escape.** Todo mide viabilidad. Y el conjunto de
pronóstico mide propagación poblacional, que está dominada por fitness y deriva.

> Se estaba buscando señal de escape **en datos que no miden escape**. El fallo no
> fue de la señal: fue de la elección del conjunto de validación.

Esto explica limpiamente por qué todas las señales convergen a viabilidad: es lo
único que hay en la etiqueta.

---

## Qué NO se concluye

- **No** se concluye que la rotación sea inútil. Se concluye que **no se puede
  demostrar** con estos datos. Es distinto y hay que decirlo así.
- **No** se activa todavía el criterio de fracaso del pre-registro (publicar solo el
  eje A). Ese criterio exige haber fallado con datos adecuados, y estos no lo eran.

## Siguiente paso, ya identificado

Conseguir **verdad de campo específica de escape**: los mapas de escape a anticuerpos
por *deep mutational scanning* (grupo de Bloom, públicos), o datos antigénicos de
inhibición de la hemaglutinación. Es contra eso que EVEscape se valida — no contra
recuentos poblacionales.

Sin esa etiqueta, cualquier «eje B» que midamos seguirá siendo el eje A con otro
nombre, por muy ingeniosa que sea la señal.
