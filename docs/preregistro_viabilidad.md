# Pre-registro — ¿Sobrevivirá esta variante?

**Autor:** Aarón Aranda Torrijos
**Estado:** documento de PRE-REGISTRO. Se escribe **antes** de medir nada.
**Sello de tiempo:** la fecha del commit de git que introduce este archivo.

> Un pre-registro sirve para una sola cosa: impedir que el autor —yo— cambie la
> pregunta, el umbral o el criterio de éxito **después** de ver los resultados. Todo
> lo que aparece aquí queda fijado. Si algo se cambia más adelante, se cambia en un
> commit aparte, con su motivo escrito, y el resultado pasa a ser exploratorio.

---

## 1. Qué es esta herramienta (y qué NO es)

**Es un EVALUADOR, no un oráculo.**

El usuario aporta dos cosas:

1. El **historial evolutivo** de un virus: secuencias de la misma proteína a lo largo
   del tiempo, con sus fechas.
2. Una o varias **variantes candidatas** que quiere someter a juicio.

La herramienta dictamina si esa variante **tiene posibilidades de sobrevivir**, y
—esto es lo que la distingue— **cuánto hay que fiarse de ese dictamen**.

**No** predice qué hará el virus. **No** dice cuál será la próxima cepa. Responde a
una hipótesis que trae el usuario.

**Es virus-agnóstica.** Todo se deriva del alineamiento y las fechas que aporte el
usuario. No hay tablas de epítopos ni estructuras 3D codificadas. Eso permite
analizar un virus que nadie ha modelado todavía, y tiene un precio declarado: en los
virus que sí tienen modelos específicos (gripe con mapas antigénicos, SARS-CoV-2 con
DMS), **esos modelos deberían ser mejores que nosotros**. Se asume y se dirá.

---

## 2. La pregunta, formalizada

Sea una variante **M** (una sustitución `sitio→aminoácido`, o un conjunto de ellas),
un corte temporal **T** y un horizonte **h**:

> ¿**M** alcanzará y **mantendrá** presencia real en la población durante el
> intervalo (T, T+h]?

**Definición operativa de SOBREVIVIR** (fijada aquí, no negociable después):

```
sobrevive(M, T, h) = 1   si   max frecuencia de M en (T, T+h] ≥ θ
                          Y   M está presente en el último tramo del intervalo
```

- **θ = 0,05** (5 %) como valor principal. Análisis de sensibilidad en 0,02 y 0,10.
- La segunda condición (presencia al final) descarta los **destellos**: algo que
  aparece un mes y desaparece no ha sobrevivido.
- **Efecto techo, resuelto:** una variante que ya está al 98 % y sigue al 98 % **SÍ
  sobrevive**, aunque no "suba". Se mide presencia sostenida, no crecimiento.
- **h = 6 meses** como principal; 3 y 12 como comprobación de robustez.

---

## 3. Las dos condiciones (ejes separados, y por qué)

Una variante sobrevive solo si cumple **las dos**. Se reportan por separado y
visibles: un único número escondería justo lo que hace útil el diagnóstico.

### Eje A — ¿Sigue funcionando? (viabilidad)

Una proteína puede mutar hasta romperse. Un virus con la proteína de entrada
inutilizada no infecta a nadie, por invisible que sea para el sistema inmune.

Señales, todas derivadas del alineamiento del usuario:

| señal | qué mide | dirección esperada |
|---|---|---|
| conservación del sitio | cuánta variación tolera esa posición | más conservado → menos viable |
| disimilitud fisicoquímica | cuánto cambia la química del residuo | más disímil → **menos** viable |
| ¿se ha visto alguna vez ese residuo ahí? | evidencia directa de que funciona | visto → más viable |
| compatibilidad con el fondo genético | si co-ocurre con lo que circula | co-ocurre → más viable |

> **Nota:** la disimilitud fisicoquímica ya fue medida en este proyecto y salió
> **invertida** respecto a lo que se esperaba, replicado en H3N2, H1N1 y linaje B.
> En su día se interpretó como un eje "que falla". Este pre-registro sostiene la
> hipótesis alternativa: **no fallaba, medía viabilidad y se estaba usando para
> predecir escape**. Aquí ocupa su lugar correcto y se predice que su signo será
> el mismo (negativo) en el eje A.

### Eje B — ¿Es nueva para el sistema inmune? (escape)

Si la variante se parece a lo que ya circuló, la población tiene defensas y no
despega.

| señal | qué mide | dirección esperada |
|---|---|---|
| exposición reciente | frecuencia de ese residuo en la ventana reciente | más visto → menos escape |
| variabilidad temporal del sitio | sitios que cambian una y otra vez | más cambiante → más escape |
| distancia al consenso actual | cuánto se aparta de lo que circula hoy | más lejos → más escape |

> **Honestidad obligatoria en el producto:** esto **no es inmunología**. Es un
> *sustituto* medible («cuánto se parece a lo que la población ya vio»), el mismo
> principio de la cartografía antigénica. Modelar la respuesta inmune de verdad
> exige datos que no tenemos. Se dirá así en la interfaz, no solo aquí.

---

## 4. El listón: contra qué hay que ganar

Ningún resultado significa nada sin esto. **El rival no es 0,5.** Se mide contra el
mejor de estos ejes gratuitos, cada uno por separado y en el mismo conjunto:

1. **Azar** (AUC 0,5).
2. **Mutabilidad del sitio** — el que ya nos engañó una vez: nuestro "AUC 0,80"
   histórico resultó ser exactamente esto.
3. **Frecuencia actual** — trivialmente fuerte para lo ya circulante.
4. **Conservación** sola.
5. **«Nunca vista → no sobrevivirá»** — regla ingenua, sorprendentemente dura.

---

## 5. Criterios de ÉXITO (fijados antes de medir)

Se declara éxito **solo si se cumplen los cuatro**:

1. **Régimen NUEVAS.** Sobre mutaciones nunca observadas antes de T, el AUC del
   modelo combinado debe superar al mejor eje trivial con **IC 95 % bootstrap que no
   se solapen**. Es la única prueba que importa: en lo ya circulante basta contar.
2. **Precisión, no solo AUC.** Con una tasa base baja, el AUC engaña. Se exige
   **precisión en el top-20** significativamente mejor que la tasa base.
3. **Calibración.** De las variantes a las que se asigne ~0,70, aproximadamente el
   70 % debe haber sobrevivido. Se reporta curva de calibración y Brier score.
4. **Replicación.** Debe sostenerse en **≥3 virus distintos**, no en uno elegido.

---

## 6. Criterios de FRACASO (declarados de antemano)

Esto es lo que convierte el trabajo en ciencia y no en publicidad:

- Si los ejes **no superan el listón trivial en el régimen NUEVAS** en al menos 2 de
  los 3 virus → **se declara que no aportan señal**, y ese resultado negativo se
  publica igual, con la misma claridad que si hubiera salido bien.
- Si la calibración es mala pero el orden es bueno → se reporta como **ordenador, no
  como estimador de probabilidad**, y se retira la cifra de probabilidad del producto.
- Si el eje B no separa nada → se publica la herramienta **solo con el eje A**
  (viabilidad), diciendo que el escape no se pudo modelar sin datos inmunológicos.

**Compromiso explícito:** ningún resultado negativo se esconde ni se reencuadra.

---

## 7. Protocolo de validación

1. **Separación temporal estricta.** Para cada corte T, el modelo solo ve datos
   anteriores a T. Nada del futuro, ni para normalizar.
2. **Muchos cortes.** No un corte elegido: una rejilla de cortes a lo largo del
   histórico, reportando la distribución de resultados, no el mejor.
3. **Detector de fuga activado**, descontando la caída de un eje trivial de control.
4. **Varios virus**, con el mismo código y sin ajustar parámetros por virus.
5. **Semillas fijadas** y todo reproducible desde cero con `tools/`.

---

## 8. Amenazas a la validez (reconocidas antes, no después)

| amenaza | por qué duele | cómo se mitiga |
|---|---|---|
| **Sesgo de muestreo** | las secuencias públicas dependen de qué países secuencian; una "subida" puede ser un cambio de vigilancia | se reporta como limitación; comprobación de sensibilidad excluyendo el país dominante |
| **Fuga temporal** | la forma nº1 de engañarse | detector automático, ya construido |
| **Desequilibrio de clases** | AUC alto con precisión ridícula | métrica de precisión en top-k obligatoria |
| **Elegir θ o h a posteriori** | convierte cualquier ruido en resultado | fijados en este documento |
| **Comparaciones múltiples** | probar 3 virus × 3 horizontes × 3 umbrales encuentra algo por azar | el criterio principal es UNO; el resto es sensibilidad declarada |
| **Proxy inmunológico** | el eje B no es inmunología | se declara en el producto, no solo aquí |

---

## 9. Qué podrá hacer el usuario

No solo mirar un veredicto:

- **Juzgar una candidata** → veredicto con los dos ejes y su fiabilidad.
- **Comparar varias** → ordenarlas y ver por qué una gana a otra.
- **Explicar** → desglose por señal: *«escapa bien, pero probablemente esté rota»*.
- **Barrer** → puntuar todas las mutaciones posibles de un sitio o una región.
- **Contexto** → evaluar la candidata sobre un fondo genético concreto.

---

## 10. Estado del arte (contra quién se compara)

| referencia | qué hace | por qué no es el mismo problema |
|---|---|---|
| EVEscape (Nature 2023) | escape con fitness + estructura | necesita estructura 3D; atado a virus modelados |
| Łuksza & Lässig (2014) | fitness de clados en gripe | predice clados, no juzga una candidata |
| Nextstrain / LBI | seguimiento en tiempo real | seguimiento, no evaluación de hipótesis |
| PyR0 (Obermeyer) | fitness de linajes | específico de SARS-CoV-2 |
| ESM-2 / Hie et al. | lenguaje de proteínas | fuga de preentrenamiento medida aquí: −0,20 |

**Dónde está el hueco:** ninguno responde *«¿cuánto debo fiarme de esta puntuación
concreta?»*, y ninguno funciona sobre un virus sin modelo previo.

---

## 11. Lo que NO se promete

- No es inmunología: el eje B es un sustituto medible.
- No supera a los modelos específicos en los virus que ellos cubren.
- No dice qué hará el virus; juzga lo que el usuario propone.
- Si los datos no alcanzan, **dirá que no lo sabe** en vez de inventar un número.
