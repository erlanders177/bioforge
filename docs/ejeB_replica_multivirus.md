# Réplica en cuatro familias de virus — el hallazgo NO sobrevive

> ## SUPERADO — su conclusión era INCORRECTA
> Este documento concluía que «lo que replica es el cambio de carga». **Falso.**
> Una auditoría posterior destapó un error PROPIO de agregación: dos scripts
> nuestros promediaban y recortaban en orden distinto, y la carga daba +0,267 o
> −0,090 sobre los MISMOS datos según cuál se usara. Con la agregación coherente,
> el destino (hidrofilia + volumen) gana en 7/7 y la carga se retira.
> Ver [`ejeB_diagnostico_y_resultado.md`](ejeB_diagnostico_y_resultado.md).
>
> Se conserva sin borrar: el recorrido del error es parte del registro.

**Reproducible con:** `python tools/bench_escape_multivirus.py`
**Corrige a:** [`ejeB_intento2_escape_medido.md`](ejeB_intento2_escape_medido.md)

> Esta mañana medimos, sobre el RBD de SARS-CoV-2, que el escape se predice
> mejor por **el residuo al que llegas** (+0,31) que por **lo lejos que vas**
> (+0,05) — el marco que usa EVEscape. Se dijo entonces, por escrito, que era
> **un solo virus y un solo dominio** y que no se afirmaba nada general hasta
> replicarlo. Esto es la réplica. **No sobrevive.**

---

## Los datos

Cinco conjuntos, **cuatro familias de virus**, dos estilos de ensayo. Todos
públicos, del laboratorio de Bloom, **no redistribuidos**.

| familia | virus / proteína | anticuerpos | métrica |
|---|---|---|---|
| *Coronaviridae* | SARS-CoV-2, RBD | ~3.000 mAbs y sueros | escape fraction |
| *Orthomyxoviridae* | gripe H3N2, HA | **sueros humanos** | diffsel |
| *Orthomyxoviridae* | gripe H3N2, HA | monoclonales | diffsel |
| *Retroviridae* | VIH-1 BG505, Env | bnAbs | diffsel |
| *Flaviviridae* | Zika, proteína E | monoclonales | diffsel |

---

## Una decisión metodológica, declarada

`diffsel` incluye selección **negativa** (mutaciones que se agotan). El escape
es, por convención del campo —y del propio laboratorio de Bloom, que grafica
*positive diffsel*—, la selección **positiva**.

**Aviso de honestidad:** ese tratamiento se aplicó **después** de ver el
resultado del crudo. Es el correcto por principio, pero la elección fue
posterior, y por eso el benchmark **reporta los dos** y la conclusión se apoya
solo en lo que vale con ambos.

---

## El resultado

Correlación media **dentro de cada sitio** (el nivel de sitio se descarta por
tautológico):

### Tratamiento principiado (solo selección positiva)

| conjunto | sitios | **destino** | **\|Δcarga\|** | \|Δhidrofobia\| |
|---|---|---|---|---|
| SARS-CoV-2 RBD | 120 | **+0,3083** | +0,2370 | +0,0485 |
| gripe H3N2 · sueros | 566 | +0,0203 | **+0,2672** | −0,0104 |
| gripe H3N2 · mAbs | 566 | +0,0187 | **+0,3335** | +0,0226 |
| VIH-1 Env | 670 | +0,0111 | **+0,2222** | +0,0080 |
| Zika E | 504 | −0,0377 | **+0,2498** | −0,0837 |

### Lo que dice

1. **El hallazgo no replica.** «Destino» da +0,31 en SARS-CoV-2 y **~0,02 en
   todo lo demás** — en Zika incluso negativo. Y esto **no depende** del
   tratamiento: en crudo también se queda en +0,06…+0,09 frente a una
   disimilitud de +0,02…+0,10, es decir, empatando.

2. **Lo que SÍ replica es el listón que intentábamos batir.** El **cambio de
   carga** da entre **+0,22 y +0,33 en las cinco**, cuatro familias de virus y
   dos estilos de ensayo. Es la señal general.

3. **El RBD de SARS-CoV-2 es un caso atípico.** Es donde el campo entero se ha
   concentrado estos años. Un resultado obtenido solo ahí —el nuestro de esta
   mañana, y conviene preguntarse si también alguno ajeno— puede no decir nada
   sobre el escape en general.

---

## Qué queda en pie de todo el día

| resultado | estado |
|---|---|
| escape ⊥ viabilidad (−0,147 / +0,131, control +0,463) | **en pie** — medición independiente |
| la accesibilidad 3D aporta ~0,07 dentro del RBD | **en pie** — con su límite declarado |
| el cambio de carga predice escape en 4 familias | **en pie, y es nuestro mejor componente** |
| «destino bate a distancia» | **CAÍDO** — solo vale en SARS-CoV-2 |

El eje B no se queda vacío: se queda con una señal **validada en cuatro
familias de virus**, gratis, sin estructura y sin IA. No es novedad científica
—el cambio de carga como determinante del escape es conocido—, pero está
**medido por nosotros, en cinco conjuntos, con la disciplina puesta**. Y ahora
sabemos qué NO añade: ni accesibilidad, ni disimilitud, ni rotación temporal.

---

## Por qué este documento existe

El pre-registro comprometía a publicar los fallos con el mismo cuidado que los
aciertos. Entre el resultado de la mañana y este han pasado unas horas; el
primero era más bonito y ya estaba escrito y subido. Se corrige porque los
datos dicen otra cosa.

Es exactamente el caso para el que se escribió la regla: **si salimos ganando,
sospechar primero de la comparación.**
