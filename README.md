# Predicción de Ventas según la Inversión Publicitaria

Proyecto de Machine Learning que predice las **ventas** de una empresa a partir de
su inversión publicitaria en **TV, Radio y Periódicos** (dataset *Advertising*).

## Introducción al problema

Toda empresa que invierte en publicidad se enfrenta a la misma pregunta:

> *«Si reparto mi presupuesto entre TV, Radio y Periódico, ¿cuántas ventas puedo esperar,
> y cómo debería distribuir ese dinero para vender lo máximo posible?»*

Invertir "a ciegas" es caro: se puede gastar mucho en un canal que apenas influye en las
ventas mientras se descuida otro más rentable. El objetivo de este proyecto es sustituir
esa intuición por **decisiones basadas en datos**.

Para ello usamos el dataset **Advertising**, con **200 registros** históricos. Cada registro
recoge cuánto se invirtió en cada canal (en miles de $) y las ventas que se obtuvieron:

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `TV` | Predictora (X) | Inversión en publicidad de TV |
| `Radio` | Predictora (X) | Inversión en publicidad de Radio |
| `Newspaper` | Predictora (X) | Inversión en publicidad de Periódico |
| `Sales` | **Objetivo (y)** | Ventas obtenidas |

Como la variable a predecir (`Sales`) es un **número continuo**, estamos ante un problema de
**regresión supervisada**: aprendemos de ejemplos históricos (inversión → ventas) para poder
predecir las ventas de una inversión futura.

## Objetivos
1. **Predecir** las ventas en función de la inversión en cada canal.
2. Determinar la **influencia de cada canal** publicitario en las ventas.
3. **Optimizar** la asignación del presupuesto para maximizar las ventas.

## Cumplimiento de requisitos
| Requisito | Dónde se implementa |
|-----------|--------------------|
| División **70% / 20% / 10%** (train/test/validación) | `src/data.py` → `dividir_datos()` |
| **2 métricas** de validación: **R²** y **RMSE** | `src/evaluate.py` |
| **2 modelos / 2 algoritmos**: Regresión Lineal + KNN | `src/models.py` |

## ¿Por qué se eligieron estos modelos?

Se seleccionaron **dos algoritmos deliberadamente distintos** para poder comparar enfoques
opuestos y quedarnos con el que mejor funcione. La idea es contrastar un modelo **simple e
interpretable** frente a uno **flexible que no asume ninguna forma concreta** en los datos.

### Modelo A — Regresión Lineal Múltiple
- **Por qué:** el EDA muestra una relación **claramente lineal** entre inversión y ventas
  (TV correlaciona 0.78 con las ventas), así que es el modelo natural de partida.
- **Ventaja clave — interpretabilidad:** produce una fórmula con un coeficiente por canal
  (`Ventas = β₀ + β₁·TV + β₂·Radio + β₃·Newspaper`). Esos coeficientes responden directamente
  al **Objetivo 2** (influencia de cada canal) y alimentan el **Objetivo 3** (optimización del
  presupuesto). Sin un modelo interpretable no podríamos justificar *por qué* recomendamos un
  reparto u otro.
- Es rápido, no tiene apenas hiperparámetros y sirve como **línea base** de referencia.

### Modelo B — KNN (K-Vecinos más Cercanos)
- **Por qué:** es un algoritmo **no paramétrico**; no asume que la relación sea lineal, sino
  que predice mirando los casos históricos **más parecidos**. Sirve de contraste: si KNN gana,
  significa que hay patrones no lineales que la regresión no capta.
- **Cuidados aplicados:** como KNN se basa en **distancias**, es imprescindible estandarizar
  las variables (`StandardScaler`), porque si no un canal con cifras grandes dominaría el cálculo.
  Además, el mejor número de vecinos `k` se elige con **validación cruzada** (`GridSearchCV`),
  no a mano.

> **En resumen:** Regresión Lineal aporta *explicación* (por qué vende cada canal) y KNN aporta
> *precisión* (mejor ajuste). Comparándolos con las mismas métricas elegimos el mejor con criterio.

## Sobre las métricas de validación

Los modelos se evalúan con **dos métricas complementarias**, calculadas sobre los tres
conjuntos (entrenamiento, test y validación). Se usan dos porque cada una cuenta algo distinto:

### R² — Coeficiente de determinación
- **Qué mide:** el **porcentaje de la variabilidad** de las ventas que el modelo consigue
  explicar. Es una medida *relativa* de calidad.
- **Cómo se lee:** va de 0 a 1 → **más cerca de 1 = mejor**. Un R²=0.88 significa que el modelo
  explica el 88% de por qué varían las ventas.
- **Ventaja:** es adimensional, así que permite comparar modelos de un vistazo.

### RMSE — Raíz del Error Cuadrático Medio
- **Qué mide:** el **error típico** de las predicciones, en las **mismas unidades que las ventas**.
- **Cómo se lee:** **más bajo = mejor**. Un RMSE≈1.96 significa que, de media, el modelo se
  equivoca en ~1.96 unidades de venta. Al elevar los errores al cuadrado, **penaliza más los
  fallos grandes**.
- **Ventaja:** es interpretable en el contexto del negocio (te dice de cuánto es el error real).

### ¿Por qué dos métricas y no una?
R² dice **qué tan bien encaja** el modelo (en %), pero no cuánto se equivoca en unidades reales;
RMSE dice **cuánto falla** en ventas, pero no si eso es mucho o poco respecto al total. Juntas dan
una imagen completa. El **conjunto de validación** (10%, datos nunca vistos) es el juez final:
sobre él comparamos ambos modelos y elegimos el ganador.

## Estructura del proyecto
```
predicciones-ventas/
├── data/
│   └── raw/
│       └── advertising.csv        # dataset original (200 registros)
├── src/                           # código fuente modular
│   ├── config.py                  # rutas, columnas y constantes (split, semilla)
│   ├── data.py                    # carga + división 70/20/10
│   ├── eda.py                     # análisis exploratorio y figuras
│   ├── models.py                  # los 2 modelos (Regresión Lineal + KNN)
│   ├── evaluate.py                # métricas R² y RMSE + comparación
│   └── optimize.py                # influencia de canales + optimización presupuesto
├── notebooks/
│   └── prediccion_ventas.ipynb    # mismo análisis paso a paso (para presentar)
├── outputs/
│   └── figuras/                   # gráficos generados automáticamente
├── main.py                        # ejecuta todo el pipeline de una vez
├── requirements.txt
└── README.md
```

## Instalación
```bash
pip install -r requirements.txt
```

## Uso

**Opción 1 — pipeline completo (script):**
```bash
python main.py
```
Imprime los resultados por consola y guarda las figuras en `outputs/figuras/`.

**Opción 2 — notebook paso a paso:**
```bash
jupyter notebook notebooks/prediccion_ventas.ipynb
```

## Metodología
1. **EDA** — estadísticas, correlaciones y dispersión de cada canal frente a las ventas.
2. **División 70/20/10** — en dos pasos: 70% train y, del 30% restante, 20% test y 10% validación.
3. **Entrenamiento** — dos algoritmos:
   - *Regresión Lineal Múltiple* (interpretable, sirve para leer la influencia de cada canal).
   - *KNN* (con `StandardScaler`; el mejor `k` se elige por validación cruzada).
4. **Evaluación** — **R²** (variabilidad explicada) y **RMSE** (error típico), en los tres conjuntos.
5. **Influencia de canales** — coeficientes estandarizados para comparar el peso de cada medio.
6. **Optimización** — reparto óptimo de un presupuesto fijo (SLSQP) frente al gasto medio histórico.

## Resultados (semilla = 42)

Resultados reales obtenidos al ejecutar `python main.py`.

### Exploración de los datos
- **200 registros, 0 valores nulos** → dataset limpio.
- **Correlación de cada canal con las ventas:** TV **0.782** · Radio **0.576** · Newspaper **0.228**.

### Modelo de Regresión Lineal (interpretable)
```
Ventas = 2.71 + 0.044·TV + 0.199·Radio + 0.007·Newspaper
```
El coeficiente casi nulo de `Newspaper` ya anticipa su escasa influencia.
El KNN eligió **k = 4** por validación cruzada.

### Evaluación — 2 métricas sobre el conjunto de VALIDACIÓN
El conjunto de validación (10%, datos nunca vistos) es el juez final para elegir el modelo.

| Modelo | R² ↑ | RMSE ↓ |
|--------|------|--------|
| **KNN** 🏆 | **0.8763** | **1.9564** |
| Regresión Lineal | 0.7881 | 2.5608 |

> **Modelo ganador: KNN** — explica el **87.6 %** de la variabilidad de las ventas con un error
> típico de solo **~1.96** unidades, superando a la Regresión Lineal en ambas métricas.

### Influencia de cada canal (coeficientes estandarizados)
| Canal | Coef. estandarizado | Importancia relativa |
|-------|:-------------------:|:--------------------:|
| **TV** | +3.92 | **58.2 %** |
| **Radio** | +2.79 | **41.5 %** |
| Newspaper | −0.02 | 0.3 % |

> **TV** y **Radio** concentran ~99.7 % del impacto en las ventas; **Newspaper** es
> prácticamente irrelevante (influencia casi nula).

### Optimización del presupuesto (~201 mil $)
| Canal | Asignación óptima | Media histórica |
|-------|:-----------------:|:---------------:|
| **TV** | **75.2 %** (151.0) | 147.0 |
| **Radio** | **24.7 %** (49.6) | 23.3 |
| Newspaper | 0.1 % (0.3) | 30.6 |

- Ventas estimadas con reparto **óptimo**: **19.25** mil unidades
- Ventas estimadas con reparto **medio histórico**: 14.03 mil unidades
- **📈 Mejora: +37.1 %** solo reasignando el presupuesto hacia TV y Radio.

### Conclusión
1. Las ventas se predicen con alta precisión (**KNN, R² ≈ 0.88**).
2. **TV y Radio** son los motores de las ventas; **Newspaper no aporta**.
3. Reasignar el presupuesto hacia los canales rentables eleva las ventas estimadas **~37 %**.

*Nota: los valores pueden variar ligeramente si se cambia `RANDOM_STATE` en `src/config.py`.*
