# Predicción de Ventas según la Inversión Publicitaria

Proyecto de Machine Learning que predice las **ventas** de una empresa a partir de
su inversión publicitaria en **TV, Radio y Periódicos** (dataset *Advertising*).

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
- **Validación:** KNN R²≈0.88 / RMSE≈1.96 · Regresión Lineal R²≈0.79 / RMSE≈2.56.
- **Influencia:** **TV** y **Radio** impulsan las ventas; **Newspaper** es prácticamente irrelevante.
- **Optimización:** reasignar el presupuesto hacia TV y Radio eleva las ventas estimadas
  **~37%** frente al gasto medio histórico.
