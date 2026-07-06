# -*- coding: utf-8 -*-
"""
Punto de entrada del proyecto — ejecuta todo el pipeline de principio a fin:

    1. Carga y exploración (EDA)
    2. División 70/20/10
    3. Entrenamiento de 2 modelos (Regresión Lineal + KNN)
    4. Evaluación con 2 métricas (R² y RMSE)
    5. Influencia de cada canal
    6. Optimización del presupuesto

Uso:
    python main.py
"""
import sys

# La consola de Windows usa cp1252 por defecto y no puede imprimir algunos
# símbolos (β, ², …). Forzamos UTF-8 en la salida estándar.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data import cargar_datos, dividir_datos
from src.eda import explorar
from src.models import entrenar_modelos
from src.evaluate import evaluar_modelos
from src.optimize import influencia_canales, optimizar_presupuesto


def seccion(titulo):
    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)


def main():
    # 1. Carga + EDA
    seccion("1. CARGA Y EXPLORACIÓN DE LOS DATOS (EDA)")
    df = cargar_datos()
    explorar(df)

    # 2. División 70/20/10
    seccion("2. DIVISIÓN DEL DATASET (70% Train / 20% Test / 10% Validación)")
    train, test, val = dividir_datos(df)

    # 3. Entrenamiento de los 2 modelos
    seccion("3. ENTRENAMIENTO DE 2 MODELOS (2 algoritmos diferentes)")
    modelos = entrenar_modelos(*train)

    # 4. Evaluación con 2 métricas
    seccion("4. EVALUACIÓN — 2 MÉTRICAS: R² y RMSE")
    evaluar_modelos(modelos, train, test, val)

    # 5. Influencia de cada canal
    seccion("5. INFLUENCIA DE CADA CANAL PUBLICITARIO")
    influencia_canales(df)

    # 6. Optimización del presupuesto
    seccion("6. OPTIMIZACIÓN DE LA ASIGNACIÓN DEL PRESUPUESTO")
    optimizar_presupuesto(modelos["Regresión Lineal"], df)

    seccion("PROCESO COMPLETADO — figuras en outputs/figuras/")


if __name__ == "__main__":
    main()
