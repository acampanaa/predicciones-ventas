# -*- coding: utf-8 -*-
"""Definición y entrenamiento de los 2 modelos (2 algoritmos diferentes)."""
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

from src import config


def entrenar_regresion_lineal(X_train, y_train):
    """Modelo A: Regresión Lineal Múltiple (interpretable)."""
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    print(">> Modelo A: Regresión Lineal Múltiple")
    print(f"   Intercepto (β0): {modelo.intercept_:.4f}")
    for col, coef in zip(config.FEATURES, modelo.coef_):
        print(f"     {col:10s}: {coef:+.4f}")
    return modelo


def entrenar_knn(X_train, y_train):
    """
    Modelo B: K-Vecinos más Cercanos (KNN).
    KNN se basa en distancias -> es imprescindible estandarizar las variables.
    El mejor k se elige por validación cruzada.
    """
    pipe = make_pipeline(StandardScaler(), KNeighborsRegressor())
    grid = GridSearchCV(
        pipe,
        param_grid={"kneighborsregressor__n_neighbors": range(3, 16)},
        cv=5, scoring="r2")
    grid.fit(X_train, y_train)
    mejor_k = grid.best_params_["kneighborsregressor__n_neighbors"]
    print(">> Modelo B: KNN (StandardScaler + KNeighborsRegressor)")
    print(f"   Mejor k por validación cruzada: k = {mejor_k}")
    return grid.best_estimator_


def entrenar_modelos(X_train, y_train):
    """Entrena ambos modelos y los devuelve en un diccionario."""
    return {
        "Regresión Lineal": entrenar_regresion_lineal(X_train, y_train),
        "KNN": entrenar_knn(X_train, y_train),
    }
