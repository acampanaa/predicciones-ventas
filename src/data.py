# -*- coding: utf-8 -*-
"""Carga del dataset y división 70/20/10 (entrenamiento / test / validación)."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def cargar_datos(ruta=config.DATA_RAW):
    """Lee el CSV y elimina la columna índice sin nombre si existe."""
    df = pd.read_csv(ruta)
    if df.columns[0].strip() in ("", "Unnamed: 0"):
        df = df.drop(columns=df.columns[0])
    return df


def dividir_datos(df, verbose=True):
    """
    Divide el dataset en 70% train / 20% test / 10% validación.

    Devuelve tres tuplas: (X_train, y_train), (X_test, y_test), (X_val, y_val).
    """
    X = df[config.FEATURES]
    y = df[config.TARGET]

    # Paso 1: 70% entrenamiento y 30% temporal.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=config.TRAIN_SIZE, random_state=config.RANDOM_STATE)

    # Paso 2: del 30% temporal -> 20% test y 10% validación (20/30 = 2/3 para test).
    proporcion_test = config.TEST_SIZE / (config.TEST_SIZE + config.VAL_SIZE)
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, train_size=proporcion_test, random_state=config.RANDOM_STATE)

    if verbose:
        total = len(df)
        print(f"Entrenamiento : {len(X_train):3d} muestras ({len(X_train)/total:.0%})")
        print(f"Test          : {len(X_test):3d} muestras ({len(X_test)/total:.0%})")
        print(f"Validación    : {len(X_val):3d} muestras ({len(X_val)/total:.0%})")

    return (X_train, y_train), (X_test, y_test), (X_val, y_val)


def guardar_division(df, destino=config.DATA_PROC, verbose=True):
    """
    Divide el dataset y escribe 3 CSV en disco: train.csv, test.csv y validation.csv.

    Cada CSV contiene las columnas completas (predictoras + objetivo).
    Devuelve un diccionario {nombre: ruta} con las rutas generadas.
    """
    destino.mkdir(parents=True, exist_ok=True)
    (X_train, y_train), (X_test, y_test), (X_val, y_val) = dividir_datos(df, verbose=verbose)

    subconjuntos = {
        "train": (X_train, y_train),
        "test": (X_test, y_test),
        "validation": (X_val, y_val),
    }

    rutas = {}
    for nombre, (X, y) in subconjuntos.items():
        ruta = destino / f"{nombre}.csv"
        # Reunimos predictoras + objetivo conservando el orden original de filas.
        X.assign(**{config.TARGET: y}).to_csv(ruta, index=False)
        rutas[nombre] = ruta
        if verbose:
            print(f"  -> {ruta.name:15s} ({len(X)} filas) guardado en {ruta}")

    return rutas
