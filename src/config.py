# -*- coding: utf-8 -*-
"""Configuración central del proyecto (rutas, columnas y constantes)."""
from pathlib import Path

# --- Rutas del proyecto ---
RAIZ = Path(__file__).resolve().parent.parent
DATA_RAW = RAIZ / "data" / "raw" / "advertising.csv"
FIG_DIR = RAIZ / "outputs" / "figuras"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- Variables del dataset ---
FEATURES = ["TV", "Radio", "Newspaper"]   # variables predictoras (inversión por canal)
TARGET = "Sales"                            # variable objetivo (ventas)

# --- Reproducibilidad y división 70/20/10 ---
RANDOM_STATE = 42
TRAIN_SIZE = 0.70      # 70% entrenamiento
TEST_SIZE = 0.20      # 20% test
VAL_SIZE = 0.10      # 10% validación
