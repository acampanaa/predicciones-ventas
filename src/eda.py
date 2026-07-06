# -*- coding: utf-8 -*-
"""Análisis exploratorio de datos (EDA) y figuras descriptivas."""
import matplotlib
matplotlib.use("Agg")  # backend sin ventana, para guardar figuras
import matplotlib.pyplot as plt
import seaborn as sns

from src import config

sns.set_theme(style="whitegrid")


def explorar(df):
    """Resumen estadístico + figuras de correlación y dispersión."""
    print(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"Valores nulos totales: {int(df.isnull().sum().sum())}")
    print("\nEstadísticas descriptivas:")
    print(df.describe().round(2).to_string())

    corr = (df.corr(numeric_only=True)[config.TARGET]
            .drop(config.TARGET).sort_values(ascending=False))
    print("\nCorrelación de cada canal con las ventas (Sales):")
    print(corr.round(3).to_string())

    _fig_correlacion(df)
    _fig_dispersion(df)
    print(f"\n[Figuras guardadas en {config.FIG_DIR}]")


def _fig_correlacion(df):
    plt.figure(figsize=(6, 5))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm",
                fmt=".2f", square=True, cbar_kws={"shrink": .8})
    plt.title("Matriz de correlación")
    plt.tight_layout()
    plt.savefig(config.FIG_DIR / "01_correlacion.png", dpi=120)
    plt.close()


def _fig_dispersion(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col in zip(axes, config.FEATURES):
        sns.regplot(data=df, x=col, y=config.TARGET, ax=ax,
                    scatter_kws={"alpha": .5, "s": 25},
                    line_kws={"color": "red"})
        ax.set_title(f"{col} vs {config.TARGET}")
    plt.suptitle("Relación entre inversión por canal y ventas", y=1.02)
    plt.tight_layout()
    plt.savefig(config.FIG_DIR / "02_dispersion.png", dpi=120, bbox_inches="tight")
    plt.close()
