# -*- coding: utf-8 -*-
"""Evaluación de los modelos con 2 métricas: R² y RMSE (+ MAE de apoyo)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src import config


def metricas(y_true, y_pred):
    """Devuelve (R², RMSE, MAE)."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae


def evaluar_modelos(modelos, train, test, val):
    """Calcula las métricas de cada modelo en los 3 conjuntos y genera figuras."""
    conjuntos = {"Entrenamiento": train, "Test": test, "Validación": val}

    filas = []
    for nombre, modelo in modelos.items():
        for cjto, (X, y) in conjuntos.items():
            r2, rmse, mae = metricas(y, modelo.predict(X))
            filas.append([nombre, cjto, r2, rmse, mae])

    tabla = pd.DataFrame(filas, columns=["Modelo", "Conjunto", "R²", "RMSE", "MAE"])
    print(tabla.to_string(index=False, formatters={
        "R²": "{:.4f}".format, "RMSE": "{:.4f}".format, "MAE": "{:.4f}".format}))

    # Comparación sobre VALIDACIÓN (las 2 métricas pedidas)
    comp = (tabla[tabla["Conjunto"] == "Validación"]
            .set_index("Modelo")[["R²", "RMSE"]])
    print("\n--- Comparación sobre el conjunto de VALIDACIÓN ---")
    print(comp.round(4).to_string())

    mejor = comp["R²"].idxmax()
    print(f"\n>> Mejor modelo según R² en validación: {mejor} "
          f"(R²={comp.loc[mejor, 'R²']:.4f}, RMSE={comp.loc[mejor, 'RMSE']:.4f})")

    _fig_comparacion(comp)
    _fig_pred_vs_real(modelos, val)
    return tabla, mejor


def _fig_comparacion(comp):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    comp["R²"].plot.bar(ax=axes[0], color="#2b8cbe", rot=0)
    axes[0].set_title("R² en Validación (mayor = mejor)")
    axes[0].set_ylim(0, 1)
    comp["RMSE"].plot.bar(ax=axes[1], color="#d7301f", rot=0)
    axes[1].set_title("RMSE en Validación (menor = mejor)")
    plt.suptitle("Comparación de modelos", y=1.03)
    plt.tight_layout()
    plt.savefig(config.FIG_DIR / "03_comparacion_modelos.png",
                dpi=120, bbox_inches="tight")
    plt.close()


def _fig_pred_vs_real(modelos, val):
    X_val, y_val = val
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, (nombre, modelo) in zip(axes, modelos.items()):
        y_pred = modelo.predict(X_val)
        r2, rmse, _ = metricas(y_val, y_pred)
        ax.scatter(y_val, y_pred, alpha=.7, edgecolor="k")
        lim = [y_val.min() - 1, y_val.max() + 1]
        ax.plot(lim, lim, "r--")
        ax.set_xlabel("Ventas reales")
        ax.set_ylabel("Ventas predichas")
        ax.set_title(f"{nombre}\nR²={r2:.3f} | RMSE={rmse:.3f}")
    plt.suptitle("Validación: Real vs Predicho", y=1.02)
    plt.tight_layout()
    plt.savefig(config.FIG_DIR / "04_pred_vs_real.png", dpi=120, bbox_inches="tight")
    plt.close()
