# -*- coding: utf-8 -*-
"""
Objetivos 2 y 3:
  - Influencia de cada canal (coeficientes estandarizados del modelo lineal).
  - Optimización de la asignación del presupuesto para maximizar las ventas.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

from src import config


# ---------------------------------------------------------------------------
# Influencia de cada canal
# ---------------------------------------------------------------------------
def influencia_canales(df):
    """Compara la importancia relativa de cada canal usando coeficientes
    estandarizados (todas las variables en la misma escala)."""
    X, y = df[config.FEATURES], df[config.TARGET]
    X_std = StandardScaler().fit_transform(X)
    modelo_std = LinearRegression().fit(X_std, y)

    imp = pd.DataFrame({
        "Canal": config.FEATURES,
        "Coef. estandarizado": modelo_std.coef_,
        "Importancia relativa": np.abs(modelo_std.coef_) / np.abs(modelo_std.coef_).sum(),
    }).sort_values("Coef. estandarizado", ascending=False)

    print(imp.to_string(index=False, formatters={
        "Coef. estandarizado": "{:+.4f}".format,
        "Importancia relativa": "{:.1%}".format}))
    orden = imp["Canal"].tolist()
    print("\nConclusión:")
    print(f"  • {orden[0]} es el canal MÁS influyente en las ventas.")
    print(f"  • {orden[-1]} es el de MENOR (o nula) influencia.")

    _fig_influencia(imp)
    return imp


def _fig_influencia(imp):
    plt.figure(figsize=(7, 4.5))
    colores = ["#2b8cbe" if c >= 0 else "#d7301f" for c in imp["Coef. estandarizado"]]
    plt.barh(imp["Canal"], imp["Coef. estandarizado"], color=colores)
    plt.axvline(0, color="k", lw=.8)
    plt.xlabel("Coeficiente estandarizado")
    plt.title("Influencia de cada canal sobre las ventas")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(config.FIG_DIR / "05_influencia_canales.png", dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Optimización del presupuesto
# ---------------------------------------------------------------------------
def optimizar_presupuesto(modelo_lr, df, presupuesto=None):
    """Reparte un presupuesto fijo entre canales para maximizar las ventas
    predichas por el modelo lineal, con límites realistas por canal."""
    if presupuesto is None:
        presupuesto = df[config.FEATURES].sum(axis=1).mean()
    print(f"Presupuesto total a repartir: {presupuesto:.1f} mil $")

    limites = [(df[c].min(), df[c].max()) for c in config.FEATURES]
    b0, coefs = modelo_lr.intercept_, modelo_lr.coef_

    # Maximizar ventas = minimizar (-ventas), sujeto a que la suma = presupuesto.
    def ventas_negativas(x):
        return -(b0 + np.dot(coefs, x))

    restriccion = {"type": "eq", "fun": lambda x: np.sum(x) - presupuesto}
    x0 = np.full(len(config.FEATURES), presupuesto / len(config.FEATURES))
    res = minimize(ventas_negativas, x0, method="SLSQP",
                   bounds=limites, constraints=restriccion)

    asignacion = res.x
    ventas_opt = b0 + np.dot(coefs, asignacion)

    # Línea base realista: la asignación MEDIA histórica (mismo presupuesto y
    # dentro de los límites observados), no un reparto uniforme inviable.
    asignacion_media = df[config.FEATURES].mean().values
    ventas_media = b0 + np.dot(coefs, asignacion_media)

    tabla = pd.DataFrame({
        "Canal": config.FEATURES,
        "Asignación óptima (mil $)": asignacion,
        "% del presupuesto": asignacion / presupuesto,
        "Asignación media histórica (mil $)": asignacion_media,
    })
    print("\n" + tabla.to_string(index=False, formatters={
        "Asignación óptima (mil $)": "{:.2f}".format,
        "% del presupuesto": "{:.1%}".format,
        "Asignación media histórica (mil $)": "{:.2f}".format}))

    print(f"\nVentas estimadas — asignación ÓPTIMA           : {ventas_opt:.2f} mil unidades")
    print(f"Ventas estimadas — asignación MEDIA histórica  : {ventas_media:.2f} mil unidades")
    mejora = (ventas_opt - ventas_media) / ventas_media
    print(f"Mejora frente a la asignación media            : {mejora:+.1%}")
    print("\nNota: al ser lineal el modelo, el óptimo concentra la inversión en los")
    print("canales de mayor impacto por dólar (Radio y TV) y minimiza Newspaper.")

    _fig_optimizacion(asignacion, asignacion_media)
    return tabla


def _fig_optimizacion(asignacion, asignacion_media):
    plt.figure(figsize=(7, 4.5))
    x = np.arange(len(config.FEATURES))
    w = 0.38
    plt.bar(x - w / 2, asignacion, w, label="Óptima", color="#2b8cbe")
    plt.bar(x + w / 2, asignacion_media, w, label="Media histórica", color="#bdbdbd")
    plt.xticks(x, config.FEATURES)
    plt.ylabel("Inversión (mil $)")
    plt.title("Asignación de presupuesto: óptima vs uniforme")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.FIG_DIR / "06_optimizacion_presupuesto.png", dpi=120)
    plt.close()
