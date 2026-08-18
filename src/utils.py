
"""Utilidades de visualización compartidas por los notebooks."""
import math
import matplotlib.pyplot as plt
import seaborn as sns

# Paleta del proyecto 
PALETA = {"principal": "#3483FA", "acento": "#E4572E", "neutro": "#6C757D"}
PALETA_CLASES = {0:"#3483FA", 1: "#E4572E"}   # legítima / fraude


def set_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 110


def grid_axes(n_items, n_cols=3, alto=4, ancho=5):
    """Crea una grilla de subplots y apaga los ejes sobrantes.
    Devuelve (fig, axes) con axes ya aplanado."""
    n_rows = math.ceil(n_items / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(ancho * n_cols, alto * n_rows))
    axes = axes.flatten()
    for j in range(n_items, len(axes)):
        fig.delaxes(axes[j])
    return fig, axes