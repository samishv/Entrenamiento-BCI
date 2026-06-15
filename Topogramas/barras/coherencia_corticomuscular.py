"""
Comparación de Coherencia Corticomuscular entre Sujetos
=======================================================
Genera gráficas de barras comparando métricas (Max/Area de Mu, Beta, Gamma)
entre sujetos, filtrando por un músculo y electrodo EEG específicos.

Uso:
    1. Coloca tu archivo CSV/Excel en la misma carpeta que este script.
    2. Ajusta las variables de configuración en la sección "CONFIGURACIÓN".
    3. Ejecuta: python coherencia_corticomuscular.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import sys

# ─────────────────────────────────────────────
#  CONFIGURACIÓN 
# ─────────────────────────────────────────────
ARCHIVO       = "CMC_envolvente_barras.xlsx"   # Ruta al archivo (CSV o Excel .xlsx)
MUSCULO       = "FCU"                    # Músculo a filtrar (ej. FCU, RECRL, ECU)
ELECTRODO     = "Fz"                     # Canal EEG a filtrar (ej. Fz, FC3, Cz)

# Métricas a graficar (deben coincidir con columnas del archivo)
METRICAS = ["Max_Mu", "Max_Beta", "Max_Gamma",
            "Area_Mu", "Area_Beta", "Area_Gamma"]

# Nombres legibles para las métricas (mismo orden que METRICAS)
ETIQUETAS_METRICAS = ["Máx Mu", "Máx Beta", "Máx Gamma",
                      "Área Mu", "Área Beta", "Área Gamma"]

# Columnas esperadas en el archivo
COL_SUJETO   = "Sujeto"
COL_MUSCULO  = "Musculo"
COL_CANAL    = "Canal_EEG"

# Colores por sujeto (se generan automáticamente si hay más sujetos)
PALETA_BASE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
               "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]

# ─────────────────────────────────────────────
#  CARGA DE DATOS
# ─────────────────────────────────────────────
def cargar_datos(ruta: str) -> pd.DataFrame:
    ext = os.path.splitext(ruta)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(ruta)
    elif ext == ".csv":
        # Intenta coma primero; si falla, punto y coma
        try:
            df = pd.read_csv(ruta, sep=",")
            if df.shape[1] < 3:
                df = pd.read_csv(ruta, sep=";")
        except Exception:
            df = pd.read_csv(ruta, sep=";")
        return df
    else:
        sys.exit(f"❌ Formato no soportado: '{ext}'. Usa .csv o .xlsx")


def validar_columnas(df: pd.DataFrame):
    requeridas = {COL_SUJETO, COL_MUSCULO, COL_CANAL} | set(METRICAS)
    faltantes = requeridas - set(df.columns)
    if faltantes:
        sys.exit(f"❌ Columnas faltantes en el archivo: {faltantes}\n"
                 f"   Columnas disponibles: {list(df.columns)}")


# ─────────────────────────────────────────────
#  FILTRADO
# ─────────────────────────────────────────────
def filtrar(df: pd.DataFrame, musculo: str, electrodo: str) -> pd.DataFrame:
    mask = (df[COL_MUSCULO].str.strip() == musculo.strip()) & \
           (df[COL_CANAL].str.strip()   == electrodo.strip())
    subset = df[mask].copy()

    if subset.empty:
        musculos_disp  = sorted(df[COL_MUSCULO].unique())
        electrodos_disp = sorted(df[COL_CANAL].unique())
        sys.exit(
            f"❌ No se encontraron datos para Músculo='{musculo}' / Electrodo='{electrodo}'.\n"
            f"   Músculos disponibles:  {musculos_disp}\n"
            f"   Electrodos disponibles: {electrodos_disp}"
        )
    return subset


# ─────────────────────────────────────────────
#  GRÁFICA
# ─────────────────────────────────────────────
def graficar(df_filtrado: pd.DataFrame, musculo: str, electrodo: str):
    sujetos = sorted(df_filtrado[COL_SUJETO].unique())
    n_sujetos  = len(sujetos)
    n_metricas = len(METRICAS)

    # Extiende la paleta si hay más sujetos que colores base
    colores = (PALETA_BASE * ((n_sujetos // len(PALETA_BASE)) + 1))[:n_sujetos]

    x        = np.arange(n_metricas)          # posición central de cada grupo
    ancho    = 0.75 / n_sujetos               # ancho de cada barra
    offsets  = np.linspace(-(n_sujetos - 1) / 2,
                            (n_sujetos - 1) / 2,
                            n_sujetos) * ancho

    fig, ax = plt.subplots(figsize=(max(10, n_metricas * 1.6), 6))

    for idx, sujeto in enumerate(sujetos):
        fila = df_filtrado[df_filtrado[COL_SUJETO] == sujeto]
        if fila.empty:
            continue
        valores = fila[METRICAS].values.flatten()
        bars = ax.bar(x + offsets[idx], valores,
                      width=ancho * 0.9,
                      color=colores[idx],
                      label=str(sujeto),
                      zorder=3)

        # Etiqueta de valor encima de cada barra
        for bar, val in zip(bars, valores):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + ax.get_ylim()[1] * 0.005,
                        f"{val:.3f}",
                        ha="center", va="bottom",
                        fontsize=7, rotation=45, color="#333333")

    # Estética
    ax.set_xticks(x)
    ax.set_xticklabels(ETIQUETAS_METRICAS, fontsize=11)
    ax.set_ylabel("Valor de Coherencia", fontsize=12)
    ax.set_xlabel("Métrica", fontsize=12)
    ax.set_title(
        f"Comparación entre Sujetos\nMúsculo: {musculo}  |  Electrodo EEG: {electrodo}",
        fontsize=13, fontweight="bold", pad=14
    )
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(title="Sujeto", bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=10, title_fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    nombre_salida = f"comparacion_{musculo}_{electrodo}.png"
    plt.savefig(nombre_salida, dpi=150, bbox_inches="tight")
    print(f"✅ Gráfica guardada como '{nombre_salida}'")
    plt.show()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print(f"📂 Cargando archivo: {ARCHIVO}")
    df = cargar_datos(ARCHIVO)
    validar_columnas(df)

    print(f"🔍 Filtrando → Músculo: {MUSCULO}, Electrodo: {ELECTRODO}")
    df_sub = filtrar(df, MUSCULO, ELECTRODO)

    sujetos = sorted(df_sub[COL_SUJETO].unique())
    print(f"👥 Sujetos encontrados: {sujetos}")
    print(df_sub[[COL_SUJETO] + METRICAS].to_string(index=False))

    graficar(df_sub, MUSCULO, ELECTRODO)


if __name__ == "__main__":
    main()
