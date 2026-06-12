"""
Calcula la frecuencia de muestreo real del dinamómetro
a partir de los timestamps guardados en el CSV.
"""

import pandas as pd
import numpy as np
import glob
import os

# ─── Configuración ────────────────────────────────────────────────────────────
CARPETA = r"C:\Users\ikerf\Desktop\Upiita\TT\Entrenamiento-BCI\resultados\todosDINAM"   # cambia a la ruta de tu carpeta de registros
# ──────────────────────────────────────────────────────────────────────────────


def analizar_archivo(filepath):
    df = pd.read_csv(filepath)

    # Parsear timestamps (formato HH:MM:SS.ffffff)
    df['t'] = pd.to_datetime(df['Timestamp'], format='%H:%M:%S.%f')
    df = df.sort_values('t').reset_index(drop=True)

    # Intervalos entre muestras en segundos
    intervalos = df['t'].diff().dt.total_seconds().dropna()

    n_muestras  = len(df)
    duracion_s  = (df['t'].iloc[-1] - df['t'].iloc[0]).total_seconds()
    fs_media    = 1 / intervalos.mean()
    fs_mediana  = 1 / intervalos.median()
    fs_global   = (n_muestras - 1) / duracion_s if duracion_s > 0 else 0
    jitter_ms   = intervalos.std() * 1000

    print(f"\n{'─'*55}")
    print(f"Archivo : {os.path.basename(filepath)}")
    print(f"Muestras: {n_muestras}  |  Duración: {duracion_s:.2f} s")
    print(f"Fs media  : {fs_media:.1f} Hz")
    print(f"Fs mediana: {fs_mediana:.1f} Hz")
    print(f"Fs global : {fs_global:.1f} Hz  (muestras-1 / duración total)")
    print(f"Jitter    : {jitter_ms:.2f} ms (desv. estándar del intervalo)")

    # Detectar huecos grandes (> 3× el intervalo mediano)
    umbral = intervalos.median() * 3
    huecos = intervalos[intervalos > umbral]
    if not huecos.empty:
        print(f"⚠  Huecos detectados (>{umbral*1000:.1f} ms): {len(huecos)}")
        for idx, val in huecos.items():
            print(f"   fila {idx}: {val*1000:.1f} ms")

    return fs_global


# ─── Buscar todos los CSV de dinamómetro ──────────────────────────────────────
patron = os.path.join(CARPETA, "**", "*DINAM*.csv")
archivos = sorted(glob.glob(patron, recursive=True))

if not archivos:
    print(f"No se encontraron archivos DINAM en: {CARPETA}")
else:
    print(f"Encontrados {len(archivos)} archivo(s):")
    resultados = {}
    for f in archivos:
        fs = analizar_archivo(f)
        resultados[os.path.basename(f)] = fs

    print(f"\n{'═'*55}")
    print("Resumen global:")
    for nombre, fs in resultados.items():
        print(f"  {nombre}: {fs:.1f} Hz")
    print(f"\nFs promedio entre archivos: {np.mean(list(resultados.values())):.1f} Hz")
