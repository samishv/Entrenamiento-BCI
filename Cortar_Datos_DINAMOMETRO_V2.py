# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 01:02:17 2026

@author: luiso
"""

import pandas as pd
import numpy as np
import os

# ── Configuración ─────────────────────────────────────────────────────────────
RUTA_CARPETA = r"D:\luiso\Descargas\S17\S17\DINAM"
T_INICIO     = 3
T_FIN        = T_INICIO + 28

# ── Procesamiento ─────────────────────────────────────────────────────────────
archivos_csv = [f for f in os.listdir(RUTA_CARPETA) if f.endswith('.csv')]
print(f"Archivos encontrados: {len(archivos_csv)}\n")

for archivo in archivos_csv:
    ruta_csv    = os.path.join(RUTA_CARPETA, archivo)
    ruta_salida = os.path.join(RUTA_CARPETA, os.path.splitext(archivo)[0])

    datos      = pd.read_csv(ruta_csv)
    timestamps = pd.to_datetime(datos['Timestamp'], format='%H:%M:%S.%f')
    t          = (timestamps - timestamps.iloc[0]).dt.total_seconds().values

    if t[-1] < T_FIN:
        print(f"[OMITIDO]  {archivo} — duración {t[-1]:.2f}s (mínimo {T_FIN}s)")
        continue

    mascara = (t >= T_INICIO) & (t < T_FIN)
    fuerza  = datos['Fuerza_N'].values[mascara]

    np.save(ruta_salida + ".npy", fuerza)
    print(f"[OK]  {archivo} → {os.path.splitext(archivo)[0]}.npy  {fuerza.shape}")

print("\nProceso completado.")