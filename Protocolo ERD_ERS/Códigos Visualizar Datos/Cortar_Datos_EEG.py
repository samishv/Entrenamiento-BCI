import pandas as pd
import numpy as np
import os

# ── Variables Locales de Configuración (Valores fijos) ────────────────────
base_dir   = r"D:\luiso\Documentos\Luis\UPIITA\TT\Entrenamiento-BCI\Protocolo ERD_ERS\Signals"
sujeto     = "SV1"
columnas   = ['Fz', 'FC3', 'FCz', 'FC4', 'Cz', 'C3', 'C4']
fs         = 250
seg_inicio = 0
seg_utiles = 21

# ── Rutas y Parámetros de Muestras ────────────────────────────────────────
ruta_carpeta = os.path.join(base_dir, sujeto)

m_inicio = fs * seg_inicio
m_fin    = m_inicio + fs * seg_utiles
m_minimo = m_fin

# ── Estructura de Columnas del CSV Original ───────────────────────────────
nombres_col = ['Timestamp', 'Fz', 'FC3', 'FCz', 'FC4', 'Cz', 'C3', 'NC', 'C4',
               'col9', 'col10', 'col11', 'col12', 'col13', 'col14', 'col15', 'col16', 'col17']

# ── Procesamiento Lineal de Archivos ──────────────────────────────────────
if os.path.exists(ruta_carpeta):
    archivos_csv = [f for f in os.listdir(ruta_carpeta) if f.endswith('.csv')]
    print(f"Sujeto: {sujeto}")
    print(f"Archivos encontrados: {len(archivos_csv)}\n")

    for archivo in archivos_csv:
        ruta_csv    = os.path.join(ruta_carpeta, archivo)
        # os.path.splitext(archivo)[0] extrae el nombre sin el '.csv'
        nombre_sin_ext = os.path.splitext(archivo)[0]
        ruta_salida    = os.path.join(ruta_carpeta, nombre_sin_ext)

        # Lectura y filtrado de columnas
        datos  = pd.read_csv(ruta_csv, header=0, names=nombres_col)
        matriz = datos[columnas].astype(float).to_numpy().T

        # Validación de longitud mínima
        if matriz.shape[1] < m_minimo:
            print(f"[OMITIDO]  {archivo} — solo tiene {matriz.shape[1]} muestras (mínimo {m_minimo})")
            continue

        # Recorte del segmento de interés
        matriz = matriz[:, m_inicio:m_fin]

        # Guardado en disco
        np.save(ruta_salida + ".npy", matriz)
        print(f"[OK]  {archivo} → {nombre_sin_ext}.npy  {matriz.shape}")

    print("\nProceso completado.")
else:
    print(f"[ERROR] La ruta no existe: {ruta_carpeta}")
