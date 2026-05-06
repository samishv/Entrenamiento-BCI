import os
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

# Bandas de frecuencia 
BANDAS = {
    "delta":  (0,   4),
    "theta":  (4,   8),
    "mu":     (8,  13),
    "beta":   (13, 30),
    "gamma":  (30, 80),
    "Otros":  (80,1000)#límite superior de gamma es de 80
}
CONDICION_MAP = {
    'OV': 'Ojos Abiertos - Movimiento Visual',
    'OK': 'Ojos Abiertos - Movimiento Cinestésico',
    'CK': 'Ojos Cerrados - Movimiento Cinestésico',
    'CV': 'Ojos Cerrados - Movimiento Visual',
}

ESTADO_MAP = {
    'ME': 'Movimiento Ejecutado',
    'MI': 'Movimiento Imaginado',
    'RE': 'Reposo',
}




data = np.load(r"C:\Users\lrl13\OneDrive - Instituto Politecnico Nacional\10\reportes_CMC\Reporte_GENERAL.npy",allow_pickle=True).item()
diccionario1 = data["S18"]

lote_datos = []

bandas_analisis = ["mu","beta","gamma"]
resultados = {}

freqs = np.linspace(0,125,126)
mask_excluye = np.zeros(len(freqs),dtype = bool)
fl, fh = BANDAS["mu"]
mask_excluye = (freqs >= fl) & (freqs <= fh)
mascara_mu = mask_excluye[:, np.newaxis, np.newaxis]

mask_excluye = np.zeros(len(freqs),dtype = bool)
fl, fh = BANDAS["beta"]
mask_excluye = (freqs >= fl) & (freqs <= fh)
mascara_beta = mask_excluye[:, np.newaxis, np.newaxis]

mask_excluye = np.zeros(len(freqs),dtype = bool)
fl, fh = BANDAS["gamma"]
mask_excluye = (freqs >= fl) & (freqs <= fh)
mascara_gamma = mask_excluye[:, np.newaxis, np.newaxis]

for sujeto in list(data.keys()):

    resultados[sujeto]={}
    freqs = data[sujeto][0]['freqs']
    mask_excluye = np.zeros(len(freqs),dtype = bool)

    for sesion in list(data[sujeto].keys()):
        resultados[sujeto][sesion] = {}


        coherencia = data[sujeto][sesion]['coherencia'] 

        coherencia_mu = coherencia*mascara_mu
        coherencia_beta = coherencia*mascara_beta
        coherencia_gamma = coherencia*mascara_gamma


        datos = {
            "sujeto" : sujeto,
            "sesion" : sesion,
            #"banda" : banda,
            "Max Mu" : np.array(coherencia_mu.max(axis=0)),
            "Max Beta" : np.array(coherencia_beta.max(axis=0)),
            "Max Gamma" : np.array(coherencia_gamma.max(axis=0)),
            "Area Mu" : np.trapezoid(coherencia_mu,axis = 0),
            "Area Beta" : np.trapezoid(coherencia_beta,axis = 0),
            "Area Gamma" : np.trapezoid(coherencia_gamma,axis = 0),
            "Promedio Mu" : coherencia_mu.mean(axis = 0),
            "Promedio Beta" : coherencia_beta.mean(axis = 0),
            "Promedio Gamma" : coherencia_gamma.mean(axis = 0)
                }
        lote_datos.append(datos)

print(len(lote_datos))
print(lote_datos[0])
    # print("\n CLAVES DEL SUJETO : \n")
    # print(list(data[sujeto].keys()))
    #print(sujeto)
EMG_NAMES = ["Flexor carpi ulnaris","Right Extensor carpi radialis longus","Extensor carpi ulnaris"]
EEG_NAMES = ["Fz","FC3","FCz","FC4","Cz","C3","C4"]
sujetos = ["S5","S7","S17","S18"]


filas_finales = []

for item in lote_datos:
    m = item["Max Mu"]
    b = item["Max Beta"]
    g = item["Max Gamma"]
    m_av = item["Promedio Mu"]
    b_av = item["Promedio Beta"]
    g_av = item["Promedio Gamma"]
    for i, musculo in enumerate(EMG_NAMES):
        for j, canal in enumerate(EEG_NAMES):
            filas_finales.append({
                "sujeto": item["sujeto"],
                "sesión": item["sesion"],
                "músculo": musculo,
                "canal eeg": canal,
                "Max Mu" : m[i, j],
                "Max Beta" : b[i, j],
                "Max Gamma" : g[i, j],
                "Area Mu" : item["Area Mu"][i][j],
                "Area Beta" : item["Area Beta"][i][j],
                "Area Gamma" : item["Area Gamma"][i][j],
                "Promedio Mu": m_av[i][j],
                "Promedio Beta": b_av[i][j],
                "Promedio Gamma": g_av[i][j]
            })

#  Crear DataFrame y guardar
df = pd.DataFrame(filas_finales)
df.to_excel("Consolidado_EEG.xlsx", index=False)

print(f"Procesadas {len(lote_datos)} matrices. Total de filas: {len(df)}")

