# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 20:36:14 2026

@author: luiso
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos, welch

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
RUTA_CSV = r"D:\luiso\Descargas\RegistrosCMC_pruebaleo\RegistrosCMC\PLEO_3\EEG_RAW_09_03_2026__13_50_35.csv"
FS = 250  # Frecuencia de muestreo en Hz

COLUMNAS_IDX = [1, 2, 3, 4, 5, 6, 8]
CANALES      = ['FZ', 'FC3', 'FCZ', 'FC4', 'CZ', 'C3', 'C4']

COLORES = ['#e63946', '#2a9d8f', '#e9c46a', '#f4a261',
           '#457b9d', '#a8dadc', '#6d6875']

# Bandas EEG completas
BANDAS = {
    'δ\n(0.5–4)'  : (0.5,  4,  '#4cc9f0'),
    'θ\n(4–8)'    : (4,    8,  '#7209b7'),
    'μ\n(8–13)'   : (8,   13,  '#f72585'),
    'β\n(13–30)'  : (13,  30,  '#fb8500'),
    'γ\n(30–45)'  : (30,  45,  '#06d6a0'),
}

# Bandas solo mu y beta (figuras 5 y 6)
BANDAS_MU_BETA = {
    'μ\n(8–13)'  : (8,  13, '#f72585'),
    'β\n(13–30)' : (13, 30, '#fb8500'),
}

# Parámetros de filtros generales
F_HIGH  = 0.5
F_LOW   = 45.0
F_NOTCH = 60.0
Q_NOTCH = 30
ORDEN   = 4

# Parámetros filtro pasa-banda mu + beta
F_MU_LOW  = 8.0
F_MU_HIGH = 30.0

# Resolución Welch: FS / nperseg = Hz por barra
nperseg = 500

# ─────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────
print("\nCargando archivo CSV...\n")
plt.close('all')
df = pd.read_csv(RUTA_CSV, header=0)

datos = df.iloc[:, COLUMNAS_IDX].values.astype(float)
n_muestras, n_canales = datos.shape
tiempo = np.arange(n_muestras) / FS

print(f"  • Muestras totales : {n_muestras}")
print(f"  • Duración         : {n_muestras / FS:.2f} s")
print(f"  • Canales cargados : {n_canales} ({', '.join(CANALES)})")

# ─────────────────────────────────────────────
# DISEÑO DE FILTROS
# ─────────────────────────────────────────────
b_n,  a_n  = iirnotch(F_NOTCH, Q_NOTCH, FS)
sos_notch  = tf2sos(b_n, a_n)

b_hp, a_hp = butter(ORDEN, F_HIGH / (FS / 2), btype='high')
b_lp, a_lp = butter(ORDEN, F_LOW  / (FS / 2), btype='low')
sos_hp     = tf2sos(b_hp, a_hp)
sos_lp     = tf2sos(b_lp, a_lp)

# Pasa-banda mu+beta (8–30 Hz), orden 4
b_mb, a_mb = butter(ORDEN, [F_MU_LOW / (FS / 2), F_MU_HIGH / (FS / 2)], btype='band')
sos_mb     = tf2sos(b_mb, a_mb)

# ─────────────────────────────────────────────
# APLICAR FILTROS
# ─────────────────────────────────────────────
# Solo Notch
datos_notch = np.zeros_like(datos)
for i in range(n_canales):
    datos_notch[:, i] = sosfiltfilt(sos_notch, datos[:, i])

# Notch + Pasa-altas + Pasa-bajas
datos_filt = np.zeros_like(datos)
for i in range(n_canales):
    sig = sosfiltfilt(sos_hp,    datos_notch[:, i])
    sig = sosfiltfilt(sos_lp,    sig)
    datos_filt[:, i] = sig

# Notch + Pasa-banda mu+beta (8–30 Hz)
datos_mubeta = np.zeros_like(datos)
for i in range(n_canales):
    datos_mubeta[:, i] = sosfiltfilt(sos_mb, datos_notch[:, i])

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def estilo_ax(ax):
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='#8b949e', labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_color('#30363d')
    ax.spines['bottom'].set_color('#30363d')

def estilo_fig(fig):
    fig.patch.set_facecolor('#0d1117')

def calcular_psd(señal):
    freqs_w, psd = welch(señal, fs=FS, nperseg=nperseg)
    return freqs_w, psd

def plot_bandas(ax, psd_max, bandas_dict, etiquetas=False):
    for nombre, (f_ini, f_fin, bc) in bandas_dict.items():
        ax.axvspan(f_ini, f_fin, alpha=0.10, color=bc, zorder=0)
        if etiquetas:
            ax.text((f_ini + f_fin) / 2, psd_max * 1.08,
                    nombre, color=bc, fontsize=6.5, ha='center', va='bottom')

def ylim_global(matriz):
    return matriz.min(), matriz.max()

def ylim_psd_global(matriz, fs, nperseg_):
    psd_max_global = 0
    for i in range(matriz.shape[1]):
        _, psd = welch(matriz[:, i], fs=fs, nperseg=nperseg_)
        psd_max_global = max(psd_max_global, psd.max())
    return psd_max_global

# ─────────────────────────────────────────────
# CALCULAR LÍMITES GLOBALES (escala compartida)
# ─────────────────────────────────────────────
print("\nCalculando límites globales...")

y_min_notch, y_max_notch = ylim_global(datos_notch)
y_min_filt,  y_max_filt  = ylim_global(datos_filt)
y_min_mb,    y_max_mb    = ylim_global(datos_mubeta)
margen_n  = (y_max_notch - y_min_notch) * 0.05
margen_f  = (y_max_filt  - y_min_filt)  * 0.05
margen_mb = (y_max_mb    - y_min_mb)    * 0.05

psd_max_notch  = ylim_psd_global(datos_notch,  FS, nperseg)
psd_max_filt   = ylim_psd_global(datos_filt,   FS, nperseg)
psd_max_mubeta = ylim_psd_global(datos_mubeta, FS, nperseg)

# ─────────────────────────────────────────────
# FIGURA 1 – Señal post-Notch
# ─────────────────────────────────────────────
# fig1, axes1 = plt.subplots(n_canales, 1, figsize=(19, 9), sharex=True, num=1)
# estilo_fig(fig1)

# for i, (ax, canal, color) in enumerate(zip(axes1, CANALES, COLORES)):
#     estilo_ax(ax)
#     ax.plot(tiempo, datos_notch[:, i], color=color, linewidth=0.6, alpha=0.9)
#     ax.set_ylabel(canal, color=color, fontsize=9, labelpad=4)
#     ax.set_xlim([tiempo[0], tiempo[-1]])
#     ax.autoscale(axis='y', tight=True)
#     ax.grid(True, color='#21262d', linewidth=0.5, linestyle='--')

# axes1[-1].set_xlabel('Tiempo (s)', color='#8b949e', fontsize=10)
# fig1.suptitle(f'Señales EEG — Filtro Notch {F_NOTCH} Hz (Q={Q_NOTCH})  |  FZ, FC3, FCZ, FC4, CZ, C3, C4',
#               color='#f0f6fc', fontsize=13, fontweight='bold', y=0.99)
# plt.tight_layout(rect=[0, 0, 1, 0.98])

# ─────────────────────────────────────────────
# FIGURA 2 – PSD señal post-Notch
# ─────────────────────────────────────────────
# fig2, axes2 = plt.subplots(n_canales, 1, figsize=(19, 9), sharex=True, sharey=True, num=2,
#                             constrained_layout=True)
# estilo_fig(fig2)

# for i, (ax, canal, color) in enumerate(zip(axes2, CANALES, COLORES)):
#     estilo_ax(ax)
#     freqs_w, psd = calcular_psd(datos_notch[:, i])
#     df_w = freqs_w[1] - freqs_w[0]
#     ax.bar(freqs_w, psd, width=df_w * 0.9, color=color, alpha=0.85, linewidth=0)
#     plot_bandas(ax, psd_max_notch, BANDAS, etiquetas=(i == 0))
#     ax.set_ylabel(f'{canal}\n(µV²/Hz)', color=color, fontsize=8, labelpad=4)
#     ax.yaxis.get_offset_text().set_color('#8b949e')
#     ax.yaxis.get_offset_text().set_fontsize(6)
#     ax.grid(True, color='#21262d', linewidth=0.5, linestyle='--', axis='y')

# axes2[0].set_ylim([0, psd_max_notch * 1.15])
# axes2[0].set_xlim([0, 45.5])
# axes2[-1].set_xlabel('Frecuencia (Hz)', color='#8b949e', fontsize=10)
# fig2.suptitle(f'PSD Señal post-Notch (Welch) — FZ, FC3, FCZ, FC4, CZ, C3, C4  |  fs = {FS} Hz',
#               color='#f0f6fc', fontsize=13, fontweight='bold')

# ─────────────────────────────────────────────
# FIGURA 3 – Señal con los 3 filtros
# ─────────────────────────────────────────────
fig3, axes3 = plt.subplots(n_canales, 1, figsize=(19, 9), sharex=True, sharey=True, num=3)
estilo_fig(fig3)

for i, (ax, canal, color) in enumerate(zip(axes3, CANALES, COLORES)):
    estilo_ax(ax)
    ax.plot(tiempo, datos_filt[:, i], color=color, linewidth=0.7, alpha=0.95)
    ax.set_ylabel(canal, color=color, fontsize=9, labelpad=4)
    ax.set_xlim([tiempo[0], tiempo[-1]])
    ax.grid(True, color='#21262d', linewidth=0.5, linestyle='--')

axes3[0].set_ylim([y_min_filt - margen_f, y_max_filt + margen_f])
axes3[-1].set_xlabel('Tiempo (s)', color='#8b949e', fontsize=10)
info_filtros = (f"Notch: {F_NOTCH} Hz (Q={Q_NOTCH})  •  "
                f"Pasa-altas: {F_HIGH} Hz  •  "
                f"Pasa-bajas: {F_LOW} Hz  •  "
                f"Butterworth orden {ORDEN}")
fig3.text(0.5, 0.005, info_filtros, ha='center', va='bottom',
          fontsize=7.5, color='#8b949e', style='italic')
fig3.suptitle(f'Señales EEG Filtradas ({F_HIGH}–{F_LOW} Hz + Notch {F_NOTCH} Hz) — FZ, FC3, FCZ, FC4, CZ, C3, C4',
              color='#f0f6fc', fontsize=13, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0.02, 1, 0.98])

# ─────────────────────────────────────────────
# FIGURA 4 – PSD señal con los 3 filtros
# ─────────────────────────────────────────────
fig4, axes4 = plt.subplots(n_canales, 1, figsize=(19, 9), sharex=True, sharey=True, num=4,
                            constrained_layout=True)
estilo_fig(fig4)

for i, (ax, canal, color) in enumerate(zip(axes4, CANALES, COLORES)):
    estilo_ax(ax)
    freqs_w, psd = calcular_psd(datos_filt[:, i])
    df_w = freqs_w[1] - freqs_w[0]
    ax.bar(freqs_w, psd, width=df_w * 0.9, color=color, alpha=0.85, linewidth=0)

    ax.axvline(F_HIGH, color='#4cc9f0', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.axvline(F_LOW,  color='#4cc9f0', linewidth=0.8, linestyle='--', alpha=0.7)

    plot_bandas(ax, psd_max_filt, BANDAS, etiquetas=(i == 0))

    if i == 0:
        ax.text(F_HIGH + 0.5, psd_max_filt * 0.85, f'HP\n{F_HIGH}Hz',
                color='#4cc9f0', fontsize=6, va='top')
        ax.text(F_LOW  + 0.5, psd_max_filt * 0.85, f'LP\n{F_LOW}Hz',
                color='#4cc9f0', fontsize=6, va='top')

    ax.set_ylabel(f'{canal}\n(µV²/Hz)', color=color, fontsize=8, labelpad=4)
    ax.yaxis.get_offset_text().set_color('#8b949e')
    ax.yaxis.get_offset_text().set_fontsize(6)
    ax.grid(True, color='#21262d', linewidth=0.5, linestyle='--', axis='y')

axes4[0].set_ylim([0, psd_max_filt * 1.15])
axes4[0].set_xlim([0, 45.5])
axes4[-1].set_xlabel('Frecuencia (Hz)', color='#8b949e', fontsize=10)
fig4.suptitle(f'PSD Señales Filtradas ({F_HIGH}–{F_LOW} Hz + Notch {F_NOTCH} Hz) — FZ, FC3, FCZ, FC4, CZ, C3, C4',
              color='#f0f6fc', fontsize=13, fontweight='bold')

# ─────────────────────────────────────────────
# FIGURA 5 – Ritmos Mu + Beta (8–30 Hz)
# ─────────────────────────────────────────────
fig5, axes5 = plt.subplots(n_canales, 1, figsize=(19, 9), sharex=True, sharey=True, num=5)
estilo_fig(fig5)

for i, (ax, canal, color) in enumerate(zip(axes5, CANALES, COLORES)):
    estilo_ax(ax)
    ax.plot(tiempo, datos_mubeta[:, i], color=color, linewidth=0.7, alpha=0.95)
    ax.set_ylabel(canal, color=color, fontsize=9, labelpad=4)
    ax.set_xlim([tiempo[0], tiempo[-1]])
    ax.grid(True, color='#21262d', linewidth=0.5, linestyle='--')

axes5[0].set_ylim([y_min_mb - margen_mb, y_max_mb + margen_mb])
axes5[-1].set_xlabel('Tiempo (s)', color='#8b949e', fontsize=10)
info_mubeta = (f"Notch: {F_NOTCH} Hz (Q={Q_NOTCH})  •  "
               f"Pasa-banda: {F_MU_LOW}–{F_MU_HIGH} Hz  •  "
               f"Butterworth orden {ORDEN}")
fig5.text(0.5, 0.005, info_mubeta, ha='center', va='bottom',
          fontsize=7.5, color='#8b949e', style='italic')
fig5.suptitle(f'Ritmos Mu + Beta ({F_MU_LOW}–{F_MU_HIGH} Hz) — FZ, FC3, FCZ, FC4, CZ, C3, C4',
              color='#f0f6fc', fontsize=13, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0.02, 1, 0.98])

# ─────────────────────────────────────────────
# FIGURA 6 – PSD ritmos Mu + Beta
# ─────────────────────────────────────────────
fig6, axes6 = plt.subplots(n_canales, 1, figsize=(19, 9), sharex=True, sharey=True, num=6,
                            constrained_layout=True)
estilo_fig(fig6)

for i, (ax, canal, color) in enumerate(zip(axes6, CANALES, COLORES)):
    estilo_ax(ax)
    freqs_w, psd = calcular_psd(datos_mubeta[:, i])
    df_w = freqs_w[1] - freqs_w[0]
    ax.bar(freqs_w, psd, width=df_w * 0.9, color=color, alpha=0.85, linewidth=0)

    ax.axvline(F_MU_LOW,  color='#4cc9f0', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.axvline(F_MU_HIGH, color='#4cc9f0', linewidth=0.8, linestyle='--', alpha=0.7)

    plot_bandas(ax, psd_max_mubeta, BANDAS_MU_BETA, etiquetas=(i == 0))

    if i == 0:
        ax.text(F_MU_LOW  + 0.3, psd_max_mubeta * 0.85, f'HP\n{F_MU_LOW}Hz',
                color='#4cc9f0', fontsize=6, va='top')
        ax.text(F_MU_HIGH + 0.3, psd_max_mubeta * 0.85, f'LP\n{F_MU_HIGH}Hz',
                color='#4cc9f0', fontsize=6, va='top')

    ax.set_ylabel(f'{canal}\n(µV²/Hz)', color=color, fontsize=8, labelpad=4)
    ax.yaxis.get_offset_text().set_color('#8b949e')
    ax.yaxis.get_offset_text().set_fontsize(6)
    ax.grid(True, color='#21262d', linewidth=0.5, linestyle='--', axis='y')

axes6[0].set_ylim([0, psd_max_mubeta * 1.15])
axes6[0].set_xlim([0, 45.5])
axes6[-1].set_xlabel('Frecuencia (Hz)', color='#8b949e', fontsize=10)
fig6.suptitle(f'PSD Ritmos Mu + Beta ({F_MU_LOW}–{F_MU_HIGH} Hz) — FZ, FC3, FCZ, FC4, CZ, C3, C4  |  fs = {FS} Hz',
              color='#f0f6fc', fontsize=13, fontweight='bold')

print("\nFiguras generadas correctamente.\n")
print(f"  Figura 1 → Señal post-Notch ({F_NOTCH} Hz, Q={Q_NOTCH})")
print(f"  Figura 2 → PSD señal post-Notch")
print(f"  Figura 3 → Señal filtrada completa ({F_HIGH}–{F_LOW} Hz + Notch)")
print(f"  Figura 4 → PSD señal filtrada completa")
print(f"  Figura 5 → Ritmos Mu + Beta ({F_MU_LOW}–{F_MU_HIGH} Hz)")
print(f"  Figura 6 → PSD ritmos Mu + Beta")

plt.show()