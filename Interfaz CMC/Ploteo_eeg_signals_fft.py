# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 23:23:24 2026

@author: luiso
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos, welch

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
RUTA_CSV = r"EEG_RAW_04_03_2026__16_14_01.csv"

FS = 250  # Frecuencia de muestreo en Hz

COLUMNAS_IDX = [1, 2, 3, 4, 5, 6, 8]
LETRAS       = ['B', 'C', 'D', 'E', 'F', 'G', 'I']
CANALES      = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'PO8']

COLORES = ['#e63946', '#2a9d8f', '#e9c46a', '#f4a261',
           '#457b9d', '#a8dadc', '#6d6875']

# Bandas EEG
BANDAS = {
    'δ\n(0.5–4)'  : (0.5,  4,  '#4cc9f0'),
    'θ\n(4–8)'    : (4,    8,  '#7209b7'),
    'α\n(8–13)'   : (8,   13,  '#f72585'),
    'β\n(13–30)'  : (13,  30,  '#fb8500'),
    'γ\n(30–45)'  : (30,  45,  '#06d6a0'),
}

# Parámetros de filtros
F_HIGH  = 0.5
F_LOW   = 50.0
F_NOTCH = 60.0
Q_NOTCH = 30
ORDEN   = 4

# Resolución Welch: FS / nperseg = Hz por barra
# Ejemplo: nperseg=500 → 0.5 Hz/barra | nperseg=250 → 1 Hz/barra
nperseg = 1000

# ─────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────
print("Cargando archivo CSV...")
plt.close('all')
df = pd.read_csv(RUTA_CSV, header=0)

datos = df.iloc[:, COLUMNAS_IDX].values.astype(float)
n_muestras, n_canales = datos.shape
tiempo = np.arange(n_muestras) / FS

print(f"  • Muestras totales : {n_muestras}")
print(f"  • Duración         : {n_muestras / FS:.2f} s")
print(f"  • Canales cargados : {n_canales} ({', '.join(CANALES)})")

# ─────────────────────────────────────────────
# FILTROS
# ─────────────────────────────────────────────
b_hp, a_hp = butter(ORDEN, F_HIGH  / (FS / 2), btype='high')
b_lp, a_lp = butter(ORDEN, F_LOW   / (FS / 2), btype='low')
b_n,  a_n  = iirnotch(F_NOTCH, Q_NOTCH, FS)
sos_hp    = tf2sos(b_hp, a_hp)
sos_lp    = tf2sos(b_lp, a_lp)
sos_notch = tf2sos(b_n,  a_n)

datos_filt = np.zeros_like(datos)
for i in range(n_canales):
    sig = sosfiltfilt(sos_hp,    datos[:, i])
    sig = sosfiltfilt(sos_lp,    sig)
    sig = sosfiltfilt(sos_notch, sig)
    datos_filt[:, i] = sig

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

def psd_canal(señal):
    freqs_w, psd = welch(señal, fs=FS, nperseg=nperseg)
    mask = (freqs_w >= 0.3) & (freqs_w <= 65)
    return freqs_w[mask], psd[mask]

def plot_bandas(ax, psd, etiquetas=False):
    for nombre, (f_ini, f_fin, bc) in BANDAS.items():
        ax.axvspan(f_ini, f_fin, alpha=0.10, color=bc, zorder=0)
        if etiquetas:
            ax.text((f_ini + f_fin) / 2, psd.max() * 1.08,
                    nombre, color=bc, fontsize=6.5, ha='center', va='bottom')

# ─────────────────────────────────────────────
# FIGURA 1 – Señales crudas
# ─────────────────────────────────────────────
fig1, axes1 = plt.subplots(n_canales, 1, figsize=(19, 9), sharex=True, num=1)
estilo_fig(fig1)

for i, (ax, canal, color) in enumerate(zip(axes1, CANALES, COLORES)):
    estilo_ax(ax)
    ax.plot(tiempo, datos[:, i], color=color, linewidth=0.6, alpha=0.9)
    ax.set_ylabel(canal, color=color, fontsize=9, labelpad=4)
    ax.set_xlim([tiempo[0], tiempo[-1]])
    ax.autoscale(axis='y', tight=True)
    ax.grid(True, color='#21262d', linewidth=0.5, linestyle='--')

axes1[-1].set_xlabel('Tiempo (s)', color='#8b949e', fontsize=10)
fig1.suptitle('Señales EEG Crudas — FZ, C3, CZ, C4, PZ, PO7, PO8',
              color='#f0f6fc', fontsize=13, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.98])

# ─────────────────────────────────────────────
# FIGURA 2 – PSD señal cruda (Welch)
# ─────────────────────────────────────────────
fig2, axes2 = plt.subplots(n_canales, 1, figsize=(19, 9), sharex=True, num=2,
                            constrained_layout=True)
estilo_fig(fig2)

for i, (ax, canal, color) in enumerate(zip(axes2, CANALES, COLORES)):
    estilo_ax(ax)
    freqs_w, psd = psd_canal(datos[:, i])
    df_w = freqs_w[1] - freqs_w[0]
    ax.bar(freqs_w, psd, width=df_w * 0.9, color=color, alpha=0.85, linewidth=0)
    plot_bandas(ax, psd, etiquetas=(i == 0))
    ax.set_ylabel(f'{canal}\n(µV²/Hz)', color=color, fontsize=8, labelpad=4)
    ax.yaxis.get_offset_text().set_color('#8b949e')
    ax.yaxis.get_offset_text().set_fontsize(6)
    ax.set_xlim([0, 65])
    ax.autoscale(axis='y', tight=True)
    ax.set_ylim(bottom=0)
    ax.grid(True, color='#21262d', linewidth=0.5, linestyle='--', axis='y')

axes2[-1].set_xlabel('Frecuencia (Hz)', color='#8b949e', fontsize=10)
fig2.suptitle('Densidad de Potencia Espectral (Welch) — FZ, C3, CZ, C4, PZ, PO7, PO8  |  fs = 250 Hz',
              color='#f0f6fc', fontsize=13, fontweight='bold')

# ─────────────────────────────────────────────
# FIGURA 3 – Señales filtradas
# ─────────────────────────────────────────────
fig3, axes3 = plt.subplots(n_canales, 1, figsize=(19, 9), sharex=True, num=3)
estilo_fig(fig3)

for i, (ax, canal, color) in enumerate(zip(axes3, CANALES, COLORES)):
    estilo_ax(ax)
    ax.plot(tiempo, datos_filt[:, i], color=color, linewidth=0.7, alpha=0.95)
    ax.set_ylabel(canal, color=color, fontsize=9, labelpad=4)
    ax.set_xlim([tiempo[0], tiempo[-1]])
    ax.autoscale(axis='y', tight=True)
    ax.grid(True, color='#21262d', linewidth=0.5, linestyle='--')

axes3[-1].set_xlabel('Tiempo (s)', color='#8b949e', fontsize=10)
info_filtros = (f"Filtros en cadena  |  Pasa-altas: {F_HIGH} Hz  •  "
                f"Pasa-bajas: {F_LOW} Hz  •  Notch: {F_NOTCH} Hz (Q={Q_NOTCH})  •  "
                f"Butterworth orden {ORDEN}")
fig3.text(0.5, 0.005, info_filtros, ha='center', va='bottom',
          fontsize=7.5, color='#8b949e', style='italic')
fig3.suptitle('Señales EEG Filtradas (0.1–50 Hz + Notch 60 Hz) — FZ, C3, CZ, C4, PZ, PO7, PO8',
              color='#f0f6fc', fontsize=13, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0.02, 1, 0.98])

# ─────────────────────────────────────────────
# FIGURA 4 – PSD señal filtrada (Welch)
# ─────────────────────────────────────────────
fig4, axes4 = plt.subplots(n_canales, 1, figsize=(19, 9), sharex=True, num=4,
                            constrained_layout=True)
estilo_fig(fig4)

for i, (ax, canal, color) in enumerate(zip(axes4, CANALES, COLORES)):
    estilo_ax(ax)
    freqs_w, psd = psd_canal(datos_filt[:, i])
    df_w = freqs_w[1] - freqs_w[0]
    ax.bar(freqs_w, psd, width=df_w * 0.9, color=color, alpha=0.85, linewidth=0)

    ax.axvline(F_HIGH,  color='#4cc9f0', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.axvline(F_LOW,   color='#4cc9f0', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.axvline(F_NOTCH, color='#f72585', linewidth=0.8, linestyle='--', alpha=0.7)

    plot_bandas(ax, psd, etiquetas=(i == 0))

    if i == 0:
        ax.text(F_HIGH  + 0.5, psd.max() * 0.85, f'HP\n{F_HIGH}Hz',
                color='#4cc9f0', fontsize=6, va='top')
        ax.text(F_LOW   + 0.5, psd.max() * 0.85, f'LP\n{F_LOW}Hz',
                color='#4cc9f0', fontsize=6, va='top')
        ax.text(F_NOTCH + 0.5, psd.max() * 0.85, f'Notch\n{F_NOTCH}Hz',
                color='#f72585', fontsize=6, va='top')

    ax.set_ylabel(f'{canal}\n(µV²/Hz)', color=color, fontsize=8, labelpad=4)
    ax.yaxis.get_offset_text().set_color('#8b949e')
    ax.yaxis.get_offset_text().set_fontsize(6)
    ax.set_xlim([0, 65])
    ax.autoscale(axis='y', tight=True)
    ax.set_ylim(bottom=0)
    ax.grid(True, color='#21262d', linewidth=0.5, linestyle='--', axis='y')

axes4[-1].set_xlabel('Frecuencia (Hz)', color='#8b949e', fontsize=10)
fig4.suptitle('PSD Señales Filtradas (0.1–50 Hz + Notch 60 Hz) — FZ, C3, CZ, C4, PZ, PO7, PO8',
              color='#f0f6fc', fontsize=13, fontweight='bold')

print("\nFiguras generadas correctamente.")
print(f"  Figura 1 → Señales crudas")
print(f"  Figura 2 → PSD señales crudas (Welch)")
print(f"  Figura 3 → Señales filtradas  [HP:{F_HIGH}Hz | LP:{F_LOW}Hz | Notch:{F_NOTCH}Hz]")
print(f"  Figura 4 → PSD señales filtradas (Welch)")

plt.show()