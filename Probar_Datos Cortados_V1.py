# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 01:04:03 2026

@author: luiso
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# ── Cargar archivos ───────────────────────────────────────────────────────────
eeg  = np.load(r"D:\luiso\Descargas\S17\S17\EEG\S17_EEG_260310_ME_1.npy")
emg  = np.load(r"D:\luiso\Descargas\S17\S17\EMG\S17_CK_EMG_260310_1.npy")
din  = np.load(r"D:\luiso\Descargas\S17\S17\DINAM\S17_DINAM_100326_ME_1.npy")

# ── Verificar que cargaron bien ───────────────────────────────────────────────
print(f"EEG  → {eeg.shape}")   # (7, 7000)
print(f"EMG  → {emg.shape}")   # (n_canales, 28000)
print(f"DIN  → {din.shape}")   # (n_muestras,)

# ── Ejes de tiempo ────────────────────────────────────────────────────────────
t_eeg = np.linspace(0, 28, eeg.shape[1])
t_emg = np.linspace(0, 28, emg.shape[1])
t_din = np.linspace(0, 28, len(din))

# ── Figura 1 - EEG ────────────────────────────────────────────────────────────
canales_eeg = ['Fz', 'FC3', 'FCz', 'FC4', 'Cz', 'C3', 'C4']
colores_eeg = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

fig1, axes = plt.subplots(7, 1, figsize=(14, 12), sharex=True)
fig1.suptitle('Señales EEG', fontsize=13, fontweight='bold')
for ax, canal, señal, color in zip(axes, canales_eeg, eeg, colores_eeg):
    ax.plot(t_eeg, señal, color=color, linewidth=0.6)
    ax.set_ylabel(canal, fontsize=9, rotation=0, labelpad=35)
    ax.grid(True, alpha=0.3, linestyle='--')
axes[-1].set_xlabel('Tiempo (s)')
plt.tight_layout()

# ── Figura 2 - EMG ────────────────────────────────────────────────────────────
canales_emg = ['Braquiorradial', 'Extensor radial largo del carpo', 'Palmar largo', 'Extensor cubital del carpo']
colores_emg = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

fig2, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
fig2.suptitle('Señales EMG', fontsize=13, fontweight='bold')
for ax, canal, señal, color in zip(axes, canales_emg, emg, colores_emg):
    ax.plot(t_emg, señal, color=color, linewidth=0.6)
    ax.set_ylabel(canal, fontsize=9, rotation=0, labelpad=55)
    ax.grid(True, alpha=0.3, linestyle='--')
axes[-1].set_xlabel('Tiempo (s)')
plt.tight_layout()

# ── Figura 3 - Dinamómetro ────────────────────────────────────────────────────
fig3, ax = plt.subplots(figsize=(14, 3))
fig3.suptitle('Señal Dinamómetro', fontsize=13, fontweight='bold')
ax.plot(t_din, din, color='steelblue', linewidth=0.8)
ax.set_ylabel('Fuerza (N)')
ax.set_xlabel('Tiempo (s)')
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

plt.show()