# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 07:55:21 2026

@author: samishv
"""
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from eeg_uhb import EEGAcquisitionManager # pip install eeg-uhb
import numpy as np

# Configuración
CANALES = {'Fz': 0, 'C3': 1, 'Cz': 2, 'C4': 3, 'Pz': 4, 'PO7': 5, 'Oz': 6, 'PO8': 7}
VENTANA = 250      # muestras a mostrar (1 segundo a 250Hz)
REFRESCO = 50      # milisegundos entre actualizaciones

# Setup
root = tk.Tk()
root.title("EEG en tiempo real")

EEG = EEGAcquisitionManager()

# Figura con un subplot por canal
fig, axes = plt.subplots(len(CANALES), 1, figsize=(10, 8), sharex=True)
fig.tight_layout(pad=2.0)

# Crear una línea por canal
lines = {}
for ax, (nombre, idx) in zip(axes, CANALES.items()):
    line, = ax.plot([], [], lw=0.8)
    ax.set_ylim(-100, 100)
    ax.set_xlim(0, VENTANA)
    ax.set_ylabel(nombre, rotation=0, labelpad=25)
    ax.yaxis.set_tick_params(labelsize=7)
    lines[nombre] = line

axes[-1].set_xlabel("Muestras")

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def actualizar():
    if EEG.data:
        ultimos = EEG.data[-VENTANA:]
        x = np.arange(len(ultimos))

        for nombre, idx in CANALES.items():
            y = [muestra[idx] for muestra in ultimos]
            lines[nombre].set_data(x, y)

        canvas.draw()

    root.after(REFRESCO, actualizar)

def iniciar():
    EEG.start_acquisition(stream_name='UN-2023.07.40', save=True, save_path='./eeg_data/')
    actualizar()

def detener():
    EEG.stop_acquisition()

# Botones
frame = tk.Frame(root)
frame.pack(pady=5)
tk.Button(frame, text="Iniciar", command=iniciar).pack(side=tk.LEFT, padx=5)
tk.Button(frame, text="Detener", command=detener).pack(side=tk.LEFT, padx=5)

root.mainloop()


# from eeg_uhb import EEGAcquisitionManager
# import time

# if __name__=='__main__':
#     EEG = EEGAcquisitionManager()
#     start_time = time.time()
#     duration = 0.04  # segundos

#     '''
#     # Connect to any available stream without saving
#     eeg.start_acquisition(stream_name='UN-2023.07.40')  

#     # Connect to specific stream and save data
#     eeg.start_acquisition(stream_name='UN-2023.07.40', 
#                         save=True,
#                         save_path='./eeg_data/')
#     '''
    
#     # the stream_name depends on the one you choose
#     EEG.start_acquisition(stream_name='UN-2023.07.40', save=True)
#     start = time.sleep(duration)
#     print(EEG.data)
#     print(f'Length: {len(EEG.data)}')
#     EEG.stop_acquisition()
