# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 07:55:21 2026

@author: samishv
"""
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from eeg_uhb import EEGAcquisitionManager
from scipy.signal import butter, filtfilt
import numpy as np
import queue

# Subclase que intercepta los datos
class EEGConGrafica(EEGAcquisitionManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grafica_queue = queue.Queue()
    
    def _acquisition_loop(self):
        try:
            while self.running:
                sample, timestamp = self.stream_inlet.pull_sample(timeout=0.1)
                if sample:
                    self.data.append(sample)
                    self.timestamps.append(timestamp)
                    self.data_queue.put((sample, timestamp))
                    self.process_queue.put((sample, timestamp))
                    self.grafica_queue.put(sample)
        except Exception as e:
            import logging
            logging.error(f"Acquisition error: {str(e)}")

# Configuración
CANALES = {'Fz': 0, 'C3': 1, 'Cz': 2, 'C4': 3, 'Pz': 4, 'PO7': 5, 'Oz': 6, 'PO8': 7}
VENTANA = 250
REFRESCO = 50
FS = 250  # frecuencia de muestreo

# Función de filtrado
def filtrar(y, lowcut=1, highcut=40, fs=FS, order=4):
    nyq = fs * 0.5
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, y)

# Setup
root = tk.Tk()
root.title("EEG en tiempo real")

EEG = EEGConGrafica()

# Figura
fig, axes = plt.subplots(len(CANALES), 1, figsize=(10, 8), sharex=True)
fig.tight_layout(pad=2.0)

lines = {}
for ax, (nombre, idx) in zip(axes, CANALES.items()):
    line, = ax.plot([], [], lw=0.8)
    ax.set_xlim(0, VENTANA)
    ax.set_ylabel(nombre, rotation=0, labelpad=25)
    ax.yaxis.set_tick_params(labelsize=7)
    lines[nombre] = line

axes[-1].set_xlabel("Muestras")

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

buffer_grafica = []

def actualizar():
    while not EEG.grafica_queue.empty():
        try:
            sample = EEG.grafica_queue.get_nowait()
            buffer_grafica.append(sample)
        except:
            break
    
    if len(buffer_grafica) > 0:
        ultimos = buffer_grafica[-VENTANA:]
        x = np.arange(len(ultimos))

        for nombre, idx in CANALES.items():
            y = np.array([muestra[idx] for muestra in ultimos])

            # Filtro pasa banda 1-40 Hz
            if len(y) >= 15:
                y = filtrar(y)

            lines[nombre].set_data(x, y)
            if len(y) > 0:
                p5, p95 = np.percentile(y, [5, 95])
                margin = (p95 - p5) * 0.2 or 1
                lines[nombre].axes.set_ylim(p5 - margin, p95 + margin)

        canvas.draw()

    root.after(REFRESCO, actualizar)

def iniciar():
    EEG.start_acquisition(
        stream_name='streamTEST', 
        save=True, 
        save_path=r'C:\Users\lipsp\Documents\Entrenamiento imaginacion'
    )
    actualizar()

def detener():
    EEG.stop_acquisition()

# Botones
frame = tk.Frame(root)
frame.pack(pady=5)
tk.Button(frame, text="Iniciar", command=iniciar).pack(side=tk.LEFT, padx=5)
tk.Button(frame, text="Detener", command=detener).pack(side=tk.LEFT, padx=5)

root.mainloop()