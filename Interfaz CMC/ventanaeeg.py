# -*- coding: utf-8 -*-


import sys
import time
import queue
import argparse
from collections import deque

import numpy as np
from scipy.signal import butter, filtfilt

from PyQt5 import QtWidgets, QtCore   # ANTES que pyqtgraph: fija el binding Qt
import pyqtgraph as pg


from eeg_uhb import EEGAcquisitionManager

# ─── Configuración ────────────────────────────────────────────────────────────
FS           = 250
EEG_VENTANA  = 2000  # 8 s × 250 Hz
EEG_REFRESCO = 50    # ms

CANALES = ['Fz', 'FC3', 'FCz', 'FC4', 'Cz', 'C3', 'C4']
INDICES = [0, 1, 2, 3, 4, 5, 7]  # 6=NC omitido

COLORES = {
    'Fz' : '#FF1B6B',
    'FC3': '#E03884',
    'FCz': '#C1559C',
    'FC4': '#A273B5',
    'Cz' : '#8390CE',
    'C3' : '#64ADE6',
    'C4' : '#45CAFF',
}

BG_COLOR   = '#0d1117'
AXIS_COLOR = '#8b949e'
GRID_COLOR = '#30363d'


# ─── Subclase EEG — solo adquisición y cola para graficar ─────────────────────
class EEGConGrafica(EEGAcquisitionManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grafica_queue = queue.Queue()

    def _acquisition_loop(self):
        try:
            while self.running:
                sample, timestamp = self.stream_inlet.pull_sample(timeout=0.1)
                if sample:
                    self.grafica_queue.put((sample, timestamp))
        except Exception as e:
            import logging
            logging.error(f"Acquisition error: {str(e)}")

    def _storage_loop(self):
        # Desactivado: este visor no guarda
        pass


# ─── Filtro (solo visualización) ──────────────────────────────────────────────
def filtrar(y, lowcut=5, highcut=35, fs=FS, order=4):
    nyq = fs * 0.5
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, y)


# ─── Ventana principal ────────────────────────────────────────────────────────
class VentanaEEG(QtWidgets.QMainWindow):
    def __init__(self, stream_name):
        super().__init__()
        self.stream_name = stream_name
        self.buffer      = deque(maxlen=EEG_VENTANA)
        self.eeg         = EEGConGrafica()

        self._build_ui()
        self._iniciar_eeg()

        self.timer = QtCore.QTimer()
        self.timer.setInterval(EEG_REFRESCO)
        self.timer.timeout.connect(self._actualizar)
        self.timer.start()

    def _build_ui(self):
        self.setWindowTitle("EEG en tiempo real")
        self.setStyleSheet(f"background-color: {BG_COLOR};")

        escritorio = QtWidgets.QDesktopWidget()
        if escritorio.screenCount() > 1:
            self.setGeometry(escritorio.screenGeometry(0))
        else:
            self.showMaximized()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 5)
        layout.setSpacing(2)

        titulo = QtWidgets.QLabel(
            'EEG — Fz, FC3, FCz, FC4, Cz, C3, C4  ')
        titulo.setStyleSheet(
            'color: #f0f6fc; font-size: 13px; font-weight: bold;')
        layout.addWidget(titulo)

        self.graficas = pg.GraphicsLayoutWidget()
        self.graficas.setBackground(BG_COLOR)
        layout.addWidget(self.graficas)

        btn = QtWidgets.QPushButton("Detener EEG")
        btn.setStyleSheet(
            'background-color: #E05555; color: white; '
            'font-size: 13px; font-weight: bold; '
            'padding: 6px 16px; border: none;')
        btn.clicked.connect(self._detener)
        layout.addWidget(btn, alignment=QtCore.Qt.AlignCenter)

        self.curvas = {}
        ultimo_plot = None
        for i, nombre in enumerate(CANALES):
            p = self.graficas.addPlot(row=i, col=0)
            p.setMenuEnabled(False)
            p.hideButtons()
            p.setXRange(0, 8.0, padding=0)  # 8 s fijos
            p.getAxis('left').setTextPen(color=COLORES[nombre])
            p.getAxis('left').setLabel(nombre, color=COLORES[nombre], size='15pt')
            p.getAxis('bottom').setTextPen(AXIS_COLOR)
            p.getAxis('left').setPen(color=COLORES[nombre], width=1)
            p.getAxis('bottom').setPen(GRID_COLOR)
            p.showGrid(x=False, y=True, alpha=0.15)

            if i < len(CANALES) - 1:
                p.hideAxis('bottom')
            else:
                p.getAxis('bottom').setLabel('Tiempo (s)', color=AXIS_COLOR, size='10pt')

            if ultimo_plot:
                p.setXLink(ultimo_plot)
            ultimo_plot = p

            curva = p.plot(pen=pg.mkPen(color=COLORES[nombre], width=1))
            self.curvas[nombre] = (curva, p)

    def _iniciar_eeg(self):
        self.eeg.start_acquisition(stream_name=self.stream_name, save=False)

    def _actualizar(self):
        """Vacía la cola y actualiza la gráfica."""
        nuevos = 0
        while True:
            try:
                sample, _ = self.eeg.grafica_queue.get_nowait()
            except queue.Empty:
                break
            self.buffer.append(sample)
            nuevos += 1

        if not nuevos:
            return

        ultimos = list(self.buffer)
        n = len(ultimos)
        x = np.linspace(max(0, 8.0 - n / FS), 8.0, n)

        for nombre, idx in zip(CANALES, INDICES):
            y = np.array([m[idx] for m in ultimos])
            if len(y) >= 15:
                try:
                    y = filtrar(y)
                except Exception:
                    pass

            curva, p = self.curvas[nombre]
            curva.setData(x, y)

            if len(y) > 0:
                p5, p95 = np.percentile(y, [5, 95])
                margin = (p95 - p5) * 0.2 or 1
                p.setYRange(p5 - margin, p95 + margin, padding=0)

    def _detener(self):
        self.close()

    def closeEvent(self, event):
        self.timer.stop()
        try:
            self.eeg.stop_acquisition()
        except Exception as e:
            print(f"[EEG] Error al detener adquisición: {e}")
        event.accept()


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stream', nargs='?', default='streamTEST',
                    help="nombre del stream LSL del EEG")
    args = ap.parse_args()

    app = QtWidgets.QApplication(sys.argv[:1])
    win = VentanaEEG(args.stream)
    win.show()
    sys.exit(app.exec_())
