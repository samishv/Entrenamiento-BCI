# -*- coding: utf-8 -*-
"""
Ventana EEG independiente con PyQtGraph.
Se lanza como subproceso desde la interfaz principal.

Argumentos (sys.argv):
    1: stream_name
    2: save_path
    3: condicion  (ME | MI | Reposo)
    4: pid        (ID del participante)
    5: indice     (índice actual de la sesión)

Ejemplo:
    python eeg_window.py streamTEST "C:\\datos" ME 1 2
"""

import sys
import os
import time
import queue
import pickle
import datetime
import numpy as np
from scipy.signal import butter, filtfilt

import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore

from eeg_uhb import EEGAcquisitionManager

# ─── Configuración ────────────────────────────────────────────────────────────
EEG_VENTANA  = 250   # muestras visibles (1 segundo a 250 Hz)
EEG_REFRESCO = 50    # ms entre actualizaciones

# Archivo flag: cuando existe → grabar; cuando no existe → solo visualizar
FLAG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'eeg_grabar.flag')

CANALES = ['Fz', 'FC3', 'FCz', 'FC4', 'Cz', 'C3', 'NC', 'C4']
INDICES = [0, 1, 2, 3, 4, 5, 6, 7]

COLORES = {
    'Fz' : '#f94144',
    'FC3': '#f3722c',
    'FCz': '#f8961e',
    'FC4': '#f9c74f',
    'Cz' : '#90be6d',
    'C3' : '#43aa8b',
    'NC' : '#4d908e',
    'C4' : '#577590',
}

BG_COLOR   = '#0d1117'
AXIS_COLOR = '#8b949e'
GRID_COLOR = '#30363d'

# ─── Subclase EEG con timeout ─────────────────────────────────────────────────
class EEGConGrafica(EEGAcquisitionManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grafica_queue  = queue.Queue()
        self._lsl_offset    = 0
        self._first_timestamp = None

    def _acquisition_loop(self):
        try:
            from pylsl import local_clock
            self._lsl_offset      = local_clock() - time.time()
            self._first_timestamp = None
            while self.running:
                sample, timestamp = self.stream_inlet.pull_sample(timeout=0.1)
                if sample:
                    if self._first_timestamp is None:
                        self._first_timestamp = timestamp
                    self.data.append(sample)
                    self.timestamps.append(timestamp)
                    self.data_queue.put((sample, timestamp))
                    self.process_queue.put((sample, timestamp))
                    self.grafica_queue.put(sample)
        except Exception as e:
            import logging
            logging.error(f"Acquisition error: {str(e)}")

    def _storage_loop(self):
        from datetime import datetime as dt
        while self.running or not self.data_queue.empty():
            try:
                sample, timestamp = self.data_queue.get(timeout=1)
                if hasattr(self, 'csv_writer'):
                    try:
                        unix_time  = timestamp - self._lsl_offset
                        hora_local = dt.fromtimestamp(unix_time).strftime(
                            '%H:%M:%S.%f')
                        self.csv_writer.writerow([hora_local] + sample)
                        self.sample_count += 1
                    except Exception as e:
                        import logging
                        logging.error(f"Error writing data: {str(e)}")
                self.data_queue.task_done()
            except queue.Empty:
                pass


# ─── Filtro ───────────────────────────────────────────────────────────────────
def filtrar(y, lowcut=1, highcut=40, fs=250, order=4):
    nyq = fs * 0.5
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, y)


# ─── Ventana principal ────────────────────────────────────────────────────────
class VentanaEEG(QtWidgets.QMainWindow):
    def __init__(self, stream_name, save_path, condicion, pid, indice):
        super().__init__()
        self.stream_name = stream_name
        self.save_path   = save_path
        self.condicion   = condicion
        self.pid         = pid
        self.indice      = indice
        self.buffer      = []
        self.eeg         = EEGConGrafica()
        self._grabando   = False  # True cuando el flag existe

        self._build_ui()
        self._iniciar_eeg()

        # Timer de refresco de gráfica
        self.timer = QtCore.QTimer()
        self.timer.setInterval(EEG_REFRESCO)
        self.timer.timeout.connect(self._actualizar)
        self.timer.start()

        # Timer que vigila el archivo flag cada 100ms
        self.timer_flag = QtCore.QTimer()
        self.timer_flag.setInterval(100)
        self.timer_flag.timeout.connect(self._revisar_flag)
        self.timer_flag.start()

        # Limpiar flag residual al arrancar
        if os.path.exists(FLAG_PATH):
            os.remove(FLAG_PATH)

    def _build_ui(self):
        self.setWindowTitle("EEG en tiempo real")
        self.setStyleSheet(f"background-color: {BG_COLOR};")

        # Posicionar en monitor primario
        escritorio = QtWidgets.QDesktopWidget()
        if escritorio.screenCount() > 1:
            pantalla = escritorio.screenGeometry(0)  # monitor primario
            self.setGeometry(pantalla)
        else:
            self.showMaximized()

        # Layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 5)
        layout.setSpacing(2)

        # Título
        titulo = QtWidgets.QLabel(
            'Señales EEG Crudas — Fz, FC3, FCz, FC4, Cz, C3, NC, C4')
        titulo.setStyleSheet(
            'color: #f0f6fc; font-size: 13px; font-weight: bold;')
        layout.addWidget(titulo)

        # Área de gráficas
        self.graficas = pg.GraphicsLayoutWidget()
        self.graficas.setBackground(BG_COLOR)
        layout.addWidget(self.graficas)

        # Botón detener
        btn = QtWidgets.QPushButton("Detener EEG")
        btn.setStyleSheet(
            'background-color: #E05555; color: white; '
            'font-size: 13px; font-weight: bold; '
            'padding: 6px 16px; border: none;')
        btn.clicked.connect(self._detener)
        layout.addWidget(btn, alignment=QtCore.Qt.AlignCenter)

        # Crear subplots por canal
        self.curvas = {}
        ultimo_plot = None
        for i, nombre in enumerate(CANALES):
            p = self.graficas.addPlot(row=i, col=0)

            # Estilo
            p.setMenuEnabled(False)
            p.hideButtons()
            p.setXRange(0, EEG_VENTANA, padding=0)
            p.getAxis('left').setTextPen(COLORES[nombre])
            p.getAxis('left').setLabel(nombre,
                color=COLORES[nombre], size='9pt')
            p.getAxis('bottom').setTextPen(AXIS_COLOR)
            p.getAxis('left').setPen(GRID_COLOR)
            p.getAxis('bottom').setPen(GRID_COLOR)
            p.showGrid(x=False, y=True,
                alpha=0.15)

            # Quitar eje X en todos menos el último
            if i < len(CANALES) - 1:
                p.hideAxis('bottom')
            else:
                p.getAxis('bottom').setLabel(
                    'Muestras', color=AXIS_COLOR, size='10pt')

            # Enlazar eje X entre subplots
            if ultimo_plot:
                p.setXLink(ultimo_plot)
            ultimo_plot = p

            curva = p.plot(pen=pg.mkPen(color=COLORES[nombre], width=1))
            self.curvas[nombre] = (curva, p)

    def _iniciar_eeg(self):
        # Arrancar sin guardar — el guardado se activa cuando llega el flag
        self.eeg.start_acquisition(
            stream_name=self.stream_name,
            save=False
        )

    def _revisar_flag(self):
        """Vigila el archivo flag y activa/desactiva el guardado."""
        flag_existe = os.path.exists(FLAG_PATH)

        if flag_existe and not self._grabando:
            # Leer condicion, pid e indice del flag
            try:
                with open(FLAG_PATH, 'r') as f:
                    partes = f.read().strip().split(',')
                self.condicion = partes[0] if len(partes) > 0 else self.condicion
                self.pid       = partes[1] if len(partes) > 1 else self.pid
                self.indice    = partes[2] if len(partes) > 2 else self.indice
            except Exception:
                pass

            # Flag apareció → empezar a guardar
            self._grabando = True
            self.eeg.stop_acquisition()
            self.eeg.start_acquisition(
                stream_name=self.stream_name,
                save=True,
                save_path=self.save_path
            )
            print(f"[EEG] Inicio de grabación — {self.condicion} S{self.pid} #{self.indice}")

        elif not flag_existe and self._grabando:
            # Flag desapareció → detener y renombrar
            self._grabando = False
            self.eeg.stop_acquisition()
            self._renombrar_archivo()
            # Reiniciar adquisición solo para visualización
            self.eeg = EEGConGrafica()
            self.eeg.start_acquisition(
                stream_name=self.stream_name,
                save=False
            )
            print("[EEG] Grabación detenida")

    def _actualizar(self):
        # Vaciar cola al buffer
        while not self.eeg.grafica_queue.empty():
            try:
                self.buffer.append(self.eeg.grafica_queue.get_nowait())
            except Exception:
                break

        if not self.buffer:
            return

        ultimos = self.buffer[-EEG_VENTANA:]
        x = np.arange(len(ultimos))

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
        self.timer.stop()
        self.timer_flag.stop()
        if os.path.exists(FLAG_PATH):
            os.remove(FLAG_PATH)
        self.eeg.stop_acquisition()
        if self._grabando:
            self._renombrar_archivo()
        self.close()

    def _renombrar_archivo(self):
        if not hasattr(self.eeg, 'filepath'):
            return
        try:
            fecha  = datetime.datetime.now().strftime("%d%m%y")
            nombre = f"S{self.pid}_EEG_{fecha}_{self.condicion}_{self.indice}.csv"
            carpeta = os.path.dirname(self.eeg.filepath)
            nuevo   = os.path.join(carpeta, nombre)
            os.rename(self.eeg.filepath, nuevo)
            print(f"[EEG] Guardado como: {nombre}")
        except Exception as e:
            print(f"[EEG] No se pudo renombrar: {e}")

    def closeEvent(self, event):
        self.timer.stop()
        self.timer_flag.stop()
        if os.path.exists(FLAG_PATH):
            os.remove(FLAG_PATH)
        self.eeg.stop_acquisition()
        if self._grabando:
            self._renombrar_archivo()
        event.accept()


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    stream_name = sys.argv[1] if len(sys.argv) > 1 else 'streamTEST'
    save_path   = sys.argv[2] if len(sys.argv) > 2 else './Data'
    condicion   = sys.argv[3] if len(sys.argv) > 3 else 'EEG'
    pid         = sys.argv[4] if len(sys.argv) > 4 else 'X'
    indice      = sys.argv[5] if len(sys.argv) > 5 else '0'

    app = QtWidgets.QApplication(sys.argv)
    win = VentanaEEG(stream_name, save_path, condicion, pid, indice)
    win.show()
    sys.exit(app.exec_())
