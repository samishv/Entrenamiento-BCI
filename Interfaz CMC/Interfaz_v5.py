# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 01:32:15 2026

@author: samishv
"""

import tkinter as tk
import random
import threading
import queue
import math
import serial        # pip install pyserial
import time
import winsound
from PIL import Image, ImageTk

# ─── EEG ──────────────────────────────────────────────────────────────────────
import numpy as np
from scipy.signal import butter, filtfilt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from eeg_uhb import EEGAcquisitionManager

EEG_STREAM_NAME = 'streamTEST'   # debe ser el mismo que en unicorn lsl
EEG_SAVE_PATH   = r'C:\Users\ikerf\Desktop\Upiita\TT' #r'C:\Users\lipsp\Documents\Entrenamiento imaginacion'
EEG_CANALES     = {'Fz': 0, 'FC3': 1, 'FCz': 2, 'FC4': 3,
                   'Cz': 4, 'C3': 5, 'NC': 6, 'C4': 7} # NC-not connected
EEG_VENTANA     = 250   # muestras visibles (1 segundo a 250 Hz)
EEG_REFRESCO    = 50    # ms entre actualizaciones de la gráfica

class EEGConGrafica(EEGAcquisitionManager):
    """Subclase que agrega timeout al pull_sample y una cola exclusiva para graficar."""
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

def _filtrar_eeg(y, lowcut=1, highcut=40, fs=250, order=4):
    nyq = fs * 0.5
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, y)

# ─── Conexión serial ──────────────────────────────────────────────────────────
SERIAL_PORT = 'COM7'
SERIAL_BAUD = 115200
CMD_START   = b'S\n'   # letra que Arduino espera para comenzar a enviar
CMD_STOP    = b'P\n'   # letra que Arduino espera para dejar de enviar

try:
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
    time.sleep(2)   # esperar a que Arduino reinicie
    SERIAL_OK = True
    print(f"Puerto {SERIAL_PORT} abierto correctamente.")
except serial.SerialException as e:
    ser = None
    SERIAL_OK = False
    print(f"⚠  No se pudo abrir {SERIAL_PORT}: {e}")
    print("   La app correrá en modo simulación.")
    
    

# ─── Paleta ──────────────────────────────────────────────────────────────────
BG_MENU     = "#5B9DB5"
BG_LIGHT    = "#EEF4F7"
CARD_WHITE  = "#FFFFFF"
GREEN_MAIN  = "#6DC14B"
GREEN_LIGHT = "#B8E4A0"
GREEN_ZONE  = "#D4F0C0"
NAVY        = "#1E3A5F"
TEXT_MUTED  = "#7A9AAF"
TEXT_MUTEDPLUS  = "#D3D3D3"
PROG_BG     = "#C8DDE6"
ZONE_BORDER = "#6DC14B"
WARN_RED    = "#E05555"
TEAL       = "#2BBCD4"
ORANGE     = "#F5A623"
GREEN      = "#5CB85C"
RED_DOT    = "#E74C3C"
GRAY_TEXT  = "#282828"
DARK_TEXT  = "#2C3E50"

# ─── Tipografías ─────────────────────────────────────────────────────────────
F_TITLE = ("Georgia",   42, "bold")
F_HEAD  = ("Helvetica", 22, "bold")
F_LABEL = ("Helvetica", 16)
F_SMALL = ("Helvetica", 13)
F_NUM   = ("Courier",   80, "bold")
F_BTN   = ("Helvetica", 16, "bold")
F_TIMER = ("Courier",   22, "bold")
F_SECTION  = ("Helvetica", 12, "bold")
F_ORDER_N  = ("Helvetica", 20, "bold")
F_ORDER_T  = ("Helvetica", 13)

CALIB_DURATION   = 10   # segundos de calibración
REGISTRO_DURATION = 32  # segundos de registro


# ─── Helpers ─────────────────────────────────────────────────────────────────
def rrect(canvas, x1, y1, x2, y2, r=14, **kw):
    pts = [x1+r, y1, x2-r, y1, x2, y1, x2, y1+r,
           x2, y2-r, x2, y2, x2-r, y2, x1+r, y2,
           x1, y2, x1, y2-r, x1, y1+r, x1, y1]
    canvas.create_polygon(pts, smooth=True, **kw)


def draw_graph(canvas, history,
               pad_l=70, pad_r=24, pad_t=25, pad_b=35,
               y_max=100, y_ticks=None,
               zone_lo=None, zone_hi=None, show_zone=False):
    """
    Dibuja la gráfica de señal en el canvas dado.
    - y_max   : valor máximo del eje Y (Newtons o %)
    - y_ticks : lista de marcas del eje Y; si es None se generan automáticamente
    - show_zone / zone_lo / zone_hi : zona objetivo sombreada
    """
    canvas.delete("all")
    w = canvas.winfo_width()
    h = canvas.winfo_height()
    if w < 20 or h < 20:
        return  

    gw = w - pad_l - pad_r
    gh = h - pad_t - pad_b

    def gy(val):
        return pad_t + gh - (val / y_max * gh)

    # Marcas del eje Y — automáticas o manuales
    ticks = y_ticks or [int(y_max * i / 5) for i in range(6)]

    # Zona objetivo (fondo sombreado)
    if show_zone and zone_lo is not None and zone_hi is not None:
        y_hi = gy(zone_hi)
        y_lo = gy(zone_lo)
        canvas.create_rectangle(pad_l, y_hi, w - pad_r, y_lo,
                                 fill=GREEN_ZONE, outline="")
        for y in [y_hi, y_lo]:
            canvas.create_line(pad_l, y, w - pad_r, y,
                               fill=ZONE_BORDER, width=2, dash=(8, 5))
        canvas.create_text(w - pad_r - 8, y_hi + 8,
                           text=f"Zona objetivo  {zone_lo}–{zone_hi}%",
                           anchor="ne", font=F_SMALL, fill=ZONE_BORDER)

    # Cuadrícula Y
    for val in ticks:
        y = gy(val)
        canvas.create_line(pad_l, y, w - pad_r, y,
                           fill="#D0E4EE", dash=(4, 6))
        canvas.create_text(pad_l - 8, y, text=str(val),
                           anchor="e", font=F_SMALL, fill=TEXT_MUTED)

    # Señal
    data = history[-150:]
    if len(data) < 2:
        return
    step = gw / max(len(data) - 1, 1)
    pts  = [(pad_l + i * step, gy(v)) for i, v in enumerate(data)]

    poly = [(pad_l, pad_t + gh)] + pts + [(pad_l + gw, pad_t + gh)]
    canvas.create_polygon([c for p in poly for c in p],
                          fill=GREEN_LIGHT, outline="", smooth=True)
    canvas.create_line([c for p in pts for c in p],
                       fill=GREEN_MAIN, width=3, smooth=True)

    # Punto actual
    cx, cy = pts[-1]
    canvas.create_oval(cx - 8, cy - 8, cx + 8, cy + 8,
                       fill=GREEN_MAIN, outline=CARD_WHITE, width=3)


def draw_timer_bar(canvas, timer_lbl, elapsed_ms, total_ms, label_fmt="{s} / {t} s"):
    """Dibuja la barra de progreso de tiempo y actualiza la etiqueta."""
    canvas.delete("all")
    w = canvas.winfo_width()
    if w < 10:
        return
    pct   = min(elapsed_ms / total_ms, 1.0)
    sec   = min(elapsed_ms // 1000, total_ms // 1000)
    total = total_ms // 1000
    timer_lbl.config(text=label_fmt.format(s=sec, t=total))
    rrect(canvas, 0, 4, w, 36, r=12, fill=PROG_BG, outline="")
    if pct > 0:
        fill_w = max(28, int(w * pct))
        color  = GREEN_MAIN if pct < 1.0 else NAVY
        rrect(canvas, 0, 4, fill_w, 36, r=12, fill=color, outline="")
        
def make_card(parent, border_color, width=100, height=380):
    """Frame con borde de color simulado usando un Frame exterior."""
    outer = tk.Frame(parent, bg=border_color, padx=3, pady=3)
    inner = tk.Frame(outer, bg=CARD_WHITE)
    inner.pack(fill="both", expand=True)
    return outer, inner


def colored_btn(parent, text, color, command=None, width=14):
    return tk.Button(
        parent, text=text, font=F_BTN,
        bg=color, fg=CARD_WHITE, activebackground=color,
        relief="flat", bd=0, padx=12, pady=8,
        cursor="hand2", width=width, command=command
    )


# ══════════════════════════════════════════════════════════════════════════════
class CoherenciaBCI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Coherencia Corticomuscular")
        self.resizable(True, True)
        self.attributes("-fullscreen", True)
        self.configure(bg=BG_MENU)
        self.bind("<Escape>", lambda e: self.attributes("-fullscreen", False))
        self.bind("<F11>",    lambda e: self.attributes("-fullscreen", True))
        self.orden_sesion = []   # se llena al generar
        self.indice_actual = 0   # apunta a la condición en curso

        # Fuerza máxima registrada en calibración (Newtons)
        self.max_force = tk.DoubleVar(value=0.0)

        # Cola donde el hilo serial deposita los valores leídos.
        # _tick() la consume con get_nowait() sin bloquear la interfaz.
        self.serial_queue = queue.Queue()

        # Hilo lector — solo arranca si el puerto está disponible
        if SERIAL_OK:
            t = threading.Thread(target=self._serial_reader, daemon=True)
            t.start()

        container = tk.Frame(self, bg=BG_MENU)
        container.pack(fill="both", expand=True)

        self.frames = {}
        for F in (MenuPage, CalibracionPage, RegistroPage, ConfiguracionPage, MIPage, REPage):
            frame = F(parent=container, controller=self)
            self.frames[F.__name__] = frame
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.show_frame("MenuPage")

        # ─── EEG ──────────────────────────────────────────────────────────────
        self.eeg = EEGConGrafica()
        self._eeg_window   = None
        self._eeg_buffer   = []
        self._eeg_lines    = {}
        self._eeg_canvas   = None

    def abrir_ventana_eeg(self, titulo="EEG en tiempo real", save_path=None):
        """Abre la ventana de visualización EEG en el monitor secundario."""
        if self._eeg_window and self._eeg_window.winfo_exists():
            return

        path = save_path or EEG_SAVE_PATH
        self.eeg.start_acquisition(
            stream_name=EEG_STREAM_NAME,
            save=True,
            save_path=path
        )

        win = tk.Toplevel(self)
        win.title(titulo)
        win.protocol("WM_DELETE_WINDOW", self.cerrar_ventana_eeg)

        # ── Posicionar en monitor secundario ──────────────────────────────────
        # Ancho total de todos los monitores
        ancho_total = win.winfo_screenwidth()
        alto        = win.winfo_screenheight()
        # El monitor principal suele ocupar la mitad izquierda.
        # El secundario empieza en ancho_total // 2 (ajusta si es al revés).
        offset_x    = ancho_total // 2
        win.geometry(f"{offset_x}x{alto}+{offset_x}+0")
        # Para pantalla completa en el secundario descomenta la línea siguiente:
        win.attributes("-fullscreen", True)
        # ---------------------------------------------------------------------

        self._eeg_window = win
        self._eeg_buffer = []

        fig, axes = plt.subplots(len(EEG_CANALES), 1,
                                 figsize=(10, 8), sharex=True)
        fig.tight_layout(pad=1.5)
        self._eeg_lines = {}

        for ax, (nombre, idx) in zip(axes, EEG_CANALES.items()):
            line, = ax.plot([], [], lw=0.8, color='#2BBCD4')
            ax.set_xlim(0, EEG_VENTANA)
            ax.set_ylabel(nombre, rotation=0, labelpad=25, fontsize=8)
            ax.yaxis.set_tick_params(labelsize=7)
            ax.set_facecolor('#F5F9FB')
            self._eeg_lines[nombre] = line

        axes[-1].set_xlabel("Muestras")

        self._eeg_canvas = FigureCanvasTkAgg(fig, master=win)
        self._eeg_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        tk.Button(win, text="Detener EEG", command=self.cerrar_ventana_eeg,
                  bg='#E05555', fg='white', font=("Helvetica", 12, "bold"),
                  relief="flat", padx=16, pady=6).pack(pady=6)

        self._actualizar_eeg()

    def cerrar_ventana_eeg(self, condicion=None):
        """Detiene la adquisición EEG, renombra el archivo y cierra la ventana."""
        self.eeg.stop_acquisition()

        # ── Renombrar archivo CSV con formato S{id}_EEG_{fecha}_{cond}_{idx} ──
        if condicion and hasattr(self.eeg, 'filepath'):
            import os
            from datetime import datetime
            try:
                cfg    = self.frames.get("ConfiguracionPage")
                pid    = cfg.participanteID.get().strip() if cfg else "X"
                fecha  = datetime.now().strftime("%d%m%y")
                indice = self.indice_actual
                nombre = f"S{pid}_EEG_{fecha}_{condicion}_{indice}.csv"

                carpeta = os.path.dirname(self.eeg.filepath)
                nuevo   = os.path.join(carpeta, nombre)
                os.rename(self.eeg.filepath, nuevo)
                print(f"[EEG] Archivo guardado como: {nombre}")
            except Exception as e:
                print(f"[EEG] No se pudo renombrar el archivo: {e}")
        # ─────────────────────────────────────────────────────────────────────

        if self._eeg_window and self._eeg_window.winfo_exists():
            self._eeg_window.destroy()
        self._eeg_window = None
        self._eeg_canvas = None
        self._eeg_lines  = {}
        self._eeg_buffer = []
        plt.close('all')

    def _actualizar_eeg(self):
        """Loop de refresco de la gráfica EEG."""
        if not self._eeg_window or not self._eeg_window.winfo_exists():
            return

        while not self.eeg.grafica_queue.empty():
            try:
                self._eeg_buffer.append(self.eeg.grafica_queue.get_nowait())
            except Exception:
                break

        if self._eeg_buffer and self._eeg_canvas:
            ultimos = self._eeg_buffer[-EEG_VENTANA:]
            x = np.arange(len(ultimos))

            for nombre, idx in EEG_CANALES.items():
                y = np.array([m[idx] for m in ultimos])
                # if len(y) >= 15:
                #     try:
                #         y = _filtrar_eeg(y)
                #     except Exception:
                #         pass
                self._eeg_lines[nombre].set_data(x, y)
                if len(y) > 0:
                    p5, p95 = np.percentile(y, [5, 95])
                    margin = (p95 - p5) * 0.2 or 1
                    self._eeg_lines[nombre].axes.set_ylim(
                        p5 - margin, p95 + margin)

            self._eeg_canvas.draw()

        self._eeg_window.after(EEG_REFRESCO, self._actualizar_eeg)

    def play_sound(self, tipo="start"):
        sonidos = {
            "start": (1000, 500),
            "finish":(900, 500),
        }
        freq, dur = sonidos.get(tipo, (500, 200))
        # SND_ASYNC para no bloquear Tkinter
        threading.Thread(
            target=lambda: winsound.Beep(freq, dur),
            daemon=True
        ).start()

    def _serial_reader(self):
        """
        Corre permanentemente en su propio hilo (daemon=True, muere con la app).
        Lee líneas del puerto serial y deposita el float en la cola.
        Ignora líneas malformadas silenciosamente.
        """
        while True:
            try:
                line = ser.readline().decode('utf-8').strip()
                if line:
                    value = float(line)
                    self.serial_queue.put(value)
                    print(f"[SERIAL] recibido: {value} N")
            except (ValueError, UnicodeDecodeError):
                pass   # línea vacía o basura — ignorar
            except serial.SerialException:
                break  # puerto desconectado — terminar hilo

    def serial_send(self, cmd):
        """Envía un comando al Arduino si el puerto está disponible."""
        if SERIAL_OK and ser and ser.is_open:
            ser.write(cmd)
            print(f"[SERIAL] enviado: {repr(cmd)}")

    def show_frame(self, name):
        self.frames[name].tkraise()
        self.frames[name].on_enter()


# ══════════════════════════════════════════════════════════════════════════════
class MenuPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=BG_MENU)
        self.controller = controller

        card = tk.Frame(self, bg=CARD_WHITE)
        card.place(relx=0.5, rely=0.5, anchor="center", width=840, height=480)

        tk.Label(card, text="Coherencia\ncorticomuscular",
                 font=F_TITLE, fg=NAVY, bg=CARD_WHITE,
                 justify="center").pack(pady=(80, 50))

        row = tk.Frame(card, bg=CARD_WHITE)
        row.pack()
        self._btn(row, "Calibración",
                  lambda: controller.show_frame("CalibracionPage")).pack(side="left", padx=20)
        self._btn(row, "Registro",
                  lambda: controller.show_frame("RegistroPage")).pack(side="left", padx=20)
        
        # El Frame se ancla abajo
        bottom_bar = tk.Frame(card, bg=CARD_WHITE)
        bottom_bar.pack(side="bottom", fill="x", padx=1,)      
        tk.Button(bottom_bar, text="Configuración ⚙️", command=lambda: controller.show_frame("ConfiguracionPage"),
                  bg=CARD_WHITE, fg=TEXT_MUTEDPLUS, font=F_SMALL,
                  relief="flat").pack(side="right")

        tk.Label(self, text="Esc — salir de pantalla completa  |  F11 — volver",
                 font=F_SMALL, fg=CARD_WHITE, bg=BG_MENU).pack(side="bottom", pady=14)

    def _btn(self, parent, text, cmd):
        return tk.Button(parent, text=text, command=cmd,
                         bg=GREEN_MAIN, fg="white", font=F_BTN,
                         relief="flat", padx=34, pady=14, cursor="hand2",
                         activebackground="#5aab35", activeforeground="white")

    def on_enter(self):
        pass


# ══════════════════════════════════════════════════════════════════════════════

class ConfiguracionPage(tk.Frame):
    def __init__(self, parent, controller=None):
        super().__init__(parent, bg=CARD_WHITE)
        self.controller = controller

        hdr = tk.Frame(self, bg=CARD_WHITE)
        hdr.pack(fill="x", padx=40, pady=(30, 10))

        tk.Label(hdr, text="Configuración", font=F_HEAD,
                 fg=NAVY, bg=CARD_WHITE).pack(side="left")
        
        tk.Button(hdr, text="← Menú", command=self._go_menu,
                  bg=CARD_WHITE, fg=TEXT_MUTED, font=F_LABEL,
                  relief="flat", cursor="hand2").pack(side="right")
        
        # Botón al fondo derecho
        bot = tk.Frame(self, bg=CARD_WHITE)
        bot.pack(side="bottom", fill="x", padx=70, pady=30)
        tk.Button(bot, text="Continuar →",
                  command=lambda: controller.show_frame("CalibracionPage"),
                  bg=CARD_WHITE, fg=TEXT_MUTED, font=F_LABEL,
                  relief="flat", padx=30, pady=12, cursor="hand2").pack(side="right")

        # ID participante (derecha del encabezado)
        id_frame = tk.Frame(hdr, bg=CARD_WHITE)
        id_frame.pack(side="left", padx=(60, 0))
        tk.Label(id_frame, text="ID de participante:", font=F_LABEL,
                 fg=TEXT_MUTED, bg=CARD_WHITE).pack(side="left", padx=(0, 8))
        self.participanteID = tk.Entry(
            id_frame, font=F_LABEL, fg=NAVY, bg=BG_LIGHT,
            relief="flat", width=12, bd=4
        )
        self.participanteID.pack(side="left", ipady=4)
        self.participanteID.insert(0, "1")

        # ── Fila de tarjetas ─────────────────────────────────────
        cards_row = tk.Frame(self, bg=CARD_WHITE)
        cards_row.pack(fill="both", expand=True, padx=70, pady=5)

        # Columnas equidistantes
        for col in range(3):
            cards_row.columnconfigure(col, weight=1, uniform="col")
        cards_row.rowconfigure(1, weight=1)


        self._build_serial_card(cards_row)
        self._build_orden_card(cards_row)
        self._build_calibracion_card(cards_row)

    # ── Tarjeta 1: Conexión serial ────────────────────────────────
    def _build_serial_card(self, parent):
        tk.Label(parent, text="Conexión serial", font=F_BTN,
                 fg=TEAL, bg=CARD_WHITE).grid(row=0, column=0, pady=(100, 8))

        outer, card = make_card(parent, TEAL, height=660)
        outer.grid(row=1, column=0, padx=15, sticky="nsew")

        # Botón
        btn_frame = tk.Frame(card, bg=CARD_WHITE)
        btn_frame.pack(side="bottom", pady=30)
        colored_btn(btn_frame, "Conectar", TEAL,
                    command=self._conectar).pack()
        
        # Puerto
        port_row = tk.Frame(card, bg=CARD_WHITE)
        port_row.pack(fill="both", expand=True, padx=20, pady=20)
        tk.Label(port_row, text="Puerto:", font=F_LABEL,
                 fg=DARK_TEXT, bg=CARD_WHITE).pack(pady=(80, 20))
        self.puerto = tk.Entry(port_row, font=F_LABEL, fg=DARK_TEXT,
                               bg="#DDE3EE", relief="flat", width=10, bd=4)
        self.puerto.pack()
        self.puerto.insert(0, "COM3")

        # Status
        tk.Label(port_row, text="Conexión Status", font=F_LABEL,
                 fg=DARK_TEXT, bg=CARD_WHITE).pack(pady=(50, 8))
        self.status_dot = tk.Label(port_row, text="△", font=("Helvetica", 32),
                                   fg=RED_DOT, bg=CARD_WHITE)
        self.status_dot.pack(pady=(0,0))

    # ── Tarjeta 2: Generar orden de sesión ───────────────────────
    def _build_orden_card(self, parent):
        tk.Label(parent, text="Generar orden de sesión", font=F_BTN,
                 fg=ORANGE, bg=CARD_WHITE).grid(row=0, column=1, pady=(100, 8))

        outer, card = make_card(parent, ORANGE, height=360)
        outer.grid(row=1, column=1, padx=15, sticky="nsew")

        # Botón
        btn_frame = tk.Frame(card, bg=CARD_WHITE)
        btn_frame.pack(side="bottom", pady=20)
        colored_btn(btn_frame, "Generar", ORANGE,
                    command=self._generar_orden).pack()

        # Lista de condiciones (se actualiza al generar)
        self.orden_labels = []
        list_frame = tk.Frame(card, bg=CARD_WHITE)
        list_frame.place(relx=0.5, rely=0.4, anchor="center")
        condiciones_default = ["RE", "MI", "ME","MI"]
        for i, cond in enumerate(condiciones_default, start=1):
            row = tk.Frame(list_frame, bg=CARD_WHITE)
            row.pack(fill="x", padx=(150,20), pady=20)
            num = tk.Label(row, text=str(i), font=F_ORDER_N,
                           fg=ORANGE, bg=CARD_WHITE, width=3)
            num.pack(side="left")
            lbl = tk.Label(row, text=cond, font=F_ORDER_T,
                           fg=GRAY_TEXT, bg=CARD_WHITE)
            lbl.pack(padx=(50,200))
            self.orden_labels.append(lbl)


    # ── Tarjeta 3: Calibración ────────────────────────────────────
    def _build_calibracion_card(self, parent):
        tk.Label(parent, text="Calibración", font=F_BTN,
                 fg=GREEN_MAIN, bg=CARD_WHITE).grid(row=0, column=2, pady=(100, 8))

        outer, card = make_card(parent, GREEN_MAIN, height=360)
        outer.grid(row=1, column=2, padx=15, sticky="nsew")

        # Botón
        btn_frame = tk.Frame(card, bg=CARD_WHITE)
        btn_frame.pack(side="bottom", pady=20)
        colored_btn(btn_frame, "Iniciar", GREEN,
                    command=self._iniciar_calibracion).pack()
        
        img_frame = tk.Frame(card, bg=CARD_WHITE)
        img_frame.pack(fill="both", expand=True, padx=20, pady=20)

        img_pil = Image.open("vernier.jpg").resize((320, 400))
        img_tk  = ImageTk.PhotoImage(img_pil)
        lbl = tk.Label(img_frame, image=img_tk, bg=CARD_WHITE)
        lbl.pack(expand=True, pady=(30, 10))
        lbl.img_ref = img_tk

    # ── Callbacks (conectar con lógica real) ─────────────────────
    def _conectar(self):
        # TODO: lógica de conexión serial
        self.status_dot.config(fg=GREEN)  # verde si conecta

    def _generar_orden(self):
        condiciones = ["Reposo", "MI", "ME"]
        self.controller.indice_actual = 0
        random.shuffle(condiciones)          # 3 sin repetir
        orden = condiciones + [random.choice(condiciones)]  # +1 que puede repetir
        self.controller.orden_sesion = orden  #  guardar en controlador
        
        for lbl, cond in zip(self.orden_labels, orden):
            lbl.config(text=cond)
        

    def _iniciar_calibracion(self):
        # TODO: navegar a CalibracionPage
        if self.controller:
            self.controller.show_frame("CalibracionPage")

    def get_filename(self):
        from datetime import datetime
        pid    = self.participanteID.get().strip()
        sesion = getattr(self, '_sesion', '1')
        fecha  = datetime.now().strftime("%d%m%y")
        return f"S{pid}_RegistroCMC_ME_{fecha}_{sesion}"
    
    def on_enter(self):
        pass
    
    def _go_menu(self):
        self.controller.show_frame("MenuPage")
        


# ══════════════════════════════════════════════════════════════════════════════
class CalibracionPage(tk.Frame):
    """
    Barra superior = temporizador del ensayo.
    Gráfica = fuerza en Newtons en tiempo real.
    Eje Y se adapta automáticamente al pico registrado.
    """
    def __init__(self, parent, controller):
        super().__init__(parent, bg=BG_LIGHT)
        self.controller  = controller
        self._running    = False
        self._elapsed_ms = 0
        self._history    = []   # valores en Newtons
        self._peak_n     = 0.0

        # Header
        hdr = tk.Frame(self, bg=BG_LIGHT)
        hdr.pack(fill="x", padx=70, pady=(50, 0))
        tk.Label(hdr, text="Calibración", font=F_HEAD, fg=NAVY,
                 bg=BG_LIGHT).pack(side="left")
        tk.Button(hdr, text="← Menú", command=self._go_menu,
                  bg=BG_LIGHT, fg=TEXT_MUTED, font=F_LABEL,
                  relief="flat", cursor="hand2").pack(side="right")
        
        # Botón al fondo derecho
        bot = tk.Frame(self, bg=BG_LIGHT)
        bot.pack(side="bottom", fill="x", padx=70, pady=30)
        tk.Button(bot, text="Continuar →",
                  command=lambda: self._siguiente(),
                  bg=BG_LIGHT, fg=TEXT_MUTED, font=F_LABEL,
                  relief="flat", padx=30, pady=12, cursor="hand2").pack(side="right")
        
        # Instrucción
        self.instr = tk.Label(self,
            text=f"Aplica tu fuerza máxima durante {CALIB_DURATION} segundos.",
            font=F_LABEL, fg=TEXT_MUTED, bg=BG_LIGHT)
        self.instr.pack(pady=(16, 0))

        # Barra de tiempo
        timer_wrap = tk.Frame(self, bg=BG_LIGHT)
        timer_wrap.pack(fill="x", padx=70, pady=(18, 0))
        timer_hdr = tk.Frame(timer_wrap, bg=BG_LIGHT)
        timer_hdr.pack(fill="x")
        tk.Label(timer_hdr, text="Tiempo del ensayo",
                 font=F_SMALL, fg=TEXT_MUTED, bg=BG_LIGHT).pack(side="left")
        self.timer_lbl = tk.Label(timer_hdr, text=f"0 / {CALIB_DURATION} s",
                                  font=F_TIMER, fg=NAVY, bg=BG_LIGHT)
        self.timer_lbl.pack(side="right")
        self.bar_cv = tk.Canvas(timer_wrap, height=40,
                                bg=BG_LIGHT, highlightthickness=0)
        self.bar_cv.pack(fill="x", pady=(8, 0))
        self.bar_cv.bind("<Configure>", lambda e: self._draw_timer())

        # Fuerza actual (número grande)
        force_row = tk.Frame(self, bg=BG_LIGHT)
        force_row.pack(pady=(20, 0))
        tk.Label(force_row, text="Fuerza:",
                 font=F_LABEL, fg=TEXT_MUTED, bg=BG_LIGHT).pack(side="left", padx=(0, 10))
        self.force_lbl = tk.Label(force_row, text="-- N",
                                  font=F_NUM, fg=GREEN_MAIN, bg=BG_LIGHT)
        self.force_lbl.pack(side="left")
        self.peak_lbl = tk.Label(force_row, text="   Pico: -- N",
                                 font=F_LABEL, fg=TEXT_MUTED, bg=BG_LIGHT)
        self.peak_lbl.pack(side="left")

        # Gráfica
        self.graph = tk.Canvas(self, bg=BG_LIGHT, highlightthickness=0)
        self.graph.pack(fill="both", expand=True, padx=70, pady=(10, 0))
        self.graph.bind("<Configure>", lambda e: self._redraw_graph())

        # Botón + resultado
        self.start_btn = tk.Button(bot, text="Iniciar", command=self._toggle,
                                   bg=GREEN_MAIN, fg="white", font=F_BTN,
                                   relief="flat", padx=30, pady=12, cursor="hand2",
                                   activebackground="#5aab35", activeforeground="white")
        self.start_btn.pack(side="left")
        self.result_lbl = tk.Label(bot, text="", font=F_LABEL, fg=NAVY, bg=BG_LIGHT)
        self.result_lbl.pack(side="left", padx=30)

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def on_enter(self):
        if self._running:
            self._stop()
        self._reset()

    def _go_menu(self):
        if self._running:
            self._stop()
        self.controller.show_frame("MenuPage")

    # ── control ───────────────────────────────────────────────────────────────
    def _toggle(self):
        if not self._running:
            self._begin()
        else:
            self._stop()

    def _begin(self):
        self._running    = True
        self._start_time = time.perf_counter()   # momento exacto de inicio
        self._history    = []
        self._peak_n     = 0.0
        self.result_lbl.config(text="")
        self.start_btn.config(text="Detener")
        self.instr.config(text="¡Aplica tu fuerza máxima ahora!")
        # Vaciar cola por si quedaron datos de una sesión anterior
        while not self.controller.serial_queue.empty():
            self.controller.serial_queue.get_nowait()
        self.controller.serial_send(CMD_START)   # Arduino: empieza a enviar
        self._tick()

    def _stop(self):
        self._running = False
        self.controller.serial_send(CMD_STOP)    # Arduino: para de enviar
        self.start_btn.config(text="Iniciar")
        self.instr.config(
            text=f"Presiona INICIAR y aplica tu fuerza máxima durante {CALIB_DURATION} segundos.")

    def _tick(self):
        if not self._running:
            return
        INTERVAL = 10  # ms — frecuencia de refresco de la interfaz
        # Tiempo real transcurrido en milisegundos
        self._elapsed_ms = (time.perf_counter() - self._start_time) * 1000

        # Leer dato de la cola (no bloqueante)
        # Si Arduino envía más lento que INTERVAL, get_nowait lanza queue.Empty
        # y repetimos el último valor para no congelar la gráfica
        
        # Vaciar la cola y quedarse solo con el dato más reciente
        force_n = None
        while not self.controller.serial_queue.empty():
            try:
                force_n = self.controller.serial_queue.get_nowait()
                print(f"[COLA] dato leído: {force_n:.1f} | restantes en cola: {self.controller.serial_queue.qsize()}")
            except queue.Empty:
                break

        if force_n is None:
            # No llegó ningún dato nuevo en este tick
            force_n = self._history[-1] if self._history else 0.
            print(f"[COLA] vacía — usando último valor: {force_n:.1f}")
        
        self._peak_n = max(self._peak_n, force_n)
        self._history.append(force_n)

        self.force_lbl.config(text=f"{force_n:.0f} N")
        self.peak_lbl.config(text=f"   Pico: {self._peak_n:.0f} N")
        self._draw_timer()
        self._redraw_graph()

        if self._elapsed_ms >= CALIB_DURATION * 1000:
            self._finish()
        else:
            self.after(INTERVAL, self._tick)

    def _finish(self):
        self._running = False
        self.controller.serial_send(CMD_STOP)    # ► Arduino: para de enviar
        self.start_btn.config(text="Iniciar de nuevo")
        self.controller.max_force.set(self._peak_n)
        self.result_lbl.config(
            text=f"✓  Fuerza máxima registrada: {self._peak_n:.0f} N")
        self.instr.config(text="Calibración completada. Puedes ir a Registro.")
        self._draw_timer(full=True)

    def _reset(self):
        self._elapsed_ms = 0
        self._history    = []
        self._peak_n     = 0.0
        self.force_lbl.config(text="-- N")
        self.peak_lbl.config(text="   Pico: -- N")
        self.result_lbl.config(text="")
        self.start_btn.config(text="Iniciar")
        self._draw_timer()
        draw_graph(self.graph, [], y_max=300)

    # ── eje Y adaptativo ──────────────────────────────────────────────────────
    def _y_max(self):
        """
        Calcula el techo del eje Y redondeando el pico al siguiente múltiplo
        de 50 N, con un mínimo de 100 N para que la gráfica no sea diminuta
        al inicio.
        """
        if not self._history:
            return 100
        peak = max(self._history)
        # Redondear hacia arriba al siguiente múltiplo de 50
        return max(100, math.ceil(peak / 50) * 50)

    def _redraw_graph(self):
        y_max = self._y_max()
        # Generar ticks cada 50 N
        ticks = list(range(0, y_max + 1, 50))
        draw_graph(self.graph, self._history,
                   y_max=y_max, y_ticks=ticks)

    # ── barra de tiempo ───────────────────────────────────────────────────────
    def _draw_timer(self, full=False):
        elapsed = CALIB_DURATION * 1000 if full else self._elapsed_ms
        draw_timer_bar(self.bar_cv, self.timer_lbl,
                       elapsed, CALIB_DURATION * 1000)
        
    def _siguiente(self):
        ctrl = self.controller
        print(ctrl.indice_actual)
    
        if ctrl.indice_actual >= len(ctrl.orden_sesion):
            ctrl.show_frame("MenuPage")     
            #ctrl.orden_sesion = []
            return
    
        siguiente_cond = ctrl.orden_sesion[ctrl.indice_actual]
    
        # Mapeo condición a frame
        mapa = {
            "Reposo" : "REPage",
            "MI"     : "MIPage",
            "ME"     : "RegistroPage",
        }
        
        ctrl.indice_actual += 1
        ctrl.show_frame(mapa[siguiente_cond])
        

# ══════════════════════════════════════════════════════════════════════════════
class RegistroPage(tk.Frame):
    """
    Ensayo de 30 segundos.
    Muestra la fuerza en % del máximo calibrado.
    Zona objetivo: 20–40 %.
    """
    def __init__(self, parent, controller):
        super().__init__(parent, bg=BG_LIGHT)
        self.controller  = controller
        self._running    = False
        self._elapsed_ms = 0
        self._history    = []   # valores en %

        # Header
        hdr = tk.Frame(self, bg=BG_LIGHT)
        hdr.pack(fill="x", padx=70, pady=(50, 0))
        tk.Label(hdr, text="M E", font=F_HEAD, fg=NAVY,
                 bg=BG_LIGHT).pack(side="left")
        tk.Button(hdr, text="← Menú", command=self._go_menu,
                  bg=BG_LIGHT, fg=TEXT_MUTED, font=F_LABEL,
                  relief="flat", cursor="hand2").pack(side="right")
        # Botón al fondo derecho
        bot = tk.Frame(self, bg=BG_LIGHT)
        bot.pack(side="bottom", fill="x", padx=70, pady=(0,30))
        tk.Button(bot, text="Continuar →",
                  command=lambda: self._siguiente(),
                  bg=BG_LIGHT, fg=TEXT_MUTED, font=F_LABEL,
                  relief="flat", padx=30, pady=0, cursor="hand2").pack(side="right")
        self.start_btn = tk.Button(bot, text="Iniciar",
                                    command=self._toggle,
                                    bg=GREEN_MAIN, fg="white", font=F_BTN,
                                    relief="flat", padx=30, pady=12, cursor="hand2",
                                    activebackground="#5aab35", activeforeground="white")
        self.start_btn.pack(side="left")

        # Instrucción
        self.instr = tk.Label(self,
            text=f"Presiona INICIAR y mantén tu fuerza en la zona verde durante {REGISTRO_DURATION} segundos.",
            font=F_LABEL, fg=TEXT_MUTED, bg=BG_LIGHT)
        self.instr.pack(pady=(8, 0))

        # Barra de tiempo
        timer_wrap = tk.Frame(self, bg=BG_LIGHT)
        timer_wrap.pack(fill="x", padx=70, pady=(18, 0))
        timer_hdr = tk.Frame(timer_wrap, bg=BG_LIGHT)
        timer_hdr.pack(fill="x")
        tk.Label(timer_hdr, text="Tiempo del ensayo",
                  font=F_SMALL, fg=TEXT_MUTED, bg=BG_LIGHT).pack(side="left")
        self.timer_lbl = tk.Label(timer_hdr, text=f"0 / {REGISTRO_DURATION} s",
                                  font=F_TIMER, fg=NAVY, bg=BG_LIGHT)
        self.timer_lbl.pack(side="right")
        
        
        
        self.bar_cv = tk.Canvas(timer_wrap, height=40,
                                bg=BG_LIGHT, highlightthickness=0)
        self.bar_cv.pack(fill="x", pady=(8, 0))
        self.bar_cv.bind("<Configure>", lambda e: self._draw_timer())
        
        # Gráfica
        self.graph = tk.Canvas(self, bg=BG_LIGHT, highlightthickness=0)
        self.graph.pack(fill="both", expand=True, padx=50, pady=(8, 0))
        self.graph.bind("<Configure>", lambda e: self._redraw_graph())

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def on_enter(self):
        if self._running:
            self._stop()

        self._reset()

    def _go_menu(self):
        if self._running:
            self._stop()
        self.controller.show_frame("MenuPage")

    # ── control ───────────────────────────────────────────────────────────────
    def _toggle(self):
        if not self._running:
            self._begin()
        else:
            self._stop()

    def _begin(self):
        self._running    = True
        self._start_time = time.perf_counter()   # momento exacto de inicio
        self.controller.play_sound("start")
        self._elapsed_ms = 0
        self._history    = []
        self.start_btn.config(text="Detener")
        self.instr.config(text="¡Mantén la fuerza en la zona verde!")
        # Vaciar cola por si quedaron datos de una sesión anterior
        while not self.controller.serial_queue.empty():
            self.controller.serial_queue.get_nowait()
        self.controller.serial_send(CMD_START)   # ► Arduino: empieza a enviar
        self.controller.abrir_ventana_eeg(titulo="EEG — Registro (ME)")
        self._tick()

    def _stop(self):
        self._running = False
        self.controller.serial_send(CMD_STOP)    # ► Arduino: para de enviar
        self.start_btn.config(text="Iniciar")
        self.instr.config(
            text=f"Presiona INICIAR y mantén tu fuerza en la zona verde durante {REGISTRO_DURATION} segundos.")
        self.controller.cerrar_ventana_eeg(condicion="ME")

    def _tick(self):
        if not self._running:
            return
        INTERVAL = 10  # ms
        self._elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        
        ref = max(self.controller.max_force.get(), 200)       
        force_n = None
        while not self.controller.serial_queue.empty():
            try:
                force_n = self.controller.serial_queue.get_nowait()
                print(f"[COLA] dato leído: {force_n:.1f} | restantes en cola: {self.controller.serial_queue.qsize()}")
            except queue.Empty:
                break

        if force_n is None:
            # No llegó ningún dato nuevo en este tick
            last_pct = self._history[-1] if self._history else ref * 0.30 / ref * 100
            force_n  = last_pct / 100 * ref
            print(f"[COLA] vacía — usando último valor: {force_n:.1f}")


        pct = min(force_n / ref * 100, 100)
        self._history.append(pct)

        # color = GREEN_MAIN if 20 <= pct <= 40 else WARN_RED
        # self.num_lbl.config(text=f"{pct:.0f}", fg=color)

        self._draw_timer()
        self._redraw_graph()

        if self._elapsed_ms >= REGISTRO_DURATION * 1000:
            self._finish()
        else:
            self.after(INTERVAL, self._tick)
        
    def _finish(self):
        self._running = False
        self.controller.serial_send(CMD_STOP)    # ► Arduino: para de enviar
        self.controller.play_sound("stop")
        self.start_btn.config(text="Iniciar de nuevo")
        self.instr.config(text=f"Presiona INICIAR y mantén tu fuerza en la zona verde durante {REGISTRO_DURATION} segundos.")
        self._draw_timer(full=True)
        self.controller.cerrar_ventana_eeg()

    def _reset(self):
        self._elapsed_ms = 0
        self._history    = []
        self.start_btn.config(text="Iniciar")
        self._draw_timer()
        self._redraw_graph()

    # ── dibujo ───────────────────────────────────────────────────────────────
    def _draw_timer(self, full=False):
        elapsed = REGISTRO_DURATION * 1000 if full else self._elapsed_ms
        draw_timer_bar(self.bar_cv, self.timer_lbl,
                       elapsed, REGISTRO_DURATION * 1000)

    def _redraw_graph(self):
        draw_graph(self.graph, self._history,
               y_max=100,
               y_ticks=[0, 20, 40, 60, 80, 100],
               zone_lo=25, zone_hi=35, show_zone=True)
    
        # Dibujar el valor encima de la gráfica, esquina superior izquierda
        pct = self._history[-1] if self._history else None
        if pct is not None:
            color = GREEN_MAIN if 20 <= pct <= 40 else WARN_RED
            self.graph.create_text(80, 30,
                                   text=f"{pct:.0f}",
                                   font=F_NUM, fill=color, anchor="nw")
            self.graph.create_text(80, 170,
                                   text="% del máximo calibrado",
                                   font=F_SMALL, fill=TEXT_MUTED, anchor="nw") 
    
    def _siguiente(self):
        ctrl = self.controller
        
    
        if ctrl.indice_actual >= len(ctrl.orden_sesion):
            ctrl.show_frame("MenuPage")
            return
    
        siguiente_cond = ctrl.orden_sesion[ctrl.indice_actual]
    
        # Mapeo condición a frame
        mapa = {
            "Reposo" : "REPage",
            "MI"     : "MIPage",
            "ME"     : "RegistroPage",
        }
        
        ctrl.indice_actual += 1
        ctrl.show_frame(mapa[siguiente_cond])
            
        
# ══════════════════════════════════════════════════════════════════════════════
class MIPage(tk.Frame):
    """
    Genera sonido e instrucción
    Timer
    
    """
    def __init__(self, parent, controller):
        super().__init__(parent, bg=NAVY)
        self.controller  = controller
        self._running    = False
        self._elapsed_ms = 0


        # Header
        hdr = tk.Frame(self, bg=NAVY)
        hdr.pack(fill="x",padx=70, pady=(50, 0))

        tk.Button(hdr, text="← Menú", command=self._go_menu,
                  bg=NAVY, fg=TEXT_MUTED, font=F_LABEL,
                  relief="flat", cursor="hand2").pack(side="right")
        
        # Instrucción
        self.instr = tk.Label(self,
            text=f"Imagina el movimiento durante {REGISTRO_DURATION} segundos.",
            font=F_LABEL, fg=TEXT_MUTED, bg=NAVY)
        self.instr.pack(pady=(16, 0))
        
        # Botón al fondo derecho
        bot = tk.Frame(self, bg=NAVY)
        bot.pack(side="bottom", fill="x", padx=70, pady=30)
        tk.Button(bot, text="Continuar →",
                  command=lambda: self._siguiente(),
                  bg=NAVY, fg=TEXT_MUTED, font=F_LABEL,
                  relief="flat", padx=30, pady=12, cursor="hand2").pack(side="right")
        
        
        self.start_btn = tk.Button(bot, text="Iniciar", command=self._toggle,
                                    bg=NAVY, fg="white", font=F_BTN,
                                    relief="flat", padx=30, pady=12, cursor="hand2")
        self.start_btn.pack(side="left")
        self.result_lbl = tk.Label(bot, text="", font=F_LABEL, fg=NAVY, bg=NAVY)
        self.result_lbl.pack(side="left", padx=30)



    # ── lifecycle ─────────────────────────────────────────────────────────────
    def on_enter(self):
        if self._running:
            self._stop()
        self._reset()

    def _go_menu(self):
        if self._running:
            self._stop()
        self.controller.show_frame("MenuPage")

    # ── control ───────────────────────────────────────────────────────────────
    def _toggle(self):
        if not self._running:
            self._begin()
        else:
            self._stop()

    def _begin(self):
        self._running    = True
        self._start_time = time.perf_counter()
        self.controller.play_sound("start")
        self._elapsed_ms = 0
        self._history    = []
        self._peak_n     = 0.0
        self.result_lbl.config(text="")
        self.start_btn.config(text="Detener")
        self._tick()
        self.controller.abrir_ventana_eeg(titulo="EEG — Imaginación Motora (MI)")
        
    def _stop(self):
        self._running = False
        self.start_btn.config(text="Iniciar")
        self.instr.config(
            text=f"Presiona INICIAR e imagina el movimiento durante {REGISTRO_DURATION} segundos.")
        self.controller.cerrar_ventana_eeg(condicion="MI")

    def _tick(self):
        if not self._running:
            return
        INTERVAL = 10  # ms — frecuencia de refresco de la interfaz
        
        # Tiempo real transcurrido en milisegundos
        self._elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        if self._elapsed_ms >= REGISTRO_DURATION * 1000:
            self._finish()
        else:
            self.after(INTERVAL, self._tick)

    def _finish(self):
        self._running = False
        self.controller.play_sound("finish")
        self.start_btn.config(text="Iniciar de nuevo")
        self.instr.config(text=f"Presiona INICIAR e imagina el movimiento durante {REGISTRO_DURATION} segundos.")
        self.controller.cerrar_ventana_eeg()

    def _reset(self):
        self._elapsed_ms = 0
        self.start_btn.config(text="Iniciar")

    def _siguiente(self):
        ctrl = self.controller
        if ctrl.indice_actual >= len(ctrl.orden_sesion):
            ctrl.show_frame("MenuPage")
            return
        siguiente_cond = ctrl.orden_sesion[ctrl.indice_actual]
        mapa = {"Reposo": "REPage", "MI": "MIPage", "ME": "RegistroPage"}
        ctrl.indice_actual += 1
        ctrl.show_frame(mapa[siguiente_cond])

# ══════════════════════════════════════════════════════════════════════════════
class REPage(tk.Frame):
    """
    Genera sonido e instrucción
    Timer
    
    """
    def __init__(self, parent, controller):
        super().__init__(parent, bg=GRAY_TEXT)
        self.controller  = controller
        self._running    = False
        self._elapsed_ms = 0


        # Header
        hdr = tk.Frame(self, bg=GRAY_TEXT)
        hdr.pack(fill="x",padx=70, pady=(50, 0))

        tk.Button(hdr, text="← Menú", command=self._go_menu,
                  bg=GRAY_TEXT, fg=CARD_WHITE, font=F_LABEL,
                  relief="flat", cursor="hand2").pack(side="right")
        
        # Instrucción
        self.instr = tk.Label(self,
            text=f"Mantente en REPOSO durante {REGISTRO_DURATION} segundos.",
            font=F_LABEL, fg=CARD_WHITE, bg=GRAY_TEXT)
        self.instr.pack(pady=(16, 0))
        
        # Botón al fondo derecho
        bot = tk.Frame(self, bg=GRAY_TEXT)
        bot.pack(side="bottom", fill="x", padx=70, pady=30)
        tk.Button(bot, text="Continuar →",
                  command=lambda: self._siguiente(),
                  bg=GRAY_TEXT, fg=CARD_WHITE, font=F_LABEL,
                  relief="flat", padx=30, pady=12, cursor="hand2").pack(side="right")
        
        
        self.start_btn = tk.Button(bot, text="Iniciar", command=self._toggle,
                                    bg=GRAY_TEXT, fg="white", font=F_BTN,
                                    relief="flat", padx=30, pady=12, cursor="hand2")
        self.start_btn.pack(side="left")
        self.result_lbl = tk.Label(bot, text="", font=F_LABEL, fg=GRAY_TEXT, bg=GRAY_TEXT)
        self.result_lbl.pack(side="left", padx=30)



    # ── lifecycle ─────────────────────────────────────────────────────────────
    def on_enter(self):
        if self._running:
            self._stop()
        self._reset()

    def _go_menu(self):
        if self._running:
            self._stop()
        self.controller.show_frame("MenuPage")

    # ── control ───────────────────────────────────────────────────────────────
    def _toggle(self):
        if not self._running:
            self._begin()
        else:
            self._stop()

    def _begin(self):
        self._running    = True
        self._start_time = time.perf_counter()
        self.controller.play_sound("start")
        self._elapsed_ms = 0
        self._history    = []
        self._peak_n     = 0.0
        self.result_lbl.config(text="")
        self.start_btn.config(text="Detener")
        self._tick()
        self.controller.abrir_ventana_eeg(titulo="EEG — Reposo")
        
    def _stop(self):
        self._running = False
        self.start_btn.config(text="Iniciar")
        self.instr.config(
            text=f"Presiona INICIAR e imagina el movimiento durante {REGISTRO_DURATION} segundos.")
        self.controller.cerrar_ventana_eeg(condicion="Reposo")

    def _tick(self):
        if not self._running:
            return
        INTERVAL = 10  # ms — frecuencia de refresco de la interfaz
        
        # Tiempo real transcurrido en milisegundos
        self._elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        if self._elapsed_ms >= REGISTRO_DURATION * 1000:
            self._finish()
        else:
            self.after(INTERVAL, self._tick)

    def _finish(self):
        self._running = False
        self.controller.play_sound("finish")
        self.start_btn.config(text="Iniciar de nuevo")
        self.instr.config(text=f"Presiona INICIAR e imagina el movimiento durante {REGISTRO_DURATION} segundos.")
        self.controller.cerrar_ventana_eeg()

    def _reset(self):
        self._elapsed_ms = 0
        self.start_btn.config(text="Iniciar")
        
    def _siguiente(self):
        ctrl = self.controller
        if ctrl.indice_actual >= len(ctrl.orden_sesion):
            ctrl.show_frame("MenuPage")
            return
        siguiente_cond = ctrl.orden_sesion[ctrl.indice_actual]
        mapa = {"Reposo": "REPage", "MI": "MIPage", "ME": "RegistroPage"}
        ctrl.indice_actual += 1
        ctrl.show_frame(mapa[siguiente_cond])




# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = CoherenciaBCI()
    app.mainloop()