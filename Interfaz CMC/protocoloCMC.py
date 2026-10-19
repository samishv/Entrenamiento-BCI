# -*- coding: utf-8 -*-
"""
Protocolo de coherencia corticomuscular (CMC): EEG + EMG + dinamómetro.

Flujo de una ejecución (= 1 bloque):
    menú de sesión -> se abre ventanaeeg.py -> CALIBRACIÓN MVC (pausa hasta ESPACIO)
    -> instrucciones (ESPACIO) -> sync inicial -> 20 trials -> sync final -> fin

Trial (≈ 19-20 s):
    ATENCIÓN (cruz '+'):  jitter 1-2 s  ->  fijación 3 s
    -> beep + cue 10 s (gráfica de fuerza con zona objetivo 15 % ± 5 % MVC)
    -> REPOSO 5 s (pantalla negra, fijo)

Señales / sincronización:
    * EEG            -> stream LSL del amplificador (LabRecorder)
    * Dinamómetro    -> este script lo lee por serial y lo publica como stream
                        LSL 'Dinamometro' (timestamp = local_clock() al leer cada muestra)
    * Marcadores     -> stream LSL 'CMC_Markers'
    * EMG (FreeEMG)  -> se graba aparte; se alinea offline con la fuerza
                        (ver bloques de 'sync' y los apretones de la calibración MVC)
    LabRecorder debe estar grabando (Update -> seleccionar EEG, Dinamometro y
    CMC_Markers -> Start) ANTES de pulsar el primer ESPACIO de la calibración.

Marcadores:
    tara_start | mvc_calib_start | mvc_calib_end | mvc_value=<raw> |
    sync_pre_start | sync_pre_end | block_start |
    jitter_start | fix_start | cue_start | reposo_start | reposo_3s |
    sync_post_start | sync_post_end | block_end | block_abort

"""
import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np

from psychopy import prefs
prefs.hardware['audioLib'] = ['ptb', 'sounddevice']   # antes de importar sound
prefs.hardware['audioLatencyMode'] = 3
from psychopy import visual, core, event, sound, gui

from pylsl import StreamInfo, StreamOutlet, local_clock, resolve_byprop

# ─── Parámetros del paradigma ────────────────────────────────────────────────
JITTER_MIN_S   = 1.0        # atención: cruz con duración aleatoria...
JITTER_MAX_S   = 2.0
FIJACION_S     = 3.0        # ...y luego 3 s de fijación (cruz)
TAREA_S        = 10.0       # cue + beep + gráfica de fuerza
REPOSO_S       = 5.0        # pantalla negra, fijo
REPOSO_MARCA_S = 3.0        # marcador 'reposo_3s' a los 3 s de reposo_start
N_TRIALS       = 20         # trials por bloque
BEEP_HZ, BEEP_S = 1000, 0.3

# ─── Fuerza ──────────────────────────────────────────────────────────────────
OBJ_PCT        = 15.0       # objetivo: 15 % del MVC
TOL_PCT        = 5.0        # ± 5 % del MVC  -> zona 10-20 %
FS_DINO        = 200.0      # Hz; debe coincidir con FS_HZ del sketch del Arduino
Y_MAX_PCT      = 40.0       # tope del eje Y de la gráfica (% MVC)
HIST_S         = 5.0        # segundos de historia visibles en la gráfica
SUAVE_DISP_S   = 0.15       # promedio móvil para mostrar / evaluar rango
SUAVE_MVC_S    = 0.5        # promedio móvil para detectar el pico del MVC
TARE_S         = 2.0        # reposo para medir la línea base
SYNC_S         = 8.0        # duración de cada bloque de sync (apretones rápidos)

FUENTE = 'Arial'

TEXTO_INSTR = (
    "1) Mira la cruz y evita moverte.\n\n"
    "2) Al escuchar el beep, aprieta el dinamómetro y mantén la línea\n"
    "dentro de la zona verde durante toda la tarea.\n\n"
    "3) Durante la pantalla negra, relájate.\n\n"
)


class Abortar(Exception):
    pass


# ═════════════════════════════════════════════════════════════════════════════
#  Dinamómetro: lectura serial -> buffer + stream LSL
# ═════════════════════════════════════════════════════════════════════════════
_NUM = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')


def parse_linea(raw):
    """Convierte una línea serial (bytes) en float. AJUSTAR al formato real del
    dinamómetro (p. ej. "12.34\\r\\n", "F:12.34", "12.34,0.1"...).
    Por defecto toma el primer número que encuentre en la línea."""
    m = _NUM.search(raw.decode('ascii', errors='ignore'))
    return float(m.group()) if m else None


class FuenteFuerza:
    """Interfaz común: buffer de (t_lsl, valor_raw) + consultas."""

    def __init__(self, outlet):
        self.outlet = outlet
        self.buf = deque(maxlen=60000)
        self.lock = threading.Lock()

    def _registrar(self, t, v):
        self.outlet.push_sample([v], t)
        with self.lock:
            self.buf.append((t, v))

    def poll(self):            # se llama cada frame (solo lo usa la fuente simulada)
        pass

    def ultimas(self, seg):
        corte = local_clock() - seg
        out = []
        with self.lock:
            for t, v in reversed(self.buf):
                if t < corte:
                    break
                out.append((t, v))
        out.reverse()
        return out

    def suave(self, seg):
        """Media de los últimos `seg` s, o None si no hay datos recientes."""
        d = self.ultimas(seg)
        return float(np.mean([v for _, v in d])) if d else None

    def cerrar(self):
        pass


class FuerzaSerial(FuenteFuerza):
    def __init__(self, puerto, baud, outlet):
        super().__init__(outlet)
        import serial                                   # pip install pyserial
        self._serial = serial
        # Abrir el puerto resetea el Arduino Uno (línea DTR): arranca el sketch de cero.
        self.ser = serial.Serial(puerto, baud, timeout=0.05)
        self._esperar_listo()
        self.ser.reset_input_buffer()
        self.ser.write(b's')                            # inicia el envío de datos
        self._stop = threading.Event()
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _esperar_listo(self, timeout=8.0):
        """Espera la línea 'READY' que el sketch imprime al terminar setup()."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.ser.readline().strip() == b'READY':
                print("[INFO] Dinamómetro listo (READY)")
                return
        print("[AVISO] El Arduino no envió READY; se continúa de todos modos "
              "(¿sketch antiguo?)")

    def _loop(self):
        while not self._stop.is_set():
            try:
                linea = self.ser.readline()
            except self._serial.SerialException as e:
                print(f"[ERROR] Serial: {e}")
                break
            t = local_clock()                           # estampa al recibir la línea
            if not linea:
                continue
            v = parse_linea(linea)
            if v is not None:
                self._registrar(t, v)

    def cerrar(self):
        self._stop.set()
        self.th.join(timeout=1)
        try:
            self.ser.write(b'p')                        # pausa el envío
        except Exception:
            pass
        self.ser.close()


class FuerzaSimulada(FuenteFuerza):
    """Pruebas sin hardware: la fuerza (0-100) sigue la posición Y del mouse."""

    def __init__(self, win, outlet):
        super().__init__(outlet)
        self.mouse = event.Mouse(win=win)

    def poll(self):
        y = self.mouse.getPos()[1]                      # unidades 'height' (-0.5..0.5)
        self._registrar(local_clock(), float(max(0.0, (y + 0.5) * 100.0)))


# ═════════════════════════════════════════════════════════════════════════════
#  Gráfica de retroalimentación (durante el cue)
# ═════════════════════════════════════════════════════════════════════════════
class Retroalimentacion:
    W, H = 1.2, 0.6                                     # área del gráfico (height units)
    VERDE, ROJO = '#2ECC71', '#E74C3C'

    def __init__(self, win, fuente, base, mvc):
        self.win, self.fuente, self.base, self.mvc = win, fuente, base, mvc
        self.lo, self.hi = OBJ_PCT - TOL_PCT, OBJ_PCT + TOL_PCT
        W, H = self.W, self.H
        yl, yh = self._y(self.lo), self._y(self.hi)
        self.marco = visual.Rect(win, width=W, height=H, fillColor='#101010',
                                 lineColor='#888888', lineWidth=2)
        self.banda = visual.Rect(win, width=W, height=yh - yl, pos=(0, (yl + yh) / 2),
                                 fillColor=self.VERDE, lineColor=None, opacity=0.35)
        self.linea_lo = visual.Line(win, start=(-W / 2, yl), end=(W / 2, yl),
                                    lineColor=self.VERDE, lineWidth=1.5)
        self.linea_hi = visual.Line(win, start=(-W / 2, yh), end=(W / 2, yh),
                                    lineColor=self.VERDE, lineWidth=1.5)
        self.txt_lo = visual.TextStim(win, text=f'{self.lo:.0f} %', pos=(W / 2 + 0.07, yl),
                                      height=0.03, color=self.VERDE, font=FUENTE)
        self.txt_hi = visual.TextStim(win, text=f'{self.hi:.0f} %', pos=(W / 2 + 0.07, yh),
                                      height=0.03, color=self.VERDE, font=FUENTE)
        self.titulo = visual.TextStim(win, text='Mantén la línea en la zona verde',
                                      pos=(0, H / 2 + 0.1), height=0.04,
                                      color='white', font=FUENTE)
        self.sin_senal = visual.TextStim(win, text='SIN SEÑAL DEL DINAMÓMETRO',
                                         pos=(0, 0), height=0.04, color=self.ROJO,
                                         font=FUENTE)
        self.traza = visual.ShapeStim(win, vertices=[(-W / 2, -H / 2), (W / 2, -H / 2)],
                                      closeShape=False, lineWidth=4,
                                      lineColor='white', fillColor=None)
        self.punto = visual.Circle(win, radius=0.016, fillColor='white', lineColor=None)
        self.reset()

    def _y(self, p):
        return -self.H / 2 + self.H * min(max(p, 0.0), Y_MAX_PCT) / Y_MAX_PCT

    def pct(self, v):
        return (v - self.base) / self.mvc * 100.0

    def reset(self):
        self.n = self.n_en = self.n_val = 0
        self.suma = 0.0

    def resumen(self):
        pct_en = 100.0 * self.n_en / self.n if self.n else float('nan')
        media = self.suma / self.n_val if self.n_val else float('nan')
        return pct_en, media

    def draw(self):
        W = self.W
        now = local_clock()
        self.marco.draw(); self.banda.draw()
        self.linea_lo.draw(); self.linea_hi.draw()
        self.txt_lo.draw(); self.txt_hi.draw(); self.titulo.draw()

        self.n += 1
        actual = self.fuente.suave(SUAVE_DISP_S)
        datos = self.fuente.ultimas(HIST_S)
        if actual is None or len(datos) < 2:
            self.sin_senal.draw()
            return
        p = self.pct(actual)
        en = self.lo <= p <= self.hi
        self.n_en += int(en)
        self.n_val += 1
        self.suma += p

        step = max(1, len(datos) // 300)                # ~300 puntos máx.
        datos = datos[::step]
        t0 = now - HIST_S
        xs = [-W / 2 + W * (t - t0) / HIST_S for t, _ in datos]
        ys = [self._y(self.pct(v)) for _, v in datos]
        color = self.VERDE if en else self.ROJO
        self.traza.vertices = list(zip(xs, ys))
        self.traza.lineColor = color
        self.traza.draw()
        self.punto.pos = (xs[-1], self._y(p))
        self.punto.fillColor = color
        self.punto.draw()


# ═════════════════════════════════════════════════════════════════════════════
#  Utilidades
# ═════════════════════════════════════════════════════════════════════════════
def lanzar_visor_eeg(stream_name):
    """Abre ventanaeeg.py como proceso independiente."""
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ventanaeeg.py')
    if not os.path.exists(ruta):
        print(f"[AVISO] No se encontró {ruta}; no se abre el visor EEG")
        return None
    try:
        return subprocess.Popen([sys.executable, ruta, stream_name])
    except Exception as e:
        print(f"[AVISO] No se pudo abrir el visor EEG: {e}")
        return None


def menu_inicio(a):
    """Diálogo de inicio: edita los argumentos de sesión en el namespace `a`."""
    k_pid, k_ses, k_blo = 'Participante (pid)', 'Sesión', 'Bloque'
    k_pan, k_n = 'Pantalla del sujeto (0, 1...)', 'Trials por bloque'
    k_eeg, k_pto, k_baud = 'Nombre del stream EEG', 'Puerto serial dinamómetro', 'Baudrate'
    k_mvc = 'MVC previo en unidades raw (vacío = calibrar)'
    k_out, k_seed = 'Carpeta de salida', 'Seed (vacío = aleatoria)'
    k_sim, k_win = 'Simular dinamómetro con mouse (pruebas)', 'Modo ventana (pruebas)'

    d = {
        k_pid: a.pid, k_ses: a.sesion, k_blo: a.bloque, k_pan: a.pantalla,
        k_n: a.n_trials, k_eeg: a.eeg_stream, k_pto: a.puerto, k_baud: a.baud,
        k_mvc: '' if a.mvc is None else a.mvc,
        k_out: a.salida, k_seed: '' if a.seed is None else a.seed,
        k_sim: a.simular, k_win: a.ventana,
    }
    dlg = gui.DlgFromDict(d, title='Protocolo CMC', order=list(d.keys()))
    if not dlg.OK:
        print("[INFO] Cancelado desde el menú")
        core.quit()
        sys.exit()

    a.pid = str(d[k_pid]).strip() or 'X'
    a.sesion = str(d[k_ses]).strip()
    a.bloque = str(d[k_blo]).strip()
    a.pantalla = int(d[k_pan])
    a.n_trials = int(d[k_n])
    a.eeg_stream = str(d[k_eeg]).strip()
    a.puerto = str(d[k_pto]).strip()
    a.baud = int(d[k_baud])
    m = str(d[k_mvc]).strip()
    a.mvc = float(m) if m else None
    a.salida = str(d[k_out]).strip() or './Data'
    s = str(d[k_seed]).strip()
    a.seed = int(s) if s else None
    a.simular = bool(d[k_sim])
    a.ventana = bool(d[k_win])
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', default='X')
    ap.add_argument('--sesion', default='D1')
    ap.add_argument('--bloque', default='B1')
    ap.add_argument('--pantalla', type=int, default=0, help='monitor del sujeto')
    ap.add_argument('--n_trials', type=int, default=N_TRIALS)
    ap.add_argument('--eeg_stream', default='streamTEST')
    ap.add_argument('--puerto', default='COM3', help='puerto serial del dinamómetro')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--mvc', type=float, default=None,
                    help='MVC previo (raw, sin tara); si se da, omite el apretón de calibración')
    ap.add_argument('--salida', default='Protocolo CMC/Signals/LOGS')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--simular', action='store_true', help='fuerza simulada con el mouse')
    ap.add_argument('--ventana', action='store_true', help='no pantalla completa (pruebas)')
    ap.add_argument('--sin_menu', action='store_true', help='omite el menú de inicio')
    ap.add_argument('--sin_visor', action='store_true', help='no abrir la ventana de EEG')
    ap.add_argument('--sin_sync', action='store_true', help='omite los bloques de sync')
    a = ap.parse_args()
    if not a.sin_menu:
        a = menu_inicio(a)

    seed = a.seed if a.seed is not None else int(datetime.now().timestamp())
    rng = random.Random(seed)
    os.makedirs(a.salida, exist_ok=True)
    base = f"{a.pid}_{datetime.now():%y%m%d}_CMC_{a.sesion}_{a.bloque}"
    sid = f'{a.pid}_{a.sesion}_{a.bloque}'

    # ── LSL: marcadores, dinamómetro y comprobación del EEG ─────────────────
    outlet = StreamOutlet(StreamInfo('CMC_Markers', 'Markers', 1, 0, 'string', f'cmc_mk_{sid}'))
    # srate nominal = la del sketch del Arduino (permite dejitter al cargar con pyxdf)
    info_f = StreamInfo('Dinamometro', 'Force', 1, FS_DINO, 'float32', f'cmc_force_{sid}')
    ch = info_f.desc().append_child('channels').append_child('channel')
    ch.append_child_value('label', 'force')
    ch.append_child_value('unit', 'N')
    fuerza_outlet = StreamOutlet(info_f)

    eeg_ok = bool(resolve_byprop('name', a.eeg_stream, timeout=3))
    if not eeg_ok:
        print(f"[AVISO] No se encontró el stream EEG '{a.eeg_stream}'")

    ultimo = {'t': None}

    def enviar(m):                       # se ejecuta justo tras el flip
        t = local_clock()
        outlet.push_sample([m], t)
        ultimo['t'] = t

    # ── Ventana, estímulos, sonido ──────────────────────────────────────────
    win = visual.Window(fullscr=not a.ventana, screen=a.pantalla, color='black',
                        units='height', allowGUI=False)
    fps = win.getActualFrameRate() or 60.0
    win.recordFrameIntervals = True
    win.refreshThreshold = 1.0 / fps * 1.2
    nf = lambda s: int(round(s * fps))
    print(f"[INFO] fps medidos: {fps:.2f} | seed: {seed}")

    fix = visual.TextStim(win, text='+', height=0.15, color='white')
    beep = sound.Sound(BEEP_HZ, secs=BEEP_S, stereo=True, hamming=True)
    nada = lambda: None

    fuente = None          # se crea dentro del try (si falla el puerto, se cierra limpio)
    cal = {'base': 0.0, 'ruido': 0.0, 'mvc': None}
    visor = None
    msg_final = None
    f = None

    def fase(n_frames, dibujar, marcador, con_beep=False):
        """Dibuja n_frames; marcador (y beep) salen en el primer flip."""
        for i in range(n_frames):
            if fuente is not None:
                fuente.poll()
            dibujar()
            if i == 0:
                win.callOnFlip(enviar, marcador)
                if con_beep:
                    beep.play(when=win.getFutureFlipTime(clock='ptb'))
            win.flip()
            if event.getKeys(['escape']):
                raise Abortar()
        return ultimo['t']

    def dib_texto(msg, pos=(0, 0), height=0.04):
        t = visual.TextStim(win, text=msg, pos=pos, height=height, color='white',
                            font=FUENTE, wrapWidth=1.5)
        return t.draw

    def esperar(msg, teclas=('space',)):
        """Pantalla de texto que pausa la interfaz hasta pulsar una de `teclas`."""
        t = visual.TextStim(win, text=msg, height=0.04, color='white',
                            font=FUENTE, wrapWidth=1.5)
        t.draw(); win.flip()
        event.clearEvents()
        k = event.waitKeys(keyList=list(teclas) + ['escape'])
        if 'escape' in k:
            raise Abortar()
        return k[0]

    # Barra vertical (calibración y sync)
    barra_marco = visual.Rect(win, width=0.14, height=0.6, pos=(0, -0.05),
                              lineColor='white', fillColor=None, lineWidth=2)
    barra_fill = visual.Rect(win, width=0.14, height=0.01, pos=(0, -0.35),
                             anchor='bottom', fillColor='#4FC3F7', lineColor=None)

    def dib_barra(frac):
        frac = min(max(frac, 0.0), 1.0)
        barra_fill.height = max(0.001, 0.6 * frac)
        barra_marco.draw(); barra_fill.draw()

    # ── Calibración MVC ─────────────────────────────────────────────────────
    def medir_tara():
        fase(nf(TARE_S), dib_texto('Relaja la mano.\n\nNo aprietes el dinamómetro.'),
             'tara_start')
        d = [v for _, v in fuente.ultimas(TARE_S * 0.8)]
        if len(d) < 5:
            print("[ERROR] No llegan datos del dinamómetro")
            esperar("No llegan datos del dinamómetro.\n\nRevisa el puerto y reinicia.")
            raise Abortar()
        cal['base'] = float(np.median(d))
        cal['ruido'] = float(np.std(d))
        print(f"[INFO] Tara: base={cal['base']:.4f} ruido(std)={cal['ruido']:.4f}")

    def calibrar():
        if a.mvc is not None:                           # MVC previo: solo tara
            esperar("Calibración (MVC previo)\n\nPresiona ESPACIO para medir la tara")
            medir_tara()
            cal['mvc'] = a.mvc - cal['base']
            enviar(f"mvc_value={a.mvc:.4f}")
            return

        txt_top = visual.TextStim(win, text='Aprieta el dinamómetro con la MÁXIMA fuerza.\n'
                                  'Puedes repetirlo varias veces.',
                                  pos=(0, 0.38), height=0.04, color='white',
                                  font=FUENTE, wrapWidth=1.5)
        txt_pico = visual.TextStim(win, text='', pos=(0, -0.45), height=0.035,
                                   color='#BDBDBD', font=FUENTE)
        while True:
            esperar("CALIBRACIÓN DE FUERZA MÁXIMA (MVC)\n\n"
                    "1) Primero relaja la mano 2 s (calibración a cero).\n"
                    "2) Luego aprieta lo más fuerte que puedas.\n\n"
                    "Presiona ESPACIO para comenzar")
            medir_tara()

            pico, primero = 0.0, True
            event.clearEvents()
            while True:
                fuente.poll()
                v = fuente.suave(SUAVE_MVC_S)
                if v is not None:
                    pico = max(pico, v - cal['base'])
                act = fuente.suave(SUAVE_DISP_S)
                frac = ((act - cal['base']) / pico) if (act is not None and pico > 0) else 0.0
                txt_pico.text = (f"Pico: {pico:.2f}   |   R = reiniciar pico   |   "
                                 f"ESPACIO = terminar")
                txt_top.draw(); dib_barra(frac); txt_pico.draw()
                if primero:
                    win.callOnFlip(enviar, 'mvc_calib_start')
                    primero = False
                win.flip()
                teclas = event.getKeys(['space', 'escape', 'r'])
                if 'escape' in teclas:
                    raise Abortar()
                if 'r' in teclas:
                    pico = 0.0
                if 'space' in teclas and pico > 10 * cal['ruido'] and pico > 0:
                    break
            enviar('mvc_calib_end')
            enviar(f"mvc_value={pico + cal['base']:.4f}")
            k = esperar(f"MVC = {pico:.2f}  (raw, sin tara)\n"
                        f"Zona objetivo: {OBJ_PCT - TOL_PCT:.0f}-{OBJ_PCT + TOL_PCT:.0f} % "
                        f"= {pico * (OBJ_PCT - TOL_PCT) / 100:.2f}-"
                        f"{pico * (OBJ_PCT + TOL_PCT) / 100:.2f}\n\n"
                        "ESPACIO = aceptar   |   R = repetir calibración",
                        teclas=('space', 'r'))
            if k == 'space':
                cal['mvc'] = pico
                break
        print(f"[INFO] MVC (sobre la tara) = {cal['mvc']:.4f}")

    # ── Sync (apretones rápidos para alinear EMG con fuerza) ────────────────
    def sync(tag):
        top = visual.TextStim(win, text='Cuando suene el beep, aprieta el dinamómetro\n'
                              'fuerte y rápido 3 veces, y relaja.',
                              pos=(0, 0.38), height=0.04, color='white',
                              font=FUENTE, wrapWidth=1.5)

        def dib():
            top.draw()
            v = fuente.suave(SUAVE_DISP_S)
            dib_barra(0.0 if v is None else (v - cal['base']) / cal['mvc'])
        fase(nf(SYNC_S), dib, f'sync_{tag}_start', con_beep=True)
        fase(nf(0.5), nada, f'sync_{tag}_end')

    def pantalla_inicio(msg):
        titulo = visual.TextStim(win, text=f'BLOQUE {a.bloque}', pos=(0, 0.38),
                                 height=0.065, color='#4FC3F7', font=FUENTE, bold=True)
        panel = visual.Rect(win, pos=(0, -0.02), width=1.8, height=0.6,
                            fillColor='#181818', lineColor='#4FC3F7', lineWidth=2)
        instr = visual.TextStim(win, text=msg, pos=(0, -0.02), height=0.034,
                                color='white', font=FUENTE, wrapWidth=1.6)
        ind = visual.TextStim(win, text='Presiona ESPACIO para comenzar', pos=(0, -0.40),
                              height=0.030, color='#BDBDBD', font=FUENTE)
        event.clearEvents()
        while True:
            panel.draw(); titulo.draw(); instr.draw(); ind.draw()
            win.flip()
            teclas = event.getKeys(['space', 'escape'])
            if 'escape' in teclas:
                raise Abortar()
            if 'space' in teclas:
                break

    def esperar_cierre(msg):
        """Pantalla final: espera ESPACIO (o ESC) sin lanzar Abortar.
        Da tiempo a pulsar Stop en LabRecorder antes de cerrar los outlets."""
        t = visual.TextStim(win, text=msg, height=0.04, color='white', wrapWidth=1.5)
        t.draw(); win.flip()
        print("[INFO] >>> Pulsa STOP en LabRecorder (y detén FreeEMG) y luego "
              "ESPACIO en la ventana de PsychoPy para cerrar.")
        event.clearEvents()
        event.waitKeys(keyList=['space', 'escape'])

    def guardar_meta(t_block_lsl):
        meta = {
            'pid': a.pid, 'sesion': a.sesion, 'bloque': a.bloque, 'seed': seed,
            'fps': fps, 'n_trials': a.n_trials,
            'mvc_sobre_tara': cal['mvc'], 'tara_raw': cal['base'],
            'ruido_tara_std': cal['ruido'],
            'mvc_raw_absoluto': None if cal['mvc'] is None else cal['mvc'] + cal['base'],
            'obj_pct': OBJ_PCT, 'tol_pct': TOL_PCT,
            'puerto': 'SIMULADO' if a.simular else a.puerto, 'baud': a.baud,
            'eeg_stream': a.eeg_stream,
            'block_start_lsl': t_block_lsl,
            'block_start_wallclock': datetime.now().isoformat(timespec='milliseconds'),
        }
        with open(os.path.join(a.salida, base + '_meta.json'), 'w', encoding='utf-8') as fj:
            json.dump(meta, fj, indent=2, ensure_ascii=False)

    # ── Ejecución ───────────────────────────────────────────────────────────
    try:
        f = open(os.path.join(a.salida, base + '_triallog.csv'), 'w', newline='',
                 encoding='utf-8')
        w = csv.writer(f)
        w.writerow(['trial', 'seed', 'fps',
                    'jitter_plan', 'fijacion_plan', 'tarea_plan', 'reposo_plan',
                    't_jitter_lsl', 't_fix_lsl', 't_cue_lsl',
                    't_reposo_lsl', 't_reposo3s_lsl',
                    'dur_jitter_real', 'dur_fijacion_real', 'dur_tarea_real',
                    'dur_reposo_3s_real', 'frames_perdidos',
                    'pct_tiempo_en_rango', 'fuerza_media_pct_mvc',
                    'mvc_sobre_tara', 'tara_raw'])

        fuente = (FuerzaSimulada(win, fuerza_outlet) if a.simular
                  else FuerzaSerial(a.puerto, a.baud, fuerza_outlet))

        if not eeg_ok:
            print("[!] EEG NO DETECTADO")
        if not a.sin_visor:
            visor = lanzar_visor_eeg(a.eeg_stream)
            time.sleep(3)
        win.winHandle.activate()

        print("[INFO] >>> Inicia la grabación en LabRecorder (EEG + Dinamometro + "
              "CMC_Markers) y en FreeEMG, y luego pulsa ESPACIO para calibrar.")
        calibrar()

        fb = Retroalimentacion(win, fuente, cal['base'], cal['mvc'])
        pantalla_inicio(TEXTO_INSTR)
        t_bs = fase(nf(1.0), nada, 'block_start')
        guardar_meta(t_bs)

        if not a.sin_sync:
            sync('pre')

        for k in range(1, a.n_trials + 1):
            jitter = rng.uniform(JITTER_MIN_S, JITTER_MAX_S)
            d0 = win.nDroppedFrames

            # Atención: cruz (jitter + fijación)
            t_jit = fase(nf(jitter), fix.draw, 'jitter_start')
            t_fix = fase(nf(FIJACION_S), fix.draw, 'fix_start')
            # Tarea: beep + gráfica de fuerza
            fb.reset()
            t_cue = fase(nf(TAREA_S), fb.draw, 'cue_start', con_beep=True)
            # Reposo: pantalla negra, fijo; marcador a los 3 s
            n_r1 = nf(REPOSO_MARCA_S)
            n_r2 = nf(REPOSO_S) - n_r1
            t_rep = fase(n_r1, nada, 'reposo_start')
            t_r3 = fase(n_r2, nada, 'reposo_3s')

            pct_en, media = fb.resumen()
            w.writerow([k, seed, f'{fps:.2f}',
                        f'{jitter:.3f}', FIJACION_S, TAREA_S, REPOSO_S,
                        f'{t_jit:.6f}', f'{t_fix:.6f}', f'{t_cue:.6f}',
                        f'{t_rep:.6f}', f'{t_r3:.6f}',
                        f'{t_fix - t_jit:.4f}', f'{t_cue - t_fix:.4f}',
                        f'{t_rep - t_cue:.4f}', f'{t_r3 - t_rep:.4f}',
                        win.nDroppedFrames - d0,
                        f'{pct_en:.1f}', f'{media:.2f}',
                        f"{cal['mvc']:.4f}", f"{cal['base']:.4f}"])
            f.flush()
            print(f"[INFO] trial {k}/{a.n_trials}: {pct_en:.0f} % en rango, "
                  f"media {media:.1f} % MVC")

        if not a.sin_sync:
            sync('post')
        fase(nf(0.5), nada, 'block_end')
        msg_final = "Fin del bloque\n\nEspera al evaluador"

    except Abortar:
        print("[INFO] Abortado por el usuario")
        enviar('block_abort')
        msg_final = "Bloque abortado\n\nEspera al evaluador"

    except Exception as e:                  # p. ej. el puerto serial no abre
        print(f"[ERROR] {type(e).__name__}: {e}")
        msg_final = f"Error: {e}\n\nAvisa al evaluador"

    finally:
        if f is not None:
            f.close()
        try:
            if msg_final:
                esperar_cierre(msg_final)
        finally:
            if fuente is not None:
                fuente.cerrar()
            if visor is not None and visor.poll() is None:
                visor.terminate()
            win.close()
            core.quit()


if __name__ == '__main__':
    main()