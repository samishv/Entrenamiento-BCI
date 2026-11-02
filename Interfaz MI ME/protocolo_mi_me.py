# -*- coding: utf-8 -*-
"""
Trial (14-15 s):
    ATENCIÓN (cruz '+'):  jitter 1-2 s  ->  fijación 3 s
    -> beep + cue (color/imagen) 5 s
    -> REPOSO 5 s (pantalla negra, fijo)
Bloque: 15 MI + 15 ME aleatorizados (máx. MAX_REPETIDAS iguales seguidas).

Marcadores (todos alineados al flip en que aparece el estímulo o arranca la fase):
    block_start | jitter_start | fix_start | cue_MI | cue_ME |
    reposo_start | reposo_3s | block_end | block_abort

    jitter_start : aparece la cruz (inicio del trial)
    fix_start    : inicio de la fijación de 3 s (la cruz ya estaba; sin cambio visual)
    cue_MI/ME    : cue + beep
    reposo_start : pantalla negra
    reposo_3s    : 3 s después de reposo_start (quedan 2 s de reposo)
Ventana de análisis prevista: fix_start -> reposo_3s (nominal 3 + 5 + 3 = 11 s).

"""
import argparse
import csv
import os
import random
import sys
import subprocess
from datetime import datetime
import time
import gc

from psychopy import prefs
#prefs.hardware['audioDevice'] = 'Altavoces (2- Realtek(R) Audio)'#COMPU LIPS
prefs.hardware['audioDevice'] = 'Altavoces (Realtek(R) Audio)'#COMPU LIPS
#prefs.hardware['audioDevice'] = 'Speaker / Headphone (Realtek(R) Audio)'#COMPU SAM 
#prefs.hardware['audioDevice'] = 'Altavoces (Realtek(R) Audio)' # COMPU LEO
prefs.hardware['audioLib'] = ['ptb', 'sounddevice']   # antes de importar sound
prefs.hardware['audioLatencyMode'] = 3
from psychopy import visual, core, event, sound, gui
 
from pylsl import StreamInfo, StreamOutlet, local_clock, resolve_byprop

# ─── Parámetros del paradigma ────────────────────────────────────────────────
JITTER_MIN_S   = 1.0        # atención: cruz con duración aleatoria...
JITTER_MAX_S   = 2.0
FIJACION_S     = 3.0        # ...y luego 3 s de fijación (cruz)
TAREA_S        = 5.0        # cue + beep
REPOSO_S       = 5.0        # pantalla negra, fijo
REPOSO_MARCA_S = 3.0        # marcador 'reposo_3s' a los 3 s de reposo_start
N_POR_CLASE    = 15
MAX_REPETIDAS  = 3          # nunca más de 3 trials seguidos de la misma clase
BEEP_HZ, BEEP_S = 700, 0.3

# Cue: color o imagen. Mapeo A/B para contrabalancear entre sujetos/sesiones.
# Colores con luminancia aproximada igualada (verifícala en tu monitor).
COLORES = {'A': {'MI': '#3A7BD5', 'ME': '#BE6E14'},
           'B': {'MI': '#BE6E14', 'ME': '#3A7BD5'}}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGENES = {
    'MI': os.path.join(BASE_DIR, 'cerebro5.png'),
    'ME': os.path.join(BASE_DIR, 'ME_L.png')
}   

TEXTO_INSTR = (
    "1) Mira la cruz.\n\n"
    "2) Al escuchar el beep, realiza la tarea hasta que desaparezca la imagen.\n\n"
    "3) Durante la pantalla negra, descansa.\n\n"
)

FUENTE = 'Minecraft'  # 'Arial', 'Helvetica', 'Verdana', 'Tahoma', 'Calibri  '



class Abortar(Exception):
    pass


def secuencia_bloque(n, rng):
    """n MI + n ME aleatorios sin rachas largas."""
    while True:
        s = ['MI'] * n + ['ME'] * n
        rng.shuffle(s)
        if not any(len(set(s[i:i + MAX_REPETIDAS + 1])) == 1
                   for i in range(len(s) - MAX_REPETIDAS)):
            return s

def lanzar_visor_eeg(stream_name):
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ventanaeeg.py')
    if not os.path.exists(ruta):
        print(f"[AVISO] No se encontró {ruta}; no se abre el visor EEG")
        return None
    try:
        flags = 0
        if sys.platform == 'win32':
            flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS   # que no compita con PsychoPy
        return subprocess.Popen([sys.executable, ruta, stream_name],
                                creationflags=flags)
    except Exception as e:
        print(f"[AVISO] No se pudo abrir el visor EEG: {e}")
        return None

def menu_inicio(a):
    """Diálogo de inicio: edita los argumentos de sesión en el namespace `a`.
    Los valores de la línea de comandos aparecen como predeterminados."""
    k_pid, k_ses, k_blo = 'Participante (pid)', 'Sesión', 'Bloque'
    k_map, k_pan = 'Mapeo de colores', 'Pantalla del sujeto (0, 1...)'
    k_n, k_eeg = 'Trials por clase', 'Nombre del stream EEG'
    k_out, k_seed = 'Carpeta de salida', 'Seed (vacío = aleatoria)'
    k_win = 'Modo ventana (pruebas)'

    d = {
        k_pid: a.pid,
        k_ses: a.sesion,
        k_blo: a.bloque,
        k_map: [a.mapeo] + [m for m in ('A', 'B') if m != a.mapeo],
        k_pan: a.pantalla,
        k_n: a.n_por_clase,
        k_eeg: a.eeg_stream,
        k_out: a.salida,
        k_seed: '' if a.seed is None else a.seed,
        k_win: a.ventana,
    }
    dlg = gui.DlgFromDict(d, title='Protocolo MI / ME', order=list(d.keys()))
    if not dlg.OK:
        print("[INFO] Cancelado desde el menú")
        core.quit()
        sys.exit()

    a.pid = str(d[k_pid]).strip() or 'X'
    a.sesion = str(d[k_ses]).strip()
    a.bloque = str(d[k_blo]).strip()
    a.mapeo = d[k_map]
    a.pantalla = int(d[k_pan])
    a.n_por_clase = int(d[k_n])
    a.eeg_stream = str(d[k_eeg]).strip()
    a.salida = str(d[k_out]).strip() or './Data'
    s = str(d[k_seed]).strip()
    a.seed = int(s) if s else None
    a.ventana = bool(d[k_win])
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', default='S')
    ap.add_argument('--sesion', default='D')
    ap.add_argument('--bloque', default='B')
    ap.add_argument('--pantalla', type=int, default=1, help='monitor del sujeto')
    ap.add_argument('--mapeo', choices=['A', 'B'], default='A')
    ap.add_argument('--n_por_clase', type=int, default=N_POR_CLASE)
    ap.add_argument('--eeg_stream', default='streamTEST')
    ap.add_argument('--salida', default='Protocolo ERD_ERS/Signals/LOGS')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--ventana', action='store_true', help='no pantalla completa (pruebas)')
    ap.add_argument('--sin_menu', action='store_true', help='omite el menú de inicio')
    ap.add_argument('--sin_visor', action='store_true', help='no abrir la ventana de EEG en tiempo real')
    ap.add_argument('--pantalla_eeg', type=int, default=None, help='(informativo) el visor ya elige monitor por sí mismo')
    a = ap.parse_args()
    if not a.sin_menu:
        a = menu_inicio(a)

    seed = a.seed if a.seed is not None else int(datetime.now().timestamp())
    rng = random.Random(seed)
    os.makedirs(a.salida, exist_ok=True)
    base = f"{a.pid}_{datetime.now():%y%m%d}_MIME_{a.sesion}_{a.bloque}"

    # ── LSL: outlet de marcadores y comprobación del EEG ────────────────────
    outlet = StreamOutlet(StreamInfo('MIME_Markers', 'Markers', 1, 0, 'string',
                                     f'mime_{a.pid}_{a.sesion}_{a.bloque}'))
    eeg_ok = bool(resolve_byprop('name', a.eeg_stream, timeout=3))
    if not eeg_ok:
        print(f"[AVISO] No se encontró el stream EEG '{a.eeg_stream}'")

    ultimo = {'t': None}

    def enviar(m):                       # se ejecuta justo tras el flip
        t = local_clock()
        outlet.push_sample([m], t)
        ultimo['t'] = t

    visor = None
    if not a.sin_visor:
        visor = lanzar_visor_eeg(a.eeg_stream)
        time.sleep(5)      # deja que el visor termine de cargar (modelo, etc.)

    # ── Ventana, estímulos, sonido ──────────────────────────────────────────
        win = visual.Window(fullscr=not a.ventana, screen=a.pantalla, color='black',
                        units='height', allowGUI=False)
    # Calentamiento: estabiliza el vsync antes de medir
    for _ in range(120):
        win.flip()
    fps = win.getActualFrameRate(nIdentical=20, nMaxFrames=240, nWarmUpFrames=20,
                                 threshold=1) or 60.0
    frame_dur = 1.0 / fps          # solo se usa como tolerancia, NO para las duraciones
    win.recordFrameIntervals = True
    win.refreshThreshold = frame_dur * 1.2
    print(f"[INFO] fps medidos: {fps:.2f} | seed: {seed}")
    fix = visual.TextStim(win, text='+', height=0.15, color='white')
    cues = {}
    for c in ('MI', 'ME'):
        if IMAGENES[c]:
            cues[c] = visual.ImageStim(win, image=IMAGENES[c], size=(0.7, 0.7))
        else:
            cues[c] = visual.Rect(win, width=0.7, height=0.7,
                                  fillColor=COLORES[a.mapeo][c], lineColor=None)
    beep = sound.Sound(BEEP_HZ, secs=BEEP_S, stereo=True, hamming=True)

    prox = {'inicio': None}    # inicio nominal de la siguiente fase (reloj LSL)

    def fase(dur_s, dibujar, marcador, con_beep=False):
        """Fase de dur_s segundos. El marcador (y beep) salen en el primer flip.
        El fin se calcula sobre un calendario absoluto: no se acumula error."""
        primero = True
        while True:
            dibujar()
            if primero:
                win.callOnFlip(enviar, marcador)
                if con_beep:
                    beep.play(when=win.getFutureFlipTime(clock='ptb'))
            win.flip()
            if event.getKeys(['escape']):
                raise Abortar()
            if primero:
                t0 = ultimo['t']                     # tiempo real del flip del marcador
                inicio_nom = prox['inicio'] if prox['inicio'] is not None else t0
                fin_nom = inicio_nom + dur_s
                prox['inicio'] = fin_nom
                primero = False
            # ¿el siguiente flip caería antes del fin nominal (con media trama de tolerancia)?
            if local_clock() + 0.5 * frame_dur >= fin_nom:
                break
        return t0

    def texto(msg, segundos=None):
        t = visual.TextStim(win, text=msg, height=0.04, color='white', wrapWidth=1.5)
        if segundos is None:
            t.draw(); win.flip()
            if 'escape' in event.waitKeys(keyList=['space', 'escape']):
                raise Abortar()
        else:
            fin = local_clock() + segundos
            while local_clock() < fin:
                t.draw(); win.flip()
                
    def pantalla_inicio(msg):
        """Muestra MI y ME junto con las instrucciones iniciales."""

        titulo = visual.TextStim(
            win,
            text=f'BLOQUE {a.bloque}',
            pos=(0, 0.42),
            height=0.065,
            color='#4FC3F7',
            font=FUENTE,
            bold=True
        )

        panel = visual.Rect(
            win,
            pos=(0, -0.02),
            width=2.0,
            height=0.7,
            fillColor='#181818',
            lineColor='#4FC3F7',
            lineWidth=2
        )

        # Imagen de imaginación motora
        imagen_mi = visual.ImageStim(
            win,
            image=IMAGENES['MI'],
            pos=(-0.30, 0.13),
            size=(0.27, 0.27)
        )

        # Imagen de movimiento ejecutado
        imagen_me = visual.ImageStim(
            win,
            image=IMAGENES['ME'],
            pos=(0.30, 0.13),
            size=(0.27, 0.27)
        )

        etiqueta_mi = visual.TextStim(
            win,
            text='IMAGINA',
            pos=(-0.30, -0.055),
            height=0.032,
            color="#EDF2F6",
            font=FUENTE,
            bold=True
        )

        etiqueta_me = visual.TextStim(
            win,
            text='EJECUTA',
            pos=(0.30, -0.055),
            height=0.032,
            color="#EDF2F6",
            font=FUENTE,
            bold=True
        )

        instrucciones = visual.TextStim(
            win,
            text=msg,
            pos=(0, -0.24),
            height=0.029,
            color='white',
            font=FUENTE,
            wrapWidth=1.25,
            alignText='center'
        )

        indicacion = visual.TextStim(
            win,
            text='Presiona ESPACIO para comenzar',
            pos=(0, -0.43),
            height=0.030,
            color='#BDBDBD',
            font=FUENTE
        )

        event.clearEvents()
        while True:
            panel.draw()
            titulo.draw()
            imagen_mi.draw()
            imagen_me.draw()
            etiqueta_mi.draw()
            etiqueta_me.draw()
            instrucciones.draw()
            indicacion.draw()
            win.flip()

            teclas = event.getKeys(['space', 'escape'])
            if 'escape' in teclas:
                raise Abortar()
            if 'space' in teclas:
                break
    
    def esperar_cierre(msg):
        """Pantalla final: espera ESPACIO (o ESC) sin lanzar Abortar.
        Sirve para pulsar Stop en LabRecorder antes de cerrar el outlet."""
        t = visual.TextStim(win, text=msg, height=0.04, color='white', wrapWidth=1.5)
        t.draw(); win.flip()
        print("[INFO] >>> Pulsa STOP en LabRecorder y luego ESPACIO en la "
              "ventana de PsychoPy para cerrar.")
        event.clearEvents()              # descarta teclas pendientes
        event.waitKeys(keyList=['space', 'escape'])

    nada = lambda: None
    seq = secuencia_bloque(a.n_por_clase, rng)

    f = open(os.path.join(a.salida, base + '_triallog.csv'), 'w', newline='', encoding='utf-8')
    w = csv.writer(f)
    w.writerow(['trial', 'clase', 'mapeo', 'seed', 'fps',
                'jitter_plan', 'fijacion_plan', 'tarea_plan', 'reposo_plan',
                't_jitter_lsl', 't_fix_lsl', 't_cue_lsl',
                't_reposo_lsl', 't_reposo3s_lsl',
                'dur_jitter_real', 'dur_fijacion_real', 'dur_tarea_real',
                'dur_reposo_3s_real', 'frames_perdidos'])

    msg_final = None
    try:
        nombre = {'#3A7BD5': 'AZUL', '#BE6E14': 'NARANJA'}
        mi = 'imagen MI' if IMAGENES['MI'] else nombre[COLORES[a.mapeo]['MI']]
        me = 'imagen ME' if IMAGENES['ME'] else nombre[COLORES[a.mapeo]['ME']]
        msg = TEXTO_INSTR.format(b=a.bloque, mi=mi, me=me)
        if not eeg_ok:
            print("[!] EEG NO DETECTADO")

        pantalla_inicio(msg)
        win.winHandle.activate()

        gc.collect(); gc.disable()          # sin pausas del recolector durante el bloque
        core.rush(True)                     # prioridad alta para PsychoPy
        prox['inicio'] = None               # el calendario arranca en block_start
        fase(1.0, nada, 'block_start')

        for k, clase in enumerate(seq, 1):
            jitter = rng.uniform(JITTER_MIN_S, JITTER_MAX_S)
            d0 = win.nDroppedFrames

            t_jit = fase(jitter,        fix.draw,          'jitter_start')
            t_fix = fase(FIJACION_S,    fix.draw,          'fix_start')
            t_cue = fase(TAREA_S,       cues[clase].draw,  f'cue_{clase}', con_beep=True)
            t_rep = fase(REPOSO_MARCA_S,           nada,   'reposo_start')
            t_r3  = fase(REPOSO_S - REPOSO_MARCA_S, nada,  'reposo_3s')

            w.writerow([k, clase, a.mapeo, seed, f'{fps:.2f}',
                        f'{jitter:.3f}', FIJACION_S, TAREA_S, REPOSO_S,
                        f'{t_jit:.6f}', f'{t_fix:.6f}', f'{t_cue:.6f}',
                        f'{t_rep:.6f}', f'{t_r3:.6f}',
                        f'{t_fix - t_jit:.4f}', f'{t_cue - t_fix:.4f}',
                        f'{t_rep - t_cue:.4f}', f'{t_r3 - t_rep:.4f}',
                        win.nDroppedFrames - d0])
            f.flush()

        fase(0.5, nada, 'block_end')
        msg_final = "Fin del bloque\n\nEspera al evaluador"

    except Abortar:
        print("[INFO] Abortado por el usuario")
        enviar('block_abort')
        msg_final = "Bloque abortado\n\nEspera al evaluador"

    finally:
        f.close()
        try:
            if msg_final:
                esperar_cierre(msg_final)
        finally:
            if visor is not None and visor.poll() is None:
                visor.terminate()
            core.rush(False)
            gc.enable()
            win.close()
            core.quit()


if __name__ == '__main__':
    main()