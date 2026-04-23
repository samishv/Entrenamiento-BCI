import os
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import iirnotch, butter, tf2sos, sosfiltfilt

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

if "RUTA_EEG" not in globals():
    RUTA_EEG = r"D:\luiso\Documentos\Luis\UPIITA\TT\CMC\S17\EEG\S17_260421_EEG_OV_2_ME.npy"

if "RUTA_EMG" not in globals():
    RUTA_EMG = r"D:\luiso\Documentos\Luis\UPIITA\TT\CMC\S17\EMG\S17_260421_EMG_OV_2_ME.npy"

if "RUTA_DINAM" not in globals():
    RUTA_DINAM = r"D:\luiso\Documentos\Luis\UPIITA\TT\CMC\S17\DINAM\S17_260421_DINAM_OV_2_ME.npy"

FS_EEG = 250
FS_EMG = 1000
ORDEN  = 4

FRECUENCIAS_NOTCH = [19.6, 60, 120, 180, 240, 300]
Q_NOTCH           = 200.0

CANALES_EEG = ['Fz', 'FC3', 'FCz', 'FC4', 'Cz', 'C3', 'C4']
MUSCULOS    = ["FCU", "ECRL", "ECU"]

COLORES_EEG = ['#ff4783', '#e96396', '#d27fab', '#bb9bc2', '#a4b7d5', '#8dd4ea', '#76f0ff']
COLORES_EMG = ["#a4f4d7", "#d2fab1", "#ffff89"]
COLOR_DINAM =  "#ffffff"

BG_FIG      = '#0a0a0a'
BG_AX       = '#0d0d0d'
COLOR_GRID  = '#404040'
COLOR_TICK  = '#a5a5a5'
COLOR_LABEL = '#d1d1d1'
COLOR_TITLE = '#ffffff'

# ─────────────────────────────────────────────────────────────────────────────
# PARSEO DE NOMBRE DE ARCHIVO
# ─────────────────────────────────────────────────────────────────────────────

CONDICION_MAP = {
    'OV': 'Ojos Abiertos - Movimiento Visual',
    'OK': 'Ojos Abiertos - Movimiento Cinestésico',
    'CK': 'Ojos Cerrados - Movimiento Cinestésico',
    'CV': 'Ojos Cerrados - Movimiento Visual',
}
ESTADO_MAP = {
    'ME': 'Movimiento Ejecutado',
    'MI': 'Movimiento Imaginado',
    'RE': 'Reposo',
}

def parsear_nombre_archivo(ruta_eeg):
    nombre = os.path.splitext(os.path.basename(ruta_eeg))[0]
    partes = nombre.split('_')
    sujeto, condicion, estado = partes[0], partes[3], partes[5]
    return (f"Sujeto {sujeto[1:]}  |  "
            f"{CONDICION_MAP.get(condicion, condicion)}  |  "
            f"{ESTADO_MAP.get(estado, estado)}")

# ─────────────────────────────────────────────────────────────────────────────
# CARGADERA DE COSAS
# ─────────────────────────────────────────────────────────────────────────────

def cargar_npy(ruta, nombre):
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró {nombre}: {ruta}")
    return np.load(ruta)

eeg   = cargar_npy(RUTA_EEG,   "EEG")
emg   = cargar_npy(RUTA_EMG,   "EMG")
dinam = cargar_npy(RUTA_DINAM, "DINAM")

t_eeg = np.arange(eeg.shape[1])  / FS_EEG
t_emg = np.arange(emg.shape[1])  / FS_EMG
t_din = np.linspace(0, emg.shape[1] / FS_EMG, dinam.shape[0])
tmax  = min(t_eeg[-1], t_emg[-1], t_din[-1])

# ─────────────────────────────────────────────────────────────────────────────
# PROCESAMIENTO EEG
#   Etapa 1 — Reducción de artefactos de parpadeo con DWT (db4, nivel 5)
#   Etapa 2 — Filtro pasa banda Butterworth 4° orden  8–45 Hz
# ─────────────────────────────────────────────────────────────────────────────

def reducir_parpadeo_dwt(senal, wavelet='db4', nivel=5):
    """Elimina el detalle de nivel 5 (artefactos de parpadeo) y reconstruye."""
    coefs = pywt.wavedec(senal, wavelet, level=nivel)
    coefs[1] = np.zeros_like(coefs[1])          # detalle nivel 5 → cero
    return pywt.waverec(coefs, wavelet)[:len(senal)]

def procesar_eeg(eeg_raw, fs, orden=ORDEN):
    sos_bp = butter(orden, [8.0, 45.0], btype='bandpass', fs=fs, output='sos')
    out = np.zeros_like(eeg_raw, dtype=np.float64)
    for ch in range(eeg_raw.shape[0]):
        sin_parpadeo = reducir_parpadeo_dwt(eeg_raw[ch].astype(np.float64))
        out[ch]      = sosfiltfilt(sos_bp, sin_parpadeo)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# PROCESAMIENTO EMG
#   Etapa 1 — Filtros Notch en cascada
#   Etapa 2 — Filtro pasa banda Butterworth 4° orden  10–450 Hz
#   Etapa 3 — Rectificación (valor absoluto)
#   Etapa 4 — Envolvente: filtro pasa bajas Butterworth 2° orden  4 Hz
# ─────────────────────────────────────────────────────────────────────────────

def procesar_emg(emg_raw, fs, freqs_notch=FRECUENCIAS_NOTCH, q=Q_NOTCH, orden=ORDEN):
    sos_bp  = butter(orden, [20.0, 150.0], btype='bandpass', fs=fs, output='sos')
    sos_lp  = butter(orden-2, 4.0,          btype='low',      fs=fs, output='sos')
    out = np.zeros_like(emg_raw, dtype=np.float64)
    for ch in range(emg_raw.shape[0]):
        senal = emg_raw[ch].astype(np.float64)
        # Notch
        for f in freqs_notch:
            if 0 < f < fs / 2:
                b, a  = iirnotch(f, q, fs=fs)
                senal = sosfiltfilt(tf2sos(b, a), senal)
        # Pasa banda → rectificación → envolvente
        senal   = sosfiltfilt(sos_bp, senal)
        senal   = np.abs(senal)
        out[ch] = sosfiltfilt(sos_lp, senal)
    return out

print("\n  Procesando señales EEG...")
eeg_proc = procesar_eeg(eeg, FS_EEG)

print("  Procesando señales EMG...")
emg_proc = procesar_emg(emg, FS_EMG)

print("\n  Procesamiento completado.")

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────────────────────────────────────

def pantalla_completa(fig):
    try:
        manager = fig.canvas.manager
        for metodo in ['manager.window.state("zoomed")',
                        'manager.window.showMaximized()',
                        'manager.frame.Maximize(True)']:
            try:    eval(metodo); return
            except: pass
        fig.set_size_inches(19, 10.5)
    except:
        fig.set_size_inches(19, 10.5)

def estilo_ax(ax):
    ax.set_facecolor(BG_AX)
    ax.tick_params(colors=COLOR_TICK, labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_color(COLOR_GRID)
    ax.spines['bottom'].set_color(COLOR_GRID)
    ax.grid(True, color=COLOR_GRID, linewidth=0.5, linestyle='--')

# ─────────────────────────────────────────────────────────────────────────────
# PLOT MA-MA-MASIVO
# ─────────────────────────────────────────────────────────────────────────────

eeg_ymin = np.min(eeg_proc) * 1.1
eeg_ymax = np.max(eeg_proc) * 1.1
emg_ylim = np.max(np.abs(emg_proc)) * 1.1

fig, axes = plt.subplots(10, 1, sharex=True)
fig.patch.set_facecolor(BG_FIG)
fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.06, hspace=0.08)

fig.suptitle("EEG · EMG — Señales Procesadas",
             color=COLOR_TITLE, fontsize=11, fontweight='bold')
fig.text(0.5, 0.005, parsear_nombre_archivo(RUTA_EEG),
         ha='center', va='bottom',
         fontsize=8, color=COLOR_LABEL, style='italic', fontweight='bold')

# EEG (canales 0–6)
for i in range(7):
    ax = axes[i]
    estilo_ax(ax)
    ax.plot(t_eeg, eeg_proc[i], color=COLORES_EEG[i], linewidth=0.75)
    ax.set_ylabel(CANALES_EEG[i], color=COLOR_LABEL, fontsize=8)
    ax.set_xlim(0, tmax)
    ax.set_ylim(eeg_ymin, eeg_ymax)

# EMG (canales 0–2, filas 7–9)
for j in range(min(3, emg_proc.shape[0])):
    ax = axes[7 + j]
    estilo_ax(ax)
    ax.plot(t_emg, emg_proc[j], color=COLORES_EMG[j] if j < len(COLORES_EMG) else "#99b2ff", linewidth=1.5)
    ax.set_ylabel(MUSCULOS[j] if j < len(MUSCULOS) else f"EMG{j+1}", color=COLOR_LABEL, fontsize=8)
    ax.set_xlim(0, tmax)
    ax.set_ylim(0, emg_ylim)

# # Dinamómetro (fila 10, sin procesamiento)
# ax = axes[10]
# estilo_ax(ax)
# ax.plot(t_din, dinam, color=COLOR_DINAM, linewidth=1.5)
# ax.set_ylabel("DINAM", color=COLOR_LABEL, fontsize=8)

ax.set_xlabel("Tiempo [s]", color=COLOR_LABEL, fontsize=8)
ax.set_xlim(0, tmax)

pantalla_completa(fig)
plt.show()
