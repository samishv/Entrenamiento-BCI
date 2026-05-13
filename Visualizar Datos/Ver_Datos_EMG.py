import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, butter, tf2sos, sosfiltfilt

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

if "RUTA_EMG" not in globals():
    RUTA_EMG   = r"D:\luiso\Documentos\Luis\UPIITA\TT\CMC\S17\EMG\S17_260421_EMG_OV_2_ME.npy"

if "RUTA_DINAM" not in globals():
    RUTA_DINAM = r"D:\luiso\Documentos\Luis\UPIITA\TT\CMC\S17\DINAM\S17_260421_DINAM_OV_2_ME.npy"

FS    = 1000                                            # Frecuencia de muestreo (Hz)
ORDEN = 4                                               # Orden filtro Butterworth

FRECUENCIAS_NOTCH = [19.6, 60, 120, 180, 240, 300]      # Hz
Q_NOTCH           = 200.0

FIGURAS = [
    ('EMG Completo',      10.0, 450.0),
    # ('Energía Dominante', 20.0, 150.0),
]

MUSCULOS = ["FCU", "ECRL", "ECU"]

COLORES_EMG = ["#99b2ff", "#90c7db", "#b0dfa9"]
COLOR_DINAM = "#fefb6e"

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

def parsear_nombre_archivo(ruta):
    nombre = os.path.splitext(os.path.basename(ruta))[0]        # 'S17_260310_EEG_CK_3_RE'
    partes = nombre.split('_')                                  # ['S17','260310','EEG','CK','3','RE']

    sujeto    = partes[0]                                       # 'S17'
    condicion = partes[3]                                       # 'CK'
    estado    = partes[5]                                       # 'RE'

    num_sujeto   = sujeto[1:]                                   # '17'
    desc_cond    = CONDICION_MAP.get(condicion, condicion)
    desc_estado  = ESTADO_MAP.get(estado, estado)

    return f"Sujeto {num_sujeto}  |  {desc_cond}  |  {desc_estado}"

PIE_FIGURA = parsear_nombre_archivo(RUTA_DINAM)

# ─────────────────────────────────────────────────────────────────────────────
# CARGADERA DE COSAS
# ─────────────────────────────────────────────────────────────────────────────

def cargar_archivo(ruta, nombre):
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró: {ruta}")
    data = np.load(ruta)
    return data

emg_data   = cargar_archivo(RUTA_EMG,   "EMG")
dinam_data = cargar_archivo(RUTA_DINAM, "DINAMÓMETRO")

if emg_data.ndim == 1:
    emg_data = emg_data.reshape(1, -1)
elif emg_data.shape[0] != 4 and emg_data.shape[1] == 4:
    emg_data = emg_data.T

if dinam_data.ndim == 2 and dinam_data.shape[0] != 1:
    dinam_data = dinam_data[0]
elif dinam_data.ndim == 2:
    dinam_data = dinam_data.flatten()

n_canales  = emg_data.shape[0]
n_muestras = emg_data.shape[1]
t_emg      = np.arange(n_muestras)

# Eje de tiempo del dinamómetro en segundos
n_dinam  = len(dinam_data)
t_dinam = np.linspace(0, emg_data.shape[1]/FS, n_dinam)

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────────────────────────────────────

def pantalla_completa(fig):
    try:
        manager = fig.canvas.manager
        try:    manager.window.state('zoomed');   return
        except: pass
        try:    manager.window.showMaximized();   return
        except: pass
        try:    manager.frame.Maximize(True);     return
        except: pass
        fig.set_size_inches(19, 10.5)
    except:
        fig.set_size_inches(19, 10.5)
        
# def nombre_salida(ruta, sufijo):
#     CARPETA_IMGS = r"D:\luiso\Documentos\Luis\UPIITA\TT\CMC\IMGS"
#     os.makedirs(CARPETA_IMGS, exist_ok=True)          # la crea si no existe
#     nombre = os.path.splitext(os.path.basename(ruta))[0]
#     return os.path.join(CARPETA_IMGS, f"{nombre}_{sufijo}.svg")

def aplicar_pasabanda(emg, f_low, f_high, fs, orden):
    sos      = butter(orden, [f_low, f_high], btype='bandpass', fs=fs, output='sos')
    filtrada = np.zeros_like(emg)
    for ch in range(emg.shape[0]):
        filtrada[ch] = sosfiltfilt(sos, emg[ch])
    return filtrada

def calcular_fft(senal, fs):
    N        = len(senal)
    fft_vals = np.fft.rfft(senal)
    freqs    = np.fft.rfftfreq(N, d=1.0 / fs)
    magnitud = np.abs(fft_vals) / N
    magnitud[1:-1] *= 2
    return freqs, magnitud

def estilo_ax(ax):
    ax.set_facecolor(BG_AX)
    ax.tick_params(colors=COLOR_TICK, labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_color(COLOR_GRID)
    ax.spines['bottom'].set_color(COLOR_GRID)
    ax.grid(True, color=COLOR_GRID, linewidth=0.5, linestyle='--')

def plot_figura(nombre, emg_filt, dinam, t_e, t_d, f_low, f_high, fs, pie):

    fig, axes = plt.subplots(
        4, 2,
        gridspec_kw={'width_ratios': [1, 4], 'hspace': 0.15, 'wspace': 0.08}
    )
    fig.patch.set_facecolor(BG_FIG)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.07)

    es_ultimo_emg = n_canales - 1

    for i in range(n_canales):
        ax_fft = axes[i, 0]
        ax_sig = axes[i, 1]
        color  = COLORES_EMG[i]
        es_ultima_fila_emg = (i == es_ultimo_emg)

        # ── Label del músculo en la columna FFT ───────────────────────────
        ax_fft.set_ylabel(MUSCULOS[i], color=color, fontsize=9,
                          rotation=0, labelpad=32, va='center')

        # ── FFT (columna izquierda) ───────────────────────────────────────
        freqs, mag = calcular_fft(emg_filt[i], fs)
        ax_fft.plot(freqs, mag, color=color, linewidth=0.75, alpha=0.92)
        ax_fft.set_xlim(0, fs / 2)
        estilo_ax(ax_fft)
        if not es_ultima_fila_emg:
            ax_fft.tick_params(labelbottom=False)
        else:
            ax_fft.set_xlabel("Frecuencia [Hz]", color=COLOR_LABEL, fontsize=8)
        if i == 0:
            ax_fft.set_title("FFT  [|X(f)|]", color=COLOR_LABEL, fontsize=8, pad=3)

        # ── Señal procesada (columna derecha) ─────────────────────────────
        ax_sig.plot(t_e, emg_filt[i], color=color, linewidth=0.75, alpha=0.92)
        ax_sig.set_xlim(0, len(t_e) - 1)
        estilo_ax(ax_sig)
        if not es_ultima_fila_emg:
            ax_sig.tick_params(labelbottom=False)
        # else:
        #     # ax_sig.set_xlabel("Muestras", color=COLOR_LABEL, fontsize=8)
        if i == 0:
            ax_sig.set_title("Señal filtrada  [µV]", color=COLOR_LABEL, fontsize=8, pad=3)

    # ── Fila 4 col 0: vacío ───────────────────────────────────────────────
    axes[3, 0].set_visible(False)

    # ── Fila 4 col 1: dinamómetro ─────────────────────────────────────────
    ax_din = axes[3, 1]
    ax_din.plot(t_d, dinam, color=COLOR_DINAM, linewidth=1.5, alpha=0.95)
    ax_din.set_ylabel("DINAMÓMETRO\n[N]", color=COLOR_DINAM,
                      fontsize=8, rotation=0, labelpad=40, va='center')
    ax_din.tick_params(axis='y', colors=COLOR_DINAM)
    ax_din.set_xlabel("Tiempo [s]", color=COLOR_LABEL, fontsize=8)
    ax_din.set_xlim(t_d[0], t_d[-1])
    estilo_ax(ax_din)

    # Título y pie
    fig.suptitle(
        f"{nombre}  —  Butterworth ord. {ORDEN}  |  {f_low}–{f_high} Hz  |  Fs = {fs} Hz",
        color=COLOR_TITLE, fontsize=11, fontweight='bold'
    )
    fig.text(0.5, 0.005, pie, ha='center', va='bottom',
             fontsize=8, color=COLOR_LABEL, style='italic', fontweight='bold')

    fig.set_size_inches(19, 10.5)    
    
    # === DESCOMENTAR PARA GUARDAR IMG EN RUTA
    
    # ruta_svg = nombre_salida(RUTA_EMG, "EMG")
    # fig.savefig(ruta_svg, format='svg', bbox_inches='tight',
    #             facecolor=fig.get_facecolor())
    # print(f"\n  Figura guardada en: {ruta_svg}")
    
    pantalla_completa(fig)
    
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# ノッチ・レクイエム
# ─────────────────────────────────────────────────────────────────────────────

print("\n  Etapa 1 — Filtros Notch en cascada...\n")
emg_notch = np.zeros_like(emg_data, dtype=np.float64)
for ch in range(n_canales):
    resultado = emg_data[ch].astype(np.float64)
    for f in FRECUENCIAS_NOTCH:
        if 0 < f < FS / 2:
            b, a      = iirnotch(f, Q_NOTCH, fs=FS)
            sos       = tf2sos(b, a)
            resultado = sosfiltfilt(sos, resultado)
    emg_notch[ch] = resultado
    print(f"    Canal {ch+1}: OK  ({FRECUENCIAS_NOTCH} Hz eliminados)")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT MA-MA-MASIVO
# ─────────────────────────────────────────────────────────────────────────────

for idx, (nombre, f_low, f_high) in enumerate(FIGURAS, start=1):
    print(f"\n    Generando Figura {idx}: {nombre}  ({f_low}–{f_high} Hz)...")
    emg_filt = aplicar_pasabanda(emg_notch, f_low, f_high, FS, ORDEN)
    plot_figura(nombre, emg_filt, dinam_data, t_emg, t_dinam,
                f_low, f_high, FS, PIE_FIGURA)

print("\n  Todas las figuras EMG/DINAM generadas.\n\n")
plt.show()
