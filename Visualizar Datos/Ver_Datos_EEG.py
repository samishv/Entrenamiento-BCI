import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos, welch

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

if "RUTA_NPY" not in globals():
    RUTA_NPY  = r"D:\luiso\Documentos\Luis\UPIITA\TT\CMC\S17\EEG\S17_260421_EEG_OV_2_ME.npy"

FS           = 250                  # Frecuencia de muestreo (Hz)
VENTANA_SEG  = 1.0                  # Resolución Welch: FS/NPERSEG = Hz por barra (500-0.5Hz) (1000-0.25Hz) MAX 125Hz
NPERSEG      = int(VENTANA_SEG*FS)

PAD_MUESTRAS = 10 * FS              # Muestras para padding: PAD - 28 segs - PAD
ORDEN        = 4                    # Orden filtro Butterworth
F_NOTCH      = 60.0                 # Frecuencia del filtro notch (Hz)
Q_NOTCH      = 30                   # Factor de calidad del notch

CANALES = ['Fz', 'FC3', 'FCz', 'FC4', 'Cz', 'C3', 'C4']

FIGURAS = [
    # ('EEG Completo',    0.5,  45.0),
    # ('Delta δ',         0.5,   4.0),
    # ('Theta θ',         4.0,   8.0),
    # ('Mu μ',            8.0,  13.0),
    # ('Beta β',         13.0,  30.0),
    # ('Gamma γ',        30.0,  45.0),
    ('μ + β + γ',   8.0,  45.0),
]

BANDAS = {
    'δ' : (0.5,  4,  '#9B3461'),
    'θ' : (4,    8,  '#C15579'),
    'μ' : (8,   13,  '#E77692'),
    'β' : (13,  30,  '#f0B8AB'),
    'γ' : (30,  45,  '#FAFAC4'),
}

COLORES = ['#6FC7B7', '#A1DAD1', '#D0EDE8','#FFFFFF',  '#BFEEF6', '#81DCEE', '#41CCE3']

BG_FIG      = '#0a0a0a'
BG_AX       = '#0d0d0d'
COLOR_GRID  = '#404040'
COLOR_TICK  = '#a5a5a5'
COLOR_LABEL = '#d1d1d1'
COLOR_TITLE = '#ffffff'

# ─────────────────────────────────────────────────────────────────────────────
# PARASEO DE LAS FIGURAS
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

PIE_FIGURA = parsear_nombre_archivo(RUTA_NPY)

# ─────────────────────────────────────────────────────────────────────────────
# CARGADERA DE COSAS
# ─────────────────────────────────────────────────────────────────────────────

datos = np.load(RUTA_NPY)
if not (datos.ndim == 2 and datos.shape[0] == 7):
    raise ValueError(f"Shape inesperado: {datos.shape}. Se esperaba (7, n_muestras).")

n_canales, n_muestras = datos.shape
tiempo = np.arange(n_muestras) / FS

b_n, a_n   = iirnotch(F_NOTCH, Q_NOTCH, FS)
sos_notch  = tf2sos(b_n, a_n)
datos      = np.array([sosfiltfilt(sos_notch, canal) for canal in datos])
print(f"\n  Filtro Notch aplicado → {F_NOTCH} Hz \n")

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────────────────────────────────────

def pantalla_completa(fig):
    try:
        manager = fig.canvas.manager
        try:    manager.window.state('zoomed');  return
        except: pass
        try:    manager.window.showMaximized();  return
        except: pass
        try:    manager.frame.Maximize(True);    return
        except: pass
        fig.set_size_inches(19, 10.5)
    except:
        fig.set_size_inches(19, 10.5)

def butter_bandpass_sos(f_low, f_high, fs, order=ORDEN):
    nyq = fs / 2
    return butter(order, [f_low / nyq, f_high / nyq], btype='band', output='sos')


def aplicar_filtro_con_padding(datos, f_low, f_high, fs, order=ORDEN, pad=PAD_MUESTRAS):
    sos = butter_bandpass_sos(f_low, f_high, fs, order)
    resultado = np.zeros_like(datos)

    for i, canal in enumerate(datos):
        inicio_espejo = canal[:pad][::-1]
        fin_espejo    = canal[-pad:][::-1]
        señal_ext     = np.concatenate([inicio_espejo, canal, fin_espejo])
        señal_filt_ext = sosfiltfilt(sos, señal_ext)
        resultado[i] = señal_filt_ext[pad:pad + len(canal)]

    return resultado

def calcular_psd(señal, fs, nperseg):
    freqs, psd = welch(señal, fs=fs, nperseg=nperseg)
    return freqs, psd

def estilo_ax(ax):
    ax.set_facecolor(BG_AX)
    ax.tick_params(colors=COLOR_TICK, labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_color('#30363d')
    ax.spines['bottom'].set_color('#30363d')

def plot_bandas_psd(ax, psd_max, f_low, f_high, etiquetas=False):
    for nombre, (b_ini, b_fin, bc) in BANDAS.items():
        if b_fin <= f_low or b_ini >= f_high:
            continue
        ax.axvspan(max(b_ini, f_low), min(b_fin, f_high),
                   alpha=0.15, color=bc, zorder=0)
        if etiquetas:
            ax.text((b_ini + b_fin) / 2, psd_max * 1.02,
                    nombre, color=bc, fontsize=6.5, ha='center', va='bottom')

def plot_figura(nombre_fig, datos_filt, f_low, f_high,
                tiempo, fs, nperseg, colores, canales, pie):
    fig, axes = plt.subplots(
        7, 2,
        gridspec_kw={'width_ratios': [1, 2], 'hspace': 0.15, 'wspace': 0.08}
    )
    fig.patch.set_facecolor(BG_FIG)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.07)

    y_min_fig  = datos_filt.min()
    y_max_fig  = datos_filt.max()
    margen_fig = (y_max_fig - y_min_fig) * 0.08

    psd_max = 0
    psds = []
    for i in range(7):
        f, p = calcular_psd(datos_filt[i], fs, nperseg)
        psds.append((f, p))
        psd_max = max(psd_max, p.max())

    for i in range(7):
        ax_s, ax_p = axes[i, 0], axes[i, 1]
        color, canal = colores[i], canales[i]
        freqs, psd = psds[i]

        # ── Señal ──────────────────────────────────────────────────────────
        estilo_ax(ax_s)
        ax_s.plot(tiempo, datos_filt[i], color=color, linewidth=0.75, alpha=0.95)
        ax_s.set_xlim(tiempo[0], tiempo[-1])
        ax_s.set_ylim(y_min_fig - margen_fig, y_max_fig + margen_fig)
        ax_s.set_ylabel(canal, color=color, fontsize=9,
                        rotation=0, labelpad=32, va='center')
        ax_s.grid(True, color=COLOR_GRID, linewidth=0.5, linestyle='--')
        if i < 6:
            ax_s.tick_params(labelbottom=False)
        else:
            ax_s.set_xlabel('Tiempo (s)', color=COLOR_LABEL, fontsize=9)

        # ── PSD (BAR PARA BARRA Y PLOT PARA CLASICO DE TODA LA VIDA)────────
        estilo_ax(ax_p)
                
        ax_p.plot(freqs, psd, color=color, alpha=0.85, linewidth=1.5)
        
        plot_bandas_psd(ax_p, psd_max, f_low, f_high, etiquetas=(i == 0))
        ax_p.set_xlim(PSD_XLIM)
        ax_p.set_ylim(0, psd_max * 1.15)
        ax_p.grid(True, color=COLOR_GRID, linewidth=0.5, linestyle='--', axis='y')
        ax_p.yaxis.get_offset_text().set_color(COLOR_TICK)
        ax_p.yaxis.get_offset_text().set_fontsize(6)
        if i < 6:
            ax_p.tick_params(labelbottom=False)
        else:
            ax_p.set_xlabel('Frecuencia (Hz)', color=COLOR_LABEL, fontsize=9)
        if i == 0:
            ax_p.set_title('PSD  (µV²/Hz)', color=COLOR_LABEL, fontsize=7.5, pad=3)

    # ── Título superior ────────────────────────────────────────────────────
    fig.suptitle(
        f'{nombre_fig}  —  Butterworth ord. {ORDEN}  |  {f_low}–{f_high} Hz  |  fs = {fs} Hz',
        color=COLOR_TITLE, fontsize=11, fontweight='bold'
    )

    # ── Pie de figura ──────────────────────────────────────────────────────
    fig.text(0.5, 0.005, pie,
             ha='center', va='bottom',
             fontsize=8, color= COLOR_LABEL,
             style='italic', fontweight='bold')

    pantalla_completa(fig)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# PLOT MA-MA-MASIVO
# ─────────────────────────────────────────────────────────────────────────────

PSD_XLIM   = (0, 45+8)

for idx, (nombre, f_low, f_high) in enumerate(FIGURAS, start=1):
    print(f"    Generando Figura {idx}: {nombre}  ({f_low}–{f_high} Hz)...")
    datos_filt = aplicar_filtro_con_padding(datos, f_low, f_high, FS, ORDEN)
    plot_figura(nombre, datos_filt, f_low, f_high,
                tiempo, FS, NPERSEG, COLORES, CANALES, PIE_FIGURA)

print("\n  Todas las figuras EEG generadas.\n\n")
plt.show()
