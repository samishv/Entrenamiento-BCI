import os
import numpy as np
import matplotlib.pyplot as plt

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

CANALES_EEG = ['Fz', 'FC3', 'FCz', 'FC4', 'Cz', 'C3', 'C4']
MUSCULOS = ["FCU", "ECRL", "ECU"]

COLORES_EEG = ['#6FC7B7', '#A1DAD1', '#D0EDE8',
               '#FFFFFF', '#BFEEF6', '#81DCEE', '#41CCE3'] 
COLORES_EMG = ["#99b2ff", "#90c7db", "#b0dfa9"]              
COLOR_DINAM = "#fefb6e"

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

def parsear_nombre_archivo(ruta_eeg):
    nombre = os.path.splitext(os.path.basename(ruta_eeg))[0]    # 'S17_260310_EEG_CK_3_RE'
    partes = nombre.split('_')                                  # ['S17','260310','EEG','CK','3','RE']

    sujeto    = partes[0]                                       # 'S17'
    condicion = partes[3]                                       # 'CK'
    estado    = partes[5]                                       # 'RE'

    num_sujeto   = sujeto[1:]                                   # '17'
    desc_cond    = CONDICION_MAP.get(condicion, condicion)
    desc_estado  = ESTADO_MAP.get(estado, estado)

    return f"Sujeto {num_sujeto}  |  {desc_cond}  |  {desc_estado}"

# ─────────────────────────────────────────────────────────────────────────────
# CARGADERA DE COSAS
# ─────────────────────────────────────────────────────────────────────────────

def cargar_npy(ruta, nombre):
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró {nombre}: {ruta}")
    return np.load(ruta)

eeg = cargar_npy(RUTA_EEG, "EEG")
emg = cargar_npy(RUTA_EMG, "EMG")
dinam = cargar_npy(RUTA_DINAM, "DINAM")

t_eeg = np.arange(eeg.shape[1]) / FS_EEG
t_emg = np.arange(emg.shape[1]) / FS_EMG

dur_emg = emg.shape[1] / FS_EMG
t_din = np.linspace(0, dur_emg, dinam.shape[0])

tmax = min(t_eeg[-1], t_emg[-1], t_din[-1])

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────────────────────────────────────

def pantalla_completa(fig):
    try:
        manager = fig.canvas.manager
        try:
            manager.window.state('zoomed')
            return
        except Exception:
            pass
        try:
            manager.window.showMaximized()
            return
        except Exception:
            pass
        try:
            manager.frame.Maximize(True)
            return
        except Exception:
            pass
        fig.set_size_inches(19, 10.5)
    except Exception:
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

fig, axes = plt.subplots(11, 1, sharex=True)
fig.patch.set_facecolor(BG_FIG)
fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.06, hspace=0.08)

# ── Pie de figura ──────────────────────────────────────────────────────
fig.text(0.5, 0.005, parsear_nombre_archivo(RUTA_EEG),
         ha='center', va='bottom',
         fontsize=8, color= COLOR_LABEL,
         style='italic', fontweight='bold')

fig.suptitle("VEMOS QUE TITULO LE PONGO", color=COLOR_TITLE, fontsize=11, fontweight='bold')

# EEG (7)
for i in range(7):
    ax = axes[i]
    estilo_ax(ax)
    ax.plot(t_eeg, eeg[i], color=COLORES_EEG[i], linewidth=0.75)
    etiqueta = CANALES_EEG[i] if i < len(CANALES_EEG) else f"EEG{i+1}"
    ax.set_ylabel(etiqueta, color=COLOR_LABEL, fontsize=8)
    ax.set_xlim(0, tmax)

# EMG (3) 
for j in range(min(3, emg.shape[0])):
    ax = axes[7 + j]
    estilo_ax(ax)
    color = COLORES_EMG[j] if j < len(COLORES_EMG) else "#99b2ff"
    ax.plot(t_emg, emg[j], color=color, linewidth=0.75)
    etiqueta = MUSCULOS[j] if j < len(MUSCULOS) else f"EMG{j+1}"
    ax.set_ylabel(etiqueta, color=COLOR_LABEL, fontsize=8)
    ax.set_xlim(0, tmax)

# DINAM (1)
ax = axes[10]
estilo_ax(ax)
ax.plot(t_din, dinam, color=COLOR_DINAM, linewidth=1.5)
ax.set_ylabel("DINAM", color=COLOR_LABEL, fontsize=8)
ax.set_xlabel("Tiempo (s)", color=COLOR_LABEL, fontsize=9)
ax.set_xlim(0, tmax)

pantalla_completa(fig)
plt.show()