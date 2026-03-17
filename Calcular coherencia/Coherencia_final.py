import os
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

plt.close('all')
# ───────────────────
# PARÁMETROS GLOBALES
# ───────────────────
FS_EMG   = 1000          
FS_EEG   = 250           
FS_FINAL = FS_EEG        

# Filtros
EMG_BAND  = (20.0, 450.0)   
EEG_BAND  = (0.1,  45.0)   
NOTCH_F   = 60.0            

# Segmentación (Welch)
SEG_DUR   = 1.0     
OVERLAP   = 0.5     

# Bandas de frecuencia 
BANDS = {
    "delta":  (1,   4),
    "theta":  (4,   8),
    "mu":     (8,  13),
    "beta":   (13, 30),
    "gamma":  (30, 80),
}
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
COLORES = [   '#6FC7B7', '#A1DAD1', '#D0EDE8',
              '#FFFFFF',  '#BFEEF6', '#81DCEE', '#41CCE3']

BG_FIG      = '#0a0a0a'
BG_AX       = '#0d0d0d'
COLOR_GRID  = '#404040'
COLOR_TICK  = '#a5a5a5'
COLOR_LABEL = '#d1d1d1'
COLOR_TITLE = '#ffffff'
PIE_FIGURA = " "
# ══════════════════════════════════════════════
# 1. CARGA DE SEÑALES
# ══════════════════════════════════════════════

def load_emg(RUTA_EMG):
    datos_emg = np.load(RUTA_EMG)
    if not (datos_emg.ndim == 2 and datos_emg.shape[0] == 4):
        raise ValueError(f"Shape inesperado: {datos_emg.shape}. Se esperaba (4, n_muestras).")

    n_canales, n_muestras = datos_emg.shape
    tiempo = np.arange(n_muestras) / FS_EMG
    print(f"\nDatos cargados → {datos_emg.shape}  |  Duración: {n_muestras/FS_EMG:.2f} s")
    return datos_emg

def load_eeg(RUTA_EEG):
    datos_eeg = np.load(RUTA_EEG)
    if not (datos_eeg.ndim == 2 and datos_eeg.shape[0] == 7):
        raise ValueError(f"Shape inesperado: {datos_eeg.shape}. Se esperaba (7, n_muestras).")

    n_canales, n_muestras = datos_eeg.shape
    tiempo = np.arange(n_muestras) / FS_EEG
    print(f"\nDatos cargados → {datos_eeg.shape}  |  Duración: {n_muestras/FS_EEG:.2f} s")

    return datos_eeg

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


# ══════════════════════════════════════════════
# 3. FILTRADO
# ══════════════════════════════════════════════
# def fir_pasabanda(lowcut, highcut, fs, numtaps=None):
#     """
#     Filtro FIR pasa-banda con ventana de Hamming.
#     numtaps: número de coeficientes (largo del filtro).
#              A más taps, más abrupto el corte pero más costo computacional.
#              Si es None, se calcula automáticamente.
#     """
#     nyq = fs / 2.0
    
#     if numtaps is None:
#         # Regla práctica: ~10 ciclos de la frecuencia de corte más baja
#         numtaps = int(10 * fs / lowcut)
#         # Debe ser impar para FIR tipo I
#         if numtaps % 2 == 0:
#             numtaps += 1
    
#     b = signal.firwin(numtaps, [lowcut / nyq, highcut / nyq],
#                       pass_zero=False, window='hamming')
#     return b
# def notch_fir(f0, fs, ancho_banda=2.0, numtaps=None):
#     """
#     Filtro notch FIR usando firwin con dos pasa-bandas que evitan f0.
#     f0         : frecuencia a eliminar (Hz)
#     ancho_banda: qué tan ancho es el notch en Hz (default ±1 Hz alrededor de f0)
#     """
#     nyq = fs / 2.0
    
#     if numtaps is None:
#         numtaps = int(10 * fs / ancho_banda)
#         if numtaps % 2 == 0:
#             numtaps += 1
    
#     # Definir las dos bandas que SÍ pasan (todo excepto f0 ± ancho_banda/2)
#     low  = (f0 - ancho_banda / 2) / nyq
#     high = (f0 + ancho_banda / 2) / nyq
    
#     b = signal.firwin(numtaps, [low, high],
#                       pass_zero=True, window='hamming')
#     return b
# def filtrar_eeg(eeg, fs=FS_EEG):
#     b = fir_pasabanda(*EEG_BAND, fs)
#     # FIR solo necesita b, no a → a = [1.0]
#     out = signal.filtfilt(b, [1.0], eeg, axis=0)
#     if NOTCH_F:
#         bn = notch_fir(NOTCH_F, fs)
#         out = signal.filtfilt(bn, [1.0], out, axis=0)
#     return out

# def filtrar_emg(emg, fs=FS_EMG):
#     b = fir_pasabanda(*EMG_BAND, fs)
#     out = signal.filtfilt(b, [1.0], emg, axis=0)
#     if NOTCH_F:
#         bn = notch_fir(NOTCH_F, fs)
#         out = signal.filtfilt(bn, [1.0], out, axis=0)
#     return out



def butter_pasabanda(lowcut: float, highcut: float, fs: float,
                     order: int = 4) -> tuple:
    nyq = fs / 2.0
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return b, a


def _notch(f0: float, fs: float, Q: float = 35) -> tuple:
    b, a = signal.iirnotch(f0, Q, fs)
    return b, a


def filtrar_emg(emg: np.ndarray, fs: float = FS_EMG) -> np.ndarray:
    b, a = butter_pasabanda(*EMG_BAND, fs)
    out  = signal.filtfilt(b, a, emg, axis=0)
    if NOTCH_F:
        bn, an = _notch(NOTCH_F, fs)
        out    = signal.filtfilt(bn, an, out, axis=0)
    return out


def filtrar_eeg(eeg: np.ndarray, fs: float = FS_EEG) -> np.ndarray:
    b, a = butter_pasabanda(*EEG_BAND, fs)
    out  = signal.filtfilt(b, a, eeg, axis=0)
    if NOTCH_F:
        bn, an = _notch(NOTCH_F, fs)
        out    = signal.filtfilt(bn, an, out, axis=0)
    return out


# ══════════════════════════════════════════════
# 4. RECTIFICACIÓN EMG
# ══════════════════════════════════════════════

def rectificar_emg(emg: np.ndarray) -> np.ndarray:
    """Rectificación de onda completa."""
    return np.abs(emg)
# ══════════════════════════════════════════════
# 5. RESAMPLE EMG → FS_FINAL
# ══════════════════════════════════════════════

def resample_signal(sig: np.ndarray, fs_in=FS_EMG, fs_out=FS_FINAL):
    from math import gcd
    g  = gcd(int(fs_out), int(fs_in))
    up = int(fs_out) // g
    dn = int(fs_in)  // g
    rs_signal = signal.resample_poly(sig, up, dn, axis=0)
    return rs_signal
# ══════════════════════════════════════════════
# 8. COHERENCIA
# ══════════════════════════════════════════════
def calcula_coherencia(emg, eeg, fs = FS_FINAL):
    n_emg = emg.shape[1]
    n_eeg = eeg.shape[1]

    freqs, _ = signal.coherence(emg[:,0],eeg[:,0], fs = fs, nperseg = int(fs*SEG_DUR))
    coherencia = np.zeros((len(freqs), n_emg,n_eeg))

    for ei in range(n_emg):
        for yi in range(n_eeg):
            _ ,coherencia[:,ei,yi] = signal.coherence(emg[:,ei],eeg[:,yi],
                                                      fs=fs,nperseg=int(fs*SEG_DUR),
                                                      noverlap=int(fs*SEG_DUR*OVERLAP))
    return freqs, coherencia
# ══════════════════════════════════════════════
# VISUALIZACIÓN
# ══════════════════════════════════════════════
def estilo_ax(ax):
    ax.set_facecolor(BG_AX)
    ax.tick_params(colors=COLOR_TICK, labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_color('#30363d')
    ax.spines['bottom'].set_color('#30363d')

def plot_coherencia(freqs, coherencia, emg_names, eeg_names, fmax=100, pie = ''):

    ylimit = coherencia.max()
    mask   = freqs <= fmax
    n_eeg  = len(eeg_names)

    colores_bandas = ['#9B3461', '#C15579', '#E77692', '#f0B8AB', '#FAFAC4']

    for ei, ename in enumerate(emg_names):

        fig, axes = plt.subplots(
            n_eeg, 1,
            figsize=(14, 2.8 * n_eeg),
            sharex=True
        )
        fig.patch.set_facecolor(BG_FIG)
        fig.subplots_adjust(
            left=0.09, right=0.98,
            top=0.93,  bottom=0.07,
            hspace=0.12
        )

        for yi, (yname, ax) in enumerate(zip(eeg_names, axes)):
            color = COLORES[yi % len(COLORES)]

            estilo_ax(ax)
            ax.plot(freqs[mask], coherencia[mask, ei, yi],
                    color=color, lw=1.5, alpha=0.95)

            for (bname, (f0, f1)), col in zip(BANDS.items(), colores_bandas):
                ax.axvspan(f0, min(f1, fmax), alpha=0.15, color=col, zorder=0)
                if yi == 0:
                    ax.text((f0 + f1) / 2, ylimit * 1.02, bname,
                            color=col, fontsize=6.5, ha='center', va='bottom')

            ax.set_ylabel(yname, color=color, fontsize=9,
                          rotation=0, labelpad=36, va='center')
            ax.set_ylim(0, ylimit * 1.15)
            ax.set_xlim(0, fmax)
            ax.grid(True, color=COLOR_GRID,
                    linewidth=0.5, linestyle='--', axis='y')

            if yi == 0:
                ax.set_title('Coherencia (CMC)', color=COLOR_LABEL,
                             fontsize=7.5, pad=3)
            if yi < n_eeg - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel('Frecuencia (Hz)', color=COLOR_LABEL, fontsize=9)       

        fig.suptitle(
            f'CMC EEG  —  EMG: {ename}  |  fs = {FS_FINAL} Hz',
            color=COLOR_TITLE, fontsize=11, fontweight='bold'
        )
        fig.text(0.5, 0.005, pie,
                 ha='center', va='bottom', fontsize=8,
                 color=COLOR_LABEL, style='italic', fontweight='bold')
# def plot_coherencia(freqs, coherence, emg_names,
#                     eeg_names, fmax=100):
#     """Muestra la coherencia para cada par EMG-EEG hasta fmax Hz."""
#     ylimit = coherence[:,:,:].max()
#     mask    = freqs <= fmax
#     # Sombreado de bandas
#     colores_bandas = ["#FFF9C4", "#C8E6C9", "#BBDEFB",
#                    "#F8BBD0", "#E1BEE7"]
#     n_emg   = len(emg_names)
#     n_eeg   = len(eeg_names)    
    
#     for ei, ename in enumerate(emg_names):
#         fig, axes = plt.subplots(len(eeg_names), 1,
#                              figsize=(12, 3 * len(eeg_names)), sharex=True)
        
#         for yi, (yname,ax) in enumerate(zip(eeg_names,axes)):
#             ax.plot(freqs[mask],coherence[mask,ei,yi],
#                     color="royalblue",lw=1.2)
        
#             for (bname, (f0, f1)), col in zip(BANDS.items(), colores_bandas):
#                 ax.axvspan(f0, min(f1, fmax), alpha=0.25,
#                            color=col, label=bname)

#             ax.set_ylabel(yname,fontsize=9)
#            # ax.set_xlabel("Frecuencia (Hz)")
#             ax.set_ylim(0, ylimit)
#             ax.grid(True, alpha=0.3)
#             if yi == 0:
#                 ax.legend(loc="upper right", fontsize=7, ncol=len(BANDS))
    
#         axes[-1].set_xlabel("Frecuencia (Hz)")
#         fig.suptitle(f"CMC - EMG: {ename}", fontsize=10)
#         plt.tight_layout()
        
        
#         #plt.savefig("coherence.png", dpi=150)
#         #plt.show()

# ══════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════

def calcula_todo(RUTA_EMG, RUTA_EEG, 
                 emg_channels=None, eeg_channels=None,
                 t_start=None, t_end=None,
                 emg_names=None, eeg_names=None):
    #1. Carga señales 
    emg = load_emg(RUTA_EMG).T
    eeg = load_eeg(RUTA_EEG).T
   
    PIE_FIGURA = parsear_nombre_archivo(RUTA_EEG)
    print(f"\nMetadatos:\n  {PIE_FIGURA}") 
    # Filtrado
    emg_filt = filtrar_emg(emg,fs = FS_EMG)
    eeg_filt = filtrar_eeg(eeg,fs = FS_EEG)

    #rectificación
    emg_rect = rectificar_emg(emg_filt)
    
    #REMUESTREAR
    emg_rs = resample_signal(emg_rect,fs_in = FS_EMG, fs_out=FS_FINAL)

    #COHERENCIA 
    freqs,coherencia = calcula_coherencia(emg_rs,eeg_filt)


    plot_coherencia(freqs, coherencia, emg_names, eeg_names, pie = PIE_FIGURA)
    plt.show()

    return {
        "freqs":     freqs,
        "coherencia": coherencia,
    }

# ══════════════════════════════════════════════
# PUNTO DE ENTRADA
# ══════════════════════════════════════════════
if __name__ == "__main__":

    #EMG_FILE = r"C:\Users\lrl13\Downloads\RegistrosCMC_pruebaleo\RegistrosCMC\PLEO_3\emg_signals_1.emt"
    #EEG_FILE = r"C:\ruta\al\archivo\EEG_RAW_09_03_2026__13_32_53.csv"

    #EEG_FILE = r"C:\Users\lrl13\Downloads\S17 (1)\S17\EEG\S17_260310_EEG_OV_4_ME.npy"
    #EMG_FILE = r"C:\Users\lrl13\Downloads\S17 (1)\S17\EMG\S17_260310_EMG_OV_2_ME.npy"
    
    EMG_FILE = r"C:\Users\lrl13\Downloads\S18\S18\EMG\S18_260313_EMG_CK_1_ME.npy"
    EEG_FILE = r"C:\Users\lrl13\Downloads\S18\S18\EEG\S18_260313_EEG_CK_2_ME.npy"

    EMG_NAMES = ["Right Brachioradialis","Right Extensor carpi radialis longus", "Right palmaris longus","Right extensor carpi ulnaris"]
    EEG_NAMES = ["Fz","FC3","FCz","FC4","Cz","C3","C4"]
    results = calcula_todo(
        RUTA_EMG     = EMG_FILE,
        RUTA_EEG     = EEG_FILE,
        emg_names = EMG_NAMES,
        eeg_names = EEG_NAMES,

        t_start      = None,          # None = solapamiento maximo
        t_end        = None,
    )

    print("\nPipeline CMC completado.")
    print(f"  Frecuencias: {results['freqs'][0]:.2f} - {results['freqs'][-1]:.2f} Hz")
    print(f"  Coherencia maxima: {results['coherencia'].max():.4f}")