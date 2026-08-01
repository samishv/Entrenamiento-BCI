import os
import numpy as np
#import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path
from Procesamiento_Funciones import (
    cargar_npy_eeg,
    aplicar_notch_multiple,
    aplicar_wavelet_dwt_multicanal,
    ica_eliminar_componente,
    aplicar_pasabanda_butter,
    rectificar_senal,
    resamplear_por_promedio,
    emg_envelope
)


plt.close('all')
# ───────────────────
# PARÁMETROS GLOBALES
# ───────────────────
FS_EMG   = 1000          
FS_EEG   = 250           
FS_FINAL = FS_EEG        

# Filtros
EMG_BAND  = (2.0,100.0)   
EEG_BAND  = (5,  45.0)   
NOTCH_F   = 60.0            

# Segmentación (Welch)
SEG_DUR   = 1.0     
OVERLAP   = 0.5     

# Bandas de frecuencia 
# BANDAS = {
#     "delta":  (0,   4),
#     "theta":  (4,   8),
#     "mu":     (8,  13),
#     "beta":   (13, 30),
#     "gamma":  (30, 45),
#     "Otros":  (45,1000)#límite superior de gamma es de 80
# }
BANDAS = {
    'δ' :       (0,  4),
    'θ' :       (4,  8),
    'μ' :       (8, 13),
    'β' :       (13,30),
    'γ' :       (30,45),
    "Otros":  (45,1000)
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
# COLORES = [   '#6FC7B7', '#A1DAD1', '#D0EDE8',
#               '#FFFFFF',  '#BFEEF6', '#81DCEE', '#41CCE3']

# BG_FIG      = '#0a0a0a'
# BG_AX       = '#0d0d0d'
# COLOR_GRID  = '#404040'
# COLOR_TICK  = '#a5a5a5'
# COLOR_LABEL = '#d1d1d1'
# COLOR_TITLE = '#ffffff'
# PIE_FIGURA = " "

# --- ESTOS COLORES SON PARA VERSION CLARA

COLORES = [    '#477e75', '#4b7168', '#4d615f',
                '#515051',  '#475e63', '#3d6e74', '#307c8a']

BG_FIG      = '#ffffff'
BG_AX       = '#ffffff'
COLOR_GRID  = '#bfbfbf'
COLOR_TICK  = '#5a5a5a'
COLOR_LABEL = '#2e2e2e'
COLOR_TITLE = '#1e1e1e'

# ══════════════════════════════════════════════
#  CARGA DE SEÑALES
# ══════════════════════════════════════════════

def load_emg(RUTA_EMG):
    datos_emg = np.load(RUTA_EMG)
    if not (datos_emg.ndim == 2 and datos_emg.shape[0] == 3):
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

    descripcion = f"Sujeto {num_sujeto}  |  {desc_cond}  |  {desc_estado}"
    tag_archivo = f"{sujeto}_{condicion}_{estado}"

    return descripcion, tag_archivo, [sujeto,condicion,estado]
def lee_carpeta(ruta_base,tipos, sujetos):
    rutas = {}
    for tipo in tipos:
        rutas[tipo] = {}
        for sujeto in sujetos :
            archivos_sujeto = []                        
            carpeta_objetivo = os.path.join(ruta_base, sujeto, tipo)        
            
            for carpeta_raiz,_, archivos in os.walk(carpeta_objetivo):
                for archivo in archivos:
                    ruta_completa = os.path.join(carpeta_raiz,archivo)
                    _,_,estado =  parsear_nombre_archivo(ruta_completa)
                    if ((os.path.splitext(os.path.basename(ruta_completa))[1] == '.npy') &
                        (estado[2] == 'ME')):
                    #archivos_sujeto.append((ruta_completa,archivo))
                        archivos_sujeto.append(ruta_completa)
            
            rutas[tipo][sujeto] = archivos_sujeto
    return rutas
def generar_pares(ruta_base,tipos,sujetos):
    rutas = lee_carpeta(ruta_base,tipos,sujetos)
    print(rutas)
    analisis = {}

    for sujeto in sujetos:
        analisis_sujeto = []
        muestras = len(rutas["EEG"][sujeto])
        if muestras == len(rutas["EMG"][sujeto]):
            for i in range(muestras):
                for j in range(muestras):
                    if parsear_nombre_archivo(rutas["EEG"][sujeto][i]) == parsear_nombre_archivo(rutas["EMG"][sujeto][j]):
                        analisis_sujeto.append((rutas["EEG"][sujeto][i],rutas["EMG"][sujeto][j]))
        analisis[sujeto] = analisis_sujeto
    return analisis

# ══════════════════════════════════════════════
#  FILTRADO
# ══════════════════════════════════════════════
def _notch(f0: float, fs: float, Q: float = 35) -> np.ndarray:
    b, a = signal.iirnotch(f0, Q, fs)
    sos  = signal.tf2sos(b, a)
    return sos

def butter_pasabanda(lowcut: float, highcut: float, fs: float,
                         orden: int = 6) -> np.ndarray:
    nyq = fs / 2.0
    sos = signal.butter(orden, [lowcut / nyq, highcut / nyq],
                        btype="band", output='sos')
    return sos

def filtrar_emg(emg: np.ndarray, fs: float = FS_EMG, bp=True, n=True) -> np.ndarray:
    if bp:
        sos = butter_pasabanda(*EMG_BAND, fs)
        out = signal.sosfiltfilt(sos, emg, axis=0)
    else:
        out = emg
    if n:
        if NOTCH_F:
            sos_n = _notch(NOTCH_F, fs)
            out   = signal.sosfiltfilt(sos_n, out, axis=0)
    return out

def filtrar_eeg(eeg: np.ndarray, fs: float = FS_EEG) -> np.ndarray:
    b, a = butter_pasabanda(*EEG_BAND, fs)
    out  = signal.filtfilt(b, a, eeg, axis=0)
    if NOTCH_F:
        bn, an = _notch(NOTCH_F, fs)
        out    = signal.filtfilt(bn, an, out, axis=0)
    return out

def rectificar_emg(emg: np.ndarray) -> np.ndarray:
    return np.abs(emg)

def procesar_emg(emg_data: np.ndarray, fs: float = FS_EMG):
    FS = 1000                 # Frecuencia de muestreo EMG (Hz)
    ORDEN_BP = 3              # Orden Butterworth
    PAD_MUESTRAS = 5 * FS     # Padding reflect (muestras). 3 s

    FREQS_NOTCH = [19.6, 60, 120, 180, 240, 300]  # Hz
    Q_NOTCH = 200

    FIGURAS = [
        # ("EMG Completo", 10, 450.0),
        # ("EMG Completo RESAMPLED", 2.0, 113.0),
        ("EMG Completo COHERENCIA", 5.0, 45.0),
    ]


    WAVELET =       'db4'
    LEVEL =         6
    NIVELES_CERO =  [1, 6]     # D6 y D1
    REMOVE_APPROX = True       # A6
    WAVE_MODE =     'reflect'

    n_canales  = emg_data.shape[0]
    n_muestras = emg_data.shape[1]
    t_emg      = np.linspace(0, emg_data.shape[1]/FS, n_muestras)

    # n_dinam  = len(dinam_data)
    # t_dinam = np.linspace(0, emg_data.shape[1]/FS, n_dinam)

    # 2) Notchs en cascada
    emg_proc = aplicar_notch_multiple(
        emg_data,
        fs=FS,
        freqs=tuple(FREQS_NOTCH),
        Q=Q_NOTCH,
        pad=PAD_MUESTRAS,
    )
    print(f"\n  [OK] Notch aplicado: {FREQS_NOTCH} Hz  |  Q={Q_NOTCH}  |  pad={PAD_MUESTRAS} muestras")

    # 3) Resampleo PROMEDIO
    emg_proc = resamplear_por_promedio(emg_proc,4)

    # 3) resampleo LIBRERIA
    # emg_proc = resamplear_profesional(emg_proc, 1000, 250)

    # 3) CONTINUACION
    FS = FS / 4  # Ahora FS pasa de 1000 a 250 Hz
    n_muestras = emg_proc.shape[1]
    t_emg = np.linspace(0, n_muestras / FS, n_muestras)

    print(f"\n  [OK] Resampleo aplicado. Nueva FS: {FS} Hz | Muestras: {n_muestras}")

    # 4) WAVELETS
    emg_proc = aplicar_wavelet_dwt_multicanal(
        emg_proc,
        wavelet=WAVELET,
        level=LEVEL,
        niveles_cero=NIVELES_CERO,
        remove_approx=REMOVE_APPROX,
        pad=PAD_MUESTRAS,
        mode=WAVE_MODE
    )
    print(f"  [OK] Wavelet aplicado: {WAVELET} | level={LEVEL} | cero={NIVELES_CERO} | remove A{LEVEL}={REMOVE_APPROX}")

    # 5) Pasa Banda y Plot
    for nombre, f_low, f_high in FIGURAS:
        print(f"\n → Generando: {nombre} ({f_low:.1f}–{f_high:.1f} Hz)")

        emg_final = aplicar_pasabanda_butter(
            emg_proc,
            fs=FS,
            f_low=f_low,
            f_high=f_high,
            orden=ORDEN_BP,
            pad=PAD_MUESTRAS,
        )
        
        # 6) Rectificar Señales
        # emg_final = rectificar_senal(emg_final, tipo='completa')
        
        # # 7) ENVOLVEMENTE SEÑAL
        for i in range(emg_final.shape[0]):
            emg_final[i, :] = emg_envelope(emg_final[i, :], fs=FS, tc_ms=20, rectify="full")
    return emg_final
        

# ══════════════════════════════════════════════
#  RESAMPLE EMG A FS_FINAL
# ══════════════════════════════════════════════

def resample_signal(sig: np.ndarray, fs_in=FS_EMG, fs_out=FS_FINAL):
    from math import gcd
    g  = gcd(int(fs_out), int(fs_in))
    up = int(fs_out) // g
    dn = int(fs_in)  // g
    rs_signal = signal.resample_poly(sig, up, dn, axis=0)
    return rs_signal
# ══════════════════════════════════════════════
# 8. COHERENCIA, LÍMITE DE CONFIANZA 
# ══════════════════════════════════════════════
def calcula_coherencia(emg, eeg, fs = FS_FINAL):
    n_emg = emg.shape[1]
    n_eeg = eeg.shape[1]

    freqs, _ = signal.coherence(emg[:,0],eeg[:,0], fs = fs, nperseg = int(fs*SEG_DUR))
    coherencia = np.zeros((len(freqs), n_emg,n_eeg))

    for ei in range(n_emg):
        for yi in range(n_eeg):
            _ ,coherencia[:,ei,yi] = signal.coherence(emg[:,ei],eeg[:,yi], fs=fs,nperseg=int(fs*SEG_DUR),
                                                      noverlap=int(fs*SEG_DUR*OVERLAP))

    return freqs, coherencia

def confidence_limit(n_segmentos, alpha=0.02):
    """
    CL = 1 - (1 - alpha)^(1 / (L - 1))
    n_segmentos: número de ventanas de Welch usadas
    alpha      : nivel de significancia (0.05 = 95%)
    """
    alpha = 0.05
    CL = 1 - alpha**(1/(n_segmentos - 1))
    return CL#1 - (1 - alpha) ** (1 / (n_segmentos - 1))

def n_segmentos(n_muestras, fs=FS_FINAL):
    seg_len = int(fs * SEG_DUR)
    step    = int(seg_len * (1 - OVERLAP))
    return len(np.arange(0, n_muestras - seg_len + 1, step))
# ══════════════════════════════════════════════
# VISUALIZACIÓN
# ══════════════════════════════════════════════
def pantalla_completa(fig):
    try:
        manager = fig.canvas.manager
        try:    manager.window.state('zoomed');  return
        except: pass
        try:    manager.window.showMaximized();  return
        except: pass
        try:    manager.frame.Maximize(False);    return
        except: pass
        fig.set_size_inches(9, 6)
    except:
        fig.set_size_inches(9, 6)
def estilo_ax(ax):
    ax.set_facecolor(BG_AX)
    ax.tick_params(colors=COLOR_TICK, labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_color('#30363d')
    ax.spines['bottom'].set_color('#30363d')

# def plot_coherencia(freqs, coherencia, emg_names, eeg_names, fmax=45, pie = '', confidence_limit =None):

#     ylimit = coherencia.max()
#     mask   = freqs <= fmax
#     n_eeg  = len(eeg_names)

#     colores_bandas = ['#9B3461', '#C15579', '#E77692', '#f0B8AB', '#FAFAC4']

#     for ei, ename in enumerate(emg_names):

#         fig, axes = plt.subplots(
#             n_eeg, 1,
#             figsize=(14, 2.8 * n_eeg),
#             sharex=True
#         )
#         fig.patch.set_facecolor(BG_FIG)
#         fig.subplots_adjust(
#             left=0.09, right=0.98,
#             top=0.93,  bottom=0.07,
#             hspace=0.12
#         )

#         for yi, (yname, ax) in enumerate(zip(eeg_names, axes)):
#             color = COLORES[yi % len(COLORES)]

#             estilo_ax(ax)
#             ax.plot(freqs[mask], coherencia[mask, ei, yi],
#                     color=color, lw=1.5, alpha=0.95)
            
#             if confidence_limit is not None:
#                 ax.axhline(confidence_limit, color='#FF6B6B', lw=1.0,
#                    linestyle='--', alpha=0.8, label=f'CL 95% = {confidence_limit:.3f}')
#                 if yi == 0:
#                     ax.legend(loc="upper right", fontsize=7)

#             for (bname, (f0, f1)), col in zip(BANDAS.items(), colores_bandas):
#                 ax.axvspan(f0, min(f1, fmax), alpha=0.15, color=col, zorder=0)
#                 if yi == 0:
#                     ax.text((f0 + f1) / 2, ylimit * 1.02, bname,
#                             color=col, fontsize=6.5, ha='center', va='bottom')

#             ax.set_ylabel(yname, color=color, fontsize=9,
#                           rotation=0, labelpad=36, va='center')
#             ax.set_ylim(0, ylimit * 1.15)
#             ax.set_xlim(0, fmax)
#             ax.grid(True, color=COLOR_GRID,
#                     linewidth=0.5, linestyle='--', axis='y')

#             if yi == 0:
#                 ax.set_title('Coherencia (CMC)', color=COLOR_LABEL,
#                              fontsize=7.5, pad=3)
#             if yi < n_eeg - 1:
#                 ax.tick_params(labelbottom=False)
#             else:
#                 ax.set_xlabel('Frecuencia (Hz)', color=COLOR_LABEL, fontsize=9)       

#         fig.suptitle(
#             f'CMC EEG  —  EMG: {ename}  |  fs = {FS_FINAL} Hz',
#             color=COLOR_TITLE, fontsize=11, fontweight='bold'
#         )
#         fig.text(0.5, 0.005, pie,
#                  ha='center', va='bottom', fontsize=8,
#                  color=COLOR_LABEL, style='italic', fontweight='bold')
#         pantalla_completa(fig)
def plot_coherencia(freqs, coherencia, emg_names, eeg_names,fmax=45,
                    pie='', confidence_limit=None, eeg_to_plot=None):
    """
    Grafica la coherencia EMG-EEG.

    Parámetros
    ----------
    freqs : array, shape (n_freqs,)
        Vector de frecuencias.
    coherencia : array, shape (n_freqs, n_emg, n_eeg)
        Matriz de coherencia.
    emg_names : list[str]
        Nombres de canales EMG.
    eeg_names : list[str]
        Nombres de canales EEG.
    fmax : float
        Frecuencia máxima a mostrar.
    pie : str
        Texto al pie de la figura.
    confidence_limit : float o None
        Límite de confianza a mostrar.
    eeg_to_plot : None, str, int, list[str], list[int]
        Canales EEG a graficar.
        - None: todos
        - "Fz": solo ese canal
        - ["Fz", "C3"]: esos canales
        - [0, 2]: por índices
    """
    mask = freqs >= 5.0
    mask = freqs <= fmax
    colores_bandas = ['#9B3461', '#C15579', '#E77692', '#f0B8AB', '#FAFAC4']

    # -----------------------------
    # Normalizar selección de canales EEG
    # -----------------------------
    if eeg_to_plot is None:
        eeg_idx = list(range(len(eeg_names)))
    else:
        # Si viene un solo elemento, convertirlo a lista
        if isinstance(eeg_to_plot, (str, int, np.integer)):
            eeg_to_plot = [eeg_to_plot]

        eeg_idx = []
        for ch in eeg_to_plot:
            if isinstance(ch, str):
                if ch not in eeg_names:
                    raise ValueError(
                        f'El canal EEG "{ch}" no existe. '
                        f'Canales disponibles: {eeg_names}'
                    )
                eeg_idx.append(eeg_names.index(ch))

            elif isinstance(ch, (int, np.integer)):
                if ch < 0 or ch >= len(eeg_names):
                    raise ValueError(
                        f'Índice EEG fuera de rango: {ch}. '
                        f'Debe estar entre 0 y {len(eeg_names)-1}'
                    )
                eeg_idx.append(int(ch))

            else:
                raise TypeError(
                    "eeg_to_plot debe ser None, str, int, list[str] o list[int]"
                )

        # Quitar duplicados conservando el orden
        eeg_idx = list(dict.fromkeys(eeg_idx))

    eeg_names_sel = [eeg_names[i] for i in eeg_idx]
    n_eeg = len(eeg_idx)

    if n_eeg == 0:
        raise ValueError("No se seleccionó ningún canal EEG para graficar.")

    # Calcular límite Y solo con los canales seleccionados
    coh_sel = coherencia[mask][:, :, eeg_idx]
    ylimit = np.nanmax(coh_sel)
    if not np.isfinite(ylimit) or ylimit <= 0:
        ylimit = 1.0

    # -----------------------------
    # Una figura por músculo EMG
    # -----------------------------
    for ei, ename in enumerate(emg_names):

        fig, axes = plt.subplots(
            n_eeg, 1,
            figsize=(14, 2.8 * n_eeg),
            sharex=True,
            squeeze=False
        )
        axes = axes.ravel()

        fig.patch.set_facecolor(BG_FIG)
        fig.subplots_adjust(
            left=0.09, right=0.98,
            top=0.93, bottom=0.07,
            hspace=0.12
        )

        for row, (yi, yname, ax) in enumerate(zip(eeg_idx, eeg_names_sel, axes)):
            color = COLORES[row % len(COLORES)]

            estilo_ax(ax)
            ax.plot(
                freqs[mask],
                coherencia[mask, ei, yi],
                color=color,
                lw=1.5,
                alpha=0.95
            )

            if confidence_limit is not None:
                ax.axhline(
                    confidence_limit,
                    color='#FF6B6B',
                    lw=1.0,
                    linestyle='--',
                    alpha=0.8,
                    label=f'CL 95% = {confidence_limit:.3f}'
                )
                if row == 0:
                    ax.legend(loc="upper right", fontsize=7)

            for (bname, (f0, f1)), col in zip(BANDAS.items(), colores_bandas):
                if f0 < fmax:
                    ax.axvspan(f0, min(f1, fmax), alpha=0.15, color=col, zorder=0)
                    if row == 0:
                        ax.text(
                            (f0 + min(f1, fmax)) / 2,
                            ylimit * 1.02,
                            bname,
                            color='k',
                            fontsize=10,
                            ha='center',
                            va='bottom'
                        )

            ax.set_ylabel(
                yname,
                color=color,
                fontsize=9,
                rotation=0,
                labelpad=36,
                va='center'
            )
            ax.set_ylim(0, ylimit * 1.15)
            ax.set_xlim(5.0, fmax)
            ax.grid(
                True,
                color=COLOR_GRID,
                linewidth=0.5,
                linestyle='--',
                axis='y'
            )

            if row == 0:
                ax.set_title(
                    'Coherencia (CMC)',
                    color=COLOR_LABEL,
                    fontsize=10,
                    pad=3
                )

            if row < n_eeg - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel('Frecuencia (Hz)', color=COLOR_LABEL, fontsize=10)

        fig.suptitle(
            f'CMC EEG  —  EMG: {ename}  |  fs = {FS_FINAL} Hz',
            color=COLOR_TITLE,
            fontsize=11,
            fontweight='bold'
        )

        fig.text(
            0.5, 0.005, pie,
            ha='center',
            va='bottom',
            fontsize=8,
            color=COLOR_LABEL,
            style='italic',
            fontweight='bold'
        )
        pantalla_completa(fig)
        
def plot_señales_psd(emg_rs, eeg_filt, emg_names, eeg_names,
                     canal_eeg=5, canal_emg=0, fmax=100):
    """
    Figura con 4 subplots:
      - Señal temporal EMG (un canal)
      - PSD del EMG
      - Señal temporal EEG (un canal)
      - PSD del EEG
    """
    # FS_EEGAUX = FS_EEG
    # FS_EEG = 250
    #FS_FINAL = 1000
    t_emg = np.arange(emg_rs.shape[0])  / FS_FINAL
    t_eeg = np.arange(eeg_filt.shape[0]) / FS_EEG

    f_emg, p_emg = signal.welch(emg_rs[:,  canal_emg], fs=FS_FINAL,
                                 nperseg=int(FS_FINAL * SEG_DUR))
    f_eeg, p_eeg = signal.welch(eeg_filt[:, canal_eeg], fs=FS_EEG,
                                 nperseg=int(FS_EEG  * SEG_DUR))

    mask_emg = f_emg <= fmax
    mask_eeg = f_eeg <= fmax

    nombre_emg = emg_names[canal_emg] if emg_names else f"EMG_{canal_emg}"
    nombre_eeg = eeg_names[canal_eeg] if eeg_names else f"EEG_{canal_eeg}"

    # FS_EEG = FS_EEGAUX


    colores_bandas = ['#9B3461', '#C15579', '#E77692', '#f0B8AB', '#FAFAC4']

    fig, axes = plt.subplots(2, 2, figsize=(16, 7))
    fig.patch.set_facecolor(BG_FIG)
    fig.subplots_adjust(left=0.07, right=0.97,
                        top=0.91, bottom=0.09,
                        hspace=0.35, wspace=0.25)

    # ── Señal temporal EMG ───────────────────────────────────────────
    ax = axes[0, 0]
    estilo_ax(ax)
    ax.plot(t_emg, emg_rs[:, canal_emg], lw=0.6, color='#6FC7B7', alpha=0.9)
    ax.set_title(f"EMG filtrado y resampleado  —  {nombre_emg}",
                 color=COLOR_LABEL, fontsize=9)
    ax.set_xlabel("Tiempo (s)", color=COLOR_LABEL, fontsize=8)
    ax.set_ylabel("Amplitud", color=COLOR_LABEL, fontsize=8)
    ax.grid(True, color=COLOR_GRID, linewidth=0.4, linestyle='--')

    # ── PSD EMG ──────────────────────────────────────────────────────
    ax = axes[0, 1]
    estilo_ax(ax)
    ax.plot(f_emg[mask_emg], p_emg[mask_emg], lw=1.4, color='#6FC7B7')
    for (bname, (f0, f1)), col in zip(BANDAS.items(), colores_bandas):
        ax.axvspan(f0, min(f1, fmax), alpha=0.15, color=col, zorder=0)
        ax.text((f0 + f1) / 2, p_emg[mask_emg].max() * 1.5, bname,
                color=col, fontsize=6, ha='center', va='bottom')
    ax.set_title(f"PSD  —  {nombre_emg}", color=COLOR_LABEL, fontsize=9)
    ax.set_xlabel("Frecuencia (Hz)", color=COLOR_LABEL, fontsize=8)
    ax.set_ylabel("PSD", color=COLOR_LABEL, fontsize=8)
    ax.set_xlim(0, fmax)
    ax.grid(True, color=COLOR_GRID, linewidth=0.4, linestyle='--')

    # ── Señal temporal EEG ───────────────────────────────────────────
    ax = axes[1, 0]
    estilo_ax(ax)
    ax.plot(t_eeg, eeg_filt[:, canal_eeg], lw=0.6, color='#81DCEE', alpha=0.9)
    ax.set_title(f"EEG filtrado  —  {nombre_eeg}",
                 color=COLOR_LABEL, fontsize=9)
    ax.set_xlabel("Tiempo (s)", color=COLOR_LABEL, fontsize=8)
    ax.set_ylabel("Amplitud (μV)", color=COLOR_LABEL, fontsize=8)
    ax.grid(True, color=COLOR_GRID, linewidth=0.4, linestyle='--')

    # ── PSD EEG ──────────────────────────────────────────────────────
    ax = axes[1, 1]
    estilo_ax(ax)
    ax.plot(f_eeg[mask_eeg], p_eeg[mask_eeg], lw=1.4, color='#81DCEE')
    for (bname, (f0, f1)), col in zip(BANDAS.items(), colores_bandas):
        ax.axvspan(f0, min(f1, fmax), alpha=0.15, color=col, zorder=0)
        ax.text((f0 + f1) / 2, p_eeg[mask_eeg].max() * 1.5, bname,
                color=col, fontsize=6, ha='center', va='bottom')
    ax.set_title(f"PSD  —  {nombre_eeg}", color=COLOR_LABEL, fontsize=9)
    ax.set_xlabel("Frecuencia (Hz)", color=COLOR_LABEL, fontsize=8)
    ax.set_ylabel("PSD (μV²/Hz)", color=COLOR_LABEL, fontsize=8)
    ax.set_xlim(0, fmax)
    ax.grid(True, color=COLOR_GRID, linewidth=0.4, linestyle='--')

    fig.suptitle("Señales filtradas y densidades espectrales de potencia",
                 color=COLOR_TITLE, fontsize=11, fontweight='bold')
    
def plot_señales_psd2(emg_rs, emg_filt, emg_names, eeg_names,
                     canal_eeg=5, canal_emg=0, fmax=200):
    """
    Figura con 4 subplots:
      - Señal temporal EMG (un canal)
      - PSD del EMG
      - Señal temporal EEG (un canal)
      - PSD del EEG
    """
    # FS_EEGAUX = FS_EEG
    # FS_EEG = 250
    #FS_FINAL = 1000
    
    t_emg = np.arange(emg_rs.shape[0])  / FS_FINAL
    t_emg2 = np.arange(emg_filt.shape[0]) / FS_EMG

    f_emg, p_emg = signal.welch(emg_rs[:,  canal_emg], fs=FS_FINAL,
                                 nperseg=int(FS_FINAL * SEG_DUR))
    f_emg2, p_emg2 = signal.welch(emg_filt[:, canal_eeg], fs=FS_EMG,
                                 nperseg=int(FS_EMG  * SEG_DUR))

    mask_emg = f_emg <= fmax
    mask_emg2 = f_emg2 <= fmax

    nombre_emg = emg_names[canal_emg] if emg_names else f"EMG_{canal_emg}"
    nombre_eeg = eeg_names[canal_eeg] if eeg_names else f"EEG_{canal_eeg}"

    # FS_EEG = FS_EEGAUX


    colores_bandas = ['#9B3461', '#C15579', '#E77692', '#f0B8AB', '#FAFAC4']

    fig, axes = plt.subplots(2, 2, figsize=(16, 7))
    fig.patch.set_facecolor(BG_FIG)
    fig.subplots_adjust(left=0.07, right=0.97,
                        top=0.91, bottom=0.09,
                        hspace=0.35, wspace=0.25)

    # ── Señal temporal EMG ───────────────────────────────────────────
    ax = axes[0, 0]
    estilo_ax(ax)
    ax.plot(t_emg, emg_rs[:, canal_emg], lw=0.6, color='#6FC7B7', alpha=0.9)
    ax.set_title(f"EMG filtrado y resampleado  —  {nombre_emg}",
                 color=COLOR_LABEL, fontsize=9)
    ax.set_xlabel("Tiempo (s)", color=COLOR_LABEL, fontsize=8)
    ax.set_ylabel("Amplitud", color=COLOR_LABEL, fontsize=8)
    ax.grid(True, color=COLOR_GRID, linewidth=0.4, linestyle='--')

    # ── PSD EMG ──────────────────────────────────────────────────────
    ax = axes[0, 1]
    estilo_ax(ax)
    ax.plot(f_emg[mask_emg], p_emg[mask_emg], lw=1.4, color='#6FC7B7')
    for (bname, (f0, f1)), col in zip(BANDAS.items(), colores_bandas):
        ax.axvspan(f0, min(f1, fmax), alpha=0.15, color=col, zorder=0)
        ax.text((f0 + f1) / 2, p_emg[mask_emg].max() * 1.5, bname,
                color=col, fontsize=6, ha='center', va='bottom')
    ax.set_title(f"PSD  —  {nombre_emg}", color=COLOR_LABEL, fontsize=9)
    ax.set_xlabel("Frecuencia (Hz)", color=COLOR_LABEL, fontsize=8)
    ax.set_ylabel("PSD", color=COLOR_LABEL, fontsize=8)
    ax.set_xlim(0, fmax)
    ax.grid(True, color=COLOR_GRID, linewidth=0.4, linestyle='--')

    # ── Señal temporal EEG ───────────────────────────────────────────
    ax = axes[1, 0]
    estilo_ax(ax)
    ax.plot(t_emg2, emg_filt[:, canal_eeg], lw=0.6, color='#81DCEE', alpha=0.9)
    ax.set_title(f"EEG filtrado  —  {nombre_eeg}",
                 color=COLOR_LABEL, fontsize=9)
    ax.set_xlabel("Tiempo (s)", color=COLOR_LABEL, fontsize=8)
    ax.set_ylabel("Amplitud (μV)", color=COLOR_LABEL, fontsize=8)
    ax.grid(True, color=COLOR_GRID, linewidth=0.4, linestyle='--')

    # ── PSD EEG ──────────────────────────────────────────────────────
    ax = axes[1, 1]
    estilo_ax(ax)
    ax.plot(f_emg2[mask_emg2], p_emg2[mask_emg2], lw=1.4, color='#81DCEE')
    for (bname, (f0, f1)), col in zip(BANDAS.items(), colores_bandas):
        ax.axvspan(f0, min(f1, fmax), alpha=0.15, color=col, zorder=0)
        ax.text((f0 + f1) / 2, p_emg2[mask_emg2].max() * 1.5, bname,
                color=col, fontsize=6, ha='center', va='bottom')
    ax.set_title(f"PSD  —  {nombre_eeg}", color=COLOR_LABEL, fontsize=9)
    ax.set_xlabel("Frecuencia (Hz)", color=COLOR_LABEL, fontsize=8)
    ax.set_ylabel("PSD (μV²/Hz)", color=COLOR_LABEL, fontsize=8)
    ax.set_xlim(0, fmax)
    ax.grid(True, color=COLOR_GRID, linewidth=0.4, linestyle='--')

    fig.suptitle("Señales filtradas y densidades espectrales de potencia",
                 color=COLOR_TITLE, fontsize=11, fontweight='bold')
    

def concatenate_multi_channel_eeg(arr1, arr2, fs=250, ms=40):
    """
    Concatena dos arreglos de forma (7, muestras) con suavizado.
    """
    # 1. Calcular número de muestras para el cross-fade
    n_samples = int((ms / 1000) * fs)
    
    # 2. Crear las rampas y expandirlas para que coincidan con (7, n_samples)
    # Usamos rampa senoidal por ser más suave para bioseñales
    t = np.linspace(0, np.pi/2, n_samples)
    fade_in = (np.sin(t)**2)   # Forma: (n_samples,)
    fade_out = (np.cos(t)**2)  # Forma: (n_samples,)
    
    # 3. Extraer las partes que se van a solapar
    # Tomamos las últimas muestras de arr1 y las primeras de arr2 en todos los canales
    overlap_zone_1 = arr1[:, -n_samples:]
    overlap_zone_2 = arr2[:, :n_samples]
    
    # 4. Aplicar el cross-fade (NumPy hace el broadcasting automáticamente)
    overlap = (overlap_zone_1 * fade_out) + (overlap_zone_2 * fade_in)
    
    # 5. Concatenar las tres partes a lo largo del eje 1 (el tiempo)
    result = np.concatenate([
        arr1[:, :-n_samples], 
        overlap, 
        arr2[:, n_samples:]
    ], axis=1)
    
    return result



def concatenate_eeg_list(signals_list, fs=250, ms=40):
    """
    Concatena una lista de arrays (7, muestras) usando cross-fade.
    """
    if not signals_list:
        return None
    
    # 1. Definir parámetros del cross-fade
    n_samples = int((ms / 1000) * fs)
    t = np.linspace(0, np.pi/2, n_samples)
    fade_in = (np.sin(t)**2)
    fade_out = (np.cos(t)**2)
    
    # 2. Tomar la primera señal como base
    combined = signals_list[0]
    
    # 3. Iterar sobre el resto de las señales
    for next_sig in signals_list[1:]:
        # Zona de solapamiento entre el acumulado y la siguiente señal
        overlap = (combined[:, -n_samples:] * fade_out) + (next_sig[:, :n_samples] * fade_in)
        
        # Unir
        combined = np.concatenate([
            combined[:, :-n_samples], 
            overlap, 
            next_sig[:, n_samples:]
        ], axis=1)
        
    return combined




# ══════════════════════════════════════════════
# PRINCIPAL
# ══════════════════════════════════════════════

def calcula_todo(emg, eeg, RUTA_SALIDA, emg_channels=None, eeg_channels=None, t_start=None, t_end=None,
                 emg_names=None, eeg_names=None, graficar=False, PIE_FIGURA = "", envolvente = True,eeg_to_plot=None):
    
    
    emg = emg.T
    eeg = eeg.T
    

    
    #PIE_FIGURA, TAG_ARCHIVO, PARTES = parsear_nombre_archivo(RUTA_EEG)
    print(f"\nMetadatos:\n  {PIE_FIGURA}") 

    #REMUESTREAR
    if envolvente:
        emg_rs = procesar_emg(emg.T)
        emg_rs = emg_rs.T

    else:
        emg_filt = filtrar_emg(emg,fs = FS_EMG)
        emg_rs = resample_signal(emg_filt,fs_in = FS_EMG, fs_out=FS_FINAL)

    eeg_filt = eeg
    #COHERENCIA 
    freqs,coherencia = calcula_coherencia(emg_rs,eeg_filt, fs = FS_FINAL)

    # Límite de confianza
    n_segs = n_segmentos(len(emg_rs))
    CL     = confidence_limit(n_segs, alpha=0.05)
    print(f"\nSegmentos usados : {n_segs}")
    print(f"Límite de confianza (95%): {CL:.4f}")

    #Encuentra la coherencia máxima registrada 
    
    #excluye_freqs = ["delta","theta","mu","gamma","Otros"] # bandas a excluir en el cálculo de la coherencia máxima
    excluye_freqs = ["δ","θ","μ","γ","Otros"]
    mask_excluye = np.zeros(len(freqs),dtype = bool)

    for banda in excluye_freqs:
        fl, fh = BANDAS[banda]
        mask_excluye |= (freqs >= fl) & (freqs <= fh)
    coherencia_excluye = coherencia.copy()
    coherencia_excluye[mask_excluye ,:,:] = 0 

    idx = coherencia_excluye.argmax()
    i_freq, i_emg, i_eeg = np.unravel_index(idx, coherencia_excluye.shape)
    print(f" Coherencia maxima: {coherencia[i_freq, i_emg, i_eeg]:.4f}")
    print(f" Frecuencia       : {freqs[i_freq]:.1f} ")
    print(f" Canal EMG        : {emg_names[i_emg]}")
    print(f" Canal EEG        : {eeg_names[i_eeg]}")
    print(f"{coherencia.shape}")


    if graficar:
        plot_señales_psd(emg_rs, eeg_filt, emg_names, eeg_names, canal_eeg=0, canal_emg=0)
        plot_coherencia(freqs, coherencia, emg_names, eeg_names, fmax = 45, pie=PIE_FIGURA, confidence_limit=CL,eeg_to_plot=eeg_to_plot)


        plt.show()

    return {
        "freqs":     freqs,
        "coherencia": coherencia,
        "limiteC": CL 
    }

# ══════════════════════════════════════════════
# PUNTO DE ENTRADA
# ══════════════════════════════════════════════
if __name__ == "__main__":
    
    ruta_base = r"C:\Users\lrl13\Downloads\DATA CMC 2"
    RUTA_SALIDA = r"C:\Users\lrl13\OneDrive - Instituto Politecnico Nacional\10\reportes_CMC"
    tipos = ["EEG","EMG"]
    sujetos = ["S4","S5","S6","S7","S17","S18"]
    #EMG_NAMES = ["Flexor carpi ulnaris","Right Extensor carpi radialis longus","Extensor carpi ulnaris"]
    EEG_NAMES = ["Fz","FC3","FCz","FC4","Cz","C3","C4"]
    EMG_NAMES = ["Flexor Cubital del Carpo","Extensor Radial Largo del Carpo","Extensor Cubital del Carpo"]#Nombres en Español
    eeg_to_plot = ["C3"]

    FREQS_NOTCH = (60.0, 120.0)
    Q_NOTCH = 5
    PAD = 5 * FS_EEG            # PADDING PARA FILTROS
    PAD_MUESTRAS = 5*FS_EMG
    # Wavelet
    WAVELET = 'db4'
    LEVEL = 6
    NIVELES_CERO = [1, 6]   # QUITAR D1 y D6
    REMOVE_APPROX = True    # QUITAR A6
    WAVE_MODE = 'reflect'

    # ICA
    IC_TO_REMOVE = 0        # IC A QUITAR [X,X]

    # PASA BANDA
    BP_LOW =    5.0         
    BP_HIGH =   45.0
    BP_ORDER =  3
    concatenar = True
    envolvente = True



    pares_analisis = generar_pares(ruta_base,tipos,sujetos)

    #print("\nPARES PARA ANALISIS\n")
    #print(pares_analisis)
   
    general = {}

    if concatenar:
        for sujeto in sujetos:
            eeg_list = []
            emg_list = []
            general[sujeto] = {}

            for i in range(4):
                eeg = load_eeg(pares_analisis[sujeto][i][0])
                eeg = eeg - eeg.mean()

                eeg = aplicar_notch_multiple(eeg, fs=FS_EEG, freqs=FREQS_NOTCH, Q=Q_NOTCH, pad=PAD)

                # 3) Wavelet
                # eeg = aplicar_wavelet_dwt_multicanal(
                #     eeg,
                #     wavelet=WAVELET,
                #     level=LEVEL,
                #     niveles_cero=NIVELES_CERO,
                #     remove_approx=REMOVE_APPROX,
                #     pad=PAD,
                #     mode=WAVE_MODE,
                # )
                if sujeto == "S5":
                # 4) ICA
                    eeg, _ica = ica_eliminar_componente(eeg, fs=FS_EEG, ch_names=EEG_NAMES, ic_to_remove=IC_TO_REMOVE)

                # 5) PASA BANDA
                eeg_final = aplicar_pasabanda_butter(eeg, fs=FS_EEG, f_low=BP_LOW, f_high=BP_HIGH, orden=BP_ORDER, pad=PAD)

                eeg_list.append(eeg_final)

                if i < 4:
                    
                    emg_tmp = load_emg(pares_analisis[sujeto][i][1])
                    emg_list.append(emg_tmp)

            for k in range(3):
                for j in range(4):
                    emg_list.append(emg_list[j])
                    eeg_list.append(eeg_list[j])
            
            emg = concatenate_eeg_list(emg_list,fs=FS_EMG, ms = 20)    
            eeg = concatenate_eeg_list(eeg_list, ms = 20)
            fig, ax = plt.subplots()
            fig.patch.set_facecolor(BG_FIG)
            ax.plot(eeg[0,:], lw=0.6, color='#6FC7B7', alpha=0.9)
            ax.set_title(f"EEG preprocesado y concatenado",
                    color=COLOR_LABEL, fontsize=9)
            ax.set_xlabel("Número de muestras", color=COLOR_LABEL, fontsize=8)
            ax.set_ylabel("Amplitud", color=COLOR_LABEL, fontsize=8)
            ax.grid(True, color=COLOR_GRID, linewidth=0.4, linestyle='--')
            estilo_ax(ax)
            plt.show()

            results = calcula_todo(
                emg    = emg,
                eeg    = eeg,
                RUTA_SALIDA = RUTA_SALIDA,
                emg_names = EMG_NAMES,
                eeg_names = EEG_NAMES,
                t_start      = None,          # None = solapamiento maximo
                t_end        = None,
                graficar     = True,
                PIE_FIGURA = sujeto,
                envolvente = envolvente,
                eeg_to_plot=eeg_to_plot
   
            )
            general[sujeto] = results

        RUTA_SALIDA = Path(RUTA_SALIDA)
        RUTA_SALIDA = RUTA_SALIDA /"Reporte_GENERAL.npy"
        RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
        np.save(RUTA_SALIDA, general)

    else:
        for i in range(4):    
            for sujeto in sujetos:
                eeg_list = []
                emg_list = []
                general[sujeto] = {}

                eeg = load_eeg(pares_analisis[sujeto][i][0])
                eeg = eeg - eeg.mean()

                eeg = aplicar_notch_multiple(eeg, fs=FS_EEG, freqs=FREQS_NOTCH, Q=Q_NOTCH, pad=PAD)

                # 3) Wavelet
                # eeg = aplicar_wavelet_dwt_multicanal(
                #     eeg,
                #     wavelet=WAVELET,
                #     level=LEVEL,
                #     niveles_cero=NIVELES_CERO,
                #     remove_approx=REMOVE_APPROX,
                #     pad=PAD,
                #     mode=WAVE_MODE,
                # )
                if sujeto == "S5":
                # 4) ICA
                    eeg, _ica = ica_eliminar_componente(eeg, fs=FS_EEG, ch_names=EEG_NAMES, ic_to_remove=IC_TO_REMOVE)

                # 5) PASA BANDA
                eeg_final = aplicar_pasabanda_butter(eeg, fs=FS_EEG, f_low=BP_LOW, f_high=BP_HIGH, orden=BP_ORDER, pad=PAD)


                if i < 4:
                    emg_tmp = load_emg(pares_analisis[sujeto][i][1])
            

                results = calcula_todo(
                    emg    = emg_tmp,
                    eeg    = eeg_final,
                    RUTA_SALIDA = RUTA_SALIDA,
                    emg_names = EMG_NAMES,
                    eeg_names = EEG_NAMES,
                    t_start      = None,          # None = solapamiento maximo
                    t_end        = None,
                    graficar     = True,
                    PIE_FIGURA = sujeto,
                    envolvente = envolvente
                )
                general[sujeto] = results

            RUTA_SALIDA = Path(RUTA_SALIDA)
            RUTA_SALIDA = RUTA_SALIDA /f"Reporte_GENERAL_{i}.npy"
            RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
            np.save(RUTA_SALIDA, general) 



        

#método de bootstrap 
