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
EMG_BAND  = (2.0,400.0)   
EEG_BAND  = (5,  100.0)   
NOTCH_F   = 60.0            

# Segmentación (Welch)
SEG_DUR   = 1.0     
OVERLAP   = 0.5     

# Bandas de frecuencia 
BANDAS = {
    "delta":  (0,   4),
    "theta":  (4,   8),
    "mu":     (8,  13),
    "beta":   (13, 30),
    "gamma":  (30, 80),
    "Otros":  (80,1000)#límite superior de gamma es de 80
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

# --- ESTOS COLORES SON PARA VERSION CLARA

# COLORES = [    '#477e75', '#4b7168', '#4d615f',
#                '#515051',  '#475e63', '#3d6e74', '#307c8a']

# BG_FIG      = '#f5f5f5'
# BG_AX       = '#f2f2f2'
# COLOR_GRID  = '#bfbfbf'
# COLOR_TICK  = '#5a5a5a'
# COLOR_LABEL = '#2e2e2e'
# COLOR_TITLE = '#1e1e1e'

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

def _notch(f0: float, fs: float, Q: float = 35) -> tuple:
    b, a = signal.iirnotch(f0, Q, fs)
    return b, a

def butter_pasabanda(lowcut: float, highcut: float, fs: float,
                     orden: int = 6) -> tuple:
    nyq = fs / 2.0
    b, a = signal.butter(orden, [lowcut / nyq, highcut / nyq], btype="band")
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

def rectificar_emg(emg: np.ndarray) -> np.ndarray:
    return np.abs(emg)
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
    return 1 - (1 - alpha) ** (1 / (n_segmentos - 1))

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
        fig.set_size_inches(13, 8)
    except:
        fig.set_size_inches(13, 8)
def estilo_ax(ax):
    ax.set_facecolor(BG_AX)
    ax.tick_params(colors=COLOR_TICK, labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_color('#30363d')
    ax.spines['bottom'].set_color('#30363d')

def plot_coherencia(freqs, coherencia, emg_names, eeg_names, fmax=100, pie = '', confidence_limit =None):

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
            
            if confidence_limit is not None:
                ax.axhline(confidence_limit, color='#FF6B6B', lw=1.0,
                   linestyle='--', alpha=0.8, label=f'CL 95% = {confidence_limit:.3f}')
                if yi == 0:
                    ax.legend(loc="upper right", fontsize=7)

            for (bname, (f0, f1)), col in zip(BANDAS.items(), colores_bandas):
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


import numpy as np

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
                 emg_names=None, eeg_names=None, graficar=False, PIE_FIGURA = ""):
    #1. Carga señales 
    # emg = load_emg(RUTA_EMG).T
    # eeg = load_eeg(RUTA_EEG).T
    emg = emg.T
    eeg = eeg.T

    
    #PIE_FIGURA, TAG_ARCHIVO, PARTES = parsear_nombre_archivo(RUTA_EEG)
    print(f"\nMetadatos:\n  {PIE_FIGURA}") 
    # Filtrado
    emg_filt = filtrar_emg(emg,fs = FS_EMG)
    eeg_filt = filtrar_eeg(eeg,fs = FS_EEG)
    # emg_filt = emg
    # eeg_filt = eeg

    #rectificación
    emg_rect = rectificar_emg(emg_filt)
    
    #REMUESTREAR

    emg_rs = resample_signal(emg_rect,fs_in = FS_EMG, fs_out=FS_FINAL)   
    
    #COHERENCIA 
    freqs,coherencia = calcula_coherencia(emg_rs,eeg_filt, fs = FS_FINAL)

    # Límite de confianza
    n_segs = n_segmentos(len(emg_rs))
    CL     = confidence_limit(n_segs, alpha=0.05)
    print(f"\nSegmentos usados : {n_segs}")
    print(f"Límite de confianza (95%): {CL:.4f}")

    #Encuentra la coherencia máxima registrada 
    excluye_freqs = ["delta","theta","mu","gamma","Otros"] # bandas a excluir en el cálculo de la coherencia máxima
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

    resultados_reporte = {"coherencia":coherencia,
                          "frecuencias": freqs}
    #path_salida = r"C:\Users\lrl13\OneDrive - Instituto Politecnico Nacional\10\reportes_CMC"
    # RUTA_SALIDA = Path(RUTA_SALIDA)
    # RUTA_SALIDA = RUTA_SALIDA / PARTES[0]/ f"Reporte_{TAG_ARCHIVO}.npy"
    # RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    # np.save(RUTA_SALIDA, resultados_reporte)



    if graficar:
        plot_señales_psd(emg_rs, eeg_filt, emg_names, eeg_names, canal_eeg=0, canal_emg=0)
        plot_coherencia(freqs, coherencia, emg_names, eeg_names, fmax = 100, pie=PIE_FIGURA, confidence_limit=CL)

        plt.show()

    return {
        "freqs":     freqs,
        "coherencia": coherencia,
    }

# ══════════════════════════════════════════════
# PUNTO DE ENTRADA
# ══════════════════════════════════════════════
if __name__ == "__main__":
    
    ruta_base = r"C:\Users\lrl13\Downloads\DATA CMC 2"
    RUTA_SALIDA = r"C:\Users\lrl13\OneDrive - Instituto Politecnico Nacional\10\reportes_CMC"
    tipos = ["EEG","EMG"]
    sujetos = ["S5","S7","S17","S18"]
    EMG_NAMES = ["Flexor carpi ulnaris","Right Extensor carpi radialis longus","Extensor carpi ulnaris"]
    EEG_NAMES = ["Fz","FC3","FCz","FC4","Cz","C3","C4"]

    pares_analisis = generar_pares(ruta_base,tipos,sujetos)

    print("\nPARES PARA ANALISIS\n")
    print(pares_analisis)
   
    general = {}

    for sujeto in sujetos:
        eeg_list = []
        emg_list = []
        general[sujeto] = {}

        for i in range(4):
            eeg_tmp = load_eeg(pares_analisis[sujeto][i][0])
            eeg_tmp -= np.mean(eeg_tmp, axis=1, keepdims=True)
            eeg_tmp = filtrar_eeg(eeg_tmp.T, fs = FS_EEG)
            eeg_tmp = eeg_tmp.T
            eeg_list.append(eeg_tmp)

            if i < 4:
                emg_tmp = load_emg(pares_analisis[sujeto][i][1])
                emg_tmp -= np.mean(emg_tmp, axis=1, keepdims=True)
                emg_tmp = filtrar_emg(emg_tmp.T, fs = FS_EMG)
                emg_tmp = emg_tmp.T
                emg_list.append(emg_tmp)

        # eeg = np.concatenate(eeg_list, axis=1)
        #emg = np.concatenate(emg_list, axis=1)
        emg = concatenate_eeg_list(emg_list,fs=FS_EMG)    
        eeg = concatenate_eeg_list(eeg_list)
                #signals_eeg_sujeto = np.concatenate((signals_eeg_sujeto,eeg))
        plt.plot(eeg[0,:])
        plt.show()



        results = calcula_todo(
            emg    = emg,
            eeg    = eeg,
            RUTA_SALIDA = RUTA_SALIDA,
            emg_names = EMG_NAMES,
            eeg_names = EEG_NAMES,
            t_start      = None,          # None = solapamiento maximo
            t_end        = None,
            graficar     = False,
            PIE_FIGURA = sujeto
        )
        general[sujeto] = results

    
    # general = {}
    # for sujeto in sujetos:
    #     general[sujeto] = {}
    #     for i in range(4):
    #         results = calcula_todo(
    #             RUTA_EMG     = pares_analisis[sujeto][i][1],
    #             RUTA_EEG     = pares_analisis[sujeto][i][0],
    #             RUTA_SALIDA = RUTA_SALIDA,
    #             emg_names = EMG_NAMES,
    #             eeg_names = EEG_NAMES,
    #             t_start      = None,          # None = solapamiento maximo
    #             t_end        = None,
    #             graficar     = False
    #         )
    #         general[sujeto][i] = results
    
    RUTA_SALIDA = Path(RUTA_SALIDA)
    RUTA_SALIDA = RUTA_SALIDA /"Reporte_GENERAL.npy"
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    np.save(RUTA_SALIDA, general)



