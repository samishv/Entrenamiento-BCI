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

    descripcion = f"Sujeto {num_sujeto}  |  {desc_cond}  |  {desc_estado}"
    tag_archivo = f"{sujeto}_{condicion}_{estado}"

    return descripcion, tag_archivo


# ══════════════════════════════════════════════
# 3. FILTRADO
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
# 8. COHERENCIA, LÍMITE DE CONFIANZA E ÍNDICE DE SINCRONÍA DE FASE
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

def calcula_plv(emg, eeg, fs=FS_FINAL):
    """
    Phase Locking Value por segmentos.
    Para cada par (canal EMG, canal EEG) calcula el PLV en cada frecuencia.
    Devuelve (freqs, plv) con plv shape (n_freqs, n_emg, n_eeg).
    """
    from math import gcd
    seg_len = int(fs * SEG_DUR)
    step    = int(seg_len * (1 - OVERLAP))
    n_emg   = emg.shape[1]
    n_eeg   = eeg.shape[1]
    n_freqs = seg_len // 2 + 1

    starts = np.arange(0, min(len(emg), len(eeg)) - seg_len + 1, step)
    win    = np.hanning(seg_len)

    # Acumula los fasores complejos de fase diferencial
    sum_fasores = np.zeros((n_freqs, n_emg, n_eeg), dtype=complex)

    for s in starts:
        for ei in range(n_emg):
            X = np.fft.rfft(emg[s:s+seg_len, ei] * win)
            fase_x = np.angle(X)
            for yi in range(n_eeg):
                Y = np.fft.rfft(eeg[s:s+seg_len, yi] * win)
                fase_y = np.angle(Y)
                # Fasor unitario de la diferencia de fase
                sum_fasores[:, ei, yi] += np.exp(1j * (fase_x - fase_y))

    plv   = np.abs(sum_fasores) / len(starts)   # promedio → [0, 1]
    freqs = np.fft.rfftfreq(seg_len, d=1.0/fs)
    return freqs, plv

def n_segmentos(n_muestras, fs=FS_FINAL):
    seg_len = int(fs * SEG_DUR)
    step    = int(seg_len * (1 - OVERLAP))
    return len(np.arange(0, n_muestras - seg_len + 1, step))

def calcula_fase(emg, eeg, fs=FS_FINAL):
    """
    Calcula el espectro cruzado y extrae la fase promedio
    para cada par (canal EMG, canal EEG).
    Devuelve (freqs, fase) con fase shape (n_freqs, n_emg, n_eeg)
    en grados, y el retraso temporal en ms.
    """
    n_emg   = emg.shape[1]
    n_eeg   = eeg.shape[1]
    nperseg = int(fs * SEG_DUR)
    noverlap= int(nperseg * OVERLAP)

    # Calcular frecuencias con una llamada de prueba
    freqs, _ = signal.csd(emg[:, 0], eeg[:, 0], fs=fs,
                          nperseg=nperseg, noverlap=noverlap)
    fase_rad  = np.zeros((len(freqs), n_emg, n_eeg))

    for ei in range(n_emg):
        for yi in range(n_eeg):
            _, Sxy = signal.csd(emg[:, ei], eeg[:, yi],
                                fs=fs, nperseg=nperseg, noverlap=noverlap)
            fase_rad[:, ei, yi] = np.angle(Sxy)   # radianes [-π, π]

    # Convertir a grados y calcular retraso en ms
    fase_deg = np.degrees(fase_rad)
    # retraso(f) = fase(f) / (2π * f)   →  en ms
    retraso_ms = np.zeros_like(fase_rad)
    for i, f in enumerate(freqs):
        if f > 0:
            retraso_ms[i] = fase_rad[i] / (2 * np.pi * f) * 1000

    return freqs, fase_deg, retraso_ms
# ══════════════════════════════════════════════
# VISUALIZACIÓN
# ══════════════════════════════════════════════
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

def plot_fase(freqs, fase_deg, retraso_ms, coherencia,
              emg_names, eeg_names, fmax=100, CL=None):
    """
    Solo grafica la fase y retraso en frecuencias donde
    la coherencia es estadísticamente significativa.
    """
    mask   = freqs <= fmax
    colores_bandas = ['#9B3461', '#C15579', '#E77692', '#f0B8AB', '#FAFAC4']

    for ei, ename in enumerate(emg_names):
        fig, axes = plt.subplots(
            len(eeg_names), 2,
            figsize=(16, 2.5 * len(eeg_names)),
            sharex=True
        )
        fig.patch.set_facecolor(BG_FIG)
        fig.subplots_adjust(left=0.07, right=0.97,
                            top=0.92, bottom=0.07,
                            hspace=0.15, wspace=0.25)

        for yi, (yname, color) in enumerate(zip(eeg_names, COLORES)):
            # Máscara de significancia: solo graficar donde coh > CL
            coh_yi = coherencia[:, ei, yi]
            sig_mask = (coh_yi > CL) & mask if CL is not None else mask

            # ── Fase ────────────────────────────────────────
            ax_fase = axes[yi, 0]
            estilo_ax(ax_fase)
            # Puntos no significativos en gris
            ax_fase.scatter(freqs[mask & ~sig_mask],
                            fase_deg[mask & ~sig_mask, ei, yi],
                            s=3, color='#404040', alpha=0.4)
            # Puntos significativos en color
            ax_fase.scatter(freqs[sig_mask],
                            fase_deg[sig_mask, ei, yi],
                            s=6, color=color, alpha=0.9, zorder=3)

            ax_fase.axhline(0, color='#606060', lw=0.8, linestyle='--')
            ax_fase.set_ylabel(yname, color=color, fontsize=8,
                               rotation=0, labelpad=36, va='center')
            ax_fase.set_ylim(-185, 185)
            ax_fase.set_yticks([-180, -90, 0, 90, 180])
            ax_fase.grid(True, color=COLOR_GRID, linewidth=0.4, linestyle='--')

            for (bname, (f0, f1)), col in zip(BANDAS.items(), colores_bandas):
                ax_fase.axvspan(f0, min(f1, fmax), alpha=0.1, color=col)

            if yi == 0:
                ax_fase.set_title('Fase (°)', color=COLOR_LABEL, fontsize=8)

            # ── Retraso ─────────────────────────────────────
            ax_ret = axes[yi, 1]
            estilo_ax(ax_ret)
            ax_ret.scatter(freqs[mask & ~sig_mask],
                           retraso_ms[mask & ~sig_mask, ei, yi],
                           s=3, color='#404040', alpha=0.4)
            ax_ret.scatter(freqs[sig_mask],
                           retraso_ms[sig_mask, ei, yi],
                           s=6, color=color, alpha=0.9, zorder=3)

            ax_ret.axhline(0,  color='#606060', lw=0.8, linestyle='--')
            # Referencia fisiológica: rango esperado corticoespinal
            ax_ret.axhspan(15, 35, color='#4CAF50', alpha=0.08,
                           label='Rango corticoespinal (15–35 ms)')
            ax_ret.set_ylim(-50, 50)
            ax_ret.grid(True, color=COLOR_GRID, linewidth=0.4, linestyle='--')

            for (bname, (f0, f1)), col in zip(BANDAS.items(), colores_bandas):
                ax_ret.axvspan(f0, min(f1, fmax), alpha=0.1, color=col)

            if yi == 0:
                ax_ret.set_title('Retraso (ms)', color=COLOR_LABEL, fontsize=8)
                ax_ret.legend(fontsize=6, loc='upper right')

            if yi == len(eeg_names) - 1:
                ax_fase.set_xlabel('Frecuencia (Hz)', color=COLOR_LABEL, fontsize=8)
                ax_ret.set_xlabel('Frecuencia (Hz)',  color=COLOR_LABEL, fontsize=8)

        fig.suptitle(f'Fase y retraso  —  EMG: {ename}',
                     color=COLOR_TITLE, fontsize=11, fontweight='bold')


def reporte_cmc(freqs, coherencia, fase, emg_names, eeg_names,
                bandas_reporte=["mu", "beta", "gamma"], CL=None,
                pie='', tag_archivo = 'cmc', exportar=True, ruta_salida=None):
  
    filas = []

    print("=" * 60)
    print("REPORTE CMC")
    if pie:
        print(pie)
    print("=" * 60)

    for ei, ename in enumerate(emg_names):
        print(f"\nEMG: {ename}")
        print("-" * 40)

        for banda in bandas_reporte:
            f0, f1     = BANDAS[banda]
            mask_banda = (freqs >= f0) & (freqs <= f1)

            coh_banda = coherencia[mask_banda, ei, :]
            idx       = coh_banda.argmax()
            i_f, i_y  = np.unravel_index(idx, coh_banda.shape)

            coh_max  = coh_banda[i_f, i_y]
            freq_max = freqs[mask_banda][i_f]
            eeg_max  = eeg_names[i_y]

            fase_val   = fase[mask_banda, ei, i_y][i_f]
            retraso_ms = (np.radians(fase_val) /
                         (2 * np.pi * freq_max)) * 1000

            es_sig = (coh_max > CL) if CL is not None else None
            sig_str = ""
            if CL is not None:
                sig_str = "✓ significativo" if es_sig else "✗ no significativo"

            print(f"  {banda:6s} | Coh = {coh_max:.4f}  {sig_str}")
            print(f"         | Freq = {freq_max:.1f} Hz  |  EEG: {eeg_max}")
            print(f"         | Retraso = {retraso_ms:.1f} ms")

            # Parsear metadatos del pie
            sujeto = condicion = estado = ""
            if pie:
                partes = [p.strip() for p in pie.split("|")]
                if len(partes) >= 1: sujeto    = partes[0].replace("Sujeto", "").strip()
                if len(partes) >= 2: condicion = partes[1].strip()
                if len(partes) >= 3: estado    = partes[2].strip()

            filas.append({
                "sujeto":        sujeto,
                "condicion":     condicion,
                "estado":        estado,
                "emg_canal":     ename,
                "banda":         banda,
                "f_lo_hz":       f0,
                "f_hi_hz":       f1,
                "coh_max":       round(coh_max, 6),
                "freq_max_hz":   round(freq_max, 2),
                "eeg_canal":     eeg_max,
                "retraso_ms":    round(retraso_ms, 2),
                "CL_95":         round(CL, 6) if CL is not None else None,
                "significativo": es_sig,
            })

    print("=" * 60)

    if not exportar or not filas:
        return pd.DataFrame(filas)

    df = pd.DataFrame(filas)

    # Definir ruta de salida
    carpeta = Path(ruta_salida) if ruta_salida else Path(".")
    carpeta.mkdir(parents=True, exist_ok=True)

    # Nombre de archivo basado en metadatos
    tag = pie.replace(" ", "_").replace("|", "").replace("/", "-") if pie else "cmc"
    tag = "".join(c for c in tag if c.isalnum() or c in "_-")

    ruta_csv   = carpeta / f"reporte_cmc_{tag_archivo}.csv"
    ruta_excel = carpeta / f"reporte_cmc_{tag_archivo}.xlsx"

    # Exportar CSV
    df.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
    print(f"\n  CSV   guardado → {ruta_csv}")

    # Exportar Excel con formato
    with pd.ExcelWriter(ruta_excel, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="CMC")

        ws = writer.sheets["CMC"]

        # Ancho de columnas automático
        for col in ws.columns:
            max_len = max(len(str(cell.value)) for cell in col if cell.value)
            ws.column_dimensions[col[0].column_letter].width = max_len + 4

        # Colorear filas según significancia
        from openpyxl.styles import PatternFill, Font
        # verde = PatternFill(start_color="1A3A2A", end_color="1A3A2A", fill_type="solid")
        # rojo  = PatternFill(start_color="3A1A1A", end_color="3A1A1A", fill_type="solid")
        bold  = Font(bold=True)

        sig_col_idx = df.columns.get_loc("significativo") + 1  # 1-indexed

        for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
            coh_val = row[df.columns.get_loc("coh_max")].value
            if coh_val == df["coh_max"].max():
                for cell in row:
                    cell.font = bold

    print(f"  Excel guardado → {ruta_excel}\n")

    return df
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
   
    PIE_FIGURA, TAG_ARCHIVO = parsear_nombre_archivo(RUTA_EEG)
    print(f"\nMetadatos:\n  {PIE_FIGURA}") 
    # Filtrado
    emg_filt = filtrar_emg(emg,fs = FS_EMG)
    eeg_filt = filtrar_eeg(eeg,fs = FS_EEG)
    #eeg_filt = eeg_filt.T
    #rectificación
    emg_rect = rectificar_emg(emg_filt)
    
    #REMUESTREAR
    emg_rs = emg_rect
    #emg_rs = resample_signal(emg_rect,fs_in = FS_EMG, fs_out=FS_FINAL)   
    eeg_filt = resample_signal(eeg_filt,fs_in = FS_EEG,fs_out = FS_EMG)
    # plt.plot(eeg_filt)
    # plt.show()
    #COHERENCIA 
    #freqs,coherencia = calcula_coherencia(emg_rs,eeg_filt)
    freqs,coherencia = calcula_coherencia(emg_rs,eeg_filt, fs = 1000)
    _, fase_deg, retraso_ms = calcula_fase(emg_rs, eeg_filt)

    # Límite de confianza
    n_segs = n_segmentos(len(emg_rs))
    CL     = confidence_limit(n_segs, alpha=0.05)
    print(f"\nSegmentos usados : {n_segs}")
    print(f"Límite de confianza (95%): {CL:.4f}")

    #Encuentra la coherencia máxima registrada 
    excluye_freqs = ["delta","theta","mu","gamma"] # bandas a excluir en el cálculo de la coherencia máxima
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
    # PLV
    #_, plv = calcula_plv(emg_rs, eeg_filt)
    path_salida = r"C:\Users\lrl13\OneDrive - Instituto Politecnico Nacional\10\reportes_CMC"

    plot_señales_psd(emg_rs, eeg_filt, emg_names, eeg_names, canal_eeg=0, canal_emg=0)
    plot_coherencia(freqs, coherencia, emg_names, eeg_names, fmax = 100, pie=PIE_FIGURA, confidence_limit=CL)
    # df_reporte = reporte_cmc(freqs, coherencia, fase_deg, emg_names, eeg_names,bandas_reporte=["beta", "gamma"], CL=CL, 
    #                          pie = PIE_FIGURA, tag_archivo = TAG_ARCHIVO, exportar = True, ruta_salida = path_salida)
    # #plot_fase(freqs, fase_deg, retraso_ms, coherencia,emg_names, eeg_names, CL=CL)
    # print(df_reporte)
    plt.show()

    # plot_señales_psd(emg_rs, eeg_filt, emg_names, eeg_names,canal_eeg=0, canal_emg=0)
    # plot_coherencia(freqs, coherencia, emg_names, eeg_names, pie = PIE_FIGURA)
    # plt.show()

    return {
        "freqs":     freqs,
        "coherencia": coherencia,
    }

# ══════════════════════════════════════════════
# PUNTO DE ENTRADA
# ══════════════════════════════════════════════
if __name__ == "__main__":

    
 


    #EEG_FILE = r"C:\Users\lrl13\Downloads\S41718\S17\EEG\S17_260310_EEG_CK_1_ME.npy"
    #EEG_FILE = r"C:\Users\lrl13\Downloads\S41718\S17\EEG\S17_260310_EEG_CK_4_ME.npy"
    # "C:\Users\lrl13\Downloads\S41718\S17\EEG\S17_260310_EEG_CV_2_ME.npy"
    #EEG_FILE = r"C:\Users\lrl13\Downloads\S41718\S17\EEG\S17_260310_EEG_CV_4_ME.npy"
    #EEG_FILE = r"C:\Users\lrl13\Downloads\S41718\S17\EEG\S17_260310_EEG_OK_3_ME.npy"
    #EEG_FILE = r"C:\Users\lrl13\Downloads\S41718\S17\EEG\S17_260310_EEG_OK_4_ME.npy"
    #EEG_FILE = r"C:\Users\lrl13\Downloads\S41718\S17\EEG\S17_260310_EEG_OV_1_ME.npy"
    #EEG_FILE = r"C:\Users\lrl13\Downloads\S41718\S17\EEG\S17_260310_EEG_OV_4_ME.npy"

    #EMG_FILE = r"C:\Users\lrl13\Downloads\S41718\S17\EMG\S17_260310_EMG_CK_1_ME.npy"
    #EMG_FILE = r"C:\Users\lrl13\Downloads\S41718\S17\EMG\S17_260310_EMG_CK_2_ME.npy"
    # "C:\Users\lrl13\Downloads\S41718\S17\EMG\S17_260310_EMG_CV_1_ME.npy"
    #EMG_FILE = r"C:\Users\lrl13\Downloads\S41718\S17\EMG\S17_260310_EMG_CV_2_ME.npy"
    #"C:\Users\lrl13\Downloads\S41718\S17\EMG\S17_260310_EMG_OK_1_ME.npy"
    #EMG_FILE = r"C:\Users\lrl13\Downloads\S41718\S17\EMG\S17_260310_EMG_OK_2_ME.npy"
    #EMG_FILE = r"C:\Users\lrl13\Downloads\S41718\S17\EMG\S17_260310_EMG_OV_1_ME.npy"
    #EMG_FILE = r"C:\Users\lrl13\Downloads\S41718\S17\EMG\S17_260310_EMG_OV_2_ME.npy"

    #EEG_FILE = r"C:\Users\lrl13\Downloads\DataCMC\S5\EEG\S5_260323_EEG_OV_1_ME.npy"
    #EEG_FILE = r"C:\Users\lrl13\Downloads\DataCMC\S5\EEG\S5_260323_EEG_OK_3_ME.npy"
    #EEG_FILE = r"C:\Users\lrl13\Downloads\DataCMC\S5\EEG\S5_260323_EEG_CV_3_ME.npy"
    EEG_FILE = r"C:\Users\lrl13\Downloads\DataCMC\S5\EEG\S5_260323_EEG_CK_2_ME.npy"
    
    EMG_FILE = r"C:\Users\lrl13\Downloads\DataCMC\S5\EMG\S5_260323_EMG_CK_1_ME.npy"
    #EMG_FILE = r"C:\Users\lrl13\Downloads\DataCMC\S5\EMG\S5_260323_EMG_CV_1_ME.npy"
    #EMG_FILE = r"C:\Users\lrl13\Downloads\DataCMC\S5\EMG\S5_260323_EMG_OK_1_ME.npy"
    #EMG_FILE = r"C:\Users\lrl13\Downloads\DataCMC\S5\EMG\S5_260323_EMG_OV_1_ME.npy"

    EMG_NAMES = ["Right Brachioradialis","Right Extensor carpi radialis longus", "Right palmaris longus","Right extensor carpi ulnaris"]
    #EMG_NAMES = ["Right palmaris longus"]
    EEG_NAMES = ["Fz","FC3","FCz","FC4","Cz","C3","C4"]
    results = calcula_todo(
        RUTA_EMG     = EMG_FILE,
        RUTA_EEG     = EEG_FILE,
        emg_names = EMG_NAMES,
        eeg_names = EEG_NAMES,

        t_start      = None,          # None = solapamiento maximo
        t_end        = None,
    )

    # print("\nPipeline CMC completado.")
    # print(f"  Frecuencias: {results['freqs'][0]:.2f} - {results['freqs'][-1]:.2f} Hz")
    # print(f"  Coherencia maxima: {results['coherencia'].max():.4f}")