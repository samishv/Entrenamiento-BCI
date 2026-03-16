"""
Pipeline de Coherencia Corticomuscular (CMC)
=============================================
Pasos:
  1. Cargar señales (EMG .emt  +  EEG .csv)
  2. Sincronizar EEG y EMG
  3. Filtrar señales
  4. Rectificar EMG
  5. Igualar frecuencia de muestreo (resample EMG → fs_eeg)
  6. Segmentar señales (ventanas de Hann con solapamiento)
  7. Calcular PSD y espectro cruzado  (Welch)
  8. Calcular coherencia
  9. Promediar o analizar bandas de interés

Dependencias: numpy, scipy, matplotlib, pandas
"""

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path


# ─────────────────────────────────────────────
# PARÁMETROS GLOBALES (ajusta según tus datos)
# ─────────────────────────────────────────────
FS_EMG   = 1000          # Hz  – frecuencia de muestreo del EMG
FS_EEG   = 250           # Hz  – frecuencia de muestreo del EEG
FS_FINAL = FS_EMG        # Hz  – fs común tras resample

# Filtros
EMG_BAND  = (20.0, 450.0)   # Hz – pasa-banda para el EMG
EEG_BAND  = (0.10,  45.0)   # Hz – pasa-banda para el EEG
NOTCH_F   = 60.0             # Hz – notch (red eléctrica); None para omitir

# Segmentación (Welch)
SEG_DUR   = 1.0     # s   – duración de cada ventana
OVERLAP   = 0.5     # fracción de solapamiento

# Bandas de frecuencia para análisis (Hz)
BANDS = {
    "delta":  (1,   4),
    "theta":  (4,   8),
    "alpha":  (8,  13),
    "beta":   (13, 30),
    "gamma":  (30, 80),
}

# Resample: up/down para pasar de FS_EMG a FS_FINAL
from math import gcd
_g   = gcd(int(FS_FINAL), int(FS_EMG))
UP   = int(FS_FINAL) // _g
DOWN = int(FS_EMG)   // _g


# ══════════════════════════════════════════════
# 1. CARGA DE SEÑALES
# ══════════════════════════════════════════════

def load_emg(filepath, channel_names=None, fs=FS_EMG):
    """
    Carga EMG desde:
      - .npy : array (n_canales, n_muestras)  [formato nuevo]
      - .emt : archivo de texto BTS freeEMG   [formato original]
 
    Devuelve (time, emg, channel_names, fs)
    emg shape: (n_muestras, n_canales)
    """
    filepath = Path(filepath)
 
    if filepath.suffix.lower() == '.npy':
        data = np.load(filepath)                  # (n_ch, n_samples)
        emg  = data.T                             # → (n_samples, n_ch)
        n_ch = emg.shape[1]
        if channel_names is None:
            channel_names = [f"EMG_{i}" for i in range(n_ch)]
        time = np.arange(emg.shape[0]) / fs
        print(f"[EMG] Archivo : {filepath.name}")
        print(f"      Canales : {channel_names}")
        print(f"      Muestras: {len(time)}  |  fs = {fs} Hz")
        return time, emg, channel_names, float(fs)
 


def load_eeg(filepath, channel_names = None, fs = FS_EEG, 
             eeg_channels = None, cnt_col = "CNT",
             timestamp_col = "Timestamp", valid_col = "VALID"):
    
    filepath = Path(filepath)

    data = np.load(filepath)                  # (n_ch, n_samples)
    eeg  = data.T                             # → (n_samples, n_ch)
    n_ch = eeg.shape[1]
    if channel_names is None:
        channel_names = [f"EEG_{i}" for i in range(n_ch)]
    time = np.arange(eeg.shape[0]) / fs
    print(f"[EEG] Archivo : {filepath.name}")
    print(f"      Canales : {channel_names}")
    print(f"      Muestras: {len(time)}  |  fs = {fs} Hz")
    print(f"      Duracion: {time[-1]:.2f} s")
    return time, eeg, channel_names, float(fs)


# ══════════════════════════════════════════════
# 2. SINCRONIZACIÓN
# ══════════════════════════════════════════════

def synchronize(time_emg: np.ndarray, emg: np.ndarray,
                time_eeg: np.ndarray, eeg: np.ndarray,
                t_start: float | None = None,
                t_end:   float | None = None
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Recorta ambas señales al intervalo [t_start, t_end].
    Si t_start / t_end son None usa el máximo solapamiento.
    """
    t0 = max(time_emg[0], time_eeg[0]) if t_start is None else t_start
    t1 = min(time_emg[-1], time_eeg[-1]) if t_end   is None else t_end

    mask_emg = (time_emg >= t0) & (time_emg <= t1)
    mask_eeg = (time_eeg >= t0) & (time_eeg <= t1)

    print(f"[SYNC] Ventana de análisis: {t0:.3f} s – {t1:.3f} s  "
          f"(EMG: {mask_emg.sum()} pts, EEG: {mask_eeg.sum()} pts)")

    return (time_emg[mask_emg], emg[mask_emg],
            time_eeg[mask_eeg], eeg[mask_eeg])


# ══════════════════════════════════════════════
# 3. FILTRADO
# ══════════════════════════════════════════════

def _butter_bandpass(lowcut: float, highcut: float, fs: float,
                     order: int = 4) -> tuple:
    nyq = fs / 2.0
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return b, a


def _notch(f0: float, fs: float, Q: float = 35) -> tuple:
    b, a = signal.iirnotch(f0, Q, fs)
    return b, a


def filter_emg(emg: np.ndarray, fs: float = FS_EMG) -> np.ndarray:
    b, a = _butter_bandpass(*EMG_BAND, fs)
    out  = signal.filtfilt(b, a, emg, axis=0)
    if NOTCH_F:
        bn, an = _notch(NOTCH_F, fs)
        out    = signal.filtfilt(bn, an, out, axis=0)
    return out


def filter_eeg(eeg: np.ndarray, fs: float = FS_EEG) -> np.ndarray:
    b, a = _butter_bandpass(*EEG_BAND, fs)
    out  = signal.filtfilt(b, a, eeg, axis=0)
    if NOTCH_F:
        bn, an = _notch(NOTCH_F, fs)
        out    = signal.filtfilt(bn, an, out, axis=0)
    return out


# ══════════════════════════════════════════════
# 4. RECTIFICACIÓN EMG
# ══════════════════════════════════════════════

def rectify_emg(emg: np.ndarray) -> np.ndarray:
    """Rectificación de onda completa."""
    return np.abs(emg)


# ══════════════════════════════════════════════
# 5. RESAMPLE EMG → FS_FINAL
# ══════════════════════════════════════════════

def resample_emg(emg: np.ndarray, fs_in: float = FS_EMG,
                 fs_out: float = FS_FINAL) -> np.ndarray:
    """
    Usa resample_poly para preservar la forma de onda.
    Recalcula UP/DOWN si fs_in difiere del global.
    """
    g   = gcd(int(fs_out), int(fs_in))
    up  = int(fs_out) // g
    dn  = int(fs_in)  // g
    out = signal.resample_poly(emg, up, dn, axis=0)
    print(f"[RESAMPLE] EMG  {len(emg)} → {len(out)} muestras  "
          f"({fs_in} Hz → {fs_out} Hz,  up={up}, down={dn})")
    return out


# ══════════════════════════════════════════════
# 6. SEGMENTACIÓN (ventanas de Hann)
# ══════════════════════════════════════════════

def make_segments(emg_rs: np.ndarray, eeg: np.ndarray,
                  fs: float = FS_FINAL,
                  seg_dur: float = SEG_DUR,
                  overlap: float = OVERLAP
                  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Divide ambas señales en segmentos de longitud seg_dur (s)
    con solapamiento overlap.
    Devuelve arrays 3-D: (n_segs, seg_len, n_channels).
    """
    seg_len  = int(seg_dur * fs)
    step     = int(seg_len * (1 - overlap))
    n_emg_ch = emg_rs.shape[1] if emg_rs.ndim > 1 else 1
    n_eeg_ch = eeg.shape[1]    if eeg.ndim   > 1 else 1

    emg_rs = emg_rs.reshape(-1, n_emg_ch)
    eeg    = eeg.reshape(-1, n_eeg_ch)

    n_samples = min(len(emg_rs), len(eeg))
    starts    = np.arange(0, n_samples - seg_len + 1, step)

    emg_segs = np.stack([emg_rs[s:s + seg_len] for s in starts])
    eeg_segs = np.stack([eeg[s:s + seg_len]    for s in starts])

    print(f"[SEGM] {len(starts)} ventanas  |  {seg_len} pts/ventana  "
          f"|  solapamiento {overlap*100:.0f}%")
    return emg_segs, eeg_segs          # (n_segs, seg_len, n_ch)


# ══════════════════════════════════════════════
# 7. PSD Y ESPECTRO CRUZADO
# ══════════════════════════════════════════════

def compute_spectra(emg_segs: np.ndarray, eeg_segs: np.ndarray,
                    fs: float = FS_FINAL
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula PSD (Welch por segmentos) y espectro cruzado promediados.

    Devuelve:
        freqs       (n_freqs,)
        Sxx_emg     (n_freqs, n_emg_ch)   – PSD EMG
        Syy_eeg     (n_freqs, n_eeg_ch)   – PSD EEG
        Sxy         (n_freqs, n_emg_ch, n_eeg_ch) – espectro cruzado complejo
    """
    n_segs, seg_len, n_emg_ch = emg_segs.shape
    n_eeg_ch                  = eeg_segs.shape[2]

    win = signal.windows.hann(seg_len)

    # Preasigna acumuladores
    half       = seg_len // 2 + 1
    Sxx_acc    = np.zeros((half, n_emg_ch))
    Syy_acc    = np.zeros((half, n_eeg_ch))
    Sxy_acc    = np.zeros((half, n_emg_ch, n_eeg_ch), dtype=complex)

    scale = 2.0 / (fs * np.sum(win ** 2))   # normalización de densidad de potencia

    for seg in range(n_segs):
        for ei in range(n_emg_ch):
            X = np.fft.rfft(emg_segs[seg, :, ei] * win)
            Sxx_acc[:, ei] += np.abs(X) ** 2

            for yi in range(n_eeg_ch):
                if ei == 0:               # calcular Y sólo una vez por canal EEG
                    Y = np.fft.rfft(eeg_segs[seg, :, yi] * win)
                    Syy_acc[:, yi] += np.abs(Y) ** 2
                else:
                    Y = np.fft.rfft(eeg_segs[seg, :, yi] * win)

                Sxy_acc[:, ei, yi] += X * np.conj(Y)

    freqs    = np.fft.rfftfreq(seg_len, d=1.0 / fs)
    Sxx_mean = Sxx_acc * scale / n_segs
    Syy_mean = Syy_acc * scale / n_segs
    Sxy_mean = Sxy_acc * scale / n_segs

    return freqs, Sxx_mean, Syy_mean, Sxy_mean


# ══════════════════════════════════════════════
# 8. COHERENCIA
# ══════════════════════════════════════════════

def compute_coherence(Sxx: np.ndarray, Syy: np.ndarray,
                      Sxy: np.ndarray) -> np.ndarray:
    """
    Coh²(f) = |Sxy(f)|² / (Sxx(f) · Syy(f))
    Devuelve array (n_freqs, n_emg_ch, n_eeg_ch) con valores en [0, 1].
    """
    numerator   = np.abs(Sxy) ** 2               # (f, emg, eeg)
    denominator = (Sxx[:, :, None] *
                   Syy[:, None, :])               # broadcasting
    coherence   = numerator / (denominator + 1e-30)
    coherence   = np.clip(coherence, 0, 1)
    return coherence


# ══════════════════════════════════════════════
# 9. ANÁLISIS POR BANDAS
# ══════════════════════════════════════════════

def band_analysis(freqs: np.ndarray, coherence: np.ndarray,
                  emg_ch_names: list[str], eeg_ch_names: list[str]) -> pd.DataFrame:
    """
    Devuelve un DataFrame con la coherencia promedio por banda de frecuencia
    para cada par (canal EMG, canal EEG).
    """
    rows = []
    for band_name, (f_lo, f_hi) in BANDS.items():
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        for ei, ename in enumerate(emg_ch_names):
            for yi, yname in enumerate(eeg_ch_names):
                mean_coh = coherence[mask, ei, yi].mean()
                rows.append({
                    "banda":    band_name,
                    "emg_ch":  ename,
                    "eeg_ch":  yname,
                    "coh_mean": mean_coh,
                    "f_lo":    f_lo,
                    "f_hi":    f_hi,
                })
    df = pd.DataFrame(rows)
    print("\n[BANDAS] Coherencia promedio por banda:")
    print(df.to_string(index=False))
    return df


# ══════════════════════════════════════════════
# VISUALIZACIÓN
# ══════════════════════════════════════════════

def plot_signals(time_emg, emg_raw, emg_filt, emg_rect,
                 time_eeg, eeg_raw, eeg_filt,
                 emg_names, eeg_names):
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=False)

    axes[0, 0].plot(time_emg, emg_raw[:, 0], lw=0.5, color="steelblue")
    axes[0, 0].set_title(f"EMG crudo – {emg_names[0]}")
    axes[0, 0].set_ylabel("μV")

    axes[1, 0].plot(time_emg, emg_filt[:, 0], lw=0.5, color="darkorange")
    axes[1, 0].set_title("EMG filtrado")
    axes[1, 0].set_ylabel("μV")

    axes[2, 0].plot(time_emg, emg_rect[:, 0], lw=0.5, color="seagreen")
    axes[2, 0].set_title("EMG rectificado")
    axes[2, 0].set_ylabel("μV")
    axes[2, 0].set_xlabel("Tiempo (s)")

    axes[0, 1].plot(time_eeg, eeg_raw[:, 0], lw=0.5, color="steelblue")
    axes[0, 1].set_title(f"EEG crudo – {eeg_names[0]}")
    axes[0, 1].set_ylabel("μV")

    axes[1, 1].plot(time_eeg, eeg_filt[:, 0], lw=0.5, color="darkorange")
    axes[1, 1].set_title("EEG filtrado")
    axes[1, 1].set_ylabel("μV")

    axes[2, 1].axis("off")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    plt.suptitle("Señales EMG y EEG", fontsize=13)
    plt.tight_layout()
    plt.savefig("signals.png", dpi=150)
    plt.show()


def plot_psd(freqs, Sxx, Syy, emg_names, eeg_names, fmax=200):
    mask = freqs <= fmax
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for i, name in enumerate(emg_names):
        axes[0].semilogy(freqs[mask], Sxx[mask, i], label=name)
    axes[0].set_title("PSD – EMG (rectificado, resampleado)")
    axes[0].set_xlabel("Frecuencia (Hz)")
    axes[0].set_ylabel("PSD  [(μV)²/Hz]")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, which="both", alpha=0.3)

    for i, name in enumerate(eeg_names):
        axes[1].semilogy(freqs[mask], Syy[mask, i], label=name)
    axes[1].set_title("PSD – EEG")
    axes[1].set_xlabel("Frecuencia (Hz)")
    axes[1].set_ylabel("PSD  [(μV)²/Hz]")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, which="both", alpha=0.3)

    plt.suptitle("Densidades espectrales de potencia", fontsize=13)
    plt.tight_layout()
    plt.savefig("psd.png", dpi=150)
    plt.show()


def plot_coherence(freqs, coherence, emg_names, eeg_names, fmax=100):
    """Muestra la coherencia para cada par EMG-EEG hasta fmax Hz."""
    mask    = freqs <= fmax
    n_emg   = len(emg_names)
    n_eeg   = len(eeg_names)
    n_pairs = n_emg * n_eeg

    fig, axes = plt.subplots(n_pairs, 1,
                             figsize=(12, 3 * n_pairs), squeeze=False)
    ax_idx = 0

    for ei, ename in enumerate(emg_names):
        for yi, yname in enumerate(eeg_names):
            ax = axes[ax_idx, 0]
            ax.plot(freqs[mask], coherence[mask, ei, yi],
                    color="royalblue", lw=1.2)

            # Sombreado de bandas
            colors_band = ["#FFF9C4", "#C8E6C9", "#BBDEFB",
                           "#F8BBD0", "#E1BEE7"]
            for (bname, (f0, f1)), col in zip(BANDS.items(), colors_band):
                ax.axvspan(f0, min(f1, fmax), alpha=0.25,
                           color=col, label=bname)

            ax.set_title(f"CMC – EMG: {ename}  /  EEG: {yname}", fontsize=10)
            ax.set_ylabel("Coherencia²")
            ax.set_xlabel("Frecuencia (Hz)")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            if ax_idx == 0:
                ax.legend(loc="upper right", fontsize=7, ncol=len(BANDS))
            ax_idx += 1

    plt.suptitle("Coherencia Corticomuscular (CMC)", fontsize=13)
    plt.tight_layout()
    plt.savefig("coherence.png", dpi=150)
    plt.show()


# ══════════════════════════════════════════════
# ══════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════

def run_cmc_pipeline(emg_path,
                     eeg_path,
                     emg_channels=None,
                     eeg_channels=None,
                     t_start=None,
                     t_end=None):
    """
    Ejecuta el pipeline completo de CMC.

    Parametros
    ----------
    emg_path    : ruta al archivo .emt
    eeg_path    : ruta al archivo .csv (formato: Timestamp, Fz, C3, Cz, C4, Pz, PO7, Oz, PO8, ...)
    emg_channels: lista de indices EMG a usar, p.ej. [0]. None = todos.
    eeg_channels: lista de nombres EEG, p.ej. ["C3","Cz","C4"]. None = autodetecta.
    t_start / t_end: ventana temporal (s); None = solapamiento maximo.

    Devuelve dict con: freqs, Sxx, Syy, Sxy, coherence, band_df
    """

    # 1. CARGA
    t_emg, emg_raw, emg_names, fs_emg_ = load_emg(emg_path)
    t_eeg, eeg_raw, eeg_names, _        = load_eeg(eeg_path, eeg_channels=eeg_channels)

    # Seleccion de canales EMG
    if emg_channels is not None:
        emg_raw   = emg_raw[:, emg_channels]
        emg_names = [emg_names[i] for i in emg_channels]

    # Garantiza 2-D
    emg_raw = np.atleast_2d(emg_raw.T).T
    eeg_raw = np.atleast_2d(eeg_raw.T).T

    # 2. SINCRONIZACION
    t_emg, emg_sync, t_eeg, eeg_sync = synchronize(
        t_emg, emg_raw, t_eeg, eeg_raw, t_start, t_end)

    # 3. FILTRADO
    emg_filt = filter_emg(emg_sync, fs=fs_emg_)
    eeg_filt = filter_eeg(eeg_sync, fs=FS_EEG)

    # 4. RECTIFICACION
    emg_rect = rectify_emg(emg_filt)

    # 5. RESAMPLE
    eeg_rs = resample_emg(eeg_filt, fs_in = FS_EEG, fs_out = FS_FINAL)
    #emg_rs = resample_emg(emg_rect, fs_in=fs_emg_, fs_out=FS_FINAL)

    # Iguala longitudes (puede haber +-1 muestra por redondeo)
    n = min(len(emg_rect), len(eeg_rs))
    emg_rect   = emg_rect[:n]
    eeg_rs = eeg_rs[:n]

    # 6. SEGMENTACION
    emg_segs, eeg_segs = make_segments(emg_rect, eeg_rs)

    # 7. PSD + ESPECTRO CRUZADO
    freqs, Sxx, Syy, Sxy = compute_spectra(emg_segs, eeg_segs)

    # 8. COHERENCIA
    coherence = compute_coherence(Sxx, Syy, Sxy)

    # 9. ANALISIS DE BANDAS
    band_df = band_analysis(freqs, coherence, emg_names, eeg_names)

    # VISUALIZACION
    plot_signals(t_emg, emg_sync, emg_filt, emg_rect,
                 t_eeg, eeg_sync, eeg_filt, emg_names, eeg_names)
    plot_psd(freqs, Sxx, Syy, emg_names, eeg_names)
    plot_coherence(freqs, coherence, emg_names, eeg_names)

    return {
        "freqs":     freqs,
        "Sxx":       Sxx,
        "Syy":       Syy,
        "Sxy":       Sxy,
        "coherence": coherence,
        "band_df":   band_df,
    }


# ══════════════════════════════════════════════
# PUNTO DE ENTRADA
# ══════════════════════════════════════════════
if __name__ == "__main__":

    #EMG_FILE = r"C:\Users\lrl13\Downloads\RegistrosCMC_pruebaleo\RegistrosCMC\PLEO_3\emg_signals_1.emt"
    #EEG_FILE = r"C:\Users\lrl13\Downloads\RegistrosCMC_pruebaleo\RegistrosCMC\PLEO_3\EEG_RAW_09_03_2026__13_48_51.csv"

    #EMG_FILE = r"C:\Users\lrl13\Downloads\S17\S17\EMG\S17_CV_EMG_260310_1.npy"
    #EEG_FILE = r"C:\Users\lrl13\Downloads\S17\S17\EEG\S17CV_EEG_260310_ME_2.npy"

    EMG_FILE = r"C:\Users\lrl13\Downloads\RegistrosCMC_pruebaleo\RegistrosCMC\PLEO_1\emg_signals_1.npy"
    EEG_FILE = r"C:\Users\lrl13\Downloads\RegistrosCMC_pruebaleo\RegistrosCMC\PLEO_1\EEG_RAW_09_03_2026__12_59_13.npy"
    results = run_cmc_pipeline(
        emg_path     = EMG_FILE,
        eeg_path     = EEG_FILE,
        emg_channels = [0],          # None = todos los canales EMG
        # Canales disponibles en el CSV: Fz, C3, Cz, C4, Pz, PO7, Oz, PO8
        # None = todos; o especifica lista, p.ej.: ["C3", "Cz", "C4"]
        eeg_channels = ["Fz","FC3", "FCz", "FC4"],
        t_start      = None,          # None = solapamiento maximo
        t_end        = None,
    )

    print("\nPipeline CMC completado.")
    print(f"  Frecuencias: {results['freqs'][0]:.2f} - {results['freqs'][-1]:.2f} Hz")
    print(f"  Coherencia maxima: {results['coherence'].max():.4f}")