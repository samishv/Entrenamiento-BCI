import numpy as np
import matplotlib.pyplot as plt
import pywt
import mne
import os

from scipy.signal import iirnotch, sosfiltfilt, tf2sos
from mne.preprocessing import ICA
from scipy.signal import welch

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────
plt.close("all")

RUTA_NPY = r"D:\luiso\Documentos\Luis\UPIITA\TT\Entrenamiento-BCI\DataCMC\Registros CMC 2\S5\EEG\S5_260429_EEG_OK_1_ME.npy"

FS = 250
Q_NOTCH = 30
BUTT_ORDER = 5
BUTT_FC_LP = 50.0

PAD = 5 * FS

CANALES = ['Fz', 'FC3', 'FCz', 'FC4', 'Cz', 'C3', 'C4']

VENTANA_SEG = 1.0
NPERSEG = int(VENTANA_SEG * FS)

LEVEL_WAVE = 6
WAVELET_MOM = 'db4'
NIVELES_CERO = [1, 6]   # quitar D1 y D6
REMOVE_APPROX = True    # quitar A6 si quieres
WAVE_MODE = 'reflect'

BANDAS = {
    'δ' : (0.5, 4,  '#9B3461'),
    'θ' : (4,   8,  '#C15579'),
    'μ' : (8,   13, '#E77692'),
    'β' : (13,  30, '#f0B8AB'),
    'γ' : (30,  45, '#FAFAC4'),
}

COLORES =     ['#6FC7B7', '#A1DAD1', '#D0EDE8', '#FFFFFF', '#BFEEF6', '#81DCEE', '#41CCE3']
COLORES_ICA = ['#FAF5CB', '#EEF7CE', '#D8F6D3', '#CCF5E6', '#C1F4F4', '#BAECF7', '#B2E8FB']

BG_FIG      = '#0a0a0a'
BG_AX       = '#0d0d0d'
COLOR_GRID  = '#404040'
COLOR_TICK  = '#a5a5a5'
COLOR_LABEL = '#d1d1d1'
COLOR_TITLE = '#ffffff'

PSD_XLIM = (0, 45 + 8)
F_LO, F_HI = 5.0, 50.0

IC_TO_REMOVE = 0

T0 = 0
DUR = 15

# ─────────────────────────────────────────────────────────────
# FUNCIONES FILTRADO
# ─────────────────────────────────────────────────────────────
def filtrar_con_padding_reflect(x, sos, pad):
    if pad is None or pad <= 0:
        return sosfiltfilt(sos, x)
    x_pad = np.pad(x, pad_width=pad, mode="reflect")
    y_pad = sosfiltfilt(sos, x_pad)
    return y_pad[pad:-pad]

def aplicar_notch_multiple(datos, fs, freqs, Q=30, pad=0):
    salida = datos.copy()
    for f0 in freqs:
        b, a = iirnotch(w0=f0, Q=Q, fs=fs)
        sos = tf2sos(b, a)
        salida = np.array([filtrar_con_padding_reflect(c, sos, pad) for c in salida])
    print(f"\n✅ Notch aplicado en {freqs} Hz (Q={Q})")
    return salida

def calcular_psd(senal, fs, nperseg):
    freqs, psd = welch(senal, fs=fs, nperseg=nperseg)
    return freqs, psd

def filtrar_dwt(signal, wavelet, level, niveles_cero=None, remove_approx=False, mode='symmetric'):
    coeffs = pywt.wavedec(signal, wavelet, level=level, mode=mode)
    coeffs_filt = [c.copy() for c in coeffs]

    if remove_approx:
        coeffs_filt[0] = np.zeros_like(coeffs_filt[0])

    if niveles_cero is not None:
        for j in niveles_cero:
            if 1 <= j <= level:
                coeffs_filt[j] = np.zeros_like(coeffs_filt[j])

    y = pywt.waverec(coeffs_filt, wavelet, mode=mode)
    return y[:len(signal)]

def filtrar_dwt_con_padding_reflect(x, wavelet, level, niveles_cero=None, remove_approx=False, pad=0, mode='symmetric'):
    if pad is None or pad <= 0:
        return filtrar_dwt(
            x,
            wavelet=wavelet,
            level=level,
            niveles_cero=niveles_cero,
            remove_approx=remove_approx,
            mode=mode)

    x_pad = np.pad(x, pad_width=pad, mode='reflect')

    y_pad = filtrar_dwt(
        x_pad,
        wavelet=wavelet,
        level=level,
        niveles_cero=niveles_cero,
        remove_approx=remove_approx,
        mode=mode)

    return y_pad[pad:-pad]

# ─────────────────────────────────────────────────────────────
# FUNCIONES DE PLOTEO
# ─────────────────────────────────────────────────────────────
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
                    nombre, color=bc, fontsize=6.5,
                    ha='center', va='bottom')

def plot_multicanal(data_2d, fs, labels, title,
                    t0=0.0, dur=10.0,
                    same_ylim=True,
                    nperseg=NPERSEG,
                    psd_xlim=PSD_XLIM,
                    colores=COLORES,
                    bandas=True,
                    f_low=None,
                    f_high=None):

    i0 = int(t0 * fs)
    i1 = int((t0 + dur) * fs)
    seg = data_2d[:, i0:i1]
    tiempo = np.arange(seg.shape[1]) / fs + t0
    n_ch = seg.shape[0]

    if colores is None:
        colores = ['#6FC7B7'] * n_ch
    if len(colores) < n_ch:
        colores = (colores * (n_ch // len(colores) + 1))[:n_ch]

    fig, axes = plt.subplots(
        n_ch, 2,
        sharex='col',
        sharey='col',
        gridspec_kw={'width_ratios': [3, 1], 'hspace': 0.15, 'wspace': 0.08}
    )

    fig.patch.set_facecolor(BG_FIG)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.07)
    fig.suptitle(title, color=COLOR_TITLE, fontsize=11, fontweight='bold')

    if n_ch == 1:
        axes = np.array([axes])

    if same_ylim:
        y_min_fig = seg.min()
        y_max_fig = seg.max()
        margen_fig = (y_max_fig - y_min_fig) * 0.08
        if margen_fig == 0:
            margen_fig = 1.0

    psd_max = 0
    psds = []
    for i in range(n_ch):
        freqs, psd = calcular_psd(seg[i], fs, nperseg)
        psds.append((freqs, psd))
        if psd.size:
            psd_max = max(psd_max, float(psd.max()))

    if f_low is None:
        f_low = psd_xlim[0]
    if f_high is None:
        f_high = psd_xlim[1]

    for i in range(n_ch):
        ax_s, ax_p = axes[i, 0], axes[i, 1]
        color = colores[i]
        canal = labels[i] if labels is not None else f"Ch {i}"
        freqs, psd = psds[i]

        estilo_ax(ax_s)
        ax_s.plot(tiempo, seg[i], color=color, linewidth=0.75, alpha=0.95)
        ax_s.set_xlim(tiempo[0], tiempo[-1])
        if same_ylim:
            ax_s.set_ylim(y_min_fig - margen_fig, y_max_fig + margen_fig)
        ax_s.set_ylabel(canal, color=color, fontsize=9,
                        rotation=0, labelpad=32, va='center')
        ax_s.grid(True, color=COLOR_GRID, linewidth=0.5, linestyle='--')
        if i < n_ch - 1:
            ax_s.tick_params(labelbottom=False)
        else:
            ax_s.set_xlabel('Tiempo (s)', color=COLOR_LABEL, fontsize=9)

        estilo_ax(ax_p)
        ax_p.plot(freqs, psd, color=color, alpha=0.85, linewidth=1.5)
        if bandas:
            plot_bandas_psd(ax_p, psd_max, f_low, f_high, etiquetas=(i == 0))
        ax_p.set_xlim(psd_xlim)
        ax_p.set_ylim(0, psd_max * 1.15 if psd_max > 0 else 1)
        ax_p.grid(True, color=COLOR_GRID, linewidth=0.5, linestyle='--', axis='y')
        ax_p.yaxis.get_offset_text().set_color(COLOR_TICK)
        ax_p.yaxis.get_offset_text().set_fontsize(6)

        if i < n_ch - 1:
            ax_p.tick_params(labelbottom=False)
        else:
            ax_p.set_xlabel('Frecuencia (Hz)', color=COLOR_LABEL, fontsize=9)

        if i == 0:
            ax_p.set_title('PSD  (µV²/Hz)', color=COLOR_LABEL, fontsize=7.5, pad=3)

    fig.set_size_inches(19, 10.5)
    pantalla_completa(fig)
    plt.show()
    return fig

# ─────────────────────────────────────────────────────────────
# 1) CARGAR DATOS EEG CRUDOS
# ─────────────────────────────────────────────────────────────
datos = np.load(RUTA_NPY)

if not (datos.ndim == 2 and datos.shape[0] == 7):
    raise ValueError(f"Shape inesperado: {datos.shape} (esperado: (7, n_muestras))")

# ─────────────────────────────────────────────────────────────
# 2) FILTROS, NOTCH Y PASA BANDA
# ─────────────────────────────────────────────────────────────
datos_f = aplicar_notch_multiple(datos, FS, freqs=[60, 120], Q=Q_NOTCH, pad=PAD)
# datos_f = aplicar_butter_pasabajas(datos_f, FS, fc=BUTT_FC_LP, order=BUTT_ORDER, pad=PAD)
datos_f = np.array([
    filtrar_dwt_con_padding_reflect(canal, wavelet=WAVELET_MOM, level=LEVEL_WAVE,
                                niveles_cero=NIVELES_CERO, remove_approx=REMOVE_APPROX,
                                pad=PAD, mode=WAVE_MODE)
    for canal in datos_f
])
print(f"✅ Wavelet DWT aplicado: wavelet={WAVELET_MOM}, level={LEVEL_WAVE}, remove_approx={REMOVE_APPROX}, pad={PAD}")

plot_multicanal(
    datos_f, FS, CANALES,
    "SEÑAL INICIAL (filtrada) — Notch 60/120 + LP Butter 50 Hz + Wavelet DWT",
    t0=T0, dur=DUR, same_ylim=True,
    nperseg=NPERSEG,
    psd_xlim=PSD_XLIM,
    colores=COLORES,
    bandas=True,
    f_low=F_LO,
    f_high=F_HI)

# ─────────────────────────────────────────────────────────────
# 3) CREAR RAW MNE
# ─────────────────────────────────────────────────────────────
info = mne.create_info(ch_names=CANALES, sfreq=FS, ch_types=['eeg'] * len(CANALES))
raw = mne.io.RawArray(datos_f, info, verbose=False)

montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage, match_case=False)

# ─────────────────────────────────────────────────────────────
# 4) SACAR TOPOGRAMAS Y COMPONENTES INDEPENDIENTES (IC) CON ICA
# ─────────────────────────────────────────────────────────────
rank = mne.compute_rank(raw, verbose=False)
n_comp = min(rank.get('eeg', len(CANALES)), len(CANALES))
print(f"🔎 Rank EEG estimado: {rank.get('eeg', None)} → usando n_components={n_comp}\n")

ica = ICA(
    n_components=n_comp,
    method='infomax',
    fit_params=dict(extended=True),
    random_state=97,
    max_iter='auto'
)
ica.fit(raw)
print("\n✅ ICA ajustada")

# === TOPOGRAMAS ===
ica.plot_components(
    extrapolate='local',
    sensors=True,
    contours=0
)

# === COMPONENTES INDEPENDIENTES ===
sources_raw = ica.get_sources(raw) 
sources = sources_raw.get_data()
ic_labels = [f"IC{i}" for i in range(sources.shape[0])]

plot_multicanal(
    sources, FS, ic_labels,
    "SEÑALES DE LOS ICs (activaciones ICA)",
    t0=T0, dur=DUR, same_ylim=False,
    nperseg=NPERSEG,
    psd_xlim=PSD_XLIM,
    colores=COLORES_ICA,
    bandas=True,
    f_low=F_LO,
    f_high=F_HI)

# ─────────────────────────────────────────────────────────────
# 5) RECONSTRUCCIÓN: excluir IC indicado por variable y aplicar
# ─────────────────────────────────────────────────────────────
if IC_TO_REMOVE is not None:
    if isinstance(IC_TO_REMOVE, int):
        exclude_list = [IC_TO_REMOVE]
    else:
        exclude_list = list(IC_TO_REMOVE)

    ica2 = ica.copy()
    ica2.exclude = exclude_list
    print(f"🚫 Reconstruyendo excluyendo: {ica2.exclude}\n")

    raw_recon = raw.copy()
    ica2.apply(raw_recon)

    plot_multicanal(
        raw_recon.get_data(), FS, CANALES,
        f"SEÑAL RECONSTRUIDA (excluyendo {ica2.exclude})",
        t0=T0, dur=DUR, same_ylim=True
    )
else:
    print("ℹ️ IC_TO_REMOVE = None → no se reconstruye (solo visualización).\n")

plt.show()

# ─────────────────────────────────────────────────────────────
# 6) RECONSTRUIR SEÑAL
# ─────────────────────────────────────────────────────────────

ica.exclude = exclude_list
raw_reconstruido = raw.copy()
ica.apply(raw_reconstruido)

datos_reconstruidos = raw_reconstruido.get_data()

print(f"✅ Señal reconstruida con shape: {datos_reconstruidos.shape}")

# ─────────────────────────────────────────────────────────────
# 7) GUARDAR SEÑAL RECONSTRUIDA EN .NPY
# ─────────────────────────────────────────────────────────────

CARPETA_SALIDA = r"D:\luiso\Documentos\Luis\UPIITA\TT\CMC\ICAS"

os.makedirs(CARPETA_SALIDA, exist_ok=True)
nombre_archivo = os.path.basename(RUTA_NPY)
ruta_salida = os.path.join(CARPETA_SALIDA, nombre_archivo)

np.save(ruta_salida, datos_reconstruidos)

print(f"💾 Archivo guardado correctamente en:\n{ruta_salida}")