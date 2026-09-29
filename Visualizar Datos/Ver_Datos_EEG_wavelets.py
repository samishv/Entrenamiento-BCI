import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pywt

from Procesamiento_Funciones import (
    cargar_npy_eeg,
    aplicar_notch_multiple,
    aplicar_pasabanda_butter,
    quitar_componente_dc
)

plt.close('all')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

RUTA_NPY = r"D:\luiso\Documentos\Luis\UPIITA\TT\Entrenamiento-BCI\DataCMC\Registros CMC 2\S5\EEG\S5_260429_EEG_OK_1_ME.npy"

FS = 250

SEGUNDO_INICIO = 0
SEGUNDO_FIN = 6

ESCALA_Y_COMPARTIDA = True

TICKAZOS_LABEL = 1.5

VENTANA_X = 11
VENTANA_Y = 10

CANALES = ['Fz', 'FC3', 'FCz', 'FC4', 'Cz', 'C3', 'C4']

# Parámetros Wavelet
LEVEL = 5
WAVELET_MOM = 'db4'

colores = [ '#204cd2', '#006ba6', '#0496ff', '#ffae00', '#d81159', '#8e34be' ]

BG_FIG      = '#ffffff'
BG_AX       = '#ffffff'
COLOR_GRID  = '#5a5a5a'
COLOR_TICK  = '#000000'
COLOR_LABEL = '#000000'
COLOR_TITLE = '#000000'

# BG_FIG      = '#000000'
# BG_AX       = '#000000'
# COLOR_GRID  = '#a5a5a5'
# COLOR_TICK  = '#ffffff'
# COLOR_LABEL = '#ffffff'
# COLOR_TITLE = '#ffffff'

SIZENAMES = 26
SIZELABELS = 22
ESPACIO_ENTRE = 0.2

FREQS_NOTCH = (60.0, 120.0)
Q_NOTCH = 5.0
F_LOW = 5.0
F_HIGH = 45.0
BP_ORDER = 8
PAD_MUESTRAS = 5 * FS


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────────────────────────────────────

def pantalla_completa(fig):

    try:
        manager = fig.canvas.manager

        try:
            manager.window.state('zoomed')
            return
        except:
            pass

        try:
            manager.window.showMaximized()
            return
        except:
            pass

        try:
            manager.frame.Maximize(True)
            return
        except:
            pass

        fig.set_size_inches(19, 10.5)

    except:
        fig.set_size_inches(19, 10.5)


def estilo_ax(ax):

    ax.set_facecolor(BG_AX)

    ax.tick_params(
        colors=COLOR_TICK,
        labelsize=SIZELABELS
    )

    ax.spines[['top', 'right']].set_visible(False)

    ax.spines['left'].set_color(COLOR_GRID)
    ax.spines['bottom'].set_color(COLOR_GRID)

    ax.grid(
        True,
        color=COLOR_GRID,
        linewidth=0.5,
        linestyle='--'
    )


def reconstruct_component(
        coeffs,
        index,
        wavelet,
        n_muestras):

    componentes = [
        np.zeros_like(c)
        for c in coeffs
    ]

    componentes[index] = coeffs[index]

    componente = pywt.waverec(
        componentes,
        wavelet
    )

    return componente[:n_muestras]


def freq_band(fs, j):

    f_low = fs / (2 ** (j + 1))
    f_high = fs / (2 ** j)

    return f_low, f_high


def plot_figura(
        componentes,
        nombres,
        tiempo,
        fs,
        colores):

    n_graficas = len(componentes)

    fig, axes = plt.subplots(
        n_graficas,
        1,
        sharex=True,
        gridspec_kw={
            'hspace': ESPACIO_ENTRE
        }
    )

    if n_graficas == 1:
        axes = [axes]

    fig.patch.set_facecolor(BG_FIG)

    fig.subplots_adjust(
        left=0.06,
        right=0.94,
        top=0.94,
        bottom=ESPACIO_ENTRE
    )

    fig.set_size_inches(
        VENTANA_X,
        VENTANA_Y
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # ESCALA Y
    # ─────────────────────────────────────────────────────────────────────────
    
    if ESCALA_Y_COMPARTIDA:
    
        y_min_global = min(
            componente.min()
            for componente in componentes
        )
    
        y_max_global = max(
            componente.max()
            for componente in componentes
        )
    
        margen_global = (
            y_max_global - y_min_global
        ) * 0.05

    # ─────────────────────────────────────────────────────────────────────────
    # GRAFICAR COMPONENTES
    # ─────────────────────────────────────────────────────────────────────────

    for i, ax in enumerate(axes):

        estilo_ax(ax)

        ax.plot(
            tiempo,
            componentes[i],
            color=colores[i],
            linewidth=2.0
        )

        # ax.set_title(
        #     nombres[i],
        #     color=COLOR_TITLE,
        #     fontsize=SIZENAMES,
        #     fontweight='bold'
        # )

        ax.set_ylabel(
            f'D{i+2}',
            color=COLOR_LABEL,
            fontsize=SIZENAMES,
            fontweight='bold'
        )

        ax.set_xlim(
            tiempo[0],
            tiempo[-1]
        )

        if ESCALA_Y_COMPARTIDA:

            ax.set_ylim(
                y_min_global - margen_global,
                y_max_global + margen_global
            )
        
        else:
        
            y_min = componentes[i].min()
            y_max = componentes[i].max()
        
            margen = (y_max - y_min) * 0.05
        
            ax.set_ylim(
                y_min - margen,
                y_max + margen
            )

        ax.xaxis.set_major_locator(
            ticker.MultipleLocator(
                TICKAZOS_LABEL
            )
        )
        
        if nombres[i].startswith('Aproximación A4'):
            ax.ticklabel_format(
                axis='y',
                style='scientific',
                scilimits=(0, 0)
            )

        if i < n_graficas - 1:
            plt.setp(
                ax.get_xticklabels(),
                visible=False
            )

    axes[-1].set_xlabel(
        'Tiempo (s)',
        color=COLOR_LABEL,
        fontsize=SIZENAMES,
        fontweight='bold'
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # GUARDAR IMAGEN
    # ─────────────────────────────────────────────────────────────────────────

    CARPETA_IMGS = r"D:\luiso\Descargas\prueba"
    os.makedirs(CARPETA_IMGS, exist_ok=True)
    
    nombre_archivo = os.path.splitext(
        os.path.basename(RUTA_NPY)
    )[0]

    ruta_img = os.path.join(
        CARPETA_IMGS,
        f"{nombre_archivo}_EEG_DESCOMP.png"
    )
    
    fig.savefig(
        ruta_img,
        dpi=300,
        bbox_inches='tight',
        facecolor=fig.get_facecolor()
    )
    
    print(f"\n[OK] Figura guardada en:\n{ruta_img}")

    pantalla_completa(fig)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CARGAR SEÑAL
# ─────────────────────────────────────────────────────────────────────────────

datos = cargar_npy_eeg(
    RUTA_NPY,
    n_canales_esperados=len(CANALES)
)

n_canales, n_muestras = datos.shape

if SEGUNDO_INICIO is not None and SEGUNDO_FIN is not None:

    muestra_inicio = int(SEGUNDO_INICIO * FS)
    muestra_fin = int(SEGUNDO_FIN * FS)

    if muestra_inicio < n_muestras and muestra_fin <= n_muestras:

        datos = datos[:, muestra_inicio:muestra_fin]

        n_canales, n_muestras = datos.shape

    else:

        print(
            f"[AVISO] El intervalo solicitado "
            f"({SEGUNDO_INICIO} - {SEGUNDO_FIN} s) "
            f"supera la duración de la señal "
            f"({n_muestras/FS:.2f} s). "
            f"Se procesará la señal completa."
        )

tiempo = np.arange(n_muestras) / FS

print(
    f"\n[OK] EEG cargado: "
    f"{n_canales} canales | "
    f"{n_muestras} muestras"
)

datos_sin_dc = quitar_componente_dc(datos)

print("\n[OK] Componente de corriente directa eliminada")

datos_notch = aplicar_notch_multiple(
    datos_sin_dc,
    fs=FS,
    freqs=FREQS_NOTCH,
    Q=Q_NOTCH,
    pad=PAD_MUESTRAS
)

print(
    f"[OK] Notch aplicado: "
    f"{FREQS_NOTCH} Hz y {FREQS_NOTCH} Hz | Q = {Q_NOTCH}"
)

datos_filtrados = aplicar_pasabanda_butter(
    datos_notch,
    fs=FS,
    f_low=F_LOW,
    f_high=F_HIGH,
    orden=BP_ORDER,
    pad=PAD_MUESTRAS
)

print(
    f"[OK] Butterworth aplicado: "
    f"{BP_ORDER}º orden | {F_LOW:.0f}-{F_HIGH:.0f} Hz"
)

signal = datos_filtrados[5]
N = len(signal)

# ─────────────────────────────────────────────────────────────────────────────
# DESCOMPOSICIÓN WAVELET
# ─────────────────────────────────────────────────────────────────────────────

coeffs = pywt.wavedec(
    signal,
    WAVELET_MOM,
    level=LEVEL
)

print(
    f"[OK] Descomposición Wavelet: "
    f"{WAVELET_MOM} | nivel {LEVEL}"
)


# ─────────────────────────────────────────────────────────────────────────────
# RECONSTRUCCIÓN DE COMPONENTES
# ─────────────────────────────────────────────────────────────────────────────

A5 = reconstruct_component(
    coeffs,
    0,
    WAVELET_MOM,
    N
)

D5 = reconstruct_component(
    coeffs,
    1,
    WAVELET_MOM,
    N
)

D4 = reconstruct_component(
    coeffs,
    2,
    WAVELET_MOM,
    N
)

D3 = reconstruct_component(
    coeffs,
    3,
    WAVELET_MOM,
    N
)

D2 = reconstruct_component(
    coeffs,
    4,
    WAVELET_MOM,
    N
)

D1 = reconstruct_component(
    coeffs,
    5,
    WAVELET_MOM,
    N
)

print(
    "[OK] Componentes reconstruidas"
)


# ─────────────────────────────────────────────────────────────────────────────
# BANDAS DE FRECUENCIA
# ─────────────────────────────────────────────────────────────────────────────

f_low_D1, f_high_D1 = freq_band(FS, 1)
f_low_D2, f_high_D2 = freq_band(FS, 2)
f_low_D3, f_high_D3 = freq_band(FS, 3)
f_low_D4, f_high_D4 = freq_band(FS, 4)
f_low_D5, f_high_D5 = freq_band(FS, 5)

fA5 = FS / (2 ** (LEVEL + 1))


# ─────────────────────────────────────────────────────────────────────────────
# TODOS LOS COMPONENTES
# ─────────────────────────────────────────────────────────────────────────────

componentes = [
    D1,
    D2,
    D3,
    D4,
    D5,
    A5
]

nombres = [
    f'Detalle D1 ({f_low_D1:.2f} - {f_high_D1:.2f} Hz)',
    f'Detalle D2 ({f_low_D2:.2f} - {f_high_D2:.2f} Hz)',
    f'Detalle D3 ({f_low_D3:.2f} - {f_high_D3:.2f} Hz)',
    f'Detalle D4 ({f_low_D4:.2f} - {f_high_D4:.2f} Hz)',
    f'Detalle D5 ({f_low_D5:.2f} - {f_high_D5:.2f} Hz)',
    f'Aproximación A5 (0 - {fA5:.2f} Hz)'
]

# ─────────────────────────────────────────────────────────────────────────────
# COMPONENTES SELECCIONADAS PARA GRAFICAR
# ─────────────────────────────────────────────────────────────────────────────

componentes_plot = [
    D2,
    D3,
    D4,
    D5,
]

nombres_plot = [
    f'Detalle 2 ({f_low_D2:.0f} ~ {f_high_D2:.0f} Hz)',
    f'Detalle 3 ({f_low_D3:.0f} ~ {f_high_D3:.0f} Hz)',
    f'Detalle 4 ({f_low_D4:.0f} ~ {f_high_D4:.0f} Hz)',
    f'Detalle 5 ({f_low_D5:.0f} ~ {f_high_D5:.0f} Hz)',
]

colores_plot = [
    colores[1],  # D2
    colores[2],  # D3
    colores[3],  # D4
    colores[4],  # D5
]


# ─────────────────────────────────────────────────────────────────────────────
# GRAFICAR
# ─────────────────────────────────────────────────────────────────────────────

plot_figura(
    componentes=componentes_plot,
    nombres=nombres_plot,
    tiempo=tiempo,
    fs=FS,
    colores=colores_plot
)

print("\n[FIN] Componentes Wavelet generadas.\n")