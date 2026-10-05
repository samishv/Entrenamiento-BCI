import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from Procesamiento_Funciones import (
    cargar_npy_emg,
    aplicar_notch_multiple,
    aplicar_pasabanda_butter,
    resamplear_profesional,
    entrenar_instar
)

plt.close('all')


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

RUTA_EMG = r"D:\luiso\Documentos\Luis\UPIITA\TT\Entrenamiento-BCI\DataCMC\Registros CMC 2\S17\EMG\S17_260421_EMG_OV_2_ME.npy"


# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS DE VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────

FS = 1000

SEGUNDO_INICIO = 1.5
SEGUNDO_FIN = 5.5

TICKAZOS_LABEL = 1.0

VENTANA_X = 9.5
VENTANA_Y = 10

PAD_MUESTRAS = 5 * FS


# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS DE INSTAR
# ─────────────────────────────────────────────────────────────────────────────

EPOCHS_INSTAR = 1000
ALPHA_INSTAR = 0.1
BIAS_INSTAR = 50


# ─────────────────────────────────────────────────────────────────────────────
# CANALES EMG
# ─────────────────────────────────────────────────────────────────────────────

MUSCULOS = [
    'FCU',
    'ECRL',
    'ECU'
]

MUSCULOS_PLOT = [
    'FCU',
    'ECRL',
    'ECU'
]

COLORES_EMG = [
    '#93D989',
    '#70BBD6',
    '#7094FF'
]


# ─────────────────────────────────────────────────────────────────────────────
# ESTILO
# ─────────────────────────────────────────────────────────────────────────────

BG_FIG      = '#ffffff'
BG_AX       = '#ffffff'
COLOR_GRID  = '#5a5a5a'
COLOR_TICK  = '#000000'
COLOR_LABEL = '#000000'
COLOR_TITLE = '#000000'

SIZENAMES = 36
SIZELABELS = 30

ESPACIO_ENTRE = 0.1


# ─────────────────────────────────────────────────────────────────────────────
# FILTROS
# ─────────────────────────────────────────────────────────────────────────────

FREQS_NOTCH = [
    19.6,
    60,
    120,
    180,
    240,
    300
]

Q_NOTCH = 200

F_LOW = 20.0
F_HIGH = 300.0
BP_ORDER = 4


# ─────────────────────────────────────────────────────────────────────────────
# PARSEO DEL NOMBRE DE ARCHIVO
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

    nombre = os.path.splitext(
        os.path.basename(ruta)
    )[0]

    partes = nombre.split('_')

    sujeto = partes[0]
    condicion = partes[3]
    estado = partes[5]

    num_sujeto = sujeto[1:]

    desc_cond = CONDICION_MAP.get(
        condicion,
        condicion
    )

    desc_estado = ESTADO_MAP.get(
        estado,
        estado
    )

    return (
        f"Sujeto {num_sujeto} | "
        f"{desc_cond} | "
        f"{desc_estado}"
    )


PIE_FIGURA = parsear_nombre_archivo(RUTA_EMG)


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

        fig.set_size_inches(
            19,
            10.5
        )

    except:

        fig.set_size_inches(
            19,
            10.5
        )


def estilo_ax(ax):

    ax.set_facecolor(BG_AX)

    ax.tick_params(
        colors=COLOR_TICK,
        labelsize=SIZELABELS
    )

    ax.spines[
        ['top', 'right']
    ].set_visible(False)

    ax.spines[
        'left'
    ].set_color(COLOR_GRID)

    ax.spines[
        'bottom'
    ].set_color(COLOR_GRID)

    ax.grid(
        True,
        color=COLOR_GRID,
        linewidth=0.5,
        linestyle='--'
    )


def seleccionar_canales(
    datos,
    canales_totales,
    canales_deseados,
    colores_totales
):

    indices = [
        canales_totales.index(ch)
        for ch in canales_deseados
    ]

    datos_sel = datos[
        indices
    ]

    colores_sel = [
        colores_totales[i]
        for i in indices
    ]

    return datos_sel, colores_sel


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA DEL VECTOR DE PESOS
# ─────────────────────────────────────────────────────────────────────────────

def plot_figura(
        nombre_fig,
        pesos_instar,
        tiempo,
        pie):

    # ─────────────────────────────────────────────────────────────────────────
    # UNA SOLA FIGURA CON 3 SUBPLOTS
    # ─────────────────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(
        3,
        1,
        sharex=True,
        gridspec_kw={
            'hspace': ESPACIO_ENTRE
        }
    )

    fig.patch.set_facecolor(
        BG_FIG
    )

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
    # SUBPLOT 1,1 → VACÍO
    # ─────────────────────────────────────────────────────────────────────────

    axes[0].set_facecolor(
        BG_FIG
    )

    axes[0].axis('off')


    # ─────────────────────────────────────────────────────────────────────────
    # SUBPLOT 2,1 → VECTOR DE PESOS INSTAR
    # ─────────────────────────────────────────────────────────────────────────

    estilo_ax(
        axes[1]
    )

    axes[1].tick_params(
        labelbottom=True
    )


    # ─────────────────────────────────────────────────────────────────────────
    # COLOR DEL ECRL
    # ─────────────────────────────────────────────────────────────────────────

    indice_ecrl = MUSCULOS.index('ECRL')

    color_ecrl = COLORES_EMG[
        indice_ecrl
    ]


    # ─────────────────────────────────────────────────────────────────────────
    # ESCALA Y
    # ─────────────────────────────────────────────────────────────────────────

    y_min = pesos_instar.min()
    y_max = pesos_instar.max()

    margen = (
        y_max - y_min
    ) * 0.01

    if margen == 0:
        margen = 1


    # ─────────────────────────────────────────────────────────────────────────
    # VECTOR DE PESOS
    # ─────────────────────────────────────────────────────────────────────────

    axes[1].plot(
        tiempo,
        pesos_instar,
        color=color_ecrl,
        linewidth=2.0
    )


    # ─────────────────────────────────────────────────────────────────────────
    # LABEL
    # ─────────────────────────────────────────────────────────────────────────

    axes[1].set_ylabel(
        r"Vector $w$",
        color=COLOR_LABEL,
        fontsize=SIZENAMES,
        fontweight='bold'
    )


    axes[1].set_xlabel(
        "Tiempo (s)",
        color=COLOR_LABEL,
        fontsize=SIZENAMES,
        fontweight='bold'
    )


    # ─────────────────────────────────────────────────────────────────────────
    # LIMITES
    # ─────────────────────────────────────────────────────────────────────────

    axes[1].set_xlim(
        tiempo[0],
        tiempo[-1]
    )

    axes[1].set_ylim(
        y_min - margen,
        y_max + margen
    )


    # ─────────────────────────────────────────────────────────────────────────
    # TICKS
    # ─────────────────────────────────────────────────────────────────────────

    axes[1].xaxis.set_major_locator(
        ticker.MultipleLocator(
            TICKAZOS_LABEL
        )
    )
    
    axes[1].yaxis.set_major_locator(
        ticker.MultipleLocator(
            150.0
        )
    )


    # ─────────────────────────────────────────────────────────────────────────
    # SUBPLOT 3,1 → VACÍO
    # ─────────────────────────────────────────────────────────────────────────

    axes[2].set_facecolor(
        BG_FIG
    )

    axes[2].axis('off')


    # ─────────────────────────────────────────────────────────────────────────
    # GUARDAR IMAGEN
    # ─────────────────────────────────────────────────────────────────────────

    CARPETA_IMGS = r"D:\luiso\Descargas\prueba"

    os.makedirs(
        CARPETA_IMGS,
        exist_ok=True
    )

    nombre_archivo = os.path.splitext(
        os.path.basename(RUTA_EMG)
    )[0]

    ruta_img = os.path.join(
        CARPETA_IMGS,
        f"{nombre_archivo}_EMG_INSTAR.png"
    )


    fig.savefig(
        ruta_img,
        dpi=300,
        bbox_inches='tight',
        facecolor=fig.get_facecolor()
    )


    print(
        f"\n[OK] Figura guardada en:\n"
        f"{ruta_img}"
    )


    pantalla_completa(
        fig
    )

    return fig

# ─────────────────────────────────────────────────────────────────────────────
# CARGAR EMG
# ─────────────────────────────────────────────────────────────────────────────

datos = cargar_npy_emg(
    RUTA_EMG
)

n_canales, n_muestras = datos.shape


# ─────────────────────────────────────────────────────────────────────────────
# RECORTAR SEÑAL
# ─────────────────────────────────────────────────────────────────────────────

if (
    SEGUNDO_INICIO is not None
    and SEGUNDO_FIN is not None
):

    muestra_inicio = int(
        SEGUNDO_INICIO * FS
    )

    muestra_fin = int(
        SEGUNDO_FIN * FS
    )

    if (
        muestra_inicio < n_muestras
        and muestra_fin <= n_muestras
    ):

        datos = datos[
            :,
            muestra_inicio:muestra_fin
        ]

        n_canales, n_muestras = datos.shape

    else:

        print(
            f"[AVISO] El intervalo solicitado "
            f"({SEGUNDO_INICIO} - {SEGUNDO_FIN} s) "
            f"supera la duración de la señal "
            f"({n_muestras / FS:.2f} s). "
            f"Se procesará la señal completa."
        )


# ─────────────────────────────────────────────────────────────────────────────
# VECTOR DE TIEMPO
# ─────────────────────────────────────────────────────────────────────────────

tiempo = np.arange(
    n_muestras
) / FS


print(
    f"\n[OK] EMG cargado: "
    f"{n_canales} canales | "
    f"{n_muestras} muestras | "
    f"Fs = {FS} Hz"
)


# ─────────────────────────────────────────────────────────────────────────────
# FILTRO NOTCH
# ─────────────────────────────────────────────────────────────────────────────

datos_notch = aplicar_notch_multiple(
    datos,
    fs=FS,
    freqs=FREQS_NOTCH,
    Q=Q_NOTCH,
    pad=PAD_MUESTRAS
)

print(
    f"[OK] Notch aplicado: "
    f"{FREQS_NOTCH} Hz | "
    f"Q = {Q_NOTCH}"
)


# ─────────────────────────────────────────────────────────────────────────────
# FILTRO PASA BANDA
# ─────────────────────────────────────────────────────────────────────────────

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
    f"{BP_ORDER}º orden | "
    f"{F_LOW:.0f}-{F_HIGH:.0f} Hz"
)


# # ─────────────────────────────────────────────────────────────────────────────
# # RESAMPLEO
# # ─────────────────────────────────────────────────────────────────────────────

# datos_submuestreados = resamplear_profesional(
#     datos_filtrados,
#     1000,
#     250
# )

# FS = FS / 4

# n_muestras = datos_submuestreados.shape[1]

# tiempo = np.arange(
#     n_muestras
# ) * (1 / FS)


# print(
#     f"[OK] Submuestreo aplicado: "
#     f"1000 Hz → {FS:.0f} Hz | "
#     f"Muestras: {n_muestras}"
# )


# ─────────────────────────────────────────────────────────────────────────────
# SELECCIONAR CANALES
# ─────────────────────────────────────────────────────────────────────────────

datos_plot, colores_plot = seleccionar_canales(
    datos_filtrados,
    MUSCULOS,
    MUSCULOS_PLOT,
    COLORES_EMG
)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAR INSTAR
# ─────────────────────────────────────────────────────────────────────────────

print(
    "\n[INFO] Entrenando red Instar..."
)

pesos_instar = entrenar_instar(
    *datos_plot,
    epochs=EPOCHS_INSTAR,
    alpha=ALPHA_INSTAR,
    bias=BIAS_INSTAR
)


pesos_instar = np.asarray(
    pesos_instar
).flatten()


print(
    f"[OK] Instar entrenada | "
    f"Epochs = {EPOCHS_INSTAR} | "
    f"Alpha = {ALPHA_INSTAR} | "
    f"Bias = {BIAS_INSTAR}"
)

print(
    f"[OK] Vector de pesos: "
    f"{len(pesos_instar)} elementos"
)


# ─────────────────────────────────────────────────────────────────────────────
# COMPROBAR LONGITUD
# ─────────────────────────────────────────────────────────────────────────────

if len(pesos_instar) != len(tiempo):

    raise ValueError(
        f"La longitud del vector de pesos "
        f"({len(pesos_instar)}) no coincide "
        f"con la longitud del vector de tiempo "
        f"({len(tiempo)})."
    )


# ─────────────────────────────────────────────────────────────────────────────
# GRAFICAR VECTOR DE PESOS
# ─────────────────────────────────────────────────────────────────────────────

nombre_archivo = os.path.splitext(
    os.path.basename(RUTA_EMG)
)[0]


plot_figura(
    nombre_fig=nombre_archivo,
    pesos_instar=pesos_instar,
    tiempo=tiempo,
    pie=PIE_FIGURA
)


print(
    "\n[FIN] Vector de pesos Instar generado.\n"
)

plt.show()