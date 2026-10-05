import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from Procesamiento_Funciones import (
    cargar_npy_emg,
    aplicar_notch_multiple,
    aplicar_pasabanda_butter,
    resamplear_profesional
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

VENTANA_X = 11
VENTANA_Y = 10

PAD_MUESTRAS = 5 * FS

# Canales EMG
MUSCULOS = ['FCU', 'ECRL', 'ECU']

# Canales que deseas visualizar
MUSCULOS_PLOT = ['FCU', 'ECRL', 'ECU']

COLORES_EMG = [
    '#93D989',
    '#70BBD6',
    '#7094FF'
]

BG_FIG      = '#ffffff'
BG_AX       = '#ffffff'
COLOR_GRID  = '#5a5a5a'
COLOR_TICK  = '#000000'
COLOR_LABEL = '#000000'
COLOR_TITLE = '#000000'

SIZENAMES = 36
SIZELABELS = 30

ESPACIO_ENTRE = 0.1

FREQS_NOTCH = [19.6, 60, 120, 180, 240, 300]  # Hz
Q_NOTCH = 200

F_LOW = 20.0
F_HIGH = 300.0
BP_ORDER = 4

# ─────────────────────────────────────────────────────────────────────────────
# PARSEO DEL NOMBRE DEL ARCHIVO
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

    datos_sel = datos[indices]

    colores_sel = [
        colores_totales[i]
        for i in indices
    ]

    return datos_sel, colores_sel


def plot_figura(
    nombre_fig,
    datos,
    tiempo,
    fs,
    colores,
    canales,
    pie
):

    n_graficas = len(canales)

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURA
    # ─────────────────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(
        n_graficas,
        1,
        sharex=True,
        gridspec_kw={
            'hspace': 0.15
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
    # ESCALA Y COMPARTIDA
    # ─────────────────────────────────────────────────────────────────────────

    y_min = datos.min()
    y_max = datos.max()

    margen = (y_max - y_min) * 0.01

    # Evitar margen cero si la señal es constante
    if margen == 0:
        margen = 1

    # ─────────────────────────────────────────────────────────────────────────
    # GRAFICAR CANALES
    # ─────────────────────────────────────────────────────────────────────────

    for i, ax in enumerate(axes):

        estilo_ax(ax)

        ax.plot(
            tiempo,
            datos[i],
            color=colores[i],
            linewidth=2.0
        )

        ax.set_ylabel(
            canales[i],
            color=COLOR_LABEL,
            fontsize=SIZENAMES,
            fontweight='bold'
        )

        ax.set_xlim(
            tiempo[0],
            tiempo[-1]
        )

        ax.set_ylim(
            y_min - margen,
            y_max + margen
        )

        ax.xaxis.set_major_locator(
            ticker.MultipleLocator(
                TICKAZOS_LABEL
            )
        )

        # Ocultar etiquetas X excepto en el último subplot
        if i < n_graficas - 1:

            plt.setp(
                ax.get_xticklabels(),
                visible=False
            )

    # ─────────────────────────────────────────────────────────────────────────
    # EJE X
    # ─────────────────────────────────────────────────────────────────────────

    axes[-1].set_xlabel(
        "Tiempo (s)",
        color=COLOR_LABEL,
        fontsize=SIZENAMES,
        fontweight='bold'
    )

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
        f"{nombre_archivo}_EMG_FILTRADA.png"
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

    pantalla_completa(fig)

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
    f"{FREQS_NOTCH} Hz y {FREQS_NOTCH} Hz | Q = {Q_NOTCH}"
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
    f"{BP_ORDER}º orden | {F_LOW:.0f}-{F_HIGH:.0f} Hz"
)

# # ─────────────────────────────────────────────────────────────────────────────
# # RESAMPLEAR
# # ─────────────────────────────────────────────────────────────────────────────

# datos_submuestreados = resamplear_profesional(datos_filtrados, 1000, 250)

# FS = FS / 4  # Ahora FS pasa de 1000 a 250 Hz
# n_muestras = datos_submuestreados.shape[1]
# tiempo = np.arange(n_muestras) * (1/FS)

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
# GRAFICAR EMG CRUDO
# ─────────────────────────────────────────────────────────────────────────────

plot_figura(
    nombre_fig="EMG Crudo",
    datos=datos_plot,
    tiempo=tiempo,
    fs=FS,
    colores=colores_plot,
    canales=MUSCULOS_PLOT,
    pie=PIE_FIGURA
)


print(
    "\n[FIN] Figura EMG crudo generada.\n"
)

plt.show()