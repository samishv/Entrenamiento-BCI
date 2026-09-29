import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from Procesamiento_Funciones import (
    cargar_npy_eeg,
    aplicar_notch_multiple,
    aplicar_pasabanda_butter,
    filtro_laplaciano,
    resamplear_profesional,
)

plt.close('all')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

if "RUTA_NPY" not in globals():
    RUTA_NPY = r"D:\luiso\Documentos\Luis\UPIITA\TT\Entrenamiento-BCI\DataCMC\Registros CMC 2\S17\EEG\S17_260421_EEG_OK_3_ME.npy"
    #RUTA_NPY = r"D:\luiso\Documentos\Luis\UPIITA\TT\Entrenamiento-BCI\DataCMC\Registros CMC 2\S5\EEG\S5_260429_EEG_OV_1_ME.npy"

FS = 250
SEGUNDOTES = 2
TICKAZOS_LABEL = 0.5

VENTANA_X = 10
VENTANA_Y = 10

CANALES = ['Fz', 'FC3', 'FCz', 'FC4', 'Cz', 'C3', 'C4']

# Canales que deseas visualizar
CANALES_PLOT = ['FC3', 'C3', 'Cz']

CANAL_INTERES_LAPLACIANO = 'C3'
CANALES_VECINOS_LAPLACIANO = ['Cz', 'FC3']

COLORES = [
    '#000000',
    '#15D18F',
    '#000000',
    '#000000',
    '#008ACF',
    '#13A2A9',
    '#000000'
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

# Parámetros de filtrado
FREQS_NOTCH = (60.0, 120.0)
Q_NOTCH = 5.0

F_LOW = 5.0
F_HIGH = 45.0
BP_ORDER = 8

PAD_MUESTRAS = 5 * FS

# ─────────────────────────────────────────────────────────────────────────────
# PARSEO DEL NOMBRE DEL ARCHIVO
# ─────────────────────────────────────────────────────────────────────────────

CONDICION_MAP = {
    'OV': 'Ojos Abiertos - Movimiento Visual (OV)',
    'OK': 'Ojos Abiertos - Movimiento Cinestésico (OK)',
    'CK': 'Ojos Cerrados - Movimiento Cinestésico (CK)',
    'CV': 'Ojos Cerrados - Movimiento Visual (CV)',
}

ESTADO_MAP = {
    'ME': 'Movimiento Ejecutado (ME)',
    'MI': 'Movimiento Imaginado (MI)',
    'RE': 'Reposo (RE)',
}


def parsear_nombre_archivo(ruta):

    nombre = os.path.splitext(os.path.basename(ruta))[0]
    partes = nombre.split('_')

    sujeto = partes[0]
    condicion = partes[3]
    estado = partes[5]

    num_sujeto = sujeto[1:]

    desc_cond = CONDICION_MAP.get(condicion, condicion)
    desc_estado = ESTADO_MAP.get(estado, estado)

    return (
        f"Sujeto {num_sujeto} | "
        f"{desc_cond} | "
        f"{desc_estado}"
    )


PIE_FIGURA = parsear_nombre_archivo(RUTA_NPY)

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


def quitar_componente_dc(datos):

    datos = np.asarray(datos)

    datos_sin_dc = datos - np.mean(
        datos,
        axis=1,
        keepdims=True
    )

    return datos_sin_dc


def seleccionar_canales(
        datos,
        canales_totales,
        canales_deseados,
        colores_totales):

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
        datos_laplaciano,
        tiempo,
        colores,
        canales,
        pie):

    # ─────────────────────────────────────────────────────────────────────────
    # UNA SOLA FIGURA CON 3 SUBPLOTS
    # ─────────────────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(
        3,
        1,
        sharex=True,
        gridspec_kw={'hspace': ESPACIO_ENTRE}
    )

    fig.patch.set_facecolor(BG_FIG)

    fig.subplots_adjust(
        left=0.06,
        right=0.94,
        top=0.94,
        bottom=ESPACIO_ENTRE
    )

    fig.set_size_inches(VENTANA_X, VENTANA_Y)

    # ─────────────────────────────────────────────────────────────────────────
    # SUBPLOT 1,1 → VACÍO (mismo color que el fondo, sin ejes ni nada)
    # ─────────────────────────────────────────────────────────────────────────

    axes[0].set_facecolor(BG_FIG)
    axes[0].axis('off')

    # ─────────────────────────────────────────────────────────────────────────
    # SUBPLOT 2,1 → RESULTADO DEL FILTRO LAPLACIANO
    # ─────────────────────────────────────────────────────────────────────────

    estilo_ax(axes[1])

    axes[1].tick_params(
        labelbottom=True
    )

    # filtro_laplaciano regresa directamente el vector 1D (n_muestras,)
    # de la señal laplaciana de C3, ya calculada como C3 - promedio(vecinos)
    indice_c3 = canales.index('C3')
    senal_laplaciano = datos_laplaciano

    y_min = senal_laplaciano.min()
    y_max = senal_laplaciano.max()

    margen = (y_max - y_min) * 0.05

    if margen == 0:
        margen = 1

    axes[1].plot(
        tiempo,
        senal_laplaciano,
        color=colores[indice_c3],
        linewidth=2.0
    )

    axes[1].set_ylabel(
        r"$\nabla^2$ C3",
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

    axes[1].set_xlim(
        tiempo[0],
        tiempo[-1]
    )

    axes[1].set_ylim(
        y_min - margen,
        y_max + margen
    )
    
    axes[1].yaxis.set_major_locator(
        ticker.MultipleLocator(2.5)
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SUBPLOT 3,1 → VACÍO (mismo color que el fondo, sin ejes ni nada)
    # ─────────────────────────────────────────────────────────────────────────

    axes[2].set_facecolor(BG_FIG)
    axes[2].axis('off')
    
    axes[2].xaxis.set_major_locator(
        ticker.MultipleLocator(TICKAZOS_LABEL)
    )

    # ─────────────────────────────────────────────────────────────────────────
    # GUARDAR IMAGEN
    # ─────────────────────────────────────────────────────────────────────────

    CARPETA_IMGS = r"D:\luiso\Descargas\prueba"
    os.makedirs(CARPETA_IMGS, exist_ok=True)

    ruta_img = os.path.join(
        CARPETA_IMGS,
        f"{nombre_fig}_EEG_LAPLACIANO.png"
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
# CARGAR, FILTRAR Y GRAFICAR EEG
# ─────────────────────────────────────────────────────────────────────────────

datos = cargar_npy_eeg(
    RUTA_NPY,
    n_canales_esperados=len(CANALES)
)

n_canales, n_muestras = datos.shape

if SEGUNDOTES is not None:

    muestras_a_procesar = int(SEGUNDOTES * FS)

    if muestras_a_procesar < n_muestras:

        datos = datos[:, :muestras_a_procesar]

        n_canales, n_muestras = datos.shape

    else:

        print(
            f"[AVISO] SEGUNDOTES = {SEGUNDOTES} s "
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

# ─────────────────────────────────────────────────────────────────────────────
# ELIMINAR DC
# ─────────────────────────────────────────────────────────────────────────────

datos_sin_dc = quitar_componente_dc(datos)

print("\n[OK] Componente de corriente directa eliminada")

# ─────────────────────────────────────────────────────────────────────────────
# NOTCH
# ─────────────────────────────────────────────────────────────────────────────

datos_notch = aplicar_notch_multiple(
    datos_sin_dc,
    fs=FS,
    freqs=FREQS_NOTCH,
    Q=Q_NOTCH,
    pad=PAD_MUESTRAS
)

print(
    f"[OK] Notch aplicado: "
    f"{FREQS_NOTCH} Hz | Q = {Q_NOTCH}"
)

# ─────────────────────────────────────────────────────────────────────────────
# BUTTERWORTH
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

# ─────────────────────────────────────────────────────────────────────────────
# RESAMPLEAR
# ─────────────────────────────────────────────────────────────────────────────

datos_submuestreados = resamplear_profesional(datos_filtrados, 250, 1000)

FS = FS * 4  # Ahora FS pasa de 250 a 1000 Hz
n_muestras = datos_submuestreados.shape[1]
tiempo = np.arange(n_muestras) * (1/FS)

# ─────────────────────────────────────────────────────────────────────────────
# FILTRO LAPLACIANO
# ─────────────────────────────────────────────────────────────────────────────

datos_laplaciano = filtro_laplaciano(
    datos_submuestreados,
    CANALES,
    CANAL_INTERES_LAPLACIANO,
    CANALES_VECINOS_LAPLACIANO
)

print(
    f"[OK] Filtro laplaciano aplicado sobre "
    f"{CANAL_INTERES_LAPLACIANO}"
)

# ─────────────────────────────────────────────────────────────────────────────
# GRAFICAR
# ─────────────────────────────────────────────────────────────────────────────

nombre_archivo = os.path.splitext(
    os.path.basename(RUTA_NPY)
)[0]

plot_figura(
    nombre_fig=nombre_archivo,
    datos_laplaciano=datos_laplaciano,
    tiempo=tiempo,
    colores=COLORES,
    canales=CANALES,
    pie=PIE_FIGURA
)

print("\n[FIN] Figura EEG filtrado + Laplaciano generada.\n")

plt.show()