import os
import numpy as np
import matplotlib.pyplot as plt

from Procesamiento_Funciones import cargar_npy_eeg

plt.close('all')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

if "RUTA_NPY" not in globals():
    RUTA_NPY = r"D:\luiso\Documentos\Luis\UPIITA\TT\Entrenamiento-BCI\DataCMC\Registros CMC 2\S17\EEG\S17_260421_EEG_OK_3_ME.npy"

FS = 250

CANALES = ['Fz', 'FC3', 'FCz', 'FC4', 'Cz', 'C3', 'C4']

# Canales que deseas visualizar
CANALES_PLOT = ['C3','Cz','C4']

COLORES = [
    '#6FC7B7',
    '#A1DAD1',
    '#D0EDE8',
    '#FFFFFF',
    '#BFEEF6',
    '#81DCEE',
    '#41CCE3'
]

BG_FIG      = '#000000'
BG_AX       = '#000000'
COLOR_GRID  = '#7a7a7a'
COLOR_TICK  = '#d1d1d1'
COLOR_LABEL = '#ffffff'
COLOR_TITLE = '#ffffff'

SIZENAMES = 32
SIZELABELS = 26
ESPACIO_ENTRE = 0.1

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


def centrar_senales(datos):

    datos = np.asarray(datos)

    return datos - np.mean(
        datos,
        axis=1,
        keepdims=True
    )


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
        datos,
        tiempo,
        fs,
        colores,
        canales,
        pie):

    n_graficas = len(canales)

    fig, axes = plt.subplots(
        n_graficas,
        1,
        sharex=True,
        gridspec_kw={'hspace': 0.15}
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

    fig.set_size_inches(16, 16)

    # fig.suptitle(
    #     f'{nombre_fig} | Señales EEG sin preprocesamiento | fs = {fs} Hz',
    #     color=COLOR_TITLE,
    #     fontsize=SIZENAMES,
    #     fontweight='bold'
    # )

    y_min = datos.min()
    y_max = datos.max()

    margen = (y_max - y_min) * 0.05

    # fig.text(
    #     0.5,
    #     0.01,
    #     pie,
    #     ha='center',
    #     va='bottom',
    #     fontsize=SIZELABELS,
    #     color=COLOR_LABEL,
    #     style='italic',
    #     fontweight='bold'
    # )

    for i, ax in enumerate(axes):

        estilo_ax(ax)

        ax.plot(
            tiempo,
            datos[i],
            color=colores[i],
            linewidth=1.5,
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

        if i < n_graficas - 1:
            plt.setp(
                ax.get_xticklabels(),
                visible=False
            )

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
    os.makedirs(CARPETA_IMGS, exist_ok=True)

    nombre_archivo = os.path.splitext(
        os.path.basename(RUTA_NPY)
    )[0]

    ruta_img = os.path.join(
        CARPETA_IMGS,
        f"{nombre_archivo}_EEG_CRUDO.png"
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
# CARGAR EEG
# ─────────────────────────────────────────────────────────────────────────────

datos = cargar_npy_eeg(
    RUTA_NPY,
    n_canales_esperados=len(CANALES)
)

n_canales, n_muestras = datos.shape

tiempo = np.arange(n_muestras) / FS

print(
    f"\n[OK] EEG cargado: "
    f"{n_canales} canales | "
    f"{n_muestras} muestras"
)

datos_centrados = centrar_senales(datos)

datos_plot, colores_plot = seleccionar_canales(
    datos_centrados,
    CANALES,
    CANALES_PLOT,
    COLORES
)

plot_figura(
    nombre_fig="EEG Crudo",
    datos=datos_plot,
    tiempo=tiempo,
    fs=FS,
    colores=colores_plot,
    canales=CANALES_PLOT,
    pie=PIE_FIGURA
)

print("\n[FIN] Figura EEG crudo generada.\n")

plt.show()