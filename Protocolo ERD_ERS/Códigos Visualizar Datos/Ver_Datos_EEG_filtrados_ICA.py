import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mne

from mne.preprocessing import ICA

from Procesamiento_Funciones import (
    cargar_npy_eeg,
    aplicar_notch_multiple,
    aplicar_pasabanda_butter
)

plt.close('all')

# RECORDAR
# 3 SEGS REPOSO | 15 SEGUNDOS EJECUCION | 3 SEGS REPOSO

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

RUTA_NPY = r"D:\luiso\Documentos\Luis\UPIITA\TT\Entrenamiento-BCI\Protocolo ERD_ERS\Signals\SV1\S17_260421_EEG_OV_2_ME.npy"

FS = 250

SEGUNDO_INICIO = 0
SEGUNDO_FIN = 21

TICKAZOS_LABEL = 3.0

VENTANA_X = 19
VENTANA_Y = 9.5

CANALES = ['Fz', 'FC3', 'FCz', 'FC4', 'Cz', 'C3', 'C4']

# Canales que deseas visualizar

CANALES_PLOT = ['Fz', 'FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4']

COLORES = [
    '#EBEF91',  # 1. Amarillo claro y luminoso (base inicial)
    '#B5E48C',  # 2. Verde lima suave (puente entre el amarillo y el verde)
    '#99D98C',  # 3. Verde manzana (tu tono de referencia central)
    '#6FC39C',  # 4. Verde azulado / Esmeralda suave (transición hacia el azul)
    '#3DA2F5',  # 5. Azul brillante (tu tono de referencia central)
    '#64B5F7',  # 6. Azul medio (aporta profundidad y cuerpo)
    '#248BE5'   # 7. Azul marino / Real (cierre con fuerza y contraste)
]

COLORES_ICA = ["#07c8f9","#08b2f5","#099bf1","#0a85ed","#0b6ee9","#0c58e5","#0d41e1"]

# BG_FIG      = '#ffffff'
# BG_AX       = '#ffffff'
# COLOR_GRID  = '#5a5a5a'
# COLOR_TICK  = '#000000'
# COLOR_LABEL = '#000000'

BG_FIG      = '#000000'
BG_AX       = '#000000'
COLOR_GRID  = '#cccccc'
COLOR_TICK  = '#ffffff'
COLOR_LABEL = '#ffffff'

SIZENAMES = 14
SIZELABELS = 12
ESPACIO_ENTRE = 0.07

LINEWIDOTHA = 1.5

# Parámetros de filtrado

FREQS_NOTCH = (60.0, 120.0)
Q_NOTCH = 5.0

F_LOW = 5.0
F_HIGH = 45.0
BP_ORDER = 3

PAD_MUESTRAS = 5 * FS


# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS ICA
# ─────────────────────────────────────────────────────────────────────────────

USAR_ICA = False

IC_TO_REMOVE = 0

ICA_METHOD = 'infomax'
ICA_RANDOM_STATE = 97
ICA_MAX_ITER = 'auto'


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
    datos,
    tiempo,
    colores,
    canales
):

    n_graficas = len(canales)

    fig, axes = plt.subplots(
        n_graficas,
        1,
        sharex=True,
        sharey=True,
        gridspec_kw={'hspace': ESPACIO_ENTRE}
    )

    if n_graficas == 1:
        axes = [axes]

    fig.patch.set_facecolor(BG_FIG)

    fig.subplots_adjust(
        left=0.045,
        right=0.98,
        top=0.98,
        bottom=ESPACIO_ENTRE
    )

    fig.set_size_inches(
        VENTANA_X,
        VENTANA_Y
    )

    y_min = datos.min()
    y_max = datos.max()

    margen = (y_max - y_min) * 0.01

    for i, ax in enumerate(axes):

        estilo_ax(ax)

        ax.plot(
            tiempo,
            datos[i],
            color=colores[i],
            linewidth=LINEWIDOTHA
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
            y_min*0.8 - margen,
            y_max*0.8 + margen
        )

        ax.xaxis.set_major_locator(
            ticker.MultipleLocator(TICKAZOS_LABEL)
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

    pantalla_completa(fig)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CARGAR, FILTRAR, ICA (OPCIONAL) Y GRAFICAR EEG
# ─────────────────────────────────────────────────────────────────────────────

datos = cargar_npy_eeg(
    RUTA_NPY,
    n_canales_esperados=len(CANALES)
)

n_canales, n_muestras = datos.shape

if SEGUNDO_INICIO is not None and SEGUNDO_FIN is not None:

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
            f"({n_muestras/FS:.2f} s). "
            f"Se procesará la señal completa."
        )


tiempo = np.arange(
    n_muestras
) / FS


print(
    f"\n[OK] EEG cargado: "
    f"{n_canales} canales | "
    f"{n_muestras} muestras"
)


# ─────────────────────────────────────────────────────────────────────────────
# 1) ELIMINAR COMPONENTE DC
# ─────────────────────────────────────────────────────────────────────────────

datos_sin_dc = quitar_componente_dc(
    datos
)

print(
    "\n[OK] Componente de corriente directa eliminada"
)


# ─────────────────────────────────────────────────────────────────────────────
# 2) FILTRO NOTCH
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
# 3) FILTRO PASA BANDA
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


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 1 — EEG ANTES DE ICA
# ─────────────────────────────────────────────────────────────────────────────

datos_plot, colores_plot = seleccionar_canales(
    datos_filtrados,
    CANALES,
    CANALES_PLOT,
    COLORES
)

plot_figura(
    datos=datos_plot,
    tiempo=tiempo,
    colores=colores_plot,
    canales=CANALES_PLOT
)

if USAR_ICA:
    print(
        "\n[OK] Figura 1: EEG antes de ICA generada."
    )
else:
    print(
        "\n[OK] Figura 1: EEG filtrado generada."
    )


if USAR_ICA:

    # ─────────────────────────────────────────────────────────────────────────
    # 4) CREAR RAW MNE PARA ICA
    # ─────────────────────────────────────────────────────────────────────────

    info = mne.create_info(
        ch_names=CANALES,
        sfreq=FS,
        ch_types=['eeg'] * len(CANALES)
    )

    raw = mne.io.RawArray(
        datos_filtrados,
        info,
        verbose=False
    )


    # ─────────────────────────────────────────────────────────────────────────
    # 5) APLICAR ICA
    # ─────────────────────────────────────────────────────────────────────────

    rank = mne.compute_rank(
        raw,
        verbose=False
    )

    n_comp = min(
        rank.get('eeg', len(CANALES)),
        len(CANALES)
    )

    print(
        f"\n[OK] Rank EEG estimado: "
        f"{rank.get('eeg', None)}"
    )

    print(
        f"[OK] Número de componentes ICA: "
        f"{n_comp}"
    )


    ica = ICA(
        n_components=n_comp,
        method=ICA_METHOD,
        fit_params=dict(extended=True),
        random_state=ICA_RANDOM_STATE,
        max_iter=ICA_MAX_ITER
    )

    ica.fit(raw)

    print(
        "\n[OK] ICA ajustada"
    )


    # ─────────────────────────────────────────────────────────────────────────
    # FIGURA 2 — COMPONENTES INDEPENDIENTES
    # ─────────────────────────────────────────────────────────────────────────

    sources_raw = ica.get_sources(
        raw
    )

    sources = sources_raw.get_data()

    ic_labels = [
        f"IC{i}"
        for i in range(sources.shape[0])
    ]


    # Crear tiempo correspondiente a las componentes
    tiempo_ica = np.arange(
        sources.shape[1]
    ) / FS


    plot_figura(
        datos=sources,
        tiempo=tiempo_ica,
        colores=COLORES_ICA,
        canales=ic_labels
    )

    print(
        "\n[OK] Figura 2: Componentes independientes generada."
    )


    # ─────────────────────────────────────────────────────────────────────────
    # 6) RECONSTRUCCIÓN DE LA SEÑAL
    # ─────────────────────────────────────────────────────────────────────────

    if IC_TO_REMOVE is None:

        exclude_list = []

        print(
            "\n[OK] IC_TO_REMOVE = None"
        )

        print(
            "[OK] No se eliminará ninguna componente."
        )

    else:

        if isinstance(IC_TO_REMOVE, int):

            exclude_list = [
                IC_TO_REMOVE
            ]

        else:

            exclude_list = list(
                IC_TO_REMOVE
            )


        # Verificar que las componentes indicadas existan
        for ic in exclude_list:

            if ic < 0 or ic >= n_comp:

                raise ValueError(
                    f"El IC {ic} no existe. "
                    f"Los componentes disponibles son "
                    f"IC0 hasta IC{n_comp - 1}."
                )


    print(
        f"\n[ICA] Componentes a excluir: "
        f"{exclude_list}"
    )


    ica_recon = ica.copy()

    ica_recon.exclude = exclude_list


    raw_reconstruido = raw.copy()

    ica_recon.apply(
        raw_reconstruido
    )


    datos_reconstruidos = (
        raw_reconstruido.get_data()
    )


    print(
        "\n[OK] Reconstrucción ICA realizada."
    )


    # ─────────────────────────────────────────────────────────────────────────
    # FIGURA 3 — EEG POST RECONSTRUCCIÓN ICA
    # ─────────────────────────────────────────────────────────────────────────

    datos_recon_plot, colores_recon_plot = seleccionar_canales(
        datos_reconstruidos,
        CANALES,
        CANALES_PLOT,
        COLORES
    )

    plot_figura(
        datos=datos_recon_plot,
        tiempo=tiempo,
        colores=colores_recon_plot,
        canales=CANALES_PLOT
    )

    print(
        "\n[OK] Figura 3: EEG post reconstrucción ICA generada."
    )


    print(
        "\n[FIN] Procesamiento completo:"
    )

    print(
        "      Figura 1 → EEG antes de ICA"
    )

    print(
        "      Figura 2 → Componentes independientes"
    )

    print(
        "      Figura 3 → EEG post reconstrucción ICA"
    )


plt.show()