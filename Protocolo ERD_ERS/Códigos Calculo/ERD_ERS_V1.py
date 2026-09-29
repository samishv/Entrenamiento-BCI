# -*- coding: utf-8 -*-

"""
ANALISIS ERSP Y ERD/ERS PORCENTUAL
=================================

Protocolo:
    0-3 s   : reposo inicial
    3-18 s  : contraccion sostenida de la mano derecha
    18-21 s : reposo posterior

Datos:
    - 4 ensayos independientes
    - 7 canales
    - 5250 muestras
    - Fs = 250 Hz
    - Unidades originales: microvoltios

Analisis:
    1. Eliminacion de componente DC.
    2. Filtro notch.
    3. Filtro pasa banda.
    4. ICA opcional.
    5. ERSP mediante wavelets de Morlet.
    6. ERD/ERS porcentual mediante FFT de ventana deslizante.
    7. Comparaciones por canal, banda y ensayo.
    8. Exportacion de resultados numericos a CSV.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scipy import signal

import mne
from mne.preprocessing import ICA
from mne.time_frequency import tfr_array_morlet

from Procesamiento_Funciones import (
    cargar_npy_eeg,
    aplicar_notch_multiple,
    aplicar_pasabanda_butter,
)


# =============================================================================
# CONFIGURACION GENERAL
# =============================================================================

plt.close("all")

# Carpeta en la que se encuentran los cuatro ensayos
CARPETA_DATOS = Path(
    r"D:\luiso\Documentos\Luis\UPIITA\TT"
    r"\Entrenamiento-BCI\Protocolo ERD_ERS\Signals\SV1"
)

# Patron usado para localizar los registros
PATRON_ARCHIVOS = "*.npy"

# Carpeta de salida
CARPETA_RESULTADOS = CARPETA_DATOS / "Resultados_ERD_ERS"

GUARDAR_RESULTADOS = True
MOSTRAR_FIGURAS = True

# Informacion de adquisicion
FS = 250.0
DURACION_ESPERADA = 21.0
N_MUESTRAS_ESPERADAS = int(FS * DURACION_ESPERADA)

CANALES = [
    "Fz",
    "FC3",
    "FCz",
    "FC4",
    "Cz",
    "C3",
    "C4",
]

N_CANALES_ESPERADOS = len(CANALES)

# Unidades originales de los archivos
UNIDAD_DATOS = "microvoltios"


# =============================================================================
# INTERVALOS DEL PROTOCOLO
# =============================================================================

REPOSO_INICIAL_COMPLETO = (0.0, 3.0)
MOVIMIENTO_COMPLETO = (3.0, 18.0)
REPOSO_FINAL_COMPLETO = (18.0, 21.0)

# Intervalos efectivos para los calculos numericos
BASELINE = (0.5, 2.5)
MOVIMIENTO = (3.2, 17.8)
REPOSO_POSTERIOR = (18.2, 20.5)

# Division complementaria de la contraccion sostenida
FASES_MOVIMIENTO = {
    "Inicio del movimiento": (3.2, 6.0),
    "Parte media": (6.0, 15.0),
    "Final del movimiento": (15.0, 17.8),
}


# =============================================================================
# BANDAS DE FRECUENCIA
# =============================================================================

BANDAS = {
    "Theta": (4.0, 8.0),
    "Mu": (8.0, 12.0),
    "Beta": (12.0, 30.0),
    "Gamma": (30.0, 45.0),
}

COLORES_BANDAS = {
    "Theta": "#9b5de5",
    "Mu": "#00bbf9",
    "Beta": "#00f5d4",
    "Gamma": "#fee440",
}

# Bandas que apareceran en las comparaciones principales
BANDAS_PRINCIPALES = ["Mu", "Beta", "Gamma"]


# =============================================================================
# PREPROCESAMIENTO
# =============================================================================

ELIMINAR_DC = True

FREQS_NOTCH = (60.0, 120.0)
Q_NOTCH = 5.0

F_LOW = 4.0
F_HIGH = 45.0
BP_ORDER = 3

PAD_MUESTRAS = int(5 * FS)

# No se utilizara referencia promedio en esta primera evaluacion
USAR_REFERENCIA_PROMEDIO = False

# ICA inicialmente desactivada
USAR_ICA = False

ICA_METHOD = "infomax"
ICA_RANDOM_STATE = 97
ICA_MAX_ITER = "auto"

# No eliminar componentes automáticamente
IC_TO_REMOVE = None


# =============================================================================
# CONFIGURACION DEL ERSP CON MORLET
# =============================================================================

FREQS_MORLET = np.arange(4.0, 46.0, 1.0)

# Configuracion aproximada creciente.
# Comienza con 3 ciclos en frecuencias bajas y alcanza 10 en 45 Hz.
N_CYCLES_MORLET = np.linspace(
    3.0,
    10.0,
    len(FREQS_MORLET),
)

# Diezmado temporal:
# 250 Hz / 5 = 50 puntos por segundo
MORLET_DECIM = 5

# Numero de trabajos paralelos usado por MNE
N_JOBS = 1

# Limites visuales iniciales del ERSP en dB.
# Si se establece None, se calculan automaticamente por percentiles.
LIMITE_ERSP_DB = None


# =============================================================================
# CONFIGURACION DE FFT PARA ERD/ERS PORCENTUAL
# =============================================================================

# El paper utilizo 200 muestras a 500 Hz:
#
#       200 / 500 = 0.4 segundos
#
# Para conservar 0.4 s con Fs = 250 Hz:
#
#       0.4 * 250 = 100 muestras
FFT_DURACION_VENTANA = 0.4
FFT_NPERSEG = int(FFT_DURACION_VENTANA * FS)

# Paso temporal de 100 ms
FFT_PASO_SEGUNDOS = 0.1
FFT_PASO_MUESTRAS = int(FFT_PASO_SEGUNDOS * FS)

FFT_NOVERLAP = FFT_NPERSEG - FFT_PASO_MUESTRAS

# Se conserva la resolucion de 2.5 Hz
FFT_NFFT = FFT_NPERSEG

# Suavizado exclusivo para visualizacion
APLICAR_SUAVIZADO_CURVAS = True
SUAVIZADO_SEGUNDOS = 0.5


# =============================================================================
# CONFIGURACION DE FIGURAS
# =============================================================================

ESTILO_OSCURO = False

FIG_DPI = 150
FIG_FORMATO = "png"

COLORES_CANALES = {
    "Fz": "#8338ec",
    "FC3": "#3a86ff",
    "FCz": "#06d6a0",
    "FC4": "#ff006e",
    "Cz": "#ffbe0b",
    "C3": "#4361ee",
    "C4": "#ef476f",
}

if ESTILO_OSCURO:
    plt.style.use("dark_background")


# =============================================================================
# FUNCIONES GENERALES
# =============================================================================

def crear_carpeta_resultados():
    """Crea la carpeta de resultados si la exportacion esta activada."""

    if GUARDAR_RESULTADOS:
        CARPETA_RESULTADOS.mkdir(
            parents=True,
            exist_ok=True,
        )


def guardar_figura(figura, nombre):
    """Guarda una figura en la carpeta de resultados."""

    if not GUARDAR_RESULTADOS:
        return

    ruta = CARPETA_RESULTADOS / f"{nombre}.{FIG_FORMATO}"

    figura.savefig(
        ruta,
        dpi=FIG_DPI,
        bbox_inches="tight",
        facecolor=figura.get_facecolor(),
    )

    print(f"[GUARDADO] {ruta}")


def cerrar_o_mostrar():
    """Muestra o cierra las figuras segun la configuracion."""

    if MOSTRAR_FIGURAS:
        plt.show()
    else:
        plt.close("all")


def intervalo_a_mascara(tiempo, intervalo):
    """Convierte un intervalo temporal en una mascara booleana."""

    inicio, fin = intervalo

    return (
        (tiempo >= inicio)
        & (tiempo <= fin)
    )


def quitar_componente_dc(datos):
    """Elimina el promedio temporal de cada canal."""

    datos = np.asarray(
        datos,
        dtype=np.float64,
    )

    return datos - np.mean(
        datos,
        axis=-1,
        keepdims=True,
    )


def aplicar_referencia_promedio(datos):
    """
    Aplica referencia promedio entre canales.

    La referencia promedio esta desactivada de manera predeterminada.
    """

    promedio_instantaneo = np.mean(
        datos,
        axis=0,
        keepdims=True,
    )

    return datos - promedio_instantaneo


def media_movil(datos, ventana):
    """
    Aplica una media movil sin modificar las dimensiones del arreglo.
    Solo se utiliza para suavizar las curvas representadas.
    """

    datos = np.asarray(datos)

    ventana = int(ventana)

    if ventana <= 1:
        return datos.copy()

    kernel = np.ones(ventana) / ventana

    resultado = np.empty_like(
        datos,
        dtype=np.float64,
    )

    if datos.ndim == 1:
        return np.convolve(
            datos,
            kernel,
            mode="same",
        )

    for indice in np.ndindex(datos.shape[:-1]):
        resultado[indice] = np.convolve(
            datos[indice],
            kernel,
            mode="same",
        )

    return resultado


# =============================================================================
# CARGA Y VALIDACION
# =============================================================================

def encontrar_archivos():
    """Localiza y ordena todos los archivos NPY."""

    archivos = sorted(
        CARPETA_DATOS.glob(PATRON_ARCHIVOS)
    )

    if len(archivos) == 0:
        raise FileNotFoundError(
            "No se encontraron archivos NPY en:\n"
            f"{CARPETA_DATOS}"
        )

    print("\n" + "=" * 79)
    print("ARCHIVOS ENCONTRADOS")
    print("=" * 79)

    for indice, archivo in enumerate(archivos, start=1):
        print(f"{indice:02d}. {archivo.name}")

    print(f"\nTotal de archivos: {len(archivos)}")

    if len(archivos) != 4:
        warnings.warn(
            "Se esperaban cuatro ensayos, pero se encontraron "
            f"{len(archivos)} archivos."
        )

    return archivos


def validar_ensayo(datos, archivo):
    """Valida dimensiones y contenido de un ensayo."""

    datos = np.asarray(datos)

    if datos.ndim != 2:
        raise ValueError(
            f"{archivo.name}: se esperaba una matriz bidimensional, "
            f"pero se obtuvo {datos.ndim} dimensiones."
        )

    forma_esperada = (
        N_CANALES_ESPERADOS,
        N_MUESTRAS_ESPERADAS,
    )

    if datos.shape != forma_esperada:
        raise ValueError(
            f"{archivo.name}: forma incorrecta {datos.shape}. "
            f"Se esperaba {forma_esperada}."
        )

    if not np.issubdtype(datos.dtype, np.number):
        raise TypeError(
            f"{archivo.name}: los datos no son numericos."
        )

    if np.isnan(datos).any():
        raise ValueError(
            f"{archivo.name}: contiene valores NaN."
        )

    if np.isinf(datos).any():
        raise ValueError(
            f"{archivo.name}: contiene valores infinitos."
        )

    duracion = datos.shape[1] / FS

    print(
        f"\n[OK] {archivo.name}\n"
        f"     Forma: {datos.shape}\n"
        f"     Tipo: {datos.dtype}\n"
        f"     Duracion: {duracion:.3f} s\n"
        f"     Minimo: {np.min(datos):.6f} microV\n"
        f"     Maximo: {np.max(datos):.6f} microV\n"
        f"     Media: {np.mean(datos):.6f} microV\n"
        f"     Desviacion estandar: {np.std(datos):.6f} microV"
    )


def cargar_todos_los_ensayos(archivos):
    """Carga todos los archivos y forma un arreglo ensayo-canal-tiempo."""

    lista_ensayos = []

    for archivo in archivos:
        datos = cargar_npy_eeg(
            str(archivo),
            n_canales_esperados=N_CANALES_ESPERADOS,
        )

        validar_ensayo(
            datos,
            archivo,
        )

        lista_ensayos.append(
            np.asarray(
                datos,
                dtype=np.float64,
            )
        )

    ensayos = np.stack(
        lista_ensayos,
        axis=0,
    )

    print("\n" + "=" * 79)
    print("BASE DE DATOS CARGADA")
    print("=" * 79)
    print(
        "Forma general: "
        f"{ensayos.shape} "
        "(ensayos, canales, muestras)"
    )

    return ensayos


# =============================================================================
# PREPROCESAMIENTO
# =============================================================================

def preprocesar_ensayo(datos):
    """Aplica DC, notch, pasa banda y referencia opcional."""

    procesados = np.array(
        datos,
        dtype=np.float64,
        copy=True,
    )

    if ELIMINAR_DC:
        procesados = quitar_componente_dc(
            procesados
        )

    procesados = aplicar_notch_multiple(
        procesados,
        fs=FS,
        freqs=FREQS_NOTCH,
        Q=Q_NOTCH,
        pad=PAD_MUESTRAS,
    )

    procesados = aplicar_pasabanda_butter(
        procesados,
        fs=FS,
        f_low=F_LOW,
        f_high=F_HIGH,
        orden=BP_ORDER,
        pad=PAD_MUESTRAS,
    )

    if USAR_REFERENCIA_PROMEDIO:
        procesados = aplicar_referencia_promedio(
            procesados
        )

    return procesados


def aplicar_ica_conjunta(ensayos):
    """
    Ajusta ICA sobre los cuatro ensayos concatenados.

    MNE trabaja internamente en voltios. Debido a que los archivos estan
    almacenados en microvoltios, los datos se convierten momentaneamente
    a voltios y posteriormente regresan a microvoltios.
    """

    if not USAR_ICA:
        print("\n[ICA] Desactivada.")
        return ensayos

    if IC_TO_REMOVE is None:
        componentes_excluir = []
    elif isinstance(IC_TO_REMOVE, int):
        componentes_excluir = [IC_TO_REMOVE]
    else:
        componentes_excluir = list(IC_TO_REMOVE)

    if len(componentes_excluir) == 0:
        warnings.warn(
            "ICA esta activada, pero no se selecciono ninguna "
            "componente para excluir."
        )

    n_ensayos, n_canales, n_muestras = ensayos.shape

    # Concatenacion conservando el orden ensayo por ensayo
    concatenados_microv = np.concatenate(
        ensayos,
        axis=-1,
    )

    concatenados_v = concatenados_microv * 1e-6

    info = mne.create_info(
        ch_names=CANALES,
        sfreq=FS,
        ch_types=["eeg"] * len(CANALES),
    )

    raw = mne.io.RawArray(
        concatenados_v,
        info,
        verbose=False,
    )

    rank = mne.compute_rank(
        raw,
        verbose=False,
    )

    n_componentes = min(
        rank.get("eeg", n_canales),
        n_canales,
    )

    ica = ICA(
        n_components=n_componentes,
        method=ICA_METHOD,
        fit_params=dict(extended=True),
        random_state=ICA_RANDOM_STATE,
        max_iter=ICA_MAX_ITER,
    )

    print("\n[ICA] Ajustando ICA conjunta...")
    print(f"[ICA] Componentes disponibles: {n_componentes}")
    print(f"[ICA] Componentes a excluir: {componentes_excluir}")

    ica.fit(
        raw,
        verbose=False,
    )

    for componente in componentes_excluir:
        if componente < 0 or componente >= n_componentes:
            raise ValueError(
                f"La componente IC{componente} no existe. "
                f"Las componentes disponibles son IC0 a "
                f"IC{n_componentes - 1}."
            )

    ica.exclude = componentes_excluir

    raw_limpio = raw.copy()

    ica.apply(
        raw_limpio,
        verbose=False,
    )

    limpios_microv = (
        raw_limpio.get_data() * 1e6
    )

    ensayos_limpios = np.stack(
        [
            limpios_microv[
                :,
                i * n_muestras:(i + 1) * n_muestras
            ]
            for i in range(n_ensayos)
        ],
        axis=0,
    )

    print("[ICA] Reconstruccion finalizada.")

    return ensayos_limpios


def preprocesar_todos_los_ensayos(ensayos):
    """Preprocesa cada ensayo y aplica ICA opcional."""

    preprocesados = []

    print("\n" + "=" * 79)
    print("PREPROCESAMIENTO")
    print("=" * 79)

    for indice, ensayo in enumerate(ensayos, start=1):
        print(
            f"Procesando ensayo {indice}/{len(ensayos)}..."
        )

        ensayo_filtrado = preprocesar_ensayo(
            ensayo
        )

        preprocesados.append(
            ensayo_filtrado
        )

    preprocesados = np.stack(
        preprocesados,
        axis=0,
    )

    preprocesados = aplicar_ica_conjunta(
        preprocesados
    )

    print(
        "\n[OK] Forma de datos preprocesados: "
        f"{preprocesados.shape}"
    )

    return preprocesados


# =============================================================================
# VISUALIZACION DE SEÑALES PREPROCESADAS
# =============================================================================

def graficar_ensayos_preprocesados(ensayos):
    """Genera una figura de EEG por cada ensayo."""

    tiempo = np.arange(
        ensayos.shape[-1]
    ) / FS

    for indice_ensayo, ensayo in enumerate(ensayos):
        fig, axes = plt.subplots(
            len(CANALES),
            1,
            figsize=(16, 11),
            sharex=True,
            constrained_layout=True,
        )

        fig.suptitle(
            f"EEG preprocesado - Ensayo {indice_ensayo + 1}",
            fontsize=16,
            fontweight="bold",
        )

        limite = np.percentile(
            np.abs(ensayo),
            99.5,
        )

        for indice_canal, canal in enumerate(CANALES):
            ax = axes[indice_canal]

            ax.plot(
                tiempo,
                ensayo[indice_canal],
                color=COLORES_CANALES[canal],
                linewidth=0.8,
            )

            ax.axvspan(
                0.0,
                3.0,
                color="#8ecae6",
                alpha=0.12,
            )

            ax.axvspan(
                3.0,
                18.0,
                color="#ffb703",
                alpha=0.10,
            )

            ax.axvspan(
                18.0,
                21.0,
                color="#8ecae6",
                alpha=0.12,
            )

            ax.axvline(
                3.0,
                color="gray",
                linestyle="--",
                linewidth=1.0,
            )

            ax.axvline(
                18.0,
                color="gray",
                linestyle="--",
                linewidth=1.0,
            )

            ax.axhline(
                0.0,
                color="gray",
                linewidth=0.5,
                alpha=0.6,
            )

            ax.set_ylabel(
                canal,
                fontweight="bold",
            )

            ax.set_ylim(
                -limite,
                limite,
            )

            ax.grid(
                True,
                linestyle="--",
                linewidth=0.4,
                alpha=0.4,
            )

        axes[-1].set_xlabel(
            "Tiempo (s)"
        )

        axes[-1].xaxis.set_major_locator(
            ticker.MultipleLocator(3.0)
        )

        guardar_figura(
            fig,
            f"01_EEG_preprocesado_ensayo_{indice_ensayo + 1}",
        )


# =============================================================================
# ERSP MEDIANTE MORLET
# =============================================================================

def calcular_ersp_morlet(ensayos):
    """
    Calcula potencia tiempo-frecuencia por wavelet de Morlet.

    Entrada:
        ensayos:
            Forma (ensayos, canales, muestras).

    Salida:
        potencia:
            Potencia individual.
            Forma (ensayos, canales, frecuencias, tiempos).

        ersp_db:
            ERSP calculado como cociente entre la potencia promedio de
            los ensayos y la potencia basal promedio.

        tiempo_tfr:
            Vector temporal despues del diezmado.
    """

    print("\n" + "=" * 79)
    print("CALCULO DEL ERSP CON WAVELETS DE MORLET")
    print("=" * 79)

    potencia = tfr_array_morlet(
        ensayos,
        sfreq=FS,
        freqs=FREQS_MORLET,
        n_cycles=N_CYCLES_MORLET,
        output="power",
        use_fft=True,
        decim=MORLET_DECIM,
        n_jobs=N_JOBS,
        zero_mean=True,
        verbose=False,
    )

    tiempo_tfr = (
        np.arange(potencia.shape[-1])
        * MORLET_DECIM
        / FS
    )

    mascara_baseline = intervalo_a_mascara(
        tiempo_tfr,
        BASELINE,
    )

    if not np.any(mascara_baseline):
        raise ValueError(
            "El intervalo baseline no contiene puntos temporales "
            "en el resultado de Morlet."
        )

    # SPEC(f,t): promedio de potencia entre ensayos
    potencia_promedio = np.mean(
        potencia,
        axis=0,
    )

    # mu_pre(f): promedio sobre ensayos y tiempos de baseline
    potencia_baseline = np.mean(
        potencia[..., mascara_baseline],
        axis=(0, 3),
    )

    # Dimensiones:
    # potencia_promedio: canales, frecuencias, tiempos
    # potencia_baseline: canales, frecuencias
    epsilon = np.finfo(np.float64).tiny

    ersp_db = 10.0 * np.log10(
        (potencia_promedio + epsilon)
        / (potencia_baseline[..., np.newaxis] + epsilon)
    )

    # ERSP individual para revisar variabilidad entre ensayos
    baseline_individual = np.mean(
        potencia[..., mascara_baseline],
        axis=-1,
        keepdims=True,
    )

    ersp_individual_db = 10.0 * np.log10(
        (potencia + epsilon)
        / (baseline_individual + epsilon)
    )

    print(
        "[OK] Potencia Morlet: "
        f"{potencia.shape}"
    )

    print(
        "[OK] ERSP promedio: "
        f"{ersp_db.shape}"
    )

    return (
        potencia,
        ersp_db,
        ersp_individual_db,
        tiempo_tfr,
    )


def obtener_limite_ersp(ersp_db):
    """Obtiene un limite simetrico robusto para los mapas ERSP."""

    if LIMITE_ERSP_DB is not None:
        return float(LIMITE_ERSP_DB)

    limite = np.percentile(
        np.abs(ersp_db),
        98.0,
    )

    limite = max(
        1.0,
        float(limite),
    )

    return limite


def graficar_ersp_promedio(ersp_db, tiempo_tfr):
    """Genera los siete mapas ERSP promedio."""

    limite = obtener_limite_ersp(
        ersp_db
    )

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(17, 15),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    axes = axes.ravel()

    fig.suptitle(
        "ERSP promedio de los cuatro ensayos",
        fontsize=18,
        fontweight="bold",
    )

    imagen = None

    for indice_canal, canal in enumerate(CANALES):
        ax = axes[indice_canal]

        imagen = ax.pcolormesh(
            tiempo_tfr,
            FREQS_MORLET,
            ersp_db[indice_canal],
            shading="auto",
            cmap="RdBu_r",
            vmin=-limite,
            vmax=limite,
        )

        ax.axvline(
            3.0,
            color="black",
            linestyle="--",
            linewidth=1.3,
        )

        ax.axvline(
            18.0,
            color="black",
            linestyle="--",
            linewidth=1.3,
        )

        ax.axhline(
            8.0,
            color="gray",
            linestyle=":",
            linewidth=0.8,
        )

        ax.axhline(
            12.0,
            color="gray",
            linestyle=":",
            linewidth=0.8,
        )

        ax.axhline(
            30.0,
            color="gray",
            linestyle=":",
            linewidth=0.8,
        )

        ax.set_title(
            canal,
            fontsize=14,
            fontweight="bold",
        )

        ax.set_xlim(
            0.0,
            21.0,
        )

        ax.set_ylim(
            4.0,
            45.0,
        )

        ax.set_xlabel(
            "Tiempo (s)"
        )

        ax.set_ylabel(
            "Frecuencia (Hz)"
        )

    axes[-1].axis("off")

    barra = fig.colorbar(
        imagen,
        ax=axes[:-1],
        location="right",
        shrink=0.85,
        pad=0.02,
    )

    barra.set_label(
        "ERSP (dB)",
        fontsize=13,
        fontweight="bold",
    )

    guardar_figura(
        fig,
        "02_ERSP_Morlet_promedio_todos_los_canales",
    )


def graficar_ersp_canal_individual(
    ersp_individual_db,
    tiempo_tfr,
    canal_objetivo,
):
    """Compara los cuatro ERSP individuales de un canal."""

    indice_canal = CANALES.index(
        canal_objetivo
    )

    datos = ersp_individual_db[
        :,
        indice_canal,
        :,
        :,
    ]

    limite = obtener_limite_ersp(
        datos
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 10),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    axes = axes.ravel()

    fig.suptitle(
        f"ERSP individual por ensayo - Canal {canal_objetivo}",
        fontsize=17,
        fontweight="bold",
    )

    imagen = None

    for indice_ensayo, ax in enumerate(axes):
        imagen = ax.pcolormesh(
            tiempo_tfr,
            FREQS_MORLET,
            datos[indice_ensayo],
            shading="auto",
            cmap="RdBu_r",
            vmin=-limite,
            vmax=limite,
        )

        ax.axvline(
            3.0,
            color="black",
            linestyle="--",
        )

        ax.axvline(
            18.0,
            color="black",
            linestyle="--",
        )

        ax.set_title(
            f"Ensayo {indice_ensayo + 1}",
            fontweight="bold",
        )

        ax.set_xlabel(
            "Tiempo (s)"
        )

        ax.set_ylabel(
            "Frecuencia (Hz)"
        )

        ax.set_xlim(
            0.0,
            21.0,
        )

        ax.set_ylim(
            4.0,
            45.0,
        )

    barra = fig.colorbar(
        imagen,
        ax=axes,
        shrink=0.85,
    )

    barra.set_label(
        "ERSP (dB)"
    )

    guardar_figura(
        fig,
        f"03_ERSP_individual_{canal_objetivo}",
    )


# =============================================================================
# FFT DESLIZANTE Y ERD/ERS PORCENTUAL
# =============================================================================

def calcular_potencia_fft(ensayos):
    """
    Calcula potencia espectral mediante FFT con ventana Hann deslizante.

    Se utiliza una ventana de 0.4 s, equivalente a 100 muestras a 250 Hz.
    El paso temporal es de 0.1 s.
    """

    print("\n" + "=" * 79)
    print("CALCULO DE POTENCIA CON FFT DESLIZANTE")
    print("=" * 79)

    ventana_hann = signal.windows.hann(
        FFT_NPERSEG,
        sym=False,
    )

    frecuencias, tiempo_fft, zxx = signal.stft(
        ensayos,
        fs=FS,
        window=ventana_hann,
        nperseg=FFT_NPERSEG,
        noverlap=FFT_NOVERLAP,
        nfft=FFT_NFFT,
        detrend=False,
        return_onesided=True,
        boundary=None,
        padded=False,
        axis=-1,
    )

    potencia = np.abs(zxx) ** 2

    mascara_frecuencia = (
        (frecuencias >= 4.0)
        & (frecuencias <= 45.0)
    )

    frecuencias = frecuencias[
        mascara_frecuencia
    ]

    potencia = potencia[
        ...,
        mascara_frecuencia,
        :,
    ]

    print(
        f"[OK] Ventana FFT: {FFT_NPERSEG} muestras "
        f"({FFT_NPERSEG / FS:.3f} s)"
    )

    print(
        f"[OK] Paso temporal: {FFT_PASO_MUESTRAS} muestras "
        f"({FFT_PASO_MUESTRAS / FS:.3f} s)"
    )

    print(
        f"[OK] Resolucion frecuencial: "
        f"{FS / FFT_NFFT:.3f} Hz"
    )

    print(
        f"[OK] Potencia FFT: {potencia.shape}"
    )

    return frecuencias, tiempo_fft, potencia


def calcular_erd_ers_porcentual(
    frecuencias,
    tiempo_fft,
    potencia,
):
    """
    Calcula ERD/ERS porcentual con el modelo de ganancia.

    Primero se calcula la potencia promedio entre ensayos:

        SPEC(f,t) = promedio_k P_k(f,t)

    Despues se calcula la potencia basal usando todos los ensayos y
    todos los puntos temporales de la linea base:

        mu_pre(f) = promedio_{k,t en baseline} P_k(f,t)

    Finalmente:

        cambio_porcentual(f,t) =
            100 * (SPEC(f,t) / mu_pre(f) - 1)

    El resultado por banda es el promedio de los bins incluidos.
    """

    mascara_baseline = intervalo_a_mascara(
        tiempo_fft,
        BASELINE,
    )

    if not np.any(mascara_baseline):
        raise ValueError(
            "No existen ventanas FFT dentro del intervalo baseline."
        )

    potencia_promedio = np.mean(
        potencia,
        axis=0,
    )

    potencia_baseline = np.mean(
        potencia[..., mascara_baseline],
        axis=(0, 3),
    )

    epsilon = np.finfo(np.float64).tiny

    razon_potencia = (
        potencia_promedio
        / (
            potencia_baseline[..., np.newaxis]
            + epsilon
        )
    )

    cambio_porcentual_frecuencia = (
        100.0
        * (razon_potencia - 1.0)
    )

    # Calculo individual de cada ensayo
    potencia_baseline_individual = np.mean(
        potencia[..., mascara_baseline],
        axis=-1,
        keepdims=True,
    )

    razon_individual = (
        potencia
        / (
            potencia_baseline_individual
            + epsilon
        )
    )

    cambio_individual_frecuencia = (
        100.0
        * (razon_individual - 1.0)
    )

    erd_ers_bandas = {}
    erd_ers_individual_bandas = {}
    bins_por_banda = {}

    nombres_bandas = list(BANDAS.keys())

    for indice_banda, nombre_banda in enumerate(nombres_bandas):
        f_inicio, f_fin = BANDAS[nombre_banda]

        es_ultima_banda = (
            indice_banda == len(nombres_bandas) - 1
        )

        if es_ultima_banda:
            mascara_banda = (
                (frecuencias >= f_inicio)
                & (frecuencias <= f_fin)
            )
        else:
            mascara_banda = (
                (frecuencias >= f_inicio)
                & (frecuencias < f_fin)
            )

        if not np.any(mascara_banda):
            raise ValueError(
                f"No existen bins FFT dentro de la banda "
                f"{nombre_banda}: {f_inicio}-{f_fin} Hz."
            )

        bins_banda = frecuencias[
            mascara_banda
        ]

        bins_por_banda[nombre_banda] = bins_banda

        erd_ers_bandas[nombre_banda] = np.mean(
            cambio_porcentual_frecuencia[
                :,
                mascara_banda,
                :,
            ],
            axis=1,
        )

        erd_ers_individual_bandas[nombre_banda] = np.mean(
            cambio_individual_frecuencia[
                :,
                :,
                mascara_banda,
                :,
            ],
            axis=2,
        )

    print("\nBins incluidos por banda:")

    for nombre, bins in bins_por_banda.items():
        print(
            f"  {nombre:7s}: "
            f"{np.array2string(bins, precision=1)} Hz"
        )

    return (
        erd_ers_bandas,
        erd_ers_individual_bandas,
        bins_por_banda,
    )


# =============================================================================
# GRAFICAS ERD/ERS PORCENTUAL
# =============================================================================

def sombrear_protocolo(ax):
    """Añade las regiones temporales del protocolo."""

    ax.axvspan(
        0.0,
        3.0,
        color="#8ecae6",
        alpha=0.12,
        label="Reposo",
    )

    ax.axvspan(
        3.0,
        18.0,
        color="#ffb703",
        alpha=0.10,
        label="Contraccion",
    )

    ax.axvspan(
        18.0,
        21.0,
        color="#8ecae6",
        alpha=0.12,
    )

    ax.axvline(
        3.0,
        color="gray",
        linestyle="--",
        linewidth=1.0,
    )

    ax.axvline(
        18.0,
        color="gray",
        linestyle="--",
        linewidth=1.0,
    )

    ax.axhline(
        0.0,
        color="black" if not ESTILO_OSCURO else "white",
        linestyle="-",
        linewidth=0.8,
        alpha=0.8,
    )


def preparar_curva_visual(curva):
    """Aplica suavizado opcional solamente para la representacion."""

    if not APLICAR_SUAVIZADO_CURVAS:
        return curva

    ventana = max(
        1,
        int(
            round(
                SUAVIZADO_SEGUNDOS
                / FFT_PASO_SEGUNDOS
            )
        ),
    )

    return media_movil(
        curva,
        ventana,
    )


def graficar_erd_ers_todos_canales(
    tiempo_fft,
    erd_ers_bandas,
):
    """Genera una figura por banda con los siete canales."""

    for banda in BANDAS_PRINCIPALES:
        fig, ax = plt.subplots(
            figsize=(16, 8),
            constrained_layout=True,
        )

        sombrear_protocolo(
            ax
        )

        for indice_canal, canal in enumerate(CANALES):
            curva = erd_ers_bandas[banda][
                indice_canal
            ]

            curva_plot = preparar_curva_visual(
                curva
            )

            ax.plot(
                tiempo_fft,
                curva_plot,
                color=COLORES_CANALES[canal],
                linewidth=1.6,
                label=canal,
            )

        f_inicio, f_fin = BANDAS[banda]

        ax.set_title(
            f"ERD/ERS porcentual - Banda {banda} "
            f"({f_inicio:g}-{f_fin:g} Hz)",
            fontsize=17,
            fontweight="bold",
        )

        ax.set_xlabel(
            "Tiempo (s)",
            fontsize=13,
        )

        ax.set_ylabel(
            "Cambio de potencia respecto a baseline (%)",
            fontsize=13,
        )

        ax.set_xlim(
            0.0,
            21.0,
        )

        ax.xaxis.set_major_locator(
            ticker.MultipleLocator(3.0)
        )

        ax.grid(
            True,
            linestyle="--",
            linewidth=0.5,
            alpha=0.4,
        )

        ax.legend(
            ncol=4,
            loc="upper right",
        )

        guardar_figura(
            fig,
            f"04_ERD_ERS_porcentual_{banda}_todos_canales",
        )


def graficar_comparaciones_contralaterales(
    tiempo_fft,
    erd_ers_bandas,
):
    """Compara C3-C4 y FC3-FC4 para cada banda principal."""

    pares = [
        ("C3", "C4"),
        ("FC3", "FC4"),
    ]

    for banda in BANDAS_PRINCIPALES:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(16, 10),
            sharex=True,
            constrained_layout=True,
        )

        fig.suptitle(
            f"Comparacion contralateral - Banda {banda}",
            fontsize=17,
            fontweight="bold",
        )

        for ax, par in zip(axes, pares):
            sombrear_protocolo(
                ax
            )

            for canal in par:
                indice_canal = CANALES.index(
                    canal
                )

                curva = erd_ers_bandas[banda][
                    indice_canal
                ]

                curva_plot = preparar_curva_visual(
                    curva
                )

                ax.plot(
                    tiempo_fft,
                    curva_plot,
                    color=COLORES_CANALES[canal],
                    linewidth=2.0,
                    label=canal,
                )

            ax.set_title(
                f"{par[0]} frente a {par[1]}",
                fontweight="bold",
            )

            ax.set_ylabel(
                "ERD/ERS (%)"
            )

            ax.grid(
                True,
                linestyle="--",
                linewidth=0.5,
                alpha=0.4,
            )

            ax.legend(
                loc="upper right"
            )

        axes[-1].set_xlabel(
            "Tiempo (s)"
        )

        axes[-1].set_xlim(
            0.0,
            21.0,
        )

        axes[-1].xaxis.set_major_locator(
            ticker.MultipleLocator(3.0)
        )

        guardar_figura(
            fig,
            f"05_Comparacion_contralateral_{banda}",
        )


def graficar_bandas_por_canal(
    tiempo_fft,
    erd_ers_bandas,
):
    """Genera una figura con mu, beta y gamma para cada canal."""

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(17, 15),
        sharex=True,
        constrained_layout=True,
    )

    axes = axes.ravel()

    fig.suptitle(
        "ERD/ERS porcentual por canal y banda",
        fontsize=18,
        fontweight="bold",
    )

    for indice_canal, canal in enumerate(CANALES):
        ax = axes[indice_canal]

        sombrear_protocolo(
            ax
        )

        for banda in BANDAS_PRINCIPALES:
            curva = erd_ers_bandas[banda][
                indice_canal
            ]

            curva_plot = preparar_curva_visual(
                curva
            )

            ax.plot(
                tiempo_fft,
                curva_plot,
                color=COLORES_BANDAS[banda],
                linewidth=1.6,
                label=banda,
            )

        ax.set_title(
            canal,
            fontsize=14,
            fontweight="bold",
        )

        ax.set_ylabel(
            "ERD/ERS (%)"
        )

        ax.set_xlim(
            0.0,
            21.0,
        )

        ax.grid(
            True,
            linestyle="--",
            linewidth=0.5,
            alpha=0.4,
        )

        ax.legend(
            loc="upper right"
        )

    axes[-1].axis("off")

    for ax in axes[:-1]:
        ax.set_xlabel(
            "Tiempo (s)"
        )

    guardar_figura(
        fig,
        "06_Bandas_por_canal",
    )


def graficar_variabilidad_ensayos(
    tiempo_fft,
    erd_ers_individual_bandas,
    canal_objetivo="C3",
):
    """Representa los cuatro ensayos y su promedio para un canal."""

    indice_canal = CANALES.index(
        canal_objetivo
    )

    fig, axes = plt.subplots(
        len(BANDAS_PRINCIPALES),
        1,
        figsize=(16, 12),
        sharex=True,
        constrained_layout=True,
    )

    fig.suptitle(
        f"Variabilidad entre ensayos - Canal {canal_objetivo}",
        fontsize=17,
        fontweight="bold",
    )

    for ax, banda in zip(axes, BANDAS_PRINCIPALES):
        sombrear_protocolo(
            ax
        )

        datos = erd_ers_individual_bandas[banda][
            :,
            indice_canal,
            :,
        ]

        for indice_ensayo in range(datos.shape[0]):
            curva = preparar_curva_visual(
                datos[indice_ensayo]
            )

            ax.plot(
                tiempo_fft,
                curva,
                linewidth=1.0,
                alpha=0.45,
                label=f"Ensayo {indice_ensayo + 1}",
            )

        promedio_individual = np.mean(
            datos,
            axis=0,
        )

        promedio_plot = preparar_curva_visual(
            promedio_individual
        )

        ax.plot(
            tiempo_fft,
            promedio_plot,
            color="#d00000",
            linewidth=2.5,
            label="Promedio",
        )

        ax.set_title(
            banda,
            fontweight="bold",
        )

        ax.set_ylabel(
            "ERD/ERS (%)"
        )

        ax.grid(
            True,
            linestyle="--",
            linewidth=0.5,
            alpha=0.4,
        )

        ax.legend(
            ncol=3,
            loc="upper right",
        )

    axes[-1].set_xlabel(
        "Tiempo (s)"
    )

    axes[-1].set_xlim(
        0.0,
        21.0,
    )

    guardar_figura(
        fig,
        f"07_Variabilidad_ensayos_{canal_objetivo}",
    )


# =============================================================================
# RESULTADOS NUMERICOS
# =============================================================================

def extraer_metricas_curva(
    tiempo,
    curva,
    intervalo,
):
    """Extrae promedio, mediana, minimo y maximo dentro de un intervalo."""

    mascara = intervalo_a_mascara(
        tiempo,
        intervalo,
    )

    valores = curva[
        mascara
    ]

    tiempos_intervalo = tiempo[
        mascara
    ]

    if valores.size == 0:
        return {
            "promedio": np.nan,
            "mediana": np.nan,
            "minimo": np.nan,
            "tiempo_minimo": np.nan,
            "maximo": np.nan,
            "tiempo_maximo": np.nan,
            "desviacion_estandar": np.nan,
        }

    indice_minimo = np.argmin(
        valores
    )

    indice_maximo = np.argmax(
        valores
    )

    return {
        "promedio": float(np.mean(valores)),
        "mediana": float(np.median(valores)),
        "minimo": float(valores[indice_minimo]),
        "tiempo_minimo": float(
            tiempos_intervalo[indice_minimo]
        ),
        "maximo": float(valores[indice_maximo]),
        "tiempo_maximo": float(
            tiempos_intervalo[indice_maximo]
        ),
        "desviacion_estandar": float(
            np.std(valores)
        ),
    }


def generar_tabla_promedio(
    tiempo_fft,
    erd_ers_bandas,
):
    """Genera métricas de las curvas obtenidas del promedio espectral."""

    filas = []

    intervalos = {
        "Baseline efectivo": BASELINE,
        "Movimiento completo efectivo": MOVIMIENTO,
        **FASES_MOVIMIENTO,
        "Reposo posterior": REPOSO_POSTERIOR,
    }

    for banda, datos_banda in erd_ers_bandas.items():
        for indice_canal, canal in enumerate(CANALES):
            curva = datos_banda[
                indice_canal
            ]

            for nombre_intervalo, intervalo in intervalos.items():
                metricas = extraer_metricas_curva(
                    tiempo_fft,
                    curva,
                    intervalo,
                )

                filas.append(
                    {
                        "Sujeto": "SV1",
                        "Canal": canal,
                        "Banda": banda,
                        "Intervalo": nombre_intervalo,
                        "Inicio_s": intervalo[0],
                        "Fin_s": intervalo[1],
                        "Promedio_porcentaje": metricas["promedio"],
                        "Mediana_porcentaje": metricas["mediana"],
                        "Minimo_porcentaje": metricas["minimo"],
                        "Tiempo_minimo_s": metricas["tiempo_minimo"],
                        "Maximo_porcentaje": metricas["maximo"],
                        "Tiempo_maximo_s": metricas["tiempo_maximo"],
                        "Desviacion_estandar": metricas[
                            "desviacion_estandar"
                        ],
                    }
                )

    return pd.DataFrame(
        filas
    )


def generar_tabla_individual(
    tiempo_fft,
    erd_ers_individual_bandas,
    archivos,
):
    """Genera métricas por ensayo, canal, banda e intervalo."""

    filas = []

    intervalos = {
        "Movimiento completo efectivo": MOVIMIENTO,
        **FASES_MOVIMIENTO,
        "Reposo posterior": REPOSO_POSTERIOR,
    }

    for banda, datos_banda in erd_ers_individual_bandas.items():
        for indice_ensayo in range(datos_banda.shape[0]):
            for indice_canal, canal in enumerate(CANALES):
                curva = datos_banda[
                    indice_ensayo,
                    indice_canal,
                ]

                for nombre_intervalo, intervalo in intervalos.items():
                    metricas = extraer_metricas_curva(
                        tiempo_fft,
                        curva,
                        intervalo,
                    )

                    filas.append(
                        {
                            "Sujeto": "SV1",
                            "Ensayo": indice_ensayo + 1,
                            "Archivo": archivos[
                                indice_ensayo
                            ].name,
                            "Canal": canal,
                            "Banda": banda,
                            "Intervalo": nombre_intervalo,
                            "Inicio_s": intervalo[0],
                            "Fin_s": intervalo[1],
                            "Promedio_porcentaje": metricas[
                                "promedio"
                            ],
                            "Mediana_porcentaje": metricas[
                                "mediana"
                            ],
                            "Minimo_porcentaje": metricas[
                                "minimo"
                            ],
                            "Tiempo_minimo_s": metricas[
                                "tiempo_minimo"
                            ],
                            "Maximo_porcentaje": metricas[
                                "maximo"
                            ],
                            "Tiempo_maximo_s": metricas[
                                "tiempo_maximo"
                            ],
                            "Desviacion_estandar": metricas[
                                "desviacion_estandar"
                            ],
                        }
                    )

    return pd.DataFrame(
        filas
    )


def guardar_resultados_numericos(
    tabla_promedio,
    tabla_individual,
    frecuencias_fft,
    tiempo_fft,
    erd_ers_bandas,
):
    """Exporta tablas y curvas completas a CSV."""

    if not GUARDAR_RESULTADOS:
        return

    ruta_promedio = (
        CARPETA_RESULTADOS
        / "Metricas_ERD_ERS_promedio.csv"
    )

    ruta_individual = (
        CARPETA_RESULTADOS
        / "Metricas_ERD_ERS_por_ensayo.csv"
    )

    tabla_promedio.to_csv(
        ruta_promedio,
        index=False,
        encoding="utf-8-sig",
    )

    tabla_individual.to_csv(
        ruta_individual,
        index=False,
        encoding="utf-8-sig",
    )

    filas_curvas = []

    for banda, matriz in erd_ers_bandas.items():
        for indice_canal, canal in enumerate(CANALES):
            for tiempo, valor in zip(
                tiempo_fft,
                matriz[indice_canal],
            ):
                filas_curvas.append(
                    {
                        "Sujeto": "SV1",
                        "Canal": canal,
                        "Banda": banda,
                        "Tiempo_s": tiempo,
                        "ERD_ERS_porcentaje": valor,
                    }
                )

    tabla_curvas = pd.DataFrame(
        filas_curvas
    )

    ruta_curvas = (
        CARPETA_RESULTADOS
        / "Curvas_ERD_ERS_completas.csv"
    )

    tabla_curvas.to_csv(
        ruta_curvas,
        index=False,
        encoding="utf-8-sig",
    )

    ruta_frecuencias = (
        CARPETA_RESULTADOS
        / "Frecuencias_FFT_utilizadas.csv"
    )

    pd.DataFrame(
        {"Frecuencia_Hz": frecuencias_fft}
    ).to_csv(
        ruta_frecuencias,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[GUARDADO] {ruta_promedio}")
    print(f"[GUARDADO] {ruta_individual}")
    print(f"[GUARDADO] {ruta_curvas}")
    print(f"[GUARDADO] {ruta_frecuencias}")


def imprimir_resumen_principal(tabla_promedio):
    """Imprime los resultados centrales de C3, C4, FC3 y FC4."""

    seleccionar = (
        tabla_promedio["Canal"].isin(
            ["C3", "C4", "FC3", "FC4"]
        )
        & tabla_promedio["Banda"].isin(
            BANDAS_PRINCIPALES
        )
        & (
            tabla_promedio["Intervalo"]
            == "Movimiento completo efectivo"
        )
    )

    resumen = tabla_promedio.loc[
        seleccionar,
        [
            "Canal",
            "Banda",
            "Promedio_porcentaje",
            "Minimo_porcentaje",
            "Maximo_porcentaje",
        ],
    ].copy()

    resumen = resumen.sort_values(
        ["Banda", "Canal"]
    )

    print("\n" + "=" * 79)
    print("RESUMEN DURANTE LA CONTRACCION SOSTENIDA")
    print("=" * 79)

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        140,
        "display.float_format",
        lambda valor: f"{valor:9.3f}",
    ):
        print(
            resumen.to_string(
                index=False
            )
        )


# =============================================================================
# GUARDADO DE MATRICES
# =============================================================================

def guardar_matrices_numpy(
    ersp_db,
    ersp_individual_db,
    tiempo_tfr,
    tiempo_fft,
    frecuencias_fft,
    erd_ers_bandas,
):
    """Guarda las matrices para analizarlas posteriormente sin recalcular."""

    if not GUARDAR_RESULTADOS:
        return

    np.save(
        CARPETA_RESULTADOS / "ERSP_promedio_db.npy",
        ersp_db,
    )

    np.save(
        CARPETA_RESULTADOS / "ERSP_individual_db.npy",
        ersp_individual_db,
    )

    np.save(
        CARPETA_RESULTADOS / "Tiempo_Morlet_s.npy",
        tiempo_tfr,
    )

    np.save(
        CARPETA_RESULTADOS / "Frecuencias_Morlet_Hz.npy",
        FREQS_MORLET,
    )

    np.save(
        CARPETA_RESULTADOS / "Tiempo_FFT_s.npy",
        tiempo_fft,
    )

    np.save(
        CARPETA_RESULTADOS / "Frecuencias_FFT_Hz.npy",
        frecuencias_fft,
    )

    for banda, matriz in erd_ers_bandas.items():
        nombre_seguro = banda.replace(
            " ",
            "_",
        )

        np.save(
            CARPETA_RESULTADOS
            / f"ERD_ERS_porcentual_{nombre_seguro}.npy",
            matriz,
        )

    print("[GUARDADO] Matrices NumPy de los resultados.")


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

def main():

    crear_carpeta_resultados()

    print("\n" + "=" * 79)
    print("ANALISIS ERSP Y ERD/ERS DE SV1")
    print("=" * 79)

    print(f"Carpeta: {CARPETA_DATOS}")
    print(f"Frecuencia de muestreo: {FS:g} Hz")
    print(f"Canales: {CANALES}")
    print(f"Unidades originales: {UNIDAD_DATOS}")
    print(f"Filtro pasa banda: {F_LOW:g}-{F_HIGH:g} Hz")
    print(f"Referencia promedio: {USAR_REFERENCIA_PROMEDIO}")
    print(f"ICA: {USAR_ICA}")
    print(f"Baseline efectivo: {BASELINE} s")
    print(f"Movimiento efectivo: {MOVIMIENTO} s")
    print(f"Reposo posterior: {REPOSO_POSTERIOR} s")

    # -------------------------------------------------------------------------
    # 1. Carga
    # -------------------------------------------------------------------------

    archivos = encontrar_archivos()

    ensayos_crudos = cargar_todos_los_ensayos(
        archivos
    )

    # -------------------------------------------------------------------------
    # 2. Preprocesamiento
    # -------------------------------------------------------------------------

    ensayos_preprocesados = preprocesar_todos_los_ensayos(
        ensayos_crudos
    )

    # -------------------------------------------------------------------------
    # 3. Señales preprocesadas
    # -------------------------------------------------------------------------

    graficar_ensayos_preprocesados(
        ensayos_preprocesados
    )

    # -------------------------------------------------------------------------
    # 4. ERSP Morlet
    # -------------------------------------------------------------------------

    (
        potencia_morlet,
        ersp_db,
        ersp_individual_db,
        tiempo_tfr,
    ) = calcular_ersp_morlet(
        ensayos_preprocesados
    )

    graficar_ersp_promedio(
        ersp_db,
        tiempo_tfr,
    )

    graficar_ersp_canal_individual(
        ersp_individual_db,
        tiempo_tfr,
        canal_objetivo="C3",
    )

    graficar_ersp_canal_individual(
        ersp_individual_db,
        tiempo_tfr,
        canal_objetivo="C4",
    )

    # -------------------------------------------------------------------------
    # 5. FFT y ERD/ERS porcentual
    # -------------------------------------------------------------------------

    (
        frecuencias_fft,
        tiempo_fft,
        potencia_fft,
    ) = calcular_potencia_fft(
        ensayos_preprocesados
    )

    (
        erd_ers_bandas,
        erd_ers_individual_bandas,
        bins_por_banda,
    ) = calcular_erd_ers_porcentual(
        frecuencias_fft,
        tiempo_fft,
        potencia_fft,
    )

    # -------------------------------------------------------------------------
    # 6. Graficas ERD/ERS porcentuales
    # -------------------------------------------------------------------------

    graficar_erd_ers_todos_canales(
        tiempo_fft,
        erd_ers_bandas,
    )

    graficar_comparaciones_contralaterales(
        tiempo_fft,
        erd_ers_bandas,
    )

    graficar_bandas_por_canal(
        tiempo_fft,
        erd_ers_bandas,
    )

    graficar_variabilidad_ensayos(
        tiempo_fft,
        erd_ers_individual_bandas,
        canal_objetivo="C3",
    )

    graficar_variabilidad_ensayos(
        tiempo_fft,
        erd_ers_individual_bandas,
        canal_objetivo="C4",
    )

    # -------------------------------------------------------------------------
    # 7. Tablas
    # -------------------------------------------------------------------------

    tabla_promedio = generar_tabla_promedio(
        tiempo_fft,
        erd_ers_bandas,
    )

    tabla_individual = generar_tabla_individual(
        tiempo_fft,
        erd_ers_individual_bandas,
        archivos,
    )

    imprimir_resumen_principal(
        tabla_promedio
    )

    guardar_resultados_numericos(
        tabla_promedio,
        tabla_individual,
        frecuencias_fft,
        tiempo_fft,
        erd_ers_bandas,
    )

    guardar_matrices_numpy(
        ersp_db,
        ersp_individual_db,
        tiempo_tfr,
        tiempo_fft,
        frecuencias_fft,
        erd_ers_bandas,
    )

    print("\n" + "=" * 79)
    print("ANALISIS FINALIZADO")
    print("=" * 79)

    if GUARDAR_RESULTADOS:
        print(
            "Resultados guardados en:\n"
            f"{CARPETA_RESULTADOS}"
        )

    cerrar_o_mostrar()


if __name__ == "__main__":
    main()