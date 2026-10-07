# -*- coding: utf-8 -*-

"""
ANALISIS ERSP (MORLET) Y ERD/ERS PORCENTUAL  -  V2
==================================================

Toma TODOS los archivos .npy de una carpeta (N ensayos, cualquier cantidad),
cada uno con forma (canales, muestras), y genera 4 figuras:

    1. ERSP Morlet promedio de los N ensayos (todos los canales).
    2. ERD/ERS porcentual de Mu     (un subplot por canal).
    3. ERD/ERS porcentual de Beta   (un subplot por canal).
    4. ERD/ERS porcentual de Gamma  (un subplot por canal).

Todas con disposicion 3x3 segun POSICIONES_CANALES.
No se genera ningun CSV ni .npy.

Los tiempos del protocolo (reposo / tarea / reposo) se editan abajo y se
aplican a todos los ensayos por igual.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scipy import signal
from scipy.ndimage import uniform_filter1d
from mne.time_frequency import tfr_array_morlet

from Procesamiento_Funciones import (
    cargar_npy_eeg,
    quitar_componente_dc,
    aplicar_notch_multiple,
    aplicar_pasabanda_butter,
    ica_eliminar_componente,
    pantalla_completa,
)


# =============================================================================
# CONFIGURACION: DATOS
# =============================================================================

plt.close("all")

CARPETA_DATOS = Path(
    r"C:\Users\lipsp\Documents\GitHub\Entrenamiento-BCI\Protocolo ERD_ERS\Signals"
    r"\SV1\MI"
)
PATRON_ARCHIVOS = "*.npy"          # se usan todos los que existan en la carpeta
CARPETA_RESULTADOS = CARPETA_DATOS / "Resultados_ERD_ERS"

FS = 250.0

# Orden de los canales tal como vienen en las filas de cada .npy
CANALES = ["Fz", "FC3", "FCz", "FC4", "Cz", "C3", "C4"]

# Posicion de cada canal en la malla 3x3: (fila, columna), empezando en 1
POSICIONES_CANALES = {
    "Fz":  (1, 2),
    "FC3": (2, 1), "FCz": (2, 2), "FC4": (2, 3),
    "C3":  (3, 1), "Cz":  (3, 2), "C4":  (3, 3),
}


# =============================================================================
# CONFIGURACION: TIEMPOS DEL PROTOCOLO (segundos, iguales para todos los ensayos)
# =============================================================================
# Deben ser consecutivos: el fin de un segmento = inicio del siguiente.
# Ejemplo alterno:  (0, 2) / (2, 5) / (5, 7)

T_REPOSO_INICIAL = (0.0, 3.0)
T_TAREA          = (3.0, 18.0)
T_REPOSO_FINAL   = (18.0, 21.0)

# El analisis termina en T_REPOSO_FINAL[1]. Si un archivo es mas largo se recorta
# ahi; si es mas corto se omite (con aviso).
T_FIN = T_REPOSO_FINAL[1]

# Baseline = reposo inicial menos un margen en cada extremo (evita bordes de
# filtros/wavelets). Puedes sobrescribirlo, p. ej.: BASELINE = (0.5, 2.5)
MARGEN_BASELINE = 0.5
BASELINE = (
    T_REPOSO_INICIAL[0] + MARGEN_BASELINE,
    T_REPOSO_INICIAL[1] - MARGEN_BASELINE,
)


# =============================================================================
# CONFIGURACION: BANDAS
# =============================================================================

BANDAS = {
    "Mu":    (8.0, 12.0),
    "Beta":  (12.0, 30.0),
    "Gamma": (30.0, 45.0),
}


# =============================================================================
# CONFIGURACION: PREPROCESAMIENTO
# =============================================================================

ELIMINAR_DC = True

FREQS_NOTCH = (60.0, 120.0)
Q_NOTCH = 5.0

F_LOW = 5.0
F_HIGH = 45.0
BP_ORDER = 3

PAD_MUESTRAS = int(5 * FS)         # padding reflejado (se limita al largo de la senal)

# ICA (usa ica_eliminar_componente de Procesamiento_Funciones; ajuste conjunto
# sobre los N ensayos concatenados). IC_TO_REMOVE: indice de la IC o None.
USAR_ICA = False
IC_TO_REMOVE = None
ICA_RANDOM_STATE = 97


# =============================================================================
# CONFIGURACION: ERSP CON MORLET  (zona para ir probando valores)
# =============================================================================

# FREQS_MORLET = np.arange(4.0, 46.0, 1.0)
FREQS_MORLET = np.arange(7.0, 32.0, 1.0)

# Ciclos por frecuencia. Opciones:
#   np.linspace(3.0, 10.0, len(FREQS_MORLET))   ciclos crecientes (actual)
#   5                                           ciclos fijos
#   FREQS_MORLET / 2                            wavelet de duracion constante
# N_CYCLES_MORLET = np.linspace(5, 8, len(FREQS_MORLET))
N_CYCLES_MORLET = np.linspace(6, 15, len(FREQS_MORLET))

MORLET_DECIM = 4                   # 250 Hz / 5 = 50 puntos por segundo
N_JOBS = 1

LIMITE_ERSP_DB = None              # p. ej. 3.0 para fijar +-3 dB; None = automatico
CMAP_ERSP = "jet"


# =============================================================================
# CONFIGURACION: FFT DESLIZANTE PARA ERD/ERS PORCENTUAL
# =============================================================================

FFT_DURACION_VENTANA = 0.8        # s  (resolucion en frecuencia = 1 / ventana) 100 muestras / 250 Hz = 0.4 segundos
FFT_PASO_SEGUNDOS = 0.2            # s
FFT_NPERSEG = int(FFT_DURACION_VENTANA * FS)
FFT_PASO_MUESTRAS = int(FFT_PASO_SEGUNDOS * FS)
FFT_NFFT = FFT_NPERSEG             # >FFT_NPERSEG interpola el espectro (zero-padding)

SUAVIZADO_SEGUNDOS = 0             # solo visual; 0 o None desactiva "MULTIPLO DE PASO"
LIMITES_ERD_ERS = None             # p. ej. (-100, 200) en %; None = automatico


# =============================================================================
# CONFIGURACION: FIGURAS
# =============================================================================

GUARDAR_FIGURAS = True
MOSTRAR_FIGURAS = True
MAXIMIZAR_FIGURAS = True

ESTILO_OSCURO = True
FIG_SIZE = (20, 12)
FIG_DPI = 300
FIG_FORMATO = "png"

COLORES_CANALES = {
    "Fz": "#EBEF91", "FC3": "#B5E48C", "FCz": "#99D98C", "FC4": "#6FC39C",
    "Cz": "#3DA2F5", "C3": "#64B5F7", "C4": "#248BE5",
}

COLOR_REPOSO = "#B7EBF6"
COLOR_TAREA = "#F5D6B7"

if ESTILO_OSCURO:
    plt.style.use("dark_background")

EPS = np.finfo(np.float64).tiny


# =============================================================================
# VALIDACION DE LA CONFIGURACION
# =============================================================================

def validar_configuracion():
    if not (T_REPOSO_INICIAL[1] == T_TAREA[0] and T_TAREA[1] == T_REPOSO_FINAL[0]):
        raise ValueError("Los segmentos del protocolo deben ser consecutivos.")
    if not (T_REPOSO_INICIAL[0] < T_REPOSO_INICIAL[1]
            and T_TAREA[0] < T_TAREA[1]
            and T_REPOSO_FINAL[0] < T_REPOSO_FINAL[1]):
        raise ValueError("Cada segmento del protocolo debe tener inicio < fin.")
    if not (T_REPOSO_INICIAL[0] <= BASELINE[0] < BASELINE[1] <= T_REPOSO_INICIAL[1]):
        raise ValueError(f"BASELINE {BASELINE} debe quedar dentro del reposo inicial.")
    faltantes = [c for c in CANALES if c not in POSICIONES_CANALES]
    if faltantes:
        raise ValueError(f"Canales sin posicion en POSICIONES_CANALES: {faltantes}")


# =============================================================================
# CARGA Y PREPROCESAMIENTO
# =============================================================================

def preprocesar_ensayo(datos):
    """DC -> notch -> pasa banda (un ensayo, forma canales x muestras)."""

    if ELIMINAR_DC:
        datos = quitar_componente_dc(datos)

    pad = min(PAD_MUESTRAS, datos.shape[1] - 1)

    datos = aplicar_notch_multiple(datos, FS, FREQS_NOTCH, Q_NOTCH, pad)
    datos = aplicar_pasabanda_butter(datos, FS, F_LOW, F_HIGH, BP_ORDER, pad)

    return datos


def cargar_ensayos():
    """Carga y preprocesa todos los .npy de la carpeta -> (ensayos, canales, muestras)."""

    archivos = sorted(CARPETA_DATOS.glob(PATRON_ARCHIVOS))

    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos {PATRON_ARCHIVOS} en:\n{CARPETA_DATOS}")

    n_fin = int(round(T_FIN * FS))
    ensayos = []

    print(f"Archivos encontrados: {len(archivos)}")

    for archivo in archivos:
        try:
            datos = cargar_npy_eeg(str(archivo), n_canales_esperados=len(CANALES))

            if not np.isfinite(datos).all():
                raise ValueError("contiene NaN o infinitos")
            if datos.shape[1] < n_fin:
                raise ValueError(
                    f"dura {datos.shape[1] / FS:.2f} s y el protocolo requiere {T_FIN:g} s"
                )
        except ValueError as error:
            print(f"  [OMITIDO] {archivo.name}: {error}")
            continue

        # Se filtra la senal completa y despues se recorta (menos efecto de borde)
        ensayos.append(preprocesar_ensayo(datos)[:, :n_fin])
        print(f"  [OK] {archivo.name}")

    if not ensayos:
        raise RuntimeError("Ningun archivo fue valido para el analisis.")

    return np.stack(ensayos, axis=0)


def aplicar_ica(ensayos):
    """ICA conjunta sobre los ensayos concatenados (MNE trabaja en voltios)."""

    if not USAR_ICA:
        return ensayos

    if IC_TO_REMOVE is None:
        print("[ICA] Activada sin componente a excluir (IC_TO_REMOVE = None).")

    concatenados_v = np.concatenate(ensayos, axis=-1) * 1e-6

    limpios_v, _ = ica_eliminar_componente(
        concatenados_v,
        fs=FS,
        ch_names=CANALES,
        ic_to_remove=IC_TO_REMOVE,
        random_state=ICA_RANDOM_STATE,
    )

    return np.stack(np.split(limpios_v * 1e6, len(ensayos), axis=-1), axis=0)


# =============================================================================
# CALCULOS
# =============================================================================

def intervalo_a_mascara(tiempo, intervalo):
    return (tiempo >= intervalo[0]) & (tiempo <= intervalo[1])


def promedio_con_baseline(ensayos, potencia_ensayo):
    """
    Promedia la potencia de los N ensayos y su baseline, ensayo por ensayo
    (la memoria no crece con N).

    Devuelve: potencia media (canales, frec, t), baseline (canales, frec), t.
    """

    suma = base = mascara = None

    for ensayo in ensayos:
        potencia, tiempo = potencia_ensayo(ensayo)

        if suma is None:
            mascara = intervalo_a_mascara(tiempo, BASELINE)
            if not mascara.any():
                raise ValueError("El baseline no contiene puntos temporales.")
            suma = np.zeros_like(potencia)
            base = np.zeros(potencia.shape[:-1])

        suma += potencia
        base += potencia[..., mascara].mean(axis=-1)

    n = len(ensayos)

    return suma / n, base / n, tiempo


def potencia_morlet(ensayo):
    potencia = tfr_array_morlet(
        ensayo[np.newaxis],
        sfreq=FS,
        freqs=FREQS_MORLET,
        n_cycles=N_CYCLES_MORLET,
        output="power",
        zero_mean=True,
        decim=MORLET_DECIM,
        n_jobs=N_JOBS,
        verbose=False,
    )[0]

    return potencia, np.arange(potencia.shape[-1]) * MORLET_DECIM / FS


def potencia_stft(ensayo):
    frecuencias, tiempo, zxx = signal.stft(
        ensayo,
        fs=FS,
        window=signal.windows.hann(FFT_NPERSEG, sym=False),
        nperseg=FFT_NPERSEG,
        noverlap=FFT_NPERSEG - FFT_PASO_MUESTRAS,
        nfft=FFT_NFFT,
        detrend=False,
        boundary=None,
        padded=False,
        axis=-1,
    )

    mascara = (frecuencias >= F_LOW) & (frecuencias <= F_HIGH)

    return np.abs(zxx[:, mascara, :]) ** 2, tiempo


def frecuencias_fft():
    frecuencias = np.fft.rfftfreq(FFT_NFFT, 1.0 / FS)
    return frecuencias[(frecuencias >= F_LOW) & (frecuencias <= F_HIGH)]


def calcular_ersp(ensayos):
    """ERSP en dB: potencia media de los N ensayos / baseline medio."""

    potencia, base, tiempo = promedio_con_baseline(ensayos, potencia_morlet)
    ersp_db = 10.0 * np.log10((potencia + EPS) / (base[..., np.newaxis] + EPS))

    return ersp_db, tiempo


def calcular_erd_ers(ensayos):
    """ERD/ERS % por banda: 100 * (SPEC(f,t) / baseline(f) - 1), promedio en la banda."""

    potencia, base, tiempo = promedio_con_baseline(ensayos, potencia_stft)
    frecuencias = frecuencias_fft()

    cambio = 100.0 * (potencia / (base[..., np.newaxis] + EPS) - 1.0)

    erd_ers = {}

    for banda, (f_ini, f_fin) in BANDAS.items():
        # El limite superior es inclusivo solo en el extremo del rango de analisis
        tope = frecuencias <= f_fin if f_fin >= frecuencias.max() else frecuencias < f_fin
        mascara = (frecuencias >= f_ini) & tope

        if not mascara.any():
            raise ValueError(f"No hay bins FFT en la banda {banda} ({f_ini}-{f_fin} Hz).")

        erd_ers[banda] = cambio[:, mascara, :].mean(axis=1)

    return erd_ers, tiempo


# =============================================================================
# FIGURAS
# =============================================================================

def crear_malla(titulo):
    """Malla 3x3 con los canales en POSICIONES_CANALES. Devuelve fig, {canal: ax}, ejes libres."""

    fig, axes = plt.subplots(3, 3, figsize=FIG_SIZE, constrained_layout=True)
    fig.suptitle(titulo, fontsize=18, fontweight="bold")

    ejes = {c: axes[POSICIONES_CANALES[c][0] - 1, POSICIONES_CANALES[c][1] - 1] for c in CANALES}
    libres = [ax for ax in axes.ravel() if all(ax is not usado for usado in ejes.values())]

    for ax in libres:
        ax.set_axis_off()

    return fig, ejes, libres


def marcar_protocolo(ax, sombrear):
    if sombrear:
        ax.axvspan(*T_REPOSO_INICIAL, color=COLOR_REPOSO, alpha=0.2)
        ax.axvspan(*T_TAREA, color=COLOR_TAREA, alpha=0.2)
        ax.axvspan(*T_REPOSO_FINAL, color=COLOR_REPOSO, alpha=0.2)

    for t in (T_TAREA[0], T_TAREA[1]):
        ax.axvline(t, color="gray" if sombrear else "black", linestyle="--", linewidth=1.2)

    ax.set_xlim(T_REPOSO_INICIAL[0], T_FIN)


def finalizar_figura(fig, nombre):
    if MAXIMIZAR_FIGURAS and MOSTRAR_FIGURAS:
        pantalla_completa(fig)

    if GUARDAR_FIGURAS:
        CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
        ruta = CARPETA_RESULTADOS / f"{nombre}.{FIG_FORMATO}"
        fig.savefig(ruta, dpi=FIG_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[GUARDADO] {ruta}")


def graficar_ersp(ersp_db, tiempo, n_ensayos):
    fig, ejes, _ = crear_malla(f"ERSP Morlet - Promedio de {n_ensayos} Ensayos")

    limite = LIMITE_ERSP_DB or max(1.0, float(np.percentile(np.abs(ersp_db), 98.0)))
    bordes_bandas = sorted({f for rango in BANDAS.values() for f in rango})

    for i, canal in enumerate(CANALES):
        ax = ejes[canal]

        imagen = ax.pcolormesh(
            tiempo, FREQS_MORLET, ersp_db[i],
            shading="auto", cmap=CMAP_ERSP, vmin=-limite, vmax=limite,
        )

        marcar_protocolo(ax, sombrear=False)

        for f in bordes_bandas:
            ax.axhline(f, color="black", linestyle=":", linewidth=1.2)

        ax.set_yscale("log")
        ax.set_ylim(FREQS_MORLET[0], FREQS_MORLET[-1])
        
        ticks_frecuencia = [8, 10, 12, 14, 17, 20, 25, 30]
        
        ax.set_yticks(ticks_frecuencia)
        ax.set_yticklabels([str(f) for f in ticks_frecuencia])
        
        ax.set_title(canal, fontsize=14, fontweight="bold")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Frecuencia (Hz)")

    barra = fig.colorbar(imagen, ax=list(ejes.values()), shrink=0.85, pad=0.02)
    barra.set_label("ERSP (dB)", fontsize=13, fontweight="bold")

    finalizar_figura(fig, "ERSP_Morlet_promedio")
    # pantalla_completa(fig)


def graficar_erd_ers(erd_ers, tiempo, banda, n_ensayos):
    f_ini, f_fin = BANDAS[banda]

    fig, ejes, libres = crear_malla(
        f"ERD/ERS porcentual - {banda} ({f_ini:g}-{f_fin:g} Hz) - Promedio de {n_ensayos} Ensayos"
    )

    ventana = int(round(SUAVIZADO_SEGUNDOS / FFT_PASO_SEGUNDOS)) if SUAVIZADO_SEGUNDOS else 1
    curvas = erd_ers[banda]

    if ventana > 1:
        curvas = uniform_filter1d(curvas, size=ventana, axis=-1, mode="nearest")

    if LIMITES_ERD_ERS is not None:
        y_min, y_max = LIMITES_ERD_ERS
    else:
        y_min, y_max = min(curvas.min(), 0.0), max(curvas.max(), 0.0)
        margen = 0.05 * (y_max - y_min or 1.0)
        y_min, y_max = y_min - margen, y_max + margen

    for i, canal in enumerate(CANALES):
        ax = ejes[canal]

        marcar_protocolo(ax, sombrear=True)
        ax.axhline(0.0, linewidth=1.0, alpha=0.8)
        ax.plot(tiempo, curvas[i], color=COLORES_CANALES.get(canal, "C0"), linewidth=1.6)

        ax.set_ylim(y_min, y_max)
        ax.set_title(canal, fontsize=14, fontweight="bold")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Cambio vs baseline (%)")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)

    if libres:
        libres[0].legend(
            handles=[
                Patch(color=COLOR_REPOSO, alpha=0.4, label="Reposo"),
                Patch(color=COLOR_TAREA, alpha=0.4, label="Tarea"),
            ],
            loc="center",
            fontsize=13,
        )

    finalizar_figura(fig, f"ERD_ERS_porcentual_{banda}")
    # pantalla_completa(fig)


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

def main():
    validar_configuracion()

    print("=" * 70)
    print(f"Carpeta: {CARPETA_DATOS}")
    print(f"Reposo inicial {T_REPOSO_INICIAL} | Tarea {T_TAREA} | Reposo final {T_REPOSO_FINAL} s")
    print(f"Baseline: {BASELINE} s")
    print("=" * 70)

    ensayos = aplicar_ica(cargar_ensayos())
    n_ensayos = ensayos.shape[0]

    print(f"\nEnsayos usados: {n_ensayos} | Forma: {ensayos.shape} (ensayos, canales, muestras)")

    ciclos = np.broadcast_to(N_CYCLES_MORLET, FREQS_MORLET.shape)
    sigma_t = ciclos / (2 * np.pi * FREQS_MORLET)
    print(
        f"Morlet: sigma temporal {sigma_t[0]:.3f} s en {FREQS_MORLET[0]:g} Hz "
        f"y {sigma_t[-1]:.3f} s en {FREQS_MORLET[-1]:g} Hz"
    )
    print(f"FFT: resolucion {FS / FFT_NFFT:.2f} Hz\n")

    ersp_db, tiempo_tfr = calcular_ersp(ensayos)
    graficar_ersp(ersp_db, tiempo_tfr, n_ensayos)

    erd_ers, tiempo_fft = calcular_erd_ers(ensayos)
    for banda in BANDAS:
        graficar_erd_ers(erd_ers, tiempo_fft, banda, n_ensayos)

    if MOSTRAR_FIGURAS:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()