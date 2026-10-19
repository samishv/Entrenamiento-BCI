from pathlib import Path
import re

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
    filtro_laplaciano,
    pantalla_completa,
)


# =============================================================================
# CONFIGURACION: DATOS
# =============================================================================

plt.close("all")

# CARPETA_DATOS = Path.cwd() / "Protocolo ERD_ERS" / "Signals" / "SV3" / "MI"

CARPETA_DATOS = Path(
    r"D:\luiso\Documentos\Luis\UPIITA\TT\Entrenamiento-BCI\Protocolo ERD_ERS\Signals"
    r"\SV3\ME"
)

PATRON_ARCHIVOS = "*.npy"
CARPETA_RESULTADOS = CARPETA_DATOS.parent / "Resultados_ERD_ERS"
MODALIDAD = CARPETA_DATOS.name

FS = 250.0

CANALES = ["Fz", "FC3", "FCz", "FC4", "Cz", "C3", "C4"]

POSICIONES_CANALES = {
    "Fz":  (1, 2),
    "FC3": (2, 1), "FCz": (2, 2), "FC4": (2, 3),
    "C3":  (3, 1), "Cz":  (3, 2), "C4":  (3, 3),
}


# =============================================================================
# CONFIGURACION: TIEMPOS DEL PROTOCOLO (segundos, iguales para todos los ensayos)
# =============================================================================

T_REPOSO_INICIAL = (0.0, 3.0)
T_TAREA          = (3.0, 8.0)
T_REPOSO_FINAL   = (8.0, 11.0)
T_FIN = T_REPOSO_FINAL[1]


MARGEN_BASELINE = 0.5                                       # Baseline = reposo inicial menos un margen en cada extremo 
BASELINE = (
    T_REPOSO_INICIAL[0] + MARGEN_BASELINE,
    T_REPOSO_INICIAL[1] - MARGEN_BASELINE,
)


# =============================================================================
# CONFIGURACION: BANDAS
# =============================================================================

BANDAS = {
    "Ritmo Mu":    (8.0, 12.0),
    "Ritmo Beta":  (12.0, 30.0),
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

PAD_MUESTRAS = int(5 * FS)                                  # padding reflejado (5 segundos)

USAR_ICA = False                                            # USAR ICA
IC_TO_REMOVE = None                                         # Componente principal a eliminar       
ICA_RANDOM_STATE = 97    

USAR_LAPLACIANO = True
CANAL_INTERES_LAPLACIANO = "C3"
CANALES_VECINOS_LAPLACIANO = ["Cz","FC3"]
NOMBRE_CANAL_LAPLACIANO = f"{CANAL_INTERES_LAPLACIANO} Laplaciano"                   


# =============================================================================
# CONFIGURACION: ERSP CON MORLET  (zona para ir probando valores)
# =============================================================================

FREQS_MORLET = np.arange(7.0, 32.0, 1.0)                    # rango del escalograma
N_CYCLES_MORLET = np.linspace(6, 15, len(FREQS_MORLET))     # de 6 a 15 ciclos de barrido

MORLET_DECIM = 4                                            # 250 Hz / 5 = 50 puntos por segundo
N_JOBS = 1

LIMITE_ERSP_DB = None                                       # p. ej. 3.0 para fijar +-3 dB; None = automatico
CMAP_ERSP = "jet"


# =============================================================================
# CONFIGURACION: FFT DESLIZANTE PARA ERD/ERS PORCENTUAL
# =============================================================================

FFT_DURACION_VENTANA = 0.8                                  # s  (resolucion en frecuencia = 1 / ventana) 100 muestras / 250 Hz = 0.4 segundos
FFT_PASO_SEGUNDOS = 0.2                                     # s
FFT_NPERSEG = int(FFT_DURACION_VENTANA * FS)
FFT_PASO_MUESTRAS = int(FFT_PASO_SEGUNDOS * FS)
FFT_NFFT = FFT_NPERSEG                                      # FFT_NPERSEG interpola el espectro (zero-padding)

SUAVIZADO_SEGUNDOS = 0                                      # solo visual; 0 o None desactiva "MULTIPLO DE PASO"
LIMITES_ERD_ERS = None                                      # p. ej. (-100, 200) en %; None = automatico

LIMITES_RELACION_MU_BETA = None


# =============================================================================
# CONFIGURACION: FIGURAS
# =============================================================================

GUARDAR_FIGURAS = True
MOSTRAR_FIGURAS = False
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
    bandas_requeridas = {"Ritmo Mu", "Ritmo Beta"}
    bandas_faltantes = bandas_requeridas.difference(BANDAS)

    if bandas_faltantes:
        raise ValueError(
            "Para calcular la relación Mu/Beta deben existir las bandas "
            f"'Ritmo Mu' y 'Ritmo Beta'. Faltan: {sorted(bandas_faltantes)}"
        )

    if USAR_LAPLACIANO:
        if CANAL_INTERES_LAPLACIANO not in CANALES:
            raise ValueError(
                f"El canal de interés '{CANAL_INTERES_LAPLACIANO}' "
                "no se encuentra en CANALES."
            )

        if not CANALES_VECINOS_LAPLACIANO:
            raise ValueError(
                "CANALES_VECINOS_LAPLACIANO debe contener al menos un canal."
            )

        vecinos_faltantes = [
            canal
            for canal in CANALES_VECINOS_LAPLACIANO
            if canal not in CANALES
        ]

        if vecinos_faltantes:
            raise ValueError(
                "Los siguientes canales vecinos no se encuentran en CANALES: "
                f"{vecinos_faltantes}"
            )

        if CANAL_INTERES_LAPLACIANO in CANALES_VECINOS_LAPLACIANO:
            raise ValueError(
                "El canal de interés no puede aparecer también como vecino."
            )

        if len(set(CANALES_VECINOS_LAPLACIANO)) != len(
            CANALES_VECINOS_LAPLACIANO
        ):
            raise ValueError(
                "CANALES_VECINOS_LAPLACIANO contiene canales repetidos."
            )


# =============================================================================
# CARGA Y PREPROCESAMIENTO
# =============================================================================

def preprocesar_ensayo(datos):
    if ELIMINAR_DC:
        datos = quitar_componente_dc(datos)
    pad = min(PAD_MUESTRAS, datos.shape[1] - 1)
    datos = aplicar_notch_multiple(datos, FS, FREQS_NOTCH, Q_NOTCH, pad)
    datos = aplicar_pasabanda_butter(datos, FS, F_LOW, F_HIGH, BP_ORDER, pad)
    return datos


def obtener_dia_desde_ruta(archivo):
    ruta_relativa = archivo.relative_to(CARPETA_DATOS)

    for parte in ruta_relativa.parts[:-1]:
        coincidencia = re.fullmatch(
            r"D(\d+)_B(\d+)",
            parte,
            flags=re.IGNORECASE,
        )

        if coincidencia:
            numero_dia = int(coincidencia.group(1))
            return f"D{numero_dia}"

    return None


def clave_orden_dia(nombre_dia):
    coincidencia = re.fullmatch(r"D(\d+)", nombre_dia, flags=re.IGNORECASE)

    if coincidencia:
        return int(coincidencia.group(1))

    return float("inf")


def cargar_ensayos_por_dia():
    archivos = sorted(CARPETA_DATOS.rglob(PATRON_ARCHIVOS))

    if not archivos:
        raise FileNotFoundError(
            f"No se encontraron archivos {PATRON_ARCHIVOS} en:\n"
            f"{CARPETA_DATOS}"
        )

    n_fin = int(round(T_FIN * FS))
    ensayos_por_dia = {}

    print(f"Archivos '.npy' encontrados: {len(archivos)}")

    for archivo in archivos:
        nombre_relativo = archivo.relative_to(CARPETA_DATOS)
        dia = obtener_dia_desde_ruta(archivo)

        if dia is None:
            print(
                f"  [OMITIDO] {nombre_relativo}: "
                "no está dentro de una carpeta con formato DX_BX"
            )
            continue

        try:
            datos = cargar_npy_eeg(
                str(archivo),
                n_canales_esperados=len(CANALES),
            )

            if not np.isfinite(datos).all():
                raise ValueError("contiene NaN o infinitos")

            if datos.shape[1] < n_fin:
                raise ValueError(
                    f"dura {datos.shape[1] / FS:.2f} s "
                    f"y el protocolo requiere {T_FIN:g} s"
                )

        except ValueError as error:
            print(f"  [OMITIDO] {nombre_relativo}: {error}")
            continue

        datos_preprocesados = preprocesar_ensayo(datos)[:, :n_fin]

        if dia not in ensayos_por_dia:
            ensayos_por_dia[dia] = []

        ensayos_por_dia[dia].append(datos_preprocesados)

        print(f"  [OK] {nombre_relativo} -> {dia}")

    if not ensayos_por_dia:
        raise RuntimeError(
            "No se encontró ningún archivo válido dentro de carpetas DX_BX."
        )

    ensayos_por_dia = {
        dia: np.stack(lista_ensayos, axis=0)
        for dia, lista_ensayos in ensayos_por_dia.items()
    }

    ensayos_por_dia = dict(
        sorted(
            ensayos_por_dia.items(),
            key=lambda elemento: clave_orden_dia(elemento[0]),
        )
    )

    return ensayos_por_dia


def aplicar_ica(ensayos):
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

def aplicar_laplaciano_ensayos(ensayos):
    senales_laplacianas = []

    for ensayo in ensayos:
        senal_laplaciana = filtro_laplaciano(
            datos=ensayo,
            canales=CANALES,
            canal_interes=CANAL_INTERES_LAPLACIANO,
            canales_vecinos=CANALES_VECINOS_LAPLACIANO,
        )

        senales_laplacianas.append(senal_laplaciana)

    ensayos_laplacianos = np.stack(
        senales_laplacianas,
        axis=0,
    )

    ensayos_laplacianos = ensayos_laplacianos[:, np.newaxis, :]

    return ensayos_laplacianos


# =============================================================================
# CALCULOS
# =============================================================================

def intervalo_a_mascara(tiempo, intervalo):
    return (tiempo >= intervalo[0]) & (tiempo <= intervalo[1])


def promedio_con_baseline(ensayos, potencia_ensayo):
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
    potencia, base, tiempo = promedio_con_baseline(ensayos, potencia_morlet)
    ersp_db = 10.0 * np.log10((potencia + EPS) / (base[..., np.newaxis] + EPS))

    return ersp_db, tiempo


def calcular_erd_ers(ensayos):
    potencia, base, tiempo = promedio_con_baseline(ensayos, potencia_stft)
    frecuencias = frecuencias_fft()

    cambio = 100.0 * (potencia / (base[..., np.newaxis] + EPS) - 1.0)

    erd_ers = {}

    for banda, (f_ini, f_fin) in BANDAS.items():
        tope = frecuencias <= f_fin if f_fin >= frecuencias.max() else frecuencias < f_fin
        mascara = (frecuencias >= f_ini) & tope

        if not mascara.any():
            raise ValueError(f"No hay bins FFT en la banda {banda} ({f_ini}-{f_fin} Hz).")

        erd_ers[banda] = cambio[:, mascara, :].mean(axis=1)

    return erd_ers, tiempo

def calcular_relacion_mu_beta(ensayos):
    potencia, _, tiempo = promedio_con_baseline(
        ensayos,
        potencia_stft,
    )

    frecuencias = frecuencias_fft()

    f_mu_ini, f_mu_fin = BANDAS["Ritmo Mu"]
    f_beta_ini, f_beta_fin = BANDAS["Ritmo Beta"]

    mascara_mu = (
        (frecuencias >= f_mu_ini)
        & (frecuencias < f_mu_fin)
    )

    mascara_beta = (
        (frecuencias >= f_beta_ini)
        & (frecuencias <= f_beta_fin)
    )

    if not mascara_mu.any():
        raise ValueError(
            "No hay bins FFT dentro de la banda Mu "
            f"({f_mu_ini:g}-{f_mu_fin:g} Hz)."
        )

    if not mascara_beta.any():
        raise ValueError(
            "No hay bins FFT dentro de la banda Beta "
            f"({f_beta_ini:g}-{f_beta_fin:g} Hz)."
        )

    # Potencia temporal promedio dentro de cada banda.
    potencia_mu = potencia[:, mascara_mu, :].mean(axis=1)
    potencia_beta = potencia[:, mascara_beta, :].mean(axis=1)

    # Relación temporal directa Mu/Beta.
    relacion_mu_beta = (
        potencia_mu + EPS
    ) / (
        potencia_beta + EPS
    )

    # Baseline de la propia relación Mu/Beta.
    mascara_baseline = intervalo_a_mascara(
        tiempo,
        BASELINE,
    )

    if not mascara_baseline.any():
        raise ValueError(
            "El baseline no contiene puntos temporales "
            "para calcular la relación Mu/Beta."
        )

    relacion_baseline = relacion_mu_beta[
        :,
        mascara_baseline,
    ].mean(axis=-1)

    # Cambio porcentual de la relación respecto al baseline.
    cambio_relacion = 100.0 * (
        relacion_mu_beta
        / (relacion_baseline[:, np.newaxis] + EPS)
        - 1.0
    )

    return cambio_relacion, tiempo


# =============================================================================
# FIGURAS
# =============================================================================

def crear_malla(titulo):
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


def finalizar_figura(fig, nombre, subcarpeta):
    if MAXIMIZAR_FIGURAS and MOSTRAR_FIGURAS:
        pantalla_completa(fig)

    if GUARDAR_FIGURAS:
        carpeta_destino = CARPETA_RESULTADOS / subcarpeta
        carpeta_destino.mkdir(parents=True, exist_ok=True)

        ruta = carpeta_destino / f"{nombre}.{FIG_FORMATO}"

        fig.savefig(
            ruta,
            dpi=FIG_DPI,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )

        print(f"[GUARDADO] {nombre}.{FIG_FORMATO} ✅")


def graficar_ersp(ersp_db, tiempo, n_ensayos, grupo):
    fig, ejes, _ = crear_malla(
        f"ERSP | {MODALIDAD} | {grupo} | "
        f"{n_ensayos} Ensayos"
    )

    limite = LIMITE_ERSP_DB or max(
        1.0,
        float(np.percentile(np.abs(ersp_db), 98.0)),
    )

    bordes_bandas = sorted({
        frecuencia
        for rango in BANDAS.values()
        for frecuencia in rango
    })

    for i, canal in enumerate(CANALES):
        ax = ejes[canal]

        imagen = ax.pcolormesh(
            tiempo,
            FREQS_MORLET,
            ersp_db[i],
            shading="auto",
            cmap=CMAP_ERSP,
            vmin=-limite,
            vmax=limite,
        )

        marcar_protocolo(ax, sombrear=False)

        for frecuencia in bordes_bandas:
            ax.axhline(
                frecuencia,
                color="black",
                linestyle=":",
                linewidth=1.2,
            )

        ax.set_yscale("log")
        ax.set_ylim(FREQS_MORLET[0], FREQS_MORLET[-1])

        ticks_frecuencia = [8, 10, 12, 14, 17, 20, 25, 30]
        ticks_frecuencia = [
            frecuencia
            for frecuencia in ticks_frecuencia
            if FREQS_MORLET[0] <= frecuencia <= FREQS_MORLET[-1]
        ]

        ax.set_yticks(ticks_frecuencia)
        ax.set_yticklabels([str(f) for f in ticks_frecuencia])

        ax.set_title(canal, fontsize=14, fontweight="bold")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Frecuencia (Hz)")

    barra = fig.colorbar(
        imagen,
        ax=list(ejes.values()),
        shrink=0.85,
        pad=0.02,
    )

    barra.set_label(
        "ERSP (dB)",
        fontsize=13,
        fontweight="bold",
    )

    nombre_archivo = f"ERSP_Morlet_{MODALIDAD}_{grupo}"

    finalizar_figura(
        fig,
        nombre=nombre_archivo,
        subcarpeta="Escalogramas",
    )


def graficar_erd_ers(erd_ers, tiempo, banda, n_ensayos, grupo):
    f_ini, f_fin = BANDAS[banda]

    fig, ejes, libres = crear_malla(
        f"% ERD/ERS | {MODALIDAD} | {grupo} | "
        f"{banda} ({f_ini:g}-{f_fin:g} Hz) | "
        f"{n_ensayos} Ensayos"
    )

    ventana = (
        int(round(SUAVIZADO_SEGUNDOS / FFT_PASO_SEGUNDOS))
        if SUAVIZADO_SEGUNDOS
        else 1
    )

    curvas = erd_ers[banda]

    if ventana > 1:
        curvas = uniform_filter1d(
            curvas,
            size=ventana,
            axis=-1,
            mode="nearest",
        )

    if LIMITES_ERD_ERS is not None:
        y_min, y_max = LIMITES_ERD_ERS
    else:
        y_min = min(curvas.min(), 0.0)
        y_max = max(curvas.max(), 0.0)

        margen = 0.05 * (y_max - y_min or 1.0)
        y_min -= margen
        y_max += margen

    for i, canal in enumerate(CANALES):
        ax = ejes[canal]

        marcar_protocolo(ax, sombrear=True)
        ax.axhline(0.0, linewidth=1.0, alpha=0.8)

        ax.plot(
            tiempo,
            curvas[i],
            color=COLORES_CANALES.get(canal, "C0"),
            linewidth=1.6,
        )

        ax.set_ylim(y_min, y_max)
        ax.set_title(canal, fontsize=14, fontweight="bold")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Cambio vs baseline (%)")
        ax.grid(
            True,
            linestyle="--",
            linewidth=0.4,
            alpha=0.4,
        )

    if libres:
        libres[0].legend(
            handles=[
                Patch(
                    color=COLOR_REPOSO,
                    alpha=0.4,
                    label="Reposo",
                ),
                Patch(
                    color=COLOR_TAREA,
                    alpha=0.4,
                    label="Tarea",
                ),
            ],
            loc="center",
            fontsize=13,
        )

    nombre_archivo = f"ERD_ERS_{MODALIDAD}_{banda}_{grupo}"

    finalizar_figura(
        fig,
        nombre=nombre_archivo,
        subcarpeta=banda,
    )
    
def graficar_relacion_mu_beta(
        relacion_mu_beta,
        tiempo,
        n_ensayos,
        grupo,
        es_laplaciano=False):

    ventana = (
        int(round(
            SUAVIZADO_SEGUNDOS
            / FFT_PASO_SEGUNDOS
        ))
        if SUAVIZADO_SEGUNDOS
        else 1
    )

    curvas = relacion_mu_beta.copy()

    if ventana > 1:
        curvas = uniform_filter1d(
            curvas,
            size=ventana,
            axis=-1,
            mode="nearest",
        )

    if LIMITES_RELACION_MU_BETA is not None:
        y_min, y_max = LIMITES_RELACION_MU_BETA
    else:
        y_min = min(float(curvas.min()), 0.0)
        y_max = max(float(curvas.max()), 0.0)

        margen = 0.05 * (y_max - y_min or 1.0)

        y_min -= margen
        y_max += margen

    if es_laplaciano:
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(14, 7),
            constrained_layout=True,
        )

        fig.suptitle(
            f"Relación Mu/Beta vs baseline | "
            f"{MODALIDAD} | {grupo} | "
            f"{NOMBRE_CANAL_LAPLACIANO} | "
            f"{n_ensayos} Ensayos",
            fontsize=18,
            fontweight="bold",
        )

        marcar_protocolo(
            ax,
            sombrear=True,
        )

        ax.axhline(
            0.0,
            linewidth=1.0,
            alpha=0.8,
        )

        ax.plot(
            tiempo,
            curvas[0],
            color=COLORES_CANALES.get(
                CANAL_INTERES_LAPLACIANO,
                "C0",
            ),
            linewidth=1.8,
        )

        ax.set_ylim(y_min, y_max)

        ax.set_title(
            (
                f"{CANAL_INTERES_LAPLACIANO} - promedio"
                f"({', '.join(CANALES_VECINOS_LAPLACIANO)})"
            ),
            fontsize=14,
            fontweight="bold",
        )

        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel(
            "Cambio de la relación Mu/Beta vs baseline (%)"
        )

        ax.grid(
            True,
            linestyle="--",
            linewidth=0.4,
            alpha=0.4,
        )

        ax.legend(
            handles=[
                Patch(
                    color=COLOR_REPOSO,
                    alpha=0.4,
                    label="Reposo",
                ),
                Patch(
                    color=COLOR_TAREA,
                    alpha=0.4,
                    label="Tarea",
                ),
            ],
            loc="best",
            fontsize=11,
        )

        nombre_archivo = (
            f"Relacion_Mu_Beta_Laplaciano_"
            f"{MODALIDAD}_{grupo}"
        )

        finalizar_figura(
            fig,
            nombre=nombre_archivo,
            subcarpeta=(
                f"Laplaciano_{CANAL_INTERES_LAPLACIANO}"
                "/Relacion Mu Beta"
            ),
        )

    else:
        fig, ejes, libres = crear_malla(
            f"Relación Mu/Beta vs baseline | "
            f"{MODALIDAD} | {grupo} | "
            f"{n_ensayos} Ensayos"
        )

        for i, canal in enumerate(CANALES):
            ax = ejes[canal]

            marcar_protocolo(
                ax,
                sombrear=True,
            )

            ax.axhline(
                0.0,
                linewidth=1.0,
                alpha=0.8,
            )

            ax.plot(
                tiempo,
                curvas[i],
                color=COLORES_CANALES.get(
                    canal,
                    "C0",
                ),
                linewidth=1.6,
            )

            ax.set_ylim(y_min, y_max)

            ax.set_title(
                canal,
                fontsize=14,
                fontweight="bold",
            )

            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel(
                "Cambio Mu/Beta vs baseline (%)"
            )

            ax.grid(
                True,
                linestyle="--",
                linewidth=0.4,
                alpha=0.4,
            )

        if libres:
            libres[0].legend(
                handles=[
                    Patch(
                        color=COLOR_REPOSO,
                        alpha=0.4,
                        label="Reposo",
                    ),
                    Patch(
                        color=COLOR_TAREA,
                        alpha=0.4,
                        label="Tarea",
                    ),
                ],
                loc="center",
                fontsize=13,
            )

        nombre_archivo = (
            f"Relacion_Mu_Beta_{MODALIDAD}_{grupo}"
        )

        finalizar_figura(
            fig,
            nombre=nombre_archivo,
            subcarpeta="Relacion Mu Beta",
        )
        
def graficar_ersp_laplaciano(
        ersp_db,
        tiempo,
        n_ensayos,
        grupo):
    """
    Genera y guarda el escalograma ERSP del único canal obtenido
    mediante el filtro laplaciano.

    Parámetros
    ----------
    ersp_db : ndarray
        ERSP normalizada respecto al baseline y expresada en dB.
        Forma esperada:
        (1, n_frecuencias, n_tiempos)

    tiempo : ndarray
        Vector temporal correspondiente al ERSP.

    n_ensayos : int
        Número de ensayos incluidos en el promedio.

    grupo : str
        Nombre del grupo analizado, por ejemplo:
        "D1", "D2" o "GENERAL".
    """

    # Validación de la forma recibida.
    if ersp_db.ndim != 3:
        raise ValueError(
            "ersp_db debe tener tres dimensiones con forma "
            "(1, n_frecuencias, n_tiempos). "
            f"Forma recibida: {ersp_db.shape}"
        )

    if ersp_db.shape[0] != 1:
        raise ValueError(
            "graficar_ersp_laplaciano espera un único canal. "
            f"Se recibieron {ersp_db.shape[0]} canales."
        )

    if ersp_db.shape[1] != len(FREQS_MORLET):
        raise ValueError(
            "La cantidad de frecuencias del ERSP no coincide con "
            "FREQS_MORLET. "
            f"ERSP: {ersp_db.shape[1]} frecuencias; "
            f"FREQS_MORLET: {len(FREQS_MORLET)} frecuencias."
        )

    if ersp_db.shape[2] != len(tiempo):
        raise ValueError(
            "La cantidad de puntos temporales del ERSP no coincide "
            "con el vector tiempo. "
            f"ERSP: {ersp_db.shape[2]} puntos; "
            f"tiempo: {len(tiempo)} puntos."
        )

    # -------------------------------------------------------------
    # Creación de la figura
    # -------------------------------------------------------------
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(14, 7),
        constrained_layout=True,
    )

    fig.suptitle(
        f"ERSP Laplaciano | {MODALIDAD} | {grupo} | "
        f"{n_ensayos} Ensayos",
        fontsize=18,
        fontweight="bold",
    )

    # -------------------------------------------------------------
    # Límite simétrico de colores
    # -------------------------------------------------------------
    # Si LIMITE_ERSP_DB tiene un valor, se utiliza directamente.
    # Si es None, se calcula automáticamente con el percentil 98
    # del valor absoluto del ERSP.
    if LIMITE_ERSP_DB is not None:
        limite = float(LIMITE_ERSP_DB)
    else:
        valores_finitos = np.abs(
            ersp_db[np.isfinite(ersp_db)]
        )

        if valores_finitos.size == 0:
            limite = 1.0
        else:
            limite = max(
                1.0,
                float(np.percentile(
                    valores_finitos,
                    98.0,
                )),
            )

    # -------------------------------------------------------------
    # Escalograma del único canal laplaciano
    # -------------------------------------------------------------
    imagen = ax.pcolormesh(
        tiempo,
        FREQS_MORLET,
        ersp_db[0],
        shading="auto",
        cmap=CMAP_ERSP,
        vmin=-limite,
        vmax=limite,
    )

    # Líneas verticales que indican el inicio y final de la tarea.
    # No se sombrean los periodos porque podrían ocultar colores
    # importantes del escalograma.
    marcar_protocolo(
        ax,
        sombrear=False,
    )

    # -------------------------------------------------------------
    # Límites de las bandas Mu y Beta
    # -------------------------------------------------------------
    bordes_bandas = sorted({
        frecuencia
        for rango in BANDAS.values()
        for frecuencia in rango
    })

    for frecuencia in bordes_bandas:
        if FREQS_MORLET[0] <= frecuencia <= FREQS_MORLET[-1]:
            ax.axhline(
                frecuencia,
                color="black",
                linestyle=":",
                linewidth=1.2,
            )

    # -------------------------------------------------------------
    # Configuración del eje de frecuencias
    # -------------------------------------------------------------
    ax.set_yscale("log")

    ax.set_ylim(
        FREQS_MORLET[0],
        FREQS_MORLET[-1],
    )

    ticks_frecuencia = [
        8,
        10,
        12,
        14,
        17,
        20,
        25,
        30,
    ]

    ticks_frecuencia = [
        frecuencia
        for frecuencia in ticks_frecuencia
        if (
            FREQS_MORLET[0]
            <= frecuencia
            <= FREQS_MORLET[-1]
        )
    ]

    ax.set_yticks(ticks_frecuencia)

    ax.set_yticklabels([
        str(frecuencia)
        for frecuencia in ticks_frecuencia
    ])

    # -------------------------------------------------------------
    # Etiquetas y título del canal
    # -------------------------------------------------------------
    vecinos_texto = ", ".join(
        CANALES_VECINOS_LAPLACIANO
    )

    ax.set_title(
        f"{CANAL_INTERES_LAPLACIANO} - "
        f"promedio({vecinos_texto})",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Tiempo (s)",
        fontsize=12,
    )

    ax.set_ylabel(
        "Frecuencia (Hz)",
        fontsize=12,
    )

    # -------------------------------------------------------------
    # Barra de colores
    # -------------------------------------------------------------
    barra = fig.colorbar(
        imagen,
        ax=ax,
        shrink=0.85,
        pad=0.02,
    )

    barra.set_label(
        "ERSP (dB)",
        fontsize=13,
        fontweight="bold",
    )

    # -------------------------------------------------------------
    # Guardado de la figura
    # -------------------------------------------------------------
    nombre_archivo = (
        f"ERSP_Morlet_Laplaciano_"
        f"{MODALIDAD}_{grupo}"
    )

    subcarpeta = (
        f"Laplaciano_{CANAL_INTERES_LAPLACIANO}"
        "/Escalogramas"
    )

    finalizar_figura(
        fig,
        nombre=nombre_archivo,
        subcarpeta=subcarpeta,
    )
        
def graficar_erd_ers_laplaciano(
        erd_ers,
        tiempo,
        banda,
        n_ensayos,
        grupo):

    f_ini, f_fin = BANDAS[banda]

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(14, 7),
        constrained_layout=True,
    )

    fig.suptitle(
        f"% ERD/ERS | {MODALIDAD} | {grupo} | "
        f"{NOMBRE_CANAL_LAPLACIANO} | "
        f"{banda} ({f_ini:g}-{f_fin:g} Hz) | "
        f"{n_ensayos} Ensayos",
        fontsize=18,
        fontweight="bold",
    )

    ventana = (
        int(round(
            SUAVIZADO_SEGUNDOS
            / FFT_PASO_SEGUNDOS
        ))
        if SUAVIZADO_SEGUNDOS
        else 1
    )

    curva = erd_ers[banda][0].copy()

    if ventana > 1:
        curva = uniform_filter1d(
            curva,
            size=ventana,
            mode="nearest",
        )

    if LIMITES_ERD_ERS is not None:
        y_min, y_max = LIMITES_ERD_ERS
    else:
        y_min = min(float(curva.min()), 0.0)
        y_max = max(float(curva.max()), 0.0)

        margen = 0.05 * (y_max - y_min or 1.0)

        y_min -= margen
        y_max += margen

    marcar_protocolo(
        ax,
        sombrear=True,
    )

    ax.axhline(
        0.0,
        linewidth=1.0,
        alpha=0.8,
    )

    ax.plot(
        tiempo,
        curva,
        color=COLORES_CANALES.get(
            CANAL_INTERES_LAPLACIANO,
            "C0",
        ),
        linewidth=1.8,
    )

    ax.set_ylim(y_min, y_max)

    ax.set_title(
        (
            f"{CANAL_INTERES_LAPLACIANO} - promedio"
            f"({', '.join(CANALES_VECINOS_LAPLACIANO)})"
        ),
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Cambio vs baseline (%)")

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.4,
        alpha=0.4,
    )

    ax.legend(
        handles=[
            Patch(
                color=COLOR_REPOSO,
                alpha=0.4,
                label="Reposo",
            ),
            Patch(
                color=COLOR_TAREA,
                alpha=0.4,
                label="Tarea",
            ),
        ],
        loc="best",
        fontsize=11,
    )

    nombre_archivo = (
        f"ERD_ERS_Laplaciano_"
        f"{MODALIDAD}_{banda}_{grupo}"
    )

    finalizar_figura(
        fig,
        nombre=nombre_archivo,
        subcarpeta=(
            f"Laplaciano_{CANAL_INTERES_LAPLACIANO}"
            f"/{banda}"
        ),
    )
      


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

def main():
    validar_configuracion()

    print(f"\nCarpeta de datos: {CARPETA_DATOS}")
    print(f"Carpeta de resultados: {CARPETA_RESULTADOS}")
    print(
        f"Reposo inicial {T_REPOSO_INICIAL} | "
        f"Tarea {T_TAREA} | "
        f"Reposo final {T_REPOSO_FINAL} s"
    )
    print(f"Baseline: {BASELINE} s\n")

    ciclos = np.broadcast_to(
        N_CYCLES_MORLET,
        FREQS_MORLET.shape,
    )

    sigma_t = ciclos / (2 * np.pi * FREQS_MORLET)

    print(
        f"Morlet: sigma temporal {sigma_t[0]:.3f} s "
        f"en {FREQS_MORLET[0]:g} Hz y "
        f"{sigma_t[-1]:.3f} s "
        f"en {FREQS_MORLET[-1]:g} Hz"
    )

    print(f"FFT: resolución {FS / FFT_NFFT:.2f} Hz\n")

    ensayos_por_dia = cargar_ensayos_por_dia()

    print("\nDías encontrados automáticamente:")

    for dia, ensayos_dia in ensayos_por_dia.items():
        print(f"  {dia}  : {ensayos_dia.shape[0]} ensayos")
    
    total_general = sum(
        ensayos_dia.shape[0]
        for ensayos_dia in ensayos_por_dia.values()
    )
    
    print(f"  GEN : {total_general} ensayos")

    # =============================================================
    # ANÁLISIS INDIVIDUAL DE CADA DÍA
    # =============================================================
    
    for dia, ensayos_dia in ensayos_por_dia.items():
        analizar_grupo(
            ensayos=ensayos_dia,
            grupo=dia,
        )

    # =============================================================
    # ANÁLISIS GENERAL DE TODOS LOS DÍAS
    # =============================================================
    
    ensayos_generales = np.concatenate(
        list(ensayos_por_dia.values()),
        axis=0,
    )

    analizar_grupo(
        ensayos=ensayos_generales,
        grupo="GENERAL",
    )

    if MOSTRAR_FIGURAS:
        plt.show()
    else:
        plt.close("all")

    print("\n" + "=" * 30)
    print("ANÁLISIS TERMINADO")
    print("=" * 30)
    print(f"Resultados guardados en:\n{CARPETA_RESULTADOS}")
        
def analizar_grupo(ensayos, grupo):
    print("\n" + "=" * 30)
    print(f"ANÁLISIS: {MODALIDAD} - {grupo}")
    print("=" * 30)

    # =============================================================
    # ICA
    # =============================================================
    # Si USAR_ICA es False, aplicar_ica devuelve las mismas señales.
    # Si es True, devuelve las señales después de retirar el
    # componente seleccionado.
    ensayos = aplicar_ica(ensayos)

    n_ensayos = ensayos.shape[0]

    print(f"Ensayos: {n_ensayos}")

    # =============================================================
    # ANÁLISIS MULTICANAL ORIGINAL
    # =============================================================
    print("[MULTICANAL] Calculando ERSP...")

    ersp_db, tiempo_tfr = calcular_ersp(ensayos)

    graficar_ersp(
        ersp_db,
        tiempo_tfr,
        n_ensayos,
        grupo,
    )

    print("[MULTICANAL] Calculando ERD/ERS...")

    erd_ers, tiempo_fft = calcular_erd_ers(ensayos)

    for banda in BANDAS:
        graficar_erd_ers(
            erd_ers,
            tiempo_fft,
            banda,
            n_ensayos,
            grupo,
        )

    print(
        "[MULTICANAL] Calculando relación "
        "Mu/Beta respecto al baseline..."
    )

    relacion_mu_beta, tiempo_relacion = (
        calcular_relacion_mu_beta(ensayos)
    )

    graficar_relacion_mu_beta(
        relacion_mu_beta,
        tiempo_relacion,
        n_ensayos,
        grupo,
        es_laplaciano=False,
    )

    # =============================================================
    # ANÁLISIS LAPLACIANO
    # =============================================================
    if USAR_LAPLACIANO:
        print(
            f"[LAPLACIANO] Canal de interés: "
            f"{CANAL_INTERES_LAPLACIANO}"
        )

        print(
            f"[LAPLACIANO] Canales vecinos: "
            f"{CANALES_VECINOS_LAPLACIANO}"
        )

        # El laplaciano se aplica después del ICA y antes de:
        # - Morlet
        # - ERSP
        # - STFT
        # - ERD/ERS
        ensayos_laplacianos = aplicar_laplaciano_ensayos(
            ensayos
        )

        print(
            "[LAPLACIANO] Calculando ERSP..."
        )

        ersp_laplaciano, tiempo_tfr_laplaciano = (
            calcular_ersp(ensayos_laplacianos)
        )

        graficar_ersp_laplaciano(
            ersp_laplaciano,
            tiempo_tfr_laplaciano,
            n_ensayos,
            grupo,
        )

        print(
            "[LAPLACIANO] Calculando ERD/ERS..."
        )

        erd_ers_laplaciano, tiempo_fft_laplaciano = (
            calcular_erd_ers(ensayos_laplacianos)
        )

        for banda in BANDAS:
            graficar_erd_ers_laplaciano(
                erd_ers_laplaciano,
                tiempo_fft_laplaciano,
                banda,
                n_ensayos,
                grupo,
            )

        print(
            "[LAPLACIANO] Calculando relación "
            "Mu/Beta respecto al baseline..."
        )

        relacion_laplaciana, tiempo_relacion_laplaciana = (
            calcular_relacion_mu_beta(
                ensayos_laplacianos
            )
        )

        graficar_relacion_mu_beta(
            relacion_laplaciana,
            tiempo_relacion_laplaciana,
            n_ensayos,
            grupo,
            es_laplaciano=True,
        )

    if not MOSTRAR_FIGURAS:
        plt.close("all")


if __name__ == "__main__":
    main()