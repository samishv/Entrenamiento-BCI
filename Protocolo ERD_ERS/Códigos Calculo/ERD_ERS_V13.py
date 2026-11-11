from pathlib import Path
import re
import random

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

plt.close("all")


# =============================================================================
# SELECCION DEL ENTORNO Y CARPETA DEL PARTICIPANTE
# =============================================================================

# --- SPYDER
#CARPETA_SUJETO = Path(
#    r"D:\luiso\Documentos\Luis\UPIITA\TT\Entrenamiento-BCI\Protocolo ERD_ERS\Signals"
#    r"\SV3"
#)

# --- VSCODE
CARPETA_SUJETO = (Path.cwd() / "Protocolo ERD_ERS" / "Signals" / "SK3")

# =============================================================================
# VARIABLES MODIFICABLES
# =============================================================================

CARPETAS_A_ANALIZAR = {
    "ME": True,
    "MI": True,
}

LATERALIDAD = "diestro"                                     # Opciones: "diestro" o "zurdo"
SEMILLA_PRUEBA_ALEATORIA = 97                               # None = una seleccion distinta en cada ejecucion

# --- Figuras -----------------------------------------------------------------
GUARDAR_FIGURAS = True
MOSTRAR_FIGURAS = False
MAXIMIZAR_FIGURAS = True

# --- Bandas ------------------------------------------------------------------
# Bandas generales: referencia visual en los escalogramas y relacion Mu/Beta "Completa".
BANDAS = {
    "Ritmo Mu":    (8.0, 12.0),
    "Ritmo Beta":  (12.0, 30.0),
}

# Subbandas: graficas temporales de ERD/ERS y relacion Mu/Beta "Delimitada".
BANDAS_ANALISIS = {
    "Ritmo Mu":    (11.0, 12.0),
    "Ritmo Beta":  (17.0, 22.0),
}

# --- Preprocesamiento --------------------------------------------------------
F_LOW = 8.0                                                 
F_HIGH = 30.0
ELIMINAR_DC = True
USAR_ICA = False
IC_TO_REMOVE = None                                         
USAR_LAPLACIANO = True

# --- ERSP con Morlet ---------------------------------------------------------
FREQS_MORLET = np.arange(7.0, 31.0, 1.0)                    # rango del escalograma
N_CYCLES_MORLET_TEMPORAL = np.linspace(6, 15, len(FREQS_MORLET))
N_CYCLES_MORLET_FRECUENCIAL = np.linspace(14, 40, len(FREQS_MORLET))
LIMITE_ERSP_DB = None                                       # p. ej. 3.0 para fijar +-3 dB; None = automatico
CMAP_ERSP = "jet"

# --- FFT deslizante para ERD/ERS porcentual ----------------------------------
FFT_DURACION_VENTANA = 0.8                                  # s (resolucion en frecuencia = 1 / ventana)
FFT_PASO_SEGUNDOS = 0.2                                     # s
SUAVIZADO_SEGUNDOS = 0                                      # solo visual; 0 o None desactiva "MULTIPLO DE PASO"
LIMITES_ERD_ERS = None                                      # p. ej. (-100, 200) en %; None = automatico
LIMITES_RELACION_MU_BETA = None                             # igual que LIMITES_ERD_ERS

# --- Graficas a generar ------------------------------------------------------
GENERAR_GRAFICAS = {
    "1_Resultados_ERD_ERS": {
        "Escalogramas Temporales": True,
        "Escalogramas Frecuenciales": True,
        "Pruebas Empalmadas": True,
        "Relacion Mu Beta": True,
        "Ritmo Beta": False,
        "Ritmo Mu": False,
        "Pruebas Empalmadas Delimitadas": True,
        "Relacion Mu Beta Delimitada": True,
    },
    "2_Resultados_ERD_ERS_Laplaciano": {
        "Escalogramas Temporales": True,
        "Escalogramas Frecuenciales": True,
        "Pruebas Empalmadas": True,
        "Relacion Mu Beta": True,
        "Ritmo Beta": False,
        "Ritmo Mu": False,
        "Pruebas Empalmadas Delimitadas": True,
        "Relacion Mu Beta Delimitada": True,
    },
    "3_Resultados_Prueba_Aleatoria": {
        "Ritmo Beta": False,
        "Ritmo Mu": False,
        "ERSP Temporal": True,
        "ERSP Frecuencial": True,
        "Relacion Mu Beta": True,
        "Relacion Mu Beta Delimitada": True,
    },
    "4_Resultados_Prueba_Aleatoria_Laplaciano": {
        "Ritmo Beta": False,
        "Ritmo Mu": False,
        "ERSP Temporal": True,
        "ERSP Frecuencial": True,
        "Relacion Mu Beta": True,
        "Relacion Mu Beta Delimitada": True,
    },
}


# =============================================================================
# VARIABLES ESTRUCTURALES 
# =============================================================================

PATRON_ARCHIVOS = "*.npy"
FS = 250.0

CANALES = ["Fz", "FC3", "FCz", "FC4", "Cz", "C3", "C4"]

POSICIONES_CANALES = {
    "Fz":  (1, 2),
    "FC3": (2, 1), "FCz": (2, 2), "FC4": (2, 3),
    "C3":  (3, 1), "Cz":  (3, 2), "C4":  (3, 3),
}

T_REPOSO_INICIAL = (0.0, 3.0)
T_TAREA          = (3.0, 8.0)
T_REPOSO_FINAL   = (8.0, 11.0)
T_FIN = T_REPOSO_FINAL[1]

MARGEN_BASELINE = 0.5                                       
BASELINE = (
    T_REPOSO_INICIAL[0] + MARGEN_BASELINE,
    T_REPOSO_INICIAL[1] - MARGEN_BASELINE,
)

FREQS_NOTCH = (60.0, 120.0)
Q_NOTCH = 5.0
BP_ORDER = 3
PAD_MUESTRAS = int(5 * FS)                                  
ICA_RANDOM_STATE = 97

# Morlet
MORLET_DECIM = 4
N_JOBS = 1

# FFT
FFT_NPERSEG = int(FFT_DURACION_VENTANA * FS)
FFT_PASO_MUESTRAS = int(FFT_PASO_SEGUNDOS * FS)
FFT_NFFT = int(FS)                                          # cuadricula FFT de 1 Hz mediante zero-padding
VENTANA_SUAVIZADO = int(round(SUAVIZADO_SEGUNDOS / FFT_PASO_SEGUNDOS)) if SUAVIZADO_SEGUNDOS else 1

# Estilo de figuras
ESTILO_OSCURO = True
FIG_SIZE = (16, 9)
FIG_DPI = 120
FIG_FORMATO = "png"

COLORES_CANALES = {
    "Fz": "#EBEF91", "FC3": "#B5E48C", "FCz": "#99D98C", "FC4": "#6FC39C",
    "Cz": "#3DA2F5", "C3": "#64B5F7", "C4": "#248BE5",
}
COLOR_REPOSO = "#B7EBF6"
COLOR_TAREA = "#F5D6B7"

ALPHA_ENSAYOS_EMPALMADOS = 0.25
GROSOR_PROMEDIO_EMPALMADO = 2.5

if ESTILO_OSCURO:
    plt.style.use("dark_background")


# =============================================================================
# VALORES DERIVADOS (no editar)
# =============================================================================

EPS = np.finfo(np.float64).tiny
SEP = "=" * 50

LATERALIDAD_NORMALIZADA = LATERALIDAD.strip().lower()
if LATERALIDAD_NORMALIZADA == "diestro":
    CANAL_INTERES_LAPLACIANO = "C3"
    CANALES_VECINOS_LAPLACIANO = ["Cz", "FC3"]
elif LATERALIDAD_NORMALIZADA == "zurdo":
    CANAL_INTERES_LAPLACIANO = "C4"
    CANALES_VECINOS_LAPLACIANO = ["Cz", "FC4"]
else:
    raise ValueError('LATERALIDAD debe ser "diestro" o "zurdo".')
NOMBRE_CANAL_LAPLACIANO = f"{CANAL_INTERES_LAPLACIANO} Laplaciano"

MODOS = (False, True) if USAR_LAPLACIANO else (False,)      # es_laplaciano
PREFIJO_ALEATORIA = "PRUEBA_ALEATORIA"

# (prueba_aleatoria, es_laplaciano) -> clave de GENERAR_GRAFICAS
CLAVES = {
    (False, False): "1_Resultados_ERD_ERS",
    (False, True):  "2_Resultados_ERD_ERS_Laplaciano",
    (True, False):  "3_Resultados_Prueba_Aleatoria",
    (True, True):   "4_Resultados_Prueba_Aleatoria_Laplaciano",
}
CARPETAS_RESULTADOS = {
    clave: CARPETA_SUJETO / (
        f"{clave}_{CANAL_INTERES_LAPLACIANO}" if clave.endswith("Laplaciano") else clave
    )
    for clave in CLAVES.values()
}

GRAFICAS_ERD_ERS = {
    "Escalogramas Temporales", "Escalogramas Frecuenciales",
    "Pruebas Empalmadas", "Relacion Mu Beta",
    "Ritmo Beta", "Ritmo Mu",
    "Pruebas Empalmadas Delimitadas", "Relacion Mu Beta Delimitada",
}
GRAFICAS_ALEATORIA = {
    "Ritmo Beta", "Ritmo Mu", "ERSP Temporal", "ERSP Frecuencial",
    "Relacion Mu Beta", "Relacion Mu Beta Delimitada",
}

ERSP_TIPOS = (
    ("Escalogramas Temporales", "ERSP Temporal", "Temporal", N_CYCLES_MORLET_TEMPORAL),
    ("Escalogramas Frecuenciales", "ERSP Frecuencial", "Frecuencial", N_CYCLES_MORLET_FRECUENCIAL),
)

RELACIONES = (
    ("Relacion Mu Beta", BANDAS, "Relacion Mu Beta", "Completa", "con bandas completas"),
    ("Relacion Mu Beta Delimitada", BANDAS_ANALISIS, "Relacion Mu Beta Delimitada", "Delimitada", "con bandas delimitadas"),
)
EMPALMADAS = (
    ("Pruebas Empalmadas", BANDAS, "Pruebas Empalmadas", "Completa", "con bandas completas"),
    ("Pruebas Empalmadas Delimitadas", BANDAS_ANALISIS, "Pruebas Empalmadas Delimitadas", "Delimitada", "con bandas delimitadas"),
)

MODALIDAD = None
CARPETA_DATOS = None


# =============================================================================
# CONFIGURACION Y VALIDACION
# =============================================================================

def configurar_modalidad(modalidad):
    global MODALIDAD, CARPETA_DATOS
    MODALIDAD = modalidad.strip().upper()
    CARPETA_DATOS = CARPETA_SUJETO / MODALIDAD


def validar_configuracion():
    for (aleatoria, _), clave in CLAVES.items():
        esperadas = GRAFICAS_ALEATORIA if aleatoria else GRAFICAS_ERD_ERS
        if clave not in GENERAR_GRAFICAS:
            raise ValueError(f"Falta la configuracion de graficas para: {clave}")
        bloque = GENERAR_GRAFICAS[clave]
        faltantes = esperadas.difference(bloque)
        if faltantes:
            raise ValueError(f"Faltan interruptores en '{clave}': {sorted(faltantes)}")
        no_booleanos = {
            n: v for n, v in bloque.items() if n in esperadas and not isinstance(v, bool)
        }
        if no_booleanos:
            raise TypeError(f"Los interruptores de '{clave}' deben ser True o False: {no_booleanos}")

    if not (T_REPOSO_INICIAL[1] == T_TAREA[0] and T_TAREA[1] == T_REPOSO_FINAL[0]):
        raise ValueError("Los segmentos del protocolo deben ser consecutivos.")
    if not all(ini < fin for ini, fin in (T_REPOSO_INICIAL, T_TAREA, T_REPOSO_FINAL)):
        raise ValueError("Cada segmento del protocolo debe tener inicio < fin.")
    if not (T_REPOSO_INICIAL[0] <= BASELINE[0] < BASELINE[1] <= T_REPOSO_INICIAL[1]):
        raise ValueError(f"BASELINE {BASELINE} debe quedar dentro del reposo inicial.")

    sin_posicion = [c for c in CANALES if c not in POSICIONES_CANALES]
    if sin_posicion:
        raise ValueError(f"Canales sin posicion en POSICIONES_CANALES: {sin_posicion}")

    for nombre, bandas in (("BANDAS", BANDAS), ("BANDAS_ANALISIS", BANDAS_ANALISIS)):
        faltantes = {"Ritmo Mu", "Ritmo Beta"}.difference(bandas)
        if faltantes:
            raise ValueError(f"{nombre} debe incluir 'Ritmo Mu' y 'Ritmo Beta'. Faltan: {sorted(faltantes)}")
    for nombre, (f_ini, f_fin) in BANDAS_ANALISIS.items():
        if not (F_LOW <= f_ini < f_fin <= F_HIGH):
            raise ValueError(
                f"La subbanda {nombre} ({f_ini:g}-{f_fin:g} Hz) debe quedar "
                f"dentro del pasabanda ({F_LOW:g}-{F_HIGH:g} Hz)."
            )

    if USAR_LAPLACIANO:
        vecinos = CANALES_VECINOS_LAPLACIANO
        fuera = [c for c in (CANAL_INTERES_LAPLACIANO, *vecinos) if c not in CANALES]
        if fuera:
            raise ValueError(f"Canales del laplaciano ausentes en CANALES: {fuera}")
        if not vecinos or CANAL_INTERES_LAPLACIANO in vecinos or len(set(vecinos)) != len(vecinos):
            raise ValueError("CANALES_VECINOS_LAPLACIANO invalido (vacio, repetido o contiene el canal de interes).")


# =============================================================================
# CARGA Y PREPROCESAMIENTO
# =============================================================================

def preprocesar_ensayo(datos):
    if ELIMINAR_DC:
        datos = quitar_componente_dc(datos)
    pad = min(PAD_MUESTRAS, datos.shape[1] - 1)
    datos = aplicar_notch_multiple(datos, FS, FREQS_NOTCH, Q_NOTCH, pad)
    return aplicar_pasabanda_butter(datos, FS, F_LOW, F_HIGH, BP_ORDER, pad)


def obtener_dia_desde_ruta(archivo):
    for parte in archivo.relative_to(CARPETA_DATOS).parts[:-1]:
        coincidencia = re.fullmatch(r"D(\d+)_B(\d+)", parte, flags=re.IGNORECASE)
        if coincidencia:
            return f"D{int(coincidencia.group(1))}"
    return None


def cargar_ensayos_por_dia():
    archivos = sorted(CARPETA_DATOS.rglob(PATRON_ARCHIVOS))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos {PATRON_ARCHIVOS} en:\n{CARPETA_DATOS}")

    n_fin = int(round(T_FIN * FS))
    ensayos_por_dia = {}
    archivos_por_dia = {}

    for archivo in archivos:
        nombre_relativo = archivo.relative_to(CARPETA_DATOS)
        dia = obtener_dia_desde_ruta(archivo)

        if dia is None:
            print(f"   [OMITIDO] {nombre_relativo}: fuera de carpeta DX_BX")
            continue

        try:
            datos = cargar_npy_eeg(str(archivo), n_canales_esperados=len(CANALES))
            if not np.isfinite(datos).all():
                raise ValueError("contiene NaN o infinitos")
            if datos.shape[1] < n_fin:
                raise ValueError(f"dura {datos.shape[1] / FS:.2f} s y se requieren {T_FIN:g} s")
        except ValueError as error:
            print(f"   [OMITIDO] {nombre_relativo}: {error}")
            continue

        ensayos_por_dia.setdefault(dia, []).append(preprocesar_ensayo(datos)[:, :n_fin])
        archivos_por_dia.setdefault(dia, []).append(archivo)

    if not ensayos_por_dia:
        raise RuntimeError("No se encontró ningún archivo válido dentro de carpetas DX_BX.")

    dias_ordenados = sorted(ensayos_por_dia, key=lambda d: int(d[1:]))
    return (
        {dia: np.stack(ensayos_por_dia[dia], axis=0) for dia in dias_ordenados},
        {dia: archivos_por_dia[dia] for dia in dias_ordenados},
    )


def aplicar_ica(ensayos):
    if not USAR_ICA:
        return ensayos

    if IC_TO_REMOVE is None:
        print(" [ICA] sin componente a excluir (IC_TO_REMOVE = None)")

    limpios_v, _ = ica_eliminar_componente(
        np.concatenate(ensayos, axis=-1) * 1e-6,
        fs=FS,
        ch_names=CANALES,
        ic_to_remove=IC_TO_REMOVE,
        random_state=ICA_RANDOM_STATE,
    )

    return np.stack(np.split(limpios_v * 1e6, len(ensayos), axis=-1), axis=0)


def aplicar_laplaciano_ensayos(ensayos):
    senales = [
        filtro_laplaciano(
            datos=ensayo,
            canales=CANALES,
            canal_interes=CANAL_INTERES_LAPLACIANO,
            canales_vecinos=CANALES_VECINOS_LAPLACIANO,
        )
        for ensayo in ensayos
    ]

    return np.stack(senales, axis=0)[:, np.newaxis, :]


# =============================================================================
# CALCULOS (potencia de todos los ensayos en una sola llamada)
# =============================================================================

def intervalo_a_mascara(tiempo, intervalo):
    return (tiempo >= intervalo[0]) & (tiempo <= intervalo[1])


def promedio_con_baseline(potencias, tiempo):
    """potencias: (ensayos, canales, frecuencias, tiempo)."""
    mascara = intervalo_a_mascara(tiempo, BASELINE)
    if not mascara.any():
        raise ValueError("El baseline no contiene puntos temporales.")

    suma = np.zeros_like(potencias[0])
    base = np.zeros(potencias[0].shape[:-1])
    for potencia in potencias:
        suma += potencia
        base += potencia[..., mascara].mean(axis=-1)

    n = len(potencias)
    return suma / n, base / n


def potencia_stft(ensayos):
    frecuencias, tiempo, zxx = signal.stft(
        ensayos,
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

    return np.abs(zxx[..., mascara, :]) ** 2, tiempo


def frecuencias_fft():
    frecuencias = np.fft.rfftfreq(FFT_NFFT, 1.0 / FS)
    return frecuencias[(frecuencias >= F_LOW) & (frecuencias <= F_HIGH)]


def calcular_ersp(ensayos, n_cycles_morlet):
    potencias = tfr_array_morlet(
        ensayos,
        sfreq=FS,
        freqs=FREQS_MORLET,
        n_cycles=n_cycles_morlet,
        output="power",
        zero_mean=True,
        decim=MORLET_DECIM,
        n_jobs=N_JOBS,
        verbose=False,
    )
    tiempo = np.arange(potencias.shape[-1]) * MORLET_DECIM / FS
    potencia, base = promedio_con_baseline(potencias, tiempo)
    ersp_db = 10.0 * np.log10((potencia + EPS) / (base[..., np.newaxis] + EPS))

    return ersp_db, tiempo


def calcular_erd_ers(potencia, base):
    frecuencias = frecuencias_fft()
    cambio = 100.0 * (potencia / (base[..., np.newaxis] + EPS) - 1.0)
    erd_ers = {}

    for banda, (f_ini, f_fin) in BANDAS_ANALISIS.items():
        mascara = (frecuencias >= f_ini) & (frecuencias <= f_fin)
        if not mascara.any():
            raise ValueError(
                f"No hay bins FFT en la banda {banda} ({f_ini:g}-{f_fin:g} Hz). "
                f"Bins disponibles: {frecuencias}"
            )
        erd_ers[banda] = cambio[:, mascara, :].mean(axis=1)

    return erd_ers


def calcular_relacion_mu_beta(potencia, tiempo, bandas):
    """Mu/Beta normalizada contra el baseline. potencia: (..., canales, frecuencias, tiempo)."""
    frecuencias = frecuencias_fft()
    f_mu_ini, f_mu_fin = bandas["Ritmo Mu"]
    f_beta_ini, f_beta_fin = bandas["Ritmo Beta"]

    mascara_mu = (frecuencias >= f_mu_ini) & (frecuencias <= f_mu_fin)
    mascara_beta = (frecuencias >= f_beta_ini) & (frecuencias <= f_beta_fin)

    for nombre, mascara, (f_ini, f_fin) in (
        ("Mu", mascara_mu, (f_mu_ini, f_mu_fin)),
        ("Beta", mascara_beta, (f_beta_ini, f_beta_fin)),
    ):
        if not mascara.any():
            raise ValueError(
                f"No hay bins FFT dentro de la banda {nombre} "
                f"({f_ini:g}-{f_fin:g} Hz). Bins disponibles: {frecuencias}"
            )

    potencia_mu = potencia[..., mascara_mu, :].mean(axis=-2)
    potencia_beta = potencia[..., mascara_beta, :].mean(axis=-2)
    relacion = (potencia_mu + EPS) / (potencia_beta + EPS)

    relacion_baseline = relacion[..., intervalo_a_mascara(tiempo, BASELINE)].mean(axis=-1)

    return 100.0 * (relacion / (relacion_baseline[..., np.newaxis] + EPS) - 1.0)


# =============================================================================
# FIGURAS
# =============================================================================

def crear_malla(titulo, es_laplaciano=False, leyenda=True):
    if es_laplaciano:
        fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
        ejes = {CANAL_INTERES_LAPLACIANO: ax}
        etiquetas = {
            CANAL_INTERES_LAPLACIANO: (
                f"{CANAL_INTERES_LAPLACIANO} - "
                f"promedio({', '.join(CANALES_VECINOS_LAPLACIANO)})"
            )
        }
        destino_leyenda, loc_leyenda, tam_leyenda = ax, "best", 11
    else:
        fig, axes = plt.subplots(3, 3, figsize=FIG_SIZE, constrained_layout=True)
        ejes = {
            c: axes[POSICIONES_CANALES[c][0] - 1, POSICIONES_CANALES[c][1] - 1]
            for c in CANALES
        }
        etiquetas = {c: c for c in CANALES}
        libres = [ax for ax in axes.ravel() if ax not in ejes.values()]

        for ax in libres:
            ax.set_axis_off()

        destino_leyenda, loc_leyenda, tam_leyenda = (
            libres[0] if libres else None, "center", 13
        )

    fig.suptitle(titulo, fontsize=18, fontweight="bold")

    for canal, ax in ejes.items():
        ax.set_title(etiquetas[canal], fontsize=14, fontweight="bold")

    if leyenda and destino_leyenda is not None:
        destino_leyenda.legend(
            handles=[
                Patch(color=COLOR_REPOSO, alpha=0.4, label="Reposo"),
                Patch(color=COLOR_TAREA, alpha=0.4, label="Tarea"),
            ],
            loc=loc_leyenda,
            fontsize=tam_leyenda,
        )

    return fig, ejes


def marcar_protocolo(ax, sombrear):
    if sombrear:
        for intervalo, color in (
            (T_REPOSO_INICIAL, COLOR_REPOSO),
            (T_TAREA, COLOR_TAREA),
            (T_REPOSO_FINAL, COLOR_REPOSO),
        ):
            ax.axvspan(*intervalo, color=color, alpha=0.2)

    for t in T_TAREA:
        ax.axvline(t, color="gray" if sombrear else "black", linestyle="--", linewidth=1.2)

    ax.set_xlim(T_REPOSO_INICIAL[0], T_FIN)


def suavizar(curvas):
    if VENTANA_SUAVIZADO > 1:
        return uniform_filter1d(curvas, size=VENTANA_SUAVIZADO, axis=-1, mode="nearest")
    return curvas


def limites_con_margen(y_min, y_max):
    y_min, y_max = min(y_min, 0.0), max(y_max, 0.0)
    margen = 0.05 * (y_max - y_min or 1.0)
    return y_min - margen, y_max + margen


def nombre_imagen_por_modalidad(nombre):
    """Coloca ME o MI al inicio del archivo y evita repetir la modalidad."""
    partes = [parte for parte in nombre.split("_") if parte != MODALIDAD]
    return f"{MODALIDAD}_{'_'.join(partes)}"


def finalizar_figura(fig, nombre, subcarpeta, grupo, es_laplaciano):
    """Guarda la figura. La prueba aleatoria va directo a su carpeta, sin grupo en el nombre."""
    if MAXIMIZAR_FIGURAS and MOSTRAR_FIGURAS:
        pantalla_completa(fig)

    if GUARDAR_FIGURAS:
        aleatoria = grupo.startswith(PREFIJO_ALEATORIA)
        carpeta = CARPETAS_RESULTADOS[CLAVES[(aleatoria, es_laplaciano)]]
        if not aleatoria:
            carpeta = carpeta / subcarpeta
            nombre = f"{nombre}_{grupo}"
        carpeta.mkdir(parents=True, exist_ok=True)

        ruta = carpeta / f"{nombre_imagen_por_modalidad(nombre)}.{FIG_FORMATO}"
        fig.savefig(ruta, dpi=FIG_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())

    if not MOSTRAR_FIGURAS:
        plt.close(fig)


def graficar_ersp(ersp_db, tiempo, n_ensayos, grupo, tipo_analisis, es_laplaciano=False):
    canales = [CANAL_INTERES_LAPLACIANO] if es_laplaciano else CANALES
    forma_esperada = (len(canales), len(FREQS_MORLET), len(tiempo))
    if ersp_db.shape != forma_esperada:
        raise ValueError(
            f"Forma del ERSP {ersp_db.shape} distinta de la esperada {forma_esperada} "
            "(canales, frecuencias, tiempos)."
        )

    fig, ejes = crear_malla(
        f"ERSP {tipo_analisis}{' Laplaciano' if es_laplaciano else ''} | "
        f"{MODALIDAD} | {grupo} | {n_ensayos} Ensayos",
        es_laplaciano,
        leyenda=False,
    )

    if LIMITE_ERSP_DB is not None:
        limite = float(LIMITE_ERSP_DB)
    else:
        valores = np.abs(ersp_db[np.isfinite(ersp_db)])
        limite = max(1.0, float(np.percentile(valores, 98.0)) if valores.size else 1.0)

    f_min, f_max = FREQS_MORLET[0], FREQS_MORLET[-1]
    bordes_bandas = [
        f for f in sorted({f for rango in BANDAS.values() for f in rango})
        if f_min <= f <= f_max
    ]
    ticks_frecuencia = [f for f in (8, 10, 12, 14, 17, 20, 25, 30) if f_min <= f <= f_max]
    fuente = {"fontsize": 12} if es_laplaciano else {}

    for i, ax in enumerate(ejes.values()):
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
            ax.axhline(frecuencia, color="black", linestyle=":", linewidth=1.2)

        ax.set_yscale("log")
        ax.set_ylim(f_min, f_max)
        ax.set_yticks(ticks_frecuencia)
        ax.set_yticklabels([str(f) for f in ticks_frecuencia])
        ax.set_xlabel("Tiempo (s)", **fuente)
        ax.set_ylabel("Frecuencia (Hz)", **fuente)

    barra = fig.colorbar(imagen, ax=list(ejes.values()), shrink=0.85, pad=0.02)
    barra.set_label("ERSP (dB)", fontsize=13, fontweight="bold")

    finalizar_figura(
        fig,
        nombre=f"ERSP_Morlet_{tipo_analisis}_{'Laplaciano_' if es_laplaciano else ''}{MODALIDAD}",
        subcarpeta=(
            "Escalogramas_Temporales"
            if tipo_analisis == "Temporal"
            else "Escalogramas_Frecuenciales"
        ),
        grupo=grupo,
        es_laplaciano=es_laplaciano,
    )


def graficar_curvas(curvas, tiempo, titulo, etiqueta_y, limites, nombre, subcarpeta, grupo, es_laplaciano):
    fig, ejes = crear_malla(titulo, es_laplaciano)

    curvas = suavizar(curvas)
    y_min, y_max = limites if limites is not None else limites_con_margen(
        float(curvas.min()), float(curvas.max())
    )

    for i, (canal, ax) in enumerate(ejes.items()):
        marcar_protocolo(ax, sombrear=True)
        ax.axhline(0.0, linewidth=1.0, alpha=0.8)
        ax.plot(
            tiempo,
            curvas[i],
            color=COLORES_CANALES.get(canal, "C0"),
            linewidth=1.8 if es_laplaciano else 1.6,
        )
        ax.set_ylim(y_min, y_max)
        ax.set(xlabel="Tiempo (s)", ylabel=etiqueta_y)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)

    finalizar_figura(fig, nombre, subcarpeta, grupo, es_laplaciano)


def graficar_erd_ers(erd_ers, tiempo, banda, n_ensayos, grupo, es_laplaciano=False):
    f_ini, f_fin = BANDAS_ANALISIS[banda]

    graficar_curvas(
        erd_ers[banda],
        tiempo,
        titulo=(
            f"% ERD/ERS | {MODALIDAD} | {grupo} | "
            f"{NOMBRE_CANAL_LAPLACIANO + ' | ' if es_laplaciano else ''}"
            f"{banda} ({f_ini:g}-{f_fin:g} Hz) | {n_ensayos} Ensayos"
        ),
        etiqueta_y="Cambio vs baseline (%)",
        limites=LIMITES_ERD_ERS,
        nombre=f"ERD_ERS_{'Laplaciano_' if es_laplaciano else ''}{MODALIDAD}_{banda}",
        subcarpeta=banda,
        grupo=grupo,
        es_laplaciano=es_laplaciano,
    )


def graficar_relacion_mu_beta(
    relacion_mu_beta, tiempo, n_ensayos, grupo, bandas_relacion,
    subcarpeta, sufijo_archivo, tipo_relacion, es_laplaciano=False,
):
    f_mu_ini, f_mu_fin = bandas_relacion["Ritmo Mu"]
    f_beta_ini, f_beta_fin = bandas_relacion["Ritmo Beta"]

    graficar_curvas(
        relacion_mu_beta,
        tiempo,
        titulo=(
            f"Relación Mu/Beta {tipo_relacion} vs baseline | "
            f"Mu {f_mu_ini:g}-{f_mu_fin:g} Hz / "
            f"Beta {f_beta_ini:g}-{f_beta_fin:g} Hz | "
            f"{MODALIDAD} | {grupo} | "
            f"{NOMBRE_CANAL_LAPLACIANO + ' | ' if es_laplaciano else ''}"
            f"{n_ensayos} Ensayos"
        ),
        etiqueta_y=(
            "Cambio de la relación Mu/Beta vs baseline (%)"
            if es_laplaciano
            else "Cambio Mu/Beta vs baseline (%)"
        ),
        limites=LIMITES_RELACION_MU_BETA,
        nombre=f"Relacion_Mu_Beta_{sufijo_archivo}_{'Laplaciano_' if es_laplaciano else ''}{MODALIDAD}",
        subcarpeta=subcarpeta,
        grupo=grupo,
        es_laplaciano=es_laplaciano,
    )


def graficar_pruebas_empalmadas(
    relaciones, tiempo, grupo, bandas_relacion,
    subcarpeta, sufijo_archivo, tipo_relacion, es_laplaciano=False,
):
    """Superpone los ensayos y destaca el promedio. relaciones: (ensayos, canales, tiempo)."""
    relaciones = suavizar(relaciones)
    promedio = relaciones.mean(axis=0)

    if LIMITES_RELACION_MU_BETA is not None:
        y_min, y_max = LIMITES_RELACION_MU_BETA
    else:
        valores = relaciones[np.isfinite(relaciones)]
        y_min, y_max = limites_con_margen(
            float(np.percentile(valores, 3)), float(np.percentile(valores, 97))
        )

    f_mu_ini, f_mu_fin = bandas_relacion["Ritmo Mu"]
    f_beta_ini, f_beta_fin = bandas_relacion["Ritmo Beta"]
    fig, ejes = crear_malla(
        f"Relación Mu/Beta {tipo_relacion} ensayo a ensayo | "
        f"Mu {f_mu_ini:g}-{f_mu_fin:g} Hz / "
        f"Beta {f_beta_ini:g}-{f_beta_fin:g} Hz | "
        f"{MODALIDAD} | {grupo} | "
        f"{NOMBRE_CANAL_LAPLACIANO + ' | ' if es_laplaciano else ''}"
        f"{relaciones.shape[0]} Ensayos",
        es_laplaciano,
        leyenda=False,
    )

    for i, (canal, ax) in enumerate(ejes.items()):
        marcar_protocolo(ax, sombrear=True)
        ax.axhline(0.0, linewidth=1.0, alpha=0.8)
        color = COLORES_CANALES.get(canal, "C0")
        ax.plot(tiempo, relaciones[:, i, :].T, color=color, linewidth=0.8, alpha=ALPHA_ENSAYOS_EMPALMADOS)
        ax.plot(
            tiempo, promedio[i], color=color,
            linewidth=GROSOR_PROMEDIO_EMPALMADO, alpha=1.0, label="Promedio",
        )
        ax.set_ylim(y_min, y_max)
        ax.set(xlabel="Tiempo (s)", ylabel="Cambio Mu/Beta vs baseline (%)")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
        ax.legend(loc="best")

    finalizar_figura(
        fig,
        nombre=f"Relacion_Mu_Beta_Empalmada_{sufijo_archivo}_{'Laplaciano_' if es_laplaciano else ''}{MODALIDAD}",
        subcarpeta=subcarpeta,
        grupo=grupo,
        es_laplaciano=es_laplaciano,
    )


# =============================================================================
# FLUJO DE ANALISIS
# =============================================================================

def generar_graficas(datos, grupo, es_laplaciano, aleatoria=False):
    """Calcula y guarda las figuras habilitadas. La STFT se calcula una sola vez."""
    activar = GENERAR_GRAFICAS[CLAVES[(aleatoria, es_laplaciano)]]
    n_figuras = sum(bool(v) for v in activar.values())
    modo = f"Laplaciano {CANAL_INTERES_LAPLACIANO}" if es_laplaciano else "Multicanal"
    print(f" [{'ALEATORIA' if aleatoria else grupo}] {modo}: {n_figuras} fig")
    if not n_figuras:
        return

    n_ensayos = datos.shape[0]
    if es_laplaciano:
        datos = aplicar_laplaciano_ensayos(datos)

    for clave, clave_aleatoria, tipo, ciclos in ERSP_TIPOS:
        if activar[clave_aleatoria if aleatoria else clave]:
            ersp_db, tiempo = calcular_ersp(datos, ciclos)
            graficar_ersp(ersp_db, tiempo, n_ensayos, grupo, tipo, es_laplaciano)

    bandas_activas = [b for b in BANDAS_ANALISIS if activar.get(b, False)]
    relaciones_activas = [r for r in RELACIONES if activar.get(r[0], False)]
    empalmadas_activas = [] if aleatoria else [e for e in EMPALMADAS if activar.get(e[0], False)]
    if not (bandas_activas or relaciones_activas or empalmadas_activas):
        return

    potencias, tiempo = potencia_stft(datos)
    potencia, base = promedio_con_baseline(potencias, tiempo)

    if bandas_activas:
        erd_ers = calcular_erd_ers(potencia, base)
        for banda in bandas_activas:
            graficar_erd_ers(erd_ers, tiempo, banda, n_ensayos, grupo, es_laplaciano)

    for _, bandas, subcarpeta, sufijo, descripcion in relaciones_activas:
        relacion = calcular_relacion_mu_beta(potencia, tiempo, bandas)
        graficar_relacion_mu_beta(
            relacion, tiempo, n_ensayos, grupo, bandas,
            subcarpeta, sufijo, descripcion, es_laplaciano,
        )

    for _, bandas, subcarpeta, sufijo, descripcion in empalmadas_activas:
        relaciones = calcular_relacion_mu_beta(potencias, tiempo, bandas)
        graficar_pruebas_empalmadas(
            relaciones, tiempo, grupo, bandas,
            subcarpeta, sufijo, descripcion, es_laplaciano,
        )


def analizar_grupo(ensayos, grupo):
    ensayos = aplicar_ica(ensayos)
    for es_laplaciano in MODOS:
        generar_graficas(ensayos, grupo, es_laplaciano)


def analizar_prueba_aleatoria(ensayo, archivo):
    identificador = re.sub(r"[^A-Za-z0-9_-]+", "_", archivo.stem)
    grupo = f"{PREFIJO_ALEATORIA}_{identificador}"
    print(f"\n [ALEATORIA] {archivo.relative_to(CARPETA_DATOS)}")
    for es_laplaciano in MODOS:
        generar_graficas(ensayo[np.newaxis, ...], grupo, es_laplaciano, aleatoria=True)


def analizar_modalidad_actual():
    sigma_t = N_CYCLES_MORLET_TEMPORAL / (2 * np.pi * FREQS_MORLET)
    sigma_f = N_CYCLES_MORLET_FRECUENCIAL / (2 * np.pi * FREQS_MORLET)
    print(f" {LATERALIDAD_NORMALIZADA} | baseline {BASELINE} s | FFT {FS / FFT_NFFT:.2f} Hz")
    print(
        f" Morlet sigma (s): T {sigma_t[0]:.3f}-{sigma_t[-1]:.3f} | "
        f"F {sigma_f[0]:.3f}-{sigma_f[-1]:.3f}"
    )

    ensayos_por_dia, archivos_por_dia = cargar_ensayos_por_dia()
    total = sum(e.shape[0] for e in ensayos_por_dia.values())
    resumen = "  ".join(f"{dia}={e.shape[0]}" for dia, e in ensayos_por_dia.items())
    print(f" Ensayos: {resumen}  GEN={total}\n")

    for dia, ensayos_dia in ensayos_por_dia.items():
        analizar_grupo(ensayos_dia, dia)
    analizar_grupo(np.concatenate(list(ensayos_por_dia.values()), axis=0), "GENERAL")

    pares_pruebas = [
        (ensayo, archivo)
        for dia in ensayos_por_dia
        for ensayo, archivo in zip(ensayos_por_dia[dia], archivos_por_dia[dia])
    ]
    ensayo_aleatorio, archivo_aleatorio = random.Random(SEMILLA_PRUEBA_ALEATORIA).choice(pares_pruebas)
    analizar_prueba_aleatoria(ensayo_aleatorio, archivo_aleatorio)

    if MOSTRAR_FIGURAS:
        plt.show()


def main():
    no_booleanos = {
        modalidad: activar
        for modalidad, activar in CARPETAS_A_ANALIZAR.items()
        if not isinstance(activar, bool)
    }
    if no_booleanos:
        raise TypeError(
            f"Los interruptores de CARPETAS_A_ANALIZAR deben ser True o False: {no_booleanos}"
        )

    modalidades_activas = [m.upper() for m, activar in CARPETAS_A_ANALIZAR.items() if activar]
    if not modalidades_activas:
        raise ValueError("No hay modalidades activadas. Activa al menos MI o ME en CARPETAS_A_ANALIZAR.")

    if not CARPETA_SUJETO.is_dir():
        raise FileNotFoundError(
            "No existe la carpeta del participante configurada en CARPETA_SUJETO:\n"
            f"{CARPETA_SUJETO}\n\n"
            "Si ejecutas en VS Code, verifica que Path.cwd() sea la carpeta raiz "
            "'Entrenamiento-BCI'. Si ejecutas en Spyder, verifica la ruta absoluta."
        )

    validar_configuracion()
    print(f"\n{SEP}\n ERD/ERS | {CARPETA_SUJETO.name} | {', '.join(modalidades_activas)}\n{SEP}")

    for numero, modalidad in enumerate(modalidades_activas, start=1):
        configurar_modalidad(modalidad)
        if not CARPETA_DATOS.is_dir():
            raise FileNotFoundError(
                f"La modalidad '{MODALIDAD}' está activada, pero no existe:\n{CARPETA_DATOS}"
            )

        print(f"\n{SEP}\n CORRIDA {numero}/{len(modalidades_activas)} | {MODALIDAD}\n{SEP}")
        analizar_modalidad_actual()
        plt.close("all")

    print(f"\n{SEP}\n TERMINADO\n Resultados en: {CARPETA_SUJETO}\n{SEP}")


if __name__ == "__main__":
    main()