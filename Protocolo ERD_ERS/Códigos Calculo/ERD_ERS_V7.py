from pathlib import Path
import re
import random
import shutil

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

# CARPETA_DATOS = Path.cwd() / "Protocolo ERD_ERS" / "Signals" / "SV2" / "MI"

CARPETA_DATOS = Path(
   r"D:\luiso\Documentos\Luis\UPIITA\TT\Entrenamiento-BCI\Protocolo ERD_ERS\Signals"
   r"\SV3\MI"
)

PATRON_ARCHIVOS = "*.npy"
LATERALIDAD = "diestro"  # Opciones: "diestro" o "zurdo"
SEMILLA_PRUEBA_ALEATORIA = 97  # Usar None para una seleccion distinta en cada ejecucion
CARPETA_RESULTADOS = CARPETA_DATOS.parent / "1_Resultados_ERD_ERS"
CARPETA_PRUEBA_ALEATORIA = CARPETA_DATOS.parent / "3_Resultados_Prueba_Aleatoria"
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
FIG_DPI = 150
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

CARPETA_LAPLACIANO = CARPETA_RESULTADOS.parent / f"2_Resultados_ERD_ERS_Laplaciano_{CANAL_INTERES_LAPLACIANO}"
CARPETA_PRUEBA_ALEATORIA_LAPLACIANO = CARPETA_RESULTADOS.parent / f"4_Resultados_Prueba_Aleatoria_Laplaciano_{CANAL_INTERES_LAPLACIANO}"
ALPHA_ENSAYOS_EMPALMADOS = 0.18
GROSOR_PROMEDIO_EMPALMADO = 2.4
VENTANA_SUAVIZADO = int(round(SUAVIZADO_SEGUNDOS / FFT_PASO_SEGUNDOS)) if SUAVIZADO_SEGUNDOS else 1

SEP_DOBLE = "=" * 70
SEP_SIMPLE = "-" * 70


def validar_configuracion():
    if not (T_REPOSO_INICIAL[1] == T_TAREA[0] and T_TAREA[1] == T_REPOSO_FINAL[0]):
        raise ValueError("Los segmentos del protocolo deben ser consecutivos.")
    if not all(ini < fin for ini, fin in (T_REPOSO_INICIAL, T_TAREA, T_REPOSO_FINAL)):
        raise ValueError("Cada segmento del protocolo debe tener inicio < fin.")
    if not (T_REPOSO_INICIAL[0] <= BASELINE[0] < BASELINE[1] <= T_REPOSO_INICIAL[1]):
        raise ValueError(f"BASELINE {BASELINE} debe quedar dentro del reposo inicial.")

    faltantes = [c for c in CANALES if c not in POSICIONES_CANALES]
    if faltantes:
        raise ValueError(f"Canales sin posicion en POSICIONES_CANALES: {faltantes}")

    bandas_faltantes = {"Ritmo Mu", "Ritmo Beta"}.difference(BANDAS)
    if bandas_faltantes:
        raise ValueError(
            "Para calcular la relación Mu/Beta deben existir las bandas "
            f"'Ritmo Mu' y 'Ritmo Beta'. Faltan: {sorted(bandas_faltantes)}"
        )

    if not USAR_LAPLACIANO:
        return

    vecinos = CANALES_VECINOS_LAPLACIANO

    if CANAL_INTERES_LAPLACIANO not in CANALES:
        raise ValueError(
            f"El canal de interés '{CANAL_INTERES_LAPLACIANO}' no se encuentra en CANALES."
        )
    if not vecinos:
        raise ValueError("CANALES_VECINOS_LAPLACIANO debe contener al menos un canal.")

    vecinos_faltantes = [c for c in vecinos if c not in CANALES]
    if vecinos_faltantes:
        raise ValueError(
            f"Los siguientes canales vecinos no se encuentran en CANALES: {vecinos_faltantes}"
        )
    if CANAL_INTERES_LAPLACIANO in vecinos:
        raise ValueError("El canal de interés no puede aparecer también como vecino.")
    if len(set(vecinos)) != len(vecinos):
        raise ValueError("CANALES_VECINOS_LAPLACIANO contiene canales repetidos.")


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
        raise FileNotFoundError(
            f"No se encontraron archivos {PATRON_ARCHIVOS} en:\n{CARPETA_DATOS}"
        )

    n_fin = int(round(T_FIN * FS))
    ensayos_por_dia = {}
    archivos_por_dia = {}

    print(f" Archivos '.npy' encontrados: {len(archivos)}\n")

    for archivo in archivos:
        nombre_relativo = archivo.relative_to(CARPETA_DATOS)
        dia = obtener_dia_desde_ruta(archivo)

        if dia is None:
            print(
                f"   [OMITIDO] {nombre_relativo}: "
                "no está dentro de una carpeta con formato DX_BX"
            )
            continue

        try:
            datos = cargar_npy_eeg(str(archivo), n_canales_esperados=len(CANALES))

            if not np.isfinite(datos).all():
                raise ValueError("contiene NaN o infinitos")

            if datos.shape[1] < n_fin:
                raise ValueError(
                    f"dura {datos.shape[1] / FS:.2f} s "
                    f"y el protocolo requiere {T_FIN:g} s"
                )

        except ValueError as error:
            print(f"   [OMITIDO] {nombre_relativo}: {error}")
            continue

        ensayos_por_dia.setdefault(dia, []).append(preprocesar_ensayo(datos)[:, :n_fin])
        archivos_por_dia.setdefault(dia, []).append(archivo)
        print(f"   [OK]      {nombre_relativo} -> {dia}")

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
        print("\n [ICA] Activada sin componente a excluir (IC_TO_REMOVE = None).")

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
    potencia, _, tiempo = promedio_con_baseline(ensayos, potencia_stft)
    frecuencias = frecuencias_fft()

    (f_mu_ini, f_mu_fin), (f_beta_ini, f_beta_fin) = BANDAS["Ritmo Mu"], BANDAS["Ritmo Beta"]

    mascara_mu = (frecuencias >= f_mu_ini) & (frecuencias < f_mu_fin)
    mascara_beta = (frecuencias >= f_beta_ini) & (frecuencias <= f_beta_fin)

    for nombre, mascara, (f_ini, f_fin) in (
        ("Mu", mascara_mu, (f_mu_ini, f_mu_fin)),
        ("Beta", mascara_beta, (f_beta_ini, f_beta_fin)),
    ):
        if not mascara.any():
            raise ValueError(
                f"No hay bins FFT dentro de la banda {nombre} ({f_ini:g}-{f_fin:g} Hz)."
            )

    relacion = (
        (potencia[:, mascara_mu, :].mean(axis=1) + EPS)
        / (potencia[:, mascara_beta, :].mean(axis=1) + EPS)
    )

    relacion_baseline = relacion[:, intervalo_a_mascara(tiempo, BASELINE)].mean(axis=-1)

    return 100.0 * (relacion / (relacion_baseline[:, np.newaxis] + EPS) - 1.0), tiempo


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


def finalizar_figura(fig, nombre, subcarpeta, es_laplaciano=False):
    if MAXIMIZAR_FIGURAS and MOSTRAR_FIGURAS:
        pantalla_completa(fig)

    if GUARDAR_FIGURAS:
        carpeta_destino = (CARPETA_LAPLACIANO if es_laplaciano else CARPETA_RESULTADOS) / subcarpeta
        carpeta_destino.mkdir(parents=True, exist_ok=True)

        ruta = carpeta_destino / f"{nombre}.{FIG_FORMATO}"

        fig.savefig(ruta, dpi=FIG_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())

        print(f"     [GUARDADO] {ruta.relative_to(CARPETA_RESULTADOS.parent)} ✅")


def graficar_ersp(ersp_db, tiempo, n_ensayos, grupo, es_laplaciano=False):
    canales = [CANAL_INTERES_LAPLACIANO] if es_laplaciano else CANALES
    forma_esperada = (len(canales), len(FREQS_MORLET), len(tiempo))

    if ersp_db.shape != forma_esperada:
        raise ValueError(
            f"Forma del ERSP {ersp_db.shape} distinta de la esperada {forma_esperada} "
            "(canales, frecuencias, tiempos)."
        )

    fig, ejes = crear_malla(
        f"ERSP{' Laplaciano' if es_laplaciano else ''} | {MODALIDAD} | {grupo} | "
        f"{n_ensayos} Ensayos",
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
        nombre=f"ERSP_Morlet_{'Laplaciano_' if es_laplaciano else ''}{MODALIDAD}_{grupo}",
        subcarpeta="Escalogramas",
        es_laplaciano=es_laplaciano,
    )


def graficar_erd_ers(erd_ers, tiempo, banda, n_ensayos, grupo, es_laplaciano=False):
    f_ini, f_fin = BANDAS[banda]

    fig, ejes = crear_malla(
        f"% ERD/ERS | {MODALIDAD} | {grupo} | "
        f"{NOMBRE_CANAL_LAPLACIANO + ' | ' if es_laplaciano else ''}"
        f"{banda} ({f_ini:g}-{f_fin:g} Hz) | {n_ensayos} Ensayos",
        es_laplaciano,
    )

    curvas = erd_ers[banda]

    if VENTANA_SUAVIZADO > 1:
        curvas = uniform_filter1d(curvas, size=VENTANA_SUAVIZADO, axis=-1, mode="nearest")

    if LIMITES_ERD_ERS is not None:
        y_min, y_max = LIMITES_ERD_ERS
    else:
        y_min = min(float(curvas.min()), 0.0)
        y_max = max(float(curvas.max()), 0.0)
        margen = 0.05 * (y_max - y_min or 1.0)
        y_min -= margen
        y_max += margen

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
        ax.set(xlabel="Tiempo (s)", ylabel="Cambio vs baseline (%)")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)

    finalizar_figura(
        fig,
        nombre=f"ERD_ERS_{'Laplaciano_' if es_laplaciano else ''}{MODALIDAD}_{banda}_{grupo}",
        subcarpeta=banda,
        es_laplaciano=es_laplaciano,
    )


def graficar_relacion_mu_beta(relacion_mu_beta, tiempo, n_ensayos, grupo, es_laplaciano=False):
    fig, ejes = crear_malla(
        f"Relación Mu/Beta vs baseline | {MODALIDAD} | {grupo} | "
        f"{NOMBRE_CANAL_LAPLACIANO + ' | ' if es_laplaciano else ''}"
        f"{n_ensayos} Ensayos",
        es_laplaciano,
    )

    curvas = relacion_mu_beta

    if VENTANA_SUAVIZADO > 1:
        curvas = uniform_filter1d(curvas, size=VENTANA_SUAVIZADO, axis=-1, mode="nearest")

    if LIMITES_RELACION_MU_BETA is not None:
        y_min, y_max = LIMITES_RELACION_MU_BETA
    else:
        y_min = min(float(curvas.min()), 0.0)
        y_max = max(float(curvas.max()), 0.0)
        margen = 0.05 * (y_max - y_min or 1.0)
        y_min -= margen
        y_max += margen

    etiqueta_y = (
        "Cambio de la relación Mu/Beta vs baseline (%)"
        if es_laplaciano
        else "Cambio Mu/Beta vs baseline (%)"
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

    finalizar_figura(
        fig,
        nombre=f"Relacion_Mu_Beta_{'Laplaciano_' if es_laplaciano else ''}{MODALIDAD}_{grupo}",
        subcarpeta="Relacion Mu Beta",
        es_laplaciano=es_laplaciano,
    )


def calcular_relaciones_individuales(ensayos):
    """Calcula una curva Mu/Beta normalizada por baseline para cada ensayo."""
    relaciones = []
    tiempo_referencia = None
    for ensayo in ensayos:
        relacion, tiempo = calcular_relacion_mu_beta(ensayo[np.newaxis, ...])
        relaciones.append(relacion)
        if tiempo_referencia is None:
            tiempo_referencia = tiempo
        elif not np.allclose(tiempo_referencia, tiempo):
            raise ValueError("Los ensayos no comparten el mismo vector temporal.")
    return np.stack(relaciones, axis=0), tiempo_referencia


def graficar_pruebas_empalmadas(ensayos, grupo, es_laplaciano=False):
    """Superpone todos los ensayos y destaca su curva promedio."""
    relaciones, tiempo = calcular_relaciones_individuales(ensayos)

    if VENTANA_SUAVIZADO > 1:
        relaciones = uniform_filter1d(
            relaciones,
            size=VENTANA_SUAVIZADO,
            axis=-1,
            mode="nearest",
        )

    promedio = relaciones.mean(axis=0)

    if LIMITES_RELACION_MU_BETA is not None:
        y_min, y_max = LIMITES_RELACION_MU_BETA
    else:
        valores = relaciones[np.isfinite(relaciones)]

        # Límites robustos basados en percentiles
        y_min = np.percentile(valores, 3)
        y_max = np.percentile(valores, 97)

        # Asegurar que el cero siempre quede visible
        y_min = min(y_min, 0.0)
        y_max = max(y_max, 0.0)

        margen = 0.05 * (y_max - y_min or 1.0)
        y_min -= margen
        y_max += margen

    fig, ejes = crear_malla(
        f"Relación Mu/Beta ensayo a ensayo | {MODALIDAD} | {grupo} | "
        f"{NOMBRE_CANAL_LAPLACIANO + ' | ' if es_laplaciano else ''}"
        f"{len(ensayos)} Ensayos",
        es_laplaciano,
        leyenda=False,
    )

    for i, (canal, ax) in enumerate(ejes.items()):
        marcar_protocolo(ax, sombrear=True)
        ax.axhline(0.0, linewidth=1.0, alpha=0.8)

        color = COLORES_CANALES.get(canal, "C0")

        for curva in relaciones[:, i, :]:
            ax.plot(
                tiempo,
                curva,
                color=color,
                linewidth=0.8,
                alpha=ALPHA_ENSAYOS_EMPALMADOS,
            )

        ax.plot(
            tiempo,
            promedio[i],
            color=color,
            linewidth=GROSOR_PROMEDIO_EMPALMADO,
            alpha=1.0,
            label="Promedio",
        )

        ax.set_ylim(y_min, y_max)
        ax.set(
            xlabel="Tiempo (s)",
            ylabel="Cambio Mu/Beta vs baseline (%)",
        )

        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
        ax.legend(loc="best")

    finalizar_figura(
        fig,
        nombre=(
            f"Relacion_Mu_Beta_Empalmada_"
            f"{'Laplaciano_' if es_laplaciano else ''}{MODALIDAD}_{grupo}"
        ),
        subcarpeta="Pruebas Empalmadas",
        es_laplaciano=es_laplaciano,
    )


def mover_figuras_prueba_aleatoria(grupo, es_laplaciano, destino):
    """Mueve las cuatro figuras de la prueba elegida a una carpeta sin subcarpetas."""
    raiz = CARPETA_LAPLACIANO if es_laplaciano else CARPETA_RESULTADOS
    prefijo_lap = "Laplaciano_" if es_laplaciano else ""
    origenes = [
        raiz / "Escalogramas" / f"ERSP_Morlet_{prefijo_lap}{MODALIDAD}_{grupo}.{FIG_FORMATO}",
        raiz / "Ritmo Mu" / f"ERD_ERS_{prefijo_lap}{MODALIDAD}_Ritmo Mu_{grupo}.{FIG_FORMATO}",
        raiz / "Ritmo Beta" / f"ERD_ERS_{prefijo_lap}{MODALIDAD}_Ritmo Beta_{grupo}.{FIG_FORMATO}",
        raiz / "Relacion Mu Beta" / f"Relacion_Mu_Beta_{prefijo_lap}{MODALIDAD}_{grupo}.{FIG_FORMATO}",
    ]
    destino.mkdir(parents=True, exist_ok=True)
    for origen in origenes:
        if not origen.exists():
            raise FileNotFoundError(f"No se encontro la figura temporal: {origen}")
        nombre_limpio = origen.name.replace(f"_{grupo}.{FIG_FORMATO}", f".{FIG_FORMATO}")
        shutil.move(str(origen), str(destino / nombre_limpio))


def analizar_prueba_aleatoria(ensayo, archivo):
    """Genera cuatro representaciones multicanal y Laplacianas de la misma prueba."""
    identificador = re.sub(r"[^A-Za-z0-9_-]+", "_", archivo.stem)
    grupo = f"PRUEBA_ALEATORIA_{identificador}"
    print(f"\n{SEP_DOBLE}\n PRUEBA ALEATORIA: {archivo.relative_to(CARPETA_DATOS)}\n{SEP_DOBLE}")
    datos_base = ensayo[np.newaxis, ...]
    for es_laplaciano in ((False, True) if USAR_LAPLACIANO else (False,)):
        datos = aplicar_laplaciano_ensayos(datos_base) if es_laplaciano else datos_base
        ersp_db, tiempo_tfr = calcular_ersp(datos)
        graficar_ersp(ersp_db, tiempo_tfr, 1, grupo, es_laplaciano)
        erd_ers, tiempo_fft = calcular_erd_ers(datos)
        for banda in BANDAS:
            graficar_erd_ers(erd_ers, tiempo_fft, banda, 1, grupo, es_laplaciano)
        relacion, tiempo_relacion = calcular_relacion_mu_beta(datos)
        graficar_relacion_mu_beta(relacion, tiempo_relacion, 1, grupo, es_laplaciano)
        destino = (
            CARPETA_PRUEBA_ALEATORIA_LAPLACIANO
            if es_laplaciano
            else CARPETA_PRUEBA_ALEATORIA
        )
        mover_figuras_prueba_aleatoria(grupo, es_laplaciano, destino)
    if not MOSTRAR_FIGURAS:
        plt.close("all")


def analizar_grupo(ensayos, grupo):
    n_ensayos = ensayos.shape[0]

    print(f"\n{SEP_DOBLE}\n ANÁLISIS: {MODALIDAD} | {grupo} | {n_ensayos} ensayos\n{SEP_DOBLE}")

    ensayos = aplicar_ica(ensayos)

    for es_laplaciano in ((False, True) if USAR_LAPLACIANO else (False,)):
        if es_laplaciano:
            print(
                f"\n [LAPLACIANO] Canal: {CANAL_INTERES_LAPLACIANO} | "
                f"Vecinos: {', '.join(CANALES_VECINOS_LAPLACIANO)}"
            )
            datos = aplicar_laplaciano_ensayos(ensayos)
        else:
            print("\n [MULTICANAL]")
            datos = ensayos

        print("   > ERSP (Morlet)")
        ersp_db, tiempo_tfr = calcular_ersp(datos)
        graficar_ersp(ersp_db, tiempo_tfr, n_ensayos, grupo, es_laplaciano)

        print("   > ERD/ERS")
        erd_ers, tiempo_fft = calcular_erd_ers(datos)
        for banda in BANDAS:
            graficar_erd_ers(erd_ers, tiempo_fft, banda, n_ensayos, grupo, es_laplaciano)

        print("   > Relación Mu/Beta respecto al baseline")
        relacion, tiempo_relacion = calcular_relacion_mu_beta(datos)
        graficar_relacion_mu_beta(relacion, tiempo_relacion, n_ensayos, grupo, es_laplaciano)
        print("   > Pruebas empalmadas ensayo a ensayo")
        graficar_pruebas_empalmadas(datos, grupo, es_laplaciano)

    if not MOSTRAR_FIGURAS:
        plt.close("all")


def main():
    validar_configuracion()

    sigma_t = np.broadcast_to(N_CYCLES_MORLET, FREQS_MORLET.shape) / (2 * np.pi * FREQS_MORLET)

    print(f"\n{SEP_DOBLE}\n ERD/ERS | {MODALIDAD}\n{SEP_DOBLE}")
    print(f" Datos       : {CARPETA_DATOS}")
    print(f" Lateralidad : {LATERALIDAD_NORMALIZADA}")
    print(f" Resultados  : {CARPETA_RESULTADOS}")
    if USAR_LAPLACIANO:
        print(f" Laplaciano  : {CARPETA_LAPLACIANO}")
    print(
        f" Protocolo   : Reposo inicial {T_REPOSO_INICIAL} | "
        f"Tarea {T_TAREA} | Reposo final {T_REPOSO_FINAL} s"
    )
    print(f" Baseline    : {BASELINE} s")
    print(
        f" Morlet      : sigma temporal {sigma_t[0]:.3f} s en {FREQS_MORLET[0]:g} Hz "
        f"y {sigma_t[-1]:.3f} s en {FREQS_MORLET[-1]:g} Hz"
    )
    print(f" FFT         : resolución {FS / FFT_NFFT:.2f} Hz")

    print(f"\n{SEP_SIMPLE}\n CARGA DE DATOS\n{SEP_SIMPLE}")

    ensayos_por_dia, archivos_por_dia = cargar_ensayos_por_dia()

    print("\n Ensayos por día:")
    for dia, ensayos_dia in ensayos_por_dia.items():
        print(f"   {dia:<4}: {ensayos_dia.shape[0]} ensayos")
    print(f"   {'GEN':<4}: {sum(e.shape[0] for e in ensayos_por_dia.values())} ensayos")

    for dia, ensayos_dia in ensayos_por_dia.items():
        analizar_grupo(ensayos_dia, dia)

    ensayos_generales = np.concatenate(list(ensayos_por_dia.values()), axis=0)
    analizar_grupo(ensayos_generales, "GENERAL")

    pares_pruebas = [
        (ensayo, archivo)
        for dia in ensayos_por_dia
        for ensayo, archivo in zip(ensayos_por_dia[dia], archivos_por_dia[dia])
    ]
    generador = random.Random(SEMILLA_PRUEBA_ALEATORIA)
    ensayo_aleatorio, archivo_aleatorio = generador.choice(pares_pruebas)
    analizar_prueba_aleatoria(ensayo_aleatorio, archivo_aleatorio)

    if MOSTRAR_FIGURAS:
        plt.show()

    print(f"\n{SEP_DOBLE}\n ANÁLISIS TERMINADO\n{SEP_DOBLE}")
    print(f" Resultados guardados en:\n   {CARPETA_RESULTADOS}")
    if USAR_LAPLACIANO:
        print(f"   {CARPETA_LAPLACIANO}")
    print(f"   {CARPETA_PRUEBA_ALEATORIA}")
    if USAR_LAPLACIANO:
        print(f"   {CARPETA_PRUEBA_ALEATORIA_LAPLACIANO}")


if __name__ == "__main__":
    main()