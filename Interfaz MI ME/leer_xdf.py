# -*- coding: utf-8 -*-
"""
Lee uno o varios .xdf (LabRecorder) del protocolo MI/ME y recorta los trials
usando los marcadores LSL de 'MIME_Markers'.

Secuencia de marcadores de cada trial:
    jitter_start  aparece la cruz (inicio del trial; duración variable)
    fix_start     inicio de los 3 s de fijación
    cue_MI|cue_ME cue + beep
    reposo_start  pantalla negra
    reposo_3s     3 s después de reposo_start (el reposo dura 3 s + 2 s)
Marcadores de bloque: block_start, block_end, block_abort.

Ventana de análisis principal: fix_start -> reposo_3s = 3 + 5 + 3 = 11 s nominales
(fija, por eso se pueden apilar todos los trials en un arreglo 3D).

Tipos de recorte (TIPO_RECORTE):
  'fix_a_reposo3'    : [fix_start, fix_start + 11 s)  largo FIJO  (por defecto)
  'jitter_a_reposo3' : [jitter_start, reposo_3s)      largo variable
  'jitter_a_jitter'  : [jitter_start, siguiente jitter_start | block_end]
                       trial completo (incluye los 2 s finales de reposo); largo variable
  'ventana_cue'      : [cue+TMIN, cue+TMAX)           largo FIJO

TIEMPOS: en todo lo que se guarda (metadatos, CSV, gráficas) el tiempo 0 es el
INICIO DEL RECORTE; no hay tiempos negativos. Ej. con 'fix_a_reposo3':
fix_start = 0 s, cue = 3 s, reposo_start = 8 s, reposo_3s = 11 s. Un marcador que
caiga antes del inicio del recorte (p. ej. jitter_start en 'fix_a_reposo3') se
guarda como NaN.

Salidas:
  - <salida>/<archivo>_trials.npz        (todos los trials del archivo)
  - <salida>/dataset_unido.npz           (todos los archivos juntos)
  - <RUTA_SIGNALS>/<sujeto>/ME/<archivo>_trialXXX_ME.npy   (un .npy por trial,
    <RUTA_SIGNALS>/<sujeto>/MI/<archivo>_trialXXX_MI.npy    shape = (canales, muestras))
    <RUTA_SIGNALS>/<sujeto>/CSV/<archivo>_indice_trials.csv (metadatos)

Todo se configura en la sección CONFIGURACIÓN INICIAL (no hay argumentos de
terminal). Cambia MODO entre 'procesar' (recortar .xdf) y 'visualizar'
(graficar .npy ya guardados) y ejecuta: python cortar_xdf.py

Nota: con los recortes de largo variable el .npz contiene un arreglo de objetos,
por lo que hay que cargarlo con np.load(ruta, allow_pickle=True).
"""
import csv
import glob
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
import pyxdf

# ==============================================================================
# CONFIGURACIÓN INICIAL (Modifica aquí tus rutas y parámetros)
# ==============================================================================
# --- Modo de ejecución -------------------------------------------------------
MODO = 'procesar'        # 'procesar' = recortar .xdf | 'visualizar' = graficar .npy

# --- Entrada (modo 'procesar') -----------------------------------------------
RUTA_ARCHIVO_XDF = r"C:\Users\lipsp\Documents\GitHub\Entrenamiento-BCI\Protocolo ERD_ERS\Signals\SK1\D0_B1.xdf"
RUTAS_XDF = [RUTA_ARCHIVO_XDF]   # lista de archivos .xdf o patrones glob (ej. r"...\SVX\*.xdf")

# --- Salida de los trials .npy -----------------------------------------------
# Se guardan en <RUTA_SIGNALS>\<CARPETA_SUJETO>\{ME, MI, CSV}
RUTA_SIGNALS = r"C:\Users\lipsp\Documents\GitHub\Entrenamiento-BCI\Protocolo ERD_ERS\Signals"
CARPETA_SUJETO = 'SK1'           # SV1, SV2, SK1, ... | None = usar el nombre de la carpeta del .xdf
GUARDAR_NPY = True               # guardar un .npy por trial

# --- Salida de los .npz ------------------------------------------------------
CARPETA_SALIDA = os.path.join(RUTA_SIGNALS, CARPETA_SUJETO)   # carpeta donde se guardan los .npz

# --- Tipo de recorte ---------------------------------------------------------
TIPO_RECORTE = 'fix_a_reposo3'   # 'fix_a_reposo3' | 'jitter_a_reposo3' | 'jitter_a_jitter' | 'ventana_cue'
DURACION_VENTANA = 11.0          # 'fix_a_reposo3': largo fijo en s (3 fijación + 5 tarea + 3 reposo)
TOL_DURACION = 0.5               # 'fix_a_reposo3': se descarta el trial si (reposo_3s - fix_start)
                                 #                  difiere del nominal en más de esto (s)
TMIN = -5.0                      # solo 'ventana_cue': inicio respecto al cue (s); solo sirve para
TMAX = 5.0                       # cortar: en lo guardado el tiempo 0 es el inicio del recorte

# --- Stream y canales --------------------------------------------------------
EEG_STREAM = None                # nombre del stream EEG si hay ambigüedad (None = automático)
EEG_STREAM_INDICE = None         # si hay streams duplicados: None = elegir el que tenga más muestras,
                                 # o un número (0, 1, ...) para forzar cuál usar (ver tabla que imprime)
CANALES = ['Fz', 'FC3', 'FCz', 'FC4', 'Cz', 'C3', 'C4']
INDICES = [0, 1, 2, 3, 4, 5, 7]  # el 6 'NC' se omite
CLASES = {'MI': 0, 'ME': 1}
NOMBRE_MARCADORES = 'MIME_Markers'

# --- Visualización (modo 'visualizar') ---------------------------------------
# Archivos .npy, carpetas (se buscan .npy dentro de todas sus subcarpetas) o
# patrones glob. Los marcadores y fs se leen de CSV\<archivo>_indice_trials.csv;
# sin ese CSV el eje se dibuja en muestras y sin marcadores.
RUTAS_NPY_VISUALIZAR = [
    os.path.join(RUTA_SIGNALS, 'SK1'),                    # todo el sujeto (ME y MI)
    # os.path.join(RUTA_SIGNALS, 'SK1', 'MI'),            # solo los MI
    # os.path.join(RUTA_SIGNALS, 'SK1', 'MI', 'D1_B1_trial000_MI.npy'),   # un trial
]
GUARDAR_FIG = False              # True = guarda PNG junto a cada .npy en vez de abrir ventanas
QUITAR_DC = True                 # restar la media de cada canal (solo para graficar)
# ==============================================================================

TIPOS_RECORTE = ('fix_a_reposo3', 'jitter_a_reposo3', 'jitter_a_jitter', 'ventana_cue')
# (clave inicio, clave fin) de cada recorte basado en marcadores
INICIO_FIN = {
    'fix_a_reposo3':    ('t_fix', 't_reposo3'),
    'jitter_a_reposo3': ('t_jitter', 't_reposo3'),
    'jitter_a_jitter':  ('t_jitter', 't_next_jitter'),
}
NOMBRE_CLAVE = {'t_fix': 'fix_start', 't_jitter': 'jitter_start', 't_reposo3': 'reposo_3s',
                't_next_jitter': 'siguiente jitter_start/block_end'}
# Marcadores que se guardan (relativos al inicio del recorte) y se dibujan:
# (nombre, clave en res, color)
EVENTOS = [
    ('jitter_start', 't_jitter', 'tab:gray'),
    ('fix_start', 't_fix', 'tab:blue'),
    ('cue', 't_cue', 'r'),
    ('reposo_start', 't_reposo', 'tab:green'),
    ('reposo_3s', 't_reposo3', 'tab:orange'),
]
_RE_NPY = re.compile(r'^(?P<base>.+)_trial\d+_[^_]+\.npy$')


def _nombres_canales(canales):
    """
    Nombres de los canales extraídos. CANALES se corresponde por posición con
    INDICES (el índice 6 'NC' se omite), así que el índice 7 del stream es 'C4'.
    Si se piden otros índices distintos en cantidad, se usan nombres genéricos.
    """
    if len(canales) == len(CANALES):
        return list(CANALES)
    return [f"Ch{c}" for c in canales]


# ─── Carga ────────────────────────────────────────────────────────────────────
def _nombre(s):
    return s['info']['name'][0]


def _tipo(s):
    return s['info']['type'][0]


def _campo(s, clave):
    """Lee un campo de s['info'] (pyxdf los guarda como listas de un elemento)."""
    v = s['info'].get(clave)
    return v[0] if isinstance(v, (list, tuple)) and len(v) else v


def _iguales(a, b):
    """True si dos streams tienen exactamente los mismos timestamps y datos."""
    try:
        return (np.array_equal(np.asarray(a['time_stamps']), np.asarray(b['time_stamps']))
                and np.array_equal(np.asarray(a['time_series']), np.asarray(b['time_series'])))
    except Exception:
        return False


def _elegir(candidatos, etiqueta, indice=None):
    """
    Elige un stream entre varios candidatos con el mismo nombre/tipo.
    Imprime una tabla para poder identificar de dónde viene cada uno.
    Criterio: `indice` si se indica; si no, el de más muestras (empate: el primero).
    """
    if len(candidatos) == 1:
        return candidatos[0]

    print(f"  [AVISO] Hay {len(candidatos)} streams candidatos para {etiqueta}:")
    for i, c in enumerate(candidatos):
        print(f"    [{i}] {_nombre(c)} ({_tipo(c)}) | muestras={len(c['time_stamps'])} "
              f"| canales={_campo(c, 'channel_count')} | fs={_campo(c, 'nominal_srate')} "
              f"| host={_campo(c, 'hostname')} | source_id={_campo(c, 'source_id')} "
              f"| uid={_campo(c, 'uid')} | stream_id={_campo(c, 'stream_id')} "
              f"| creado={_campo(c, 'created_at')}")

    if indice is not None:
        if not 0 <= indice < len(candidatos):
            raise ValueError(f"EEG_STREAM_INDICE={indice} fuera de rango (0..{len(candidatos) - 1})")
        elegido = indice
    else:
        n = [len(c['time_stamps']) for c in candidatos]
        elegido = int(np.argmax(n))

    otros = [i for i in range(len(candidatos)) if i != elegido]
    if all(_iguales(candidatos[elegido], candidatos[i]) for i in otros):
        print(f"  -> Son duplicados exactos (mismos datos y timestamps). Se usa el [{elegido}].")
    else:
        print(f"  -> OJO: los streams NO son idénticos. Se usa el [{elegido}]; "
              f"revisa la tabla y fuerza otro con EEG_STREAM_INDICE si no es el correcto.")
    return candidatos[elegido]


def cargar_streams(ruta, eeg_name=None, eeg_indice=None, marker_name=NOMBRE_MARCADORES):
    """Devuelve (stream_eeg, stream_marcadores) de un .xdf."""
    streams, _ = pyxdf.load_xdf(ruta)
    disponibles = [f"{_nombre(s)} ({_tipo(s)})" for s in streams]

    marc = [s for s in streams if _nombre(s) == marker_name]
    if len(marc) == 0:
        raise ValueError(f"No hay ningún stream '{marker_name}'. "
                         f"Streams en el archivo: {disponibles}")

    # Forzar la búsqueda por el nombre personalizado o el tipo 'Data' si falla el tipo 'EEG'
    if eeg_name:
        eeg = [s for s in streams if _nombre(s) == eeg_name]
    else:
        # Intenta primero con 'EEG' y si no, busca por 'DATA' (caso de streamTEST)
        eeg = [s for s in streams if _tipo(s).upper() == 'EEG']
        if len(eeg) == 0:
            eeg = [s for s in streams if _tipo(s).upper() == 'DATA' or _nombre(s) == 'streamTEST']

    if len(eeg) == 0:
        raise ValueError(f"No hay ningún stream EEG/Data. "
                         f"Streams en el archivo: {disponibles}")

    return _elegir(eeg, 'EEG', eeg_indice), _elegir(marc, f"marcadores '{marker_name}'")


def leer_marcadores(s_marc):
    """Lista ordenada de (timestamp, etiqueta)."""
    t = np.asarray(s_marc['time_stamps'], dtype=float)
    etiquetas = []
    for m in s_marc['time_series']:
        val = m[0] if isinstance(m, (list, tuple, np.ndarray)) else m
        etiquetas.append(str(val).strip())
    orden = np.argsort(t, kind='stable')
    return [(t[i], etiquetas[i]) for i in orden]


# ─── Segmentación por marcadores ──────────────────────────────────────────────
def segmentar(marcadores):
    """
    Agrupa marcadores en trials completos:
        jitter_start -> fix_start -> cue_X -> reposo_start -> reposo_3s
    Cada trial guarda t_jitter, t_fix, t_cue, clase, t_reposo, t_reposo3 y, si
    existe, t_next_jitter: el siguiente jitter_start (o block_end en el último
    trial del bloque). jitter_start es opcional para los recortes que no lo usan.
    """
    trials, cur = [], None
    ultimo = None   # último trial completo que aún espera su t_next_jitter
    info = {'abort': False, 'block_start': False, 'block_end': False, 'incompletos': 0}

    etiquetas_unicas = set([etq for _, etq in marcadores])

    for t, etq in marcadores:
        if etq == 'block_start':
            info['block_start'] = True
            if cur is not None:
                info['incompletos'] += 1
            cur, ultimo = None, None
        elif etq == 'jitter_start':
            if ultimo is not None:
                ultimo['t_next_jitter'] = t
                ultimo = None
            if cur is not None:
                info['incompletos'] += 1
            cur = {'t_jitter': t}
        elif etq == 'fix_start':
            if cur is not None and 't_fix' in cur:      # fix_start repetido
                info['incompletos'] += 1
                cur = {}
            elif cur is None:                           # trial sin jitter_start
                cur = {}
            cur['t_fix'] = t
        elif etq.startswith('cue_'):
            clase = etq[4:]
            if cur is not None and 't_fix' in cur and 't_cue' not in cur and clase in CLASES:
                cur['t_cue'], cur['clase'] = t, clase
            else:
                info['incompletos'] += 1
                cur = None
        elif etq == 'reposo_start':
            if cur is not None and 't_cue' in cur and 't_reposo' not in cur:
                cur['t_reposo'] = t
            else:
                if cur is not None:
                    info['incompletos'] += 1
                cur = None
        elif etq == 'reposo_3s':
            if cur is not None and 't_reposo' in cur:
                cur['t_reposo3'] = t
                trials.append(cur)
                ultimo = cur
            elif cur is not None:
                info['incompletos'] += 1
            cur = None
        elif etq == 'block_end':
            info['block_end'] = True
            if cur is not None:
                info['incompletos'] += 1
            if ultimo is not None:
                ultimo['t_next_jitter'] = t
                ultimo = None
            cur = None
        elif etq == 'block_abort':
            info['abort'] = True
            if cur is not None:
                info['incompletos'] += 1
            cur, ultimo = None, None

    if cur is not None:
        info['incompletos'] += 1

    if len(trials) == 0:
        print(f"  [ALERTA] No se segmentaron trials. Marcadores detectados: {etiquetas_unicas}")

    return trials, info


# ─── Épocas ───────────────────────────────────────────────────────────────────
def extraer_epocas(eeg_t, eeg_x, trials, fs, tipo='fix_a_reposo3',
                   dur_nominal=11.0, tol=0.5, tmin=-5.0, tmax=5.0):
    """
    Recorta cada trial por conteo de muestras.

    'ventana_cue'    : [t_cue+tmin, t_cue+tmax)
    'fix_a_reposo3'  : [t_fix, t_fix + dur_nominal); se verifica que
                       t_reposo3 - t_fix esté a menos de `tol` s del nominal.
    'jitter_a_reposo3' / 'jitter_a_jitter': de t_inicio a t_fin (largo variable).
    Los dos primeros dan X 3D (trials, canales, muestras); los demás, un arreglo
    1D de objetos donde cada elemento es (canales, muestras).

    Devuelve X, kept, descartes, ep_t0 (timestamp LSL donde empieza cada época).
    """
    X, kept, descartes, ep_t0 = [], [], [], []
    if tipo == 'ventana_cue':
        n_fijo = int(round((tmax - tmin) * fs))
    elif tipo == 'fix_a_reposo3':
        n_fijo = int(round(dur_nominal * fs))
    else:
        n_fijo = None

    for k, tr in enumerate(trials):
        if tipo == 'ventana_cue':
            t0, n = tr['t_cue'] + tmin, n_fijo
        else:
            k0, k1 = INICIO_FIN[tipo]
            t0, t1 = tr.get(k0), tr.get(k1)
            if t0 is None:
                descartes.append((k, f'falta {NOMBRE_CLAVE[k0]}'))
                continue
            if t1 is None:
                descartes.append((k, f'falta {NOMBRE_CLAVE[k1]}'))
                continue
            if tipo == 'fix_a_reposo3':
                real = t1 - t0
                if abs(real - dur_nominal) > tol:
                    descartes.append((k, f'fix_start→reposo_3s dura {real:.2f} s '
                                         f'(nominal {dur_nominal:g} s, tolerancia {tol:g} s)'))
                    continue
                n = n_fijo
            else:
                n = int(round((t1 - t0) * fs))

        if n < 2:
            descartes.append((k, 'duración nula o negativa'))
            continue
        if t0 < eeg_t[0]:
            descartes.append((k, 'empieza antes del EEG'))
            continue
        i0 = int(np.searchsorted(eeg_t, t0))
        if i0 + n > len(eeg_t):
            descartes.append((k, 'termina después del EEG'))
            continue

        # --- Chequeo de huecos con tolerancia de 0.1 s para admitir jitter de hardware ---
        dur = eeg_t[i0 + n - 1] - eeg_t[i0]
        if abs(dur - (n - 1) / fs) > 0.1:
            descartes.append((k, f'hueco excesivo en el stream EEG (dur {dur:.3f} s)'))
            continue

        X.append(eeg_x[i0:i0 + n].T)
        kept.append(k)
        ep_t0.append(t0)

    if n_fijo is not None:
        X = np.stack(X) if X else np.empty((0, eeg_x.shape[1], n_fijo))
    else:
        arr = np.empty(len(X), dtype=object)
        for i, x in enumerate(X):
            arr[i] = x
        X = arr
    return X, kept, descartes, np.array(ep_t0, dtype=float)


# ─── Guardar cada trial como .npy ─────────────────────────────────────────────
def guardar_trials_npy(res, nombre_base, carpeta_sujeto):
    """
    Guarda cada trial de `res` como un .npy individual de shape (canales, muestras).

    Estructura:
        <carpeta_sujeto>/ME/<nombre_base>_trial000_ME.npy
        <carpeta_sujeto>/MI/<nombre_base>_trial001_MI.npy
        <carpeta_sujeto>/CSV/<nombre_base>_indice_trials.csv

    El número de trial es el del bloque (por eso en cada carpeta de clase puede
    haber saltos). Antes de guardar se borran los .npy previos de ese mismo
    archivo para que no queden trials viejos mezclados con los nuevos.

    El CSV guarda fs, la duración y la posición de cada marcador en segundos
    desde el inicio del recorte (tiempo 0); NaN si el marcador cae antes del inicio.
    Devuelve la carpeta del sujeto.
    """
    carpeta_csv = os.path.join(carpeta_sujeto, 'CSV')
    os.makedirs(carpeta_csv, exist_ok=True)
    for clase in CLASES:
        os.makedirs(os.path.join(carpeta_sujeto, clase), exist_ok=True)

    previos = 0
    for clase in CLASES:
        for f in glob.glob(os.path.join(carpeta_sujeto, clase, f"{glob.escape(nombre_base)}_trial*.npy")):
            os.remove(f)
            previos += 1

    ruta_csv = os.path.join(carpeta_csv, f"{nombre_base}_indice_trials.csv")
    with open(ruta_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['archivo_npy', 'trial', 'clase', 'y', 'epoca_t0_lsl']
                   + [f'{nombre}_s' for nombre, _, _ in EVENTOS]
                   + ['t_next_jitter_s', 'duracion_s', 'fs', 'canales'])
        for i in range(len(res['X'])):
            clase = str(res['clase'][i])
            nombre_npy = f"{nombre_base}_trial{i:03d}_{clase}.npy"
            carpeta_clase = os.path.join(carpeta_sujeto, clase)
            os.makedirs(carpeta_clase, exist_ok=True)
            np.save(os.path.join(carpeta_clase, nombre_npy), res['X'][i])
            w.writerow([nombre_npy, i, clase, int(res['y'][i]), f"{res['epoca_t0_lsl'][i]:.6f}"]
                       + [f"{res[clave][i]:.4f}" for _, clave, _ in EVENTOS]
                       + [f"{res['t_next_jitter'][i]:.4f}", f"{res['duracion'][i]:.4f}",
                          res['fs'], '|'.join(res['ch_names'])])

    extra = f" (se reemplazaron {previos} .npy previos de este archivo)" if previos else ""
    print(f"  -> {len(res['X'])} trials .npy guardados en: {carpeta_sujeto}{extra}")
    print(f"  -> CSV: {ruta_csv}")
    return carpeta_sujeto


# ─── Visualizar .npy ──────────────────────────────────────────────────────────
def _meta_desde_csv(ruta_npy):
    """Busca fs y marcadores del .npy en <sujeto>/CSV/<archivo>_indice_trials.csv."""
    nombre = os.path.basename(ruta_npy)
    m = _RE_NPY.match(nombre)
    if not m:
        return None
    carpeta_sujeto = os.path.dirname(os.path.dirname(os.path.abspath(ruta_npy)))
    ruta_csv = os.path.join(carpeta_sujeto, 'CSV', f"{m.group('base')}_indice_trials.csv")
    if not os.path.isfile(ruta_csv):
        return None
    try:
        with open(ruta_csv, newline='', encoding='utf-8') as f:
            for fila in csv.DictReader(f):
                if fila['archivo_npy'] == nombre:
                    return {'fs': float(fila['fs']),
                            'eventos': {n: float(fila[f'{n}_s']) for n, _, _ in EVENTOS}}
    except (KeyError, ValueError):
        return None   # CSV de una versión anterior
    return None


def visualizar_npy(ruta_npy, ch_names=None, quitar_dc=True, guardar_fig=False):
    """
    Grafica un trial .npy (canales, muestras): un subplot por canal, con el eje
    de tiempo desde el inicio del recorte (t = 0) y los marcadores jitter_start,
    fix_start, cue, reposo_start y reposo_3s que caigan dentro de la época.

    fs y marcadores se leen del CSV del sujeto. Sin CSV, el eje va en muestras
    y no se dibujan marcadores.
    quitar_dc: resta la media de cada canal (sólo para visualizar).
    guardar_fig: guarda un .png junto al .npy en lugar de abrir la ventana.
    """
    import matplotlib
    if guardar_fig:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = np.load(ruta_npy)
    if x.ndim != 2:
        raise ValueError(f"{ruta_npy}: se esperaba un arreglo 2D (canales, muestras), "
                         f"shape = {x.shape}")
    n_ch, n = x.shape

    meta = _meta_desde_csv(ruta_npy)
    if meta:
        fs, eventos = meta['fs'], meta['eventos']
        t = np.arange(n) / fs
        etiqueta_x, margen = 'Tiempo desde el inicio del recorte (s)', 1.5 / fs
    else:
        fs, eventos = None, {}
        t = np.arange(n, dtype=float)
        etiqueta_x, margen = 'Muestras desde el inicio del recorte (sin CSV)', 0.0

    if ch_names is None or len(ch_names) != n_ch:
        ch_names = [f"Ch{i}" for i in range(n_ch)]

    datos = x - x.mean(axis=1, keepdims=True) if quitar_dc else x

    fig, axes = plt.subplots(n_ch, 1, sharex=True, figsize=(12, max(3, 1.3 * n_ch)))
    axes = np.atleast_1d(axes)
    lineas = [(eventos[nom], nom, col) for nom, _, col in EVENTOS
              if nom in eventos and np.isfinite(eventos[nom])
              and -margen <= eventos[nom] <= t[-1] + margen]
    for j, (ax, y, nombre) in enumerate(zip(axes, datos, ch_names)):
        ax.plot(t, y, lw=0.7)
        for pos, nom, col in lineas:
            ax.axvline(pos, color=col, ls='--', lw=1, label=nom if j == 0 else None)
        ax.set_ylabel(nombre, rotation=0, ha='right', va='center')
        ax.set_xlim(0, t[-1])
        ax.grid(alpha=0.3)
    if lineas:
        axes[0].legend(loc='upper right', ncol=len(lineas), fontsize=7)
    axes[-1].set_xlabel(etiqueta_x)

    stem = os.path.splitext(os.path.basename(ruta_npy))[0]
    info_fs = f"{n / fs:.1f} s  |  fs ≈ {fs:.1f} Hz" if fs else f"{n} muestras"
    fig.suptitle(f"{stem}  |  {info_fs}" + ("  |  DC removido" if quitar_dc else ""))
    fig.tight_layout()

    if guardar_fig:
        ruta_png = os.path.splitext(ruta_npy)[0] + '.png'
        fig.savefig(ruta_png, dpi=120)
        plt.close(fig)
        print(f"  -> Figura guardada: {ruta_png}")
    return fig


def visualizar_varios(entradas, ch_names, quitar_dc, guardar_fig):
    """Acepta archivos .npy, carpetas (búsqueda recursiva) o patrones glob y los grafica uno a uno."""
    rutas = []
    for e in entradas:
        if os.path.isdir(e):
            rutas.extend(sorted(glob.glob(os.path.join(glob.escape(e), '**', '*.npy'), recursive=True)))
        else:
            rutas.extend(sorted(glob.glob(e, recursive=True)))
    if not rutas:
        print(f"Error: no se encontraron .npy en: {entradas}")
        return

    print(f"Graficando {len(rutas)} trial(s)...")
    for r in rutas:
        visualizar_npy(r, ch_names, quitar_dc, guardar_fig)

    if not guardar_fig:
        import matplotlib.pyplot as plt
        plt.show()


# ─── Procesar un archivo ──────────────────────────────────────────────────────
def procesar(ruta, a):
    print(f"\n=== {os.path.basename(ruta)} ===")
    s_eeg, s_marc = cargar_streams(ruta, a.eeg_stream, a.eeg_indice)

    eeg_t = np.asarray(s_eeg['time_stamps'], dtype=float)
    eeg_x_all = np.asarray(s_eeg['time_series'], dtype=float)
    fs = float(s_eeg['info']['nominal_srate'][0])
    n_ch = eeg_x_all.shape[1]

    if max(a.canales) >= n_ch:
        raise ValueError(f"El stream EEG tiene {n_ch} canales y pediste el índice {max(a.canales)}")
    eeg_x = eeg_x_all[:, a.canales]

    dur = eeg_t[-1] - eeg_t[0]
    fs_ef = (len(eeg_t) - 1) / dur if dur > 0 else float('nan')
    print(f"  EEG: {len(eeg_t)} muestras, {n_ch} canales en el stream, "
          f"{dur:.1f} s | fs nominal {fs:g} Hz, efectiva {fs_ef:.2f} Hz")

    marcadores = leer_marcadores(s_marc)
    trials, info = segmentar(marcadores)
    print(f"  Marcadores: {len(marcadores)} | trials completos: {len(trials)} | incompletos: {info['incompletos']}")

    if len(trials) == 0:
        return None

    d_real = np.array([t['t_reposo3'] - t['t_fix'] for t in trials])
    print(f"  fix_start→reposo_3s (real): min {d_real.min():.2f} s | media {d_real.mean():.2f} s "
          f"| max {d_real.max():.2f} s (nominal {a.dur_ventana:g} s)")

    X, kept, descartes, ep_t0 = extraer_epocas(
        eeg_t, eeg_x, trials, fs, a.tipo_recorte,
        dur_nominal=a.dur_ventana, tol=a.tol_duracion, tmin=a.tmin, tmax=a.tmax)
    print(f"  Épocas extraídas ({a.tipo_recorte}): {len(X)} (Descartadas: {len(descartes)})")
    for k, mot in descartes:
        print(f"    - Trial {k} descartado: {mot}")

    if len(X) == 0:
        return None

    n_muestras = np.array([x.shape[-1] for x in X])
    durs = n_muestras / fs
    print(f"  Duración de las épocas: min {durs.min():.2f} s | media {durs.mean():.2f} s | max {durs.max():.2f} s")

    trials_kept = [trials[i] for i in kept]
    es_ventana = a.tipo_recorte == 'ventana_cue'

    def rel(clave):
        """Instante del marcador en s desde el inicio del recorte (NaN si no existe
        o si cae antes del inicio, para que no haya tiempos negativos)."""
        v = np.array([t.get(clave, np.nan) for t in trials_kept], dtype=float) - ep_t0
        with np.errstate(invalid='ignore'):
            v[v < -1e-9] = np.nan
        return v

    res = {
        'X': X,
        'y': np.array([CLASES[t['clase']] for t in trials_kept], dtype=int),
        'clase': np.array([t['clase'] for t in trials_kept], dtype=object),
        'epoca_t0_lsl': ep_t0,                   # timestamp LSL del inicio de cada época
        # Marcadores en s desde el inicio del recorte (tiempo 0):
        't_jitter': rel('t_jitter'),
        't_fix': rel('t_fix'),
        't_cue': rel('t_cue'),
        't_reposo': rel('t_reposo'),
        't_reposo3': rel('t_reposo3'),
        't_next_jitter': rel('t_next_jitter'),
        'dur_fix_reposo3': np.array([t['t_reposo3'] - t['t_fix'] for t in trials_kept], dtype=float),
        'duracion': durs,                        # duración de cada época (s)
        'fs': fs,
        'tmin': a.tmin if es_ventana else np.nan,
        'tmax': a.tmax if es_ventana else np.nan,
        'modo_recorte': a.tipo_recorte,
        'ch_names': np.array(_nombres_canales(a.canales), dtype=object)
    }
    return res


# ─── Bloque Principal de Ejecución ────────────────────────────────────────────
if __name__ == '__main__':
    if TIPO_RECORTE not in TIPOS_RECORTE:
        print(f"Error: TIPO_RECORTE debe ser uno de {TIPOS_RECORTE} (actual: {TIPO_RECORTE!r})")
        sys.exit(1)

    args = SimpleNamespace(
        archivos=RUTAS_XDF, salida=CARPETA_SALIDA, tmin=TMIN, tmax=TMAX,
        canales=INDICES, eeg_stream=EEG_STREAM, eeg_indice=EEG_STREAM_INDICE,
        tipo_recorte=TIPO_RECORTE, dur_ventana=DURACION_VENTANA, tol_duracion=TOL_DURACION,
    )

    # Modo visualización: no toca los .xdf
    if MODO == 'visualizar':
        visualizar_varios(RUTAS_NPY_VISUALIZAR, _nombres_canales(args.canales),
                          quitar_dc=QUITAR_DC, guardar_fig=GUARDAR_FIG)
        sys.exit(0)
    elif MODO != 'procesar':
        print(f"Error: MODO debe ser 'procesar' o 'visualizar' (actual: {MODO!r})")
        sys.exit(1)

    lista_archivos = []
    for f in args.archivos:
        lista_archivos.extend(glob.glob(f))

    if not lista_archivos:
        print(f"Error: No se encontró el archivo en la ruta especificada: {args.archivos[0]}")
        sys.exit(1)

    os.makedirs(args.salida, exist_ok=True)
    datasets = []

    for idx, ruta in enumerate(lista_archivos):
        resultado = procesar(ruta, args)
        if resultado is not None:
            nombre_base = os.path.splitext(os.path.basename(ruta))[0]
            ruta_out = os.path.join(args.salida, f"{nombre_base}_trials.npz")
            np.savez_compressed(ruta_out, **resultado)
            print(f"  -> Guardado individual: {ruta_out}")

            if GUARDAR_NPY:
                sujeto = CARPETA_SUJETO or os.path.basename(os.path.dirname(os.path.abspath(ruta)))
                guardar_trials_npy(resultado, nombre_base, os.path.join(RUTA_SIGNALS, sujeto))

            resultado['bloque'] = np.full(len(resultado['X']), idx, dtype=int)
            datasets.append(resultado)

    if len(datasets) > 0:
        unido = {}
        for llave in datasets[0].keys():
            if llave in ['fs', 'tmin', 'tmax', 'ch_names', 'modo_recorte']:
                unido[llave] = datasets[0][llave]
            else:
                unido[llave] = np.concatenate([d[llave] for d in datasets], axis=0)

        ruta_unido = os.path.join(args.salida, 'dataset_unido.npz')
        np.savez_compressed(ruta_unido, **unido)
        print(f"\n=== PROCESO TERMINADO ===")
        print(f"Dataset combinado guardado en: {ruta_unido} (Total trials: {len(unido['X'])})")
    else:
        print("\nNo se pudo generar ningún dataset. Revisa las alertas de marcadores impresas arriba.")