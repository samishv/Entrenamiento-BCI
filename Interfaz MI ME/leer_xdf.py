# -*- coding: utf-8 -*-
"""
Lee uno o varios .xdf (LabRecorder) del protocolo MI/ME y recorta los trials
usando los marcadores LSL de 'MIME_Markers'.

Secuencia de marcadores esperada:
    block_start -> [ rest_start -> cue_MI | cue_ME -> iti_start ] x N -> block_end
    (block_abort si el bloque se interrumpe)

Tipos de recorte (TIPO_RECORTE):
  'rest_a_rest' : de rest_start hasta el siguiente rest_start (o block_end en el
                  último trial). Largo variable por trial.
  'ventana_cue' : ventana fija [cue+TMIN, cue+TMAX). Largo igual para todos.

Salidas:
  - <salida>/<archivo>_trials.npz        (todos los trials del archivo)
  - <salida>/dataset_unido.npz           (todos los archivos juntos)
  - <salida>/npy/<archivo>/<archivo>_trialXXX_<clase>.npy  (un .npy por trial,
    shape = (canales, muestras)) + indice_trials.csv con los metadatos

Todo se configura en la sección CONFIGURACIÓN INICIAL (no hay argumentos de
terminal). Cambia MODO entre 'procesar' (recortar .xdf) y 'visualizar'
(graficar .npy ya guardados) y ejecuta: python cortar_xdf.py

Nota: con 'rest_a_rest' el .npz contiene un arreglo de objetos (trials de
distinto largo), por lo que hay que cargarlo con np.load(ruta, allow_pickle=True).
"""
import csv
import glob
import os
import sys
from types import SimpleNamespace

import numpy as np
import pyxdf

# ==============================================================================
# CONFIGURACIÓN INICIAL (Modifica aquí tus rutas y parámetros)
# ==============================================================================
# --- Modo de ejecución -------------------------------------------------------
MODO = 'visualizar'        # 'procesar' = recortar .xdf | 'visualizar' = graficar .npy

# --- Entrada / salida (modo 'procesar') --------------------------------------
RUTA_ARCHIVO_XDF = r"C:\Users\lipsp\Documents\GitHub\Entrenamiento-BCI\Protocolo ERD_ERS\Signals\SVY\D1_B1.xdf"
RUTAS_XDF = [RUTA_ARCHIVO_XDF]   # lista de archivos .xdf o patrones glob (ej. r"...\SVX\*.xdf")
CARPETA_SALIDA = './Trials'
GUARDAR_NPY = True               # guardar un .npy por trial

# --- Tipo de recorte ---------------------------------------------------------
TIPO_RECORTE = 'rest_a_rest'     # 'rest_a_rest' | 'ventana_cue'
FIN_TRIAL = 'siguiente_rest'     # solo 'rest_a_rest': 'siguiente_rest' (incluye el ITI)
                                 #                  o 'iti_start' (termina al iniciar el ITI)
TMIN = -5.0                      # solo 'ventana_cue': inicio respecto al cue (s)
TMAX = 5.0                       # solo 'ventana_cue': fin respecto al cue (s)

# --- Stream y canales --------------------------------------------------------
EEG_STREAM = None                # nombre del stream EEG si hay ambigüedad (None = automático)
EEG_STREAM_INDICE = None          # si hay streams duplicados: None = elegir el que tenga más muestras,
                                 # o un número (0, 1, ...) para forzar cuál usar (ver tabla que imprime)
CANALES = ['Fz', 'FC3', 'FCz', 'FC4', 'Cz', 'C3', 'C4']
INDICES = [0, 1, 2, 3, 4, 5, 7]  # el 6 'NC' se omite
CLASES = {'MI': 0, 'ME': 1}
NOMBRE_MARCADORES = 'MIME_Markers'

# --- Visualización (modo 'visualizar') ---------------------------------------
# Archivos .npy, carpetas o patrones glob. Si junto a los .npy está el
# indice_trials.csv, se lee de ahí fs y el inicio de cada época (TMIN/TMAX solo
# se usan como respaldo cuando no hay CSV).
RUTAS_NPY_VISUALIZAR = [
    r"./Trials/npy/D1_B1",                       # carpeta completa
    # r"./Trials/npy/D1_B1/D1_B1_trial000_MI.npy",   # un trial
    # r"./Trials/npy/D1_B1/*_MI.npy",                # solo los MI
]
GUARDAR_FIG = False              # True = guarda PNG junto a cada .npy en vez de abrir ventanas
QUITAR_DC = True                 # restar la media de cada canal (solo para graficar)
# ==============================================================================


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
    Agrupa marcadores en trials completos: rest_start -> cue_X -> iti_start.
    Cada trial guarda t_rest, t_cue, clase, t_iti y, si existe, t_next_rest: el
    instante del siguiente rest_start (o del block_end en el último trial).
    """
    trials, cur = [], None
    ultimo = None   # último trial completo que aún espera su t_next_rest
    info = {'abort': False, 'block_start': False, 'block_end': False, 'incompletos': 0}

    etiquetas_unicas = set([etq for _, etq in marcadores])

    for t, etq in marcadores:
        if etq == 'block_start':
            info['block_start'] = True
            ultimo = None
        elif etq == 'rest_start':
            if ultimo is not None:
                ultimo['t_next_rest'] = t
                ultimo = None
            if cur is not None:
                info['incompletos'] += 1
            cur = {'t_rest': t}
        elif etq.startswith('cue_'):
            clase = etq[4:]
            if cur is not None and 't_cue' not in cur and clase in CLASES:
                cur['t_cue'], cur['clase'] = t, clase
            else:
                info['incompletos'] += 1
                cur = None
        elif etq == 'iti_start':
            if cur is not None and 't_cue' in cur:
                cur['t_iti'] = t
                trials.append(cur)
                ultimo = cur
            elif cur is not None:
                info['incompletos'] += 1
            cur = None
        elif etq == 'block_end':
            info['block_end'] = True
            if ultimo is not None:
                ultimo['t_next_rest'] = t
                ultimo = None
            cur = None
        elif etq == 'block_abort':
            info['abort'] = True
            if cur is not None:
                info['incompletos'] += 1
            cur = None
            ultimo = None

    if cur is not None:
        info['incompletos'] += 1

    if len(trials) == 0:
        print(f"  [ALERTA] No se segmentaron trials. Marcadores detectados: {etiquetas_unicas}")

    return trials, info


# ─── Épocas ───────────────────────────────────────────────────────────────────
def extraer_epocas(eeg_t, eeg_x, trials, fs, tmin, tmax,
                   tipo='rest_a_rest', fin_trial='siguiente_rest'):
    """
    Recorta cada trial por conteo de muestras.

    tipo='ventana_cue': [t_cue+tmin, t_cue+tmax)  -> X es un array 3D (trials, canales, muestras)
    tipo='rest_a_rest': [t_rest, t_fin)           -> X es un array 1D de objetos; cada
        elemento es (canales, muestras) y puede tener distinto largo.
        t_fin = t_next_rest (fin_trial='siguiente_rest') o t_iti (fin_trial='iti_start').

    Devuelve X, kept, descartes, ep_tmin (inicio de cada época respecto al cue, en s).
    """
    X, kept, descartes, ep_tmin = [], [], [], []
    for k, tr in enumerate(trials):
        if tipo == 'ventana_cue':
            t0 = tr['t_cue'] + tmin
            n = int(round((tmax - tmin) * fs))
            ini = tmin
        else:
            t0 = tr['t_rest']
            t_fin = tr.get('t_next_rest') if fin_trial == 'siguiente_rest' else tr['t_iti']
            if t_fin is None:
                descartes.append((k, 'no hay rest_start/block_end posterior que cierre el trial'))
                continue
            n = int(round((t_fin - t0) * fs))
            ini = t0 - tr['t_cue']

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
        ep_tmin.append(ini)

    if tipo == 'ventana_cue':
        n = int(round((tmax - tmin) * fs))
        X = np.stack(X) if X else np.empty((0, eeg_x.shape[1], n))
    else:
        arr = np.empty(len(X), dtype=object)
        for i, x in enumerate(X):
            arr[i] = x
        X = arr
    return X, kept, descartes, np.array(ep_tmin, dtype=float)


# ─── Guardar cada trial como .npy ─────────────────────────────────────────────
def guardar_trials_npy(res, nombre_base, carpeta_salida):
    """
    Guarda cada trial de `res` como un .npy individual de shape (canales, muestras).

    Estructura:
        <carpeta_salida>/npy/<nombre_base>/<nombre_base>_trial000_MI.npy
        <carpeta_salida>/npy/<nombre_base>/indice_trials.csv   (metadatos)

    Como un .npy no guarda metadatos, se escribe un CSV índice con fs y con el
    inicio/fin de cada época respecto al cue (lo usa la visualización).
    Devuelve la carpeta creada.
    """
    carpeta = os.path.join(carpeta_salida, 'npy', nombre_base)
    os.makedirs(carpeta, exist_ok=True)

    ruta_csv = os.path.join(carpeta, 'indice_trials.csv')
    with open(ruta_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['archivo_npy', 'trial', 'clase', 'y', 't_cue_lsl',
                    't_rest_rel_cue', 't_iti_rel_cue', 'epoca_tmin', 'epoca_tmax',
                    'fs', 'canales'])
        for i in range(len(res['X'])):
            clase = str(res['clase'][i])
            nombre_npy = f"{nombre_base}_trial{i:03d}_{clase}.npy"
            np.save(os.path.join(carpeta, nombre_npy), res['X'][i])
            w.writerow([nombre_npy, i, clase, int(res['y'][i]),
                        f"{res['t_cue'][i]:.6f}",
                        f"{res['t_rest'][i]:.4f}", f"{res['t_iti'][i]:.4f}",
                        f"{res['epoca_tmin'][i]:.4f}", f"{res['epoca_tmax'][i]:.4f}",
                        res['fs'], '|'.join(res['ch_names'])])

    print(f"  -> {len(res['X'])} trials .npy guardados en: {carpeta}")
    return carpeta


# ─── Visualizar .npy ──────────────────────────────────────────────────────────
def _meta_desde_csv(ruta_npy):
    """Busca fs, inicio de época y t_iti del .npy en el indice_trials.csv vecino."""
    ruta_csv = os.path.join(os.path.dirname(os.path.abspath(ruta_npy)), 'indice_trials.csv')
    if not os.path.isfile(ruta_csv):
        return None
    nombre = os.path.basename(ruta_npy)
    try:
        with open(ruta_csv, newline='', encoding='utf-8') as f:
            for fila in csv.DictReader(f):
                if fila['archivo_npy'] == nombre:
                    return {'fs': float(fila['fs']),
                            'tmin': float(fila['epoca_tmin']),
                            't_iti': float(fila['t_iti_rel_cue'])}
    except (KeyError, ValueError):
        return None   # CSV de una versión anterior: se usa el respaldo TMIN/TMAX
    return None


def visualizar_npy(ruta_npy, tmin=-5.0, tmax=5.0, ch_names=None,
                   quitar_dc=True, guardar_fig=False):
    """
    Grafica un trial .npy (canales, muestras): un subplot por canal, con el eje
    de tiempo relativo al cue (t = 0, línea roja) y, si se conoce, el inicio del
    ITI (línea verde).

    fs y el inicio de la época se leen de indice_trials.csv (misma carpeta). Si
    no existe, se deduce fs = n_muestras / (tmax - tmin) y el inicio = tmin.
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
        fs, t_ini, t_iti = meta['fs'], meta['tmin'], meta['t_iti']
    else:
        fs, t_ini, t_iti = n / (tmax - tmin), tmin, None
    t = t_ini + np.arange(n) / fs

    if ch_names is None or len(ch_names) != n_ch:
        ch_names = [f"Ch{i}" for i in range(n_ch)]

    datos = x - x.mean(axis=1, keepdims=True) if quitar_dc else x

    fig, axes = plt.subplots(n_ch, 1, sharex=True, figsize=(12, max(3, 1.3 * n_ch)))
    axes = np.atleast_1d(axes)
    for ax, y, nombre in zip(axes, datos, ch_names):
        ax.plot(t, y, lw=0.7)
        ax.axvline(0, color='r', ls='--', lw=1)
        if t_iti is not None and t[0] <= t_iti <= t[-1]:
            ax.axvline(t_iti, color='g', ls='--', lw=1)
        ax.set_ylabel(nombre, rotation=0, ha='right', va='center')
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel('Tiempo respecto al cue (s)   [rojo = cue, verde = inicio ITI]')

    stem = os.path.splitext(os.path.basename(ruta_npy))[0]
    fuente = 'CSV' if meta else 'TMIN/TMAX'
    fig.suptitle(f"{stem}  |  {n / fs:.1f} s  |  fs ≈ {fs:.1f} Hz ({fuente})"
                 + ("  |  DC removido" if quitar_dc else ""))
    fig.tight_layout()

    if guardar_fig:
        ruta_png = os.path.splitext(ruta_npy)[0] + '.png'
        fig.savefig(ruta_png, dpi=120)
        plt.close(fig)
        print(f"  -> Figura guardada: {ruta_png}")
    return fig


def visualizar_varios(entradas, tmin, tmax, ch_names, quitar_dc, guardar_fig):
    """Acepta archivos .npy, carpetas o patrones glob y los grafica uno a uno."""
    rutas = []
    for e in entradas:
        if os.path.isdir(e):
            rutas.extend(sorted(glob.glob(os.path.join(e, '*.npy'))))
        else:
            rutas.extend(sorted(glob.glob(e)))
    if not rutas:
        print(f"Error: no se encontraron .npy en: {entradas}")
        return

    print(f"Graficando {len(rutas)} trial(s)...")
    for r in rutas:
        visualizar_npy(r, tmin, tmax, ch_names, quitar_dc, guardar_fig)

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

    X, kept, descartes, ep_tmin = extraer_epocas(
        eeg_t, eeg_x, trials, fs, a.tmin, a.tmax, a.tipo_recorte, a.fin_trial)
    print(f"  Épocas extraídas: {len(X)} (Descartadas: {len(descartes)})")
    for k, mot in descartes:
        print(f"    - Trial {k} descartado: {mot}")

    if len(X) == 0:
        return None

    n_muestras = np.array([x.shape[-1] for x in X])
    durs = n_muestras / fs
    print(f"  Duración de las épocas: min {durs.min():.2f} s | media {durs.mean():.2f} s | max {durs.max():.2f} s")

    trials_kept = [trials[i] for i in kept]
    es_ventana = a.tipo_recorte == 'ventana_cue'

    res = {
        'X': X,
        'y': np.array([CLASES[t['clase']] for t in trials_kept], dtype=int),
        'clase': np.array([t['clase'] for t in trials_kept], dtype=object),
        't_cue': np.array([t['t_cue'] for t in trials_kept], dtype=float),
        't_rest': np.array([t['t_rest'] - t['t_cue'] for t in trials_kept], dtype=float),
        't_iti': np.array([t['t_iti'] - t['t_cue'] for t in trials_kept], dtype=float),
        'epoca_tmin': ep_tmin,                   # inicio de la época respecto al cue (s)
        'epoca_tmax': ep_tmin + durs,            # fin de la época respecto al cue (s)
        'fs': fs,
        'tmin': a.tmin if es_ventana else np.nan,
        'tmax': a.tmax if es_ventana else np.nan,
        'modo_recorte': a.tipo_recorte,
        'ch_names': np.array([CANALES[c] if c < len(CANALES) else f"Ch{c}" for c in a.canales], dtype=object)
    }
    return res


# ─── Bloque Principal de Ejecución ────────────────────────────────────────────
if __name__ == '__main__':
    if TIPO_RECORTE not in ('rest_a_rest', 'ventana_cue'):
        print(f"Error: TIPO_RECORTE debe ser 'rest_a_rest' o 'ventana_cue' (actual: {TIPO_RECORTE!r})")
        sys.exit(1)
    if FIN_TRIAL not in ('siguiente_rest', 'iti_start'):
        print(f"Error: FIN_TRIAL debe ser 'siguiente_rest' o 'iti_start' (actual: {FIN_TRIAL!r})")
        sys.exit(1)

    args = SimpleNamespace(
        archivos=RUTAS_XDF, salida=CARPETA_SALIDA, tmin=TMIN, tmax=TMAX,
        canales=INDICES, eeg_stream=EEG_STREAM, eeg_indice=EEG_STREAM_INDICE,
        tipo_recorte=TIPO_RECORTE, fin_trial=FIN_TRIAL,
    )

    # Modo visualización: no toca los .xdf
    if MODO == 'visualizar':
        nombres = [CANALES[c] if c < len(CANALES) else f"Ch{c}" for c in args.canales]
        visualizar_varios(RUTAS_NPY_VISUALIZAR, args.tmin, args.tmax, nombres,
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
                guardar_trials_npy(resultado, nombre_base, args.salida)

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