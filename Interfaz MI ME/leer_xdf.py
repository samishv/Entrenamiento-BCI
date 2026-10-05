# -*- coding: utf-8 -*-
"""
Lee uno o varios .xdf (LabRecorder) del protocolo MI/ME y recorta los trials
usando los marcadores LSL de 'MIME_Markers'.

Por cada trial se guarda una época alineada al cue (marcador cue_MI / cue_ME):
    tiempo 0 = flip en que aparece el cue (y suena el beep)
    por defecto de -5 s (reposo con '+') a +5 s (tarea)

No filtra ni re-referencia: las épocas salen crudas (µV). El preprocesamiento
debe vivir en un módulo común para entrenamiento y fase en línea.

Uso:
    python leer_xdf.py bloque1.xdf bloque2.xdf bloque3.xdf
    python leer_xdf.py "Data/*.xdf" --salida ./Trials
    python leer_xdf.py bloque1.xdf --triallog S1_..._B1_triallog.csv   # verificación cruzada
    python leer_xdf.py bloque1.xdf --tmin -2 --tmax 5 --canales 0 1 2 3 4 5 7

Salida (.npz por archivo; y 'dataset_unido.npz' si hay varios archivos):
    X        (n_trials, n_canales, n_muestras)  float64, µV
    y        (n_trials,)  0 = MI, 1 = ME
    clase    (n_trials,)  'MI' / 'ME'
    t_cue    (n_trials,)  timestamp LSL del cue
    t_rest   (n_trials,)  inicio del reposo, relativo al cue (s)
    t_iti    (n_trials,)  inicio del descanso, relativo al cue (s)
    bloque   (n_trials,)  índice del archivo de origen (solo en el unido)
    fs, tmin, tmax, ch_names

Cargar después:
    d = np.load('Trials/dataset_unido.npz', allow_pickle=True)
    X, y = d['X'], d['y']
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import pyxdf

CANALES = ['Fz', 'FC3', 'FCz', 'FC4', 'Cz', 'C3', 'C4']
INDICES = [0, 1, 2, 3, 4, 5, 7]          # el 6 'NC' se omite
CLASES = {'MI': 0, 'ME': 1}
NOMBRE_MARCADORES = 'MIME_Markers'


# ─── Carga ────────────────────────────────────────────────────────────────────
def _nombre(s):
    return s['info']['name'][0]


def _tipo(s):
    return s['info']['type'][0]


def cargar_streams(ruta, eeg_name=None, marker_name=NOMBRE_MARCADORES):
    """Devuelve (stream_eeg, stream_marcadores) de un .xdf."""
    streams, _ = pyxdf.load_xdf(ruta)
    disponibles = [f"{_nombre(s)} ({_tipo(s)})" for s in streams]

    marc = [s for s in streams if _nombre(s) == marker_name]
    if len(marc) != 1:
        raise ValueError(f"Se esperaba 1 stream '{marker_name}' y hay {len(marc)}. "
                         f"Streams en el archivo: {disponibles}")

    if eeg_name:
        eeg = [s for s in streams if _nombre(s) == eeg_name]
    else:
        eeg = [s for s in streams if _tipo(s).upper() == 'EEG']
    if len(eeg) != 1:
        raise ValueError(f"Se esperaba 1 stream EEG y hay {len(eeg)}. "
                         f"Usa --eeg_stream. Streams en el archivo: {disponibles}")
    return eeg[0], marc[0]


def leer_marcadores(s_marc):
    """Lista ordenada de (timestamp, etiqueta)."""
    t = np.asarray(s_marc['time_stamps'], dtype=float)
    etiquetas = []
    for m in s_marc['time_series']:
        etiquetas.append(str(m[0] if isinstance(m, (list, tuple, np.ndarray)) else m))
    orden = np.argsort(t, kind='stable')
    return [(t[i], etiquetas[i]) for i in orden]


# ─── Segmentación por marcadores ──────────────────────────────────────────────
def segmentar(marcadores):
    """
    Agrupa marcadores en trials completos: rest_start -> cue_X -> iti_start.
    Un trial interrumpido (abort, falta de marcador) se descarta.
    Devuelve (trials, info) con info = dict de conteos y banderas.
    """
    trials, cur = [], None
    info = {'abort': False, 'block_start': False, 'block_end': False,
            'incompletos': 0}
    for t, etq in marcadores:
        if etq == 'block_start':
            info['block_start'] = True
        elif etq == 'rest_start':
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
            elif cur is not None:
                info['incompletos'] += 1
            cur = None
        elif etq == 'block_end':
            info['block_end'] = True
            cur = None
        elif etq == 'block_abort':
            info['abort'] = True
            if cur is not None:
                info['incompletos'] += 1
            cur = None
    if cur is not None:
        info['incompletos'] += 1
    return trials, info


# ─── Épocas ───────────────────────────────────────────────────────────────────
def extraer_epocas(eeg_t, eeg_x, trials, fs, tmin, tmax):
    """Recorta [t_cue+tmin, t_cue+tmax) por conteo de muestras.
    Descarta épocas fuera del rango del EEG o con huecos en el stream.
    Devuelve (X, kept, motivos_descarte)."""
    n = int(round((tmax - tmin) * fs))
    X, kept, descartes = [], [], []
    for k, tr in enumerate(trials):
        t0 = tr['t_cue'] + tmin
        if t0 < eeg_t[0]:
            descartes.append((k, 'empieza antes del EEG'))
            continue
        i0 = int(np.searchsorted(eeg_t, t0))
        if i0 + n > len(eeg_t):
            descartes.append((k, 'termina después del EEG'))
            continue
        dur = eeg_t[i0 + n - 1] - eeg_t[i0]
        if abs(dur - (n - 1) / fs) > 2.0 / fs:
            descartes.append((k, f'hueco en el stream EEG (dur {dur:.3f} s)'))
            continue
        X.append(eeg_x[i0:i0 + n].T)          # (canales, muestras)
        kept.append(k)
    X = np.stack(X) if X else np.empty((0, eeg_x.shape[1], n))
    return X, kept, descartes


# ─── Verificación cruzada con el triallog ─────────────────────────────────────
def comparar_triallog(ruta_csv, trials):
    with open(ruta_csv, newline='', encoding='utf-8') as f:
        filas = list(csv.DictReader(f))
    clases_csv = [r['clase'] for r in filas]
    clases_xdf = [t['clase'] for t in trials]
    print(f"  [triallog] trials CSV={len(filas)} | XDF={len(trials)}")
    if clases_csv != clases_xdf:
        print("  [triallog] ¡LAS SECUENCIAS DE CLASES NO COINCIDEN!")
        return
    print("  [triallog] secuencia de clases coincide")
    dt = np.array([float(r['t_cue_lsl']) for r in filas]) - \
         np.array([t['t_cue'] for t in trials])
    print(f"  [triallog] t_cue CSV - XDF: media {dt.mean()*1000:.2f} ms, "
          f"máx |.| {np.abs(dt).max()*1000:.2f} ms")


# ─── Procesar un archivo ──────────────────────────────────────────────────────
def procesar(ruta, a):
    print(f"\n=== {os.path.basename(ruta)} ===")
    s_eeg, s_marc = cargar_streams(ruta, a.eeg_stream)

    eeg_t = np.asarray(s_eeg['time_stamps'], dtype=float)
    eeg_x_all = np.asarray(s_eeg['time_series'], dtype=float)
    fs = float(s_eeg['info']['nominal_srate'][0])
    n_ch = eeg_x_all.shape[1]
    if max(a.canales) >= n_ch:
        raise ValueError(f"El stream EEG tiene {n_ch} canales y pediste el índice "
                         f"{max(a.canales)}")
    eeg_x = eeg_x_all[:, a.canales]

    dur = eeg_t[-1] - eeg_t[0]
    fs_ef = (len(eeg_t) - 1) / dur if dur > 0 else float('nan')
    max_dt = np.diff(eeg_t).max() if len(eeg_t) > 1 else float('nan')
    print(f"  EEG: {len(eeg_t)} muestras, {n_ch} canales en el stream, "
          f"{dur:.1f} s | fs nominal {fs:g} Hz, efectiva {fs_ef:.2f} Hz")
    if max_dt > 5.0 / fs:
        print(f"  [AVISO] hueco máximo entre muestras: {max_dt*1000:.0f} ms "
              f"(posible pérdida de datos por Bluetooth)")

    marcadores = leer_marcadores(s_marc)
    trials, info = segmentar(marcadores)
    print(f"  Marcadores: {len(marcadores)} | trials completos: {len(trials)} "
          f"| incompletos: {info['incompletos']}")
    if not info['block_start']:
        print("  [AVISO] no hay 'block_start' (¿Start de LabRecorder tras empezar?)")
    if info['abort']:
        print("  [AVISO] el bloque terminó con 'block_abort'")
    elif not info['block_end']:
        print("  [AVISO] no hay 'block_end' (grabación cortada antes del final)")

    t_m0, t_m1 = marcadores[0][0], marcadores[-1][0]
    if t_m0 < eeg_t[0] or t_m1 > eeg_t[-1]:
        print(f"  [AVISO] marcadores fuera del rango del EEG "
              f"(EEG {eeg_t[0]:.2f}-{eeg_t[-1]:.2f}, marcadores {t_m0:.2f}-{t_m1:.2f})")

    if a.triallog:
        comparar_triallog(a.triallog, trials)

    X, kept, descartes = extraer_epocas(eeg_t, eeg_x, trials, fs, a.tmin, a.tmax)
    for k, motivo in descartes:
        print(f"  [DESCARTADO] trial {k + 1} ({trials[k]['clase']}): {motivo}")

    sel = [trials[k] for k in kept]
    clase = np.array([t['clase'] for t in sel])
    y = np.array([CLASES[c] for c in clase], dtype=int)
    print(f"  Épocas guardadas: {len(sel)} "
          f"(MI={int((clase == 'MI').sum())}, ME={int((clase == 'ME').sum())}) "
          f"| forma {X.shape}")

    # Duraciones reales (diagnóstico del paradigma)
    if sel:
        d_rest = np.array([t['t_cue'] - t['t_rest'] for t in sel])
        d_task = np.array([t['t_iti'] - t['t_cue'] for t in sel])
        print(f"  Duración real reposo {d_rest.mean():.3f} s "
              f"[{d_rest.min():.3f}, {d_rest.max():.3f}] | "
              f"tarea {d_task.mean():.3f} s [{d_task.min():.3f}, {d_task.max():.3f}]")

    return dict(
        X=X, y=y, clase=clase,
        t_cue=np.array([t['t_cue'] for t in sel]),
        t_rest=np.array([t['t_rest'] - t['t_cue'] for t in sel]),
        t_iti=np.array([t['t_iti'] - t['t_cue'] for t in sel]),
        fs=fs, tmin=a.tmin, tmax=a.tmax,
        ch_names=np.array([CANALES[INDICES.index(i)] if i in INDICES else f'ch{i}'
                           for i in a.canales]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('archivos', nargs='+', help='.xdf (admite comodines)')
    ap.add_argument('--salida', default='./Trials')
    ap.add_argument('--eeg_stream', default=None,
                    help="nombre del stream EEG (por defecto: el de tipo 'EEG')")
    ap.add_argument('--tmin', type=float, default=-5.0, help='s respecto al cue')
    ap.add_argument('--tmax', type=float, default=5.0, help='s respecto al cue')
    ap.add_argument('--canales', type=int, nargs='+', default=INDICES,
                    help='índices de canal del stream (por defecto 0-5 y 7)')
    ap.add_argument('--triallog', default=None,
                    help='triallog.csv para verificar (solo con un archivo)')
    a = ap.parse_args()

    rutas = []
    for p in a.archivos:
        rutas.extend(sorted(glob.glob(p)) or [p])
    if a.triallog and len(rutas) != 1:
        sys.exit("--triallog solo se puede usar con un archivo .xdf")

    os.makedirs(a.salida, exist_ok=True)
    resultados = []
    for i, ruta in enumerate(rutas):
        try:
            r = procesar(ruta, a)
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue
        r['bloque'] = np.full(len(r['y']), i, dtype=int)
        base = os.path.splitext(os.path.basename(ruta))[0]
        destino = os.path.join(a.salida, base + '_trials.npz')
        np.savez_compressed(destino, **{k: v for k, v in r.items() if k != 'bloque'})
        print(f"  -> {destino}")
        resultados.append((ruta, r))

    if len(resultados) > 1:
        rs = [r for _, r in resultados]
        unido = {k: np.concatenate([r[k] for r in rs])
                 for k in ('X', 'y', 'clase', 't_cue', 't_rest', 't_iti', 'bloque')}
        unido.update(fs=rs[0]['fs'], tmin=a.tmin, tmax=a.tmax,
                     ch_names=rs[0]['ch_names'],
                     archivos=np.array([os.path.basename(p) for p, _ in resultados]))
        destino = os.path.join(a.salida, 'dataset_unido.npz')
        np.savez_compressed(destino, **unido)
        print(f"\nDataset unido: {unido['X'].shape} "
              f"(MI={int((unido['y'] == 0).sum())}, ME={int((unido['y'] == 1).sum())}) "
              f"-> {destino}")


if __name__ == '__main__':
    main()
