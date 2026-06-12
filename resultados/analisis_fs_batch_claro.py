# -*- coding: utf-8 -*-
"""
Análisis batch de frecuencia de muestreo — EEG Unicorn Hybrid Black
====================================================================
Procesa todos los CSV de una carpeta, calcula la frecuencia efectiva
y el jitter de cada archivo, genera una tabla resumen y una figura
con la distribución entre sesiones.

Uso:
    python analisis_fs_batch.py                        # pide la carpeta
    python analisis_fs_batch.py ruta/a/la/carpeta      # directo

Salida (en la misma carpeta que los CSV):
    resumen_fs_eeg.csv          — tabla con métricas por archivo
    resumen_fs_eeg.png          — figura de distribución

Dependencias:
    pip install pandas numpy matplotlib scipy
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─── Configuración ────────────────────────────────────────────────────────────
CANALES    = ['Fz', 'FC3', 'FCz', 'FC4', 'Cz', 'C3', 'C4']
FS_NOMINAL = 250.0

BG  = '#ffffff'
FG  = '#1a1a1a'
MUT = '#555555'
GRD = '#cccccc'


# ─── Carga y análisis de un CSV ───────────────────────────────────────────────
def analizar_archivo(ruta):
    df = pd.read_csv(
        ruta,
        usecols=range(8),
        names=['Timestamp'] + CANALES,
        skiprows=1
    )
    if len(df) < 10:
        return None

    ts       = pd.to_datetime(df['Timestamp'], format='%H:%M:%S.%f')
    duracion = (ts.iloc[-1] - ts.iloc[0]).total_seconds()
    if duracion <= 0:
        return None

    fs_global    = (len(df) - 1) / duracion
    intervalos   = ts.diff().dt.total_seconds().dropna() * 1000
    jitter_std   = intervalos.std()
    mediana_int  = intervalos.median()

    return {
        'archivo'    : os.path.basename(ruta),
        'n_muestras' : len(df),
        'duracion_s' : round(duracion, 3),
        'fs_global'  : round(fs_global, 2),
        'jitter_std' : round(jitter_std, 3),
        'mediana_int': round(mediana_int, 3),
    }


# ─── Figura resumen ───────────────────────────────────────────────────────────
def generar_figura(resumen):
    fs   = resumen['fs_global'].values
    jit  = resumen['jitter_std'].values

    fig = plt.figure(figsize=(13, 8), facecolor=BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.35)

    def estilo(ax):
        ax.set_facecolor(BG)
        ax.tick_params(colors=MUT, labelsize=9)
        for sp in ax.spines.values():
            sp.set_color(GRD)
        ax.xaxis.label.set_color(MUT)
        ax.yaxis.label.set_color(MUT)
        ax.title.set_color(FG)

    # ── Panel 1: histograma de fs_global ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    estilo(ax1)
    bins = np.linspace(fs.min() - 1, fs.max() + 1, 30)
    ax1.hist(fs, bins=bins, color='#5A6FBF', edgecolor='none', alpha=0.85)
    ax1.axvline(FS_NOMINAL, color='#1A7FBF', lw=1.3, ls='--',
                label=f'{FS_NOMINAL:.0f} Hz nominal')
    ax1.axvline(fs.mean(), color='#CC2255', lw=1.3, ls='--',
                label=f'Media = {fs.mean():.2f} Hz')
    ax1.set_xlabel('Frecuencia de muestreo efectiva (Hz)', fontsize=10)
    ax1.set_ylabel('Número de archivos', fontsize=10)
    ax1.set_title('Distribución de $f_s$ entre sesiones', fontsize=11)
    ax1.legend(fontsize=8, facecolor='#f5f5f5', labelcolor='#1a1a1a', framealpha=0.7)

    # ── Panel 2: fs por archivo (scatter) ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    estilo(ax2)
    idx = np.arange(len(fs))
    ax2.scatter(idx, fs, s=18, color='#2A7EC0', alpha=0.7, zorder=3)
    ax2.axhline(FS_NOMINAL, color='#1A7FBF', lw=1.0, ls='--', alpha=0.8,
                label=f'{FS_NOMINAL:.0f} Hz nominal')
    ax2.axhline(fs.mean(), color='#CC2255', lw=1.0, ls='--', alpha=0.8,
                label=f'Media = {fs.mean():.2f} Hz')
    ax2.fill_between(idx,
                     fs.mean() - fs.std(),
                     fs.mean() + fs.std(),
                     color='#CC2255', alpha=0.08,
                     label=f'±1σ = {fs.std():.2f} Hz')
    ax2.set_xlabel('Índice de archivo', fontsize=10)
    ax2.set_ylabel('$f_s$ efectiva (Hz)', fontsize=10)
    ax2.set_title('$f_s$ por sesión', fontsize=11)
    ax2.legend(fontsize=8, facecolor='#f5f5f5', labelcolor='#1a1a1a', framealpha=0.7)

    # ── Panel 3: histograma de jitter ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    estilo(ax3)
    bins_j = np.linspace(jit.min() - 0.5, jit.max() + 0.5, 30)
    ax3.hist(jit, bins=bins_j, color='#7A4FA0', edgecolor='none', alpha=0.85)
    ax3.axvline(jit.mean(), color='#CC2255', lw=1.3, ls='--',
                label=f'Media = {jit.mean():.2f} ms')
    ax3.set_xlabel('Jitter std (ms)', fontsize=10)
    ax3.set_ylabel('Número de archivos', fontsize=10)
    ax3.set_title('Distribución de jitter entre sesiones', fontsize=11)
    ax3.legend(fontsize=8, facecolor='#f5f5f5', labelcolor='#1a1a1a', framealpha=0.7)

    # ── Panel 4: tabla de estadísticos ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(BG)
    ax4.axis('off')

    stats = [
        ['Métrica',              'Valor'],
        ['Archivos analizados',  f'{len(resumen)}'],
        ['$f_s$ nominal',        f'{FS_NOMINAL:.1f} Hz'],
        ['$f_s$ media ± std',    f'{fs.mean():.2f} ± {fs.std():.2f} Hz'],
        ['$f_s$ mínima',         f'{fs.min():.2f} Hz'],
        ['$f_s$ máxima',         f'{fs.max():.2f} Hz'],
        ['Jitter media ± std',   f'{jit.mean():.2f} ± {jit.std():.2f} ms'],
        ['Duración media',       f'{resumen["duracion_s"].mean():.1f} s'],
        ['Muestras media',       f'{resumen["n_muestras"].mean():.0f}'],
    ]

    tabla = ax4.table(
        cellText=stats[1:],
        colLabels=stats[0],
        cellLoc='left',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    for (row, col), cell in tabla.get_celld().items():
        cell.set_facecolor('#f0f0f0' if row % 2 == 0 else '#ffffff')
        cell.set_text_props(color='#1a1a1a' if row > 0 else '#1A7FBF')
        cell.set_edgecolor(GRD)
    ax4.set_title('Estadísticos globales', fontsize=11, color=FG, pad=10)

    # ── Pie ───────────────────────────────────────────────────────────────
    pie = (f"N = {len(resumen)} archivos  |  "
           f"$f_s$ = {fs.mean():.2f} ± {fs.std():.2f} Hz  |  "
           f"Unicorn Hybrid Black @ {FS_NOMINAL:.0f} Hz nominal")
    fig.text(0.5, 0.005, pie, ha='center', color=MUT, fontsize=8.5)

    return fig


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) > 1:
        carpeta = sys.argv[1]
    else:
        carpeta = input('Ruta de la carpeta con los CSV: ').strip().strip('"')

    if not os.path.isdir(carpeta):
        print(f'Error: no se encontró la carpeta "{carpeta}"')
        sys.exit(1)

    archivos = sorted([
        os.path.join(carpeta, f)
        for f in os.listdir(carpeta)
        if f.lower().endswith('.csv')
    ])

    if not archivos:
        print('No se encontraron archivos CSV en la carpeta.')
        sys.exit(1)

    print(f'\nArchivos encontrados: {len(archivos)}')
    print('Procesando...\n')

    resultados = []
    errores    = []

    for i, ruta in enumerate(archivos, 1):
        nombre = os.path.basename(ruta)
        try:
            r = analizar_archivo(ruta)
            if r:
                resultados.append(r)
                print(f'  [{i:>3}/{len(archivos)}] {nombre:<45} '
                      f'fs={r["fs_global"]:.2f} Hz   '
                      f'jitter={r["jitter_std"]:.2f} ms')
            else:
                errores.append(nombre)
                print(f'  [{i:>3}/{len(archivos)}] {nombre:<45} OMITIDO (muy corto)')
        except Exception as e:
            errores.append(nombre)
            print(f'  [{i:>3}/{len(archivos)}] {nombre:<45} ERROR: {e}')

    if not resultados:
        print('\nNo se pudo analizar ningún archivo.')
        sys.exit(1)

    resumen = pd.DataFrame(resultados)

    # Imprimir estadísticos
    fs  = resumen['fs_global']
    jit = resumen['jitter_std']
    sep = '─' * 50
    print(f'\n{sep}')
    print(f'  Archivos analizados : {len(resumen)} / {len(archivos)}')
    print(f'  fs nominal          : {FS_NOMINAL:.1f} Hz')
    print(f'  fs media ± std      : {fs.mean():.2f} ± {fs.std():.2f} Hz')
    print(f'  fs mínima / máxima  : {fs.min():.2f} / {fs.max():.2f} Hz')
    print(f'  Jitter media ± std  : {jit.mean():.2f} ± {jit.std():.2f} ms')
    print(f'  Duración media      : {resumen["duracion_s"].mean():.1f} s')
    if errores:
        print(f'  Archivos con error  : {len(errores)}')
    print(sep)

    # Guardar CSV resumen
    ruta_csv = os.path.join(carpeta, 'resumen_fs_eeg.csv')
    resumen.to_csv(ruta_csv, index=False)
    print(f'\nTabla guardada en:\n  {ruta_csv}')

    # Generar y guardar figura
    print('\nGenerando figura ...')
    fig      = generar_figura(resumen)
    ruta_fig = os.path.join(carpeta, 'resumen_fs_eeg.png')
    fig.savefig(ruta_fig, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'Figura guardada en:\n  {ruta_fig}\n')
    plt.show()


if __name__ == '__main__':
    main()
