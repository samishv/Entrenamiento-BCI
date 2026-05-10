
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.viz import plot_topomap

df = pd.read_excel("datosCMC.xlsx")
sujetos = df["sujeto"].unique()
musculos = df["musculo"].unique()

# rango global (para comparar colores correctamente)
vmin = df["Area Beta"].min()
vmax = df["Area Beta"].max()

fig, axes = plt.subplots(len(sujetos), len(musculos), figsize=(5 * len(musculos), 4*len(sujetos)))

for a,s in enumerate(sujetos):

    #fig, axes = plt.subplots(1, len(musculos), figsize=(5 * len(musculos), 4))

    # en caso de que haya un solo músculo
    if len(musculos) == 1:
        axes = [axes]

    for i, m in enumerate(musculos):

        subset = df[
            (df["sujeto"] == s) &
            (df["musculo"] == m)
        ]

        if subset.empty:
            axes[a,i].set_title(f"{m}\n(sin datos)")
            continue

        valores = subset["Area Beta"].values
        canales = subset["canal eeg"].values

        # limpiar nombres si aún tienen sufijos
        canales = [c.split("-")[0] for c in canales]

        info = mne.create_info(
            ch_names=list(canales),
            sfreq=256,
            ch_types='eeg'
        )
        info.set_montage('standard_1020')

        im, _ = plot_topomap(
            valores,
            info,
            axes=axes[i],
            extrapolate='local',
            contours=0,
            sensors=True,
            cmap='viridis',
            vlim=(vmin, vmax),
            show=False
        )

        axes[a,i].set_title(s,m)

    # UNA SOLA barra de color para todos los músculos
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.05)

    plt.suptitle(f"Sujeto {s} - Area Beta")
    #plt.tight_layout()

    # GUARDAR ANTES de show
    plt.savefig(f"topomap_{s}_Area_Beta.png", dpi=300)

    plt.show()
