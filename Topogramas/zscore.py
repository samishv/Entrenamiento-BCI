import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.viz import plot_topomap

df = pd.read_excel("datosCMC2.xlsx")
#print(plt.colormaps())
metrica = "Max gamma Z" 
sujetos = df["sujeto"].unique()
musculos = df["musculo"].unique()

# Escala fija para z-scores
vmin, vmax = -3, 3

fig, axes = plt.subplots(len(musculos), len(sujetos), figsize=(5 * len(sujetos), 4 * len(musculos)))

# Asegurar que axes siempre sea 2D
if len(musculos) == 1 and len(sujetos) == 1:
    axes = np.array([[axes]])
elif len(musculos) == 1:
    axes = axes[np.newaxis, :]
elif len(sujetos) == 1:
    axes = axes[:, np.newaxis]

for a, m in enumerate(musculos):
    for i, s in enumerate(sujetos):

        subset = df[
            (df["sujeto"] == s) &
            (df["musculo"] == m)
        ]

        if subset.empty:
            axes[a, i].set_title("(sin datos)" if a == 0 else "")
            axes[a, i].axis("off")
            continue

        valores = subset[metrica].values
        canales = subset["canal eeg"].values
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
            axes=axes[a, i],
            extrapolate='local',
            contours=0,
            sensors=True,
            cmap='viridis',
            vlim=(vmin, vmax),
            show=False
        )

        # Anotar el z-score en cada canal
        layout = mne.channels.make_eeg_layout(info)
        pos = layout.pos[:, :2]

        # Normalizar posiciones al rango del eje
        pos_norm = pos.copy()
        pos_norm[:, 0] = (pos[:, 0] - pos[:, 0].min()) / (pos[:, 0].max() - pos[:, 0].min())
        pos_norm[:, 1] = (pos[:, 1] - pos[:, 1].min()) / (pos[:, 1].max() - pos[:, 1].min())

        ax_x0, ax_y0 = axes[a, i].get_position().x0, axes[a, i].get_position().y0
        ax_w, ax_h = axes[a, i].get_position().width, axes[a, i].get_position().height

        # Título solo en primera fila
        if a == 0:
            axes[a, i].set_title(f"{s}", fontsize=12)
        else:
            axes[a, i].set_title("")

        # Etiqueta de músculo solo en primera columna
        if i == 0:
            axes[a, i].set_ylabel(f"{m}", fontsize=12, labelpad=10)

plt.suptitle(metrica, fontsize=14, y=1.01)

# Una sola barra de color para todos
cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.05, pad=0.04)
cbar.set_label("Z-score", fontsize=12)
cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])

# Línea en 0 en la barra de color para referencia visual
cbar.ax.axhline(y=0, color='black', linewidth=1.5)

plt.savefig(f"topomap_{metrica.replace(' ', '_')}_score.png", dpi=300, bbox_inches='tight')
plt.show()