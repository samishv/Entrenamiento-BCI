import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.viz import plot_topomap
from datetime import datetime

ahora = datetime.now()
df = pd.read_excel("metricas_CMC.xlsx")
metricas = ["Area_Beta", "Area_Gamma", "Area_Mu", "Max_Beta", "Max_Gamma", "Max_Mu", "Area_Beta_S", "Area_Gamma_S", "Area_Mu_S", "Max_Beta_S", "Max_Gamma_S", "Max_Mu_S", "Area_Beta_M", "Area_Gamma_M", "Area_Mu_M", "Max_Beta_M", "Max_Gamma_M", "Max_Mu_M"]  # Puedes agregar más métricas aquí

for metrica in metricas:
    sujetos = df["Sujeto"].unique()
    musculos = df["Musculo"].unique()

    # rango global (para comparar colores correctamente)
    vmin = df[metrica].min()
    vmax = df[metrica].max()

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
                (df["Sujeto"] == s) &
                (df["Musculo"] == m)
            ]

            if subset.empty:
                axes[a, i].set_title(f"(sin datos)" if a == 0 else "")
                axes[a, i].axis("off")
                continue

            valores = subset[metrica].values
            canales = subset["Canal_EEG"].values

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
                axes=axes[a, i],
                extrapolate='local',
                contours=4,
                sensors=True,
                cmap='viridis',
                vlim=(vmin, vmax),
                show=False
            )

            # Título solo en la primera fila
            if a == 0:
                axes[a, i].set_title(f"{s}", fontsize=12)
            else:
                axes[a, i].set_title("")

            # Etiqueta de músculo solo en la primera columna
            if i == 0:
                axes[a, i].set_ylabel(f"{m}", fontsize=12, labelpad=10)

    plt.suptitle(metrica, fontsize=14, y=1.01)

    # Una sola barra de color para todos
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.05)

    # Guardar
    plt.savefig(f"topomap_{metrica.replace(' ', '_')}_{ahora.strftime('%y-%m-%d_%H-%M')}.png", dpi=300, bbox_inches='tight')

    #plt.show()