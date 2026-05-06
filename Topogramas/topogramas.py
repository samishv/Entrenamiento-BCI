import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.viz import plot_topomap


electrodos = ['Fz', 'FC3', 'FCz', 'FC4', 'Cz', 'C3', 'C4']
valores = np.array([5.2, 4.8, 6.1, 5.9, 7.0, 5.5, 5.7])
# Crear un objeto Info con la información de los electrodos
info = mne.create_info(ch_names=electrodos, sfreq=250, ch_types='eeg')

montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage, match_case=False)

fig, ax = plt.subplots()

im, _ = plot_topomap(
    valores,
    info,
    axes=ax,
    extrapolate='local', 
    outlines='head',
    contours=0,
    sensors=True,
    cmap='viridis',
    show=False
)

plt.colorbar(im, ax=ax)

ax.set_title("Topograma EEG (10-10, 7 electrodos)")
plt.show()
