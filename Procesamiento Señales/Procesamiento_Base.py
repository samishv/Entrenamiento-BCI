import matplotlib.pyplot as plt

from Procesamiento_Funciones import (
    cargar_npy_eeg,
    aplicar_notch_multiple,
    aplicar_wavelet_dwt_multicanal,
    ica_eliminar_componente,
    aplicar_pasabanda_butter,
)

plt.close('all')


# %% CONFIGURACION

RUTA_NPY = r"D:\\luiso\\Documentos\\Luis\\UPIITA\\TT\\CMC\\S5\\EEG\\S5_260429_EEG_OV_2_MI.npy"

FS = 250
CANALES = ['Fz', 'FC3', 'FCz', 'FC4', 'Cz', 'C3', 'C4']

# Notch
FREQS_NOTCH = (60.0, 120.0)
Q_NOTCH = 30
PAD = 5 * FS            # PADDING PARA FILTROS

# Wavelet
WAVELET = 'db4'
LEVEL = 6
NIVELES_CERO = [1, 6]   # QUITAR D1 y D6
REMOVE_APPROX = True    # QUITAR A6
WAVE_MODE = 'reflect'

# ICA
IC_TO_REMOVE = 0        # IC A QUITAR [X,X]

# PASA BANDA
BP_LOW =    5.0         
BP_HIGH =   45.0
BP_ORDER =  3


# %% PROCESAMIENTO DE LA SEÑAL 

# 1) Cargar el npy
datos = cargar_npy_eeg(RUTA_NPY, n_canales_esperados=len(CANALES))

# 2) Notch
datos = aplicar_notch_multiple(datos, fs=FS, freqs=FREQS_NOTCH, Q=Q_NOTCH, pad=PAD)

# 3) Wavelet
datos = aplicar_wavelet_dwt_multicanal(
    datos,
    wavelet=WAVELET,
    level=LEVEL,
    niveles_cero=NIVELES_CERO,
    remove_approx=REMOVE_APPROX,
    pad=PAD,
    mode=WAVE_MODE,
)

# 4) ICA
datos, _ica = ica_eliminar_componente(datos, fs=FS, ch_names=CANALES, ic_to_remove=IC_TO_REMOVE)

# 5) PASA BANDA
datos_final = aplicar_pasabanda_butter(datos, fs=FS, f_low=BP_LOW, f_high=BP_HIGH, orden=BP_ORDER, pad=PAD)


# %% PLOTEO

fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(19, 9))

for i in range(7):
    axes[i].plot(datos_final[i])
    axes[i].set_title(f'{CANALES[i]}')
    axes[i].set_ylabel('Valor')

plt.tight_layout()
plt.show()