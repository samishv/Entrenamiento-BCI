import numpy as np
import matplotlib.pyplot as plt
import pywt

plt.close('all')

# %% Parámetros

fs = 250
level = 6
wavelet_mom = 'db4'

FACTORES = [2.0, 0.5, 0.1, 0.25, 0.1, 2.0]
#          [ D6,  D5,  D4,  D3,  D2,  D1]

ruta = r"D:\luiso\Descargas\RegistrosCMC\RegistrosCMC\S5\EEG\S5_260323_EEG_CK_1_MI.npy"
signal = np.load(ruta)
signal = signal[:250*28][0]

N = len(signal)
t = np.arange(N) / fs

# %% Funciones auxiliares

def freq_band(fs, j):
    f_low = fs / (2**(j+1))
    f_high = fs / (2**j)
    return f_low, f_high

def reconstruct_component(coeffs, keep_index, wavelet_type):
    zeros = [np.zeros_like(c) for c in coeffs]
    zeros[keep_index] = coeffs[keep_index]
    return pywt.waverec(zeros, wavelet_type)

def filtrar_dwt(signal, wavelet, level, factores, remove_approx=False):

    coeffs = pywt.wavedec(signal, wavelet, level=level)
    coeffs_filt = [c.copy() for c in coeffs]

    if remove_approx:
        coeffs_filt[0] = np.zeros_like(coeffs_filt[0])

    if isinstance(factores, list):
        if len(factores) != level:
            raise ValueError("La lista de factores debe tener longitud = level")
        factores_dict = {j: factores[j-1] for j in range(1, level+1)}
    elif isinstance(factores, dict):
        factores_dict = factores
    else:
        raise TypeError("factores debe ser lista o diccionario")

    for j in range(1, level+1):
        c = coeffs[j]

        sigma = np.median(np.abs(c)) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(c)))

        factor = factores_dict.get(j, 1.0)
        thresh = uthresh * factor

        coeffs_filt[j] = pywt.threshold(c, thresh, mode='soft')

    return pywt.waverec(coeffs_filt, wavelet)[:len(signal)]

# %% Descomposición

coeffs = pywt.wavedec(signal, wavelet_mom, level=level)

# %% --- GRÁFICA 1: COEFICIENTES ---

fig1, axes1 = plt.subplots(level + 3, 1, figsize=(19, 9.3))

# Señal original
axes1[0].plot(t, signal)
axes1[0].set_title('Señal Original')

# Vector completo
c_all = np.concatenate(coeffs)
axes1[1].plot(c_all)
axes1[1].set_title('Descomposición Wavelet')

# Aproximación
fA = fs / (2**(level+1))
axes1[2].plot(coeffs[0])
axes1[2].set_title(f'Aproximación (0 - {fA:.2f} Hz)')

# Detalles
detalles = coeffs[1:][::-1]  # D1 → Dn

for j, (detalle, ax) in enumerate(zip(detalles, axes1[3:]), start=1):
    f_low, f_high = freq_band(fs, j)
    ax.plot(detalle)
    ax.set_title(f'Detalle {j} ({f_low:.2f} - {f_high:.2f} Hz)')

fig1.tight_layout()

# %% --- GRÁFICA 2: RECONSTRUCCIÓN ---

fig2, axes2 = plt.subplots(level + 3, 1, figsize=(19, 9.3))

# Señal original
axes2[0].plot(t, signal)
axes2[0].set_title('Señal Original')

# Señal reconstruida
signal_rec = filtrar_dwt(signal, wavelet=wavelet_mom, level=level, factores=FACTORES, remove_approx=True)
axes2[1].plot(t, signal_rec)
axes2[1].set_title('Señal Procesada FINAL')

# Aproximación reconstruida
a_rec = reconstruct_component(coeffs, 0, wavelet_mom)[:N]
axes2[2].plot(t, a_rec)
axes2[2].set_title(f'Aproximación (0 - {fA:.2f} Hz)')

# Detalles reconstruidos
for j in range(1, level+1):
    idx = level - j + 1
    d_rec = reconstruct_component(coeffs, idx, wavelet_mom)[:N]
    
    f_low, f_high = freq_band(fs, j)
    
    axes2[j+2].plot(t, d_rec)
    axes2[j+2].set_title(f'Detalle {j} ({f_low:.2f} - {f_high:.2f} Hz)')
    axes2[j+2].grid(True)

for ax in axes2:
    ax.grid(True)

fig2.tight_layout()
plt.show()
