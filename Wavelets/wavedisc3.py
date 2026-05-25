import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import iirnotch, filtfilt  # <-- NUEVO

plt.close('all')

# %% Parámetros

fs = 1000
level = 6
wavelet_mom = 'sym4'

ruta = r"D:\luiso\Documentos\Luis\UPIITA\TT\Entrenamiento-BCI\DataCMC\Registros CMC 2\S18\EMG\S18_260429_EMG_OV_1_ME.npy"
FACTORES = [1.5, 1.5, 0.0, 0.0, 0.0, 0.0] 
#          [ D6,  D5,  D4,  D3,  D2,  D1]

signal_original = np.load(ruta)
signal_original = signal_original[0]
signal_original = signal_original[0:1000*2]

N = len(signal_original)
t = np.arange(N) / fs

# %% NOTCH

f0 = 60
Q = 30

b, a = iirnotch(w0=f0, Q=Q, fs=fs)
signal_notch = filtfilt(b, a, signal_original)

f0 = 18
Q = 3

b, a = iirnotch(w0=f0, Q=Q, fs=fs)
signal_notch = filtfilt(b, a, signal_notch)

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

    for i in range(1, len(coeffs)):
        factor = factores[i-1]
        
        if factor > 0:
            c = coeffs[i]
            sigma = np.median(np.abs(c)) / 0.6745
            uthresh = sigma * np.sqrt(2 * np.log(len(c)))
            coeffs_filt[i] = pywt.threshold(c, uthresh * factor, mode='soft')
            
    return pywt.waverec(coeffs_filt, wavelet)[:len(signal)]

# %% Descomposición

coeffs = pywt.wavedec(signal_notch, wavelet_mom, level=level)

# %% --- GRÁFICA 1: COEFICIENTES ---

fig1, axes1 = plt.subplots(level + 3, 1, figsize=(19, 9.3))

axes1[0].plot(t, signal_original)
axes1[0].set_title('Señal Original (con notch 60 Hz)')

c_all = np.concatenate(coeffs)
axes1[1].plot(c_all)
axes1[1].set_title('Descomposición Wavelet')

fA = fs / (2**(level+1))
axes1[2].plot(coeffs[0])
axes1[2].set_title(f'Aproximación (0 - {fA:.2f} Hz)')

detalles = coeffs[1:][::-1]

for j, (detalle, ax) in enumerate(zip(detalles, axes1[3:]), start=1):
    f_low, f_high = freq_band(fs, j)
    ax.plot(detalle)
    ax.set_title(f'Detalle {j} ({f_low:.2f} - {f_high:.2f} Hz)')

fig1.tight_layout()

# %% --- GRÁFICA 2: RECONSTRUCCIÓN ---

fig2, axes2 = plt.subplots(level + 3, 1, figsize=(19, 9.3))

axes2[0].plot(t, signal_original)
axes2[0].set_title('Señal Original (con notch)')

signal_wavelet = filtrar_dwt(signal_notch, wavelet=wavelet_mom, level=level, factores=FACTORES, remove_approx=True)
axes2[1].plot(t, signal_wavelet)
axes2[1].set_title('Señal Procesada FINAL')

a_rec = reconstruct_component(coeffs, 0, wavelet_mom)[:N]
axes2[2].plot(t, a_rec)
axes2[2].set_title(f'Aproximación (0 - {fA:.2f} Hz)')

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

# %% FFT

fft_vals = np.fft.fft(signal_wavelet)
fft_vals = np.abs(fft_vals) / N

freqs = np.fft.fftfreq(N, d=1/fs)

# Solo parte positiva
mask = freqs >= 0
freqs_pos = freqs[mask]
fft_pos = fft_vals[mask]

# %% FIGURA 3

fig3, axes3 = plt.subplots(4, 1, figsize=(19, 9.3))

# 1. Señal original
axes3[0].plot(t, signal_original)
axes3[0].set_title('Señal Original')
axes3[0].grid(True)

# 2. Señal con notch
axes3[1].plot(t, signal_notch)
axes3[1].set_title('Señal con Filtro Notch (60 Hz, Q=30)')
axes3[1].grid(True)

# 3. Señal después de wavelets
axes3[2].plot(t, signal_wavelet)
axes3[2].set_title('Señal después de DWT')
axes3[2].grid(True)

# 4. FFT señal final
axes3[3].plot(freqs_pos, fft_pos)
axes3[3].set_title('FFT de la Señal (post-DWT)')
axes3[3].set_xlim(0, fs/2)
axes3[3].grid(True)

fig3.tight_layout()

plt.show()