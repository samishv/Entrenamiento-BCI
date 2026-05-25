import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# 1. Simulación de una señal EMG a 1000 Hz
fs_original = 1000
fs_nueva = 250
duration = 0.2  # Graficamos solo 0.2 segundos para ver los detalles con claridad
t_original = np.linspace(0, duration, int(fs_original * duration), endpoint=False)

# Señal muscular base (50 Hz) + Ruido eléctrico/artefacto de alta frecuencia (300 Hz)
emg_puro = np.sin(2 * np.pi * 50 * t_original) 
ruido_alta_frec = 0.5 * np.sin(2 * np.pi * 300 * t_original)
senal_emg = emg_puro + ruido_alta_frec

# 2. MÉTODO 1: Promedio de 4 muestras (Tu propuesta)
muestras_a_promediar = 4
longitud_truncada = (len(senal_emg) // muestras_a_promediar) * muestras_a_promediar
emg_promediado = senal_emg[:longitud_truncada].reshape(-1, muestras_a_promediar).mean(axis=1)
t_promediado = np.linspace(0, duration, len(emg_promediado), endpoint=False)

# 3. MÉTODO 2: Resampleo profesional (Filtro Antialiasing + Diezmado)
emg_correcto = signal.resample_poly(senal_emg, up=1, down=4)
t_correcto = np.linspace(0, duration, len(emg_correcto), endpoint=False)

# 4. Generación de las gráficas
plt.figure(figsize=(12, 6))

# Gráfica del Método 1
plt.subplot(2, 1, 1)
plt.plot(t_original, senal_emg, label='Original (1000 Hz con ruido)', color='gray', alpha=0.5)
plt.plot(t_promediado, emg_promediado, 'o-', label='Resampleo por Promedio (250 Hz)', color='crimson', markersize=4)
plt.title('Método por Promedio Móvil (Genera Aliasing)')
plt.ylabel('Amplitud (mV)')
plt.grid(True)
plt.legend()

# Gráfica del Método 2
plt.subplot(2, 1, 2)
plt.plot(t_original, senal_emg, label='Original (1000 Hz con ruido)', color='gray', alpha=0.5)
plt.plot(t_correcto, emg_correcto, 'o-', label='Resampleo Polyphase (250 Hz)', color='teal', markersize=4)
plt.title('Método Profesional con scipy.signal (Limpio de Aliasing)')
plt.xlabel('Tiempo (segundos)')
plt.ylabel('Amplitud (mV)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
