import numpy as np
import pywt

import mne
from mne.preprocessing import ICA

from scipy.signal import iirnotch, sosfiltfilt, tf2sos, butter, resample_poly



# %% CARGAR DATOS EEG

def cargar_npy_eeg(ruta_npy: str, n_canales_esperados: int | None = None) -> np.ndarray:
    """Carga un archivo NPY con señales EEG.
    
        Entradas:
            ruta_npy: Ruta del archivo .npy.
            n_canales_esperados: Cantidad esperada de canales, o None para no validarla.
    
        Retorna:
            Arreglo float con forma (canales, muestras).
    
        Excepciones:
            ValueError: Si los datos no son 2D o no coinciden con los canales esperados."""
    datos = np.load(ruta_npy)
    if datos.ndim != 2:
        raise ValueError(f"Se esperaba un arreglo 2D (canales x muestras). Recibido: {datos.shape}")
    if n_canales_esperados is not None and datos.shape[0] != n_canales_esperados:
        raise ValueError(
            f"Se esperaban {n_canales_esperados} canales, pero el archivo trae {datos.shape[0]} canales: {datos.shape}"
        )
    return np.asarray(datos, dtype=float)


# %% CARGAR DATOS EMG

def cargar_npy_emg(
    ruta_npy: str,
    n_canales_esperados: int | None = None,
    permitir_transponer: bool = True,
) -> np.ndarray:
    """Carga un archivo NPY con señales EMG y normaliza su orientación.
    
        Entradas:
            ruta_npy: Ruta del archivo .npy.
            n_canales_esperados: Cantidad esperada de canales, o None.
            permitir_transponer: Permite convertir (muestras, canales) a (canales, muestras).
    
        Retorna:
            Arreglo float con forma (canales, muestras).
    
        Excepciones:
            ValueError: Si la dimensión u orientación no es válida."""
    datos = np.load(ruta_npy)

    if datos.ndim == 1:
        datos = datos.reshape(1, -1)
    elif datos.ndim != 2:
        raise ValueError(f"Se esperaba 1D o 2D para EMG. Recibido: {datos.shape}")
    if permitir_transponer and n_canales_esperados is not None:
        if datos.shape[0] != n_canales_esperados and datos.shape[1] == n_canales_esperados:
            datos = datos.T
    if permitir_transponer and n_canales_esperados is None:
        if datos.shape[0] > datos.shape[1]:
            datos = datos.T
    if n_canales_esperados is not None and datos.shape[0] != n_canales_esperados:
        raise ValueError(
            f"Se esperaban {n_canales_esperados} canales EMG, pero el archivo trae {datos.shape[0]}: {datos.shape}"
        )
    return np.asarray(datos, dtype=float)


# %% CARGAR DATOS DINAM

def cargar_npy_dinam(
    ruta_npy: str,
    permitir_multicanal: bool = False,
) -> np.ndarray:
    """Carga una señal de dinamómetro desde un archivo NPY.
    
        Entradas:
            ruta_npy: Ruta del archivo .npy.
            permitir_multicanal: Si es True, toma el primer canal de una matriz multicanal.
    
        Retorna:
            Vector float con forma (muestras,).
    
        Excepciones:
            ValueError: Si el archivo no puede interpretarse como una señal unidimensional."""
    datos = np.load(ruta_npy)
    if datos.ndim == 1:
        vec = datos
    elif datos.ndim == 2:
        if 1 in datos.shape:
            vec = datos.reshape(-1)
        else:
            if not permitir_multicanal:
                raise ValueError(f"DINAM debe ser 1D (o 2D con una dimensión = 1). Recibido: {datos.shape}")
            vec = datos[0].reshape(-1)
    else:
        raise ValueError(f"DINAM debe ser 1D o 2D. Recibido: {datos.shape}")
    return np.asarray(vec, dtype=float)


# %% FILTRADO DE NOTCH

def _filtrar_con_padding_reflect(x: np.ndarray, sos: np.ndarray, pad: int) -> np.ndarray:
    """Aplica un filtro SOS de fase cero con padding reflectante opcional.
    
        Entradas:
            x: Vector de señal.
            sos: Coeficientes del filtro en secciones de segundo orden.
            pad: Muestras añadidas en cada extremo; 0 desactiva el padding.
    
        Retorna:
            Vector filtrado con la misma longitud que x."""
    if pad is None or pad <= 0:
        return sosfiltfilt(sos, x)
    x_pad = np.pad(x, pad_width=pad, mode="reflect")
    y_pad = sosfiltfilt(sos, x_pad)
    return y_pad[pad:-pad]


def aplicar_notch_multiple(
    datos: np.ndarray,
    fs: float,
    freqs: list[float] | tuple[float, ...] = (60.0, 120.0),
    Q: float = 30.0,
    pad: int = 0,
) -> np.ndarray:
    """Aplica filtros notch consecutivos para eliminar frecuencias específicas.
    
        Entradas:
            datos: Matriz (canales, muestras).
            fs: Frecuencia de muestreo en Hz.
            freqs: Frecuencias que se desean atenuar, por ejemplo 60 y 120 Hz.
            Q: Factor de calidad de cada notch.
            pad: Muestras de padding reflectante para reducir efectos de borde.
    
        Retorna:
            Matriz filtrada con la misma forma que datos."""
    salida = np.asarray(datos, dtype=float).copy()

    for f0 in freqs:
        b, a = iirnotch(w0=f0, Q=Q, fs=fs)
        sos = tf2sos(b, a)
        salida = np.array([_filtrar_con_padding_reflect(canal, sos, pad) for canal in salida])

    return salida


# %% FILTRO POR WAVELETS

def _filtrar_dwt(
    signal: np.ndarray,
    wavelet: str,
    level: int,
    niveles_cero: list[int] | None = None,
    remove_approx: bool = False,
    mode: str = "symmetric",
) -> np.ndarray:
    """Filtra un vector anulando coeficientes seleccionados de su DWT.
    
        Entradas:
            signal: Vector de señal.
            wavelet: Familia wavelet, por ejemplo db4.
            level: Nivel de descomposición.
            niveles_cero: Índices de coeficientes de detalle que se anularán.
            remove_approx: Si es True, también anula la aproximación.
            mode: Modo de extensión usado por PyWavelets.
    
        Retorna:
            Vector reconstruido con la longitud original."""
    coeffs = pywt.wavedec(signal, wavelet, level=level, mode=mode)
    coeffs_filt = [c.copy() for c in coeffs]

    if remove_approx:
        coeffs_filt[0] = np.zeros_like(coeffs_filt[0])

    if niveles_cero is not None:
        for j in niveles_cero:
            if 1 <= j <= level:
                coeffs_filt[j] = np.zeros_like(coeffs_filt[j])

    y = pywt.waverec(coeffs_filt, wavelet, mode=mode)
    return y[: len(signal)]


def _filtrar_dwt_con_padding_reflect(
    x: np.ndarray,
    wavelet: str,
    level: int,
    niveles_cero: list[int] | None = None,
    remove_approx: bool = False,
    pad: int = 0,
    mode: str = "symmetric",
) -> np.ndarray:
    """Aplica el filtrado DWT con padding reflectante opcional.
    
        Entradas:
            x: Vector de señal.
            wavelet, level, niveles_cero, remove_approx, mode: Parámetros de la DWT.
            pad: Muestras añadidas en cada extremo; 0 desactiva el padding.
    
        Retorna:
            Vector filtrado con la misma longitud que x."""
    if pad is None or pad <= 0:
        return _filtrar_dwt(
            x,
            wavelet=wavelet,
            level=level,
            niveles_cero=niveles_cero,
            remove_approx=remove_approx,
            mode=mode,
        )

    x_pad = np.pad(x, pad_width=pad, mode="reflect")
    y_pad = _filtrar_dwt(
        x_pad,
        wavelet=wavelet,
        level=level,
        niveles_cero=niveles_cero,
        remove_approx=remove_approx,
        mode=mode,
    )
    return y_pad[pad:-pad]


def aplicar_wavelet_dwt_multicanal(
    datos: np.ndarray,
    wavelet: str = "db4",
    level: int = 6,
    niveles_cero: list[int] | None = None,
    remove_approx: bool = False,
    pad: int = 0,
    mode: str = "reflect",
) -> np.ndarray:
    """Aplica el filtrado DWT de forma independiente a cada canal.
    
        Entradas:
            datos: Matriz (canales, muestras).
            wavelet: Familia wavelet.
            level: Nivel de descomposición.
            niveles_cero: Coeficientes de detalle que se anularán.
            remove_approx: Indica si se elimina la aproximación.
            pad: Padding reflectante en muestras.
            mode: Modo de extensión de PyWavelets.
    
        Retorna:
            Matriz filtrada con forma (canales, muestras)."""
    datos = np.asarray(datos, dtype=float)
    return np.array(
        [
            _filtrar_dwt_con_padding_reflect(
                canal,
                wavelet=wavelet,
                level=level,
                niveles_cero=niveles_cero,
                remove_approx=remove_approx,
                pad=pad,
                mode=mode,
            )
            for canal in datos
        ]
    )


# %% ICA

def ica_eliminar_componente(
    datos: np.ndarray,
    fs: float,
    ch_names: list[str],
    ic_to_remove: int | None = 0,
    montage: str | None = "standard_1020",
    random_state: int = 97,
) -> tuple[np.ndarray, ICA]:
    """Ajusta ICA con MNE y elimina un componente independiente.
    
        Entradas:
            datos: Matriz EEG (canales, muestras).
            fs: Frecuencia de muestreo en Hz.
            ch_names: Nombres de los canales en el mismo orden que datos.
            ic_to_remove: Índice del componente a eliminar; None no excluye componentes.
            montage: Nombre del montaje estándar de MNE, o None.
            random_state: Semilla para reproducibilidad.
    
        Retorna:
            Tupla (datos_limpios, ica), con la matriz procesada y el objeto ICA ajustado.
    
        Nota:
            El componente debe seleccionarse tras inspeccionar sus mapas y series temporales."""
    datos = np.asarray(datos, dtype=float)

    if datos.shape[0] != len(ch_names):
        raise ValueError(
            f"ch_names tiene {len(ch_names)} nombres, pero datos tiene {datos.shape[0]} canales." 
        )

    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=["eeg"] * len(ch_names))
    raw = mne.io.RawArray(datos, info, verbose=False)

    if montage:
        try:
            mont = mne.channels.make_standard_montage(montage)
            raw.set_montage(mont, match_case=False)
        except Exception:
            pass

    rank = mne.compute_rank(raw, verbose=False)
    n_comp = min(rank.get("eeg", len(ch_names)), len(ch_names))

    ica = ICA(
        n_components=n_comp,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=random_state,
        max_iter="auto",
    )
    ica.fit(raw)

    if ic_to_remove is None:
        ica.exclude = []
    else:
        ica.exclude = [int(ic_to_remove)]

    raw_clean = raw.copy()
    ica.apply(raw_clean)

    return raw_clean.get_data(), ica


# %% PASA BANDA

def aplicar_pasabanda_butter(
    datos: np.ndarray,
    fs: float,
    f_low: float,
    f_high: float,
    orden: int = 4,
    pad: int = 0,
) -> np.ndarray:

    """Aplica un filtro pasabanda Butterworth de fase cero.
    
        Entradas:
            datos: Matriz (canales, muestras).
            fs: Frecuencia de muestreo en Hz.
            f_low: Frecuencia de corte inferior en Hz.
            f_high: Frecuencia de corte superior en Hz.
            orden: Orden del filtro.
            pad: Padding reflectante en muestras.
    
        Retorna:
            Matriz filtrada con la misma forma que datos.
    
        Excepciones:
            ValueError: Si las frecuencias u orden no son válidos."""
    datos = np.asarray(datos, dtype=float)

    if f_low <= 0:
        raise ValueError("f_low debe ser > 0")
    if f_high >= fs / 2:
        raise ValueError(f"f_high debe ser < fs/2 = {fs/2:.2f} Hz")
    if f_low >= f_high:
        raise ValueError("Se requiere f_low < f_high")
    if orden < 1:
        raise ValueError("El orden debe ser >= 1")

    sos = butter(
        N=orden,
        Wn=[f_low, f_high],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    salida = np.array([_filtrar_con_padding_reflect(canal, sos, pad) for canal in datos])
    return salida


# %% RESAMPLEAR POR PROMEDIO

def resamplear_por_promedio(senal, factor=4):
    """Reduce la cantidad de muestras promediando bloques consecutivos.
    
        Entradas:
            senal: Matriz (canales, muestras).
            factor: Número de muestras originales por cada muestra de salida.
    
        Retorna:
            Matriz (canales, muestras_reducidas). Las muestras sobrantes se descartan."""
    senal = np.asarray(senal)
    longitud_truncada = (senal.shape[1] // factor) * factor
    senal_resampleada = (
        senal[:, :longitud_truncada]
        .reshape(senal.shape[0], -1, factor)
        .mean(axis=2)
    )

    return senal_resampleada


# %% RESAMPLEAR POR FILTRO CHIDO

def resamplear_profesional(senal, fs_original, fs_nueva):
    """Remuestrea una señal mediante filtrado polifásico.
    
        Entradas:
            senal: Matriz (canales, muestras).
            fs_original: Frecuencia de muestreo original en Hz.
            fs_nueva: Frecuencia de muestreo deseada en Hz.
    
        Retorna:
            Matriz remuestreada conservando los canales en el eje 0."""
    gcd = np.gcd(int(fs_original), int(fs_nueva))
    up = int(fs_nueva // gcd)
    down = int(fs_original // gcd)

    senal_resampleada = resample_poly(
        senal,
        up=up,
        down=down,
        axis=1
    )

    return senal_resampleada

# %% RECTIFICAR

def rectificar_senal(senal, tipo='completa'):
    """Rectifica una señal completa o por media onda positiva.
    
        Entradas:
            senal: Vector o matriz de señal.
            tipo: "completa" para valor absoluto o "media" para conservar positivos.
    
        Retorna:
            Arreglo rectificado con la misma forma que senal."""
    if tipo == 'completa':
        return np.abs(senal)
        
    elif tipo == 'media':
        return np.maximum(0, senal)
        
    else:
        raise ValueError("El tipo de rectificación debe ser 'completa' o 'media'.")
        
# %%  ENVOLVENTE

def emg_envelope(x, fs=None, t=None, tc_ms=20.0, rectify="full"):
    """Obtiene la envolvente EMG mediante rectificación y promedio móvil.
    
        Entradas:
            x: Vector EMG.
            fs: Frecuencia de muestreo en Hz; puede omitirse si se proporciona t.
            t: Vector de tiempo en segundos, con la misma longitud que x.
            tc_ms: Duración de la ventana de suavizado en milisegundos.
            rectify: "full", "half" o None.
    
        Retorna:
            Vector de envolvente con la misma longitud que x."""
    x = np.asarray(x, dtype=float)

    # --- determinar fs ---
    if fs is None:
        if t is None:
            raise ValueError("Debes proporcionar fs (Hz) o t (vector de tiempo en s).")
        t = np.asarray(t, dtype=float)
        if len(t) < 2:
            return x.copy()
        dt = float(np.mean(np.diff(t)))
        if dt <= 0:
            return x.copy()
        fs = 1.0 / dt

    # --- rectificación ---
    if rectify == "half":
        x_rect = np.maximum(x, 0.0)  # como en tus scripts
    elif rectify == "full":
        x_rect = np.abs(x)
    elif rectify is None:
        x_rect = x
    else:
        raise ValueError("rectify debe ser 'half', 'full' o None.")

    # --- ventana en muestras ---
    window_samples = int((tc_ms / 1000.0) * float(fs))
    if window_samples < 3:
        return x_rect.copy()

    # forzar impar (ventana simétrica) como en tus scripts
    if window_samples % 2 == 0:
        window_samples += 1

    kernel = np.ones(window_samples, dtype=float) / window_samples
    env = np.convolve(x_rect, kernel, mode="same")  # centrado ("zero-phase" aproximado)
    return env

# %% LAPLACIANO

def filtro_laplaciano(
        datos,
        canales,
        canal_interes,
        canales_vecinos):

    """Calcula un Laplaciano espacial simple para un canal EEG.
    
        Entradas:
            datos: Matriz (canales, muestras).
            canales: Lista con los nombres de todos los canales.
            canal_interes: Canal central que se desea realzar.
            canales_vecinos: Canales usados para calcular el promedio de referencia.
    
        Retorna:
            Vector (muestras,) igual al canal de interés menos el promedio de vecinos."""
    idx_interes = canales.index(canal_interes)

    indices_vecinos = [
        canales.index(ch)
        for ch in canales_vecinos
    ]

    senal_interes = datos[idx_interes]

    promedio_vecinos = np.mean(
        datos[indices_vecinos, :],
        axis=0
    )

    senal_laplaciana = senal_interes - promedio_vecinos

    return senal_laplaciana

# %% QUITAR COMPONENTE DC

def quitar_componente_dc(datos):

    """Elimina la componente DC restando la media temporal de cada canal.
    
        Entradas:
            datos: Matriz (canales, muestras).
    
        Retorna:
            Matriz centrada en cero por canal."""
    datos = np.asarray(datos)

    datos_sin_dc = datos - np.mean(
        datos,
        axis=1,
        keepdims=True
    )

    return datos_sin_dc

# %% PANTALLA COMPLETA

def pantalla_completa(fig):

    """Intenta maximizar una figura de Matplotlib en distintos backends.
    
        Entradas:
            fig: Objeto matplotlib.figure.Figure.
    
        Retorna:
            None. Si el backend no permite maximizar, fija un tamaño aproximado."""
    try:
        manager = fig.canvas.manager

        try:
            manager.window.state('zoomed')
            return
        except:
            pass

        try:
            manager.window.showMaximized()
            return
        except:
            pass

        try:
            manager.frame.Maximize(True)
            return
        except:
            pass

        fig.set_size_inches(19, 10.5)

    except:
        fig.set_size_inches(19, 10.5)
        
# %% INSTAR

def hardlim(x):
    """Aplica una función escalón binaria elemento a elemento.
    
        Entradas:
            x: Escalar o arreglo numérico.
    
        Retorna:
            1 donde x es mayor o igual que cero y 0 en otro caso."""
    return np.where(x >= 0, 1, 0)

def entrenar_instar(*vectores, epochs=1000, alpha=0.1, bias=50):

    """Entrena una neurona Instar con vectores de igual longitud.
    
        Entradas:
            *vectores: Vectores unidimensionales de entrenamiento.
            epochs: Número de épocas.
            alpha: Tasa de aprendizaje.
            bias: Sesgo usado antes de la función hardlim.
    
        Retorna:
            Vector de pesos aprendidos.
    
        Excepciones:
            ValueError: Si algún vector no es 1D o las longitudes son distintas."""
    data = [np.asarray(v) for v in vectores]

    for v in data:
        if v.ndim != 1:
            raise ValueError("Todos los datos de entrada deben ser vectores de una dimensión.")

    longitudes = [len(v) for v in data]

    if len(set(longitudes)) != 1:
        raise ValueError("Todos los vectores deben tener la misma longitud.")

    data = np.array(data)
    num_features = data.shape[1]
    weights = np.random.rand(num_features)

    for epoch in range(epochs):
        for sample in data:

            z = np.dot(weights, sample) + bias
            a = hardlim(z)

            weights = weights + alpha * a * (sample - weights)

    return weights
