import numpy as np
import pywt

import mne
from mne.preprocessing import ICA

from scipy.signal import iirnotch, sosfiltfilt, tf2sos, butter


# %% CARGAR DATOS

def cargar_npy_eeg(ruta_npy: str, n_canales_esperados: int | None = None) -> np.ndarray:
    datos = np.load(ruta_npy)
    if datos.ndim != 2:
        raise ValueError(f"Se esperaba un arreglo 2D (canales x muestras). Recibido: {datos.shape}")
    if n_canales_esperados is not None and datos.shape[0] != n_canales_esperados:
        raise ValueError(
            f"Se esperaban {n_canales_esperados} canales, pero el archivo trae {datos.shape[0]} canales: {datos.shape}"
        )
    return np.asarray(datos, dtype=float)


# %% FILTRADO DE NOTCH

def _filtrar_con_padding_reflect(x: np.ndarray, sos: np.ndarray, pad: int) -> np.ndarray:
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