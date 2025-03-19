import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from vmdpy import VMD

# Опціональний імпорт PyEMD для EMD
try:
    from PyEMD import EMD
    HAS_EMD = True
except ImportError:
    HAS_EMD = False


# 1. VMD з бібліотеки vmdpy
def vmd(signal, alpha=2000, tau=0, K=3, DC=0, init=1, tol=1e-7, max_iter=500):
    """
    Виконує Variational Mode Decomposition (VMD) із використанням бібліотеки vmdpy.
    Повертає масив мод (K, N), де N відповідає довжині вхідного сигналу.
    """
    original_length = len(signal)
    # Якщо довжина непарна, обрізаємо останній елемент
    if original_length % 2 == 1:
        signal = signal[:-1]
    # Виконуємо VMD через vmdpy
    u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)

    if u.shape[1] < original_length:
        pad_width = original_length - u.shape[1]
        u = np.pad(u, ((0, 0), (0, pad_width)), mode='edge')
    # Якщо довжина мод більша за вхідний сигнал, обрізаємо
    elif u.shape[1] > original_length:
        u = u[:, :original_length]

    return u


# 2. EMD (опціонально)
def emd_decompose(signal):
    """Виконує EMD (якщо доступний PyEMD)."""
    if not HAS_EMD:
        return None
    emd_obj = EMD()
    imfs = emd_obj.emd(signal)
    return imfs


# 3. Обробка даних
def remove_outliers_zscore(series, threshold=3.0):
    """Видалення аномалій за Z-score."""
    mean_val = series.mean()
    std_val = series.std()
    if std_val == 0:
        return series
    zscore = (series - mean_val) / std_val
    outliers_mask = abs(zscore) > threshold
    series[outliers_mask] = np.nan
    return series


def interpolate_time(df, time_col="Datetime", method="time"):
    """Інтерполяція часових рядів."""
    df = df.sort_values(time_col)
    df = df.set_index(time_col)
    df = df.interpolate(method=method)
    df = df.reset_index()
    return df


def normalize_series(series, method="zscore"):
    """Нормалізація часового ряду."""
    if method == "zscore":
        mean_val = series.mean()
        std_val = series.std()
        if std_val == 0:
            return series - mean_val
        else:
            return (series - mean_val) / std_val
    elif method == "minmax":
        min_val = series.min()
        max_val = series.max()
        rng = max_val - min_val
        if rng == 0:
            return series - min_val
        else:
            return (series - min_val) / rng
    else:
        return series


# Додаткові функції для аналізу IMF

def compute_imf_correlations(imfs: np.ndarray) -> np.ndarray:
    """
    Обчислює матрицю кореляції (K x K) між IMF.
    """
    K = imfs.shape[0]
    corr_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            corr_matrix[i, j] = np.corrcoef(imfs[i], imfs[j])[0, 1]
    return corr_matrix


def imf_energy(imf: np.ndarray) -> float:
    """Обчислює енергію IMF."""
    return np.sum(imf ** 2)


def plot_imf_spectra(imfs: np.ndarray, sampling_rate: float = 1.0):
    """
    Побудова спектрів (FFT) для кожної IMF.
    """
    from scipy.fft import fft, fftfreq
    K, N = imfs.shape
    freqs = fftfreq(N, d=1.0 / sampling_rate)[: N // 2]

    fig, axes = plt.subplots(K, 1, figsize=(10, 2 * K), sharex=True)
    for i in range(K):
        yf = fft(imfs[i])
        axes[i].plot(freqs, np.abs(yf[: N // 2]))
        axes[i].set_title(f"IMF {i+1} Spectrum")
        axes[i].grid(True)
    axes[-1].set_xlabel("Frequency")
    plt.tight_layout()
    return fig


# 4. Візуалізація
def plot_vmd_subplots(
    dates, original_signal, imfs, title="VMD Decomposition",
    figsize=(12, 8), linewidth=1.2, original_color="black",
    imf_colors=None, ylabels=None, date_format="%Y-%m-%d"
):
    """
    Побудова графіків для IMF після VMD.
    """
    K = imfs.shape[0]
    fig, axes = plt.subplots(nrows=K + 1, ncols=1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=14)

    # Оригінальний сигнал
    axes[0].plot(dates, original_signal, color=original_color,
                 linewidth=linewidth, label="Original")
    axes[0].set_ylabel("Original", fontsize=12)
    axes[0].grid(True)
    axes[0].legend(loc="upper right")

    # Автоматичні кольори та підписи
    if imf_colors is None:
        imf_colors = [f"C{i}" for i in range(K)]
    if not ylabels:
        ylabels = [f"IMF {i+1}" for i in range(K)]

    for i in range(K):
        axes[i + 1].plot(dates, imfs[i], label=ylabels[i],
                         color=imf_colors[i], linewidth=linewidth)
        axes[i + 1].set_ylabel(ylabels[i], fontsize=12)
        axes[i + 1].grid(True)
        axes[i + 1].legend(loc="upper right")

    if len(dates) > 0 and np.issubdtype(type(dates[0]), np.datetime64):
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        fig.autofmt_xdate()

    axes[-1].set_xlabel("Date/Time", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
