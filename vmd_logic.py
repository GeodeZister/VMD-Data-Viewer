import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Спроба імпорту PyEMD (для опціонального EMD)
try:
    from PyEMD import EMD
    HAS_EMD = True
except ImportError:
    HAS_EMD = False


# 1. VMD
def vmd(
    signal,
    alpha=2000,
    tau=0,
    K=3,
    DC=0,
    init=1,
    tol=1e-7,
    max_iter=500
):
    """
    Покращена реалізація Variational Mode Decomposition (VMD) з віддзеркаленням.
    Повертає масив розмірності (K, N) із K модами.
    """

    N = len(signal)
    # Якщо непарна довжина, приберемо останню точку
    if N % 2 == 1:
        signal = signal[:-1]
        N = len(signal)

    # Віддзеркалення
    half = N // 2
    signal_mirror = np.concatenate([
        np.flip(signal[:half]),
        signal,
        np.flip(signal[-half:])
    ])
    N_ext = len(signal_mirror)

    f_signal = np.fft.fft(signal_mirror)
    freqs = np.fft.fftfreq(N_ext, d=1.0 / N_ext)

    # Ініціалізація центрів частот omega_k
    if init == 1:
        omega_k = np.linspace(0, 0.5, K, endpoint=False) * N_ext
    elif init == 2:
        omega_k = np.sort(np.random.uniform(0, 0.5, K)) * N_ext
    else:
        omega_k = np.zeros(K)

    if DC == 1:
        omega_k[0] = 0.0

    u_hat = np.zeros((K, N_ext), dtype=np.complex128)
    lambda_hat = np.zeros(N_ext, dtype=np.complex128)

    for _ in range(max_iter):
        u_hat_old = u_hat.copy()

        for k in range(K):
            # Сума усіх інших IMF
            sum_others = np.sum(u_hat[[i for i in range(K) if i != k]], axis=0)
            residue = f_signal - sum_others - lambda_hat / 2.0

            # Модулюємо залишок
            mod_factor = np.exp(-1j * 2.0 * np.pi * freqs * (omega_k[k] / N_ext))
            modulated = residue * mod_factor

            # Оновлення центру частоти (окрім DC=1 для k=0)
            if not (DC == 1 and k == 0):
                numerator = np.sum(freqs * np.abs(modulated)**2)
                denominator = np.sum(np.abs(modulated)**2) + 1e-14
                new_omega = numerator / denominator
                omega_k[k] = new_omega * N_ext
            else:
                omega_k[k] = 0.0

            # Оновлення u_hat[k]
            u_hat[k, :] = modulated / (
                1.0 + 2.0 * alpha * (freqs - (omega_k[k] / N_ext))**2
            )

        # Оновлення Lagrange-множника
        sum_u_hat = np.sum(u_hat, axis=0)
        lambda_hat = lambda_hat + tau * (sum_u_hat - f_signal)

        diff = np.sum(np.abs(u_hat - u_hat_old)**2)
        if diff < tol:
            break

    # ifft і зрізання
    u_modes_ext = np.fft.ifft(u_hat, axis=1).real
    start = half
    end = half + N
    u_modes = u_modes_ext[:, start:end]

    return u_modes


# 2. EMD (опціонально)
def emd_decompose(signal):
    """
    Виконує EMD (якщо доступний PyEMD). Повертає (K, N) або None, якщо PyEMD недоступний.
    """
    if not HAS_EMD:
        return None
    emd_obj = EMD()
    imfs = emd_obj.emd(signal)
    return imfs


# 3. Обробка даних
def remove_outliers_zscore(series, threshold=3.0):
    mean_val = series.mean()
    std_val = series.std()
    if std_val == 0:
        return series
    zscore = (series - mean_val) / std_val
    outliers_mask = abs(zscore) > threshold
    series[outliers_mask] = np.nan
    return series


def interpolate_time(df, time_col="Datetime", method="time"):
    df = df.sort_values(time_col)
    df = df.set_index(time_col)
    df = df.interpolate(method=method)
    df = df.reset_index()
    return df


def normalize_series(series, method="zscore"):
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


# 4. Візуалізація
def plot_vmd_subplots(
    dates,
    original_signal,
    imfs,
    title="VMD Decomposition",
    figsize=(12, 8),
    linewidth=1.2,
    original_color="black",
    imf_colors=None,
    ylabels=None,
    date_format="%Y-%m-%d"
):
    K = imfs.shape[0]
    fig, axes = plt.subplots(nrows=K + 1, ncols=1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=14)

    # Оригінальний
    axes[0].plot(dates, original_signal, color=original_color,
                 linewidth=linewidth, label="Original")
    axes[0].set_ylabel("Original", fontsize=12)
    axes[0].grid(True)
    axes[0].legend(loc="upper right")

    # Автоматичні кольори
    if imf_colors is None:
        imf_colors = [f"C{i}" for i in range(K)]

    # Автоматичні ylabels
    if not ylabels:
        ylabels = [f"IMF {i+1}" for i in range(K)]

    for i in range(K):
        axes[i + 1].plot(dates, imfs[i], label=ylabels[i],
                         color=imf_colors[i], linewidth=linewidth)
        axes[i + 1].set_ylabel(ylabels[i], fontsize=12)
        axes[i + 1].grid(True)
        axes[i + 1].legend(loc="upper right")

    # Формат осі X для дати
    if len(dates) > 0 and np.issubdtype(type(dates[0]), np.datetime64):
        import matplotlib.dates as mdates
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        fig.autofmt_xdate()

    axes[-1].set_xlabel("Date/Time", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
