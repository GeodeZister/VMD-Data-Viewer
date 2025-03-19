import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import matplotlib.dates as mdates

# Спробуємо імпортувати PyEMD (для порівняння з VMD)
try:
    from PyEMD import EMD

    HAS_EMD = True
except ImportError:
    HAS_EMD = False
    # Якщо PyEMD не встановлено, EMD недоступний. Використовуйте: pip install EMD-signal


# ------------------------------------------------------
# 1. Покращена реалізація VMD з віддзеркаленням сигналу
# ------------------------------------------------------
def vmd(signal, alpha=2000, tau=0, K=3, DC=0, init=1, tol=1e-7, max_iter=500):
    """
    Покращена реалізація Variational Mode Decomposition (VMD) з віддзеркаленням сигналу.
    Повертає масив розмірності (K, N) із K модами.

    Параметри:
      - signal: вхідний сигнал (1D numpy array)
      - alpha: параметр балансування за даними (data-fidelity)
      - tau: крок у подвійному підйомі (dual ascent step)
      - K: кількість режимів
      - DC: 1, якщо потрібно зафіксувати перший режим на DC (нулева частота)
      - init: 0 - всі частоти нульові, 1 - рівномірно розподілені, 2 - випадкова ініціалізація
      - tol: толерантність зупинки
      - max_iter: максимальна кількість ітерацій
    """
    # Переконаємось, що довжина сигналу парна для віддзеркалення
    N = len(signal)
    if N % 2:
        signal = signal[:-1]
        N = len(signal)

    # Віддзеркалення сигналу
    half = N // 2
    signal_mirror = np.concatenate([np.flip(signal[:half]), signal, np.flip(signal[-half:])])
    N_ext = len(signal_mirror)

    # Спектральне представлення (FFT) розширеного сигналу
    freqs = np.fft.fftfreq(N_ext, d=1.0 / N_ext)
    f_signal = np.fft.fft(signal_mirror)

    # Ініціалізація центрів частот omega_k
    if init == 1:
        # Рівномірно від 0 до 0.5 (нормована частота), масштабовано на N_ext
        omega_k = np.linspace(0, 0.5, K, endpoint=False) * N_ext
    elif init == 2:
        # Випадкова ініціалізація від 0 до 0.5 (масштабовано)
        omega_k = np.sort(np.random.uniform(0, 0.5, K)) * N_ext
    else:
        omega_k = np.zeros(K)

    # Якщо DC режим вимкнений для першої моди
    if DC:
        omega_k[0] = 0

    # Ініціалізація спектральних компонент та Lagrange-множника
    u_hat = np.zeros((K, N_ext), dtype=np.complex128)
    lambda_hat = np.zeros(N_ext, dtype=np.complex128)

    # Головний цикл VMD
    for it in range(max_iter):
        u_hat_old = u_hat.copy()
        for k in range(K):
            # Обчислюємо суму інших режимів
            idx_other = [i for i in range(K) if i != k]
            sum_others = np.sum(u_hat[idx_other, :], axis=0)
            residue = f_signal - sum_others - lambda_hat / 2.0

            # Модульне множення для корекції фази
            mod_factor = np.exp(-1j * 2 * np.pi * freqs * (omega_k[k] / N_ext))
            # Оновлення omega_k через обчислення спектрального центру
            modulated = residue * mod_factor
            numerator = np.sum(freqs * np.abs(modulated) ** 2)
            denominator = np.sum(np.abs(modulated) ** 2) + 1e-12  # для уникнення ділення на 0
            # Якщо режим не закріплений на DC, оновлюємо його центр
            if not (DC and k == 0):
                omega_k[k] = alpha * numerator / denominator
            else:
                omega_k[k] = 0

            # Оновлення спектрального представлення u_hat[k]
            u_hat[k, :] = residue * mod_factor / (1.0 + 2.0 * alpha * (freqs - omega_k[k] / N_ext) ** 2)

        # Оновлення Lagrange-множника
        lambda_hat += tau * (np.sum(u_hat, axis=0) - f_signal)

        # Перевірка зупинки за критерієм зміни режимів
        diff = np.sum(np.abs(u_hat - u_hat_old) ** 2)
        if diff < tol:
            break

    # Обчислення режимів за допомогою зворотного FFT
    u_modes_ext = np.fft.ifft(u_hat, axis=1).real
    # Обрізаємо віддзеркалену частину: беремо центральну частину довжиною N
    start = half
    end = half + N
    u_modes = u_modes_ext[:, start:end]

    return u_modes


# ------------------------------------------------------
# 2. Рекомендовані параметри VMD
# ------------------------------------------------------
def recommend_vmd_params(signal):
    """
    Евристики для рекомендації параметрів VMD.
    """
    N = len(signal)
    data_range = np.max(signal) - np.min(signal)

    # K
    if N < 2000:
        K_rec = 3
    elif N < 10000:
        K_rec = 4
    else:
        K_rec = 6

    # alpha
    alpha_rec = 2000 if data_range > 1 else 1000

    # tol
    tol_rec = 1e-7

    # max_iter
    max_iter_rec = 1000 if N > 50000 else 500

    # DC
    mean_val = np.mean(signal)
    dc_rec = 1 if abs(mean_val) > 0.5 * data_range else 0

    return {
        "K": K_rec,
        "alpha": alpha_rec,
        "tol": tol_rec,
        "max_iter": max_iter_rec,
        "DC": dc_rec,
    }


# ------------------------------------------------------
# 3. Обробка аномалій та інтерполяція
# ------------------------------------------------------
def remove_outliers_zscore(series, threshold=3.0):
    """
    Замінює значення, що перевищують поріг z-score, на NaN.
    """
    mean_val = series.mean()
    std_val = series.std()
    if std_val == 0:
        return series
    zscore = (series - mean_val) / std_val
    outliers_mask = zscore.abs() > threshold
    series[outliers_mask] = np.nan
    return series


def interpolate_time(df, time_col="Datetime", method="linear"):
    """
    Інтерполяція NaN за часовою колонкою 'Datetime'.
    """
    df = df.sort_values(time_col)
    df_interp = df.interpolate(method=method)
    return df_interp


def normalize_series(series, method="zscore"):
    """
    Нормалізація даних:
      - zscore: (x - mean) / std
      - minmax: (x - min) / (max - min)
    """
    if method == "zscore":
        std_val = series.std()
        return (series - series.mean()) / (std_val if std_val != 0 else 1)
    elif method == "minmax":
        return (series - series.min()) / (series.max() - series.min() + 1e-12)
    else:
        return series


# ------------------------------------------------------
# 4. Побудова багаторядного графіка (Original + IMF)
# ------------------------------------------------------
def plot_vmd_subplots(dates, original_signal, imfs, title="VMD Decomposition"):
    """
    Створює графік з оригінальним сигналом і кожною IMF на окремих підграфіках.
    """
    K = imfs.shape[0]
    fig, axes = plt.subplots(nrows=K + 1, ncols=1, figsize=(10, 8), sharex=True)
    fig.suptitle(title)

    # Оригінал
    axes[0].plot(dates, original_signal, color="C0", label="Original")
    axes[0].set_ylabel("Original")
    axes[0].grid(True)
    axes[0].legend(loc="upper right")

    # Режими (IMF)
    for i in range(K):
        axes[i + 1].plot(dates, imfs[i], label=f"IMF {i + 1}", color=f"C{i + 1}")
        axes[i + 1].set_ylabel(f"IMF {i + 1}")
        axes[i + 1].grid(True)
        axes[i + 1].legend(loc="upper right")

    # Форматування дат, якщо dates є datetime
    if len(dates) > 0 and (np.issubdtype(type(dates[0]), np.datetime64) or "datetime" in str(type(dates[0])).lower()):
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ------------------------------------------------------
# 5. Головна функція Streamlit
# ------------------------------------------------------
def main():
    st.title("Покращений додаток VMD/EMD із багаторядним графіком")

    st.write("""
    **Крок 1**: Завантажте CSV-файл із вашим часовим рядом.  
    Файл повинен містити:
    - Колонку `"Point"` (ім'я точки),
    - Колонку з датою/часом `"Datetime"` (опційно),
    - Числові колонки (наприклад, Height, North, East тощо).
    """)

    uploaded_file = st.file_uploader("Оберіть CSV-файл", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Перетворення та сортування за "Datetime", якщо є
        if "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
            df.sort_values("Datetime", inplace=True)

        # Перевірка наявності колонки "Point"
        if "Point" not in df.columns:
            st.error("Не знайдено стовпець 'Point' у даних!")
            st.stop()

        unique_points = df["Point"].unique()
        selected_point = st.selectbox("Оберіть точку (Point)", unique_points)
        df_selected = df[df["Point"] == selected_point].copy()

        # Вибір числових колонок (крім "Point")
        numeric_cols = df_selected.select_dtypes(include=[np.number]).columns.tolist()
        if "Point" in numeric_cols:
            numeric_cols.remove("Point")
        if not numeric_cols:
            st.error("Немає числових колонок для аналізу у вибраній точці.")
            st.stop()
        col_name = st.selectbox("Оберіть параметр (числова колонка):", numeric_cols)

        st.subheader("Налаштування попередньої обробки (Preprocessing)")
        remove_outliers = st.checkbox("Прибрати аномалії (z-score > 3)", value=False)
        interpolate_data = False
        if "Datetime" in df.columns:
            interpolate_data = st.checkbox("Інтерполювати пропуски за часом", value=False)
        norm_method = st.selectbox("Нормалізація (опційно)", ["none", "zscore", "minmax"])

        st.subheader("Вибір методу декомпозиції")
        method = st.radio("Метод", ["VMD", "EMD (якщо доступно)"])
        subtract_mean = st.checkbox("Віднімати середнє перед декомпозицією?", value=True)

        st.subheader("Рекомендовані параметри (лише для VMD)")
        if st.button("Отримати рекомендовані параметри"):
            sig_temp = df_selected[col_name].astype(float)
            rec = recommend_vmd_params(sig_temp)
            st.write("**Рекомендовані значення:**")
            st.write(rec)
            st.info("Ви можете вручну застосувати ці параметри нижче, якщо обрано VMD.")

        st.subheader("Налаштування VMD")
        K = st.slider("K (кількість мод)", 1, 10, 3, 1)
        alpha = st.slider("alpha", 100, 5000, 2000, 100)
        tol_exp = st.slider("log10(tol)", -12, -1, -7, 1)
        tol_val = 10.0 ** tol_exp
        max_iter = st.slider("max_iter", 100, 2000, 500, 100)
        DC_bool = st.checkbox("DC=1 (виділити сталу складову)", value=False)
        DC_param = 1 if DC_bool else 0

        if st.button("Запустити декомпозицію"):
            signal = df_selected[col_name].astype(float).copy()

            # 1) Видалення аномалій
            if remove_outliers:
                signal = remove_outliers_zscore(signal, threshold=3.0)
            df_selected[col_name] = signal  # записуємо NaN в оригінальний DataFrame

            # 2) Інтерполяція пропусків, якщо обрано та є "Datetime"
            if interpolate_data and "Datetime" in df_selected.columns:
                df_selected = df_selected.sort_values("Datetime")
                df_selected = df_selected.interpolate(method="linear")

            # Оновлюємо сигнал після попередньої обробки
            signal = df_selected[col_name].astype(float).values

            # 3) Нормалізація
            if norm_method != "none":
                signal = normalize_series(pd.Series(signal), method=norm_method).values

            # Відображення сигналу після попередньої обробки
            st.subheader("Сигнал після попередньої обробки")
            fig_prep, ax_prep = plt.subplots()
            ax_prep.plot(signal, label="Signal (preprocessed)")
            ax_prep.legend()
            st.pyplot(fig_prep)

            # Виконання декомпозиції
            if method == "VMD":
                if subtract_mean:
                    mean_val = np.mean(signal)
                    sig_centered = signal - mean_val
                    imfs = vmd(
                        sig_centered,
                        alpha=alpha,
                        tau=0,
                        K=K,
                        DC=DC_param,
                        init=1,
                        tol=tol_val,
                        max_iter=max_iter
                    )
                    # Розподіляємо середнє між режимами
                    imfs += mean_val / K
                else:
                    imfs = vmd(
                        signal,
                        alpha=alpha,
                        tau=0,
                        K=K,
                        DC=DC_param,
                        init=1,
                        tol=tol_val,
                        max_iter=max_iter
                    )
            else:  # EMD
                if not HAS_EMD:
                    st.warning("EMD недоступний. Встановіть PyEMD (pip install EMD-signal).")
                    st.stop()
                if subtract_mean:
                    mean_val = np.mean(signal)
                    sig_centered = signal - mean_val
                    imfs = EMD().emd(sig_centered)
                else:
                    imfs = EMD().emd(signal)
                K = imfs.shape[0]

            st.subheader(f"Результати {method} (K={K})")
            # Використовуємо "Datetime", якщо є, або індекс
            if "Datetime" in df_selected.columns and df_selected["Datetime"].notna().any():
                dates = df_selected["Datetime"].values
            else:
                dates = np.arange(len(signal))

            fig_subplots = plot_vmd_subplots(
                dates=dates,
                original_signal=signal,
                imfs=imfs,
                title=f"{method} decomposition for point '{selected_point}'"
            )
            st.pyplot(fig_subplots)
            st.success("Декомпозицію виконано!")

            # Збереження результатів
            st.subheader("Зберегти IMF у CSV?")
            if st.button("Зберегти IMFs"):
                df_imfs = pd.DataFrame(imfs.T, columns=[f"IMF_{i + 1}" for i in range(imfs.shape[0])])
                df_imfs.insert(0, "Index", np.arange(len(signal)))
                csv_buffer = BytesIO()
                df_imfs.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Завантажити IMF.csv",
                    data=csv_buffer.getvalue(),
                    file_name="imfs_result.csv",
                    mime="text/csv"
                )


if __name__ == "__main__":
    main()
