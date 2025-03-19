import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Імпортуємо все з файлу “vmd_logic2.py”
from vmd_logic_main import (
    vmd,
    emd_decompose,
    remove_outliers_zscore,
    interpolate_time,
    normalize_series,
    plot_vmd_subplots,
    HAS_EMD
)
from scipy.fft import fft, fftfreq


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


def main():
    st.title("VMD/EMD для часових рядів GNSS, TPS")

    # 1. Завантаження CSV
    uploaded_file = st.file_uploader(
        "Завантажте CSV-файл (Point,Base,Datetime,North,East,Height)", type=["csv"]
    )
    if not uploaded_file:
        st.stop()

    df = pd.read_csv(uploaded_file)
    st.write("**Загальний розмір датафрейму**:", df.shape)

    # 2. Перетворення колонки Datetime
    if "Datetime" not in df.columns:
        st.error("У файлі немає стовпця 'Datetime'.")
        st.stop()

    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    if df["Datetime"].isna().all():
        st.warning("Стовпець Datetime не вдалося перетворити на дійсні дати/часи.")

    # 3. Вибір станції (Base)
    if "Base" not in df.columns:
        st.error("Немає стовпця 'Base' у файлі!")
        st.stop()
    unique_bases = df["Base"].unique()
    selected_base = st.selectbox("Оберіть станцію (Base):", unique_bases)
    df_base = df[df["Base"] == selected_base].copy()
    st.write("**Розмір для обраної станції**:", df_base.shape)

    # 4. Вибір точки (Point)
    if "Point" not in df.columns:
        st.error("Немає стовпця 'Point' у файлі!")
        st.stop()
    unique_points = df_base["Point"].unique()
    selected_point = st.selectbox("Оберіть точку (Point):", unique_points)
    df_point = df_base[df_base["Point"] == selected_point].copy()
    st.write("**Розмір для обраної точки**:", df_point.shape)

    # 5. Вибір параметра (North/East/Height)
    param_options = ["North", "East", "Height"]
    available_params = [col for col in param_options if col in df_point.columns]
    if not available_params:
        st.error("Немає колонок North/East/Height у файлі!")
        st.stop()
    selected_param = st.selectbox("Оберіть параметр:", available_params)
    st.write(f"Ви обрали: **Base={selected_base}**, **Point={selected_point}**, **Param={selected_param}**")

    # 6. Вибір часових меж
    if df_point["Datetime"].notna().any():
        min_date = df_point["Datetime"].min().to_pydatetime()
        max_date = df_point["Datetime"].max().to_pydatetime()
        start_date, end_date = st.slider(
            "Виберіть часовий діапазон:",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
        )
        df_point = df_point[(df_point["Datetime"] >= start_date) & (df_point["Datetime"] <= end_date)]
        st.write("**Розмір для обраного діапазону**:", df_point.shape)
    else:
        st.warning("У вибраній точці немає валідних дат.")

    # 7. Налаштування попередньої обробки
    remove_outliers = st.checkbox("Прибрати аномалії (z-score>3)", value=False)
    interpolate_check = st.checkbox("Інтерполювати пропуски за часом", value=True)
    norm_method = st.selectbox("Метод нормалізації:", ["none", "zscore", "minmax"])
    subtract_mean = st.checkbox("Відняти середнє перед декомпозицією", value=False)

    # 8. Вибір методу декомпозиції
    method = st.radio("Метод декомпозиції:", ["VMD", "EMD (PyEMD)"])

    # Параметри VMD (якщо обрано VMD)
    if method.startswith("VMD"):
        st.subheader("Параметри VMD")
        K = st.slider("K (кількість мод)", 1, 10, 4, 1)
        alpha = st.slider("alpha (ширина частотних смуг)", 10, 5000, 2000, 10)
        tau_val = st.number_input("tau (dual ascent step)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        tol_exp = st.slider("log10(tol)", -12, -1, -7, 1)
        tol_val = 10.0 ** tol_exp
        max_iter = st.slider("max_iter", 100, 5000, 1000, 100)
        DC_param = st.checkbox("DC=1 (перша мода - константа)", value=False)

    # --- Запуск декомпозиції ---
    if st.button("Запустити декомпозицію"):
        data_series = df_point[selected_param].astype(float).copy()

        # 1) Видалення аномалій
        if remove_outliers:
            data_series = remove_outliers_zscore(data_series, threshold=3.0)

        # 2) Інтерполяція
        if interpolate_check:
            df_point[selected_param] = data_series
            df_point = interpolate_time(df_point, time_col="Datetime", method="time")
            data_series = df_point[selected_param].astype(float)

        # 3) Перевірка даних
        if len(data_series) == 0:
            st.error("Немає даних у вибраному діапазоні!")
            st.stop()

        # 4) Нормалізація
        if norm_method != "none":
            data_series = normalize_series(data_series, method=norm_method)

        # 5) Віднімання середнього (опціонально)
        if subtract_mean:
            mean_val = np.mean(data_series)
            signal = data_series - mean_val
        else:
            signal = data_series.copy()

        signal = signal.values if hasattr(signal, "values") else signal

        # 6) Декомпозиція
        if method.startswith("VMD"):
            imfs = vmd(
                signal,
                alpha=alpha,
                tau=tau_val,
                K=K,
                DC=int(DC_param),
                init=1,
                tol=tol_val,
                max_iter=max_iter,
            )
        else:
            if not HAS_EMD:
                st.warning("EMD недоступний: встановіть PyEMD ('pip install EMD-signal').")
                st.stop()
            imfs_emd = emd_decompose(signal)
            if imfs_emd is None:
                st.warning("EMD недоступний (PyEMD не встановлено).")
                st.stop()
            imfs = imfs_emd
            K = imfs.shape[0]

        # 7) Перевірка якості відновлення
        sum_imfs = np.sum(imfs, axis=0)
        rec_error = np.linalg.norm(signal - sum_imfs) / (np.linalg.norm(signal) + 1e-14)

        # 8) Візуалізація декомпозиції
        dates = df_point["Datetime"].values
        fig_result = plot_vmd_subplots(
            dates=dates,
            original_signal=signal,
            imfs=imfs,
            title=f"{method} - Base={selected_base}, Point={selected_point}, Param={selected_param}"
        )
        st.pyplot(fig_result)
        st.write(f"**Relative Reconstruction Error**: {rec_error:.6f}")

        # 9) Аналіз IMF: Кореляції та Енергія
        corr_matrix = compute_imf_correlations(imfs)
        st.write("### Матриця кореляції між IMF")
        st.dataframe(pd.DataFrame(corr_matrix, columns=[f"IMF_{i+1}" for i in range(K)],
                                  index=[f"IMF_{i+1}" for i in range(K)]))

        energies = [imf_energy(imfs[i]) for i in range(K)]
        st.write("### Енергія IMF")
        for i, e in enumerate(energies):
            st.write(f"IMF {i+1}: енергія = {e:.6f}")

        # 10) Побудова спектрів IMF (FFT)
        st.write("### Спектри IMF (FFT)")
        fig_fft = plot_imf_spectra(imfs, sampling_rate=1.0)
        st.pyplot(fig_fft)

        # 11) Збереження результатів
        if st.checkbox("Зберегти IMF у CSV"):
            df_imfs = pd.DataFrame(imfs.T, columns=[f"IMF_{i+1}" for i in range(K)])
            df_imfs.insert(0, "Datetime", dates)
            csv_data = df_imfs.to_csv(index=False)
            st.download_button("Завантажити IMF.csv", data=csv_data, file_name="IMF.csv", mime="text/csv")


if __name__ == "__main__":
    main()
