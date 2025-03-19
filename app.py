import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Імпортуємо все з файлу “vmd_logic.py”
from vmd_logic import (
    vmd,
    emd_decompose,
    remove_outliers_zscore,
    interpolate_time,
    normalize_series,
    plot_vmd_subplots,
    HAS_EMD
)


def main():
    st.title("VMD/EMD для ГНСС сигналів (з вибором станції, точки, параметра)")

    # 1. Завантаження CSV
    uploaded_file = st.file_uploader("Завантажте CSV-файл (Point,Base,Datetime,North,East,Height)", type=["csv"])
    if not uploaded_file:
        st.stop()

    df = pd.read_csv(uploaded_file)
    st.write("**Загальний розмір датафрейму**:", df.shape)

    # 2. Перетворення колонки Datetime (за умови, що вона існує)
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

    # Фільтруємо DataFrame за обраною станцією
    df_base = df[df["Base"] == selected_base].copy()
    st.write("**Розмір для обраної станції**:", df_base.shape)

    # 4. Вибір точки (Point)
    if "Point" not in df.columns:
        st.error("Немає стовпця 'Point' у файлі!")
        st.stop()

    unique_points = df_base["Point"].unique()
    selected_point = st.selectbox("Оберіть точку (Point):", unique_points)

    # Фільтруємо далі
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

    # 6. Вибір часових меж (range slider)
    if df_point["Datetime"].notna().any():
        # Перетворюємо мінімальну та максимальну дату на стандартний datetime
        min_date = df_point["Datetime"].min().to_pydatetime()
        max_date = df_point["Datetime"].max().to_pydatetime()

        start_date, end_date = st.slider(
            "Виберіть часовий діапазон:",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD"
        )
        # Фільтруємо за обраним діапазоном
        df_point = df_point[(df_point["Datetime"] >= start_date) & (df_point["Datetime"] <= end_date)]
        st.write("**Розмір для обраного діапазону**:", df_point.shape)
    else:
        st.warning("У вибраній точці немає валідних дат.")

    # 7. Налаштування попередньої обробки
    remove_outliers = st.checkbox("Прибрати аномалії (z-score>3)?", value=False)
    interpolate_check = st.checkbox("Інтерполювати пропуски за часом?", value=True)
    norm_method = st.selectbox("Метод нормалізації:", ["none", "zscore", "minmax"])
    subtract_mean = st.checkbox("Відняти середнє?", value=True)
    add_mean_back = st.checkbox("Додати середнє назад у IMF?", value=False)

    # 8. Вибір методу декомпозиції
    method = st.radio("Метод декомпозиції:", ["VMD", "EMD (за наявності PyEMD)"])

    # 9. Параметри VMD (актуальні, якщо VMD)
    st.subheader("Параметри VMD")
    K = st.slider("K (кількість мод)", 1, 10, 3, 1)
    alpha = st.slider("alpha (впливає на вузькосмуговість)", 10, 5000, 2000, 10)
    tau_val = st.number_input("tau (dual ascent step)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    tol_exp = st.slider("log10(tol)", -12, -1, -7, 1)
    tol_val = 10.0 ** tol_exp
    max_iter = st.slider("max_iter", 100, 5000, 500, 100)
    DC_bool = st.checkbox("DC=1 (перша мода - константа)?", value=False)
    DC_param = 1 if DC_bool else 0

    # --- Показ вибраних налаштувань ---
    st.write("### Обрані налаштування:")
    st.write(f"- **Метод декомпозиції**: {method}")
    st.write(f"- **Застосовані кроки**:")
    st.write(f"  - Прибрати аномалії: {remove_outliers}")
    st.write(f"  - Інтерполювати час: {interpolate_check}")
    st.write(f"  - Нормалізація: {norm_method}")
    st.write(f"  - Відняти середнє: {subtract_mean}")
    st.write(f"  - Додати середнє назад: {add_mean_back}")

    if method.startswith("VMD"):
        st.write(f"- **Параметри VMD**:")
        st.write(f"  - K = {K}")
        st.write(f"  - alpha = {alpha}")
        st.write(f"  - tau = {tau_val}")
        st.write(f"  - DC = {DC_param}")
        st.write(f"  - tol = {tol_val}")
        st.write(f"  - max_iter = {max_iter}")

    # 10. Кнопка запуску
    if st.button("Запустити декомпозицію"):
        # Копіюємо дані з відфільтрованого DataFrame
        data_series = df_point[selected_param].astype(float).copy()

        # 1) Прибрати аномалії
        if remove_outliers:
            data_series = remove_outliers_zscore(data_series, threshold=3.0)

        # 2) Інтерполяція за часом
        if interpolate_check:
            df_point[selected_param] = data_series
            df_point = interpolate_time(df_point, time_col="Datetime", method="time")
            data_series = df_point[selected_param].astype(float)

        # ---- Статистика вхідного ряду (до нормалізації) ----
        N_input = len(data_series)
        if N_input == 0:
            st.error("Немає даних у вибраному діапазоні!")
            st.stop()

        min_input = data_series.min()
        max_input = data_series.max()
        mean_input = data_series.mean()
        std_input = data_series.std()

        st.write("### Статистика вхідних даних (після фільтрації/інтерполяції):")
        st.write(f"- Кількість точок: {N_input}")
        st.write(f"- min = {min_input:.6f}")
        st.write(f"- max = {max_input:.6f}")
        st.write(f"- mean = {mean_input:.6f}")
        st.write(f"- std = {std_input:.6f}")

        # 3) Нормалізація
        if norm_method != "none":
            data_series = normalize_series(data_series, method=norm_method)

        # 4) Відняти середнє
        mean_val = np.mean(data_series)
        if subtract_mean:
            data_centered = data_series - mean_val
        else:
            data_centered = data_series.copy()

        signal = data_centered.values if hasattr(data_centered, "values") else data_centered

        # 5) Декомпозиція
        if method.startswith("VMD"):
            imfs = vmd(
                signal,
                alpha=alpha,
                tau=tau_val,
                K=K,
                DC=DC_param,
                init=1,
                tol=tol_val,
                max_iter=max_iter
            )
        else:
            if not HAS_EMD:
                st.warning("EMD недоступний: немає PyEMD. Установіть 'pip install EMD-signal'.")
                st.stop()
            imfs_emd = emd_decompose(signal)
            if imfs_emd is None:
                st.warning("EMD недоступний (PyEMD не встановлено).")
                st.stop()
            imfs = imfs_emd
            K = imfs.shape[0]  # перевизначаємо K, якщо EMD

        # 6) Додавання середнього назад
        if add_mean_back and subtract_mean:
            imfs += mean_val / K

        # 7) Перевірка відновлення
        sum_imfs = np.sum(imfs, axis=0)
        norm_signal = np.linalg.norm(signal) + 1e-14
        rec_error = np.linalg.norm(signal - sum_imfs) / norm_signal

        # 8) Формуємо масив дат
        dates = df_point["Datetime"].values

        # 9) Малюємо
        fig_result = plot_vmd_subplots(
            dates=dates,
            original_signal=signal,
            imfs=imfs,
            title=f"{method} - Base={selected_base}, Point={selected_point}, Param={selected_param}"
        )
        st.pyplot(fig_result)

        st.write(f"**Relative Reconstruction Error**: {rec_error:.6f}")

        st.write("**IMFs (мін/макс)**:")
        for i in range(K):
            st.write(f"IMF {i+1}: min={imfs[i].min():.6f}, max={imfs[i].max():.6f}")

        # 10) Збереження
        if st.checkbox("Зберегти IMFs у CSV"):
            df_imfs = pd.DataFrame(imfs.T, columns=[f"IMF_{i+1}" for i in range(K)])
            df_imfs.insert(0, "Datetime", dates)
            csv_data = df_imfs.to_csv(index=False)
            st.download_button(
                label="Завантажити IMFs.csv",
                data=csv_data,
                file_name=f"IMFs_{selected_base}_{selected_point}_{selected_param}.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
