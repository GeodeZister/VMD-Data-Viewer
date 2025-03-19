import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime



# ==========================================================================
#  1. Функція Variational Mode Decomposition
# ==========================================================================

def vmd(signal, alpha=2000, tau=0, K=3, DC=0, init=1, tol=1e-7, max_iter=500):
    """
    Спрощена реалізація Variational Mode Decomposition (VMD).

    Параметри:
    ----------
    signal : np.array
        Вхідний часовий ряд.
    alpha : float
        Коефіцієнт, пов'язаний зі штрафом за ширину смуги для кожної моди.
    tau : float
        Коєфіцієнт ADMM (зазвичай 0).
    K : int
        Кількість мод (IMF), які хочемо отримати.
    DC : int
        Чи вилучати DC компоненту (0 = так, 1 = ні).
    init : int
        Ініціалізація частот: 0 = всі нулі, 1 = розгортка лінійно.
    tol : float
        Допуск на зупинку ітерацій.
    max_iter : int
        Максимальна кількість ітерацій.

    Повертає:
    --------
    u : np.ndarray
        Масив розмірності (K, N) із K модами, на які розклали початковий сигнал.
    """

    # Довжина сигналу
    N = len(signal)

    # Індекси для перетворення Фур'є
    freqs = np.fft.fftfreq(N, 1.0 / N)  # крок = 1

    # Перетворення Фур'є від початкового сигналу
    f_signal = np.fft.fft(signal)

    # Ініціалізація
    if init == 1:
        omega_k = np.linspace(0, 0.5 * N, K, endpoint=False)
    else:
        omega_k = np.zeros(K)
    u = np.zeros((K, N), dtype=np.float64)
    u_hat = np.zeros((K, N), dtype=np.complex128)
    lambda_hat = np.zeros(N, dtype=np.complex128)

    # Допоміжна функція для зсуву частоти
    def shift_freq(f, freq_center):
        return np.exp(1j * 2 * np.pi * f * freq_center / N)

    # AL iteration (псевдо-ітерація ADMM)
    for n in range(max_iter):
        u_hat_old = u_hat.copy()

        for k in range(K):
            # Тимчасовий внесок решти мод
            sum_others = np.sum(u_hat[[i for i in range(K) if i != k]], axis=0)
            residue = f_signal - sum_others - lambda_hat / 2

            # Оновлюємо частоту k-го IMF (omega_k)
            # Часто роблять update через зважений момент "f * |(residue * shift_freq())|^2"
            # Тут можна знайти "центр мас" у спектральній області
            numerator = np.sum(freqs * np.abs(residue * shift_freq(freqs, -omega_k[k])) ** 2)
            denominator = np.sum(np.abs(residue * shift_freq(freqs, -omega_k[k])) ** 2) + 1e-12
            omega_k[k] = alpha * numerator / denominator

            # Оновлюємо u_hat[k]
            # Приблизна формула (див. офіційний псевдокод VMD)
            u_hat[k] = residue * shift_freq(freqs, -omega_k[k]) / (1 + 2 * alpha * (freqs - omega_k[k]) ** 2)

        # Оновлюємо lambda_hat
        sum_u = np.sum(u_hat, axis=0)
        lambda_hat = lambda_hat + tau * (sum_u - f_signal)

        # Перевірка зупинки за приростом
        diff = 0
        for k in range(K):
            diff += np.sum(np.abs(u_hat[k] - u_hat_old[k]) ** 2)
        if diff < tol:
            break

    # Робимо зворотне перетворення Фур'є
    for k in range(K):
        u[k, :] = np.fft.ifft(u_hat[k]).real

    return u


# ==========================================================================
#  2. Основний приклад використання: зчитуємо CSV, робимо VMD для "East"
# ==========================================================================

def main():
    # !!! Якщо ваш файл називається інакше — змініть шлях тут:
    csv_file = "R1G_prepared.csv"  # <-- Задайте свій шлях, наприклад "R1G_prepared.csv"

    # Зчитуємо
    df = pd.read_csv(csv_file, parse_dates=["Datetime"])

    # Припустимо, ми хочемо розглянути пункт "PP4", і колонку "East"
    point_name = "ACP1"
    col_name = "Height"

    df_point = df[df["Point"] == point_name].copy()
    df_point.sort_values(by="Datetime", inplace=True)

    # Витягаємо час (для візуалізації) та сам сигнал
    time = df_point["Datetime"].values
    signal = df_point[col_name].values

    # Виконуємо VMD (налаштовуємо параметри на власний розсуд)
    K = 3  # Кількість мод
    alpha = 2000
    tol = 1e-7
    max_iter = 500

    imfs = vmd(signal, alpha=alpha, K=K, tol=tol, max_iter=max_iter)

    # 3. Будуємо графіки
    plt.figure(figsize=(10, 6))
    plt.plot(time, signal, label="Original signal")
    plt.title(f"Original '{col_name}' for point '{point_name}'")
    plt.xlabel("Time")
    plt.ylabel(col_name)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Кожен IMF
    for i in range(K):
        plt.figure(figsize=(10, 3))
        plt.plot(time, imfs[i], label=f"IMF {i + 1}")
        plt.title(f"IMF {i + 1} (of {K})")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
