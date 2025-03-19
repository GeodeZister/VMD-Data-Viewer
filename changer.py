import sys
import pandas as pd
import datetime
from datetime import timedelta


def prepare_gnss_data_to_csv(input_txt_file: str, output_csv_file: str = None):
    """
    Зчитує GNSS-дані з текстового файлу, створює колонку Datetime
    та зберігає таблицю у форматі CSV (з автоматично згенерованою
    або заданою вручну назвою).

    Формат вхідних рядків:
      Point Base Year Month Day North East Height Hour
    без рядка заголовків.
    """
    # Якщо не задано вручну, формуємо назву вихідного файлу
    if output_csv_file is None:
        base_name = input_txt_file.rsplit('.', 1)[0]
        output_csv_file = f"{base_name}_prepared.csv"

    # 1. Імена колонок у тому порядку, що відповідає вхідному файлу
    col_names = ["Point", "Base", "Year", "Month", "Day",
                 "North", "East", "Height", "Hour"]

    # 2. Зчитуємо дані
    df = pd.read_csv(
        input_txt_file,
        sep=r"\s+",  # Замість delim_whitespace=True
        names=col_names,
        header=None
    )

    # 3. Створюємо єдину колонку Datetime
    def build_datetime(row):
        y = int(row["Year"])
        m = int(row["Month"])
        d = int(row["Day"])
        h = int(row["Hour"])

        # Спочатку створюємо дату (без години)
        base_date = datetime.datetime(y, m, d)
        # Додаємо зсув годин (якщо h=24, то це буде +24 години, тобто наступний день)
        result = base_date + timedelta(hours=h)
        return result

    df["Datetime"] = df.apply(build_datetime, axis=1)

    # 4. Видаляємо непотрібні стовпці
    df.drop(["Year", "Month", "Day", "Hour"], axis=1, inplace=True)

    # 5. Сортуємо за часом
    df = df.sort_values("Datetime")

    # 6. Змінюємо порядок колонок (якщо треба)
    df = df[["Point", "Base", "Datetime", "North", "East", "Height"]]

    # 7. Зберігаємо в CSV
    df.to_csv(output_csv_file, index=False)
    print(f"Файл успішно збережено: {output_csv_file}")


def main():
    """
    Використання:
      python changer.py path/to/input.txt [path/to/output.csv]
    """
    if len(sys.argv) < 2:
        print("Помилка: потрібно вказати шлях до вхідного текстового файлу.")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    prepare_gnss_data_to_csv(input_file, output_file)


if __name__ == "__main__":
    main()
