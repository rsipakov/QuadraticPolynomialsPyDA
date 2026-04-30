from pathlib import Path
import pandas as pd
import glob, os, sys

# --- 1. Переходим в папку со скриптом (на всякий случай) ---------------------
os.chdir(Path(__file__).parent)

# --- 2. Собираем файлы с учётом регистра -------------------------------------
files = sorted(glob.glob("csv_data/*.csv-csv_data-combined")) + sorted(glob.glob("csv_data/*.CSV"))

# можно одним паттерном:  glob.glob("csv_data/*.[cC][sS][vV]")

print("Найдено файлов:", len(files))
if not files:
    sys.exit("❌ CSV-файлы не найдены. Проверьте путь/расширение.")

# --- 3. Читаем и объединяем ---------------------------------------------------
dfs = [pd.read_csv(f) for f in files]
combined = pd.concat(dfs, ignore_index=True)

# --- 4. Сохраняем результат ---------------------------------------------------
Path("output").mkdir(exist_ok=True)
combined.to_csv("output/measurements_all_days.csv-csv_data-combined", index=False)
print("✅  measurements_all_days.csv-csv_data-combined создан")

