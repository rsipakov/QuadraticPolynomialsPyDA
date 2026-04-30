import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. Пути и параметры -----------------------------------------
csv_path = "measurements_all_days.csv"   # путь к исходным данным
out_dir  = "output_2"                                # куда сохранять рисунки

pm_col = "PM2.5(ug/m3)"
candidate_cols = [
    "PM10(ug/m3)",
    "PARTICLES(per/L)",
    "HCHO(mg/m3)",
    "TVOC(mg/m3)",
    "TEMP",
    "HUMI(%RH)"
]

# ---------- 2. Подготовка выходной папки --------------------------------
os.makedirs(out_dir, exist_ok=True)  # создаёт "output", если её нет

# ---------- 3. Загрузка и очистка ---------------------------------------
df = pd.read_csv(csv_path)
df.replace({',': ''}, regex=True, inplace=True)    # убираем запятые в числах
df = df.apply(pd.to_numeric, errors='ignore')

cols = [pm_col] + [c for c in candidate_cols if c in df.columns]
data = df[cols].dropna(how="any")

# ---------- 4. Корреляционная матрица -----------------------------------
corr = data.corr(method="pearson")                 # можно заменить на "spearman"

# ---------- 5. Визуализация ---------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="viridis")

ax.set_xticks(np.arange(len(cols)))
ax.set_yticks(np.arange(len(cols)))
ax.set_xticklabels(cols, rotation=45, ha="right")
ax.set_yticklabels(cols)

for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f"{corr.values[i, j]:+.2f}",
                ha="center", va="center", fontsize=8, color="white")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Pearson r")

fig.tight_layout()

# ---------- 6. Сохранение ------------------------------------------------
pdf_path = os.path.join(out_dir, "fig_cross_sensor_corr.pdf")
png_path = os.path.join(out_dir, "fig_cross_sensor_corr.png")
plt.savefig(pdf_path, dpi=600)
plt.savefig(png_path, dpi=600)
plt.show()

print(f"Графики сохранены в: {pdf_path} и {png_path}")

