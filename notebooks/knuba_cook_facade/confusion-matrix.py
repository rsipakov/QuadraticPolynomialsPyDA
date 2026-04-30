# cross_sensor_corr_confusionstyle.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

# ---------- 1. Пути и параметры -----------------------------------------
csv_path = "measurements_all_days.csv"
out_dir  = "output"

pm_col = "PM2.5(ug/m3)"
candidate_cols = [
    "PM10(ug/m3)", "PARTICLES(per/L)",
    "HCHO(mg/m3)", "TVOC(mg/m3)",
    "TEMP", "HUMI(%RH)"
]

os.makedirs(out_dir, exist_ok=True)

# ---------- 2. Загрузка данных ------------------------------------------
df = pd.read_csv(csv_path)
df.replace({',': ''}, regex=True, inplace=True)
df = df.apply(pd.to_numeric, errors='ignore')

cols = [pm_col] + [c for c in candidate_cols if c in df.columns]
data = df[cols].dropna(how="any")
corr = data.corr(method="pearson").values   # ndarray (n×n)

# ---------- 3. Подготовка «процентной» матрицы для подписи --------------
# переводим |r| → 0–100 %
pct = np.abs(corr)*100

# ---------- 4. Визуализация ---------------------------------------------
fig_sz = 3.2 + 0.15*len(cols)   # чуть расширяем под количество каналов
fig, ax = plt.subplots(figsize=(fig_sz, fig_sz))

# Бело-синяя gamma (0 → белый, 100 → насыщенный синий)
cmap = plt.cm.Blues
im = ax.imshow(pct, vmin=0, vmax=100, cmap=cmap)

# Квадратные ячейки + тонкая сетка
ax.set_xticks(np.arange(len(cols)))
ax.set_yticks(np.arange(len(cols)))
ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(cols, fontsize=8)
ax.set_xlim(-0.5, len(cols)-0.5)
ax.set_ylim(len(cols)-0.5, -0.5)            # origin at top-left

# Тонкие серые линии сетки (как в confusion matrix)
ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(cols), 1), minor=True)
ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
ax.tick_params(which="minor", bottom=False, left=False)

# Подписи внутри клеток — формат «95.29», центрирование
for i in range(len(cols)):
    for j in range(len(cols)):
        txt = f"{pct[i, j]:.0f}"
        ax.text(j, i, txt, ha="center", va="center",
                fontsize=7, color="black")

# Цветовая шкала сбоку
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("magnitude of Pearson $r$ (%)", fontsize=8)
cbar.ax.tick_params(labelsize=8)
cbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter())

fig.tight_layout()

# ---------- 5. Сохранение -----------------------------------------------
pdf_path = os.path.join(out_dir, "fig_cross_sensor_corr_confusion.pdf")
png_path = os.path.join(out_dir, "fig_cross_sensor_corr_confusion.png")
plt.savefig(pdf_path, dpi=600)
plt.savefig(png_path, dpi=600)
plt.show()

print(f"saved to: {pdf_path}  and  {png_path}")
