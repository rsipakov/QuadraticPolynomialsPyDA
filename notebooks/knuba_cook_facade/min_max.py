import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Готовим папку
os.makedirs("output", exist_ok=True)

# Загрузка данных
df = pd.read_csv("measurements_all_days.csv")
pm_channels = ["PM2.5(ug/m3)", "PM10(ug/m3)", "PARTICLES(per/L)"]
df_pm = df[pm_channels].dropna()

# Стандартное отклонение и min-max нормализация для PM-каналов
values_pm = df_pm.std().values
values_pm_norm = (values_pm - values_pm.min()) / (values_pm.max() - values_pm.min())
# Замыкаем круг
values_pm_norm = np.concatenate((values_pm_norm, [values_pm_norm[0]]))
num_vars_pm = len(pm_channels)
angles_pm = np.linspace(0, 2 * np.pi, num_vars_pm, endpoint=False)
angles_pm = np.concatenate((angles_pm, [angles_pm[0]]))

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.plot(angles_pm, values_pm_norm, color='r', linewidth=2)
ax.fill(angles_pm, values_pm_norm, color='r', alpha=0.25)
ax.set_xticks(angles_pm[:-1])
ax.set_xticklabels(pm_channels, fontsize=12)
ax.set_title("Normalized Std for PM-channels", va='bottom')
plt.tight_layout()
plt.savefig("output/radar_std_pm.png", dpi=300)
plt.close()
