import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Готовим папку для вывода
os.makedirs("output", exist_ok=True)

df = pd.read_csv("measurements_all_days.csv")
channels = ["PM2.5(ug/m3)", "PM10(ug/m3)", "PARTICLES(per/L)",
            "HCHO(mg/m3)", "TVOC(mg/m3)", "TEMP", "HUMI(%RH)"]
df = df[channels].dropna()

values = df.std().values
values_log = np.log10(values + 1)
values_log = np.concatenate((values_log, [values_log[0]]))

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.plot(angles, values_log, color='g', linewidth=2)
ax.fill(angles, values_log, color='g', alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=10)
ax.set_title("Log10(Standard deviation) across all sensor channels", va='bottom')
plt.tight_layout()
plt.savefig("output/radar_std_log.png", dpi=300)
plt.close()
