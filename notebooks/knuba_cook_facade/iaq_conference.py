import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from dateutil import parser
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from datetime import date


CSV      = "measurements_all_days.csv"
BAR_ON  = pd.Timestamp("2024-12-20")
BAR_OFF = pd.Timestamp("2025-01-17")
OUT_DIR  = Path("output_final")


df_raw = pd.read_csv(CSV)

dt_col = next(c for c in df_raw.columns if "date" in c.lower() or "time" in c.lower())
pm_col = next(c for c in df_raw.columns if "pm2" in c.lower() or "pm2.5" in c.lower())

def parse_dt(s: str):

    for fmt in ("%m/%d/%y %I:%M:%S %p", "%m/%d/%y %H:%M", "%m/%d/%y %H:%M:%S"):
        try:
            return pd.to_datetime(s, format=fmt)
        except Exception:
            pass
    try:
        return parser.parse(s, dayfirst=False)
    except Exception:
        return pd.NaT

df_raw[dt_col] = df_raw[dt_col].astype(str).apply(parse_dt)
df = (df_raw[[dt_col, pm_col]]
        .dropna()
        .rename(columns={dt_col: "datetime", pm_col: "PM25"})
        .sort_values("datetime")
        .set_index("datetime"))

if df.empty:
    raise SystemExit("✖ Данные не распарсились — проверьте исходный CSV")


US_HOLIDAYS = {
    date(2024, 12, 25),  # Christmas Day
    date(2025, 1, 1),    # New Year’s Day
    # add others as needed
}


def phase(ts: pd.Timestamp) -> str:
    h = ts.hour + ts.minute / 60

    # cooking always has priority
    if (9 <= h < 10) or (12 <= h < 14) or (20 <= h < 21):
        return "cooking"

    is_weekday = ts.weekday() < 5          # Monday–Friday
    holiday    = ts.date() in US_HOLIDAYS  # True if federal holiday

    if is_weekday and not holiday:
        if (8.5 <= h < 12) or (13 <= h < 17):
            return "facade"

    return "background"

def period(ts):
    if ts < BAR_ON:
        return "pre-barrier"
    elif ts < BAR_OFF:
        return "barrier-on"
    else:
        return "post-barrier"

df["phase"]  = df.index.map(phase)
df["period"] = df.index.map(period)
df["hour"]   = df.index.hour + df.index.minute/60

print("\nPhase counts:\n",   df.phase.value_counts())
print("\nPeriod counts:\n",  df.period.value_counts())


df["baseline"] = np.nan
quad = make_pipeline(PolynomialFeatures(2), Ridge(alpha=1.0))

for p in df.period.unique():
    bg = df[(df.period == p) & (df.phase == "background")]
    if len(bg) < 10:
        df.loc[df.period == p, "baseline"] = bg.PM25.median() if len(bg) else df.PM25.median()
        print(f"• period {p}: baseline = median (not enough data)")
    else:
        quad.fit(bg[["hour"]], bg["PM25"])
        df.loc[df.period == p, "baseline"] = quad.predict(df.loc[df.period == p, ["hour"]])

df["delta"] = df["PM25"] - df["baseline"]


src = (df.groupby("phase")
         .agg(mean_PM25=("PM25","mean"),
              max_PM25=("PM25","max"),
              dose     =("delta", lambda x: x[x>0].sum())))
src["dose_pct"] = 100 * src.dose / src.dose.sum()

bg_period = (df[df.phase == "background"]
              .groupby("period")
              .agg(mean_PM25=("PM25","mean"),
                   p95_PM25=("PM25", lambda x: np.percentile(x, 95))))

print("\n=== Source contribution ===")
print(src.round(2))
print("\n=== Background per period ===")
print(bg_period.round(2))


bg_all = df[df.phase == "background"].copy()
X = bg_all[["hour"]].values
y = bg_all["PM25"].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)
mae_list = []

for train_idx, test_idx in kf.split(X):
    model = make_pipeline(PolynomialFeatures(2), Ridge(alpha=1.0))
    model.fit(X[train_idx], y[train_idx])
    y_pred = model.predict(X[test_idx])
    mae_list.append(mean_absolute_error(y[test_idx], y_pred))

mae_mean = np.mean(mae_list)
mae_std  = np.std(mae_list)
print(f"\n5-fold CV MAE = {mae_mean:.2f} ± {mae_std:.2f} µg/m³")



OUT_DIR.mkdir(exist_ok=True)


plt.figure(figsize=(8,4))
for p, g in df[df.phase=="background"].groupby("period"):
    g.groupby("hour").PM25.mean().plot(label=p)
plt.xlabel("Hour"); plt.ylabel("Mean PM₂.₅ (µg/m³)")
plt.title("Background PM₂.₅ — three periods"); plt.legend()
plt.tight_layout(); plt.savefig(OUT_DIR/"baseline_periods.png", dpi=300); plt.close()


src.loc[["background","facade","cooking"]].dose_pct.plot(
        kind="bar", figsize=(4,4), ylabel="Dose Share (%)",
        title="Source Contribution to Daily PM₂.₅ Dose")
plt.tight_layout(); plt.savefig(OUT_DIR/"dose_share.png", dpi=300); plt.close()


colors = {"pre-barrier":"tab:blue", "barrier-on":"tab:orange", "post-barrier":"tab:green"}
plt.figure(figsize=(10,3))
for p, c in colors.items():
    df[df.period == p].PM25.plot(lw=0.6, color=c, label=p)
plt.ylabel("PM₂.₅ (µg/m³)"); plt.title("PM₂.₅ Time Series (period colour)")
plt.legend(); plt.tight_layout()
plt.savefig(OUT_DIR/"timeline_colored.png", dpi=300); plt.close()

print(f"\nPNG files saved to {OUT_DIR.resolve()}")
