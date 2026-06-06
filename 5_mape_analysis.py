import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

GRAPHS_DIR    = 'graphs'
REGION_COLORS = {'Lithuania': '#2980B9', 'Germany': '#E74C3C', 'SE4': '#27AE60'}

# Standard MAPE excluding hours where |actual price| < 10 €/MWh.
# Near-zero and negative prices cause division blow-up in MAPE, distorting results.
def mape(actual, predicted, floor=10):
    mask = actual.abs() >= floor
    a, p = actual[mask], predicted[mask]
    if len(a) == 0:
        return np.nan
    return (np.abs((a - p) / a) * 100).mean()


print("Loading master_forecast_results.csv...")
df         = pd.read_csv('master_forecast_results.csv')
time_col   = df.columns[0]
df[time_col] = pd.to_datetime(df[time_col], utc=True)
df.set_index(time_col, inplace=True)

test_df = df[df.index >= '2025-01-01'].copy()
test_df['YearMonth'] = test_df.index.to_period('M')

# ─────────────────────────────────────────────────────────────────────────────
#  COMPUTE MONTHLY MAPE PER REGION
# ─────────────────────────────────────────────────────────────────────────────
monthly_results = {}

print("\nMonthly MAPE across test period (2025-2026):\n")
print(f"{'Month':<12}", end='')
for region in ['Lithuania', 'Germany', 'SE4']:
    print(f"{region:>16}", end='')
print()
print("-" * 60)

all_periods = sorted(test_df['YearMonth'].unique())

for period in all_periods:
    print(f"{str(period):<12}", end='')
    for region in ['Lithuania', 'Germany', 'SE4']:
        rdf = test_df[(test_df['Region'] == region) & (test_df['YearMonth'] == period)]
        val = mape(rdf['Actual_Price'], rdf['Predicted_Price'])
        monthly_results.setdefault(region, []).append(val)
        print(f"{val:>15.1f}%", end='')
    print()

print()
for region in ['Lithuania', 'Germany', 'SE4']:
    vals = pd.Series(monthly_results[region]).dropna()
    print(f"{region}: range {vals.min():.1f}% – {vals.max():.1f}%  |  "
          f"median {vals.median():.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
#  GRAPH 1: Monthly MAPE Over Time — All Regions
# ─────────────────────────────────────────────────────────────────────────────
period_labels = [str(p) for p in all_periods]
x = np.arange(len(period_labels))

fig, ax = plt.subplots(figsize=(16, 6))

for region in ['Lithuania', 'Germany', 'SE4']:
    vals = monthly_results[region]
    ax.plot(x, vals, marker='o', markersize=6, linewidth=2.2,
            color=REGION_COLORS[region], label=region)
    ax.fill_between(x, vals, alpha=0.07, color=REGION_COLORS[region])

# Reference band — the overall MAPE figures cited in the report
ax.axhspan(14, 18, color='gold', alpha=0.18, label='Report baseline (14–18%)')
ax.axhline(50, color='grey', linewidth=0.8, linestyle=':', alpha=0.6)

ax.set_xticks(x)
ax.set_xticklabels(period_labels, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
ax.set_title('Monthly MAPE Evolution Across Test Period (2025–2026)',
             fontsize=14, fontweight='bold', pad=12)
ax.legend(frameon=True, fontsize=10)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax.set_ylim(0, max(max(v) for v in monthly_results.values()) * 1.12)

# Annotate the January 2026 dip (best accuracy period)
best_idx  = period_labels.index('2026-01') if '2026-01' in period_labels else None
if best_idx is not None:
    ax.annotate('Best accuracy\nwindow', xy=(best_idx, 21.4),
                xytext=(best_idx - 1.5, 10),
                arrowprops=dict(arrowstyle='->', color='grey'),
                fontsize=8, color='grey')

plt.tight_layout()
plt.savefig(f'{GRAPHS_DIR}/MAPE_Over_Time.png', dpi=300)
plt.close()
print(f"\nSaved → {GRAPHS_DIR}/MAPE_Over_Time.png")

# ─────────────────────────────────────────────────────────────────────────────
#  GRAPH 2: Grouped Bar Chart — Monthly MAPE per Region
# ─────────────────────────────────────────────────────────────────────────────
bar_width = 0.28
fig, ax   = plt.subplots(figsize=(18, 6))

for i, region in enumerate(['Lithuania', 'Germany', 'SE4']):
    offset = (i - 1) * bar_width
    ax.bar(x + offset, monthly_results[region], bar_width,
           label=region, color=REGION_COLORS[region], alpha=0.85, edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(period_labels, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
ax.set_title('MAPE per Region per Month — Test Period 2025–2026',
             fontsize=14, fontweight='bold', pad=12)
ax.legend(frameon=True, fontsize=10)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))

plt.tight_layout()
plt.savefig(f'{GRAPHS_DIR}/MAPE_Monthly_Bars.png', dpi=300)
plt.close()
print(f"Saved → {GRAPHS_DIR}/MAPE_Monthly_Bars.png")

# ─────────────────────────────────────────────────────────────────────────────
#  GRAPH 3: Rolling 30-Day MAPE — Continuous View
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
fig.suptitle('Rolling 30-Day MAPE — Forecast Accuracy Over Time (2025–2026)',
             fontsize=14, fontweight='bold')

for ax, region in zip(axes, ['Lithuania', 'Germany', 'SE4']):
    rdf = test_df[test_df['Region'] == region].copy()

    # Compute hourly absolute percentage error, then roll 720h (30 days)
    rdf['APE'] = np.where(
        rdf['Actual_Price'].abs() >= 10,
        np.abs((rdf['Actual_Price'] - rdf['Predicted_Price']) / rdf['Actual_Price']) * 100,
        np.nan
    )
    rolling_mape = rdf['APE'].rolling(window=720, min_periods=168).mean()

    ax.plot(rdf.index, rolling_mape, color=REGION_COLORS[region], linewidth=1.8)
    ax.fill_between(rdf.index, rolling_mape, alpha=0.15, color=REGION_COLORS[region])

    ax.axhline(rolling_mape.median(), color='black', linewidth=1,
               linestyle='--', alpha=0.5,
               label=f'Median: {rolling_mape.median():.1f}%')

    ax.set_ylabel('MAPE (%)', fontsize=11)
    ax.set_title(region, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))

plt.tight_layout()
plt.savefig(f'{GRAPHS_DIR}/MAPE_Rolling_30Day.png', dpi=300)
plt.close()
print(f"Saved → {GRAPHS_DIR}/MAPE_Rolling_30Day.png")

print("\nDone. Three MAPE analysis charts saved to graphs/")