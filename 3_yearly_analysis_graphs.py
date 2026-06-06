import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

GRAPHS_DIR = 'graphs'

COLORS = {
    'Sun':       '#FFD700',
    'Water':     '#3498DB',
    'Biomass':   '#27AE60',
    'Fossil':    '#7F8C8D',
    'Actual':    '#2C3E50',
    'Predicted': '#E74C3C',
}

print("Loading master_forecast_results.csv...")
df         = pd.read_csv('master_forecast_results.csv')
time_col   = df.columns[0]
df[time_col] = pd.to_datetime(df[time_col], utc=True)
df.set_index(time_col, inplace=True)
df['Year'] = df.index.year

yearly  = df.groupby(['Region', 'Year']).mean(numeric_only=True).reset_index()
GEN_COLS = [c for c in ['Fossil', 'Biomass', 'Water', 'Sun'] if c in df.columns]


def plot_price_bars(region_data, region_name):
    years     = region_data['Year'].astype(str).tolist()
    actuals   = region_data['Actual_Price'].values
    predicted = region_data['Predicted_Price'].values
    x, width  = np.arange(len(years)), 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    bars_a  = ax.bar(x - width/2, actuals,   width, label='Actual Day-Ahead Price',
                     color=COLORS['Actual'],    alpha=0.85, edgecolor='white')
    bars_p  = ax.bar(x + width/2, predicted, width, label='AI Predicted Price',
                     color=COLORS['Predicted'], alpha=0.85, edgecolor='white')

    for bar in bars_a:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f'€{h:.0f}',
                ha='center', va='bottom', fontsize=8, color=COLORS['Actual'], fontweight='bold')
    for bar in bars_p:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f'€{h:.0f}',
                ha='center', va='bottom', fontsize=8, color=COLORS['Predicted'], fontweight='bold')

    ax.set_ylabel('Average Price (€/MWh)', fontsize=12, fontweight='bold')
    ax.set_title(f'Day-Ahead Forecast Accuracy (2020–2026) — {region_name}',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.legend(frameon=True)
    ax.set_ylim(0, max(max(actuals), max(predicted)) * 1.18)

    plt.tight_layout()
    plt.savefig(f'{GRAPHS_DIR}/Yearly_Price_Forecast_{region_name}.png', dpi=300)
    plt.close()


def plot_resource_vs_price(region_data, region_name):
    years       = region_data['Year'].astype(str).tolist()
    gen_present = [c for c in GEN_COLS if region_data[c].sum() > 0]

    fig, ax1 = plt.subplots(figsize=(13, 7))

    bottom = np.zeros(len(years))
    for res in gen_present:
        ax1.bar(years, region_data[res].values, bottom=bottom, label=res,
                color=COLORS.get(res, '#AAAAAA'), alpha=0.82, edgecolor='white')
        bottom += region_data[res].values

    ax1.set_ylabel('Average Hourly Generation (MW)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, title='Generation Source')

    ax2 = ax1.twinx()
    ax2.plot(years, region_data['Actual_Price'].values,   color=COLORS['Actual'],
             marker='o', linewidth=2.5, markersize=8, label='Actual Price')
    ax2.plot(years, region_data['Predicted_Price'].values, color=COLORS['Predicted'],
             marker='s', linewidth=2.5, markersize=8, linestyle='--', label='Predicted Price')
    ax2.set_ylabel('Electricity Price (€/MWh)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True)

    plt.title(f'Generation Mix vs. Electricity Prices (2020–2026) — {region_name}',
              fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(f'{GRAPHS_DIR}/Yearly_Resources_vs_Price_{region_name}.png', dpi=300)
    plt.close()


def plot_cross_region_comparison(yearly_df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors    = {'Lithuania': '#2980B9', 'Germany': '#E74C3C', 'SE4': '#27AE60'}

    for region in yearly_df['Region'].unique():
        rdata = yearly_df[yearly_df['Region'] == region]
        axes[0].plot(rdata['Year'], rdata['Actual_Price'],   marker='o', linewidth=2.5,
                     markersize=8, label=region, color=colors.get(region))
        axes[1].plot(rdata['Year'], rdata['Predicted_Price'], marker='s', linewidth=2.5,
                     markersize=8, linestyle='--', label=region, color=colors.get(region))

    for ax, title in zip(axes, ['Actual Day-Ahead Prices', 'AI Predicted Prices']):
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylabel('Average Price (€/MWh)', fontsize=11)
        ax.set_xlabel('Year', fontsize=11)
        ax.legend(frameon=True)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.suptitle('Cross-Region Electricity Price Comparison (2020–2026)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{GRAPHS_DIR}/CrossRegion_Comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


for region in df['Region'].unique():
    print(f"Generating yearly charts: {region}...")
    rdata = yearly[yearly['Region'] == region]
    plot_price_bars(rdata, region)
    plot_resource_vs_price(rdata, region)

plot_cross_region_comparison(yearly)
print(f"\nDone. All charts saved to {GRAPHS_DIR}/")