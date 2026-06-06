import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

GRAPHS_DIR    = 'graphs'
COLOR_ACTUAL  = '#0047AB'
COLOR_PREDICT = '#CC0000'
COLOR_BAND    = '#AAAAFF'

print("Loading master_forecast_results.csv...")
df         = pd.read_csv('master_forecast_results.csv')
time_col   = df.columns[0]
df[time_col] = pd.to_datetime(df[time_col], utc=True)
df.set_index(time_col, inplace=True)
df['Year']  = df.index.year
df['Month'] = df.index.month
df['Hour']  = df.index.hour
df['Error'] = df['Predicted_Price'] - df['Actual_Price']


def save_macro_trend(rdf, region, year):
    actual_sm  = rdf['Actual_Price'].rolling(168, center=True).mean()
    predict_sm = rdf['Predicted_Price'].rolling(168, center=True).mean()
    std_sm     = rdf['Error'].rolling(168, center=True).std()

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.fill_between(rdf.index, predict_sm - std_sm, predict_sm + std_sm,
                    color=COLOR_BAND, alpha=0.25, label='±1 Std Error Band')
    ax.plot(rdf.index, actual_sm,  color=COLOR_ACTUAL,  linewidth=2,   label='Actual (7-Day Avg)')
    ax.plot(rdf.index, predict_sm, color=COLOR_PREDICT, linewidth=1.8,
            linestyle='--', label='AI Forecast (7-Day Avg)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.set_title(f'{region} — Electricity Price Trend {year} (7-Day Rolling Average)',
                 fontsize=15, fontweight='bold', pad=12)
    ax.set_ylabel('Price (€/MWh)', fontsize=12)
    ax.legend(fontsize=10, frameon=True)
    ax.set_xlim(rdf.index.min(), rdf.index.max())
    plt.tight_layout()
    plt.savefig(f'{GRAPHS_DIR}/Macro_Trend_{region}_{year}.png', dpi=300)
    plt.close()


def save_daily_profile(rdf, region, year):
    profile  = rdf.groupby('Hour')[['Actual_Price', 'Predicted_Price']].mean()
    err_std  = rdf.groupby('Hour')['Error'].std()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(profile.index,
                    profile['Predicted_Price'] - err_std,
                    profile['Predicted_Price'] + err_std,
                    color=COLOR_BAND, alpha=0.3, label='Prediction ±1 Std')
    ax.plot(profile.index, profile['Actual_Price'],    color=COLOR_ACTUAL,
            marker='o', markersize=7, linewidth=2.5, label='Actual Average')
    ax.plot(profile.index, profile['Predicted_Price'], color=COLOR_PREDICT,
            marker='s', markersize=7, linewidth=2.5, linestyle='--', label='Predicted Average')

    # Shade intraday peak demand windows
    ax.axvspan(7,  11, alpha=0.07, color='orange', label='Morning Peak')
    ax.axvspan(17, 21, alpha=0.07, color='red',    label='Evening Peak')

    ax.set_title(f'Average Daily Price Profile — {region} ({year})',
                 fontsize=15, fontweight='bold', pad=12)
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Average Price (€/MWh)', fontsize=12)
    ax.set_xticks(range(0, 24))
    ax.legend(fontsize=9, frameon=True, ncol=2)
    plt.tight_layout()
    plt.savefig(f'{GRAPHS_DIR}/Daily_Profile_{region}_{year}.png', dpi=300)
    plt.close()


def save_monthly_heatmap(region_df, region):
    pivot = region_df.groupby(['Year', 'Month'])['Actual_Price'].mean().unstack(level=1)
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:pivot.shape[1]]

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.7)))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn_r',
                linewidths=0.5, ax=ax, cbar_kws={'label': '€/MWh'})
    ax.set_title(f'Monthly Average Electricity Price — {region}',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Month', fontsize=11)
    ax.set_ylabel('Year',  fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{GRAPHS_DIR}/Monthly_Heatmap_{region}.png', dpi=300)
    plt.close()


def save_error_distribution(region_df, region):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(region_df['Error'].dropna(), bins=60,
                 color=COLOR_PREDICT, alpha=0.75, edgecolor='white')
    axes[0].axvline(0, color='black', linewidth=1.5, linestyle='--')
    mean_err = region_df['Error'].mean()
    axes[0].axvline(mean_err, color='blue', linewidth=1.5, label=f'Mean error: €{mean_err:.1f}')
    axes[0].set_title(f'Prediction Error Distribution — {region}', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Error (€/MWh) = Predicted − Actual', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].legend()

    sample = region_df.dropna().sample(min(3000, len(region_df)), random_state=42)
    axes[1].scatter(sample['Actual_Price'], sample['Predicted_Price'],
                    alpha=0.25, s=8, color=COLOR_ACTUAL)
    lo = min(sample['Actual_Price'].min(), sample['Predicted_Price'].min())
    hi = max(sample['Actual_Price'].max(), sample['Predicted_Price'].max())
    axes[1].plot([lo, hi], [lo, hi], 'r--', linewidth=2, label='Perfect Forecast')
    axes[1].set_title(f'Actual vs Predicted — {region}', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Actual Price (€/MWh)', fontsize=11)
    axes[1].set_ylabel('Predicted Price (€/MWh)', fontsize=11)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{GRAPHS_DIR}/Error_Distribution_{region}.png', dpi=300)
    plt.close()


for region in df['Region'].unique():
    print(f"Generating report graphs: {region}...")
    region_df = df[df['Region'] == region]

    for year in sorted(region_df['Year'].unique()):
        ydf = region_df[region_df['Year'] == year]
        if len(ydf) < 24:
            continue
        save_macro_trend(ydf, region, year)
        save_daily_profile(ydf, region, year)

    save_monthly_heatmap(region_df, region)
    save_error_distribution(region_df, region)
    print(f"  Done.")

print(f"\nDone. All report graphs saved to {GRAPHS_DIR}/")