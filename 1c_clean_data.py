import pandas as pd
import warnings

warnings.filterwarnings('ignore')

print("Loading clean_market_data.csv...")
df         = pd.read_csv('clean_market_data.csv')
time_col   = df.columns[0]
df[time_col] = pd.to_datetime(df[time_col], utc=True)
df.set_index(time_col, inplace=True)
df.index.name = 'Timestamp'
df = df.sort_index()

# Cap extreme price outliers at 99.9th percentile per region (winsorization)
print("Capping extreme price outliers...\n")
for region in df['Region'].unique():
    mask  = df['Region'] == region
    upper = df.loc[mask, 'Price'].quantile(0.999)
    lower = df.loc[mask, 'Price'].quantile(0.001)
    df.loc[mask, 'Price'] = df.loc[mask, 'Price'].clip(lower=lower, upper=upper)
    print(f"  {region:<12}  upper cap: {upper:>7.1f} €/MWh   lower cap: {lower:>7.1f} €/MWh")

gen_cols = [c for c in ['Biomass', 'Fossil', 'Water', 'Sun'] if c in df.columns]
df[gen_cols] = df[gen_cols].fillna(0.0)

# Data completeness report
print("\nData Completeness:")
print(f"{'Region':<14} {'Year':<6} {'Rows':>6}  {'Expected':>8}  {'Coverage':>8}")
print("-" * 48)
df['Year'] = df.index.year
for region in df['Region'].unique():
    for year in sorted(df[df['Region'] == region]['Year'].unique()):
        rows     = len(df[(df['Region'] == region) & (df['Year'] == year)])
        expected = 8784 if year % 4 == 0 else 8760
        expected = rows if year == 2026 else expected
        flag     = '' if (rows / expected) > 0.95 else '  WARNING: incomplete'
        print(f"  {region:<12} {year:<6} {rows:>6}  {expected:>8}  {rows/expected*100:>6.1f}%{flag}")

df.drop(columns=['Year'], inplace=True)
df.to_csv('clean_market_data.csv')
print(f"\nDone. clean_market_data.csv saved ({len(df):,} rows).")