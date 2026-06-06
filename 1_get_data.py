import pandas as pd
from entsoe import EntsoePandasClient
import time
import warnings

warnings.filterwarnings('ignore')

API_KEY = 'f58fcb8e-41ec-43f6-ba4f-7e3433d16538'
client  = EntsoePandasClient(api_key=API_KEY)

REGIONS = {
    'Lithuania': 'LT',
    'Germany':   'DE_LU',
    'SE4':       'SE_4',
}

RESOURCE_MAPPING = {
    'Biomass': ['Biomass'],
    'Fossil':  ['Fossil Gas', 'Fossil Hard coal', 'Fossil Brown coal/Lignite',
                'Fossil Oil', 'Fossil Coal-derived gas', 'Fossil Peat'],
    'Water':   ['Hydro Run-of-river and poundage', 'Hydro Water Reservoir',
                'Hydro Pumped Storage'],
    'Sun':     ['Solar'],
}

YEARS    = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
all_data = []

print("Starting ENTSO-E data download (2020-2026)...")
print("Expected time: 10-20 minutes. Do not close the terminal.\n")


def flatten_generation_columns(gen_raw):
    # ENTSO-E sometimes returns MultiIndex columns; keep only Actual Aggregated
    if isinstance(gen_raw.columns, pd.MultiIndex):
        if 'Actual Aggregated' in gen_raw.columns.get_level_values(0):
            gen_raw = gen_raw['Actual Aggregated']
        else:
            gen_raw.columns = gen_raw.columns.droplevel(0)
    if gen_raw.columns.duplicated().any():
        gen_raw = gen_raw.T.groupby(level=0).sum().T
    return gen_raw


for region_name, country_code in REGIONS.items():
    print(f"Downloading: {region_name} ({country_code})")

    for year in YEARS:
        start = pd.Timestamp(f'{year}-01-01', tz='UTC')
        end   = (pd.Timestamp.now(tz='UTC').floor('D')
                 if year == 2026
                 else pd.Timestamp(f'{year}-12-31 23:59', tz='UTC'))

        print(f"  Fetching {year}...", end=' ', flush=True)

        try:
            prices  = client.query_day_ahead_prices(country_code, start=start, end=end)
            prices  = prices.resample('1h').mean().to_frame(name='Price')

            gen_raw = client.query_generation(country_code, start=start, end=end)
            gen_raw = flatten_generation_columns(gen_raw)

            gen_clean = pd.DataFrame(index=gen_raw.index)
            for category, entsoe_cols in RESOURCE_MAPPING.items():
                matched           = [c for c in entsoe_cols if c in gen_raw.columns]
                gen_clean[category] = gen_raw[matched].sum(axis=1) if matched else 0.0

            gen_clean           = gen_clean.resample('1h').mean()
            yearly_df           = pd.concat([gen_clean, prices], axis=1).dropna(subset=['Price'])
            yearly_df['Region'] = region_name
            yearly_df[list(RESOURCE_MAPPING.keys())] = yearly_df[list(RESOURCE_MAPPING.keys())].fillna(0.0)

            all_data.append(yearly_df)
            print(f"{len(yearly_df):,} rows saved.")

        except Exception as e:
            print(f"Skipped. Error: {e}")

        time.sleep(2)

if all_data:
    final_dataset = pd.concat(all_data).sort_index()
    final_dataset.to_csv('clean_market_data.csv')
    print(f"\nDone. clean_market_data.csv saved ({len(final_dataset):,} rows).")
else:
    print("No data downloaded. Check API key and internet connection.")