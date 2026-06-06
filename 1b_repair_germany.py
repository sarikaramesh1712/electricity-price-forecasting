import pandas as pd
from entsoe import EntsoePandasClient
import time
import warnings

warnings.filterwarnings('ignore')

# Run this script if Germany 2021, 2022, 2023 are missing from clean_market_data.csv
API_KEY       = 'f58fcb8e-41ec-43f6-ba4f-7e3433d16538'
client        = EntsoePandasClient(api_key=API_KEY)
GERMANY_CODE  = 'DE_LU'
MISSING_YEARS = [2021, 2022, 2023]

RESOURCE_MAPPING = {
    'Biomass': ['Biomass'],
    'Fossil':  ['Fossil Gas', 'Fossil Hard coal', 'Fossil Brown coal/Lignite',
                'Fossil Oil', 'Fossil Coal-derived gas', 'Fossil Peat'],
    'Water':   ['Hydro Run-of-river and poundage', 'Hydro Water Reservoir',
                'Hydro Pumped Storage'],
    'Sun':     ['Solar'],
}


def flatten_generation_columns(gen_raw):
    if isinstance(gen_raw.columns, pd.MultiIndex):
        if 'Actual Aggregated' in gen_raw.columns.get_level_values(0):
            gen_raw = gen_raw['Actual Aggregated']
        else:
            gen_raw.columns = gen_raw.columns.droplevel(0)
    if gen_raw.columns.duplicated().any():
        gen_raw = gen_raw.T.groupby(level=0).sum().T
    return gen_raw


print("Repairing Germany data: 2021, 2022, 2023...\n")
new_data = []

for year in MISSING_YEARS:
    start = pd.Timestamp(f'{year}-01-01', tz='UTC')
    end   = pd.Timestamp(f'{year}-12-31 23:59', tz='UTC')
    print(f"  Fetching Germany {year}...", end=' ', flush=True)

    try:
        prices = client.query_day_ahead_prices(GERMANY_CODE, start=start, end=end)
        prices = prices.resample('1h').mean().to_frame(name='Price')
    except Exception as e:
        print(f"Prices failed: {e}")
        continue

    time.sleep(3)

    try:
        gen_raw   = client.query_generation(GERMANY_CODE, start=start, end=end)
        gen_raw   = flatten_generation_columns(gen_raw)
        gen_clean = pd.DataFrame(index=gen_raw.index)
        for category, entsoe_cols in RESOURCE_MAPPING.items():
            matched           = [c for c in entsoe_cols if c in gen_raw.columns]
            gen_clean[category] = gen_raw[matched].sum(axis=1) if matched else 0.0
        gen_clean = gen_clean.resample('1h').mean()
    except Exception as e:
        print(f"Generation failed ({e}). Using zeros.")
        gen_clean = pd.DataFrame(0.0, index=prices.index, columns=list(RESOURCE_MAPPING.keys()))

    time.sleep(3)

    yearly_df             = pd.concat([gen_clean, prices], axis=1).dropna(subset=['Price'])
    yearly_df['Region']   = 'Germany'
    yearly_df[list(RESOURCE_MAPPING.keys())] = yearly_df[list(RESOURCE_MAPPING.keys())].fillna(0.0)
    new_data.append(yearly_df)
    print(f"{len(yearly_df):,} rows saved.")

if new_data:
    existing      = pd.read_csv('clean_market_data.csv', index_col=0, parse_dates=True)
    existing.index = pd.to_datetime(existing.index, utc=True)
    # Remove any existing Germany rows for those years before merging
    existing      = existing[~((existing['Region'] == 'Germany') &
                                (existing.index.year.isin(MISSING_YEARS)))]
    combined      = pd.concat([existing, pd.concat(new_data)]).sort_index()
    combined.to_csv('clean_market_data.csv')
    print(f"\nDone. Germany now has {len(combined[combined['Region'] == 'Germany']):,} rows.")
else:
    print("Nothing downloaded. Check API key or try again later.")