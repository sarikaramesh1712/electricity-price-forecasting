import pandas as pd

df = pd.read_csv('master_forecast_results.csv')
time_col = df.columns[0]
df[time_col] = pd.to_datetime(df[time_col], utc=True)
df.set_index(time_col, inplace=True)
df = df.sort_index()

# Remove timezone so Excel accepts it
df.index = df.index.tz_localize(None)

df_2026 = df[df.index.year == 2026][['Region', 'Actual_Price', 'Predicted_Price']].copy()
df_2026.index.name = 'Timestamp'

fname = 'Price_Predictions_2026.xlsx'
with pd.ExcelWriter(fname, engine='openpyxl') as writer:
    for region in ['Lithuania', 'Germany', 'SE4']:
        region_df = df_2026[df_2026['Region'] == region].copy()
        region_df = region_df.drop(columns='Region')
        region_df.columns = ['Actual Price (€/MWh)', 'Predicted Price (€/MWh)']
        region_df.to_excel(writer, sheet_name=region)
        print(f"  Sheet added: {region}  ({len(region_df):,} rows)")

print(f"\nSaved: {fname}")