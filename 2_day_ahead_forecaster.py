import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import joblib
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

GRAPHS_DIR = 'graphs'
SPLIT_DATE = '2025-01-01'
FEATURES   = [
    'Fossil', 'Biomass', 'Water', 'Sun',
    'Hour', 'DayOfWeek', 'Month', 'Quarter', 'IsWeekend', 'IsMonday',
    'Price_Lag_24', 'Price_Lag_48', 'Price_Lag_168',
    'Price_RollMean_24', 'Price_RollStd_24', 'Price_RollMean_168',
]

print("Loading clean_market_data.csv...")
df         = pd.read_csv('clean_market_data.csv')
time_col   = df.columns[0]
df[time_col] = pd.to_datetime(df[time_col], utc=True)
df.set_index(time_col, inplace=True)
df = df.sort_index()


def build_features(region_df):
    d = region_df.copy()
    d['Hour']      = d.index.hour
    d['DayOfWeek'] = d.index.dayofweek
    d['Month']     = d.index.month
    d['Quarter']   = d.index.quarter
    d['IsWeekend'] = (d.index.dayofweek >= 5).astype(int)
    d['IsMonday']  = (d.index.dayofweek == 0).astype(int)

    # Lag features use only past prices — no data leakage into the forecast horizon
    d['Price_Lag_24']      = d['Price'].shift(24)
    d['Price_Lag_48']      = d['Price'].shift(48)
    d['Price_Lag_168']     = d['Price'].shift(168)
    d['Price_RollMean_24'] = d['Price'].shift(24).rolling(24).mean()
    d['Price_RollStd_24']  = d['Price'].shift(24).rolling(24).std()
    d['Price_RollMean_168']= d['Price'].shift(168).rolling(168).mean()
    return d


all_results  = []
metrics_list = []

for region in df['Region'].unique():
    print(f"\nTraining model: {region}")

    region_df = df[df['Region'] == region].copy()
    region_df = build_features(region_df)

    # Target: price 24 hours ahead (day-ahead forecast)
    region_df['Target'] = region_df['Price'].shift(-24)
    region_df.dropna(subset=FEATURES + ['Target'], inplace=True)

    X = region_df[FEATURES]
    y = region_df['Target']

    X_train = X[X.index < SPLIT_DATE]
    y_train = y[y.index < SPLIT_DATE]
    X_test  = X[X.index >= SPLIT_DATE]
    y_test  = y[y.index >= SPLIT_DATE]

    print(f"  Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    
    #XGBoost model trained on 2020–2024, tested on 2025–2026
    model = xgb.XGBRegressor(
        n_estimators          = 800,
        learning_rate         = 0.04,
        max_depth             = 6,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        min_child_weight      = 5,
        reg_lambda            = 1.0,
        random_state          = 42,
        n_jobs                = -1,
        tree_method           = 'hist',
        early_stopping_rounds = 30,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    joblib.dump(model, f'model_{region}.pkl')

    region_df['Predicted_Price'] = model.predict(X)
    test_preds                   = model.predict(X_test)

    mae  = mean_absolute_error(y_test, test_preds)
    rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    print(f"  MAE: €{mae:.2f}/MWh   RMSE: €{rmse:.2f}/MWh")

    metrics_list.append({'Region': region, 'MAE': round(mae, 2), 'RMSE': round(rmse, 2),
                         'Train_Rows': len(X_train), 'Test_Rows': len(X_test)})

    # Feature importance chart
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    fig, ax     = plt.subplots(figsize=(10, 7))
    importances.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title(f'Feature Importance — {region}', fontsize=14, fontweight='bold')
    ax.set_xlabel('XGBoost Importance Score')
    plt.tight_layout()
    plt.savefig(f'{GRAPHS_DIR}/Feature_Importance_{region}.png', dpi=300)
    plt.close()

    region_df['Actual_Price'] = region_df['Target']
    all_results.append(region_df[['Region', 'Fossil', 'Biomass', 'Water', 'Sun',
                                   'Actual_Price', 'Predicted_Price']])

pd.concat(all_results).to_csv('master_forecast_results.csv')
pd.DataFrame(metrics_list).to_csv('model_metrics.csv', index=False)

print("\nMetrics Summary:")
print(pd.DataFrame(metrics_list).to_string(index=False))
print("\nDone. master_forecast_results.csv and model_metrics.csv saved.")