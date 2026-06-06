# Electricity Price Forecasting
LEI Research Internship — Kaunas, Lithuania (2026)

## What it does
Forecasts Day-Ahead electricity market prices across Lithuania,
Germany, and Scandinavia using a Random Forest ML model trained
on 6 years of market data (2020–2026).

## Results
- MAPE of 14–18% across 3 regional markets
- Tracked the 2022 energy crisis price spikes (up to €600/MWh)
- Validated morning and evening demand peaks accurately

## Tech Stack
Python · Pandas · Scikit-learn · NumPy · SHAP · FastAPI · Matplotlib · ENTSO-E API

## Project Structure
- `*.py` — data pipeline, model training, forecasting, visualisation
- `Price_Predictions_2026` — predictions of prices csv file
- `Daily_Profile_()_2026` - Actual vs AI detected prices graph
