"""
Time-Series Models (ARIMA/SARIMA, Prophet) - CPU Optimized
Uses smaller dataset for faster training on i3 CPU
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TIME-SERIES MODELS (CPU OPTIMIZED)")
print("=" * 60)

# Load cleaned data
df = pd.read_csv('ferry_tickets_cleaned.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp').reset_index(drop=True)

# Use smaller subset for CPU efficiency (last 20k records for time-series models)
df_subset = df.iloc[-20000:].copy()
print(f"Using subset for CPU efficiency: {len(df_subset)} records")

# Set timestamp as index
df_subset = df_subset.set_index('Timestamp')

# Resample to hourly for faster ARIMA/Prophet training
df_hourly = df_subset.resample('H').sum()
print(f"Hourly resampled data: {len(df_hourly)} records")

# Train-test split
train_size = int(len(df_hourly) * 0.8)
train_df = df_hourly.iloc[:train_size]
test_df = df_hourly.iloc[train_size:]

print(f"\nTrain size: {len(train_df)}")
print(f"Test size: {len(test_df)}")

# Results storage
results = {}

# ==================== ARIMA MODEL ====================
print("\n" + "=" * 60)
print("ARIMA MODEL")
print("=" * 60)

# Use simple ARIMA(1,1,1) for speed
print("Training ARIMA(1,1,1) for Sales...")
try:
    arima_model = ARIMA(train_df['Sales Count'], order=(1,1,1))
    arima_fit = arima_model.fit()
    
    # Forecast for test period
    forecast_steps = len(test_df)
    arima_forecast = arima_fit.forecast(steps=forecast_steps)
    
    # Evaluate
    mae = mean_absolute_error(test_df['Sales Count'], arima_forecast)
    rmse = np.sqrt(mean_squared_error(test_df['Sales Count'], arima_forecast))
    mape = np.mean(np.abs((test_df['Sales Count'] - arima_forecast) / (test_df['Sales Count'] + 1))) * 100
    
    results['ARIMA_Sales'] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    print(f"ARIMA_Sales: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
except Exception as e:
    print(f"ARIMA failed: {e}")

# ==================== PROPHET MODEL ====================
print("\n" + "=" * 60)
print("FACEBOOK PROPHET MODEL")
print("=" * 60)

# Prepare data for Prophet
train_prophet = train_df.reset_index().rename(columns={'Timestamp': 'ds', 'Sales Count': 'y'})
test_prophet = test_df.reset_index().rename(columns={'Timestamp': 'ds', 'Sales Count': 'y'})

print("Training Prophet for Sales...")
try:
    prophet_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    prophet_model.fit(train_prophet)
    
    # Make future dataframe
    future_dates = prophet_model.make_future_dataframe(periods=len(test_df), freq='H')
    prophet_forecast = prophet_model.predict(future_dates)
    
    # Extract forecast for test period
    forecast_test = prophet_forecast.iloc[-len(test_df):]['yhat']
    
    # Evaluate
    mae = mean_absolute_error(test_df['Sales Count'], forecast_test)
    rmse = np.sqrt(mean_squared_error(test_df['Sales Count'], forecast_test))
    mape = np.mean(np.abs((test_df['Sales Count'] - forecast_test) / (test_df['Sales Count'] + 1))) * 100
    
    results['Prophet_Sales'] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    print(f"Prophet_Sales: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
except Exception as e:
    print(f"Prophet failed: {e}")

# Save time-series results
ts_results_df = pd.DataFrame(results).T
ts_results_df.to_csv('timeseries_results.csv')
print("\n" + "=" * 60)
print("TIME-SERIES RESULTS SAVED TO: timeseries_results.csv")
print("=" * 60)

# Combine with previous ML results
ml_results = pd.read_csv('model_results.csv', index_col=0)
all_results = pd.concat([ml_results, ts_results_df])
all_results.to_csv('all_model_results.csv')
print("Combined results saved to: all_model_results.csv")

print("\n" + "=" * 60)
print("ALL MODEL PERFORMANCE COMPARISON (MAE)")
print("=" * 60)
print(all_results[['MAE']].sort_values('MAE'))
