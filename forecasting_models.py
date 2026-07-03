"""
Comprehensive Forecasting Models for Ferry Ticket Demand
Includes baseline, ML, and time-series models with multi-horizon forecasting
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("FORECASTING MODELS IMPLEMENTATION")
print("=" * 60)

# Load engineered features
df = pd.read_csv('ferry_tickets_features.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp').reset_index(drop=True)

print(f"Dataset shape: {df.shape}")

# Define forecast horizons (in 15-minute intervals) - reduced for CPU
horizons = {
    '15min': 1,
    '1hr': 4,
    '2hr': 8
}

# Train-test split (time-based) - use smaller subset for CPU
# Use last 50k records for faster training
df_subset = df.iloc[-100000:].copy()
train_size = int(len(df_subset) * 0.8)
train_df = df_subset.iloc[:train_size].copy()
test_df = df_subset.iloc[train_size:].copy()

print(f"\nUsing subset for CPU efficiency: {len(df_subset)} records")
print(f"Train size: {len(train_df)}")
print(f"Test size: {len(test_df)}")

# Define feature columns (exclude target and timestamp) - use only key features for CPU
exclude_cols = ['Timestamp', 'Sales Count', 'Redemption Count', 'total_demand']
# Use only lag and temporal features for speed
feature_cols = [col for col in df.columns if col not in exclude_cols and 
                ('lag' in col or 'hour' in col or 'day' in col or 'month' in col or 
                 'weekend' in col or 'sin' in col or 'cos' in col)]

print(f"Number of features (optimized for CPU): {len(feature_cols)}")

# Results storage
results = {}

# ==================== BASELINE MODELS ====================
print("\n" + "=" * 60)
print("BASELINE MODELS")
print("=" * 60)

class BaselineModels:
    """Baseline forecasting models"""
    
    @staticmethod
    def naive_forecast(series, horizon):
        """Naïve forecast: last observed value"""
        return series.shift(horizon)
    
    @staticmethod
    def moving_average(series, window, horizon):
        """Moving average forecast"""
        return series.rolling(window=window).mean().shift(horizon)
    
    @staticmethod
    def linear_regression(X_train, y_train, X_test):
        """Linear regression with lag features"""
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model.predict(X_test)

# Evaluate baseline models
print("\nEvaluating Naïve Forecast...")
for horizon_name, horizon_steps in horizons.items():
    # Sales forecasting
    train_naive = BaselineModels.naive_forecast(train_df['Sales Count'], horizon_steps)
    test_naive = BaselineModels.naive_forecast(test_df['Sales Count'], horizon_steps)
    
    # Remove NaN
    valid_idx = test_naive.notna()
    y_true = test_df.loc[valid_idx, 'Sales Count']
    y_pred = test_naive[valid_idx]
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
    
    key = f'Naive_Sales_{horizon_name}'
    results[key] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    print(f"  {key}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")

print("\nEvaluating Moving Average (4-period window)...")
for horizon_name, horizon_steps in horizons.items():
    train_ma = BaselineModels.moving_average(train_df['Sales Count'], 4, horizon_steps)
    test_ma = BaselineModels.moving_average(test_df['Sales Count'], 4, horizon_steps)
    
    valid_idx = test_ma.notna()
    y_true = test_df.loc[valid_idx, 'Sales Count']
    y_pred = test_ma[valid_idx]
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
    
    key = f'MA_Sales_{horizon_name}'
    results[key] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    print(f"  {key}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")

# ==================== LINEAR REGRESSION ====================
print("\n" + "=" * 60)
print("LINEAR REGRESSION WITH LAG FEATURES")
print("=" * 60)

# Use only lag features for linear regression
lag_features = [col for col in feature_cols if 'lag' in col and 'sales' in col.lower()]

for horizon_name, horizon_steps in horizons.items():
    # Create target (shifted by horizon)
    y_train = train_df['Sales Count'].shift(-horizon_steps).dropna()
    y_test = test_df['Sales Count'].shift(-horizon_steps).dropna()
    
    # Align features
    X_train = train_df[lag_features].iloc[:len(y_train)]
    X_test = test_df[lag_features].iloc[:len(y_test)]
    
    # Train model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Predict
    y_pred = lr_model.predict(X_test)
    
    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100
    
    key = f'LR_Sales_{horizon_name}'
    results[key] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    print(f"  {key}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")

# ==================== RANDOM FOREST ====================
print("\n" + "=" * 60)
print("RANDOM FOREST REGRESSOR")
print("=" * 60)

for horizon_name, horizon_steps in horizons.items():
    # Create target
    y_train = train_df['Sales Count'].shift(-horizon_steps).dropna()
    y_test = test_df['Sales Count'].shift(-horizon_steps).dropna()
    
    # Align features
    X_train = train_df[feature_cols].iloc[:len(y_train)]
    X_test = test_df[feature_cols].iloc[:len(y_test)]
    
    # Train model - optimized for CPU
    rf_model = RandomForestRegressor(n_estimators=30, max_depth=8, 
                                     n_jobs=-1, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Predict
    y_pred = rf_model.predict(X_test)
    
    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100
    
    key = f'RF_Sales_{horizon_name}'
    results[key] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    print(f"  {key}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")

# ==================== GRADIENT BOOSTING ====================
print("\n" + "=" * 60)
print("GRADIENT BOOSTING REGRESSOR")
print("=" * 60)

for horizon_name, horizon_steps in horizons.items():
    # Create target
    y_train = train_df['Sales Count'].shift(-horizon_steps).dropna()
    y_test = test_df['Sales Count'].shift(-horizon_steps).dropna()
    
    # Align features
    X_train = train_df[feature_cols].iloc[:len(y_train)]
    X_test = test_df[feature_cols].iloc[:len(y_test)]
    
    # Train model - optimized for CPU
    gb_model = GradientBoostingRegressor(n_estimators=30, max_depth=5, 
                                         random_state=42)
    gb_model.fit(X_train, y_train)
    
    # Predict
    y_pred = gb_model.predict(X_test)
    
    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100
    
    key = f'GB_Sales_{horizon_name}'
    results[key] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    print(f"  {key}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")

# ==================== XGBOOST ====================
print("\n" + "=" * 60)
print("XGBOOST REGRESSOR")
print("=" * 60)

for horizon_name, horizon_steps in horizons.items():
    # Create target
    y_train = train_df['Sales Count'].shift(-horizon_steps).dropna()
    y_test = test_df['Sales Count'].shift(-horizon_steps).dropna()
    
    # Align features
    X_train = train_df[feature_cols].iloc[:len(y_train)]
    X_test = test_df[feature_cols].iloc[:len(y_test)]
    
    # Train model - optimized for CPU
    xgb_model = xgb.XGBRegressor(n_estimators=30, max_depth=5, 
                                  learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    
    # Predict
    y_pred = xgb_model.predict(X_test)
    
    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100
    
    key = f'XGB_Sales_{horizon_name}'
    results[key] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    print(f"  {key}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")

# Save results
results_df = pd.DataFrame(results).T
results_df.to_csv('model_results.csv')
print("\n" + "=" * 60)
print("MODEL RESULTS SAVED TO: model_results.csv")
print("=" * 60)

# Display summary
print("\n" + "=" * 60)
print("MODEL PERFORMANCE SUMMARY (MAE)")
print("=" * 60)
summary = results_df[['MAE']].sort_values('MAE')
print(summary)

print("\n" + "=" * 60)
print("BEST MODEL BY HORIZON")
print("=" * 60)
for horizon in horizons.keys():
    horizon_results = results_df[results_df.index.str.contains(horizon)]
    best = horizon_results['MAE'].idxmin()
    print(f"{horizon}: {best} (MAE={results_df.loc[best, 'MAE']:.2f})")
