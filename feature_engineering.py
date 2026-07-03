"""
Feature Engineering for Time-Series Forecasting
Creates lag features, rolling statistics, and temporal encodings
"""
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("FEATURE ENGINEERING FOR TIME-SERIES FORECASTING")
print("=" * 60)

# Load cleaned data
df = pd.read_csv('ferry_tickets_cleaned.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp').reset_index(drop=True)

print(f"Original data shape: {df.shape}")

# Handle missing intervals by creating complete time index
print("\nHandling missing intervals...")
full_range = pd.date_range(start=df['Timestamp'].min(), 
                           end=df['Timestamp'].max(), 
                           freq='15min')
df_full = pd.DataFrame({'Timestamp': full_range})
df_full = df_full.merge(df, on='Timestamp', how='left')
df_full['Sales Count'] = df_full['Sales Count'].fillna(0)
df_full['Redemption Count'] = df_full['Redemption Count'].fillna(0)

print(f"Complete time series shape: {df_full.shape}")
print(f"Filled missing intervals: {df_full['Sales Count'].isna().sum()}")

df = df_full.copy()

# Temporal Features
print("\nCreating temporal features...")
df['hour'] = df['Timestamp'].dt.hour
df['day_of_week'] = df['Timestamp'].dt.dayofweek
df['day_of_month'] = df['Timestamp'].dt.day
df['month'] = df['Timestamp'].dt.month
df['quarter'] = df['Timestamp'].dt.quarter
df['year'] = df['Timestamp'].dt.year
df['week_of_year'] = df['Timestamp'].dt.isocalendar().week
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Cyclical encoding for temporal features
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

print("✓ Temporal features created")

# Lag Features for Sales
print("\nCreating lag features for Sales...")
lag_periods = [1, 2, 4, 8, 12, 16, 24, 48, 96]  # 15min intervals: 1=15min, 4=1hr, 96=24hr
for lag in lag_periods:
    df[f'sales_lag_{lag}'] = df['Sales Count'].shift(lag)
print(f"✓ Created {len(lag_periods)} lag features for Sales")

# Lag Features for Redemptions
print("\nCreating lag features for Redemptions...")
for lag in lag_periods:
    df[f'redemptions_lag_{lag}'] = df['Redemption Count'].shift(lag)
print(f"✓ Created {len(lag_periods)} lag features for Redemptions")

# Rolling Statistics for Sales
print("\nCreating rolling statistics for Sales...")
windows = [4, 8, 12, 24, 48, 96]  # 1hr, 2hr, 3hr, 6hr, 12hr, 24hr
for window in windows:
    df[f'sales_rolling_mean_{window}'] = df['Sales Count'].rolling(window=window).mean()
    df[f'sales_rolling_std_{window}'] = df['Sales Count'].rolling(window=window).std()
    df[f'sales_rolling_max_{window}'] = df['Sales Count'].rolling(window=window).max()
    df[f'sales_rolling_min_{window}'] = df['Sales Count'].rolling(window=window).min()
print(f"✓ Created {len(windows) * 4} rolling statistics for Sales")

# Rolling Statistics for Redemptions
print("\nCreating rolling statistics for Redemptions...")
for window in windows:
    df[f'redemptions_rolling_mean_{window}'] = df['Redemption Count'].rolling(window=window).mean()
    df[f'redemptions_rolling_std_{window}'] = df['Redemption Count'].rolling(window=window).std()
    df[f'redemptions_rolling_max_{window}'] = df['Redemption Count'].rolling(window=window).max()
    df[f'redemptions_rolling_min_{window}'] = df['Redemption Count'].rolling(window=window).min()
print(f"✓ Created {len(windows) * 4} rolling statistics for Redemptions")

# Difference Features
print("\nCreating difference features...")
df['sales_diff_1'] = df['Sales Count'].diff(1)
df['sales_diff_4'] = df['Sales Count'].diff(4)
df['sales_diff_24'] = df['Sales Count'].diff(24)
df['redemptions_diff_1'] = df['Redemption Count'].diff(1)
df['redemptions_diff_4'] = df['Redemption Count'].diff(4)
df['redemptions_diff_24'] = df['Redemption Count'].diff(24)
print("✓ Created difference features")

# Total Demand Feature
df['total_demand'] = df['Sales Count'] + df['Redemption Count']

# Lag features for total demand
for lag in [1, 2, 4, 8, 12, 24, 48, 96]:
    df[f'demand_lag_{lag}'] = df['total_demand'].shift(lag)

# Rolling statistics for total demand
for window in [4, 8, 12, 24, 48, 96]:
    df[f'demand_rolling_mean_{window}'] = df['total_demand'].rolling(window=window).mean()
    df[f'demand_rolling_std_{window}'] = df['total_demand'].rolling(window=window).std()

print("✓ Created total demand features")

# Remove rows with NaN values (from lags and rolling windows)
print(f"\nOriginal rows: {len(df)}")
df = df.dropna()
print(f"Rows after removing NaN: {len(df)}")
print(f"Rows removed: {len(df_full) - len(df)}")

# Feature summary
feature_cols = [col for col in df.columns if col not in ['Timestamp', '_id']]
print(f"\nTotal features created: {len(feature_cols)}")
print(f"\nFeature categories:")
print(f"- Temporal features: 13")
print(f"- Lag features (Sales): {len(lag_periods)}")
print(f"- Lag features (Redemptions): {len(lag_periods)}")
print(f"- Lag features (Demand): {len(lag_periods)}")
print(f"- Rolling statistics (Sales): {len(windows) * 4}")
print(f"- Rolling statistics (Redemptions): {len(windows) * 4}")
print(f"- Rolling statistics (Demand): {len(windows) * 2}")
print(f"- Difference features: 6")
print(f"- Total demand: 1")

# Save engineered features
df.to_csv('ferry_tickets_features.csv', index=False)
print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 60)
print(f"Engineered dataset saved to: ferry_tickets_features.csv")
print(f"Final dataset shape: {df.shape}")
