"""
Data Exploration and Quality Analysis for Ferry Ticket Dataset
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load dataset
df = pd.read_csv('Toronto Island Ferry Tickets.csv')

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"\nShape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nBasic statistics:\n{df.describe()}")

# Check for missing values
print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

# Convert timestamp
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp').reset_index(drop=True)

print("\n" + "=" * 60)
print("TIME RANGE ANALYSIS")
print("=" * 60)
print(f"Start date: {df['Timestamp'].min()}")
print(f"End date: {df['Timestamp'].max()}")
print(f"Total duration: {df['Timestamp'].max() - df['Timestamp'].min()}")
print(f"Number of 15-minute intervals: {len(df)}")

# Check for missing intervals
print("\n" + "=" * 60)
print("INTERVAL COMPLETENESS CHECK")
print("=" * 60)
expected_intervals = pd.date_range(start=df['Timestamp'].min(), 
                                    end=df['Timestamp'].max(), 
                                    freq='15min')
missing_intervals = expected_intervals.difference(df['Timestamp'])
print(f"Expected intervals: {len(expected_intervals)}")
print(f"Actual intervals: {len(df)}")
print(f"Missing intervals: {len(missing_intervals)}")

if len(missing_intervals) > 0:
    print(f"\nFirst 10 missing intervals:")
    print(missing_intervals[:10])

# Analyze sales and redemption patterns
print("\n" + "=" * 60)
print("SALES AND REDEMPTION STATISTICS")
print("=" * 60)
print(f"\nSales Count - Mean: {df['Sales Count'].mean():.2f}, Std: {df['Sales Count'].std():.2f}")
print(f"Sales Count - Min: {df['Sales Count'].min()}, Max: {df['Sales Count'].max()}")
print(f"\nRedemption Count - Mean: {df['Redemption Count'].mean():.2f}, Std: {df['Redemption Count'].std():.2f}")
print(f"Redemption Count - Min: {df['Redemption Count'].min()}, Max: {df['Redemption Count'].max()}")

# Check for negative values
print("\n" + "=" * 60)
print("DATA QUALITY CHECKS")
print("=" * 60)
print(f"Negative Sales Count: {(df['Sales Count'] < 0).sum()}")
print(f"Negative Redemption Count: {(df['Redemption Count'] < 0).sum()}")

# Hourly patterns
df['hour'] = df['Timestamp'].dt.hour
hourly_sales = df.groupby('hour')['Sales Count'].mean()
hourly_redemptions = df.groupby('hour')['Redemption Count'].mean()

print("\n" + "=" * 60)
print("HOURLY PATTERNS (Average by Hour)")
print("=" * 60)
print("\nAverage Sales by Hour:")
print(hourly_sales)
print("\nAverage Redemptions by Hour:")
print(hourly_redemptions)

# Day of week patterns
df['day_of_week'] = df['Timestamp'].dt.dayofweek
df['day_name'] = df['Timestamp'].dt.day_name()
daily_sales = df.groupby('day_name')['Sales Count'].mean()
daily_redemptions = df.groupby('day_name')['Redemption Count'].mean()

print("\n" + "=" * 60)
print("DAILY PATTERNS (Average by Day of Week)")
print("=" * 60)
print("\nAverage Sales by Day:")
print(daily_sales)
print("\nAverage Redemptions by Day:")
print(daily_redemptions)

# Save cleaned data
df_clean = df[['Timestamp', 'Sales Count', 'Redemption Count']].copy()
df_clean.to_csv('ferry_tickets_cleaned.csv', index=False)
print("\n" + "=" * 60)
print("CLEANED DATA SAVED TO: ferry_tickets_cleaned.csv")
print("=" * 60)
