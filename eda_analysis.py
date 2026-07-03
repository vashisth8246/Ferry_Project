"""
Exploratory Data Analysis (EDA) for Ferry Ticket Demand
Comprehensive visualization and pattern analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load cleaned data
df = pd.read_csv('ferry_tickets_cleaned.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp').reset_index(drop=True)

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Create output directory for plots
import os
os.makedirs('eda_plots', exist_ok=True)

# 1. Time Series Plot - Overall Trends
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# Sample data for readability (every 100th point)
sample_idx = np.arange(0, len(df), 100)
df_sample = df.iloc[sample_idx]

axes[0].plot(df_sample['Timestamp'], df_sample['Sales Count'], alpha=0.6, linewidth=0.5)
axes[0].set_title('Ferry Ticket Sales Over Time (Sampled)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Sales Count', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)

axes[1].plot(df_sample['Timestamp'], df_sample['Redemption Count'], alpha=0.6, linewidth=0.5, color='orange')
axes[1].set_title('Ferry Ticket Redemptions Over Time (Sampled)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Redemption Count', fontsize=12)
axes[1].set_xlabel('Timestamp', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('eda_plots/01_time_series_overview.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_plots/01_time_series_overview.png")
plt.close()

# 2. Hourly Patterns
df['hour'] = df['Timestamp'].dt.hour
hourly_stats = df.groupby('hour').agg({
    'Sales Count': ['mean', 'median', 'std'],
    'Redemption Count': ['mean', 'median', 'std']
}).round(2)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Hourly mean sales
hourly_sales = df.groupby('hour')['Sales Count'].mean()
axes[0, 0].bar(hourly_sales.index, hourly_sales.values, color='steelblue', alpha=0.7)
axes[0, 0].set_title('Average Sales by Hour of Day', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Hour', fontsize=11)
axes[0, 0].set_ylabel('Average Sales', fontsize=11)
axes[0, 0].grid(axis='y', alpha=0.3)

# Hourly mean redemptions
hourly_redemptions = df.groupby('hour')['Redemption Count'].mean()
axes[0, 1].bar(hourly_redemptions.index, hourly_redemptions.values, color='coral', alpha=0.7)
axes[0, 1].set_title('Average Redemptions by Hour of Day', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Hour', fontsize=11)
axes[0, 1].set_ylabel('Average Redemptions', fontsize=11)
axes[0, 1].grid(axis='y', alpha=0.3)

# Hourly boxplot for sales
sales_by_hour = [df[df['hour'] == h]['Sales Count'].values for h in range(24)]
bp1 = axes[1, 0].boxplot(sales_by_hour, labels=range(24), patch_artist=True)
for patch in bp1['boxes']:
    patch.set_facecolor('lightblue')
axes[1, 0].set_title('Sales Distribution by Hour', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Hour', fontsize=11)
axes[1, 0].set_ylabel('Sales Count', fontsize=11)
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].set_ylim([0, 200])

# Hourly boxplot for redemptions
redemptions_by_hour = [df[df['hour'] == h]['Redemption Count'].values for h in range(24)]
bp2 = axes[1, 1].boxplot(redemptions_by_hour, labels=range(24), patch_artist=True)
for patch in bp2['boxes']:
    patch.set_facecolor('lightcoral')
axes[1, 1].set_title('Redemptions Distribution by Hour', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Hour', fontsize=11)
axes[1, 1].set_ylabel('Redemption Count', fontsize=11)
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].set_ylim([0, 200])

plt.tight_layout()
plt.savefig('eda_plots/02_hourly_patterns.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_plots/02_hourly_patterns.png")
plt.close()

# 3. Day of Week Patterns
df['day_of_week'] = df['Timestamp'].dt.dayofweek
df['day_name'] = df['Timestamp'].dt.day_name()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df['day_name'] = pd.Categorical(df['day_name'], categories=day_order, ordered=True)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Daily mean sales
daily_sales = df.groupby('day_name')['Sales Count'].mean()
axes[0, 0].bar(range(len(daily_sales)), daily_sales.values, color='steelblue', alpha=0.7)
axes[0, 0].set_xticks(range(len(daily_sales)))
axes[0, 0].set_xticklabels(daily_sales.index, rotation=45)
axes[0, 0].set_title('Average Sales by Day of Week', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('Average Sales', fontsize=11)
axes[0, 0].grid(axis='y', alpha=0.3)

# Daily mean redemptions
daily_redemptions = df.groupby('day_name')['Redemption Count'].mean()
axes[0, 1].bar(range(len(daily_redemptions)), daily_redemptions.values, color='coral', alpha=0.7)
axes[0, 1].set_xticks(range(len(daily_redemptions)))
axes[0, 1].set_xticklabels(daily_redemptions.index, rotation=45)
axes[0, 1].set_title('Average Redemptions by Day of Week', fontsize=13, fontweight='bold')
axes[0, 1].set_ylabel('Average Redemptions', fontsize=11)
axes[0, 1].grid(axis='y', alpha=0.3)

# Sales heatmap by hour and day
heatmap_data_sales = df.groupby(['day_name', 'hour'])['Sales Count'].mean().unstack()
sns.heatmap(heatmap_data_sales, cmap='YlOrRd', ax=axes[1, 0], cbar_kws={'label': 'Average Sales'})
axes[1, 0].set_title('Sales Heatmap: Hour vs Day of Week', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Hour', fontsize=11)
axes[1, 0].set_ylabel('Day of Week', fontsize=11)

# Redemptions heatmap by hour and day
heatmap_data_redemptions = df.groupby(['day_name', 'hour'])['Redemption Count'].mean().unstack()
sns.heatmap(heatmap_data_redemptions, cmap='YlOrRd', ax=axes[1, 1], cbar_kws={'label': 'Average Redemptions'})
axes[1, 1].set_title('Redemptions Heatmap: Hour vs Day of Week', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Hour', fontsize=11)
axes[1, 1].set_ylabel('Day of Week', fontsize=11)

plt.tight_layout()
plt.savefig('eda_plots/03_daily_patterns.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_plots/03_daily_patterns.png")
plt.close()

# 4. Monthly Seasonality
df['month'] = df['Timestamp'].dt.month
df['month_name'] = df['Timestamp'].dt.month_name()
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
df['month_name'] = pd.Categorical(df['month_name'], categories=month_order, ordered=True)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

monthly_sales = df.groupby('month_name')['Sales Count'].mean()
axes[0].bar(range(len(monthly_sales)), monthly_sales.values, color='steelblue', alpha=0.7)
axes[0].set_xticks(range(len(monthly_sales)))
axes[0].set_xticklabels(monthly_sales.index, rotation=45)
axes[0].set_title('Average Sales by Month', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Average Sales', fontsize=12)
axes[0].grid(axis='y', alpha=0.3)

monthly_redemptions = df.groupby('month_name')['Redemption Count'].mean()
axes[1].bar(range(len(monthly_redemptions)), monthly_redemptions.values, color='coral', alpha=0.7)
axes[1].set_xticks(range(len(monthly_redemptions)))
axes[1].set_xticklabels(monthly_redemptions.index, rotation=45)
axes[1].set_title('Average Redemptions by Month', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Average Redemptions', fontsize=12)
axes[1].set_xlabel('Month', fontsize=12)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('eda_plots/04_monthly_seasonality.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_plots/04_monthly_seasonality.png")
plt.close()

# 5. Distribution Analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Sales distribution
axes[0, 0].hist(df['Sales Count'], bins=100, color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Distribution of Sales Count', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Sales Count', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_xlim([0, 500])

# Redemptions distribution
axes[0, 1].hist(df['Redemption Count'], bins=100, color='coral', alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Distribution of Redemption Count', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Redemption Count', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].set_xlim([0, 500])

# Log-transformed sales
axes[1, 0].hist(np.log1p(df['Sales Count']), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Log-Transformed Sales Distribution', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Log(Sales + 1)', fontsize=11)
axes[1, 0].set_ylabel('Frequency', fontsize=11)

# Log-transformed redemptions
axes[1, 1].hist(np.log1p(df['Redemption Count']), bins=50, color='coral', alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Log-Transformed Redemptions Distribution', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Log(Redemptions + 1)', fontsize=11)
axes[1, 1].set_ylabel('Frequency', fontsize=11)

plt.tight_layout()
plt.savefig('eda_plots/05_distribution_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_plots/05_distribution_analysis.png")
plt.close()

# 6. Correlation and Scatter Analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot
sample_idx = np.random.choice(len(df), min(10000, len(df)), replace=False)
df_sample_scatter = df.iloc[sample_idx]
axes[0].scatter(df_sample_scatter['Sales Count'], df_sample_scatter['Redemption Count'], 
                alpha=0.3, s=10, color='purple')
axes[0].plot([0, max(df['Sales Count'].max(), df['Redemption Count'].max())], 
             [0, max(df['Sales Count'].max(), df['Redemption Count'].max())], 
             'r--', linewidth=2, label='y = x')
axes[0].set_title('Sales vs Redemptions Scatter Plot', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Sales Count', fontsize=11)
axes[0].set_ylabel('Redemption Count', fontsize=11)
axes[0].set_xlim([0, 500])
axes[0].set_ylim([0, 500])
axes[0].legend()
axes[0].grid(alpha=0.3)

# Correlation matrix
corr_matrix = df[['Sales Count', 'Redemption Count']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, ax=axes[1], cbar_kws={'label': 'Correlation'})
axes[1].set_title('Correlation Matrix', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('eda_plots/06_correlation_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_plots/06_correlation_analysis.png")
plt.close()

# 7. Peak Demand Analysis
df['total_demand'] = df['Sales Count'] + df['Redemption Count']
peak_threshold = df['total_demand'].quantile(0.95)
peak_periods = df[df['total_demand'] > peak_threshold]

print("\n" + "=" * 60)
print("PEAK DEMAND ANALYSIS")
print("=" * 60)
print(f"95th percentile threshold: {peak_threshold:.2f}")
print(f"Number of peak periods: {len(peak_periods)}")
print(f"Percentage of peak periods: {len(peak_periods)/len(df)*100:.2f}%")

# Peak hours
peak_hours = peak_periods.groupby('hour').size()
print(f"\nTop 5 peak hours:")
print(peak_hours.nlargest(5))

# Peak days
peak_days = peak_periods.groupby('day_name').size()
print(f"\nTop 3 peak days:")
print(peak_days.nlargest(3))

# 8. Weekly Trend Analysis (last 8 weeks)
df['week'] = df['Timestamp'].dt.isocalendar().week
df['year'] = df['Timestamp'].dt.year
recent_weeks = df[df['year'] >= 2025].copy()
if len(recent_weeks) > 0:
    weekly_sales = recent_weeks.groupby(['year', 'week'])['Sales Count'].mean().reset_index()
    weekly_sales['period'] = weekly_sales['year'].astype(str) + '-W' + weekly_sales['week'].astype(str)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(range(len(weekly_sales)), weekly_sales['Sales Count'], 
            marker='o', linewidth=2, markersize=6, color='steelblue')
    ax.set_title('Weekly Average Sales Trend (Recent Data)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('Average Sales', fontsize=12)
    ax.grid(alpha=0.3)
    plt.xticks(range(len(weekly_sales)), weekly_sales['period'], rotation=45)
    plt.tight_layout()
    plt.savefig('eda_plots/07_weekly_trend.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: eda_plots/07_weekly_trend.png")
    plt.close()

# Save summary statistics
summary_stats = {
    'total_records': len(df),
    'date_range_start': str(df['Timestamp'].min()),
    'date_range_end': str(df['Timestamp'].max()),
    'avg_sales': df['Sales Count'].mean(),
    'avg_redemptions': df['Redemption Count'].mean(),
    'max_sales': df['Sales Count'].max(),
    'max_redemptions': df['Redemption Count'].max(),
    'peak_hour_sales': hourly_sales.idxmax(),
    'peak_hour_redemptions': hourly_redemptions.idxmax(),
    'peak_day_sales': daily_sales.idxmax(),
    'peak_day_redemptions': daily_redemptions.idxmax(),
    'sales_redemption_correlation': float(df['Sales Count'].corr(df['Redemption Count']))
}

import json
with open('eda_plots/summary_statistics.json', 'w') as f:
    json.dump(summary_stats, f, indent=2, default=str)

print("\n" + "=" * 60)
print("EDA COMPLETE - Summary Statistics Saved")
print("=" * 60)
print(json.dumps(summary_stats, indent=2, default=str))
