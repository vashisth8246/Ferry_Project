"""
Forecast Uncertainty Analysis and Prediction Intervals
CPU-optimized implementation
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("FORECAST UNCERTAINTY ANALYSIS")
print("=" * 60)

# Load engineered features
df = pd.read_csv('ferry_tickets_features.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp').reset_index(drop=True)

# Use smaller subset for CPU
df_subset = df.iloc[-50000:].copy()
train_size = int(len(df_subset) * 0.8)
train_df = df_subset.iloc[:train_size].copy()
test_df = df_subset.iloc[train_size:].copy()

print(f"Using subset: {len(df_subset)} records")
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# Define features (optimized)
exclude_cols = ['Timestamp', 'Sales Count', 'Redemption Count', 'total_demand']
feature_cols = [col for col in df.columns if col not in exclude_cols and 
                ('lag' in col or 'hour' in col or 'day' in col or 'month' in col or 
                 'weekend' in col or 'sin' in col or 'cos' in col)]

print(f"Features: {len(feature_cols)}")

# ==================== PREDICTION INTERVALS USING RANDOM FOREST ====================
print("\nCalculating prediction intervals using Random Forest...")

# Train model for 1-hour forecast
horizon_steps = 4
y_train = train_df['Sales Count'].shift(-horizon_steps).dropna()
y_test = test_df['Sales Count'].shift(-horizon_steps).dropna()

X_train = train_df[feature_cols].iloc[:len(y_train)]
X_test = test_df[feature_cols].iloc[:len(y_test)]

# Train Random Forest with more trees for better uncertainty estimation
rf_model = RandomForestRegressor(n_estimators=50, max_depth=8, 
                                  n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train)

# Get predictions from individual trees for uncertainty
predictions = np.array([tree.predict(X_test) for tree in rf_model.estimators_])

# Calculate statistics
mean_pred = np.mean(predictions, axis=0)
std_pred = np.std(predictions, axis=0)
lower_90 = np.percentile(predictions, 5, axis=0)
upper_90 = np.percentile(predictions, 95, axis=0)
lower_50 = np.percentile(predictions, 25, axis=0)
upper_50 = np.percentile(predictions, 75, axis=0)

# Evaluate
mae = np.mean(np.abs(y_test - mean_pred))
rmse = np.sqrt(np.mean((y_test - mean_pred)**2))

print(f"Mean Prediction MAE: {mae:.2f}")
print(f"Mean Prediction RMSE: {rmse:.2f}")
print(f"Average 90% CI Width: {np.mean(upper_90 - lower_90):.2f}")
print(f"Average 50% CI Width: {np.mean(upper_50 - lower_50):.2f}")

# Calculate coverage
coverage_90 = np.mean((y_test >= lower_90) & (y_test <= upper_90))
coverage_50 = np.mean((y_test >= lower_50) & (y_test <= upper_50))

print(f"90% CI Coverage: {coverage_90*100:.1f}%")
print(f"50% CI Coverage: {coverage_50*100:.1f}%")

# Save uncertainty results
uncertainty_df = pd.DataFrame({
    'actual': y_test.values,
    'predicted': mean_pred,
    'lower_90': lower_90,
    'upper_90': upper_90,
    'lower_50': lower_50,
    'upper_50': upper_50,
    'std': std_pred
})
uncertainty_df.to_csv('uncertainty_results.csv', index=False)
print("\nUncertainty results saved to: uncertainty_results.csv")

# ==================== VISUALIZATION ====================
print("\nCreating uncertainty visualization...")

import os
os.makedirs('uncertainty_plots', exist_ok=True)

# Sample for visualization (first 500 points)
sample_size = min(500, len(uncertainty_df))
uncertainty_sample = uncertainty_df.iloc[:sample_size]

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Predictions with 90% confidence intervals
axes[0].plot(range(sample_size), uncertainty_sample['actual'], 
             label='Actual', alpha=0.7, linewidth=1.5, color='blue')
axes[0].plot(range(sample_size), uncertainty_sample['predicted'], 
             label='Predicted', alpha=0.7, linewidth=1.5, color='red')
axes[0].fill_between(range(sample_size), uncertainty_sample['lower_90'], 
                     uncertainty_sample['upper_90'], alpha=0.2, color='red', 
                     label='90% CI')
axes[0].set_title('Ferry Sales Forecast with 90% Prediction Intervals', 
                  fontsize=14, fontweight='bold')
axes[0].set_xlabel('Time Step', fontsize=12)
axes[0].set_ylabel('Sales Count', fontsize=12)
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot 2: Prediction uncertainty over time
axes[1].plot(range(sample_size), uncertainty_sample['std'], 
             color='purple', linewidth=1.5, label='Prediction Std Dev')
axes[1].set_title('Forecast Uncertainty (Standard Deviation)', 
                  fontsize=14, fontweight='bold')
axes[1].set_xlabel('Time Step', fontsize=12)
axes[1].set_ylabel('Standard Deviation', fontsize=12)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('uncertainty_plots/prediction_intervals.png', dpi=300, bbox_inches='tight')
print("✓ Saved: uncertainty_plots/prediction_intervals.png")
plt.close()

# Error distribution analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

errors = uncertainty_sample['actual'] - uncertainty_sample['predicted']

axes[0].hist(errors, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[0].set_title('Forecast Error Distribution', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Error (Actual - Predicted)', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].grid(alpha=0.3)

# Q-Q plot
from scipy import stats
stats.probplot(errors, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot of Forecast Errors', fontsize=13, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('uncertainty_plots/error_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: uncertainty_plots/error_distribution.png")
plt.close()

# ==================== UNCERTAINTY METRICS ====================
print("\n" + "=" * 60)
print("UNCERTAINTY ANALYSIS SUMMARY")
print("=" * 60)

# Calculate additional metrics
mean_absolute_scaled_error = mae / np.mean(np.abs(np.diff(y_test)))
print(f"Mean Absolute Scaled Error (MASE): {mean_absolute_scaled_error:.4f}")

# Peak demand detection
peak_threshold = np.percentile(y_test, 90)
peak_mask = y_test > peak_threshold
peak_mae = np.mean(np.abs(y_test[peak_mask] - mean_pred[peak_mask]))
print(f"Peak Demand MAE (90th percentile): {peak_mae:.2f}")

# Low demand detection
low_threshold = np.percentile(y_test, 10)
low_mask = y_test < low_threshold
low_mae = np.mean(np.abs(y_test[low_mask] - mean_pred[low_mask]))
print(f"Low Demand MAE (10th percentile): {low_mae:.2f}")

# Save summary
summary = {
    'mae': float(mae),
    'rmse': float(rmse),
    'coverage_90': float(coverage_90),
    'coverage_50': float(coverage_50),
    'avg_ci_width_90': float(np.mean(upper_90 - lower_90)),
    'avg_ci_width_50': float(np.mean(upper_50 - lower_50)),
    'mase': float(mean_absolute_scaled_error),
    'peak_demand_mae': float(peak_mae),
    'low_demand_mae': float(low_mae)
}

import json
with open('uncertainty_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\nUncertainty summary saved to: uncertainty_summary.json")
print(json.dumps(summary, indent=2))

print("\n" + "=" * 60)
print("UNCERTAINTY ANALYSIS COMPLETE")
print("=" * 60)
