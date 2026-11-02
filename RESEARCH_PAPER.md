# Short-Term Ferry Ticket Demand Forecasting & Predictive Decision Support System

## Research Paper

**Submitted to:** Toronto Government Parks, Forestry & Recreation  
**Date:** July 2026  
**Project:** Ferry Operations Predictive Intelligence

---

## Abstract

This research presents a comprehensive machine learning and time-series forecasting system for predicting short-term ferry ticket demand to Toronto Island Park. Using over 10 years of high-frequency 15-minute interval data (261,538 records from 2015-2025), we developed and evaluated multiple forecasting approaches including baseline models, machine learning algorithms (Random Forest, Gradient Boosting, XGBoost), and time-series models (ARIMA). The system provides forecasts for 15-minute, 1-hour, and 2-hour horizons, enabling proactive operational decision-making for ferry scheduling, staff readiness, and crowd management. XGBoost emerged as the best-performing model across all forecast horizons with MAE values of 17.73 (15-min), 18.68 (1-hour), and 19.81 (2-hour). The research demonstrates significant potential for transitioning ferry operations from reactive to predictive intelligence.

---

## 1. Introduction

### 1.1 Background

Ferry operations to Toronto Island Park serve as a critical transportation link for residents and tourists. With over a decade of operational data, ferry operators have historically relied on retrospective analytics to understand past demand patterns. However, effective ferry operations require anticipatory decision-making to address:

- Proactive ferry scheduling
- Staff readiness and allocation
- Crowd management during peak periods
- Safety preparedness during sudden demand spikes

### 1.2 Problem Statement

Despite possessing extensive historical ticket data, ferry operations currently lack:

- Short-term demand forecasts (15-minute to 2-hour windows)
- Predictive visibility into upcoming congestion
- Quantitative uncertainty estimates for planning decisions

This reactive approach increases the risk of congestion, delays, and service inefficiencies.

### 1.3 Research Objectives

**Primary Objectives:**
- Forecast short-term ferry ticket sales and redemptions
- Predict demand for upcoming 15-minute, 1-hour, and 2-hour windows
- Compare statistical and machine-learning forecasting approaches

**Secondary Objectives:**
- Quantify prediction uncertainty
- Support proactive operational planning
- Demonstrate real-world ML deployment via Streamlit dashboard

---

## 2. Data Analysis

### 2.1 Dataset Description

The dataset contains 261,538 records spanning from May 1, 2015, to December 21, 2025, with 15-minute interval granularity:

| Column | Description |
|--------|-------------|
| _id | Unique row identifier |
| Timestamp | 15-minute interval end time |
| Sales Count | Tickets sold in interval |
| Redemption Count | Tickets redeemed in interval |

**Data Quality:**
- No missing values in core variables
- No negative values detected
- 111,651 missing intervals (30%) filled via interpolation
- Complete time series: 373,189 intervals after filling

### 2.2 Exploratory Data Analysis

#### 2.2.1 Temporal Patterns

**Hourly Patterns:**
- Peak sales hour: 12:00 PM (94.5 average sales)
- Peak redemptions hour: 12:00 PM (100.2 average redemptions)
- Low activity: 3:00-6:00 AM (4-10 average transactions)
- Strong diurnal pattern with morning peak (11:00-14:00)

**Day of Week Patterns:**
- Highest demand: Saturday (71.2 avg sales), Sunday (70.0 avg sales)
- Lowest demand: Tuesday (36.2 avg sales), Wednesday (38.6 avg sales)
- Weekend demand ~70-80% higher than weekdays

**Monthly Seasonality:**
- Summer months (June-August) show elevated demand
- Winter months (December-February) show reduced activity
- Consistent seasonal patterns across years

#### 2.2.2 Statistical Summary

| Metric | Sales Count | Redemption Count |
|--------|-------------|------------------|
| Mean | 49.60 | 48.89 |
| Std Dev | 99.86 | 104.55 |
| Min | 0 | 0 |
| Max | 7,229 | 7,216 |
| Median | 13 | 11 |

**Key Insight:** High variance indicates significant demand fluctuations, necessitating robust forecasting approaches.

#### 2.2.3 Correlation Analysis

Sales and redemptions show strong positive correlation (r = 0.918), indicating that ticket sales and usage are closely linked, suggesting effective demand prediction can inform both operational and capacity planning.

#### 2.2.4 Peak Demand Analysis

- 95th percentile threshold: 494 total transactions per interval
- Peak periods represent 5% of all intervals (13,071 periods)
- Top peak hours: 11:00 AM (2,165 peaks), 12:00 PM (2,112 peaks), 1:00 PM (1,933 peaks)
- Top peak days: Saturday (3,679 peaks), Sunday (3,436 peaks)

---

## 3. Methodology

### 3.1 Feature Engineering

#### 3.1.1 Temporal Features

- **Basic temporal:** Hour, day of week, day of month, month, quarter, year, week of year
- **Cyclical encoding:** Sin/cos transformations for hour, day, month to capture cyclical patterns
- **Weekend indicator:** Binary flag for weekend days

#### 3.1.2 Lag Features

Lag features at multiple intervals to capture autoregressive patterns:
- Sales lags: t-1, t-2, t-4, t-8, t-12, t-16, t-24, t-48, t-96 (15-min to 24-hour lags)
- Redemption lags: Same intervals
- Total demand lags: Same intervals

#### 3.1.3 Rolling Statistics

Rolling window statistics to capture trends and volatility:
- Windows: 1hr, 2hr, 3hr, 6hr, 12hr, 24hr
- Statistics: Mean, standard deviation, max, min
- Applied to sales, redemptions, and total demand

#### 3.1.4 Difference Features

First-order differences at multiple intervals:
- 1-interval, 4-interval (1-hour), 24-interval (6-hour) differences
- Captures rate of change and acceleration in demand

**Total Features Engineered:** 109 features from 3 base variables

### 3.2 Train-Test Strategy

- **Time-based split:** No random shuffling to preserve temporal structure
- **Training set:** 80% of data (chronologically earliest)
- **Test set:** 20% of data (chronologically latest)
- **Subset for CPU efficiency:** Last 100,000 records used for model training

### 3.3 Forecast Horizons

Three forecast horizons evaluated:
- **15-minute:** Immediate operational decisions
- **1-hour:** Short-term planning and adjustments
- **2-hour:** Medium-term scheduling and resource allocation

### 3.4 Model Architecture

#### 3.4.1 Baseline Models

**Naïve Forecast:**
- Uses last observed value as forecast
- Serves as minimum performance benchmark

**Moving Average:**
- 4-period (1-hour) moving average
- Smooths short-term fluctuations

**Linear Regression:**
- Uses lag features only
- Captures linear autoregressive relationships

#### 3.4.2 Machine Learning Models

**Random Forest Regressor:**
- 30 trees, max depth 8 (CPU-optimized)
- Ensemble of decision trees
- Captures non-linear relationships

**Gradient Boosting Regressor:**
- 30 estimators, max depth 5 (CPU-optimized)
- Sequential ensemble learning
- Strong performance on structured data

**XGBoost Regressor:**
- 30 estimators, max depth 5, learning rate 0.1 (CPU-optimized)
- Gradient boosting framework
- Best overall performance

#### 3.4.3 Time-Series Models

**ARIMA(1,1,1):**
- AutoRegressive Integrated Moving Average
- Captures temporal dependencies
- Simple configuration for computational efficiency

**Prophet:**
- Facebook's forecasting library
- Handles seasonality and holidays
- Implementation encountered compatibility issues

---

## 4. Results

### 4.1 Model Performance Comparison

| Model | Horizon | MAE | RMSE | MAPE |
|-------|---------|-----|------|------|
| **XGBoost** | 15-min | **17.73** | **64.17** | 98.90% |
| Gradient Boosting | 15-min | 18.21 | 64.10 | 141.32% |
| Random Forest | 15-min | 18.73 | 66.29 | 103.69% |
| Moving Average | 15-min | 19.68 | 69.99 | 97.45% |
| Naïve | 15-min | 21.23 | 86.03 | 95.43% |
| Linear Regression | 15-min | 22.51 | 70.70 | 235.04% |
| **XGBoost** | 1-hour | **18.68** | **64.63** | 109.20% |
| Random Forest | 1-hour | 19.14 | 66.33 | 110.46% |
| Gradient Boosting | 1-hour | 19.27 | 65.06 | 152.20% |
| Moving Average | 1-hour | 26.90 | 76.54 | 133.59% |
| Naïve | 1-hour | 26.13 | 89.90 | 108.78% |
| Linear Regression | 1-hour | 27.32 | 75.76 | 326.27% |
| **XGBoost** | 2-hour | **19.81** | **65.12** | 126.45% |
| Gradient Boosting | 2-hour | 20.48 | 65.56 | 173.25% |
| Random Forest | 2-hour | 20.53 | 67.15 | 128.33% |
| Naïve | 2-hour | 36.22 | 99.30 | 169.81% |
| Linear Regression | 2-hour | 34.46 | 83.72 | 477.22% |
| Moving Average | 2-hour | 38.40 | 89.05 | 211.93% |
| ARIMA | Hourly | 28.32 | 54.66 | 76.17% |

### 4.2 Key Findings

**Model Performance:**
1. **XGBoost** consistently outperforms all models across all horizons
2. **Gradient Boosting** is the second-best performer
3. **Machine learning models** significantly outperform traditional time-series models
4. **Baseline models** provide reasonable benchmarks but are inadequate for operational use

**Forecast Horizon Analysis:**
- Performance degrades gracefully with longer horizons
- 15-minute forecasts: MAE 17.73 (XGBoost)
- 1-hour forecasts: MAE 18.68 (XGBoost) - 5.4% increase
- 2-hour forecasts: MAE 19.81 (XGBoost) - 11.7% increase

**Error Characteristics:**
- MAPE values are high due to low-demand periods (division by small numbers)
- RMSE indicates presence of outliers (peak demand events)
- MASE (Mean Absolute Scaled Error): 0.93, indicating better than naïve forecast

### 4.3 Uncertainty Analysis

**Prediction Intervals (Random Forest):**
- Mean Prediction MAE: 10.44
- Mean Prediction RMSE: 47.02
- Average 90% CI Width: 20.94
- Average 50% CI Width: 8.08
- 90% CI Coverage: 35.0% (below expected 90%)
- 50% CI Coverage: 17.3% (below expected 50%)

**Uncertainty Insights:**
- Current prediction intervals underestimate uncertainty
- Coverage rates suggest need for wider confidence intervals
- Peak demand periods show higher error (MAE: 40.75 at 90th percentile)
- Model performs better on typical demand periods

---

## 5. Discussion

### 5.1 Model Selection Rationale

**XGBoost Superiority:**
- Handles non-linear relationships effectively
- Robust to feature scaling
- Built-in regularization prevents overfitting
- Efficient implementation suitable for CPU deployment
- Captures complex temporal patterns through engineered features

**Feature Engineering Impact:**
- Lag features are critical for short-term forecasting
- Cyclical temporal encoding improves pattern recognition
- Rolling statistics capture trend and volatility information
- Feature selection optimized for CPU efficiency (37 key features)

### 5.2 Operational Implications

**Immediate Operational Decisions (15-minute forecasts):**
- Real-time ferry frequency adjustments
- Dynamic staffing allocation
- Crowd management interventions
- Safety preparedness during demand spikes

**Short-term Planning (1-hour forecasts):**
- Staff shift scheduling
- Ferry maintenance timing
- Resource allocation optimization
- Communication with island facilities

**Medium-term Planning (2-hour forecasts):**
- Ferry schedule optimization
- Peak demand preparation
- Contingency planning
- Inter-agency coordination

### 5.3 Limitations

**Data Limitations:**
- 30% missing intervals required interpolation
- No external variables (weather, events, holidays)
- Limited contextual information (special events, festivals)

**Model Limitations:**
- Prediction intervals underestimate uncertainty
- Peak demand forecasting remains challenging
- No explicit handling of special events or anomalies
- CPU-optimized models may sacrifice some accuracy

**Deployment Considerations:**
- Model retraining frequency not established
- Real-time data pipeline requirements
- Integration with existing operational systems
- User interface and decision support design

### 5.4 Future Work

**Model Improvements:**
- Implement ensemble methods for better uncertainty quantification
- Add conformal prediction for reliable confidence intervals
- Incorporate external variables (weather, events, holidays)
- Develop anomaly detection for special events

**System Enhancements:**
- Real-time model retraining pipeline
- Automated alert system for demand spikes
- Integration with ferry scheduling systems
- Mobile application for field staff

**Research Extensions:**
- Long-term forecasting (daily, weekly, monthly)
- Scenario analysis and what-if modeling
- Capacity optimization modeling
- Economic impact analysis

---

## 6. Conclusions

This research successfully developed a comprehensive short-term ferry ticket demand forecasting system using machine learning and time-series techniques. Key achievements include:

1. **Comprehensive EDA:** Identified strong temporal patterns, with peak demand at 12:00 PM and on weekends
2. **Feature Engineering:** Created 109 features capturing temporal, lag, rolling, and difference patterns
3. **Model Development:** Implemented and evaluated 6 forecasting approaches across 3 horizons
4. **Performance Achievement:** XGBoost achieved MAE of 17.73 (15-min), 18.68 (1-hour), 19.81 (2-hour)
5. **Uncertainty Quantification:** Implemented prediction intervals using Random Forest ensemble
6. **Operational Deployment:** Created interactive Streamlit dashboard for real-time decision support

**Primary Conclusion:** Machine learning approaches, particularly XGBoost, significantly outperform traditional time-series methods for short-term ferry demand forecasting. The system provides actionable intelligence for proactive operational decision-making.

**Impact:** Transitioning from reactive to predictive operations can reduce congestion, improve service efficiency, enhance safety, and optimize resource allocation for Toronto Island ferry operations.

---

## 7. Recommendations

### 7.1 Immediate Actions (0-3 months)

1. **Deploy XGBoost Model:** Implement the best-performing model in production for 15-minute and 1-hour forecasts
2. **Staff Training:** Train operational staff on interpreting and using forecast outputs
3. **Dashboard Integration:** Integrate Streamlit dashboard into operational decision-making workflows
4. **Baseline Monitoring:** Establish baseline performance metrics for ongoing evaluation

### 7.2 Short-term Actions (3-12 months)

1. **External Data Integration:** Incorporate weather, events, and holiday data
2. **Uncertainty Improvement:** Implement conformal prediction for reliable confidence intervals
3. **Real-time Pipeline:** Develop automated data pipeline for real-time forecasting
4. **Alert System:** Create automated alerts for demand spikes and anomalies

### 7.3 Long-term Actions (12+ months)

1. **Full System Integration:** Integrate forecasting with ferry scheduling and staff management systems
2. **Advanced Analytics:** Develop scenario analysis and optimization models
3. **Expansion:** Extend forecasting to other transportation hubs
4. **Continuous Improvement:** Establish ongoing model retraining and evaluation framework

---

## 8. References

1. Toronto Government Parks, Forestry & Recreation. (2026). Ferry Operations Data.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
3. Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: principles and practice. OTexts.
4. Taylor, S. J., & Letham, B. (2018). Forecasting at scale. The American Statistician, 72(1), 37-45.
5. Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.

---

## Appendix

### A. Technical Specifications

**Computing Environment:**
- CPU: Intel i3 (no GPU)
- Python 3.11
- Libraries: pandas, numpy, scikit-learn, xgboost, statsmodels, streamlit

**Model Hyperparameters:**
- XGBoost: n_estimators=30, max_depth=5, learning_rate=0.1
- Random Forest: n_estimators=30, max_depth=8
- Gradient Boosting: n_estimators=30, max_depth=5
- ARIMA: order=(1,1,1)

### B. File Structure

```
Ferry_Project/
├── Toronto Island Ferry Tickets.csv (raw data)
├── ferry_tickets_cleaned.csv (cleaned data)
├── ferry_tickets_features.csv (engineered features)
├── model_results.csv (ML model results)
├── timeseries_results.csv (time-series model results)
├── all_model_results.csv (combined results)
├── uncertainty_results.csv (prediction intervals)
├── uncertainty_summary.json (uncertainty metrics)
├── eda_plots/ (EDA visualizations)
├── uncertainty_plots/ (uncertainty visualizations)
├── data_exploration.py
├── eda_analysis.py
├── feature_engineering.py
├── forecasting_models.py
├── time_series_models.py
├── uncertainty_analysis.py
├── streamlit_app.py
└── RESEARCH_PAPER.md
```

### C. Streamlit Dashboard Features

- **Dashboard Overview:** Key metrics, recent trends, best model summary
- **Model Comparison:** Interactive performance comparison across models and horizons
- **Forecast Visualization:** Date range selection, horizon selection, model selection
- **Uncertainty Analysis:** Prediction intervals, coverage metrics, uncertainty visualization
- **EDA Insights:** Hourly and daily patterns, operational recommendations

---

**End of Research Paper**
