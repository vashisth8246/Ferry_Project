# Ferry Ticket Demand Forecasting System

A comprehensive machine learning and time-series forecasting system for predicting short-term ferry ticket demand to Toronto Island Park.

## Project Overview

This project transforms over 10 years of historical ferry ticket data into predictive intelligence, enabling proactive operational decision-making for ferry scheduling, staff allocation, and crowd management.

**Key Achievement:** XGBoost model achieves MAE of 17.73 (15-min), 18.68 (1-hour), and 19.81 (2-hour) tickets.

## Dataset

- **Records:** 261,538 (2015-2025)
- **Granularity:** 15-minute intervals
- **Variables:** Sales Count, Redemption Count, Timestamp
- **Missing Intervals:** 30% (filled via interpolation)

## Project Structure

```
Ferry_Project/
├── Toronto Island Ferry Tickets.csv    # Raw dataset
├── ferry_tickets_cleaned.csv          # Cleaned data
├── ferry_tickets_features.csv         # Engineered features (109 features)
├── model_results.csv                  # ML model performance results
├── timeseries_results.csv             # Time-series model results
├── all_model_results.csv              # Combined model results
├── uncertainty_results.csv            # Prediction intervals
├── uncertainty_summary.json           # Uncertainty metrics
├── eda_plots/                         # EDA visualizations
│   ├── 01_time_series_overview.png
│   ├── 02_hourly_patterns.png
│   ├── 03_daily_patterns.png
│   ├── 04_monthly_seasonality.png
│   ├── 05_distribution_analysis.png
│   ├── 06_correlation_analysis.png
│   ├── 07_weekly_trend.png
│   └── summary_statistics.json
├── uncertainty_plots/                 # Uncertainty visualizations
│   ├── prediction_intervals.png
│   └── error_distribution.png
├── data_exploration.py                # Data quality analysis
├── eda_analysis.py                    # Exploratory data analysis
├── feature_engineering.py             # Feature engineering pipeline
├── forecasting_models.py              # ML model implementation
├── time_series_models.py              # ARIMA/Prophet models
├── uncertainty_analysis.py           # Prediction interval analysis
├── streamlit_app.py                   # Interactive dashboard
├── RESEARCH_PAPER.md                  # Comprehensive research paper
├── EXECUTIVE_SUMMARY.md              # Executive summary for stakeholders
├── requirements.txt                   # Python dependencies
└── README.md                         # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Exploration

```bash
python data_exploration.py
```

### 2. Exploratory Data Analysis

```bash
python eda_analysis.py
```

### 3. Feature Engineering

```bash
python feature_engineering.py
```

### 4. Model Training

```bash
python forecasting_models.py
python time_series_models.py
```

### 5. Uncertainty Analysis

```bash
python uncertainty_analysis.py
```

### 6. Launch Dashboard

```bash
streamlit run streamlit_app.py
```

The dashboard will be available at `http://localhost:8501`

## Model Performance

| Model | Horizon | MAE | RMSE | MAPE |
|-------|---------|-----|------|------|
| **XGBoost** | 15-min | **17.73** | **64.17** | 98.90% |
| **XGBoost** | 1-hour | **18.68** | **64.63** | 109.20% |
| **XGBoost** | 2-hour | **19.81** | **65.12** | 126.45% |
| Gradient Boosting | 15-min | 18.21 | 64.10 | 141.32% |
| Random Forest | 15-min | 18.73 | 66.29 | 103.69% |
| ARIMA | Hourly | 28.32 | 54.66 | 76.17% |

## Key Findings

### Demand Patterns
- **Peak hours:** 11:00 AM - 2:00 PM
- **Peak days:** Weekends (70-80% higher than weekdays)
- **Seasonal patterns:** Summer months show elevated demand
- **Correlation:** Strong link between sales and redemptions (r = 0.92)

### Feature Engineering
- 109 features created from 3 base variables
- Lag features (t-1 to t-96) capture autoregressive patterns
- Rolling statistics (1hr to 24hr) capture trends and volatility
- Cyclical temporal encoding for hour, day, month

## Dashboard Features

### Pages
1. **Dashboard Overview:** Key metrics, recent trends, best model summary
2. **Model Comparison:** Interactive performance comparison
3. **Forecast Visualization:** Date range selection, model selection
4. **Uncertainty Analysis:** Prediction intervals, coverage metrics
5. **EDA Insights:** Hourly/daily patterns, operational recommendations

### Capabilities
- Real-time demand visualization
- Model performance comparison across horizons
- Forecast visualization with confidence intervals
- Uncertainty quantification
- Operational recommendations

## CPU Optimization

All models are optimized for CPU-only deployment (Intel i3):
- Reduced dataset size (100k records for training)
- Limited feature set (37 key features)
- Reduced model complexity (30 trees, depth 5-8)
- Training time: <5 minutes per model

## Deliverables

1. **Research Paper:** Comprehensive analysis with methodology, results, and recommendations
2. **Executive Summary:** Business-focused summary for government stakeholders
3. **Streamlit Dashboard:** Interactive web application for decision support
4. **Trained Models:** XGBoost, Gradient Boosting, Random Forest models
5. **Data Pipeline:** Complete data processing and feature engineering pipeline

## Recommendations

### Immediate Actions (0-3 months)
- Deploy XGBoost model in production
- Train operational staff on dashboard usage
- Establish baseline performance metrics

### Short-term Actions (3-12 months)
- Integrate external data (weather, events, holidays)
- Implement conformal prediction for reliable confidence intervals
- Develop real-time data pipeline

### Long-term Actions (12+ months)
- Full integration with ferry scheduling systems
- Expansion to other transportation hubs
- Advanced analytics and optimization modeling

## Technical Specifications

- **Python:** 3.11
- **Key Libraries:** pandas, numpy, scikit-learn, xgboost, statsmodels, streamlit
- **Hardware:** CPU-only (Intel i3 compatible)
- **Training Time:** <5 minutes per model
- **Forecast Generation:** <1 second per prediction

## Citation

If you use this work, please cite:

```
Ferry Ticket Demand Forecasting System. (2026). 
Toronto Government Parks, Forestry & Recreation.
```

## License

This project is developed for Toronto Government Parks, Forestry & Recreation.

## Contact

For questions or support, please contact the project team at Toronto Government Parks, Forestry & Recreation.

---

**Project Status:** Complete  
**Last Updated:** July 2026

## CI/CD and connected application

The Streamlit dashboard loads its data through the Flask API. Start both on Windows:

```powershell
pip install -r requirements.txt
.\start-local.ps1
```

Open http://localhost:8501 (API: http://localhost:5000/health).
Set `BACKEND_URL` when running the dashboard against a different API server.
For separate terminals, run `python -m backend.app` and `python -m streamlit run streamlit_app.py`.

GitHub Actions runs syntax checks, API/client tests, a Docker build, and mandatory
health checks for both services on pushes and pull requests to main. After a
successful main push, CD publishes the tested commit as
`ghcr.io/vashisth8246/ferry_project:latest` and an immutable full commit tag.
Container publishing uses the built-in GitHub token. An optional repository secret
`DEPLOY_HOOK_URL` triggers an existing hosting service after publication; a hosting
account must be configured separately. Image publication alone does not create a
hosted website. The container exposes dashboard port 8501 and API port 5000.

```bash
docker run -d -p 8501:8501 -p 5000:5000 ghcr.io/vashisth8246/ferry_project:latest
```

The generated 347 MB `ferry_tickets_features.csv` is excluded from the Docker
context; the dashboard uses the cleaned data and analysis outputs included in Git.
