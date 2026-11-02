# Tidal — Ferry operations intelligence

Tidal turns the Toronto Island ferry research project into an interactive operating
workspace. Open **http://localhost:5000/app** after starting the project.
The original research dashboard remains at http://localhost:8501.

## Product workspace

- Overview with daily demand, capacity, utilization and queue signals.
- Demand and capacity lab: choose a planning date, vessel seats, sailing frequency,
  demand adjustment, ticket value and variable sailing cost.
- Hourly queue carryover, suggested departures and scenario economics.
- Saved operating plans backed by SQLite, plus CSV and JSON exports.
- Network seasonality and model evidence with a chronological holdout.
- Responsive desktop and mobile layouts, accessible controls and error states.

### Start locally

```powershell
pip install -r requirements.txt
.\start-local.ps1
```

Go to http://localhost:5000/app. Product UI and product API use the same server,
so the new interface needs no frontend build step or separate CORS configuration.

### Persistent plans

Local plans are stored in `instance/plans.sqlite3` (excluded from Git and images).
For Docker, attach a persistent volume at `/app/instance`:

```bash
docker run -d -p 5000:5000 -p 8501:8501 -v tidal-plans:/app/instance ghcr.io/vashisth8246/ferry_project:latest
```

This is a shared investor demo workspace, without authentication or tenant isolation.
Do not expose it as an authenticated customer service. Production rollout requires
operator identity, permissions, backups, monitoring, current ticket/boarding feeds,
and route/vessel constraints. Plans do not dispatch actual vessels.

### What the product model does

The running product uses mean hourly ticket sales grouped by month, weekend and
hour, with an hourly mean fallback. Historical variation is shown as mean plus or
minus 1.28 standard deviations, clipped at zero. This is **not a calibrated
prediction interval**. Evaluation holds out the last 28 calendar days before fitting.
Planning uses all available observations, ending December 21, 2025. No saved
XGBoost artifact, live ticket feed, weather, or events integration is claimed.
Ticket sales are a planning proxy rather than confirmed boardings. Financial
results use user-selected CAD assumptions and exclude fixed costs.

### Investor demo flow

1. Open Overview and explain the demand/capacity mismatch.
2. Open Demand & capacity and raise departures per hour from 2 to 3, then select
   **Update scenario**. Compare the queue and contribution.
3. Save a named operating plan and reopen it from Saved plans.
4. Export the hourly CSV for operations review.
5. Open Model & data to show the evaluation method and limitations.

No market valuation, customer traction or verified commercial savings are implied.

### Validation

```bash
python -m pytest tests -q
node --check frontend/app.js
```

GitHub Actions tests the APIs and product assets, builds the container, checks
both services and product endpoints, and renders the original research dashboard.
CD publishes the successful main commit to GitHub Container Registry. Hosting
remains optional through `DEPLOY_HOOK_URL`.

---

## Original research documentation

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

## Render deployment

Create a Render Blueprint from this repository's `render.yaml`. The default is a
free shared demo, with automatic Render deploys disabled so GitHub Actions gates
updates. The initial Blueprint deployment is initiated by Render.

The public product path is `/app`, health path `/health`, and version path
`/api/product/version`. Render runs only the product/API using `render-start.sh`.

After service creation:
1. Add its secret deploy hook as GitHub repository secret `DEPLOY_HOOK_URL`.
2. Add its public base URL as repository variable `RENDER_APP_URL` (no `/app`).
3. Push to main. CI tests the commit; CD publishes its image, asks Render to deploy
   that exact Git commit, and waits for its version endpoint and health check.

Free Render storage is ephemeral: saved plans can disappear on restart or deploy.
For durable plans, select a paid service and mount a persistent disk at
`/app/instance`. Do not put Render deploy hook URLs or API credentials in Git.
