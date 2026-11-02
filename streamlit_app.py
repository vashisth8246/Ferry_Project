"""
Streamlit Dashboard for Ferry Ticket Demand Forecasting
CPU-optimized interactive visualization
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Ferry Demand Forecasting System",
    page_icon="⛴️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Hide Streamlit footer */
    .stApp { footer { visibility: hidden; } }
    .stApp { footer:after { content: ""; visibility: hidden; display: block; height: 0; } }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Fetch dashboard data through the Flask API.
from api_client import fetch_json

@st.cache_data(ttl=60)
def load_data():
    df = pd.DataFrame(fetch_json('/api/data/recent', n=1000000))
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df.sort_values('Timestamp').reset_index(drop=True)

@st.cache_data(ttl=60)
def load_model_results():
    return pd.DataFrame.from_dict(fetch_json('/api/models/results'), orient='index')

@st.cache_data(ttl=60)
def load_uncertainty_results():
    return pd.DataFrame(fetch_json('/api/uncertainty/predictions', n=1000000))

@st.cache_data(ttl=60)
def load_eda_summary():
    return fetch_json('/api/eda/summary')

# Main application
def main():
    st.markdown('<h1 class="main-header">⛴️ Ferry Ticket Demand Forecasting System</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    try:
        df = load_data()
        model_results = load_model_results()
        uncertainty_results = load_uncertainty_results()
        eda_summary = load_eda_summary()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()
    st.sidebar.success('Backend connected')

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Dashboard Overview", "Model Comparison", "Forecast Visualization", 
         "Uncertainty Analysis", "EDA Insights"]
    )
    
    # ==================== DASHBOARD OVERVIEW ====================
    if page == "Dashboard Overview":
        st.header("📊 Dashboard Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Avg Sales/Interval", f"{eda_summary['avg_sales']:.1f}")
        with col3:
            st.metric("Avg Redemptions/Interval", f"{eda_summary['avg_redemptions']:.1f}")
        with col4:
            st.metric("Date Range", f"{eda_summary['date_range_start'][:10]} to {eda_summary['date_range_end'][:10]}")
        
        st.markdown("---")
        
        # Recent trends
        st.subheader("Recent Demand Trends (Last 7 Days)")
        recent_data = df.tail(672)  # 7 days * 96 intervals/day
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent_data['Timestamp'],
            y=recent_data['Sales Count'],
            mode='lines',
            name='Sales',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=recent_data['Timestamp'],
            y=recent_data['Redemption Count'],
            mode='lines',
            name='Redemptions',
            line=dict(color='#ff7f0e', width=2)
        ))
        fig.update_layout(
            title='Sales vs Redemptions (Last 7 Days)',
            xaxis_title='Timestamp',
            yaxis_title='Count',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model summary
        st.subheader("🏆 Best Performing Models")
        best_15min = model_results[model_results.index.str.contains('15min')]['MAE'].idxmin()
        best_1hr = model_results[model_results.index.str.contains('1hr')]['MAE'].idxmin()
        best_2hr = model_results[model_results.index.str.contains('2hr')]['MAE'].idxmin()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**15-min Forecast:** {best_15min}\n\nMAE: {model_results.loc[best_15min, 'MAE']:.2f}")
        with col2:
            st.info(f"**1-hour Forecast:** {best_1hr}\n\nMAE: {model_results.loc[best_1hr, 'MAE']:.2f}")
        with col3:
            st.info(f"**2-hour Forecast:** {best_2hr}\n\nMAE: {model_results.loc[best_2hr, 'MAE']:.2f}")
    
    # ==================== MODEL COMPARISON ====================
    elif page == "Model Comparison":
        st.header("🤖 Model Performance Comparison")
        
        # Model selection
        st.subheader("Model Performance Metrics")
        
        metric = st.selectbox("Select Metric", ["MAE", "RMSE", "MAPE"])
        
        # Group by horizon
        horizons = ['15min', '1hr', '2hr']
        
        fig = go.Figure()
        
        for horizon in horizons:
            horizon_data = model_results[model_results.index.str.contains(horizon)]
            fig.add_trace(go.Bar(
                name=horizon,
                x=horizon_data.index,
                y=horizon_data[metric],
                text=horizon_data[metric].round(2),
                textposition='outside'
            ))
        
        fig.update_layout(
            title=f'Model Comparison by Forecast Horizon ({metric})',
            xaxis_title='Model',
            yaxis_title=metric,
            barmode='group',
            height=500,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.subheader("Detailed Results")
        st.dataframe(model_results.style.background_gradient(cmap='Blues', subset=['MAE', 'RMSE', 'MAPE']))
        
        # Model recommendations
        st.subheader("💡 Model Recommendations")
        st.success("""
        **XGBoost** performs best across all forecast horizons with the lowest MAE.
        
        - **15-minute forecasts**: Use XGBoost for immediate operational decisions
        - **1-hour forecasts**: XGBoost provides reliable short-term planning
        - **2-hour forecasts**: XGBoost maintains good performance for medium-term planning
        
        **Gradient Boosting** is a strong alternative with similar performance.
        """)
    
    # ==================== FORECAST VISUALIZATION ====================
    elif page == "Forecast Visualization":
        st.header("📈 Forecast Visualization")
        
        # Date range selector
        st.subheader("Select Time Range")
        
        min_date = df['Timestamp'].min().date()
        max_date = df['Timestamp'].max().date()
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        # Filter data
        mask = (df['Timestamp'].dt.date >= start_date) & (df['Timestamp'].dt.date <= end_date)
        filtered_data = df[mask].copy()
        
        if len(filtered_data) == 0:
            st.warning("No data available for selected date range.")
        else:
            # Forecast horizon selector
            horizon = st.selectbox("Forecast Horizon", ["15 minutes", "1 hour", "2 hours"])
            
            # Model selector
            model = st.selectbox("Select Model", ["XGBoost", "Gradient Boosting", "Random Forest", "Naïve", "Moving Average"])
            
            # Plot actual vs simulated forecast
            fig = go.Figure()
            
            # Actual data
            fig.add_trace(go.Scatter(
                x=filtered_data['Timestamp'],
                y=filtered_data['Sales Count'],
                mode='lines',
                name='Actual Sales',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Simulated forecast (using lag for visualization)
            if horizon == "15 minutes":
                lag = 1
            elif horizon == "1 hour":
                lag = 4
            else:
                lag = 8
            
            forecast_values = filtered_data['Sales Count'].shift(-lag)
            fig.add_trace(go.Scatter(
                x=filtered_data['Timestamp'],
                y=forecast_values,
                mode='lines',
                name=f'Simulated {horizon} Forecast',
                line=dict(color='#ff7f0e', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title=f'Actual vs Simulated Forecast ({model} - {horizon})',
                xaxis_title='Timestamp',
                yaxis_title='Sales Count',
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("Note: This shows a simulated forecast using lagged values. In production, the trained model would generate actual predictions.")
    
    # ==================== UNCERTAINTY ANALYSIS ====================
    elif page == "Uncertainty Analysis":
        st.header("🎯 Forecast Uncertainty Analysis")
        
        # Uncertainty metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with open('uncertainty_summary.json', 'r') as f:
            uncertainty_summary = json.load(f)
        
        with col1:
            st.metric("MAE", f"{uncertainty_summary['mae']:.2f}")
        with col2:
            st.metric("RMSE", f"{uncertainty_summary['rmse']:.2f}")
        with col3:
            st.metric("90% CI Width", f"{uncertainty_summary['avg_ci_width_90']:.2f}")
        with col4:
            st.metric("MASE", f"{uncertainty_summary['mase']:.4f}")
        
        st.markdown("---")
        
        # Prediction intervals visualization
        st.subheader("Prediction Intervals")
        
        sample_size = min(500, len(uncertainty_results))
        sample_data = uncertainty_results.iloc[:sample_size]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(sample_size)),
            y=sample_data['actual'],
            mode='lines',
            name='Actual',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(sample_size)),
            y=sample_data['predicted'],
            mode='lines',
            name='Predicted',
            line=dict(color='#ff7f0e', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(sample_size)),
            y=sample_data['upper_90'],
            mode='lines',
            name='Upper 90% CI',
            line=dict(color='rgba(255,127,14,0.3)', width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(sample_size)),
            y=sample_data['lower_90'],
            mode='lines',
            name='Lower 90% CI',
            line=dict(color='rgba(255,127,14,0.3)', width=0),
            fill='tonexty',
            fillcolor='rgba(255,127,14,0.2)',
            showlegend=False
        ))
        
        fig.update_layout(
            title='Forecast with 90% Prediction Intervals',
            xaxis_title='Time Step',
            yaxis_title='Sales Count',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Coverage analysis
        st.subheader("Prediction Interval Coverage")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("90% CI Coverage", f"{uncertainty_summary['coverage_90']*100:.1f}%")
        with col2:
            st.metric("50% CI Coverage", f"{uncertainty_summary['coverage_50']*100:.1f}%")
        
        st.warning("""
        **Note on Coverage**: The current prediction intervals have lower coverage than expected. 
        This suggests the model uncertainty is underestimated. For operational use, consider:
        - Using wider confidence intervals
        - Implementing ensemble methods
        - Adding more uncertainty quantification techniques
        """)
    
    # ==================== EDA INSIGHTS ====================
    elif page == "EDA Insights":
        st.header("🔍 Exploratory Data Analysis Insights")
        
        # Key findings
        st.subheader("Key Findings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Peak Demand Hours**
            - Sales: {eda_summary['peak_hour_sales']}:00
            - Redemptions: {eda_summary['peak_hour_redemptions']}:00
            
            **Peak Days**
            - Sales: {eda_summary['peak_day_sales']}
            - Redemptions: {eda_summary['peak_day_redemptions']}
            """)
        
        with col2:
            st.info(f"""
            **Correlation**
            - Sales ↔ Redemptions: {eda_summary['sales_redemption_correlation']:.3f}
            
            **Maximum Values**
            - Max Sales: {eda_summary['max_sales']}
            - Max Redemptions: {eda_summary['max_redemptions']}
            """)
        
        st.markdown("---")
        
        # Hourly patterns
        st.subheader("Hourly Demand Patterns")
        
        df['hour'] = df['Timestamp'].dt.hour
        hourly_avg = df.groupby('hour')[['Sales Count', 'Redemption Count']].mean()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Sales',
            x=hourly_avg.index,
            y=hourly_avg['Sales Count'],
            marker_color='#1f77b4'
        ))
        fig.add_trace(go.Bar(
            name='Redemptions',
            x=hourly_avg.index,
            y=hourly_avg['Redemption Count'],
            marker_color='#ff7f0e'
        ))
        fig.update_layout(
            title='Average Sales and Redemptions by Hour',
            xaxis_title='Hour of Day',
            yaxis_title='Average Count',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Day of week patterns
        st.subheader("Day of Week Patterns")
        
        df['day_name'] = df['Timestamp'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['day_name'] = pd.Categorical(df['day_name'], categories=day_order, ordered=True)
        daily_avg = df.groupby('day_name')[['Sales Count', 'Redemption Count']].mean()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Sales',
            x=daily_avg.index,
            y=daily_avg['Sales Count'],
            marker_color='#1f77b4'
        ))
        fig.add_trace(go.Bar(
            name='Redemptions',
            x=daily_avg.index,
            y=daily_avg['Redemption Count'],
            marker_color='#ff7f0e'
        ))
        fig.update_layout(
            title='Average Sales and Redemptions by Day of Week',
            xaxis_title='Day of Week',
            yaxis_title='Average Count',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Operational recommendations
        st.subheader("📋 Operational Recommendations")
        
        st.success("""
        **Staffing Recommendations:**
        - Peak hours (11:00-14:00) require maximum staffing
        - Weekends (Saturday, Sunday) have 70-80% higher demand than weekdays
        - Consider dynamic staffing based on forecasted demand
        
        **Scheduling Recommendations:**
        - Increase ferry frequency during peak hours (11:00-14:00)
        - Weekend schedules should accommodate 2x weekday demand
        - Use 15-minute forecasts for real-time adjustments
        
        **Capacity Planning:**
        - Monitor 2-hour forecasts to anticipate demand surges
        - Implement crowd management during peak periods
        - Use prediction intervals to plan for uncertainty
        """)

if __name__ == "__main__":
    import json
    main()
