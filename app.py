import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go

st.set_page_config(page_title="Inventory Drift Monitor", layout="wide", page_icon="🛍️")

# --- CORE LOGIC FUNCTIONS ---

def get_mock_historical_data(product_name, season, days=90):
    """Generates data in memory instead of using a database"""
    np.random.seed(hash(product_name + season) % (2**32))
    
    season_multipliers = {
        'Winter': {'Jackets': 1.8, 'Gloves': 1.9, 'Boots': 1.7, 'Scarves': 1.6},
        'Summer': {'Jackets': 0.4, 'Gloves': 0.3, 'Boots': 0.5, 'Scarves': 0.3},
        'Spring': {'Jackets': 0.9, 'Gloves': 0.7, 'Boots': 0.8, 'Scarves': 0.7},
        'Fall': {'Jackets': 1.3, 'Gloves': 1.2, 'Boots': 1.2, 'Scarves': 1.1}
    }
    
    base_sales = {'Jackets': 150, 'Gloves': 200, 'Boots': 120, 'Scarves': 180}
    data = []
    start_date = datetime.now() - timedelta(days=days)
    base = base_sales.get(product_name, 100)
    multiplier = season_multipliers.get(season, {}).get(product_name, 1.0)
    
    for i in range(days):
        current_date = start_date + timedelta(days=i)
        temp = np.random.normal(15 if season == 'Spring' else 25 if season == 'Summer' else 5 if season == 'Winter' else 12, 5)
        sales = base * multiplier
        sales = max(0, int(sales + np.random.normal(0, sales * 0.15)))
        
        data.append({
            'date': current_date.date(),
            'sales': sales,
            'temperature': temp,
            'day_of_week': current_date.weekday(),
            'is_weekend': current_date.weekday() >= 5,
            'is_holiday': np.random.random() < 0.05,
            'has_promotion': np.random.random() < 0.2,
            'month': current_date.month
        })
    return pd.DataFrame(data)

def train_forecasting_model(historical_df):
    """Trains the Random Forest model and returns metrics"""
    X = historical_df[['temperature', 'day_of_week', 'is_weekend', 'is_holiday', 'has_promotion', 'month']].astype(float)
    y = historical_df['sales']
    
    model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
    model.fit(X, y)
    
    predictions = model.predict(X)
    
    # FIXED: Added the missing closing parenthesis below
    return model, {
        'MAE': mean_absolute_error(y, predictions),
        'RMSE': np.sqrt(mean_squared_error(y, predictions)),
        'R2': r2_score(y, predictions)
    }

def calculate_drift_metrics(historical_sales, current_sales):
    """Calculates statistical drift scores"""
    hist_mean = np.mean(historical_sales)
    hist_std = np.std(historical_sales)
    z_score = (current_sales - hist_mean) / hist_std if hist_std > 0 else 0
    return {
        'z_score': z_score,
        'drift_score': abs(z_score) * 10,
        'historical_mean': hist_mean,
        'is_anomaly': abs(z_score) > 2
    }

# --- MAIN APP UI ---

def main():
    st.title("🛍️ ML Inventory & Drift Monitor")
    st.success("🚀 System running in High-Stability Mode")
    
    with st.sidebar:
        st.header("Settings")
        product = st.selectbox("Select Product", ["Jackets", "Gloves", "Boots", "Scarves"])
        season = st.radio("Season", ["Spring", "Summer", "Fall", "Winter"])
        
    col1, col2 = st.columns(2)
    with col1:
        avg_temp = st.number_input("Average Temp (°C)", value=15)
        actual_sales_today = st.number_input("Actual Sales (Today)", value=450)
    with col2:
        active_promotion = st.checkbox("Active Promotion", value=False)
        st.info("The model will use these inputs to detect drift against historical patterns.")
    
    if
