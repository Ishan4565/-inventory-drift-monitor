import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go

# 1. Page Config
st.set_page_config(page_title="Inventory Drift Monitor", layout="wide", page_icon="🛍️")

# 2. Data Generation Logic
def get_mock_historical_data(product_name, season, days=90):
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
        sales = max(0, int((base * multiplier) + np.random.normal(0, 20)))
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

# 3. ML Logic
def train_forecasting_model(historical_df):
    X = historical_df[['temperature', 'day_of_week', 'is_weekend', 'is_holiday', 'has_promotion', 'month']].astype(float)
    y = historical_df['sales']
    model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    metrics = {
        'MAE': mean_absolute_error(y, predictions),
        'RMSE': np.sqrt(mean_squared_error(y, predictions)),
        'R2': r2_score(y, predictions)
    }
    return model, metrics

# 4. Main UI
def main():
    st.title("🛍️ ML Inventory & Drift Monitor")
    st.success("✅ App is Live and Stable")
    
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
        st.write("Click the button below to trigger the Random Forest model.")
    
    if st.button("Run AI Analysis", type="primary"):
        with st.spinner("Calculating..."):
            df = get_mock_historical_data(product, season)
            model, metrics = train_forecasting_model(df)
            
            # Predict
            now = datetime.now()
            features = np.array([[avg_temp, now.weekday(), now.weekday()>=5, False, active_promotion, now.month]])
            expected = int(model.predict(features)[0])
            
            # Drift
            diff = abs(expected - actual_sales_today)
            is_anomaly = diff > (np.std(df['sales']) * 2)
            
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted", expected)
            m2.metric("Actual", actual_sales_today, delta=f"{actual_sales_today - expected}")
            m3.metric("Status", "⚠️ DRIFT" if is_anomaly else "✅ STABLE")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['date'], y=df['sales'], name="History"))
            fig.add_hline(y=actual_sales_today, line_color="red", annotation_text="Today")
            st.plotly_chart(fig, use_container_width=True)

# 5. Safe Execution
if __name__ == "__main__":
    main()
