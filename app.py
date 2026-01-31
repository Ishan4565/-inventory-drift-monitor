import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go

# 1. Page Config
st.set_page_config(page_title="Universal Drift Monitor", layout="wide", page_icon="📈")

# 2. Universal Data Generation Logic
def get_mock_historical_data(product_name, season, days=90):
    # The seed is now based on whatever name the user types!
    np.random.seed(hash(product_name + season) % (2**32))
    
    # Generic multipliers for the 4 seasons
    season_multipliers = {
        'Winter': 1.5,
        'Summer': 0.8,
        'Spring': 1.1,
        'Fall': 1.2
    }
    
    base_sales = 100 # Default starting point
    data = []
    start_date = datetime.now() - timedelta(days=days)
    multiplier = season_multipliers.get(season, 1.0)
    
    for i in range(days):
        current_date = start_date + timedelta(days=i)
        temp = np.random.normal(15 if season == 'Spring' else 25 if season == 'Summer' else 5 if season == 'Winter' else 12, 5)
        # Random variance + seasonal trend
        sales = max(0, int((base_sales * multiplier) + np.random.normal(0, 15)))
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
        'R2': r2_score(y, predictions)
    }
    return model, metrics

# 4. Main UI
def main():
    st.title("🛍️ Universal Inventory & Drift Monitor")
    st.markdown("### Type in any product to begin analysis")
    
    with st.sidebar:
        st.header("Product Configuration")
        # NEW: User can now type their own product name
        user_product = st.text_input("What is your product name?", placeholder="e.g. Sunglasses, Coffee Beans...")
        season = st.radio("Current Season", ["Spring", "Summer", "Fall", "Winter"])
        
    if not user_product:
        st.warning("Please enter a product name in the sidebar to load the model.")
        return

    col1, col2 = st.columns(2)
    with col1:
        avg_temp = st.number_input("Current Temp (°C)", value=20)
        actual_sales_today = st.number_input("Actual Sales (Today)", value=120)
    with col2:
        active_promotion = st.checkbox("Running a Promotion?", value=False)
        st.info(f"Analyzing trends for: **{user_product}**")
    
    if st.button("Run AI Drift Analysis", type="primary"):
        with st.spinner(f"Simulating market patterns for {user_product}..."):
            df = get_mock_historical_data(user_product, season)
            model, metrics = train_forecasting_model(df)
            
            # Predict
            now = datetime.now()
            features = np.array([[avg_temp, now.weekday(), now.weekday()>=5, False, active_promotion, now.month]])
            expected = int(model.predict(features)[0])
            
            # Drift Logic
            diff = abs(expected - actual_sales_today)
            threshold = np.std(df['sales']) * 1.5
            is_anomaly = diff > threshold
            
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Demand", expected)
            m2.metric("Actual Demand", actual_sales_today, delta=f"{actual_sales_today - expected}")
            m3.metric("Market Status", "🚨 DRIFT" if is_anomaly else "✅ STABLE")

            # Charting
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['date'], y=df['sales'], name=f"Estimated {user_product} History", line=dict(color='#00CC96')))
            fig.add_hline(y=actual_sales_today, line_color="red", annotation_text="Today's Entry")
            fig.update_layout(title=f"90-Day Market Trend: {user_product}", height=450)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
