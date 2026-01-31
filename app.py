import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import psycopg2
from psycopg2.extras import execute_values
import os

st.set_page_config(page_title="Inventory Drift Monitor", layout="wide", page_icon="🛍️")

@st.cache_resource
def get_db_connection():
    database_url = os.getenv('postgresql://inventory_drift_db_user:Xf9BpwHH8zNTqmjap0W1bCLXKd3kUzni@dpg-d5uarfiqcgvc73asnf80-a/inventory_drift_db')
    
    if database_url:
        if database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
        conn = psycopg2.connect(database_url)
    else:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            database=os.getenv('DB_NAME', 'inventory_ml'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'postgres'),
            port=os.getenv('DB_PORT', '5432')
        )
    
    return conn

def init_database():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            product_id SERIAL PRIMARY KEY,
            product_name VARCHAR(255) NOT NULL,
            category VARCHAR(100),
            base_price DECIMAL(10, 2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sales_data (
            sales_id SERIAL PRIMARY KEY,
            product_name VARCHAR(255) NOT NULL,
            season VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            sales INTEGER NOT NULL,
            temperature DECIMAL(5, 2),
            day_of_week INTEGER,
            is_weekend BOOLEAN,
            is_holiday BOOLEAN,
            has_promotion BOOLEAN,
            month INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS drift_analysis (
            analysis_id SERIAL PRIMARY KEY,
            product_name VARCHAR(255) NOT NULL,
            season VARCHAR(20) NOT NULL,
            analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            actual_sales INTEGER,
            expected_sales INTEGER,
            drift_score DECIMAL(10, 4),
            z_score DECIMAL(10, 4),
            psi_score DECIMAL(10, 4),
            is_anomaly BOOLEAN,
            temperature DECIMAL(5, 2),
            has_promotion BOOLEAN
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            metric_id SERIAL PRIMARY KEY,
            product_name VARCHAR(255),
            season VARCHAR(20),
            mae DECIMAL(10, 4),
            rmse DECIMAL(10, 4),
            r2_score DECIMAL(10, 4),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        conn.commit()
        cursor.close()
        
        return True
        
    except Exception as e:
        st.error(f"Database initialization error: {e}")
        return False

def insert_historical_data(conn, product_name, season, days=90):
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) FROM sales_data 
        WHERE product_name = %s AND season = %s
    """, (product_name, season))
    
    count = cursor.fetchone()[0]
    
    if count >= days:
        cursor.close()
        return count
    
    np.random.seed(hash(product_name + season) % (2**32))
    
    season_multipliers = {
        'Winter': {'Jackets': 1.8, 'Gloves': 1.9, 'Boots': 1.7, 'Scarves': 1.6},
        'Summer': {'Jackets': 0.4, 'Gloves': 0.3, 'Boots': 0.5, 'Scarves': 0.3},
        'Spring': {'Jackets': 0.9, 'Gloves': 0.7, 'Boots': 0.8, 'Scarves': 0.7},
        'Fall': {'Jackets': 1.3, 'Gloves': 1.2, 'Boots': 1.2, 'Scarves': 1.1}
    }
    
    base_sales = {
        'Jackets': 150,
        'Gloves': 200,
        'Boots': 120,
        'Scarves': 180
    }
    
    data = []
    start_date = datetime.now() - timedelta(days=days)
    
    base = base_sales.get(product_name, 100)
    multiplier = season_multipliers.get(season, {}).get(product_name, 1.0)
    
    for i in range(days):
        current_date = start_date + timedelta(days=i)
        day_of_week = current_date.weekday()
        month = current_date.month
        
        is_weekend = day_of_week >= 5
        is_holiday = np.random.random() < 0.05
        has_promotion = np.random.random() < 0.2
        
        temp = np.random.normal(15 if season == 'Spring' else 25 if season == 'Summer' else 5 if season == 'Winter' else 12, 5)
        
        sales = base * multiplier
        
        if is_weekend:
            sales *= 1.2
        if is_holiday:
            sales *= 1.4
        if has_promotion:
            sales *= 1.3
        
        temp_factor = 1 + (15 - temp) * 0.02 if product_name in ['Jackets', 'Gloves', 'Boots', 'Scarves'] else 1
        sales *= temp_factor
        
        sales = max(0, int(sales + np.random.normal(0, sales * 0.15)))
        
        data.append((
            product_name,
            season,
            current_date.date(),
            sales,
            temp,
            day_of_week,
            is_weekend,
            is_holiday,
            has_promotion,
            month
        ))
    
    execute_values(cursor, """
        INSERT INTO sales_data 
        (product_name, season, date, sales, temperature, day_of_week, is_weekend, is_holiday, has_promotion, month)
        VALUES %s
        ON CONFLICT DO NOTHING
    """, data)
    
    conn.commit()
    cursor.close()
    
    return len(data)

def get_historical_data(conn, product_name, season):
    query = """
        SELECT date, sales, temperature, day_of_week, is_weekend, is_holiday, has_promotion, month
        FROM sales_data
        WHERE product_name = %s AND season = %s
        ORDER BY date DESC
        LIMIT 90
    """
    
    df = pd.read_sql(query, conn, params=(product_name, season))
    return df

def save_drift_analysis(conn, product_name, season, actual_sales, expected_sales, drift_metrics, temperature, has_promotion):
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO drift_analysis 
        (product_name, season, actual_sales, expected_sales, drift_score, z_score, psi_score, is_anomaly, temperature, has_promotion)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        product_name,
        season,
        actual_sales,
        expected_sales,
        drift_metrics['drift_score'],
        drift_metrics['z_score'],
        drift_metrics['psi'],
        drift_metrics['is_anomaly'],
        temperature,
        has_promotion
    ))
    
    conn.commit()
    cursor.close()

def save_model_metrics(conn, product_name, season, mae, rmse, r2):
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO model_metrics (product_name, season, mae, rmse, r2_score)
        VALUES (%s, %s, %s, %s, %s)
    """, (product_name, season, mae, rmse, r2))
    
    conn.commit()
    cursor.close()

def get_past_analyses(conn, product_name, season, limit=10):
    query = """
        SELECT analysis_date, actual_sales, expected_sales, drift_score, is_anomaly
        FROM drift_analysis
        WHERE product_name = %s AND season = %s
        ORDER BY analysis_date DESC
        LIMIT %s
    """
    
    df = pd.read_sql(query, conn, params=(product_name, season, limit))
    return df

def calculate_drift_metrics(historical_sales, current_sales, historical_temp, current_temp):
    if len(historical_sales) < 10:
        return None
    
    hist_mean = np.mean(historical_sales)
    hist_std = np.std(historical_sales)
    
    z_score = (current_sales - hist_mean) / hist_std if hist_std > 0 else 0
    
    percentile = stats.percentileofscore(historical_sales, current_sales)
    
    ks_statistic, ks_pvalue = stats.ks_2samp(historical_sales, [current_sales] * 10)
    
    reference_counts, bin_edges = np.histogram(historical_sales, bins=10)
    current_bin = np.digitize([current_sales], bin_edges)[0] - 1
    current_bin = min(max(0, current_bin), 9)
    
    ref_percents = reference_counts / len(historical_sales)
    curr_percents = np.zeros(10)
    curr_percents[current_bin] = 1.0
    
    ref_percents = np.where(ref_percents == 0, 0.0001, ref_percents)
    curr_percents = np.where(curr_percents == 0, 0.0001, curr_percents)
    
    psi = np.sum((curr_percents - ref_percents) * np.log(curr_percents / ref_percents))
    
    drift_score = abs(psi) * 100
    
    temp_drift = abs(current_temp - np.mean(historical_temp)) / np.std(historical_temp) if np.std(historical_temp) > 0 else 0
    
    return {
        'z_score': z_score,
        'percentile': percentile,
        'psi': psi,
        'drift_score': drift_score,
        'ks_statistic': ks_statistic,
        'ks_pvalue': ks_pvalue,
        'historical_mean': hist_mean,
        'historical_std': hist_std,
        'temp_drift': temp_drift,
        'is_anomaly': abs(z_score) > 2 or drift_score > 20
    }

def train_forecasting_model(historical_df):
    X = historical_df[['temperature', 'day_of_week', 'is_weekend', 'is_holiday', 'has_promotion', 'month']].astype(float)
    y = historical_df['sales']
    
    model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
    model.fit(X, y)
    
    predictions = model.predict(X)
    
    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)
    
    return model, {'MAE': mae, 'RMSE': rmse, 'R2': r2}

def predict_expected_sales(model, current_temp, day_of_week, is_weekend, is_holiday, has_promotion, month):
    features = np.array([[current_temp, day_of_week, is_weekend, is_holiday, has_promotion, month]])
    prediction = model.predict(features)[0]
    return max(0, int(prediction))

def main():
    st.title("🛍️ ML Inventory & Drift Monitor with PostgreSQL")
    st.markdown("### Real-time Sales Drift Detection with Database")
    
    if not init_database():
        st.error("❌ Could not initialize database. Please check your DATABASE_URL environment variable.")
        st.info("""
        **For Render Deployment:**
        1. Create PostgreSQL database in Render
        2. Add DATABASE_URL environment variable
        
        **For Local Development:**
        Set these environment variables:
        - DB_HOST=localhost
        - DB_NAME=inventory_ml
        - DB_USER=postgres
        - DB_PASSWORD=ishan123
        """)
        return
    
    st.success("✅ Connected to PostgreSQL Database")
    
    conn = get_db_connection()
    
    with st.sidebar:
        st.header("Settings")
        
        st.subheader("Product Category")
        product_category = st.selectbox(
            "Select Product",
            ["Jackets", "Gloves", "Boots", "Scarves"],
            label_visibility="collapsed"
        )
        
        st.subheader("Select Current Season")
        season = st.radio(
            "Season",
            ["Spring", "Summer", "Fall", "Winter"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        if st.button("🔄 Generate Historical Data"):
            with st.spinner("Generating data..."):
                records = insert_historical_data(conn, product_category, season, days=90)
                st.success(f"✅ Generated {records} records!")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        avg_temp = st.number_input("Average Temp (°C)", min_value=-20, max_value=45, value=15, step=1)
    
    with col2:
        active_promotion = st.checkbox("Active Promotion", value=False)
    
    col3, col4 = st.columns([1, 1])
    
    with col3:
        last_month_units = st.number_input("Last Month's Units", min_value=0, max_value=10000, value=500, step=10)
    
    with col4:
        actual_sales_today = st.number_input("Actual Sales (Today)", min_value=0, max_value=5000, value=450, step=10)
    
    if st.button("Run AI Analysis", type="primary"):
        with st.spinner("Analyzing data from PostgreSQL database..."):
            historical_df = get_historical_data(conn, product_category, season)
            
            if len(historical_df) < 30:
                st.warning("⚠️ Not enough historical data. Click 'Generate Historical Data' in the sidebar first.")
                return
            
            current_date = datetime.now()
            day_of_week = current_date.weekday()
            is_weekend = day_of_week >= 5
            is_holiday = False
            month = current_date.month
            
            model, metrics = train_forecasting_model(historical_df)
            
            save_model_metrics(conn, product_category, season, metrics['MAE'], metrics['RMSE'], metrics['R2'])
            
            expected_sales = predict_expected_sales(
                model, avg_temp, day_of_week, is_weekend, is_holiday, active_promotion, month
            )
            
            drift_metrics = calculate_drift_metrics(
                historical_df['sales'].values,
                actual_sales_today,
                historical_df['temperature'].values,
                avg_temp
            )
            
            save_drift_analysis(
                conn, product_category, season, actual_sales_today, 
                expected_sales, drift_metrics, avg_temp, active_promotion
            )
            
            st.success("✅ Analysis Complete! Results saved to database.")
            
            st.markdown("---")
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric(
                    "Expected Sales",
                    f"{expected_sales:,}",
                    delta=None
                )
            
            with metric_col2:
                variance = ((actual_sales_today - expected_sales) / expected_sales * 100) if expected_sales > 0 else 0
                st.metric(
                    "Actual Sales",
                    f"{actual_sales_today:,}",
                    delta=f"{variance:+.1f}%",
                    delta_color="normal" if abs(variance) < 10 else "inverse"
                )
            
            with metric_col3:
                st.metric(
                    "Drift Score",
                    f"{drift_metrics['drift_score']:.1f}",
                    delta="Anomaly" if drift_metrics['is_anomaly'] else "Normal",
                    delta_color="inverse" if drift_metrics['is_anomaly'] else "normal"
                )
            
            with metric_col4:
                st.metric(
                    "Historical Avg",
                    f"{drift_metrics['historical_mean']:.0f}",
                    delta=f"±{drift_metrics['historical_std']:.0f}"
                )
            
            st.markdown("---")
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Drift Analysis", 
                "📈 Historical Trends", 
                "🎯 Model Performance",
                "📋 Past Analyses",
                "💾 Database Stats"
            ])
            
            with tab1:
                st.subheader("Drift Detection Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    drift_status = "🔴 DRIFT DETECTED" if drift_metrics['is_anomaly'] else "🟢 NORMAL"
                    st.markdown(f"### {drift_status}")
                    
                    st.markdown(f"""
                    **Z-Score:** {drift_metrics['z_score']:.2f}  
                    **Percentile:** {drift_metrics['percentile']:.1f}%  
                    **PSI Score:** {drift_metrics['psi']:.4f}  
                    **Temperature Drift:** {drift_metrics['temp_drift']:.2f}σ
                    """)
                    
                    if drift_metrics['is_anomaly']:
                        st.warning("⚠️ **Anomaly Detected!** Sales significantly different from historical patterns")
                    else:
                        st.info("✅ **Sales Within Normal Range**")
                
                with col2:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Histogram(
                        x=historical_df['sales'],
                        name='Historical Distribution',
                        nbinsx=30,
                        marker_color='lightblue',
                        opacity=0.7
                    ))
                    
                    fig.add_vline(x=actual_sales_today, line_dash="dash", line_color="red", annotation_text="Today")
                    fig.add_vline(x=drift_metrics['historical_mean'], line_dash="dot", line_color="green", annotation_text="Mean")
                    
                    fig.update_layout(title="Sales Distribution", height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("Historical Sales from PostgreSQL")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=historical_df['date'], y=historical_df['sales'], mode='lines', name='Sales'))
                fig.add_hline(y=drift_metrics['historical_mean'], line_dash="dash", annotation_text="Avg")
                fig.update_layout(title=f"{product_category} - {season} Sales", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(historical_df.head(10), use_container_width=True)
            
            with tab3:
                st.subheader("ML Model Performance")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("MAE", f"{metrics['MAE']:.2f}")
                with col2:
                    st.metric("RMSE", f"{metrics['RMSE']:.2f}")
                with col3:
                    st.metric("R² Score", f"{metrics['R2']:.4f}")
            
            with tab4:
                st.subheader("Past Analysis Results from Database")
                
                past_analyses = get_past_analyses(conn, product_category, season, limit=20)
                
                if not past_analyses.empty:
                    st.dataframe(past_analyses, use_container_width=True)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=past_analyses['analysis_date'], 
                        y=past_analyses['actual_sales'], 
                        mode='markers+lines',
                        name='Actual',
                        marker=dict(color=['red' if x else 'green' for x in past_analyses['is_anomaly']])
                    ))
                    fig.add_trace(go.Scatter(
                        x=past_analyses['analysis_date'], 
                        y=past_analyses['expected_sales'], 
                        mode='lines',
                        name='Expected',
                        line=dict(dash='dash')
                    ))
                    fig.update_layout(title="Analysis History", height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No past analyses found")
            
            with tab5:
                st.subheader("Database Statistics")
                
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM sales_data")
                total_sales_records = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM drift_analysis")
                total_analyses = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT product_name) FROM sales_data")
                total_products = cursor.fetchone()[0]
                
                cursor.close()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Sales Records", total_sales_records)
                with col2:
                    st.metric("Total Analyses", total_analyses)
                with col3:
                    st.metric("Products Tracked", total_products)

if __name__ == "__main__":
    main()