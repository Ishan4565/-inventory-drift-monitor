import os
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

def train_universal_model():
    # Core ML logic remains untouched
    data = pd.DataFrame({
        'season_index': [1, 2, 3, 4, 1, 2, 3, 4],
        'temp': [15, 30, 10, -5, 18, 35, 8, -10],
        'promo': [0, 1, 0, 1, 1, 0, 0, 1],
        'past_sales': [300, 800, 400, 600, 350, 850, 420, 650],
        'actual': [320, 820, 390, 610, 360, 810, 400, 630]
    })
    X = data[['season_index', 'temp', 'promo', 'past_sales']]
    y = data['actual']
    return GradientBoostingRegressor().fit(X, y)

def predict_and_log(product, season_name, temp, promo, past, actual_sales, model):
    mapping = {"Spring": 1, "Summer": 2, "Fall": 3, "Winter": 4}
    s_index = mapping[season_name]
    
    input_data = pd.DataFrame([[s_index, temp, promo, past]], 
                               columns=['season_index', 'temp', 'promo', 'past_sales'])
    
    prediction = model.predict(input_data)[0]
    drift = abs(prediction - actual_sales) / (prediction + 1e-9)
    status = "Critical Drift" if drift > 0.2 else "Stable"
    
    # We are returning only the ML results to the Streamlit UI
    return prediction, drift
