from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib
import os

app = Flask(__name__)

# Helper function to find the close price column
def find_close_column(df):
    possible_names = ['Close', 'close', 'CLOSE', 'Close Price', 'Closing Price',
                      'Close_Price', 'Adj Close', 'adj_close', 'ADJ_CLOSE']
    for col in df.columns:
        if col.strip() in possible_names:
            return col
    return None

@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/abstract')
def abstract():
    return render_template('abstract.html')

@app.route('/login')
def login():
    return render_template('login.html')




@app.route('/predict', methods=['POST'])
def predict():
    try:
        stock_name = request.form['stock']

        stock_mapping = {
            'NIFTY_IT': ('data/NIFTY IT_data.csv', 'nifty_it_model.h5'),
            'NIFTY_FIN': ('data/NIFTY FIN SERVICE_data.csv', 'nifty_fin_service_model.h5'),
            'NIFTY_METAL': ('data/NIFTY METAL_data.csv', 'nifty_metal_model.h5')
        }

        if stock_name not in stock_mapping:
            return jsonify({'error': 'Unknown stock selected'}), 400

        csv_file, model_file = stock_mapping[stock_name]
        df = pd.read_csv(csv_file)

        close_column = find_close_column(df)
        if not close_column:
            return jsonify({'error': f'No close price column found. Available columns: {df.columns.tolist()}'}), 400

        close_data = df[[close_column]].values
        if len(close_data) < 60:
            return jsonify({'error': 'Not enough data points'}), 400

        model_path = f'models/{model_file}'
        scaler_path = model_path.replace(".h5", "_scaler.pkl")

        model = load_model(model_path)
        scaler = joblib.load(scaler_path)

        scaled_data = scaler.transform(close_data)
        last_60 = scaled_data[-60:].reshape(1, 60, 1)

        predicted_scaled = model.predict(last_60)
        predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]
        last_actual_price = float(close_data[-1][0])
        percentage_change = ((predicted_price - last_actual_price) / last_actual_price) * 100

        last_5_dates = df['date'].astype(str).tolist()[-5:] if 'date' in df.columns else ['N/A'] * 5
        last_5_prices = [float(x) for x in df[close_column].tolist()[-5:]]

        return render_template('index.html',
            predicted_price=round(predicted_price, 2),
            last_actual_price=round(last_actual_price, 2),
            percentage_change=round(percentage_change, 2),
            model_used=model_file,
            last_5_dates=last_5_dates,
            last_5_prices=last_5_prices
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    available_models = []
    if os.path.exists('models'):
        available_models = [f for f in os.listdir('models') if f.endswith('.h5')]

    return jsonify({
        'status': 'healthy',
        'available_models': available_models
    })

if __name__ == '__main__':
    if not os.path.exists('models'):
        os.makedirs('models')
    app.run(debug=True)
