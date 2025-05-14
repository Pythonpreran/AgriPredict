from flask import Flask, render_template, request, jsonify
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from io import BytesIO
import base64
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
matplotlib.use('Agg')

app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Route for the login page
@app.route('/login')
def login():
    return render_template('login.html')

# Route for the dashboard page
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# Placeholder routes for other pages (future pages)
@app.route('/aboutus')
def aboutus():
    return "About Us page coming soon..."

@app.route('/prediction')
def prediction():
    return render_template('commodities.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    selected_commodity = data['commodity'].title().strip()  # Normalize input
    selected_center = data['center']

    # Load and preprocess data
    df = pd.read_csv('synthetic_commodity_data.csv')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df.set_index('Date', inplace=True)

    # Filter data for the selected commodity and center
    if selected_center in df['Reporting_Centre'].unique() and selected_commodity in df['Commodity'].unique():
        df_filtered = df[(df['Reporting_Centre'] == selected_center) & (df['Commodity'] == selected_commodity)]
        df_filtered = df_filtered['Price'].resample('W').mean()
        df_filtered.fillna(method='ffill', inplace=True)
    else:
        return jsonify({"error": "Invalid Commodity or Center"}), 400

     # Set SARIMA parameters based on the commodity
    if selected_commodity == 'Soya Oil':
        sarima_order = (1, 3, 0)
        seasonal_order = (0, 0, 0, 52)  # RMSE: 5.03
    elif selected_commodity == 'Mustard Oil':
        sarima_order = (3, 3, 3)
        seasonal_order = (1, 1, 0, 52)  # RMSE: 4.3392
    elif selected_commodity == 'Potato':
        sarima_order = (2, 0, 4)
        seasonal_order = (0, 0, 0, 52)  # RMSE: 2.684
    elif selected_commodity == 'Sugar':
        sarima_order = (1, 1, 3)
        seasonal_order = (1, 0, 0, 52)  # RMSE: 0.9877
    elif selected_commodity == 'Rice':
        sarima_order = (3, 0, 3)
        seasonal_order = (1, 0, 0, 52)  # RMSE: 2.0723
    elif selected_commodity == 'Moong Dal':
        sarima_order = (2, 3, 3)
        seasonal_order = (1, 1, 0, 52)  # RMSE: 3.9831
    elif selected_commodity == 'Gram Dal':
        sarima_order = (1, 0, 3)
        seasonal_order = (1, 0, 0, 52)  # RMSE: 4.4731
    elif selected_commodity == 'Urad Dal':
        sarima_order = (3, 2, 0)
        seasonal_order = (0, 0, 0, 52)  # RMSE: 2.7388
    elif selected_commodity == 'Onion':
        sarima_order = (0, 0, 1)
        seasonal_order = (1, 0, 1, 52)  # RMSE: 2.664
    elif selected_commodity == 'Wheat':
        sarima_order = (2, 1, 3)
        seasonal_order = (1, 0, 0, 52)  # RMSE: 0.870
    elif selected_commodity == 'Groundnut Oil':
        sarima_order = (0, 3, 0)
        seasonal_order = (1, 0, 1, 52)  # RMSE: 2.319, High but acceptable

    # Split data into train and test sets
    train_size = int(len(df_filtered) * 0.8)
    train_df, test_df = df_filtered[:train_size], df_filtered[train_size:]

    # Fit SARIMA model
    sarima_model = SARIMAX(train_df, order=sarima_order, seasonal_order=seasonal_order)
    sarima_result = sarima_model.fit(disp=False)

    # Predict using SARIMA model for the test set
    sarima_predictions = sarima_result.predict(start=test_df.index[0], end=test_df.index[-1])
    residuals = test_df - sarima_predictions

    # Split residuals into train and test sets for XGBoost
    x = np.arange(len(residuals)).reshape(-1, 1)
    y = residuals.values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Fit XGBoost model to residuals
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    xgb_model.fit(x_train, y_train)

    # Predict residuals using XGBoost for test set
    xgb_residual_predictions = xgb_model.predict(x_test)

    # Combine SARIMA predictions with XGBoost residual predictions for test data
    xgb_full_residuals = xgb_model.predict(x)
    hybrid_test_predictions = sarima_predictions + xgb_full_residuals

    # Generate future predictions for SARIMA (1 month, 3 months, 6 months)
    weeks_in_month = 4
    weeks_in_3_months = 12
    weeks_in_6_months = 26

    future_steps = weeks_in_6_months  # Predict up to 6 months (26 weeks)
    sarima_future_predictions = sarima_result.get_forecast(steps=future_steps).predicted_mean

    # Create feature set for predicting residuals over the future steps
    future_x = np.arange(len(residuals), len(residuals) + future_steps).reshape(-1, 1)
    xgb_future_residuals = xgb_model.predict(future_x)

    # Combine SARIMA future predictions with XGBoost future residuals
    hybrid_future_predictions = sarima_future_predictions + xgb_future_residuals

    # Plot the results including future predictions
    plt.figure(figsize=(12, 6))
    plt.plot(df_filtered.index, df_filtered, label='Actual Prices', color='blue', marker='o', linestyle='-', linewidth=2, markersize=5)
    plt.plot(test_df.index, hybrid_test_predictions, label='SARIMA + XGBoost Predictions', color='red', marker='s', linestyle='--', linewidth=2, markersize=5)
    future_dates = pd.date_range(start=test_df.index[-1], periods=future_steps+1, freq='W')[1:]
    plt.title(f'{selected_commodity.title()} - {selected_center.title()}', fontsize=16, weight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (₹/Kg)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Convert plot to PNG image and then to base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode('utf-8')

    # Close the plot
    plt.close()

    # Extract predictions for 1 month, 3 months, and 6 months
    one_month_pred = round(hybrid_future_predictions[weeks_in_month - 1], 3) if future_steps >= weeks_in_month else None
    three_month_pred = round(hybrid_future_predictions[weeks_in_3_months - 1], 3) if future_steps >= weeks_in_3_months else None
    six_month_pred = round(hybrid_future_predictions[weeks_in_6_months - 1], 3) if future_steps >= weeks_in_6_months else None

    # Return JSON response including chart image
    return jsonify({
        'currentPrice': round(df_filtered.iloc[-1], 3),
        'oneMonthPrediction': one_month_pred,
        'threeMonthPrediction': three_month_pred,
        'sixMonthPrediction': six_month_pred,
        'chartData': {
            'chart': graph_url
        }
    })

@app.route('/nationalp', methods=['POST'])
def national_prediction():
    data = request.get_json()
    selected_commodity = data.get('commodity').title().strip().replace(' ', '_')  # Normalize input and replace spaces with underscores

    # Load and preprocess data for the selected commodity
    filename = f'average_daily_{selected_commodity.lower()}_prices.csv'
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        return jsonify({"error": "Data for the selected commodity is not available"}), 400

    # Data preprocessing
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.set_index('Date', inplace=True)
    df.dropna(subset=['Price'], inplace=True)

    # Resample to weekly data
    commodity_weekly = df['Price'].resample('W').mean()

    # Create a unique filename for the plot
    plot_filename = f'static/national_prediction_{selected_commodity.lower()}.png'

    # Prepare the static plot separately
    plt.figure(figsize=(12, 6))
    plt.plot(commodity_weekly, label='Historical Data', color='blue')
    plt.title(f'{selected_commodity.capitalize()} - National Average')
    plt.xlabel('Date')
    plt.ylabel('Price (₹/Kg)')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_filename)  # Save to a static folder
    plt.close()

    # Return the historical data results as JSON
    return jsonify({
        'currentPrice': round(df['Price'].iloc[-1], 3),
        'chartData': {
            'chart': f'/static/national_prediction_{selected_commodity.lower()}.png'  # Unique static image path
        },
    })









@app.route('/features')
def features():
    return "Features page coming soon..."

@app.route('/commodities')
def commodities():
    return "Commodities page coming soon..."

@app.route('/supply')
def supply():
    return render_template('disaster.html')

@app.route('/disaster')
def disaster():
    return render_template('disaster.html')
    

@app.route('/my_account')
def my_account():
    return "My Account page coming soon..."

if __name__ == '__main__':
    app.run()  # Set debug=True for development, False for production
