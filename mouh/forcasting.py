import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from django.db import connections
from datetime import date, timedelta
from io import BytesIO
import base64

def load_data_from_sql(id_service):
    with connections['default'].cursor() as cursor:
        cursor.execute("""
            SELECT date, nombre_operations
            FROM NOPJ
            WHERE id_service = %s
            ORDER BY date DESC
        """, [id_service])
        results = cursor.fetchall()
    
    data = pd.DataFrame(results, columns=['date', 'nombre_operations'])
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    return data.sort_index()

def fit_arima_model(data, order=(12, 1, 12)):
    model = ARIMA(data, order=order)
    fitted_model = model.fit()
    return fitted_model

def predict_future(fitted_model, steps=50):
    forecast = fitted_model.get_forecast(steps=steps)
    forecast_mean = forecast.predicted_mean
    confidence_intervals = forecast.conf_int()
    return forecast_mean, confidence_intervals

def calculate_errors(true_values, predicted_values):
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    mape = np.mean(np.abs((true_values - predicted_values) / true_values)) * 100
    return mae, rmse, mape

def plot_results(data, forecast, confidence_intervals):
    plt.figure(figsize=(10, 5))
    plt.plot(data, label='Observations')
    plt.plot(forecast, label='Prévisions', color='red')
    plt.fill_between(forecast.index,
                     confidence_intervals.iloc[:, 0],
                     confidence_intervals.iloc[:, 1],
                     color='pink', alpha=0.3)
    plt.legend()
    plt.title('Prévisions des opérations')
    plt.xlabel('Date')
    plt.ylabel('Nombre d\'opérations')
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')
    
    return graphic

def generate_forecast(id_service):
    data = load_data_from_sql(id_service)
    data_series = data['nombre_operations']
    
    if data_series.empty:
        return None, None, None, None
    
    fitted_model = fit_arima_model(data_series)
    forecast, confidence_intervals = predict_future(fitted_model, steps=50)
    
    observed_future_values = [data_series[-1]] * len(forecast)
    mae, rmse, mape = calculate_errors(observed_future_values, forecast)
    
    graphic = plot_results(data_series, forecast, confidence_intervals)
    
    return mae, rmse, mape, graphic