import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import os
import pyodbc
from celery import shared_task

# Configuration de la connexion à la base de données
conn_str = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=.;DATABASE=stage3;UID=sa;PWD=May2015++'

def load_data_from_sql(id_service):
    query = """
    SELECT date, nombre_operations
    FROM NOPJ
    WHERE id_service = ?
    ORDER BY date DESC
    """
    with pyodbc.connect(conn_str) as conn:
        data = pd.read_sql(query, conn, params=(id_service,), index_col='date', parse_dates=['date'])
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

def save_model(model, id_service):
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, f'arima_model_{id_service}.pkl')
    # Sauvegarder le modèle avec le temps actuel
    joblib.dump((model, datetime.now()), model_path)
    print(f"Modèle sauvegardé : {model_path}")

def get_all_service_ids():
    with pyodbc.connect(conn_str) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT id_service FROM NOPJ")
        return [row.id_service for row in cursor.fetchall()]
import os

def model_exists(id_service):
    model_path = os.path.join('models', f'arima_model_{id_service}.pkl')
    return os.path.exists(model_path)
def load_model(id_service):
    model_path = os.path.join('models', f'arima_model_{id_service}.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Le modèle pour le service {id_service} n'existe pas.")

@shared_task
def generate_and_save_model(id_service):
    try:
        if model_exists(id_service):
            print(f"Modèle déjà existant pour le service {id_service}. Aucune action nécessaire.")
            return  # Si le modèle existe, on ne fait rien.

        data = load_data_from_sql(id_service)
        data_series = data['nombre_operations']
        fitted_model = fit_arima_model(data_series)
        save_model(fitted_model, id_service)
        print(f"Modèle généré et sauvegardé pour le service {id_service}")
    except Exception as e:
        print(f"Erreur lors de la génération du modèle pour le service {id_service}: {str(e)}")


@shared_task
def generate_all_models():
    service_ids = get_all_service_ids()
    for service_id in service_ids:
        data = load_data_from_sql(service_id)
        data_series = data['nombre_operations']
        fitted_model = fit_arima_model(data_series)
        save_model(fitted_model, service_id)
    print("Tâche de génération de tous les modèles terminée")

@shared_task
def clean_old_models(days=7):
    model_dir = 'models'
    now = datetime.now()
    for filename in os.listdir(model_dir):
        file_path = os.path.join(model_dir, filename)
        if os.path.isfile(file_path):
            file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
            if (now - file_modified).days > days:
                os.remove(file_path)
                print(f"Ancien modèle supprimé : {file_path}")

def calculate_errors(true_values, predicted_values):
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    return mae, rmse