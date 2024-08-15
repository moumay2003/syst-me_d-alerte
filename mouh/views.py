from django.shortcuts import render
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import io
import base64
from django.db import connection
from django.shortcuts import render, redirect
from django.contrib import messages
from datetime import date, datetime, timedelta
from django.utils import timezone
from django.db.models import Count
from django.db.models.functions import TruncHour
from .forcasting import generate_forecast

# views.py
import pyodbc
from django.shortcuts import render

def index(request):
    # Connexion à la base de données SQL Server
    conn_str = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=.;DATABASE=stage4;UID=sa;PWD=May2015++'
    conn = pyodbc.connect(conn_str)

    # Créer un curseur
    cursor = conn.cursor()

    # Exécuter une requête SQL pour récupérer les 50 dernières lignes
    cursor.execute("SELECT TOP 50 * FROM transactionsmain2 ORDER BY date DESC")
    rows = cursor.fetchall()

    # Passer les résultats au template
    return render(request, 'index2.html', {'rows': rows})    

def forecast_view(request):
    context = {}
    if request.method == 'POST':
        id_service = request.POST.get('id_service')
        mae, rmse, mape, graphic = generate_forecast(int(id_service))
        
        context = {
            'id_service': id_service,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'graphic': graphic,
        }
    
    return render(request, 'forecast.html', context)


def plot_operations(request):
    message = None
    graphic = None

    if request.method == 'POST':
        date = request.POST.get('date')
        service_id = request.POST.get('service_id')
        
        # Votre requête SQL
        query = f"""
        SELECT DATEPART(HOUR, heure) as hour, COUNT(*) as count
        FROM transactionsmain2
        WHERE date = '{date}' AND id_service = {service_id}
        GROUP BY DATEPART(HOUR, heure)
        ORDER BY DATEPART(HOUR, heure)
        """
        
        # Exécuter la requête
        with connection.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()
        
        if not results:
            message = "Aucune donnée trouvée pour la date et l'ID de service spécifiés."
        else:
            # Convertir les résultats en DataFrame
            df = pd.DataFrame(results, columns=['hour', 'count'])
            
            # Créer un DataFrame avec toutes les heures (0-23)
            all_hours = pd.DataFrame({'hour': range(24)})
            df = all_hours.merge(df, on='hour', how='left').fillna(0)
            
            # Créer le graphique
            plt.figure(figsize=(12, 6))
            plt.bar(df['hour'], df['count'], width=0.8)
            plt.title(f"Nombre d'opérations par heure pour le service {service_id} le {date}")
            plt.xlabel("Heure")
            plt.ylabel("Nombre d'opérations")
            plt.xticks(range(24))
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            for i, v in df.iterrows():
                plt.text(v['hour'], v['count'], str(int(v['count'])), ha='center', va='bottom')
            
            # Sauvegarder le graphique en format base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            graphic = base64.b64encode(image_png)
            graphic = graphic.decode('utf-8')
    
    return render(request, 'plot_operations.html', {'graphic': graphic, 'message': message})

from django.shortcuts import render
import pyodbc
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def resultats_comparaison(request):
    resultats = None
    error = None
    total_operations = 0
    moyenne_operations = 0
    ecart_max = 0
    graph_image = None

    if request.method == 'POST':
        service_id = int(request.POST.get('service_id'))
        date_input = request.POST.get('date')
        date = datetime.strptime(date_input, "%Y-%m-%d")

        # Connexion à la base de données
        conn_str = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=.;DATABASE=stage4;UID=sa;PWD=May2015++'
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        # Récupération de la profondeur moyenne
        cursor.execute("SELECT TOP 1 profondeur_moyenne FROM transactionsmain2 WHERE id_service = ?", service_id)
        result = cursor.fetchone()
        if result:
            profondeur_moyenne = result[0]
        else:
            error = f"Aucune profondeur moyenne trouvée pour le service {service_id}"
            return render(request, 'resultats_comparaison.html', {'error': error})

        date_debut = date - timedelta(days=30*profondeur_moyenne)
        nombre_jours = (date - date_debut).days + 1

        query = """
        SELECT DATEPART(HOUR, heure) as heure, COUNT(*) as nb_operations
        FROM transactionsmain2
        WHERE date BETWEEN ? AND ?
        AND id_service = ?
        GROUP BY DATEPART(HOUR, heure)
        """

        df_jour = pd.read_sql(query, conn, params=(date, date, service_id))
        df_periode = pd.read_sql(query, conn, params=(date_debut, date, service_id))

        df_moyenne = df_periode.groupby('heure')['nb_operations'].sum().reset_index()
        df_moyenne['nb_operations_moyenne'] = df_moyenne['nb_operations'] / nombre_jours

        df_resultat = pd.merge(df_jour, df_moyenne[['heure', 'nb_operations_moyenne']], on='heure')
        df_resultat['difference'] = df_resultat['nb_operations'] - df_resultat['nb_operations_moyenne']
        df_resultat['inferieur_a_la_moyenne'] = df_resultat['difference'] < 0

        df_resultat_filtre = df_resultat[df_resultat['inferieur_a_la_moyenne']]

        # Création et insertion dans la table ResultatsComparaison
        cursor.execute("""
        IF OBJECT_ID('ResultatsComparaison', 'U') IS NOT NULL
            DROP TABLE ResultatsComparaison
        CREATE TABLE ResultatsComparaison (
            date DATE,
            service_id INT,
            heure INT,
            nb_operations INT,
            nb_operations_moy FLOAT,
            difference FLOAT
        )
        """)

        for _, row in df_resultat_filtre.iterrows():
            cursor.execute("""
                INSERT INTO ResultatsComparaison (date, service_id, heure, nb_operations, nb_operations_moy, difference)
                VALUES (?, ?, ?, ?, ?, ?)
                """, date, service_id, int(row['heure']), int(row['nb_operations']), float(row['nb_operations_moyenne']), float(row['difference']))

        conn.commit()

        # Récupération des résultats pour l'affichage
        cursor.execute("SELECT * FROM ResultatsComparaison WHERE date = ? AND service_id = ?", date, service_id)
        resultats = cursor.fetchall()

        if resultats:
            total_operations = sum(r[3] for r in resultats)
            moyenne_operations = sum(r[4] for r in resultats) / len(resultats)
            ecart_max = max(abs(r[5]) for r in resultats)

        # Génération du graphique si demandé
        if 'analyse_graphique' in request.POST:
            graph_image = plot_operations_per_hour(date_input, service_id, conn)

        conn.close()

    return render(request, 'resultats_comparaison.html', {
        'resultats': resultats, 
        'error': error,
        'total_operations': total_operations,
        'moyenne_operations': moyenne_operations,
        'ecart_max': ecart_max,
        'graph_image': graph_image
    })

def plot_operations_per_hour(date, service_id, conn):
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    cursor = conn.cursor()
    cursor.execute("SELECT TOP 1 profondeur_moyenne FROM transactionsmain2 WHERE id_service = ?", service_id)
    result = cursor.fetchone()
    if result:
        profondeur_moyenne = result[0]
    else:
        return None

    date_debut = date_obj - timedelta(days=30*profondeur_moyenne)
    nombre_jours = (date_obj - date_debut).days + 1

    query_jour = f"""
     SELECT DATEPART(HOUR, heure) as hour, COUNT(*) as count
     FROM transactionsmain2
     WHERE date = '{date}' AND id_service = {service_id}
     GROUP BY DATEPART(HOUR, heure)
     """
    query_periode = f"""
     SELECT DATEPART(HOUR, heure) as hour, COUNT(*) as count
     FROM transactionsmain2
     WHERE date BETWEEN '{date_debut.strftime('%Y-%m-%d')}' AND '{date}'
     AND id_service = {service_id}
     GROUP BY DATEPART(HOUR, heure)
     """

    df_jour = pd.read_sql(query_jour, conn)
    df_periode = pd.read_sql(query_periode, conn)

    df_moyenne = df_periode.groupby('hour')['count'].sum().reset_index()
    df_moyenne['count_moyenne'] = df_moyenne['count'] / nombre_jours

    all_hours = pd.DataFrame({'hour': range(24)})
    df_jour = all_hours.merge(df_jour, on='hour', how='left').fillna(0)
    df_moyenne = all_hours.merge(df_moyenne, on='hour', how='left').fillna(0)

    plt.figure(figsize=(14, 7))
    bars = plt.bar(df_jour['hour'], df_jour['count'], width=0.8, label='Nombre d\'opérations')
    plt.plot(df_moyenne['hour'], df_moyenne['count_moyenne'], color='red', linestyle='--', label='Moyenne')

    for i, (bar, jour_count, moy_count) in enumerate(zip(bars, df_jour['count'], df_moyenne['count_moyenne'])):
        if jour_count < moy_count:
            bar.set_color('orange')

    plt.title(f"Nombre d'opérations par heure pour le service {service_id} le {date}")
    plt.xlabel("Heure")
    plt.ylabel("Nombre d'opérations")
    plt.xticks(range(24))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, v in df_jour.iterrows():
        plt.text(v['hour'], v['count'], str(int(v['count'])), ha='center', va='bottom')

    plt.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return image_base64
import pyodbc
from django.shortcuts import render

def my_view(request):
    # Connexion à la base de données SQL Server
    conn_str = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=.;DATABASE=stage2;UID=sa;PWD=May2015++'
    conn = pyodbc.connect(conn_str)

    # Créer un curseur
    cursor = conn.cursor()

    # Exécuter une requête SQL pour récupérer les 50 dernières lignes
    cursor.execute("SELECT TOP 50 * FROM transactionsmain2 ORDER BY ID DESC")
    rows = cursor.fetchall()

    # Passer les résultats au template
    return render(request, 'index2.html', {'rows': rows})


from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pyodbc
from .task import load_data_from_sql, fit_arima_model, predict_future, save_model,model_exists,load_model
import matplotlib.pyplot as plt
import io
import base64

conn_str = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=.;DATABASE=stage3;UID=sa;PWD=May2015++'

def get_services():
    with pyodbc.connect(conn_str) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT id_service FROM NOPJ")
        return [{'id': row.id_service, 'name': f'Service {row.id_service}'} for row in cursor.fetchall()]

def should_generate_new_model(last_generated_time):
    """Vérifie si 30 minutes se sont écoulées depuis la dernière génération du modèle."""
    return datetime.now() - last_generated_time > timedelta(minutes=30)

@csrf_exempt
def forecast_view2(request):
    if request.method == 'POST':
        id_service = request.POST.get('id_service')
        
        try:
            # Charger le modèle existant si disponible
            if model_exists(id_service):
                fitted_model, model_date = load_model(id_service)
                if should_generate_new_model(model_date):
                    # Générer un nouveau modèle si 30 minutes sont passées
                    data = load_data_from_sql(id_service)
                    data_series = data['nombre_operations']
                    fitted_model = fit_arima_model(data_series)
                    save_model(fitted_model, id_service)
                    message = f"Nouveau modèle généré pour le service {id_service}."
                else:
                    message = f"Le modèle existant pour le service {id_service} est encore récent (généré le {model_date})."
                    # Charger les données pour le graphe
                    data = load_data_from_sql(id_service)
                    data_series = data['nombre_operations']
            else:
                # Charger les données et générer un nouveau modèle
                data = load_data_from_sql(id_service)
                data_series = data['nombre_operations']
                fitted_model = fit_arima_model(data_series)
                save_model(fitted_model, id_service)
                message = f"Nouveau modèle généré pour le service {id_service}."
            
            # Faire des prévisions et générer le graphe
            forecast_mean, confidence_intervals = predict_future(fitted_model)
            
            plt.figure(figsize=(10, 5))
            plt.plot(data_series, label='Observations')
            plt.plot(forecast_mean, label='Prévisions', color='red')
            plt.fill_between(forecast_mean.index,
                             confidence_intervals.iloc[:, 0],
                             confidence_intervals.iloc[:, 1],
                             color='pink', alpha=0.3)
            plt.legend()
            plt.title(f'Prévisions des opérations pour le service {id_service}')
            plt.xlabel('Date')
            plt.ylabel('Nombre d\'opérations')
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            graphic = base64.b64encode(image_png).decode('utf-8')
            
            return JsonResponse({'status': 'success', 'graphic': graphic, 'message': message})
        
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    services = get_services()
    return render(request, 'forecast2.html', {'services': services})
