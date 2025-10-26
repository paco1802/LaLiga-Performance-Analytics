import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sns

# --- Configuración y Rutas ---
# El dataset matches_laliga.csv ya está en la carpeta data/
DATA_FOLDER = 'data'
INPUT_FILE = os.path.join(DATA_FOLDER, 'matches_laliga.csv')
OUTPUT_FILE = os.path.join(DATA_FOLDER, 'processed_data.csv')
IMAGES_FOLDER = 'images'
WINDOW_SIZE = 5 # Ventana de partidos para el promedio móvil

def run_data_prep():
    # TEMPORAL: Este es el primer print que debería aparecer
    print("--- 🟢 TEST DE EJECUCIÓN EXITOSO ---") 

    

    # A. CARGA Y CONCATENACIÓN DE DATOS
    try:
        # El archivo ya está 'desplegado' (una fila por equipo/partido)
        df_raw = pd.read_csv(INPUT_FILE)
        print(f"Cargado el dataset: {INPUT_FILE}. Filas totales: {len(df_raw)}")
    except FileNotFoundError:
        print(f"ERROR: Archivo no encontrado. Asegúrate de que '{INPUT_FILE}' esté en la carpeta 'data/'.")
        return

    # Asegurarse de que las carpetas existan
    os.makedirs(IMAGES_FOLDER, exist_ok=True)
    os.makedirs(DATA_FOLDER, exist_ok=True)
    
    # Continuar con el Feature Engineering
    df_final = data_cleaning_and_feature_engineering(df_raw)

    # G. GUARDADO FINAL Y RESUMEN
    print(f"\nGuardando el DataFrame procesado en: {OUTPUT_FILE}")
    
    # Columnas finales esenciales para el modelado
    cols_to_keep = [
        'Date', 'Season', 'HomeTeam', 'AwayTeam', 'Target', 
        'Feature_Delta_xG', 'Feature_Delta_Points', 
        # Incluimos las features base para posibles análisis futuros
        'Home_xG', 'Away_xG', 'Home_Possession', 'Away_Possession' 
    ]
    df_final[cols_to_keep].to_csv(OUTPUT_FILE, index=False)
    
    print("¡Preparación de datos completada con éxito!")


def calculate_points(goals_diff):
    """Calcula los puntos ganados en un partido."""
    if goals_diff > 0:
        return 3
    elif goals_diff == 0:
        return 1
    return 0


def data_cleaning_and_feature_engineering(df_actuaciones):
    """
    Limpia, crea métricas de partido y aplica medias móviles.
    """
    print("\n--- B. Limpieza y C. Creación de Métricas de Partidos ---")
    
    # B. LIMPIEZA BÁSICA y estandarización de nombres (según matches_laliga.csv)
    df_actuaciones = df_actuaciones.rename(columns={
        'date': 'Date', 'team': 'Team', 'opponent': 'Opponent', 
        'gf': 'Goals_For', 'ga': 'Goals_Against', 
        'xg': 'xG_For', 'xga': 'xG_Against', 
        'poss': 'Possession', 'season': 'Season', 'venue': 'Venue'
    })
    
    # Convertir la fecha y ordenar
    df_actuaciones['Date'] = pd.to_datetime(df_actuaciones['Date'])
    
    # Crear métricas de partido para cada fila (actuación del equipo)
    df_actuaciones['Goals_Diff'] = df_actuaciones['Goals_For'] - df_actuaciones['Goals_Against']
    df_actuaciones['xG_Diff'] = df_actuaciones['xG_For'] - df_actuaciones['xG_Against']
    
    # C. CÁLCULO DE PUNTOS POR PARTIDO
    df_actuaciones['Points'] = df_actuaciones['Goals_Diff'].apply(calculate_points)

    
    print("\n--- D. Cálculo de Promedios Móviles (Features) ---")

    # Ordena por Equipo y Fecha. ¡CRUCIAL!
    df_actuaciones = df_actuaciones.sort_values(by=['Team', 'Date']).reset_index(drop=True)

    # CÁLCULO DE PROMEDIOS MÓVILES (LAST 5 GAMES)
    # .shift(1) garantiza que solo se usen datos anteriores al partido actual.
    # min_periods=1 permite tener un valor (media de 1, 2, 3 o 4 partidos) al inicio de la temporada.

    # 1. Promedio móvil de Diferencial de xG (la feature principal)
    df_actuaciones['Avg_xG_Diff_Last_5'] = df_actuaciones.groupby('Team')['xG_Diff'].transform(
        lambda x: x.shift(1).rolling(window=WINDOW_SIZE, min_periods=1).mean()
    )
    # 2. Promedio móvil de Puntos (rendimiento reciente)
    df_actuaciones['Avg_Points_Last_5'] = df_actuaciones.groupby('Team')['Points'].transform(
        lambda x: x.shift(1).rolling(window=WINDOW_SIZE, min_periods=1).mean()
    )
    
    print(f"Calculadas las features de media móvil para {WINDOW_SIZE} partidos.")
    
    # E. RE-UNIFICACIÓN Y CREACIÓN DE FEATURES DIFERENCIALES
    
    # 1. Filtrar solo filas de partidos como local (HOME) para obtener la vista de "partido"
    df_home = df_actuaciones[df_actuaciones['Venue'] == 'Home'].copy()
    
    # 2. Definición de la Variable Objetivo (Solo para la fila del partido HOME)
    # (El Goal_Diff de la fila 'Home' es la diferencia del partido)
    df_home['Target'] = df_home['Goals_Diff'].apply(lambda x: 1 if x > 0 else (0 if x < 0 else 2))

    # Seleccionar y renombrar las features del Equipo Local
    df_home = df_home.rename(columns={
        'Avg_xG_Diff_Last_5': 'Home_Avg_xG_Diff_Last_5',
        'Avg_Points_Last_5': 'Home_Avg_Points_Last_5',
        'Team': 'HomeTeam', 'Opponent': 'AwayTeam', 
        'Possession': 'Home_Possession',
        'xG_For': 'Home_xG', 'xG_Against': 'Away_xG'
    })

    # Seleccionar y renombrar las features del Equipo Visitante (las usaremos para el merge)
    df_away_features = df_actuaciones[['Date', 'Team', 'Avg_xG_Diff_Last_5', 'Avg_Points_Last_5', 'Possession']].copy()
    df_away_features = df_away_features.rename(columns={
        'Team': 'AwayTeam',
        'Avg_xG_Diff_Last_5': 'Away_Avg_xG_Diff_Last_5',
        'Avg_Points_Last_5': 'Away_Avg_Points_Last_5',
        'Possession': 'Away_Possession'
    })
    
    # Merge: Unir el rendimiento del Visitante al partido del Local usando Date y AwayTeam
    df_final = pd.merge(df_home, df_away_features, 
                         left_on=['Date', 'AwayTeam'], 
                         right_on=['Date', 'AwayTeam'], 
                         how='inner')
    
    # CREACIÓN DE FEATURES DIFERENCIALES (Nuestras X finales)
    df_final['Feature_Delta_xG'] = df_final['Home_Avg_xG_Diff_Last_5'] - df_final['Away_Avg_xG_Diff_Last_5']
    df_final['Feature_Delta_Points'] = df_final['Home_Avg_Points_Last_5'] - df_final['Away_Avg_Points_Last_5']
    
    # Puedes añadir un feature de Localía (dummy variable 1/0, ya implícito, o simplemente 1 ya que todas las filas son Home)
    df_final['Feature_Home_Advantage'] = 1 

    print("Creadas las Features Diferenciales (Delta xG y Delta Puntos).")
    
    # F. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
    print("\n--- F. Verificación de Features (EDA) ---")
    
    # Boxplot para validar si Delta_xG es predictivo
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Target', y='Feature_Delta_xG', data=df_final, 
                palette={1: 'green', 0: 'red', 2: 'orange'})
    plt.title('Feature Delta xG por Resultado (1=Victoria, 0=Derrota, 2=Empate)')
    plt.xlabel('Resultado del Equipo Local')
    plt.ylabel('Diferencia de xG Promedio (Local - Visitante)')
    plt.xticks([0, 1, 2], ['Derrota', 'Victoria', 'Empate'])
    
    plot_path = os.path.join(IMAGES_FOLDER, 'delta_xg_boxplot.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Gráfico de Boxplot de Delta xG guardado en: {plot_path}")
    
    # Imprimir un resumen
    print("\nPrimeras 5 filas del DataFrame procesado con las Features:")
    print(df_final[['Date', 'HomeTeam', 'AwayTeam', 'Target', 
                     'Feature_Delta_xG', 'Feature_Delta_Points']].head())
    
    return df_final

if __name__ == '__main__':
    run_data_prep() 
    print("--- 🟢 FIN DEL SCRIPT ---") # Nuevo mensaje de fin
