import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def entrenar_modelos():
    print("=== Iniciando el proceso de entrenamiento ===")
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'datos_horarios_contaminacion_lima_limpio.xlsx')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    
    # Verificar que el dataset exista de forma local
    if not os.path.exists(DATA_PATH):
        print(f"Error: No se encontró el archivo en {DATA_PATH}")
        return

    # 2. Cargar el dataset limpio
    print("Cargando datos...")
    df = pd.read_excel(DATA_PATH)  
    
    # 3. Separar Características (X) y Variable Objetivo (y)
    # NOTA: Asegúrate de que los nombres coincidan exactamente con las columnas de tu CSV.
    # Si tu procesamiento generó columnas dummies para las estaciones, inclúyelas aquí.
    COLUMNAS_INPUT = ['HORA', 'PM 10', 'NO2', 'SO2', 'O3', 'CO'] 
    VARIABLE_TARGET = 'PM 2.5'
    
    # Filtramos solo las columnas que existan en el DataFrame para evitar caídas
    columnas_validas = [c for c in COLUMNAS_INPUT if c in df.columns]
    # Si hiciste codificación One-Hot para las estaciones, puedes descomentar la siguiente línea:
    # columnas_validas = [c for c in df.columns if c != VARIABLE_TARGET and c != 'Fecha']

    X = df[columnas_validas]
    y = df[VARIABLE_TARGET]
    
    print(f"Variables predictoras utilizadas: {list(X.columns)}")
    print(f"Variable a predecir: {VARIABLE_TARGET}")

    # 4. Dividir en set de entrenamiento (80%) y prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Instanciar y entrenar Lasso Regression
    print("Entrenando Lasso Regression...")
    modelo_lasso = Lasso(alpha=1.0, random_state=42)
    modelo_lasso.fit(X_train, y_train)
    
    # 6. Instanciar y entrenar Decision Tree Regressor
    print("Entrenando Decision Tree Regressor...")
    modelo_tree = DecisionTreeRegressor(max_depth=10, random_state=42) # Ajusta max_depth según veas el overfitting
    modelo_tree.fit(X_train, y_train)
    
    # 7. Guardar los modelos entrenados en la carpeta models/
    os.makedirs(MODELS_DIR, exist_ok=True) # Asegura que la carpeta exista
    
    lasso_path = os.path.join(MODELS_DIR, 'lasso.pkl')
    tree_path = os.path.join(MODELS_DIR, 'decision_tree.pkl')
    
    joblib.dump(modelo_lasso, lasso_path)
    joblib.dump(modelo_tree, tree_path)
    
    print(f"¡Modelos entrenados con éxito!")
    print(f"-> Guardado: {lasso_path}")
    print(f"-> Guardado: {tree_path}")
    print("==============================================")

if __name__ == '__main__':
    entrenar_modelos()