import os
import joblib
import pandas as pd

# Definir la ruta a la carpeta de modelos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def realizar_prediccion(datos_entrada, tipo_modelo='lasso'):
    """
    Recibe un diccionario con los datos ambientales y devuelve la predicción de PM2.5.
    """
    try:
        # 1. Seleccionar la ruta del modelo elegido
        if tipo_modelo == 'lasso':
            modelo_path = os.path.join(MODELS_DIR, 'lasso.pkl')
        else:
            modelo_path = os.path.join(MODELS_DIR, 'decision_tree.pkl')
        
        # 2. Cargar el modelo
        if not os.path.exists(modelo_path):
            return f"Error: No se encontró el modelo en {modelo_path}"
        
        modelo = joblib.load(modelo_path)
        
        # 3. Convertir el diccionario de entrada a un DataFrame de pandas
        df_entrada = pd.DataFrame([datos_entrada])
        
        # Asegurarnos de usar exactamente las mismas columnas que en el entrenamiento
        columnas_requeridas = ['Hora', 'PM10', 'NO2', 'SO2', 'O3', 'CO']
        df_entrada = df_entrada[columnas_requeridas]
        
        # 4. Realizar la predicción
        prediccion = modelo.predict(df_entrada)
        
        # Retornar el valor redondeado a 2 decimales
        return round(float(prediccion[0]), 2)

    except Exception as e:
        return f"Error al predecir: {str(e)}"

# ==========================================
# Bloque de prueba manual
# ==========================================
if __name__ == '__main__':
    # Simulamos los datos que un usuario ingresaría en la página web
    datos_prueba = {
        'Hora': 14,
        'PM10': 55.5,
        'NO2': 20.1,
        'SO2': 5.0,
        'O3': 30.2,
        'CO': 1.1
    }
    
    print("=== Probando los archivos .pkl ===")
    
    resultado_lasso = realizar_prediccion(datos_prueba, tipo_modelo='lasso')
    print(f"Predicción PM2.5 (Lasso Regression): {resultado_lasso} µg/m³")
    
    resultado_tree = realizar_prediccion(datos_prueba, tipo_modelo='tree')
    print(f"Predicción PM2.5 (Decision Tree):    {resultado_tree} µg/m³")