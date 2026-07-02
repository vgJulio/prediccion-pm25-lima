# Sistema Inteligente de Predicción de PM2.5 en Lima

Aplicación web desarrollada para predecir la concentración de material particulado fino (PM2.5) en Lima Metropolitana mediante técnicas de Machine Learning, utilizando los modelos **Lasso Regression** y **Decision Tree Regressor**.

El sistema permite comparar el desempeño de ambos modelos a través de métricas de evaluación, visualizaciones y predicciones sobre nuevos datos.

---

## Objetivos

- Predecir la concentración de PM2.5 utilizando algoritmos de regresión.
- Comparar el rendimiento de Lasso Regression y Decision Tree Regressor.
- Analizar la influencia de las variables ambientales en la contaminación del aire.
- Visualizar los resultados mediante gráficos y reportes.

---

## Modelos Implementados

### Lasso Regression

Modelo de regresión lineal con regularización L1.

Permite:

- Predecir valores de PM2.5.
- Identificar las variables más relevantes.
- Reducir el efecto de variables poco importantes.

---

### Decision Tree Regressor

Modelo de aprendizaje supervisado basado en árboles de decisión.

Permite:

- Capturar relaciones no lineales.
- Encontrar patrones entre las variables ambientales.
- Explicar las reglas utilizadas para realizar una predicción.

---

## Variables del Proyecto

Variables utilizadas como entrada del modelo:

- Hora
- Fecha
- Estación de monitoreo
- PM10
- NO₂
- SO₂
- O₃
- CO

Variable objetivo:

- PM2.5

---

## Tecnologías Utilizadas

- Python 3.13
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- OpenPyXL
- PySide6 / Flask (según la versión final)
- Joblib

---

## Estructura del Proyecto

```
Proyecto/
│
├── app.py
├── requirements.txt
│
├── data/
│   ├── datos_horarios_contaminacion_lima.xlsx
│   └── datos_limpios.csv
│
├── controllers/
│   ├── train.py
│   ├── predict.py
│   └── metrics.py
│
├── models/
│   ├── lasso.pkl
│   └── decision_tree.pkl
│
├── graphics/
│   ├── lasso/
│   ├── decision_tree/
│   └── comparison/
│
├── reports/
│
├── templates/
│
├── static/
│
└── README.md
```

---

## Funcionalidades

- Carga del conjunto de datos.
- Entrenamiento automático de modelos.
- Predicción de nuevos registros.
- Comparación entre modelos.
- Visualización de métricas.
- Generación de gráficos.
- Exportación de resultados.

---

## Métricas de Evaluación

El sistema compara ambos modelos mediante:

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score

---

## Resultados Esperados

Se espera que:

- **Decision Tree Regressor** obtenga una mayor precisión al modelar relaciones no lineales presentes en los datos.
- **Lasso Regression** permita interpretar la importancia de las variables mediante sus coeficientes, facilitando el análisis de los factores que influyen en la concentración de PM2.5.

---

## Instalación

Clonar el repositorio:

```bash
git clone https://github.com/usuario/prediccion-pm25-lima.git
```

Ingresar al proyecto:

```bash
cd prediccion-pm25-lima
```

Crear entorno virtual:

```bash
python -m venv venv
```

Activar entorno virtual:

Windows

```bash
venv\Scripts\activate
```

Linux

```bash
source venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar la aplicación:

```bash
python app.py
```

---

## Autores

Proyecto desarrollado para el curso de Inteligencia Artificial.

Universidad

Ingeniería Informática

2026