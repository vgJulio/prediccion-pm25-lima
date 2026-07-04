# Prediccion PM2.5 Lima

Proyecto para limpiar datos horarios de contaminacion en Lima, entrenar dos
modelos de regresion y explorar los resultados en un dashboard Flask con
analisis en vivo (pandas, numpy, matplotlib y scikit-learn corriendo al
momento del request, no solo en los scripts offline).

## Objetivo

Predecir la concentracion de PM2.5 usando contaminantes atmosfericos,
fecha, hora y estacion de monitoreo.

## Modelos

- Lasso Regression
- Decision Tree Regressor

## Paginas del dashboard

| Ruta | Que muestra |
|---|---|
| `/` | Inicio: resumen del proyecto y acceso a todas las secciones. |
| `/dashboard` | Metricas, importancia de variables y reglas del arbol de decision. |
| `/comparacion` | Lasso vs Decision Tree lado a lado, con boton para recalcular metricas en vivo con `sklearn.metrics`. |
| `/modelo-matematico` | Ecuacion real de Lasso (coeficientes + intercepto) y explicacion matematica del arbol (criterio de division, prediccion por hoja). |
| `/exploracion` | EDA en vivo: `describe()`, correlaciones, promedio por estacion y por hora, filtrable por estacion. Incluye la prueba de hipotesis de horas punta (con `scipy.stats`). Histograma generado en memoria con matplotlib. |
| `/residuales` | Error (MAE) y sesgo de cada modelo por estacion, calculado sobre `data/test/datos_prueba.csv`. |
| `/mapa` | Mapa interactivo (Leaflet) con el promedio historico de PM2.5 por estacion. |
| `/resultados` | Formulario de prediccion: calcula ambos modelos, clasifica la calidad de aire y guarda un historial. |
| `/api/predecir` (POST JSON) | Version API de `/resultados`, sin HTML. |

## Estructura

```text
prediccion-pm25-lima-Julio/
├── app/
│   ├── app.py            # Rutas Flask
│   ├── predictor.py       # Carga de modelos, prediccion, historial
│   ├── analytics.py       # EDA, graficos y metricas en vivo (pandas/matplotlib/sklearn)
│   ├── static/            # CSS y JS
│   └── templates/         # Paginas del dashboard
├── data/
│   ├── raw/               # Dataset original
│   ├── processed/         # Datos limpios/modelo
│   └── test/               # Datos de prueba
├── models/                # Modelos .pkl
├── notebooks/             # Notebooks del proyecto
├── reports/                # Metricas, imagenes y historial de predicciones
├── scripts/                # Limpieza, entrenamiento y evaluacion
├── app.py                  # Entrada simple
├── requirements.txt
└── README.md
```

## Archivos importantes

- `data/raw/datos_horarios_contaminacion_lima.xlsx`: dataset sucio original.
- `notebooks/01_Limpieza_Datos.ipynb`: notebook de limpieza trabajado.
- `notebooks/04_Lasso_Regression.ipynb`: implementacion del modelo Lasso.
- `notebooks/05_Decision_Tree_Regressor.ipynb`: implementacion del arbol de decision.
- `notebooks/06_Comparacion_Modelos.ipynb`: comparacion de resultados.

El notebook de regresion logistica solo se uso como referencia de estilo para
organizar los notebooks; no forma parte del entrenamiento de PM2.5.

## Ejecucion

Activar el entorno:

```bash
venv\Scripts\activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Limpiar datos:

```bash
python scripts/limpiar_datos.py
```

Entrenar modelos:

```bash
python scripts/entrenar_modelos.py
```

Ver metricas:

```bash
python scripts/evaluar_modelos.py
```

Abrir dashboard:

```bash
python app.py
```

Luego entrar a:

```text
http://127.0.0.1:5003/
```

## Antes de exponer

- Corran `python scripts/entrenar_modelos.py` una vez en la maquina/laptop
  que van a usar para la sustentacion. Esto regenera los `.pkl` con la
  version de scikit-learn instalada ahi y evita el warning de version al
  cargar los modelos.
- Prueben el formulario de `/resultados` al menos una vez antes de exponer
  para que el historial no aparezca vacio en la demo.

