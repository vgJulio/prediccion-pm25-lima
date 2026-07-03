# Prediccion PM2.5 Lima

Proyecto simple para limpiar datos horarios de contaminacion en Lima,
entrenar dos modelos de regresion y mostrar resultados en un dashboard Flask.

## Objetivo

Predecir la concentracion de PM2.5 usando contaminantes atmosfericos,
fecha, hora y estacion de monitoreo.

## Modelos

- Lasso Regression
- Decision Tree Regressor

## Estructura

```text
Prediccion_PM25_Lima/
├── app/                  # Dashboard Flask
├── data/
│   ├── raw/              # Dataset original
│   ├── processed/        # Datos limpios/modelo
│   └── test/             # Datos de prueba
├── docs/                 # Documento del proyecto
├── models/               # Modelos .pkl
├── notebooks/            # Notebooks del proyecto
├── reports/              # Metricas e imagenes
├── scripts/              # Limpieza, entrenamiento y evaluacion
├── app.py                # Entrada simple
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
