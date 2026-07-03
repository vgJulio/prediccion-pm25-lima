# Prediccion de PM2.5 en Lima con Lasso Regression y Decision Tree Regressor

## 1. Introduccion

El presente informe desarrolla un proyecto de Machine Learning orientado a la
prediccion de la concentracion de material particulado fino PM2.5 en Lima
Metropolitana. El problema es ambiental y de salud publica: el PM2.5 puede
penetrar profundamente en el sistema respiratorio, y sus variaciones horarias
estan relacionadas con dinamicas urbanas como trafico vehicular, actividad
industrial, dispersion atmosferica y diferencias entre estaciones de monitoreo.

La implementacion sigue la misma logica metodologica del notebook de clase de
Regresion Logistica: primero se cargan los datos, luego se inspeccionan, se
limpian, se transforman, se separan las variables independientes y la variable
objetivo, se divide el dataset en entrenamiento y prueba, se entrena el modelo,
se generan predicciones y finalmente se evalua el desempeno. La diferencia
central es que el ejercicio de clase era de clasificacion, mientras que este
proyecto es de regresion porque PM2.5 es una variable numerica continua.

## 2. Objetivos

El objetivo general es construir y evaluar modelos de regresion capaces de
estimar PM2.5 a partir de variables ambientales, temporales y espaciales.

Objetivos especificos:

- Analizar la metodologia del notebook guia de Regresion Logistica.
- Adaptar esa metodologia a un problema de regresion.
- Preparar el dataset de contaminacion atmosferica de Lima.
- Entrenar un modelo Lasso Regression y un Decision Tree Regressor.
- Comparar los modelos usando MAE, MSE, RMSE, R2 y tiempos de ejecucion.
- Interpretar los resultados desde una perspectiva ambiental.
- Guardar modelos, graficos, tablas y codigo reproducible.

## 3. Descripcion del problema

La pregunta ambiental de base es: ¿como se puede estimar la concentracion de
PM2.5 en Lima Metropolitana usando otros contaminantes, la hora, el calendario y
la estacion de monitoreo?

En terminos de Machine Learning, esta pregunta se formula como un problema de
aprendizaje supervisado de regresion. La variable dependiente es PM2.5 y las
variables independientes son PM10, SO2, NO2, O3, CO, ano, mes, dia, hora y
estacion.

## 4. Analisis del notebook de Regresion Logistica

El notebook de clase tenia como objetivo predecir la supervivencia de pasajeros
del Titanic. La variable objetivo era `Survived`, que toma valores discretos:
0 para no sobrevivio y 1 para sobrevivio. Por ello se utilizo Regresion
Logistica, un modelo de clasificacion.

La carga de datos se realizo con Pandas mediante `read_csv`. Despues se reviso
el dataset con `head`, `columns`, mapas de calor de nulos y graficos de
frecuencia. Esta inspeccion permitio entender que columnas existian, cuales
tenian valores faltantes y que relacion preliminar habia entre variables como
sexo, clase del pasajero, edad y supervivencia.

La preparacion del dataset siguio una logica progresiva. Primero se analizaron
los valores nulos; luego se imputo la edad usando una funcion condicionada por
la clase del pasajero; despues se eliminaron columnas con poca utilidad o muchos
nulos, como `Cabin`, y columnas identificadoras o textuales que no se usarian
directamente, como nombre, ticket e identificador del pasajero. Las variables
categoricas, como sexo y puerto de embarque, se transformaron con variables
dummy para que el modelo pudiera procesarlas numericamente.

La division de datos se hizo separando `X` e `y`: `X` contenia las variables
explicativas y `y` la variable objetivo. Luego se uso `train_test_split` con
30% para prueba y `random_state=42`, lo que permite reproducir la particion.
El modelo se entreno con `fit`, se predijo con `predict` y se evaluo con
accuracy, matriz de confusion y reporte de clasificacion.

Las buenas practicas principales fueron: inspeccionar antes de modelar, tratar
nulos, eliminar variables no pertinentes, codificar categorias, separar
entrenamiento y prueba, fijar semilla aleatoria y evaluar con metricas
coherentes con el tipo de problema. En este proyecto se conserva esa estructura,
pero se reemplazan las metricas de clasificacion por metricas de regresion.

## 5. Dataset

El archivo original contiene 703056 filas y
12 columnas. Luego de eliminar duplicados, imputar
predictores y conservar filas con PM2.5 observado, el dataset de modelado queda
con 339970 filas y 12
columnas. El rango temporal del dataset limpio va de 2014 a
2020, con 10 estaciones de monitoreo.

La variable dependiente es `PM 2.5`. Las variables independientes son `PM 10`,
`SO2`, `NO2`, `O3`, `CO`, `ANO`, `MES`, `DIA`, `HORA` y `ESTACION`.

### Valores faltantes

| Variable | Nulos antes | Nulos despues |
| --- | --- | --- |
| CODIGO ESTACION | 0 | 0 |
| ESTACION | 0 | 0 |
| ANO | 0 | 0 |
| MES | 0 | 0 |
| DIA | 0 | 0 |
| HORA | 0 | 0 |
| PM 10 | 206372 | 0 |
| PM 2.5 | 363084 | 0 |
| SO2 | 408776 | 0 |
| NO2 | 351802 | 0 |
| O3 | 354354 | 0 |
| CO | 381705 | 0 |

Los contaminantes se convierten a numericos porque algunos valores pueden estar
almacenados como texto o con coma decimal. Los predictores contaminantes se
imputan mediante interpolacion temporal por estacion y medianas por
estacion-hora. PM2.5 se conserva como objetivo y se eliminan las filas sin
valor observado de PM2.5 para evitar evaluar los modelos contra una respuesta
inventada por imputacion.

## 6. Variables

Variables independientes:

- `PM 10`: material particulado grueso. Es fisicamente cercano a PM2.5, por lo
  que se espera que tenga alta relacion predictiva.
- `SO2`: asociado a combustion y fuentes industriales.
- `NO2`: relacionado con trafico vehicular y combustion.
- `O3`: contaminante secundario influido por radiacion solar y reacciones
  atmosfericas.
- `CO`: indicador de combustion incompleta, comun en trafico urbano.
- `ANO`, `MES`, `DIA`, `HORA`: variables temporales que capturan patrones por
  ano, estacionalidad, dia y ciclo diario.
- `ESTACION`: variable categorica que representa ubicacion de monitoreo y
  diferencias espaciales dentro de Lima.

Variable dependiente:

- `PM 2.5`: concentracion esperada de material particulado fino.

## 7. Metodologia

La metodologia replica el orden del notebook guia:

1. Importacion de librerias.
2. Lectura del dataset.
3. Inspeccion inicial.
4. Limpieza de datos.
5. Transformacion de variables.
6. Separacion de `X` e `y`.
7. Division train/test con 70% entrenamiento y 30% prueba.
8. Entrenamiento de modelos.
9. Prediccion.
10. Evaluacion con metricas de regresion.
11. Visualizacion e interpretacion.
12. Guardado de modelos con Joblib.

## 8. Implementacion del modelo Lasso

Lasso responde la pregunta: ¿como afectan conjuntamente PM10, NO2, SO2, CO,
O3, hora, calendario y estacion a la concentracion esperada de PM2.5?

La pregunta es adecuada para Lasso porque este modelo estima una combinacion
lineal de variables. Cada coeficiente indica la direccion y fuerza de la
relacion, mientras que la penalizacion L1 puede reducir algunos coeficientes a
cero o cerca de cero, simplificando la interpretacion.

La formulacion matematica de Lasso es:

$$
\min_{\beta_0,\beta} \frac{1}{n} \sum_{i=1}^n
(y_i - \beta_0 - x_i^T\beta)^2 + \alpha \sum_{j=1}^p |\beta_j|
$$

Donde:

- $y_i$ es el valor real de PM2.5.
- $x_i$ es el vector de variables predictoras.
- $\beta_0$ es el intercepto.
- $\beta_j$ son los coeficientes.
- El primer termino mide error cuadratico.
- El segundo termino es la penalizacion L1.
- $\alpha$ controla la intensidad de la regularizacion.

En la implementacion, las variables numericas se imputan y escalan con
`StandardScaler`, porque Lasso es sensible a la escala. La estacion se codifica
con One-Hot Encoding porque es categorica.

Principales coeficientes encontrados:

| Variable | Coeficiente | Valor absoluto |
| --- | --- | --- |
| PM 10 | 9.3766 | 9.3766 |
| ESTACION_VILLA MARIA DEL TRIUNFO | -7.5920 | 7.5920 |
| ESTACION_SAN BORJA  | -3.8678 | 3.8678 |
| ESTACION_SANTA ANITA | 3.3394 | 3.3394 |
| HORA | -2.9073 | 2.9073 |
| ESTACION_ATE | 2.0922 | 2.0922 |
| CO | 1.9660 | 1.9660 |
| NO2 | 1.8741 | 1.8741 |
| ESTACION_SAN JUAN DE LURIGANCHO | 1.8685 | 1.8685 |
| SO2 | 1.3822 | 1.3822 |

Interpretacion: `PM 10` obtuvo el coeficiente positivo mas alto. Fisicamente,
esto indica que cuando aumenta PM10 tambien tiende a aumentar PM2.5, lo cual es
coherente porque ambos son fracciones de material particulado y pueden compartir
fuentes como polvo resuspendido, combustion y trafico. La variable `HORA` tuvo
coeficiente negativo en escala estandarizada, lo que sugiere que, manteniendo
otras variables constantes, ciertas horas tempranas o patrones del ciclo diario
estan asociadas a niveles mas altos que horas posteriores.

## 9. Implementacion del Decision Tree Regressor

El Decision Tree Regressor responde la pregunta: ¿que reglas y patrones
presentes en las variables ambientales permiten estimar la concentracion de
PM2.5?

Esta pregunta es adecuada para un arbol porque el modelo no construye una sola
ecuacion lineal, sino reglas del tipo "si PM10 es menor o mayor que cierto
umbral, entonces seguir por una rama". Por eso puede capturar relaciones no
lineales e interacciones entre variables.

La formulacion CART para regresion busca divisiones que reduzcan el error
dentro de los nodos. Para un nodo $R$, la prediccion de una hoja es:

$$
\hat{y}_R = \frac{1}{|R|} \sum_{i \in R} y_i
$$

El error de una region se mide con RSS:

$$
RSS(R) = \sum_{i \in R} (y_i - \hat{y}_R)^2
$$

Para dividir un nodo, CART prueba variables y umbrales, y elige la particion
que minimiza:

$$
RSS(R_1) + RSS(R_2)
$$

El MSE es el RSS dividido entre el numero de observaciones. Cada hoja almacena
el promedio de PM2.5 de los registros que llegan a ella, y ese promedio es la
prediccion final.

La primera division del arbol fue por `PM 10` con
umbral aproximado 86.1450. Esto significa que el
modelo encontro en esa variable la separacion inicial que mas reduce el error
cuadratico. En este caso, PM10 domina la estructura inicial del arbol, lo cual
refuerza la relacion fisica entre particulas gruesas y finas.

Importancia de variables:

| Variable | Importancia |
| --- | --- |
| PM 10 | 0.6681 |
| HORA | 0.1217 |
| SO2 | 0.0601 |
| MES | 0.0420 |
| ESTACION_ATE | 0.0170 |
| ANO | 0.0162 |
| NO2 | 0.0159 |
| CO | 0.0124 |
| ESTACION_VILLA MARIA DEL TRIUNFO | 0.0113 |
| DIA | 0.0083 |

## 10. Explicacion detallada del codigo

El codigo se encuentra en `proyecto_pm25_modelos.py` y esta organizado en
funciones. La logica es la siguiente:

- Las importaciones cargan Pandas y NumPy para datos, Matplotlib para graficos,
  Scikit-Learn para modelado y Joblib para guardar modelos.
- Las constantes definen rutas, nombre de la variable objetivo, columnas
  predictoras, semilla aleatoria y porcentaje de prueba.
- `load_dataset` lee el Excel original.
- `convert_pollutants_to_numeric` transforma contaminantes a numeros y resuelve
  comas decimales.
- `remove_duplicate_records` elimina filas repetidas para la misma estacion y
  hora.
- `impute_predictor_pollutants` completa valores faltantes de predictores con
  interpolacion temporal, mediana por estacion-hora, mediana por estacion y
  mediana global.
- `prepare_dataset` aplica toda la limpieza y elimina registros sin PM2.5.
- `split_features_target` separa `X` e `y`, igual que en el notebook guia.
- `build_lasso_pipeline` crea el preprocesamiento de Lasso: imputacion,
  escalamiento y codificacion categorica.
- `build_tree_pipeline` crea el preprocesamiento del arbol: imputacion y
  codificacion categorica.
- `evaluate_model` ejecuta `fit`, `predict` y calcula MAE, MSE, RMSE y R2.
- Las funciones de graficos guardan visualizaciones de exploracion, prediccion,
  residuos, coeficientes e importancias.
- `run_experiment` coordina todo el flujo y guarda tablas, graficos y modelos.

Esta estructura resuelve un problema practico: evita repetir codigo, hace que
el experimento sea reproducible y permite comparar modelos bajo la misma
particion de datos.

## 11. Resultados obtenidos

| Modelo | MAE | MSE | RMSE | R2 | Tiempo entrenamiento (s) | Tiempo prediccion (s) |
| --- | --- | --- | --- | --- | --- | --- |
| Lasso Regression | 8.8321 | 169.3128 | 13.0120 | 0.4492 | 1.1070 | 0.0709 |
| Decision Tree Regressor | 7.8480 | 140.1924 | 11.8403 | 0.5439 | 1.9277 | 0.0849 |

El conjunto de entrenamiento tuvo 237979 filas y el de
prueba 101991 filas. La media de PM2.5 en el dataset limpio
fue 26.3540 y la mediana fue 22.4000.

Graficos generados:

![Distribucion de PM2.5](graficos/distribucion_pm25.png)

![PM2.5 promedio por hora](graficos/pm25_promedio_hora.png)

![Comparacion de metricas](graficos/comparacion_metricas.png)

![Coeficientes Lasso](graficos/coeficientes_lasso.png)

![Importancia de variables del arbol](graficos/importancia_variables_arbol.png)

## 12. Comparacion

| Criterio | Lasso | Decision Tree | Interpretacion |
| --- | --- | --- | --- |
| MAE | 8.8321 | 7.8480 | menor es mejor |
| MSE | 169.3128 | 140.1924 | penaliza errores grandes |
| RMSE | 13.0120 | 11.8403 | error promedio en unidades de PM2.5 |
| R2 | 0.4492 | 0.5439 | mayor es mejor |
| Tiempo entrenamiento (s) | 1.1070 | 1.9277 | costo de ajuste |
| Tiempo prediccion (s) | 0.0709 | 0.0849 | costo de inferencia |
| Interpretabilidad | alta por coeficientes | alta por reglas | ambos son explicables |
| No linealidad | limitada | alta | el arbol captura umbrales |
| Sensibilidad al ruido | media, regularizada | media-alta | se controla con profundidad y hojas minimas |

El Decision Tree Regressor obtuvo mejores metricas: menor MAE, menor MSE,
menor RMSE y mayor R2. Esto indica que capto mejor la estructura del problema.
La razon principal es que la contaminacion atmosferica rara vez se comporta de
forma estrictamente lineal. Existen umbrales, interacciones por hora, diferencias
entre estaciones y variaciones temporales que un arbol puede representar mejor.

Lasso sigue siendo valioso porque ofrece una interpretacion directa mediante
coeficientes y reduce complejidad con regularizacion. Sin embargo, su forma
lineal limita su capacidad para capturar patrones como "PM10 alto en ciertas
horas y estaciones produce incrementos distintos de PM2.5".

## 13. Interpretacion ambiental

El resultado mas consistente es el peso de PM10. En Lasso aparece como el
coeficiente dominante y en el arbol concentra la mayor importancia. Esto es
coherente con el fenomeno ambiental: PM10 y PM2.5 son contaminantes particulados
y pueden estar asociados a fuentes comunes, como trafico, polvo resuspendido,
combustion y actividades urbanas.

La importancia de `HORA` en el arbol sugiere que el ciclo diario es relevante.
Esto puede relacionarse con horas de mayor movilidad vehicular, estabilidad
atmosferica en ciertos momentos del dia y cambios en dispersion. Si el modelo
divide por hora o asigna alta importancia a esa variable, esta reconociendo que
la contaminacion no depende solo de la cantidad de contaminantes presentes, sino
tambien del momento en que se mide.

Las variables de estacion indican diferencias espaciales dentro de Lima. Por
ejemplo, estaciones como ATE, SANTA ANITA o VILLA MARIA DEL TRIUNFO aparecen en
coeficientes o importancias relevantes, lo que sugiere que la ubicacion del
sensor resume condiciones urbanas locales: trafico, densidad, industria,
topografia y ventilacion.

## 14. Conclusiones

El proyecto demuestra que la metodologia de un notebook de clasificacion puede
adaptarse correctamente a regresion si se cambian el modelo, la variable
objetivo y las metricas de evaluacion. En lugar de accuracy y matriz de
confusion, se emplean MAE, MSE, RMSE y R2.

Decision Tree Regressor fue el modelo con mejor desempeno en esta ejecucion,
con RMSE de 11.8403 y R2 de 0.5439. Lasso obtuvo RMSE de
13.0120 y R2 de 0.4492. Por tanto, el arbol representa
mejor las relaciones no lineales del problema, mientras que Lasso aporta una
explicacion lineal clara.

Ambos modelos son adecuados para una primera fase academica de prediccion de
PM2.5 porque son interpretables, estan disponibles en Scikit-Learn, permiten un
flujo reproducible y ofrecen mecanismos claros de explicacion: coeficientes en
Lasso y reglas/importancias en el arbol.

## 15. Trabajo futuro

Como trabajo futuro se recomienda:

- Validar el modelo con particiones temporales, no solo aleatorias.
- Analizar variables meteorologicas como viento, humedad y temperatura.
- Ajustar hiperparametros con validacion cruzada.
- Evaluar errores por estacion y por franja horaria.
- Crear un dashboard que muestre predicciones y patrones ambientales.
- Incorporar monitoreo de drift si el modelo se usa con datos nuevos.

## Entregables generados

- Codigo completo: `proyecto_pm25_modelos.py`.
- Modelos guardados: `modelos/modelo_lasso_pm25.pkl` y
  `modelos/modelo_arbol_pm25.pkl`.
- Tablas comparativas: carpeta `tablas`.
- Graficos: carpeta `graficos`.
- Reglas exportadas del arbol: `tablas/reglas_arbol_pm25.txt`.
- Notebook pedagogico: `proyecto_pm25_lasso_arbol.ipynb`.
