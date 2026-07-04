from __future__ import annotations

from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_from_directory

# Permite ejecutar el archivo de dos formas:
# 1. python app.py desde la raiz del proyecto.
# 2. python app/app.py desde la carpeta app.
if __package__:
    from . import analytics
    from .predictor import DEFAULT_VALUES, append_history, load_history, load_metrics, predict_both
else:
    import sys

    APP_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(APP_DIR))
    import analytics
    from predictor import DEFAULT_VALUES, append_history, load_history, load_metrics, predict_both


app = Flask(__name__)

# Recarga templates y evita cache para ver cambios rapido durante la exposicion.
app.config["TEMPLATES_AUTO_RELOAD"] = True
PROJECT_DIR = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_DIR / "reports" / "imagenes"
APP_VERSION = "dashboard-v2-decision-tree"


@app.context_processor
def inject_version():
    # Agrega una version a CSS/JS/imagenes para que el navegador no muestre cache viejo.
    return {"app_version": APP_VERSION}


@app.after_request
def add_no_cache_headers(response):
    # Fuerza al navegador a pedir la version actual de cada pagina.
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/")
def index():
    # Pagina inicial con resumen general del proyecto.
    metrics_data = load_metrics()
    return render_template("index.html", metrics_data=metrics_data)


@app.route("/dashboard")
def dashboard():
    # Dashboard principal: graficos, resumen y decision del arbol.
    metrics_data = load_metrics()
    return render_template("dashboard.html", metrics_data=metrics_data)


@app.route("/exploracion")
def exploracion():
    # Analisis exploratorio en vivo: pandas/numpy calculan estadisticas al momento del
    # request, filtrando por la estacion elegida en el select.
    metrics_data = load_metrics()
    estacion = request.args.get("estacion") or None
    eda = analytics.build_eda_summary(estacion)
    return render_template(
        "exploracion.html",
        metrics_data=metrics_data,
        eda=eda,
        estacion_seleccionada=estacion,
        stations=metrics_data.get("stations", ["ATE"]),
    )


@app.route("/grafico/histograma.png")
def grafico_histograma():
    # Genera el histograma de PM2.5 en memoria con matplotlib, sin guardar archivo en disco.
    estacion = request.args.get("estacion") or None
    png_bytes = analytics.build_histogram_png(estacion)
    return Response(png_bytes, mimetype="image/png")


@app.route("/comparacion")
def comparacion():
    # Tabla y graficos para comparar Lasso contra Decision Tree.
    metrics_data = load_metrics()
    live_metrics = None
    if request.args.get("recalcular"):
        # Vuelve a calcular MAE/RMSE/R2 con sklearn sobre data/test/datos_prueba.csv,
        # en vez de solo leer los valores guardados en metricas_modelos.json.
        live_metrics = analytics.recalcular_metricas_en_vivo()
    return render_template("comparacion.html", metrics_data=metrics_data, live_metrics=live_metrics)


@app.route("/modelo-matematico")
def modelo_matematico():
    # Ecuacion de Lasso y explicacion matematica del Decision Tree.
    metrics_data = load_metrics()
    return render_template("modelo_matematico.html", metrics_data=metrics_data)


@app.route("/resultados", methods=["GET", "POST"])
def resultados():
    # Formulario de prediccion. GET muestra la pagina; POST calcula PM 2.5 con ambos modelos.
    metrics_data = load_metrics()
    form_values = DEFAULT_VALUES.copy()
    predictions = None
    error = None

    if request.method == "POST":
        # Toma los valores escritos por el usuario y llama al predictor.
        form_values.update(request.form.to_dict())
        try:
            predictions = predict_both(form_values)
            # Registra la prueba en reports/historial_predicciones.csv.
            append_history(form_values, predictions)
        except (ValueError, FileNotFoundError) as exc:
            error = str(exc)

    return render_template(
        "resultados.html",
        error=error,
        form_values=form_values,
        metrics_data=metrics_data,
        predictions=predictions,
        stations=metrics_data.get("stations", ["ATE"]),
        historial=load_history(),
    )


@app.route("/api/predecir", methods=["POST"])
def api_predecir():
    # Version JSON de /resultados: recibe las mismas variables y devuelve ambas predicciones
    # sin renderizar HTML. Util para consumir el modelo como servicio.
    form_values = DEFAULT_VALUES.copy()
    form_values.update(request.get_json(silent=True) or {})
    try:
        predictions = predict_both(form_values)
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"entrada": form_values, "predicciones": predictions})


@app.route("/residuales")
def residuales():
    # Analisis de residuales por estacion sobre el set de prueba: cuanto se equivoca cada
    # modelo, agrupado con pandas (analytics.residuales_por_estacion).
    metrics_data = load_metrics()
    datos = analytics.residuales_por_estacion()
    return render_template("residuales.html", metrics_data=metrics_data, datos=datos)


@app.route("/grafico/residuales.png")
def grafico_residuales():
    # Grafico de barras del MAE por estacion, generado en memoria con matplotlib.
    png_bytes = analytics.build_residuales_png()
    return Response(png_bytes, mimetype="image/png")


@app.route("/mapa")
def mapa():
    # Mapa de estaciones con el promedio historico de PM2.5, usando Leaflet en el navegador.
    metrics_data = load_metrics()
    estaciones = analytics.build_station_map_data()
    return render_template("mapa.html", metrics_data=metrics_data, estaciones=estaciones)


@app.route("/imagenes/<path:filename>")
def imagenes(filename: str):
    # Sirve las imagenes generadas por los scripts para usarlas en el HTML.
    return send_from_directory(IMAGES_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5003)
