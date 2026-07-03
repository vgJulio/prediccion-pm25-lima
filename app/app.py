from __future__ import annotations

from pathlib import Path

from flask import Flask, render_template, request, send_from_directory

# Permite ejecutar el archivo de dos formas:
# 1. python app.py desde la raiz del proyecto.
# 2. python app/app.py desde la carpeta app.
if __package__:
    from .predictor import DEFAULT_VALUES, load_metrics, predict
else:
    import sys

    APP_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(APP_DIR))
    from predictor import DEFAULT_VALUES, load_metrics, predict


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


@app.route("/comparacion")
def comparacion():
    # Tabla y graficos para comparar Lasso contra Decision Tree.
    metrics_data = load_metrics()
    return render_template("comparacion.html", metrics_data=metrics_data)


@app.route("/resultados", methods=["GET", "POST"])
def resultados():
    # Formulario de prediccion. GET muestra la pagina; POST calcula PM 2.5.
    metrics_data = load_metrics()
    form_values = DEFAULT_VALUES.copy()
    prediction = None
    selected_model = "tree"
    error = None

    if request.method == "POST":
        # Toma los valores escritos por el usuario y llama al predictor.
        form_values.update(request.form.to_dict())
        selected_model = request.form.get("modelo", selected_model)
        try:
            prediction = predict(selected_model, form_values)
        except (ValueError, FileNotFoundError) as exc:
            error = str(exc)

    return render_template(
        "resultados.html",
        error=error,
        form_values=form_values,
        metrics_data=metrics_data,
        prediction=prediction,
        selected_model=selected_model,
        stations=metrics_data.get("stations", ["ATE"]),
    )


@app.route("/imagenes/<path:filename>")
def imagenes(filename: str):
    # Sirve las imagenes generadas por los scripts para usarlas en el HTML.
    return send_from_directory(IMAGES_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5003)
