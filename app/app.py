from __future__ import annotations

from pathlib import Path

from flask import Flask, render_template, request, send_from_directory

if __package__:
    from .predictor import DEFAULT_VALUES, load_metrics, predict
else:
    import sys

    APP_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(APP_DIR))
    from predictor import DEFAULT_VALUES, load_metrics, predict


app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
PROJECT_DIR = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_DIR / "reports" / "imagenes"
APP_VERSION = "dashboard-v2-decision-tree"


@app.context_processor
def inject_version():
    return {"app_version": APP_VERSION}


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/")
def index():
    metrics_data = load_metrics()
    return render_template("index.html", metrics_data=metrics_data)


@app.route("/dashboard")
def dashboard():
    metrics_data = load_metrics()
    return render_template("dashboard.html", metrics_data=metrics_data)


@app.route("/comparacion")
def comparacion():
    metrics_data = load_metrics()
    return render_template("comparacion.html", metrics_data=metrics_data)


@app.route("/resultados", methods=["GET", "POST"])
def resultados():
    metrics_data = load_metrics()
    form_values = DEFAULT_VALUES.copy()
    prediction = None
    selected_model = "tree"
    error = None

    if request.method == "POST":
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
    return send_from_directory(IMAGES_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5003)
