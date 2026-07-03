from __future__ import annotations

import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_DIR / "reports" / "metricas_modelos.json"


def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError("Primero ejecute: python scripts/entrenar_modelos.py")

    data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    for row in data["metrics"]:
        print(
            f"{row['modelo']}: "
            f"MAE={row['mae']:.4f}, RMSE={row['rmse']:.4f}, R2={row['r2']:.4f}"
        )


if __name__ == "__main__":
    main()
