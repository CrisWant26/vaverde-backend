"""
main.py — VaVerde backend API
==============================
Servidor mínimo: sirve predictions.json generado por generate_predictions.py.
Cero base de datos, cero auth. El estado vive en un archivo.

Desarrollo local:
    pip install fastapi uvicorn
    python generate_predictions.py   # genera/actualiza predicciones
    uvicorn main:app --host 0.0.0.0 --port 8000

La app iOS (simulador) consume: http://127.0.0.1:8000/v1/predictions
Desde un iPhone físico en la misma red: http://<IP-de-tu-Mac>:8000/v1/predictions
"""
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

PREDICTIONS_PATH = Path(__file__).parent / "predictions.json"

app = FastAPI(title="VaVerde API", version="0.1.0")

# Cache en memoria: se recarga solo si el archivo cambió (mtime)
_cache = {"mtime": 0.0, "data": None}


def load_predictions():
    if not PREDICTIONS_PATH.exists():
        raise HTTPException(503, "Predicciones no generadas todavía. Corre generate_predictions.py")
    mtime = os.path.getmtime(PREDICTIONS_PATH)
    if mtime != _cache["mtime"]:
        with open(PREDICTIONS_PATH) as f:
            _cache["data"] = json.load(f)
        _cache["mtime"] = mtime
    return _cache["data"]


@app.get("/v1/predictions")
def predictions():
    """Todas las predicciones del Mundial + metadata del modelo."""
    data = load_predictions()
    return JSONResponse(
        content=data,
        headers={"Cache-Control": "public, max-age=3600"},  # 1h de cache cliente/CDN
    )


@app.get("/health")
def health():
    return {"status": "ok"}
