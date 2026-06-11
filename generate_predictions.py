"""
generate_predictions.py — VaVerde backend
==========================================
Ajusta el Dixon-Coles con datos al día y genera predictions.json
con las probabilidades de todos los fixtures futuros del Mundial.

Correr 1 vez al día (cron). El servidor FastAPI solo sirve el archivo.

Uso:
    python generate_predictions.py
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from poisson_model import cargar_datos, ajustar_dixon_coles, matriz_marcadores

DATA_PATH = "results.csv"
OUTPUT_PATH = "predictions.json"


def predecir_fixture(params, home, away, neutral):
    """Predicción completa de un fixture. Devuelve None si falta algún equipo."""
    if home not in params["attack"] or away not in params["attack"]:
        return None
    M, lh, la = matriz_marcadores(params, home, away, neutral=neutral)
    total = np.add.outer(np.arange(M.shape[0]), np.arange(M.shape[1]))
    i, j = np.unravel_index(M.argmax(), M.shape)
    return {
        "p_home": round(float(np.tril(M, -1).sum()), 3),
        "p_draw": round(float(np.trace(M)), 3),
        "p_away": round(float(np.triu(M, 1).sum()), 3),
        "p_over25": round(float(M[total > 2.5].sum()), 3),
        "p_btts": round(float(1 - M[0, :].sum() - M[:, 0].sum() + M[0, 0]), 3),
        "xg_home": round(float(lh), 2),
        "xg_away": round(float(la), 2),
        "likely_score": f"{i}-{j}",
    }


def main():
    print("Cargando datos y ajustando modelo...")
    df_train = cargar_datos(path=DATA_PATH, desde_anio=2018)
    params = ajustar_dixon_coles(df_train)
    print(f"  {len(df_train):,} partidos | último: {df_train['date'].max().date()}")

    # Fixtures futuros del Mundial (sin marcador todavía)
    raw = pd.read_csv(DATA_PATH)
    raw["date"] = pd.to_datetime(raw["date"])
    today = pd.Timestamp.now().normalize()
    fixtures = raw[
        raw["home_score"].isna()
        & (raw["tournament"] == "FIFA World Cup")
        & (raw["date"] >= today)
    ].sort_values("date")

    matches, skipped = [], []
    for _, row in fixtures.iterrows():
        pred = predecir_fixture(params, row["home_team"], row["away_team"],
                                neutral=bool(row["neutral"]))
        if pred is None:
            skipped.append(f'{row["home_team"]} vs {row["away_team"]}')
            continue
        matches.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "home": row["home_team"],
            "away": row["away_team"],
            "city": row["city"],
            "neutral": bool(row["neutral"]),
            **pred,
        })

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "dixon-coles-poisson",
        "training_matches": int(len(df_train)),
        "last_result_date": str(df_train["date"].max().date()),
        "matches": matches,
        # Slot para el simulador Monte Carlo de torneo completo (probabilidades
        # de campeón / avance por ronda). Se llena cuando conectemos tu MC.
        "champion_probs": None,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    print(f"  {len(matches)} predicciones escritas en {OUTPUT_PATH}")
    if skipped:
        print(f"  Sin datos para {len(skipped)} fixtures: {skipped[:5]}")


if __name__ == "__main__":
    main()
