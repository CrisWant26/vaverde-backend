"""
generate_predictions.py — VaVerde backend (v2: pipeline unificado Poisson+Elo)
===============================================================================
Re-entrena TODO el modelo cada corrida con el dataset al día:

  1. Elo de las 200+ selecciones sobre el histórico completo (elo.py).
  2. Regresión Poisson+Elo por máxima verosimilitud (modelo_elo.py).
  3. Calibración Platt de las probabilidades 1X2 (modelo_elo.py).
  4. Predicciones de los fixtures del Mundial (mismo modelo).
  5. Monte Carlo de 10,000 torneos para probabilidades de campeón
     (simulacion.py, que consume el pkl recién entrenado).

Un solo modelo para todo: lo que dice la fila de un partido y lo que dice
la tabla de campeón salen de los mismos parámetros.

Requiere junto a este script: elo.py, modelo_elo.py, simulacion.py, results.csv
"""
import json
import pickle
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from scipy.stats import poisson

from elo import calcular_elo
from modelo_elo import ajustar_modelo_elo, ajustar_calibrador, lambdas
from historial import actualizar_historial, cargar_historial

DATA_PATH = "results.csv"
OUTPUT_PATH = "predictions.json"
MODEL_PATH = "modelo_elo.pkl"

DESDE_ANIO = 2014        # ventana de entrenamiento de la regresión
MIN_PARTIDOS = 20        # filtro de equipos con muestra sólida (para la regresión)
N_SIMULACIONES = 10_000


def entrenar():
    """Elo + regresión + calibrador, todo fresco. Guarda el pkl y devuelve las piezas."""
    raw = pd.read_csv(DATA_PATH)
    raw["date"] = pd.to_datetime(raw["date"])

    played = raw.dropna(subset=["home_score", "away_score"]).copy()
    played["home_score"] = played["home_score"].astype(int)
    played["away_score"] = played["away_score"].astype(int)

    print(f"Calculando Elo sobre {len(played):,} partidos históricos...")
    df_elo, elos = calcular_elo(played)

    train = df_elo[df_elo["date"].dt.year >= DESDE_ANIO]
    counts = pd.concat([train["home_team"], train["away_team"]]).value_counts()
    solidos = set(counts[counts >= MIN_PARTIDOS].index)
    train = train[
        train["home_team"].isin(solidos) & train["away_team"].isin(solidos)
    ].reset_index(drop=True)

    # ---- Auditoría de calibración (split temporal honesto) ----
    # Entrena un modelo SOLO con el 80% más antiguo y mide Brier en el 20%
    # restante, usando el Elo PRE-partido de cada fila (sin fuga del futuro).
    corte = train["date"].quantile(0.8)
    tr_eval = train[train["date"] <= corte].reset_index(drop=True)
    ev_eval = train[train["date"] > corte].reset_index(drop=True)
    print(f"Auditoría de calibración (train {len(tr_eval):,} / eval {len(ev_eval):,})...")
    params_eval = ajustar_modelo_elo(tr_eval)
    calib_eval = ajustar_calibrador(tr_eval, params_eval, elos)
    brier = brier_pre_partido(ev_eval, params_eval, calib_eval)
    print(f"  Brier score (P(local) calibrada, out-of-sample): {brier:.4f}")

    # ---- Modelo de producción: entrenado con TODO ----
    print(f"Ajustando regresión Poisson+Elo ({len(train):,} partidos desde {DESDE_ANIO})...")
    params = ajustar_modelo_elo(train)
    print(f"  b_elo={params['b_elo']:.3f}  home_adv={params['home_adv']:.3f}")

    print("Calibrando (Platt)...")
    calibrador = ajustar_calibrador(train, params, elos)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"params": params, "calibrador": calibrador, "elos": elos}, f)

    last_date = played["date"].max().date()
    return raw, params, calibrador, elos, len(train), last_date, brier


def brier_pre_partido(ev_df, params, calibrador, max_goals=10, n_max=3000):
    """Brier de P(local) calibrada sobre el set de evaluación.

    Rigor: usa el Elo PRE-partido de cada fila (columnas elo_home/elo_away de
    calcular_elo), no los ratings actuales — así la evaluación no ve el futuro.
    """
    eps = 1e-6
    logit = lambda p: np.log((p + eps) / (1 - p + eps))
    sub = ev_df.sample(min(n_max, len(ev_df)), random_state=1)

    errores = []
    i = np.arange(max_goals + 1)
    for r in sub.itertuples():
        elos_row = {r.home_team: r.elo_home, r.away_team: r.elo_away}
        neutral = r.neutral in (True, "TRUE", "True")
        lh, la = lambdas(params, elos_row, r.home_team, r.away_team, neutral=neutral)
        M = np.outer(poisson.pmf(i, lh), poisson.pmf(i, la))
        M /= M.sum()
        ph = float(np.tril(M, -1).sum())
        pd_ = float(np.trace(M))
        pa = float(np.triu(M, 1).sum())
        cal = calibrador.predict_proba(
            np.array([[logit(ph), logit(pd_), logit(pa)]])
        )[0]
        y = 1.0 if r.home_score > r.away_score else 0.0
        errores.append((cal[0] - y) ** 2)

    return float(np.mean(errores))


def predecir_fixture(params, calibrador, elos, home, away, neutral, max_goals=10):
    """Predicción completa de un fixture: 1X2 calibrado + mercados desde la matriz."""
    lh, la = lambdas(params, elos, home, away, neutral=neutral)
    i = np.arange(max_goals + 1)
    M = np.outer(poisson.pmf(i, lh), poisson.pmf(i, la))
    M /= M.sum()

    p_home = float(np.tril(M, -1).sum())
    p_draw = float(np.trace(M))
    p_away = float(np.triu(M, 1).sum())

    # Calibración Platt sobre los logits del 1X2 (igual que modelo_elo.predecir)
    eps = 1e-6
    logit = lambda p: np.log((p + eps) / (1 - p + eps))
    cal = calibrador.predict_proba(
        np.array([[logit(p_home), logit(p_draw), logit(p_away)]])
    )[0]

    total = np.add.outer(i, i)
    gi, gj = np.unravel_index(M.argmax(), M.shape)

    return {
        "p_home": round(float(cal[0]), 3),
        "p_draw": round(float(cal[1]), 3),
        "p_away": round(float(cal[2]), 3),
        "p_over25": round(float(M[total > 2.5].sum()), 3),
        "p_btts": round(float(1 - M[0, :].sum() - M[:, 0].sum() + M[0, 0]), 3),
        "xg_home": round(float(lh), 2),
        "xg_away": round(float(la), 2),
        "likely_score": f"{gi}-{gj}",
    }


def champion_probs(n_sims=N_SIMULACIONES):
    """Monte Carlo del torneo con el pkl recién entrenado (degradación elegante)."""
    try:
        from simulacion import simular
        seed = int(datetime.now(timezone.utc).strftime("%Y%m%d"))
        df = simular(N=n_sims, seed=seed)
        print(f"  Monte Carlo: {n_sims:,} torneos simulados")
        return [
            {
                "team": r.equipo,
                "elo": int(r.elo),
                "p_advance": round(r._3 / 100, 3),
                "p_final": round(r._4 / 100, 3),
                "p_champion": round(r._5 / 100, 3),
            }
            for r in df.itertuples()
        ]
    except Exception as e:
        print(f"  Monte Carlo omitido: {e}")
        return None


def main():
    raw, params, calibrador, elos, n_train, last_date, brier = entrenar()

    # "Hoy" en hora de México (CDMX = UTC-6), no UTC, para que coincida con
    # el teléfono del usuario. Incluimos los partidos de HOY aunque ya se
    # hayan jugado: la app los mantiene visibles toda la jornada y los oculta
    # a la medianoche local.
    now_cdmx = datetime.now(timezone.utc) - timedelta(hours=6)
    today = pd.Timestamp(now_cdmx.date())
    fixtures = raw[
        (raw["tournament"] == "FIFA World Cup")
        & (raw["date"] >= today)
    ].sort_values("date")

    matches = []
    for _, row in fixtures.iterrows():
        pred = predecir_fixture(
            params, calibrador, elos,
            row["home_team"], row["away_team"], neutral=bool(row["neutral"]),
        )
        matches.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "home": row["home_team"],
            "away": row["away_team"],
            "city": row["city"],
            "neutral": bool(row["neutral"]),
            **pred,
        })

    # Actualiza el historial usando las predicciones actuales y el dataset
    actualizar_historial(matches, raw)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "poisson-elo-calibrado",
        "training_matches": int(n_train),
        "last_result_date": str(last_date),
        "model_brier": round(brier, 4),
        "matches": matches,
        "champion_probs": champion_probs(),
        "history": cargar_historial(),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    print(f"  {len(matches)} predicciones escritas en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
