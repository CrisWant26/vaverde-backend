"""
comparar_dc.py — A/B honesto: Poisson+Elo vs Poisson+Elo+Dixon-Coles
=====================================================================
Mide el Brier out-of-sample de AMBOS modelos sobre las mismas particiones
de datos, para las tres competencias. La decisión la toman los números.

Metodología (idéntica a la auditoría que ya usa el pipeline):
  - Split temporal 80/20 (nada de shuffle: el futuro no se entrena).
  - Elo PRE-partido de cada fila (columnas elo_home/elo_away), sin fuga.
  - Brier multiclase sobre 1X2 calibrado, no solo P(local): mide TODO
    el vector de probabilidades, que es lo que la app muestra.
  - Baseline naïve como referencia de "cuánta señal hay disponible".

Uso:
    python comparar_dc.py                 # las tres competencias
    python comparar_dc.py arg             # solo una

Colocar en la RAÍZ del repo (junto a elo.py y modelo_elo.py).
"""
import sys
import os

import numpy as np
import pandas as pd
from scipy.stats import poisson

from elo import calcular_elo
from modelo_elo import ajustar_modelo_elo, ajustar_calibrador, lambdas
from modelo_dc import (ajustar_modelo_dc, ajustar_calibrador_dc,
                       lambdas_dc, matriz_dc)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------
# Competencias a evaluar
# ------------------------------------------------------------
COMPETENCIAS = {
    "selecciones": {
        "tipo": "csv_local",
        "path": "results.csv",
        "desde_anio": 2014,
        "min_partidos": 20,
    },
    "ligamx": {
        "tipo": "football_data",
        "url": "https://www.football-data.co.uk/new/MEX.csv",
        "nombre": "Liga MX",
        "desde_anio": 2012,
        "min_partidos": 30,
    },
    "arg": {
        "tipo": "football_data",
        "url": "https://www.football-data.co.uk/new/ARG.csv",
        "nombre": "Liga Argentina",
        "desde_anio": 2012,
        "min_partidos": 30,
    },
}


def cargar(cfg):
    if cfg["tipo"] == "csv_local":
        raw = pd.read_csv(cfg["path"])
        raw["date"] = pd.to_datetime(raw["date"])
        df = raw.dropna(subset=["home_score", "away_score"]).copy()
        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)
        return df
    raw = pd.read_csv(cfg["url"], encoding="utf-8-sig")
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["Date"], format="%d/%m/%Y", errors="coerce"),
        "home_team": raw["Home"].astype(str).str.strip(),
        "away_team": raw["Away"].astype(str).str.strip(),
        "home_score": pd.to_numeric(raw["HG"], errors="coerce"),
        "away_score": pd.to_numeric(raw["AG"], errors="coerce"),
    })
    df["tournament"] = cfg["nombre"]
    df["neutral"] = False
    df = df.dropna(subset=["date", "home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df.sort_values("date").reset_index(drop=True)


def brier_multiclase(ev_df, params, calib, usar_dc, max_goals=10, n_max=3000):
    """Brier del vector 1X2 completo, con Elo pre-partido (sin fuga)."""
    eps = 1e-6
    logit = lambda p: np.log((p + eps) / (1 - p + eps))
    sub = ev_df.sample(min(n_max, len(ev_df)), random_state=1)
    i = np.arange(max_goals + 1)

    errores = []
    for r in sub.itertuples():
        elos_row = {r.home_team: r.elo_home, r.away_team: r.elo_away}
        neutral = r.neutral in (True, "TRUE", "True")
        if usar_dc:
            lh, la = lambdas_dc(params, elos_row, r.home_team, r.away_team, neutral)
            M = matriz_dc(lh, la, params.get("rho", 0.0), max_goals)
        else:
            lh, la = lambdas(params, elos_row, r.home_team, r.away_team, neutral)
            M = np.outer(poisson.pmf(i, lh), poisson.pmf(i, la))
            M /= M.sum()
        ph = float(np.tril(M, -1).sum())
        pdw = float(np.trace(M))
        pa = float(np.triu(M, 1).sum())
        cal = calib.predict_proba(np.array([[logit(ph), logit(pdw), logit(pa)]]))[0]
        y = np.zeros(3)
        y[0 if r.home_score > r.away_score else
          (1 if r.home_score == r.away_score else 2)] = 1.0
        errores.append(float(((cal - y) ** 2).sum()))
    return float(np.mean(errores))


def brier_naive(ev_df):
    """Baseline: predice siempre las tasas base del set de evaluación."""
    n = len(ev_df)
    ph = (ev_df["home_score"] > ev_df["away_score"]).mean()
    pdw = (ev_df["home_score"] == ev_df["away_score"]).mean()
    pa = 1 - ph - pdw
    base = np.array([ph, pdw, pa])
    err = 0.0
    for r in ev_df.itertuples():
        y = np.zeros(3)
        y[0 if r.home_score > r.away_score else
          (1 if r.home_score == r.away_score else 2)] = 1.0
        err += float(((base - y) ** 2).sum())
    return err / n


def evaluar(key):
    cfg = COMPETENCIAS[key]
    print(f"\n{'='*58}\n  {key.upper()}\n{'='*58}")

    df = cargar(cfg)
    print(f"Partidos: {len(df):,}  ({df['date'].min().date()} → {df['date'].max().date()})")

    df_elo, elos = calcular_elo(df)
    train = df_elo[df_elo["date"].dt.year >= cfg["desde_anio"]]
    counts = pd.concat([train["home_team"], train["away_team"]]).value_counts()
    solidos = set(counts[counts >= cfg["min_partidos"]].index)
    train = train[train["home_team"].isin(solidos) &
                  train["away_team"].isin(solidos)].reset_index(drop=True)

    corte = train["date"].quantile(0.8)
    tr = train[train["date"] <= corte].reset_index(drop=True)
    ev = train[train["date"] > corte].reset_index(drop=True)
    print(f"Train: {len(tr):,}   Eval: {len(ev):,}")

    naive = brier_naive(ev)

    print("Entrenando modelo ACTUAL (Poisson+Elo)...")
    p_act = ajustar_modelo_elo(tr)
    c_act = ajustar_calibrador(tr, p_act, elos)
    b_act = brier_multiclase(ev, p_act, c_act, usar_dc=False)

    print("Entrenando modelo DIXON-COLES...")
    p_dc = ajustar_modelo_dc(tr)
    c_dc = ajustar_calibrador_dc(tr, p_dc, elos)
    b_dc = brier_multiclase(ev, p_dc, c_dc, usar_dc=True)

    delta = b_act - b_dc
    print(f"\n  {'Baseline naïve':<26} {naive:.4f}")
    print(f"  {'Actual (Poisson+Elo)':<26} {b_act:.4f}   (vs naïve: {naive-b_act:+.4f})")
    print(f"  {'Dixon-Coles':<26} {b_dc:.4f}   (vs naïve: {naive-b_dc:+.4f})")
    print(f"  {'rho estimado':<26} {p_dc['rho']:+.4f}")
    print(f"\n  MEJORA DC sobre actual: {delta:+.4f}", end="  ")
    if delta > 0.003:
        print("→ ADOPTAR: mejora clara")
    elif delta > 0.0005:
        print("→ mejora leve, evaluar si compensa la complejidad")
    elif delta > -0.0005:
        print("→ EMPATE: no aporta, quedarse con el actual")
    else:
        print("→ RECHAZAR: empeora")

    return {"competencia": key, "naive": naive, "actual": b_act,
            "dixon_coles": b_dc, "rho": p_dc["rho"], "delta": delta}


if __name__ == "__main__":
    claves = [sys.argv[1]] if len(sys.argv) > 1 else list(COMPETENCIAS)
    resultados = [evaluar(k) for k in claves]

    print(f"\n\n{'='*58}\n  RESUMEN\n{'='*58}")
    print(f"{'Competencia':<14}{'naïve':>9}{'actual':>9}{'DC':>9}{'delta':>10}{'rho':>9}")
    for r in resultados:
        print(f"{r['competencia']:<14}{r['naive']:>9.4f}{r['actual']:>9.4f}"
              f"{r['dixon_coles']:>9.4f}{r['delta']:>+10.4f}{r['rho']:>+9.4f}")
    print("\nRegla: adoptar solo si la mejora es consistente y >0.003.")
