"""
generate_league.py — VaVerde ligas de clubes (Fase 2)
======================================================
Pipeline por liga, REUSANDO el motor existente sin tocarlo:
  - elo.py            → calcular_elo (Elo propio de la liga, todos parten de 1500)
  - modelo_elo.py     → ajustar_modelo_elo + ajustar_calibrador (Platt por liga)
  - generate_predictions.py → predecir_fixture + brier_pre_partido
  - historial.py      → historial por liga (archivos propios por liga)

Diferencias clave vs selecciones:
  - Fuente: football-data.co.uk (descarga automática, actualizan ~2x/semana).
  - neutral=False SIEMPRE (en clubes no hay cancha neutral).
  - Fixtures: CSV manual por jornada (football-data no publica calendario
    de las ligas "extra"); los RESULTADOS sí llegan solos.
  - Sin Monte Carlo de campeón (la liguilla es otro proyecto): champion_probs=None.
  - Emparejamiento de resultados con tolerancia de ±1 día (football-data
    registra fechas en hora UK: un partido de viernes 21:00 CDMX cae en
    sábado UTC).

Uso:  python leagues/generate_league.py ligamx
"""
import sys
import os
import json
from datetime import datetime, timezone, timedelta

import pandas as pd

# El script vive en leagues/ pero los módulos del motor viven en la raíz.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from elo import calcular_elo
from modelo_elo import ajustar_modelo_elo, ajustar_calibrador
from generate_predictions import predecir_fixture, brier_pre_partido
import historial

# ============================================================
# Configuración por liga. Agregar Premier/LaLiga = nueva entrada
# (ojo: las ligas "main" de football-data usan archivos POR TEMPORADA
   #  con otro esquema — eso se resuelve cuando las agreguemos, no hoy).
# ============================================================
LEAGUES = {
    "arg": {
        "name": "Liga Argentina",
        "csv_url": "https://www.football-data.co.uk/new/ARG.csv",
        "fixtures_csv": "leagues/fixtures_arg.csv",
        "output_json": "docs/leagues/arg.json",
        "history_file": "leagues/history_arg.json",
        "pending_file": "leagues/pending_arg.json",
        "desde_anio": 2012,
        "min_partidos": 30,
    },
    "ligamx": {
            "name": "Liga MX",
            "csv_url": "https://www.football-data.co.uk/new/MEX.csv",
            "fixtures_csv": "leagues/fixtures_ligamx.csv",
            "output_json": "docs/leagues/ligamx.json",
            "history_file": "leagues/history_ligamx.json",
            "pending_file": "leagues/pending_ligamx.json",
            "desde_anio": 2012,     # ventana de entrenamiento de la regresión
            "min_partidos": 30,     # filtro de equipos con muestra sólida
        },
    }
    
    
    # ------------------------------------------------------------
    # Carga y normalización al esquema canónico del motor
    # ------------------------------------------------------------
    def cargar_datos(cfg):
    """Descarga el CSV de football-data y lo normaliza a las columnas
    que esperan elo.py / modelo_elo.py."""
    raw = pd.read_csv(cfg["csv_url"], encoding="utf-8-sig")
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["Date"], format="%d/%m/%Y", errors="coerce"),
        "home_team": raw["Home"].astype(str).str.strip(),
        "away_team": raw["Away"].astype(str).str.strip(),
        "home_score": pd.to_numeric(raw["HG"], errors="coerce"),
        "away_score": pd.to_numeric(raw["AG"], errors="coerce"),
    })
    df["tournament"] = cfg["name"]   # K=30 en k_por_torneo (liga doméstica)
    df["neutral"] = False            # clubes: localía SIEMPRE
    df = df.dropna(subset=["date", "home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df.sort_values("date").reset_index(drop=True)
    
    
    def cargar_fixtures(cfg):
    """Fixtures manuales de la jornada (date,home,away). Las líneas que
    empiezan con # son comentarios."""
    path = os.path.join(ROOT, cfg["fixtures_csv"])
    if not os.path.exists(path):
        print(f"  AVISO: no existe {cfg['fixtures_csv']} — sin fixtures que predecir.")
        return pd.DataFrame(columns=["date", "home", "away"])
    fx = pd.read_csv(path, comment="#")
    fx.columns = [c.strip().lower() for c in fx.columns]
    fx["date"] = pd.to_datetime(fx["date"])
    fx["home"] = fx["home"].astype(str).str.strip()
    fx["away"] = fx["away"].astype(str).str.strip()
    return fx.sort_values("date").reset_index(drop=True)


# ------------------------------------------------------------
# Emparejador de resultados con tolerancia de ±1 día
# (reemplaza al de historial.py solo dentro de este proceso)
# ------------------------------------------------------------
def _resultado_real_tolerante(results_df, date, home, away):
    d = pd.Timestamp(date)
    m = results_df[
        (results_df["home_team"] == home)
        & (results_df["away_team"] == away)
        & (results_df["date"] >= d - pd.Timedelta(days=1))
        & (results_df["date"] <= d + pd.Timedelta(days=1))
    ]
    if len(m) == 0:
        return None
    row = m.iloc[0]
    try:
        hs, as_ = row["home_score"], row["away_score"]
        if pd.isna(hs) or pd.isna(as_):
            return None
        return int(hs), int(as_)
    except (ValueError, TypeError):
        return None


# ------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------
def main(league_key):
    if league_key not in LEAGUES:
        print(f"Liga desconocida: {league_key}. Disponibles: {list(LEAGUES)}")
        sys.exit(1)
    cfg = LEAGUES[league_key]

    print(f"=== {cfg['name']} ===")
    df = cargar_datos(cfg)
    print(f"Partidos históricos: {len(df):,} "
          f"({df['date'].min().date()} → {df['date'].max().date()})")

    # Elo de la liga (todos arrancan en 1500; ~13 temporadas lo asientan)
    df_elo, elos = calcular_elo(df)

    # Ventana de entrenamiento + equipos con muestra sólida
    train = df_elo[df_elo["date"].dt.year >= cfg["desde_anio"]]
    counts = pd.concat([train["home_team"], train["away_team"]]).value_counts()
    solidos = set(counts[counts >= cfg["min_partidos"]].index)
    train = train[
        train["home_team"].isin(solidos) & train["away_team"].isin(solidos)
    ].reset_index(drop=True)

    # Auditoría honesta (split temporal 80/20, Elo pre-partido)
    corte = train["date"].quantile(0.8)
    tr_eval = train[train["date"] <= corte].reset_index(drop=True)
    ev_eval = train[train["date"] > corte].reset_index(drop=True)
    print(f"Auditoría (train {len(tr_eval):,} / eval {len(ev_eval):,})...")
    params_eval = ajustar_modelo_elo(tr_eval)
    calib_eval = ajustar_calibrador(tr_eval, params_eval, elos)
    brier = brier_pre_partido(ev_eval, params_eval, calib_eval)
    p_base = (ev_eval["home_score"] > ev_eval["away_score"]).mean()
    brier_naive = p_base * (1 - p_base)
    print(f"  Brier naïve local ({p_base:.1%}): {brier_naive:.4f}")
    print(f"  Brier out-of-sample: {brier:.4f}")
    # Modelo de producción con TODO
    print(f"Entrenando producción ({len(train):,} partidos desde {cfg['desde_anio']})...")
    params = ajustar_modelo_elo(train)
    print(f"  b_elo={params['b_elo']:.3f}  home_adv={params['home_adv']:.3f}")
    calibrador = ajustar_calibrador(train, params, elos)

    # Fixtures de hoy en adelante (hora CDMX, igual que el torneo)
    now_cdmx = datetime.now(timezone.utc) - timedelta(hours=6)
    today = pd.Timestamp(now_cdmx.date())
    fixtures = cargar_fixtures(cfg)
    fixtures = fixtures[fixtures["date"] >= today]

    matches = []
    for _, row in fixtures.iterrows():
        for team in (row["home"], row["away"]):
            if team not in elos:
                print(f"  AVISO: '{team}' no existe en el histórico — "
                      f"¿grafía distinta a football-data? (Elo 1500 por default)")
        pred = predecir_fixture(
            params, calibrador, elos, row["home"], row["away"], neutral=False,
        )
        matches.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "home": row["home"],
            "away": row["away"],
            "city": "",
            "neutral": False,
            **pred,
        })

    # Historial POR LIGA: redirigimos los archivos del módulo historial
    # y su emparejador de resultados, solo dentro de este proceso.
    historial.HISTORY_FILE = os.path.join(ROOT, cfg["history_file"])
    historial.PENDING_FILE = os.path.join(ROOT, cfg["pending_file"])
    historial._resultado_real = _resultado_real_tolerante
    # Verificación: la inyección DEBE quedar activa o nada empareja.
    assert historial._resultado_real is _resultado_real_tolerante
    print("  Emparejador tolerante (±1 día) activo")
    historial.actualizar_historial(matches, df)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "poisson-elo-calibrado",
        "league": cfg["name"],
        "training_matches": int(len(train)),
        "last_result_date": str(df["date"].max().date()),
        "model_brier": round(brier, 4),
        "model_brier_baseline": round(brier_naive, 4),
        "matches": matches,
        "champion_probs": None,
        "history": historial.cargar_historial(),
    }

    out = os.path.join(ROOT, cfg["output_json"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"  {len(matches)} predicciones escritas en {cfg['output_json']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ligamx")

