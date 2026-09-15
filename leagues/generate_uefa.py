"""
generate_uefa.py — Predicciones de Champions League
====================================================
A diferencia de las ligas domésticas, aquí los resultados NO llegan solos:
football-data no cubre competencias UEFA y openfootball va con retraso.
Los capturas tú en leagues/results_uefa.csv, igual que hacías con el
torneo de selecciones.

CÓMO FUNCIONA EL ELO:
El Elo se calcula sobre TODO junto y en orden cronológico:
  1. Las 7 ligas domésticas (football-data, automático)
  2. El histórico UEFA de openfootball (traducido con mapeo_uefa.csv)
  3. Tus resultados capturados a mano (results_uefa.csv)

Los partidos europeos son los "puentes" que hacen comparables los Elos
entre ligas. Sin ellos, el 1750 del Madrid y el 1750 del Ajax medirían
cosas distintas.

VALIDACIÓN (medida el 14-sep-2026):
  - Neutral en predicciones domésticas: +0.0001 promedio en 7 ligas
  - +0.0278 sobre baseline en partidos UEFA (n=179)
  - LIMITACIÓN: la validación cubrió élite vs élite. Los cruces contra
    equipos de ligas no cubiertas están peor calibrados.

FORMATO de results_uefa.csv:
    date,home,away,home_score,away_score,competicion
    2026-09-16,Real Madrid,Ajax,3,1,CL
    2026-09-17,Ath Bilbao,Napoli,NA,NA,CL

  - NA,NA = partido futuro (el modelo lo predice)
  - Con marcador = ya se jugó (alimenta Elo e historial)
  - Usa grafías de football-data para equipos de las 7 ligas cubiertas
    (Real Madrid, Ath Bilbao, Bayern Munich, Man City, Paris SG...)
  - Para equipos de otras ligas usa el nombre que quieras, pero sé
    CONSISTENTE: el mismo equipo siempre igual.

Uso:  python leagues/generate_uefa.py
"""
import os
import sys
import json
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from elo import factor_goles
from modelo_elo import ajustar_modelo_elo, ajustar_calibrador
from generate_predictions import predecir_fixture, brier_pre_partido
from adaptador_main import cargar_liga_main
import historial

CFG = {
    "name": "Champions League",
    "results_csv": "leagues/results_uefa.csv",
    "mapeo_csv": "leagues/mapeo_uefa.csv",
    "uefa_historico": "leagues/uefa_historico.csv",
    "output_json": "docs/leagues/uefa.json",
    "history_file": "leagues/history_uefa.json",
    "pending_file": "leagues/pending_uefa.json",
    "desde_anio": 2013,
    "min_partidos": 20,
}

LIGAS = {
    "SP1": "LaLiga", "E0": "Premier League", "I1": "Serie A",
    "D1": "Bundesliga", "F1": "Ligue 1", "N1": "Eredivisie",
    "P1": "Primeira Liga",
}

K_LIGA, K_UEFA, VENTAJA_LOCAL = 30, 40, 65

# Equipos de ligas NO cubiertas cuyo nombre en tu CSV difiere del que
# openfootball usa en el histórico. Sin esto arrancarían con Elo 1500
# y perderían todo su recorrido europeo.
MOSTRAR_UEFA = {
    "aek athen": "AEK Athens",
    "bodø/glimt": "Bodø/Glimt",
    "fenerbahçe": "Fenerbahçe",
    "slavia praha": "Slavia Praha",
}

ALIAS_UEFA = {
    "AEK Athens": "aek athen",
    "Bodo/Glimt": "bodø/glimt",
    "Fenerbahce": "fenerbahçe",
    "Slavia Prague": "slavia praha",
}



# ------------------------------------------------------------
# Carga de las tres fuentes
# ------------------------------------------------------------
def cargar_ligas():
    frames = []
    for cod, nombre in LIGAS.items():
        df = cargar_liga_main(cod, nombre=nombre, verbose=False)
        df = df[["date", "home_team", "away_team", "home_score", "away_score"]].copy()
        df["es_uefa"] = False
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def cargar_uefa_historico():
    """Histórico de openfootball, con los nombres traducidos."""
    path = os.path.join(ROOT, CFG["uefa_historico"])
    if not os.path.exists(path):
        print(f"  AVISO: no existe {path} — el Elo se calculará sin el")
        print(f"  histórico europeo (los puentes entre ligas serán pocos).")
        return pd.DataFrame(columns=["date", "home_team", "away_team",
                                     "home_score", "away_score", "es_uefa"])

    mapeo_path = os.path.join(ROOT, CFG["mapeo_csv"])
    m = pd.read_csv(mapeo_path).fillna("")
    m = m[m["openfootball"] != ""]
    mapeo = dict(zip(m["openfootball"], m["football_data"]))

    u = pd.read_csv(path)
    u["date"] = pd.to_datetime(u["date"])
    u["home_team"] = u["home_team"].map(lambda x: mapeo.get(x, f"UEFA:{x}"))
    u["away_team"] = u["away_team"].map(lambda x: mapeo.get(x, f"UEFA:{x}"))
    u = u[["date", "home_team", "away_team", "home_score", "away_score"]].copy()
    u["es_uefa"] = True
    return u


def cargar_results_manuales():
    """Tus capturas. Devuelve (jugados, futuros)."""
    path = os.path.join(ROOT, CFG["results_csv"])
    if not os.path.exists(path):
        print(f"  AVISO: no existe {CFG['results_csv']}")
        vacio = pd.DataFrame(columns=["date", "home", "away",
                                      "home_score", "away_score"])
        return vacio, vacio

    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df["home"] = df["home"].astype(str).str.strip().map(
        lambda x: ALIAS_UEFA.get(x, x))
    df["away"] = df["away"].astype(str).str.strip().map(
        lambda x: ALIAS_UEFA.get(x, x))
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    jugados = df.dropna(subset=["home_score", "away_score"]).copy()
    jugados["home_score"] = jugados["home_score"].astype(int)
    jugados["away_score"] = jugados["away_score"].astype(int)
    futuros = df[df["home_score"].isna()].copy()
    return jugados, futuros


# ------------------------------------------------------------
# Elo unificado
# ------------------------------------------------------------
def calcular_elo_unificado(df, elo_inicial=1500):
    df = df.sort_values("date").reset_index(drop=True)
    elos = {}
    eh = np.zeros(len(df))
    ea = np.zeros(len(df))

    for i, r in enumerate(df.itertuples()):
        h, a = r.home_team, r.away_team
        e_h = elos.get(h, elo_inicial)
        e_a = elos.get(a, elo_inicial)
        eh[i], ea[i] = e_h, e_a
        we = 1 / (1 + 10 ** (-(e_h - e_a + VENTAJA_LOCAL) / 400))
        w = 1.0 if r.home_score > r.away_score else (0.0 if r.home_score < r.away_score else 0.5)
        K = K_UEFA if r.es_uefa else K_LIGA
        c = K * factor_goles(r.home_score - r.away_score) * (w - we)
        elos[h] = e_h + c
        elos[a] = e_a - c

    df = df.copy()
    df["elo_home"] = eh
    df["elo_away"] = ea
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    df["neutral"] = False
    df["tournament"] = "UEFA"
    return df, elos


# ------------------------------------------------------------
# Emparejador: para UEFA los resultados los pones TÚ
# ------------------------------------------------------------
def _resultado_manual(results_df, date, home, away):
    d = pd.Timestamp(date)
    m = results_df[
        (results_df["home"] == home) & (results_df["away"] == away)
        & (results_df["date"] >= d - pd.Timedelta(days=2))
        & (results_df["date"] <= d + pd.Timedelta(days=2))
    ]
    if len(m) == 0:
        return None
    row = m.iloc[0]
    if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
        return None
    return int(row["home_score"]), int(row["away_score"])


# ------------------------------------------------------------
def main():
    print(f"=== {CFG['name']} ===")

    print("Cargando ligas domésticas...")
    ligas = cargar_ligas()
    print(f"  {len(ligas):,} partidos de 7 ligas")

    print("Cargando histórico UEFA (openfootball)...")
    uefa_hist = cargar_uefa_historico()
    print(f"  {len(uefa_hist):,} partidos europeos históricos")

    print("Cargando resultados capturados a mano...")
    jugados, futuros = cargar_results_manuales()
    print(f"  {len(jugados)} jugados, {len(futuros)} por jugar")

    manuales = pd.DataFrame()
    if len(jugados):
        manuales = pd.DataFrame({
            "date": jugados["date"],
            "home_team": jugados["home"],
            "away_team": jugados["away"],
            "home_score": jugados["home_score"],
            "away_score": jugados["away_score"],
            "es_uefa": True,
        })

    todo = pd.concat([ligas, uefa_hist, manuales], ignore_index=True)
    todo = todo.dropna(subset=["date", "home_score", "away_score"])
    todo = todo[todo["date"].dt.year >= CFG["desde_anio"]]
    print(f"\nTotal para el Elo: {len(todo):,} partidos "
          f"({todo['date'].min().date()} → {todo['date'].max().date()})")

    print("Calculando Elo unificado...")
    df_elo, elos = calcular_elo_unificado(todo)

    # Entrenamiento: solo partidos domésticos (para ataque/defensa fiable)
    dom = df_elo[~df_elo["es_uefa"]]
    counts = pd.concat([dom["home_team"], dom["away_team"]]).value_counts()
    solidos = set(counts[counts >= CFG["min_partidos"]].index)
    train = dom[dom["home_team"].isin(solidos) &
                dom["away_team"].isin(solidos)].reset_index(drop=True)

    corte = train["date"].quantile(0.8)
    tr_ev = train[train["date"] <= corte].reset_index(drop=True)
    ev = train[train["date"] > corte].reset_index(drop=True)
    print(f"Auditoría (train {len(tr_ev):,} / eval {len(ev):,})...")
    p_ev = ajustar_modelo_elo(tr_ev)
    c_ev = ajustar_calibrador(tr_ev, p_ev, elos)
    brier = brier_pre_partido(ev, p_ev, c_ev)
    print(f"  Brier out-of-sample (doméstico): {brier:.4f}")

    print(f"Entrenando producción ({len(train):,} partidos)...")
    params = ajustar_modelo_elo(train)
    print(f"  b_elo={params['b_elo']:.3f}  home_adv={params['home_adv']:.3f}")
    calibrador = ajustar_calibrador(train, params, elos)

    # Predicciones de los partidos futuros
    now_cdmx = datetime.now(timezone.utc) - timedelta(hours=6)
    hoy = pd.Timestamp(now_cdmx.date())
    pendientes = futuros[futuros["date"] >= hoy]

    matches = []
    for _, row in pendientes.iterrows():
        for t in (row["home"], row["away"]):
            if t not in elos:
                print(f"  AVISO: '{t}' sin historial — Elo 1500 por default")
        pred = predecir_fixture(params, calibrador, elos,
                                row["home"], row["away"], neutral=False)
        matches.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "home": MOSTRAR_UEFA.get(row["home"], row["home"]),
            "away": MOSTRAR_UEFA.get(row["away"], row["away"]),
            "city": "",
            "neutral": False,
            **pred,
        })

    # Historial: el emparejador lee TU csv, no football-data
    historial.HISTORY_FILE = os.path.join(ROOT, CFG["history_file"])
    historial.PENDING_FILE = os.path.join(ROOT, CFG["pending_file"])
    historial._resultado_real = _resultado_manual
    todos_manuales = pd.concat([jugados, futuros], ignore_index=True)
    historial.actualizar_historial(matches, todos_manuales)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "poisson-elo-unificado",
        "league": CFG["name"],
        "training_matches": int(len(train)),
        "last_result_date": str(todo["date"].max().date()),
        "model_brier": round(brier, 4),
        "matches": matches,
        "champion_probs": None,
        "history": historial.cargar_historial(),
    }

    out = os.path.join(ROOT, CFG["output_json"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"  {len(matches)} predicciones escritas en {CFG['output_json']}")


if __name__ == "__main__":
    main()
