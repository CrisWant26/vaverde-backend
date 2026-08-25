"""
historial.py — Memoria del modelo VaVerde
==========================================
Acumula las predicciones que el modelo hizo ANTES de cada partido, y las
empareja con el resultado real una vez jugado. Esto alimenta la pantalla
"Historial" de la app, donde el usuario ve predicción vs realidad.

Cómo se conecta (1 línea al final de generate_predictions.py):

    from historial import actualizar_historial
    actualizar_historial(matches, results_df)   # antes de escribir el JSON
    # luego incluye  "history": cargar_historial()  en el dict que serializas

La clave del diseño: el historial vive en un archivo propio (history.json)
que el Action commitea junto al resto. Cada corrida:
  1. Carga el historial previo.
  2. Toma las predicciones "vivas" de AYER y anteriores que ya tienen
     resultado en el dataset, y las archiva con su marcador real.
  3. Guarda. Así nunca se pierde una predicción aunque el partido ya no
     esté en `matches`.
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd

HISTORY_FILE = "history.json"
# También guardamos las predicciones vivas de cada día para poder
# "congelarlas" cuando el partido se juegue (el modelo cambia a diario).
PENDING_FILE = "pending_predictions.json"


def _cargar(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def _guardar(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def _key(date, home, away):
    return f"{date}|{home}|{away}"


def _resultado_real(results_df, date, home, away):
    """Busca el marcador real en el dataset. None si aún no se juega.

    Compara por fecha EXACTA + equipos, lo que ya evita confundir un
    amistoso histórico (otra fecha) con el partido del Mundial.
    """
    m = results_df[
        (results_df["date"].astype(str).str[:10] == date)
        & (results_df["home_team"] == home)
        & (results_df["away_team"] == away)
    ]
    if len(m) == 0:
        return None
    row = m.iloc[0]
    try:
        hs, as_ = row["home_score"], row["away_score"]
        if pd.isna(hs) or pd.isna(as_):
            return None  # fixture sin marcador todavía
        return int(hs), int(as_)
    except (ValueError, TypeError):
        return None


def actualizar_historial(matches, results_df):
    """
    matches: lista de dicts de predicciones VIVAS de esta corrida
             (las mismas que van a 'matches' en predictions.json).
    results_df: DataFrame del dataset con columnas date, home_team,
                away_team, home_score, away_score.

    Efecto: actualiza history.json (archivados con resultado real) y
    pending_predictions.json (vivos, congelados para emparejar después).
    """
    history = _cargar(HISTORY_FILE)
    pending = _cargar(PENDING_FILE)

    hist_keys = {_key(h["date"], h["home"], h["away"]) for h in history}
    pending_by_key = {_key(p["date"], p["home"], p["away"]): p for p in pending}

    # 1) Congelar las predicciones vivas de hoy en 'pending' (sin pisar las ya guardadas;
    #    la PRIMERA predicción que hicimos de un partido es la que cuenta).
    for m in matches:
        k = _key(m["date"], m["home"], m["away"])
        if k not in pending_by_key and k not in hist_keys:
            pending_by_key[k] = {
                "date": m["date"], "home": m["home"], "away": m["away"],
                "p_home": m["p_home"], "p_draw": m["p_draw"], "p_away": m["p_away"],
                "likely_score": m["likely_score"],
                "xg_home": m["xg_home"], "xg_away": m["xg_away"],
            }

    # 2) Mover de pending -> history los que ya tienen resultado real.
    nuevos_pending = {}
    for k, p in pending_by_key.items():
        res = _resultado_real(results_df, p["date"], p["home"], p["away"])
        if res is None:
            nuevos_pending[k] = p  # sigue pendiente
            continue
        hs, as_ = res
        outcome = "home" if hs > as_ else ("away" if as_ > hs else "draw")
        probs = {"home": p["p_home"], "draw": p["p_draw"], "away": p["p_away"]}
        fav = max(probs, key=probs.get)

        # Resultado que IMPLICA el marcador probable del modelo (lo que el
        # usuario ve en pantalla, p.ej. "1-1" => empate). Esta es la señal
        # principal de acierto: es más intuitiva y, sobre los datos del
        # Mundial, acierta el resultado un poco más que el favorito 1X2.
        try:
            mh, ma = map(int, str(p.get("likely_score", "")).split("-"))
            score_outcome = "home" if mh > ma else ("away" if ma > mh else "draw")
        except (ValueError, AttributeError):
            score_outcome = fav  # fallback si el marcador probable no parsea

        # Clasificación de tres niveles:
        #   pleno   = el resultado 1X2 MÁS PROBABLE fue el que salió
        #             (se juzga por la probabilidad que ve el usuario,
        #              no por el marcador probable)
        #   parcial = el marcador probable falló el resultado, PERO el resultado
        #             real estaba a <=12 pts del más probable (zona razonable)
        #   fallo   = falló el resultado y además era poco probable (p.ej.
        #             Canadá: marcador probable 2-0 y empató al 11%)
        gap = probs[fav] - probs[outcome]
        if fav == outcome:
            tier = "pleno"
        elif gap <= 0.12:
            tier = "parcial"
        else:
            tier = "fallo"
        history.append({
            **p,
            "real_home": hs, "real_away": as_,
            "outcome": outcome,
            "fav_hit": score_outcome == outcome,   # acierto = marcador probable acertó el resultado
            "tier": tier,
            "p_fav": round(probs[fav], 3),
            "p_outcome": round(probs[outcome], 3),
            "score_outcome": score_outcome,
        })

    _guardar(HISTORY_FILE, history)
    _guardar(PENDING_FILE, list(nuevos_pending.values()))
    return history


def resumen_historial(history):
    """Métricas agregadas honestas: tres niveles + Brier 1X2.

    No mezclamos plenos con parciales en una sola cifra para no inflar la
    tasa. El Brier (continuo) sigue siendo la métrica estrella: premia poco
    un empate predicho al 37% y castiga mucho un favorito al 84% que falla.
    """
    if not history:
        return {"n": 0, "plenos": 0, "parciales": 0, "fallos": 0,
                "fav_hits": 0, "fav_rate": 0.0, "brier": None}
    n = len(history)
    plenos = sum(1 for h in history if h.get("tier") == "pleno")
    parciales = sum(1 for h in history if h.get("tier") == "parcial")
    fallos = sum(1 for h in history if h.get("tier") == "fallo")
    brier = 0.0
    for h in history:
        probs = {"home": h["p_home"], "draw": h["p_draw"], "away": h["p_away"]}
        y = {"home": 0, "draw": 0, "away": 0}
        y[h["outcome"]] = 1
        brier += sum((probs[k] - y[k]) ** 2 for k in probs)
    return {
        "n": n,
        "plenos": plenos,
        "parciales": parciales,
        "fallos": fallos,
        "fav_hits": plenos,                    # solo plenos cuentan como acierto pleno
        "fav_rate": round(plenos / n, 3),
        "brier": round(brier / n, 3),
    }


def cargar_historial():
    """Para incluir en predictions.json: lista ordenada (más reciente primero) + resumen."""
    history = _cargar(HISTORY_FILE)
    history_sorted = sorted(history, key=lambda h: h["date"], reverse=True)
    return {
        "matches": history_sorted,
        "summary": resumen_historial(history),
    }
