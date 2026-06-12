"""
Cálculo de ratings Elo de selecciones (World Football Elo)
===========================================================
No dependemos de fuentes externas: calculamos el Elo desde el histórico
de 49k partidos con el algoritmo estándar.

Fórmula (World Football Elo Ratings):
  - Resultado esperado del local:  We = 1 / (1 + 10^(-(Elo_l - Elo_v + ventaja)/400))
  - Nuevo Elo = Elo_viejo + K * G * (W - We)
        W  = resultado real (1 gana, 0.5 empata, 0 pierde)
        K  = constante de importancia del partido (Mundial alto, amistoso bajo)
        G  = factor de margen de goles (ganar por mucho suma más)
        ventaja = ~65 pts si el local NO juega en cancha neutral

Esto produce, para cada partido, el Elo de ambos equipos JUSTO ANTES de jugarlo,
que es lo que luego le daremos al modelo como variable (sin fuga de información).
"""
import numpy as np
import pandas as pd


# K por importancia del torneo (valores estándar World Football Elo)
def k_por_torneo(torneo):
    t = str(torneo).lower()
    if "world cup" in t and "qualif" not in t:
        return 60
    if "confederations" in t or "continental" in t:
        return 50
    if any(x in t for x in ["uefa euro", "copa am", "african cup", "asian cup", "gold cup", "nations league"]):
        return 50
    if "qualif" in t:
        return 40
    if "friendly" in t:
        return 20
    return 30  # otros torneos


def factor_goles(dif):
    """G: ganar por más goles ajusta más el rating."""
    d = abs(dif)
    if d <= 1:
        return 1.0
    if d == 2:
        return 1.5
    return (11 + d) / 8.0  # 3 goles -> 1.75, 4 -> 1.875, etc.


def calcular_elo(df, elo_inicial=1500, ventaja_local=65):
    """
    Recorre los partidos en orden cronológico y mantiene el Elo de cada equipo.
    Devuelve el df con dos columnas nuevas: elo_home, elo_away (ANTES del partido)
    y un diccionario con el Elo final de cada selección.
    """
    df = df.sort_values("date").reset_index(drop=True)
    elos = {}
    elo_h_list = np.zeros(len(df))
    elo_a_list = np.zeros(len(df))

    for i, r in enumerate(df.itertuples()):
        h, a = r.home_team, r.away_team
        eh = elos.get(h, elo_inicial)
        ea = elos.get(a, elo_inicial)
        elo_h_list[i] = eh
        elo_a_list[i] = ea

        # ventaja de local solo si no es neutral
        va = 0 if getattr(r, "neutral") in (True, "TRUE", "True") else ventaja_local
        we_h = 1 / (1 + 10 ** (-(eh - ea + va) / 400))

        hs, as_ = r.home_score, r.away_score
        if hs > as_:
            w_h = 1.0
        elif hs < as_:
            w_h = 0.0
        else:
            w_h = 0.5

        K = k_por_torneo(getattr(r, "tournament"))
        G = factor_goles(hs - as_)
        cambio = K * G * (w_h - we_h)
        elos[h] = eh + cambio
        elos[a] = ea - cambio  # juego de suma cero

    df = df.copy()
    df["elo_home"] = elo_h_list
    df["elo_away"] = elo_a_list
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    return df, elos


if __name__ == "__main__":
    df = pd.read_csv("/mnt/user-data/uploads/results.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    df_elo, elos = calcular_elo(df)

    # Top 20 selecciones por Elo actual
    rank = sorted(elos.items(), key=lambda x: -x[1])[:20]
    print("Ranking Elo actual (top 20):")
    for i, (t, e) in enumerate(rank, 1):
        print(f"  {i:2d}. {t:22s} {e:6.0f}")

    # Guardar para usar en el modelo
    import pickle
    with open("/home/claude/elos_finales.pkl", "wb") as f:
        pickle.dump(elos, f)
    df_elo.to_pickle("/home/claude/df_con_elo.pkl")
    print("\nElo calculado y guardado.")

    # comparar Elo de los partidos de la captura
    print("\nElo de los equipos de tu captura:")
    for t in ["Mexico", "South Africa", "South Korea", "Czech Republic"]:
        print(f"  {t:18s} {elos.get(t, 1500):.0f}")
