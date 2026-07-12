"""
Simulación Monte Carlo del Mundial 2026 — DESDE SEMIFINALES
============================================================
Los cuartos terminaron (4 de 4 para el escenario principal del modelo).
La simulación arranca con los 2 cruces REALES de semifinales (los 4
equipos vivos) y juega hasta la final.

Cada simulación:
  1. Juega las 2 semifinales (empate -> penales). Todo neutral.
  2. Juega la final entre los dos ganadores.
  3. Registra hasta dónde llegó cada equipo.

Los eliminados en cuartos ya NO aparecen.

NIVELES de ronda:
  4 = semifinal (punto de partida de los 4 vivos)
  5 = final (jugó la final)
  6 = campeón

NOTA de compatibilidad: con 4 equipos, ganar tu semifinal ES llegar a la
final, así que p_avanzar_% y p_final_% son idénticas por definición.
Se mantienen AMBAS columnas para no romper champion_probs() de
generate_predictions.py.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pickle
from collections import defaultdict


def lambdas(params, elos, home, away, neutral=True):
    """Goles esperados del modelo Poisson+Elo (idéntica a la original)."""
    atk, dfn = params["attack"], params["defense"]
    eh = elos.get(home, 1500); ea = elos.get(away, 1500)
    elo_diff = (eh - ea) / 400.0
    ha = 0.0 if neutral else params["home_adv"]
    c, b = params["c"], params["b_elo"]
    a_h = atk.get(home, 0.0); a_a = atk.get(away, 0.0)
    d_h = dfn.get(home, 0.0); d_a = dfn.get(away, 0.0)
    lh = np.exp(np.clip(c + a_h - d_a + b*elo_diff + ha, -2, 3))
    la = np.exp(np.clip(c + a_a - d_h - b*elo_diff, -2, 3))
    return lh, la


# ============================================================
# BRACKET OFICIAL — Semifinales REALES.
# El ganador de SEMIS[0] juega la final contra el ganador de SEMIS[1].
# Todo neutral: ningún anfitrión vivo.
# ============================================================
SEMIS = [
    ("Spain", "France"),        # 0  SF1 — 2026-07-14
    ("Argentina", "England"),   # 1  SF2 — 2026-07-15
]


def cargar_modelo():
    with open("modelo_elo.pkl", "rb") as f:
        M = pickle.load(f)
    return M["params"], M["elos"]


def jugar_partido(h, a, params, elos, rng, neutral=True):
    """Partido a muerte: empate -> penales (calibrado 54.4%, divisor 1545)."""
    lh, la = lambdas(params, elos, h, a, neutral=neutral)
    gh, ga = rng.poisson(lh), rng.poisson(la)
    if gh > ga: return h
    if ga > gh: return a
    eh, ea = elos.get(h, 1500), elos.get(a, 1500)
    p_h = 1 / (1 + 10 ** (-(eh - ea) / 1545))
    return h if rng.random() < p_h else a


def simular_torneo(params, elos, rng):
    """Una simulación desde semifinales. Devuelve dict equipo -> nivel máximo."""
    ronda = {}
    for a, b in SEMIS:
        ronda[a] = 4; ronda[b] = 4  # los 4 vivos arrancan en semifinales

    # --- Semifinales ---
    finalistas = []
    for h, a in SEMIS:
        w = jugar_partido(h, a, params, elos, rng, neutral=True)
        ronda[w] = 5  # llegó a la final
        finalistas.append(w)

    # --- Final ---
    campeon = jugar_partido(finalistas[0], finalistas[1], params, elos, rng, neutral=True)
    ronda[campeon] = 6
    return ronda


def simular(N=10000, seed=0):
    params, elos = cargar_modelo()
    rng = np.random.default_rng(seed)
    equipos = [t for par in SEMIS for t in par]

    avanza = defaultdict(int)   # gana su semi (= llega a la final, nivel>=5)
    final = defaultdict(int)    # llega a la final (nivel>=5) — igual a avanza
    campeon = defaultdict(int)  # campeón (nivel>=6)
    suma_ronda = defaultdict(int)

    for _ in range(N):
        r = simular_torneo(params, elos, rng)
        for t in equipos:
            nivel = r.get(t, 4)
            suma_ronda[t] += nivel
            if nivel >= 5: avanza[t] += 1
            if nivel >= 5: final[t] += 1
            if nivel >= 6: campeon[t] += 1

    import pandas as pd
    filas = []
    for t in equipos:
        filas.append({
            "equipo": t,
            "elo": round(elos.get(t, 1500)),
            "p_avanzar_%": round(100*avanza[t]/N, 1),
            "p_final_%": round(100*final[t]/N, 1),
            "p_campeon_%": round(100*campeon[t]/N, 1),
            "ronda_media": round(suma_ronda[t]/N, 2),
        })
    return pd.DataFrame(filas).sort_values("p_campeon_%", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    import pandas as pd
    print("Simulando desde semifinales (10,000 torneos)...")
    df = simular(N=10000, seed=42)
    pd.set_option("display.max_rows", 10)
    print("\n=== PROBABILIDADES (4 equipos vivos) ===\n")
    print(df.to_string(index=False))
    df.to_csv("simulacion_mundial.csv", index=False)
    print("\nGuardado en simulacion_mundial.csv")
