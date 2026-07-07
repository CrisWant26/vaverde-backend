"""
Simulación Monte Carlo del Mundial 2026 — DESDE CUARTOS
========================================================
Los octavos terminaron. La simulación arranca con los 4 cruces REALES
de cuartos (los 8 equipos que siguen vivos) y juega el bracket oficial
hasta la final.

Cada simulación:
  1. Juega los 4 cuartos (empate -> penales). Todo neutral (ya no hay
     anfitriones vivos: USA, México y Canadá quedaron eliminados).
  2. Cascada por el bracket oficial: semifinales y final.
  3. Registra hasta dónde llegó cada equipo.

Los eliminados en octavos ya NO aparecen.

NIVELES de ronda:
  3 = cuartos (punto de partida de los 8 vivos)
  4 = semifinal
  5 = final (jugó la final)
  6 = campeón
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
# BRACKET OFICIAL — Cuartos en ORDEN DE LLAVE.
# Pares consecutivos se cruzan en semifinales:
#   SF1: ganador(0) vs ganador(1)   [France/Morocco vs Spain/Belgium]
#   SF2: ganador(2) vs ganador(3)   [Norway/England vs Argentina/Switzerland]
# Final: SF1 vs SF2.
# ============================================================
CUARTOS = [
    # --- Mitad izquierda ---
    ("France", "Morocco"),               # 0  @ Foxborough
    ("Spain", "Belgium"),                # 1  @ Inglewood
    # --- Mitad derecha ---
    ("Norway", "England"),               # 2  @ Miami Gardens
    ("Argentina", "Switzerland"),        # 3  @ Kansas City
]
# Todo neutral: ya no queda ningún anfitrión vivo.


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
    """Una simulación desde cuartos. Devuelve dict equipo -> nivel máximo."""
    ronda = {}
    for a, b in CUARTOS:
        ronda[a] = 3; ronda[b] = 3  # los 8 vivos arrancan en cuartos

    # --- Cuartos ---
    ganadores = []
    for h, a in CUARTOS:
        w = jugar_partido(h, a, params, elos, rng, neutral=True)
        ronda[w] = 4  # avanzó a semifinales
        ganadores.append(w)

    # --- Semifinales y final en cascada (neutral) ---
    nivel = 5
    actual = ganadores
    while len(actual) > 1:
        siguiente = []
        for k in range(0, len(actual), 2):
            w = jugar_partido(actual[k], actual[k+1], params, elos, rng, neutral=True)
            ronda[w] = nivel
            siguiente.append(w)
        actual = siguiente
        nivel += 1
    # Semifinalistas ganadores quedan en 5 (finalistas); el campeón en 6.
    return ronda


def simular(N=10000, seed=0):
    params, elos = cargar_modelo()
    rng = np.random.default_rng(seed)
    equipos = [t for par in CUARTOS for t in par]

    avanza = defaultdict(int)   # gana su cuarto (llega a semis, nivel>=4)
    final = defaultdict(int)    # llega a la final (nivel>=5)
    campeon = defaultdict(int)  # campeón (nivel>=6)
    suma_ronda = defaultdict(int)

    for _ in range(N):
        r = simular_torneo(params, elos, rng)
        for t in equipos:
            nivel = r.get(t, 3)
            suma_ronda[t] += nivel
            if nivel >= 4: avanza[t] += 1
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
    print("Simulando desde cuartos (10,000 torneos)...")
    df = simular(N=10000, seed=42)
    pd.set_option("display.max_rows", 10)
    print("\n=== PROBABILIDADES (8 equipos vivos) ===\n")
    print(df.to_string(index=False))
    df.to_csv("simulacion_mundial.csv", index=False)
    print("\nGuardado en simulacion_mundial.csv")
