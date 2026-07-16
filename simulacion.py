"""
Simulación Monte Carlo del Mundial 2026 — LA FINAL
===================================================
Última migración del torneo. Las semifinales terminaron (el escenario
principal del modelo va 6/6 en eliminación directa: 4/4 cuartos, 2/2
semis). Queda UN partido: la final en East Rutherford.

Cada simulación juega la final (empate -> penales, cancha neutral)
y corona a un campeón.

NIVELES de ronda:
  5 = finalista (punto de partida de los 2 vivos)
  6 = campeón

NOTA de compatibilidad: con 2 equipos, p_avanzar_% y p_final_% son 100
por definición (ya están EN la final). Se mantienen ambas columnas para
no romper champion_probs() de generate_predictions.py.
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
# LA FINAL — 2026-07-19, East Rutherford. Cancha neutral.
# (Si la designación oficial local/visita es al revés, voltea el par;
#  al modelo le da igual en neutral.)
# ============================================================
FINAL = ("Spain", "Argentina")


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
    """Una simulación de la final. Devuelve dict equipo -> nivel máximo."""
    h, a = FINAL
    ronda = {h: 5, a: 5}  # ambos ya son finalistas
    campeon = jugar_partido(h, a, params, elos, rng, neutral=True)
    ronda[campeon] = 6
    return ronda


def simular(N=10000, seed=0):
    params, elos = cargar_modelo()
    rng = np.random.default_rng(seed)
    equipos = list(FINAL)

    avanza = defaultdict(int)   # finalista (nivel>=5): 100% para ambos
    final = defaultdict(int)    # idem
    campeon = defaultdict(int)  # campeón (nivel>=6)
    suma_ronda = defaultdict(int)

    for _ in range(N):
        r = simular_torneo(params, elos, rng)
        for t in equipos:
            nivel = r.get(t, 5)
            suma_ronda[t] += nivel
            if nivel >= 5:
                avanza[t] += 1
                final[t] += 1
            if nivel >= 6:
                campeon[t] += 1

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
    print("Simulando la final (10,000 partidos)...")
    df = simular(N=10000, seed=42)
    pd.set_option("display.max_rows", 10)
    print("\n=== LA FINAL: PROBABILIDADES DE CAMPEÓN ===\n")
    print(df.to_string(index=False))
    df.to_csv("simulacion_mundial.csv", index=False)
    print("\nGuardado en simulacion_mundial.csv")
