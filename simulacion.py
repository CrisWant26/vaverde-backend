"""
Simulación Monte Carlo del Mundial 2026 — DESDE OCTAVOS
========================================================
Los dieciseisavos terminaron. La simulación arranca con los 8 cruces
REALES de octavos (los 16 equipos que siguen vivos) y juega el bracket
oficial hasta la final.

Cada simulación:
  1. Juega los 8 octavos (empate -> penales), con localía donde aplica.
  2. Cascada por el bracket oficial: cuartos, semis, final.
  3. Registra hasta dónde llegó cada equipo.

Los eliminados en 16vos ya NO aparecen.

NIVELES de ronda:
  2 = octavos (punto de partida de los 16 vivos)
  3 = cuartos
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
# BRACKET OFICIAL — Octavos en ORDEN DE LLAVE.
# Pares consecutivos se cruzan en cuartos:
#   QF1: ganador(0) vs ganador(1)   [Paraguay/France vs Canada/Morocco]
#   QF2: ganador(2) vs ganador(3)   [Portugal/Spain vs USA/Belgium]
#   QF3: ganador(4) vs ganador(5)   [Brazil/Norway vs Mexico/England]
#   QF4: ganador(6) vs ganador(7)   [Argentina/Egypt vs Switzerland/Colombia]
# Luego SF: QF1vsQF2, QF3vsQF4. Final: SF1vsSF2.
# ============================================================
OCTAVOS = [
    # --- Mitad izquierda ---
    ("Paraguay", "France"),              # 0
    ("Canada", "Morocco"),               # 1
    ("Portugal", "Spain"),               # 2
    ("United States", "Belgium"),        # 3
    # --- Mitad derecha ---
    ("Brazil", "Norway"),                # 4
    ("Mexico", "England"),               # 5
    ("Argentina", "Egypt"),              # 6
    ("Switzerland", "Colombia"),         # 7
]

# Localía en octavos: False = el home es anfitrión jugando en su país.
NEUTRAL_R16 = [
    True,   # 0 Paraguay/France @ Philadelphia
    True,   # 1 Canada/Morocco @ Houston (Canadá NO juega en su país)
    True,   # 2 Portugal/Spain @ Arlington
    False,  # 3 USA/Belgium @ Seattle (USA local)
    True,   # 4 Brazil/Norway @ East Rutherford
    False,  # 5 Mexico/England @ Azteca (México local)
    True,   # 6 Argentina/Egypt @ Atlanta
    True,   # 7 Switzerland/Colombia @ Vancouver
]
# De cuartos en adelante: neutral (sedes sin local definido).


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
    """Una simulación desde octavos. Devuelve dict equipo -> nivel máximo."""
    ronda = {}
    for a, b in OCTAVOS:
        ronda[a] = 2; ronda[b] = 2  # los 16 vivos arrancan en octavos

    # --- Octavos (con localía donde aplica) ---
    ganadores = []
    for idx, (h, a) in enumerate(OCTAVOS):
        w = jugar_partido(h, a, params, elos, rng, neutral=NEUTRAL_R16[idx])
        ronda[w] = 3  # avanzó a cuartos
        ganadores.append(w)

    # --- Cuartos, semis, final en cascada (neutral) ---
    nivel = 4
    actual = ganadores
    while len(actual) > 1:
        siguiente = []
        for k in range(0, len(actual), 2):
            w = jugar_partido(actual[k], actual[k+1], params, elos, rng, neutral=True)
            ronda[w] = nivel
            siguiente.append(w)
        actual = siguiente
        nivel += 1
    # El último nivel asignado al ganador final es 6 (campeón).
    return ronda


def simular(N=10000, seed=0):
    params, elos = cargar_modelo()
    rng = np.random.default_rng(seed)
    equipos = [t for par in OCTAVOS for t in par]

    avanza = defaultdict(int)   # gana su octavo (llega a cuartos, nivel>=3)
    final = defaultdict(int)    # llega a la final (nivel>=5)
    campeon = defaultdict(int)  # campeón (nivel>=6)
    suma_ronda = defaultdict(int)

    for _ in range(N):
        r = simular_torneo(params, elos, rng)
        for t in equipos:
            nivel = r.get(t, 2)
            suma_ronda[t] += nivel
            if nivel >= 3: avanza[t] += 1
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
    print("Simulando desde octavos (10,000 torneos)...")
    df = simular(N=10000, seed=42)
    pd.set_option("display.max_rows", 20)
    print("\n=== PROBABILIDADES (16 equipos vivos) ===\n")
    print(df.to_string(index=False))
    df.to_csv("simulacion_mundial.csv", index=False)
    print("\nGuardado en simulacion_mundial.csv")
