"""
Simulación Monte Carlo del Mundial 2026 — DESDE DIECISEISAVOS
=============================================================
Versión de eliminatorias: los grupos YA terminaron. Esta simulación arranca
directo con los 16 cruces REALES de dieciseisavos (las 32 selecciones que
clasificaron) y juega el bracket oficial hasta la final.

Cada simulación:
  1. Juega los 16 dieciseisavos (empate -> penales).
  2. Cruza los ganadores en octavos según el BRACKET OFICIAL.
  3. Cascada: cuartos, semis, final.
  4. Registra hasta dónde llegó cada equipo.

Las selecciones eliminadas en grupos ya NO aparecen (no están en el bracket).

NIVELES de ronda (para compatibilidad con generate_predictions.py):
  1 = dieciseisavos (R32, punto de partida de todos)
  2 = octavos (R16)
  3 = cuartos (QF)
  4 = semifinal (SF)
  5 = final (jugó la final)
  6 = campeón (ganó la final)
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
# BRACKET OFICIAL — Dieciseisavos en ORDEN DE LLAVE (imagen FIFA).
# El orden define el camino: pares consecutivos se cruzan en octavos.
# Mitad izquierda (índices 0-7), mitad derecha (índices 8-15).
# ============================================================
DIECISEISAVOS = [
    # --- Mitad izquierda ---
    ("Germany", "Paraguay"),                       # 0
    ("France", "Sweden"),                          # 1
    ("South Africa", "Canada"),                    # 2
    ("Netherlands", "Morocco"),                    # 3
    ("Portugal", "Croatia"),                       # 4
    ("Spain", "Austria"),                          # 5
    ("United States", "Bosnia and Herzegovina"),   # 6
    ("Belgium", "Senegal"),                        # 7
    # --- Mitad derecha ---
    ("Brazil", "Japan"),                           # 8
    ("Ivory Coast", "Norway"),                     # 9
    ("Mexico", "Ecuador"),                         # 10
    ("England", "DR Congo"),                       # 11
    ("Argentina", "Cape Verde"),                   # 12
    ("Australia", "Egypt"),                        # 13
    ("Switzerland", "Algeria"),                    # 14
    ("Colombia", "Ghana"),                         # 15
]

# Sedes de los dieciseisavos: True = neutral, False = local (anfitrión en casa).
# Solo México (Azteca) y USA (Santa Clara) son locales en esta ronda.
NEUTRAL_R32 = [
    True,   # 0 Germany/Paraguay @ Foxborough
    True,   # 1 France/Sweden @ East Rutherford
    True,   # 2 South Africa/Canada @ Inglewood (Canadá NO juega en casa)
    True,   # 3 Netherlands/Morocco @ Guadalupe
    True,   # 4 Portugal/Croatia @ Toronto
    True,   # 5 Spain/Austria @ Inglewood
    False,  # 6 USA/Bosnia @ Santa Clara (USA local)
    True,   # 7 Belgium/Senegal @ Seattle
    True,   # 8 Brazil/Japan @ Houston
    True,   # 9 Ivory Coast/Norway @ Arlington
    False,  # 10 Mexico/Ecuador @ Azteca (México local)
    True,   # 11 England/DR Congo @ Atlanta
    True,   # 12 Argentina/Cape Verde @ Miami Gardens
    True,   # 13 Australia/Egypt @ Arlington
    True,   # 14 Switzerland/Algeria @ Vancouver
    True,   # 15 Colombia/Ghana @ Kansas City
]
# A partir de octavos asumimos neutral (sedes aún no fijadas a un local).


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
    """Una simulación desde dieciseisavos. Devuelve dict equipo -> nivel máx."""
    ronda = {}
    for a, b in DIECISEISAVOS:
        ronda[a] = 1; ronda[b] = 1  # todos arrancan en R32 (nivel 1)

    # --- Dieciseisavos (con localía donde aplica) ---
    ganadores = []
    for idx, (h, a) in enumerate(DIECISEISAVOS):
        w = jugar_partido(h, a, params, elos, rng, neutral=NEUTRAL_R32[idx])
        ronda[w] = 2  # avanzó a octavos
        ganadores.append(w)

    # --- Octavos, cuartos, semis, final en cascada (neutral) ---
    # nivel 2=octavos ya asignado a ganadores R32; ahora jugamos rondas siguientes
    nivel = 3  # ganador de octavos llega a nivel 3 (cuartos), etc.
    actual = ganadores
    while len(actual) > 1:
        siguiente = []
        for k in range(0, len(actual), 2):
            w = jugar_partido(actual[k], actual[k+1], params, elos, rng, neutral=True)
            ronda[w] = nivel
            siguiente.append(w)
        actual = siguiente
        nivel += 1
    # Cuando queda 1, ese es el campeón. El último 'nivel' asignado fue 6.
    return ronda


def simular(N=10000, seed=0):
    params, elos = cargar_modelo()
    rng = np.random.default_rng(seed)
    equipos = [t for par in DIECISEISAVOS for t in par]

    avanza = defaultdict(int)   # gana dieciseisavos (llega a octavos, nivel>=2)
    final = defaultdict(int)    # llega a la final (nivel>=5)
    campeon = defaultdict(int)  # gana (nivel>=6)
    suma_ronda = defaultdict(int)

    for _ in range(N):
        r = simular_torneo(params, elos, rng)
        for t in equipos:
            nivel = r.get(t, 1)
            suma_ronda[t] += nivel
            if nivel >= 2: avanza[t] += 1
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
    print("Simulando desde dieciseisavos (10,000 torneos)...")
    df = simular(N=10000, seed=42)
    pd.set_option("display.max_rows", 40)
    print("\n=== PROBABILIDADES (32 equipos en eliminatorias) ===\n")
    print(df.to_string(index=False))
    df.to_csv("simulacion_mundial.csv", index=False)
    print("\nGuardado en simulacion_mundial.csv")
