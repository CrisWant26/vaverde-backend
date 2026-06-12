"""
Simulación Monte Carlo del Mundial 2026
========================================
Simula el torneo completo N veces usando el modelo Poisson+Elo calibrado.

Cada simulación:
  1. Juega los 72 partidos de grupos (marcadores muestreados de Poisson).
  2. Aplica reglas FIFA: puntos, luego diferencia de goles, luego goles a favor.
  3. Clasifica 1ro y 2do de cada grupo + los 8 mejores terceros = 32 equipos.
  4. Arma el bracket de la Ronda de 32 y lo juega (empate -> penales).
  5. Registra hasta dónde llegó cada equipo y quién ganó.

Sobre N torneos, las frecuencias dan las probabilidades.

NOTA sobre el formato 48 equipos: es nuevo (primer Mundial así), y el
emparejamiento exacto de los 8 mejores terceros con los grupos es complejo
(FIFA tiene una tabla de asignación según QUÉ grupos aportan los terceros).
Aquí usamos un emparejamiento simplificado por siembra (mejor vs peor) que
captura bien las probabilidades sin reproducir la tabla oficial exacta.
Esto es una aproximación razonable y está documentada como tal.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pickle
from collections import defaultdict
from modelo_elo import lambdas

# 12 grupos oficiales (sorteo 5-dic-2025). Nombres EXACTOS del dataset.
GRUPOS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Uzbekistan", "Colombia", "DR Congo"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}
ANFITRIONES = {"Mexico", "United States", "Canada"}


def cargar_modelo():
    with open("/mnt/user-data/outputs/modelo_elo.pkl", "rb") as f:
        M = pickle.load(f)
    return M["params"], M["elos"]


def muestrear_marcador(params, elos, h, a, neutral, rng):
    """Muestrea un marcador (goles_h, goles_a) de las distribuciones Poisson."""
    lh, la = lambdas(params, elos, h, a, neutral=neutral)
    return rng.poisson(lh), rng.poisson(la)


def jugar_grupo(equipos, params, elos, rng):
    """Round-robin. Devuelve equipos ordenados por reglas FIFA con sus stats."""
    pts = defaultdict(int); gf = defaultdict(int); gc = defaultdict(int)
    for i in range(len(equipos)):
        for j in range(i+1, len(equipos)):
            h, a = equipos[i], equipos[j]
            # anfitrión juega "en casa" si está en el partido
            neutral = not (h in ANFITRIONES)
            gh, ga = muestrear_marcador(params, elos, h, a, neutral, rng)
            gf[h] += gh; gc[h] += ga; gf[a] += ga; gc[a] += gh
            if gh > ga: pts[h] += 3
            elif ga > gh: pts[a] += 3
            else: pts[h] += 1; pts[a] += 1
    tabla = sorted(equipos, key=lambda t: (pts[t], gf[t]-gc[t], gf[t]) ,
                   reverse=True)
    stats = {t: (pts[t], gf[t]-gc[t], gf[t]) for t in equipos}
    return tabla, stats


def jugar_eliminatoria(h, a, params, elos, rng):
    """Partido a muerte: empate -> penales (~50/50 con leve sesgo al mejor Elo)."""
    gh, ga = muestrear_marcador(params, elos, h, a, neutral=True, rng=rng)
    if gh > ga: return h
    if ga > gh: return a
    # penales: prob proporcional a Elo (leve), basado en shootouts históricos ~50/50
    eh, ea = elos.get(h, 1500), elos.get(a, 1500)
    p_h = 1 / (1 + 10 ** (-(eh - ea) / 800))  # /800 = efecto suave
    return h if rng.random() < p_h else a


def simular_torneo(params, elos, rng):
    """Una simulación completa. Devuelve dict equipo -> ronda alcanzada."""
    ronda = {}  # equipo -> ronda máxima (0=grupos,1=R32,2=R16,3=QF,4=SF,5=Final,6=Campeón)
    primeros, segundos, terceros = [], [], []
    tercero_stats = []

    for g, equipos in GRUPOS.items():
        for t in equipos:
            ronda[t] = 0
        tabla, stats = jugar_grupo(equipos, params, elos, rng)
        primeros.append(tabla[0]); segundos.append(tabla[1])
        terceros.append(tabla[2])
        tercero_stats.append((tabla[2], stats[tabla[2]]))

    # 8 mejores terceros por (pts, dg, gf)
    terceros_ord = sorted(tercero_stats, key=lambda x: x[1], reverse=True)
    mejores_terceros = [t for t, _ in terceros_ord[:8]]

    # 32 clasificados
    clasificados = primeros + segundos + mejores_terceros
    for t in clasificados:
        ronda[t] = 1  # llegó al menos a R32

    # Bracket: sembrar por Elo (mejor vs peor) — aproximación al cruce real
    clasificados_ord = sorted(clasificados, key=lambda t: elos.get(t, 1500), reverse=True)
    # emparejar 1-32, 2-31, ... estilo siembra
    bracket = []
    n = len(clasificados_ord)
    for i in range(n // 2):
        bracket.append((clasificados_ord[i], clasificados_ord[n-1-i]))

    rondas_nombres = [2, 3, 4, 5, 6]  # R16, QF, SF, Final, Campeón
    for nivel in rondas_nombres:
        ganadores = []
        for h, a in bracket:
            w = jugar_eliminatoria(h, a, params, elos, rng)
            ganadores.append(w)
            ronda[w] = nivel
        if len(ganadores) == 1:
            break
        # re-emparejar ganadores en orden
        bracket = [(ganadores[i], ganadores[i+1]) for i in range(0, len(ganadores), 2)]

    return ronda


def simular(N=10000, seed=0):
    params, elos = cargar_modelo()
    rng = np.random.default_rng(seed)
    todos = [t for eq in GRUPOS.values() for t in eq]
    avanza = defaultdict(int)   # pasa de grupos (R32)
    campeon = defaultdict(int)
    final = defaultdict(int)
    suma_ronda = defaultdict(int)

    for _ in range(N):
        r = simular_torneo(params, elos, rng)
        for t in todos:
            nivel = r.get(t, 0)
            suma_ronda[t] += nivel
            if nivel >= 1: avanza[t] += 1
            if nivel >= 5: final[t] += 1
            if nivel >= 6: campeon[t] += 1

    filas = []
    for t in todos:
        filas.append({
            "equipo": t,
            "elo": round(elos.get(t, 1500)),
            "p_avanzar_%": round(100*avanza[t]/N, 1),
            "p_final_%": round(100*final[t]/N, 1),
            "p_campeon_%": round(100*campeon[t]/N, 1),
            "ronda_media": round(suma_ronda[t]/N, 2),
        })
    import pandas as pd
    return pd.DataFrame(filas).sort_values("p_campeon_%", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    import pandas as pd
    print("Simulando el Mundial 2026 (10,000 torneos)...")
    df = simular(N=10000, seed=42)
    pd.set_option("display.max_rows", 60)
    print("\n=== PROBABILIDADES (ordenado por prob. de ser campeón) ===\n")
    print(df.to_string(index=False))
    df.to_csv("/mnt/user-data/outputs/simulacion_mundial.csv", index=False)
    print("\nGuardado en simulacion_mundial.csv")
