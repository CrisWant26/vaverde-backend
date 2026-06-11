"""
Motor de predicción de partidos de selecciones — Modelo Poisson / Dixon-Coles
================================================================================
Fundamento teórico (lo que pidió tu profe):

  - PROCESO DE POISSON: los goles de cada equipo se modelan como conteos Poisson
    con tasa lambda. Justificado por la LEY DE LOS EVENTOS RAROS (muchos ataques,
    cada uno con prob. baja de gol -> el conteo converge a Poisson).

  - REGRESIÓN MULTIVARIADA (GLM Poisson): estimamos parámetros de ataque y defensa
    por selección + ventaja de local resolviendo:
        log(lambda) = mu + ataque[local] - defensa[visitante] + ventaja_local
    (para el visitante: log(lambda) = mu + ataque[visitante] - defensa[local])

  - CORRECCIÓN DIXON-COLES (rho): ajusta la dependencia en marcadores bajos
    (0-0, 1-0, 0-1, 1-1) que el Poisson puro subestima/sobreestima.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


def cargar_datos(path="results.csv", desde_anio=2018, ponderar_recientes=True):
    """Carga y filtra partidos. Pondera partidos recientes y competitivos."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    # Solo partidos ya jugados (con marcador)
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    # Ventana temporal: forma reciente importa más que historia antigua
    df = df[df["date"].dt.year >= desde_anio].reset_index(drop=True)

    # Peso por recencia (decaimiento exponencial) y por importancia del torneo
    max_date = df["date"].max()
    half_life_days = 365 * 2  # vida media de 2 años
    age_days = (max_date - df["date"]).dt.days
    df["w_time"] = 0.5 ** (age_days / half_life_days) if ponderar_recientes else 1.0

    # Amistosos pesan menos que competitivos
    df["w_comp"] = np.where(df["tournament"].str.contains("Friendly", case=False, na=False), 0.5, 1.0)
    df["weight"] = df["w_time"] * df["w_comp"]
    return df


def ajustar_dixon_coles(df):
    """Estima parámetros ataque/defensa por equipo, ventaja de local y rho (DC)."""
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    # Vector de parámetros: [ataque(n), defensa(n), home_adv, rho]
    # Restricción: suma de ataques = 0 (identificabilidad)
    init = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [-0.05]])

    h_idx = df["home_team"].map(idx).values
    a_idx = df["away_team"].map(idx).values
    hg = df["home_score"].values
    ag = df["away_score"].values
    w = df["weight"].values

    def dc_adjust(hg, ag, lh, la, rho):
        """Término de corrección Dixon-Coles para marcadores bajos."""
        tau = np.ones_like(lh, dtype=float)
        m00 = (hg == 0) & (ag == 0)
        m10 = (hg == 1) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m11 = (hg == 1) & (ag == 1)
        tau[m00] = 1 - lh[m00] * la[m00] * rho
        tau[m10] = 1 + la[m10] * rho
        tau[m01] = 1 + lh[m01] * rho
        tau[m11] = 1 - rho
        return np.maximum(tau, 1e-10)

    def neg_log_lik(params):
        atk = params[:n]
        dfn = params[n:2*n]
        home_adv = params[2*n]
        rho = params[2*n + 1]
        # centrar ataques para identificabilidad
        atk = atk - atk.mean()
        log_lh = atk[h_idx] - dfn[a_idx] + home_adv
        log_la = atk[a_idx] - dfn[h_idx]
        lh = np.exp(log_lh)
        la = np.exp(log_la)
        ll = (poisson.logpmf(hg, lh) + poisson.logpmf(ag, la))
        tau = dc_adjust(hg, ag, lh, la, rho)
        ll = ll + np.log(tau)
        return -np.sum(w * ll)

    res = minimize(neg_log_lik, init, method="L-BFGS-B",
                   options={"maxiter": 500})
    p = res.x
    atk = p[:n]; atk = atk - atk.mean()
    params = {
        "teams": teams, "idx": idx,
        "attack": dict(zip(teams, atk)),
        "defense": dict(zip(teams, p[n:2*n])),
        "home_adv": p[2*n], "rho": p[2*n+1],
    }
    return params


def matriz_marcadores(params, home, away, neutral=True, max_goals=8):
    """Devuelve matriz de probabilidades P(home=i, away=j)."""
    atk, dfn = params["attack"], params["defense"]
    ha = 0.0 if neutral else params["home_adv"]  # Mundial = casi todo neutral
    log_lh = atk[home] - dfn[away] + ha
    log_la = atk[away] - dfn[home]
    lh, la = np.exp(log_lh), np.exp(log_la)

    i = np.arange(max_goals + 1)
    ph = poisson.pmf(i, lh)
    pa = poisson.pmf(i, la)
    M = np.outer(ph, pa)

    # Corrección Dixon-Coles en celdas bajas
    rho = params["rho"]
    M[0, 0] *= 1 - lh * la * rho
    M[1, 0] *= 1 + la * rho
    M[0, 1] *= 1 + lh * rho
    M[1, 1] *= 1 - rho
    M /= M.sum()  # renormalizar
    return M, lh, la


def predecir(params, home, away, neutral=True):
    """Probabilidades de los mercados principales."""
    M, lh, la = matriz_marcadores(params, home, away, neutral)
    p_home = np.tril(M, -1).sum()   # home > away
    p_draw = np.trace(M)
    p_away = np.triu(M, 1).sum()
    # Over/Under 2.5
    total = np.add.outer(np.arange(M.shape[0]), np.arange(M.shape[1]))
    p_over = M[total > 2.5].sum()
    p_btts = 1 - M[0, :].sum() - M[:, 0].sum() + M[0, 0]  # ambos anotan
    # marcador más probable
    i, j = np.unravel_index(M.argmax(), M.shape)
    return {
        "xg_home": round(lh, 2), "xg_away": round(la, 2),
        "P(local)": round(p_home, 3), "P(empate)": round(p_draw, 3),
        "P(visita)": round(p_away, 3),
        "P(over2.5)": round(p_over, 3), "P(ambos anotan)": round(p_btts, 3),
        "marcador_probable": f"{i}-{j}",
    }


if __name__ == "__main__":
    print("Cargando datos...")
    df = cargar_datos(desde_anio=2018)
    print(f"Partidos de entrenamiento: {len(df):,}")
    print("Ajustando modelo Dixon-Coles (regresión Poisson)...")
    params = ajustar_dixon_coles(df)
    print(f"Ventaja de local estimada: {params['home_adv']:.3f}")
    print(f"Rho (Dixon-Coles): {params['rho']:.3f}\n")

    # Top 10 ataques
    rank = sorted(params["attack"].items(), key=lambda x: -x[1])[:10]
    print("Top 10 fuerza ofensiva:")
    for t, v in rank:
        print(f"  {t:20s} ataque={v:+.2f}  defensa={params['defense'][t]:+.2f}")

    print("\n--- Predicciones de prueba (sede neutral) ---")
    for h, a in [("Mexico", "United States"), ("Argentina", "France"),
                 ("Spain", "Brazil"), ("Mexico", "Brazil")]:
        try:
            r = predecir(params, h, a)
            print(f"\n{h} vs {a}")
            for k, v in r.items():
                print(f"   {k}: {v}")
        except KeyError as e:
            print(f"  (sin datos para {e})")
