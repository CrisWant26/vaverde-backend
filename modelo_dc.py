"""
modelo_dc.py — Poisson + Elo + corrección Dixon-Coles
======================================================
Versión experimental PARALELA a modelo_elo.py (que NO se toca).

Qué agrega respecto al modelo actual:
El Poisson independiente asume que los goles del local y del visitante
no se afectan entre sí. En la realidad eso falla en los marcadores bajos:
los 0-0, 1-0, 0-1 y 1-1 ocurren MÁS seguido de lo que predice el Poisson
(partidos trabados donde los equipos ajustan su comportamiento).

Dixon-Coles corrige exactamente esas 4 celdas con un parámetro rho:
    tau(0,0) = 1 - lh*la*rho
    tau(0,1) = 1 + lh*rho
    tau(1,0) = 1 + la*rho
    tau(1,1) = 1 - rho
    (todas las demás celdas: tau = 1, sin cambio)

rho se ESTIMA por máxima verosimilitud junto con el resto de parámetros,
no se fija a mano. Si para una liga sale cerca de 0, esa liga no
necesitaba la corrección — y eso también es información.

API compatible con modelo_elo.py: ajustar_modelo_dc, lambdas_dc,
prob_1x2_dc, ajustar_calibrador_dc. Así el comparador puede correr
ambos modelos con el mismo código alrededor.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression


# ------------------------------------------------------------
# Corrección Dixon-Coles
# ------------------------------------------------------------
def tau_dc(hg, ag, lh, la, rho):
    """Factor de corrección para las 4 celdas bajas. Vectorizado.

    Devuelve 1.0 para cualquier marcador que no sea 0-0, 0-1, 1-0 o 1-1.
    Se acota por abajo para que el logaritmo nunca explote.
    """
    t = np.ones_like(lh, dtype=float)

    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)

    t[m00] = 1.0 - lh[m00] * la[m00] * rho
    t[m01] = 1.0 + lh[m01] * rho
    t[m10] = 1.0 + la[m10] * rho
    t[m11] = 1.0 - rho

    return np.clip(t, 1e-9, None)


def ajustar_modelo_dc(df):
    """Igual que ajustar_modelo_elo, pero estimando además rho (Dixon-Coles).

    Devuelve el mismo dict de params + la clave 'rho'.
    """
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    h_idx = df["home_team"].map(idx).values
    a_idx = df["away_team"].map(idx).values
    hg = df["home_score"].values
    ag = df["away_score"].values
    elo_diff = (df["elo_diff"].values) / 400.0
    neutral = df["neutral"].isin([True, "TRUE", "True"]).values

    # params: [atk(n), def(n), home_adv, b_elo, c, rho]
    init = np.concatenate([np.zeros(n), np.zeros(n), [0.2], [0.3], [0.0], [0.0]])

    def nll(p):
        atk = p[:n]; atk = atk - atk.mean()
        dfn = p[n:2*n]
        ha = p[2*n]; b_elo = p[2*n+1]; c = p[2*n+2]; rho = p[2*n+3]
        adv = np.where(neutral, 0.0, ha)
        log_lh = c + atk[h_idx] - dfn[a_idx] + b_elo*elo_diff + adv
        log_la = c + atk[a_idx] - dfn[h_idx] - b_elo*elo_diff
        lh = np.exp(np.clip(log_lh, -2, 3))
        la = np.exp(np.clip(log_la, -2, 3))
        base = poisson.logpmf(hg, lh) + poisson.logpmf(ag, la)
        correccion = np.log(tau_dc(hg, ag, lh, la, rho))
        return -(base + correccion).sum()

    # rho acotado: fuera de este rango tau se vuelve negativo con lambdas altas
    bounds = [(None, None)] * (2*n + 3) + [(-0.25, 0.25)]
    res = minimize(nll, init, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 600})
    p = res.x
    atk = p[:n]; atk = atk - atk.mean()
    return {
        "teams": teams, "idx": idx,
        "attack": dict(zip(teams, atk)),
        "defense": dict(zip(teams, p[n:2*n])),
        "home_adv": p[2*n], "b_elo": p[2*n+1], "c": p[2*n+2],
        "rho": p[2*n+3],
    }


def lambdas_dc(params, elos, home, away, neutral=True):
    """Idéntica a modelo_elo.lambdas (rho no afecta las medias, solo la matriz)."""
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


def matriz_dc(lh, la, rho, max_goals=10):
    """Matriz de marcadores con la corrección aplicada a las 4 celdas bajas."""
    i = np.arange(max_goals + 1)
    M = np.outer(poisson.pmf(i, lh), poisson.pmf(i, la))
    M[0, 0] *= max(1.0 - lh * la * rho, 1e-9)
    M[0, 1] *= max(1.0 + lh * rho, 1e-9)
    M[1, 0] *= max(1.0 + la * rho, 1e-9)
    M[1, 1] *= max(1.0 - rho, 1e-9)
    return M / M.sum()


def prob_1x2_dc(params, elos, home, away, neutral=True, max_goals=10):
    lh, la = lambdas_dc(params, elos, home, away, neutral)
    M = matriz_dc(lh, la, params.get("rho", 0.0), max_goals)
    return float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum())


def ajustar_calibrador_dc(df, params, elos):
    """Platt sobre los logits del 1X2 corregido (misma receta que el actual)."""
    feats, ys = [], []
    sub = df.sample(min(4000, len(df)), random_state=0)
    eps = 1e-6
    lo = lambda p: np.log((p + eps) / (1 - p + eps))
    for r in sub.itertuples():
        neutral = r.neutral in (True, "TRUE", "True")
        ph, pd_, pa = prob_1x2_dc(params, elos, r.home_team, r.away_team, neutral)
        feats.append([lo(ph), lo(pd_), lo(pa)])
        ys.append(0 if r.home_score > r.away_score else
                  (1 if r.home_score == r.away_score else 2))
    clf = LogisticRegression(max_iter=1000)
    clf.fit(np.array(feats), np.array(ys))
    return clf
