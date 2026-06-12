"""
Modelo híbrido Poisson + Elo
=============================
Idea: en vez de estimar la fuerza de cada selección SOLO con un parámetro
ataque/defensa de regresión (que pesa todos los partidos igual y se confunde
con muestras chicas), incorporamos la diferencia de Elo —que ya condensa la
forma reciente y la calidad de rivales— como variable explicativa de los goles.

Modelo de goles esperados (regresión Poisson con Elo):
    log(lambda_local)  = c + b_atk*atk_l - b_def*def_v + b_elo*elo_diff + ventaja
    log(lambda_visita) = c + b_atk*atk_v - b_def*def_l - b_elo*elo_diff

Ajustamos los coeficientes por máxima verosimilitud sobre el histórico.
Luego, como antes, calibramos las probabilidades 1X2 (Platt).
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression

from elo import calcular_elo


def cargar_con_elo(desde_anio=2010):
    df = pd.read_csv("/mnt/user-data/uploads/results.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    # Elo se calcula con TODA la historia (para que los ratings sean buenos),
    # pero luego entrenamos/evaluamos solo desde desde_anio.
    df_elo, elos = calcular_elo(df)
    df_elo = df_elo[df_elo["date"].dt.year >= desde_anio].reset_index(drop=True)
    return df_elo, elos


def ajustar_modelo_elo(df):
    """Estima ataque/defensa por equipo + coeficiente de Elo + ventaja local."""
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    h_idx = df["home_team"].map(idx).values
    a_idx = df["away_team"].map(idx).values
    hg = df["home_score"].values
    ag = df["away_score"].values
    elo_diff = (df["elo_diff"].values) / 400.0  # escalar
    neutral = df["neutral"].isin([True, "TRUE", "True"]).values

    # params: [atk(n), def(n), home_adv, b_elo, c]
    init = np.concatenate([np.zeros(n), np.zeros(n), [0.2], [0.3], [0.0]])

    def nll(p):
        atk = p[:n]; atk = atk - atk.mean()
        dfn = p[n:2*n]
        ha = p[2*n]; b_elo = p[2*n+1]; c = p[2*n+2]
        adv = np.where(neutral, 0.0, ha)
        log_lh = c + atk[h_idx] - dfn[a_idx] + b_elo*elo_diff + adv
        log_la = c + atk[a_idx] - dfn[h_idx] - b_elo*elo_diff
        lh = np.exp(np.clip(log_lh, -2, 3))
        la = np.exp(np.clip(log_la, -2, 3))
        return -(poisson.logpmf(hg, lh) + poisson.logpmf(ag, la)).sum()

    res = minimize(nll, init, method="L-BFGS-B", options={"maxiter": 400})
    p = res.x
    atk = p[:n]; atk = atk - atk.mean()
    return {
        "teams": teams, "idx": idx,
        "attack": dict(zip(teams, atk)),
        "defense": dict(zip(teams, p[n:2*n])),
        "home_adv": p[2*n], "b_elo": p[2*n+1], "c": p[2*n+2],
    }


def lambdas(params, elos, home, away, neutral=True):
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


def prob_1x2(params, elos, home, away, neutral=True, max_goals=10):
    lh, la = lambdas(params, elos, home, away, neutral)
    i = np.arange(max_goals+1)
    M = np.outer(poisson.pmf(i, lh), poisson.pmf(i, la))
    M /= M.sum()
    return np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()


def ajustar_calibrador(df, params, elos):
    feats, ys = [], []
    sub = df.sample(min(4000, len(df)), random_state=0)
    eps = 1e-6
    lo = lambda p: np.log((p+eps)/(1-p+eps))
    for r in sub.itertuples():
        neutral = r.neutral in (True, "TRUE", "True")
        ph, pd_, pa = prob_1x2(params, elos, r.home_team, r.away_team, neutral)
        feats.append([lo(ph), lo(pd_), lo(pa)])
        ys.append(0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2))
    clf = LogisticRegression(max_iter=1000)
    clf.fit(np.array(feats), np.array(ys))
    return clf


def predecir(params, calib, elos, home, away, neutral=True):
    ph, pd_, pa = prob_1x2(params, elos, home, away, neutral)
    eps = 1e-6
    lo = lambda p: np.log((p+eps)/(1-p+eps))
    cal = calib.predict_proba(np.array([[lo(ph), lo(pd_), lo(pa)]]))[0]
    return {"cruda": (round(ph,3), round(pd_,3), round(pa,3)),
            "P(local)": round(cal[0],3), "P(empate)": round(cal[1],3), "P(visita)": round(cal[2],3)}


def brier_local(df, params, elos, calib):
    rows = []
    sub = df.sample(min(3000, len(df)), random_state=1)
    for r in sub.itertuples():
        neutral = r.neutral in (True, "TRUE", "True")
        pred = predecir(params, calib, elos, r.home_team, r.away_team, neutral)
        rows.append((pred["P(local)"], 1 if r.home_score > r.away_score else 0))
    d = pd.DataFrame(rows, columns=["p", "y"])
    return ((d["p"]-d["y"])**2).mean()


if __name__ == "__main__":
    print("Cargando datos y calculando Elo...")
    df, elos = cargar_con_elo(desde_anio=2014)
    df = df[df["date"] < pd.Timestamp("2026-06-01")]
    counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
    solidos = set(counts[counts >= 20].index)
    df = df[df["home_team"].isin(solidos) & df["away_team"].isin(solidos)].reset_index(drop=True)

    corte = df["date"].quantile(0.8)
    tr = df[df["date"] <= corte]; ev = df[df["date"] > corte]
    print(f"Train: {len(tr):,}  Eval: {len(ev):,}")

    print("Ajustando modelo Poisson+Elo...")
    params = ajustar_modelo_elo(tr)
    print(f"  Coeficiente de Elo (b_elo): {params['b_elo']:.3f}  (>0 = Elo ayuda a predecir goles)")
    print(f"  Ventaja de local: {params['home_adv']:.3f}")

    calib = ajustar_calibrador(tr, params, elos)
    b = brier_local(ev, params, elos, calib)
    print(f"\n  Brier score (Poisson+Elo calibrado): {b:.4f}")
    print(f"  (recordatorio: Poisson solo calibrado daba 0.1788)")

    # Reentrenar con todo para uso real y guardar
    params_full = ajustar_modelo_elo(df)
    calib_full = ajustar_calibrador(df, params_full, elos)
    import pickle
    with open("/mnt/user-data/outputs/modelo_elo.pkl", "wb") as f:
        pickle.dump({"params": params_full, "calibrador": calib_full, "elos": elos}, f)

    print("\n--- Comparación contra el mercado (partidos de la captura) ---")
    mercado = {
        ("Mexico", "South Africa", False): (1.44, 4.60, 9.50),
        ("South Korea", "Czech Republic", True): (2.68, 3.20, 2.95),
    }
    for (h, a, neu), (oh, od, oa) in mercado.items():
        pred = predecir(params_full, calib_full, elos, h, a, neutral=neu)
        s = 1/oh + 1/od + 1/oa
        q = [(1/oh)/s, (1/od)/s, (1/oa)/s]
        print(f"\n=== {h} vs {a} ===")
        print(f"  {'':8s}  ELO+POIS  MERCADO   EV")
        for lab, pm, qq, o in [("Local", pred['P(local)'], q[0], oh),
                                ("Empate", pred['P(empate)'], q[1], od),
                                ("Visita", pred['P(visita)'], q[2], oa)]:
            ev_ = pm*o - 1
            flag = "  <-- valor" if ev_ > 0.03 else ""
            print(f"  {lab:8s}  {pm*100:6.1f}%   {qq*100:5.1f}%   {ev_:+.3f}{flag}")
