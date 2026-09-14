"""
elo_unificado.py — Un solo Elo para toda Europa
================================================
PROBLEMA QUE RESUELVE:
Hoy cada liga tiene su Elo aislado: todos arrancan en 1500 y suben o
bajan jugando solo entre ellos. El 1750 del Madrid y el 1750 del Ajax
miden cosas distintas — son universos que nunca se tocaron.

SOLUCIÓN:
Calcular un Elo sobre TODOS los partidos juntos (7 ligas domésticas +
competencias UEFA), en orden cronológico. Los partidos europeos actúan
como "puentes" que calibran la relación entre ligas: cada Madrid-Ajax
dice algo sobre cuánto vale LaLiga contra la Eredivisie.

LIMITACIÓN CONOCIDA (medida, no supuesta):
El dataset UEFA solo tiene Champions de 2011 a 2020, y Europa/Conference
desde 2020-21. Eso significa que los puentes conectan sobre todo a la
ÉLITE de cada liga. La relación Madrid-Bayern quedará bien calibrada;
la de un equipo de media tabla contra otro de media tabla, mucho menos.

QUÉ EVALÚA:
  1. ¿El Elo unificado empeora las predicciones DOMÉSTICAS?
     (no debería; si las mejora, mejor)
  2. ¿Predice partidos UEFA mejor que un baseline?
     (si no, el proyecto no sirve para Champions)

Uso:  python elo_unificado.py
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import poisson

from elo import factor_goles
from modelo_elo import ajustar_modelo_elo, ajustar_calibrador, lambdas
from adaptador_main import cargar_liga_main

MAPEO = "leagues/mapeo_uefa.csv"
UEFA_CSV = os.path.expanduser("~/uefa_parseado.csv")

LIGAS = {
    "laliga": ("SP1", "ESP", "LaLiga"),
    "premier": ("E0", "ENG", "Premier League"),
    "seriea": ("I1", "ITA", "Serie A"),
    "bundesliga": ("D1", "GER", "Bundesliga"),
    "ligue1": ("F1", "FRA", "Ligue 1"),
    "eredivisie": ("N1", "NED", "Eredivisie"),
    "primeira": ("P1", "POR", "Primeira Liga"),
}

K_LIGA = 30       # mismo K que usa el pipeline actual
K_UEFA = 40       # los partidos europeos pesan un poco más
VENTAJA_LOCAL = 65


# ------------------------------------------------------------
# Construcción del dataset unificado
# ------------------------------------------------------------
def cargar_mapeo():
    """openfootball -> football_data (el sentido que necesitamos)."""
    m = pd.read_csv(MAPEO).fillna("")
    m = m[m["openfootball"] != ""]
    return dict(zip(m["openfootball"], m["football_data"]))


def cargar_dataset():
    frames = []

    # Ligas domésticas
    for clave, (cod, pais, nombre) in LIGAS.items():
        df = cargar_liga_main(cod, nombre=nombre, verbose=False)
        df = df[["date", "home_team", "away_team", "home_score", "away_score"]].copy()
        df["competicion"] = clave
        df["es_uefa"] = False
        df["neutral"] = False
        frames.append(df)
        print(f"  {nombre:<16} {len(df):>6,} partidos")

    # Competencias UEFA
    if not os.path.exists(UEFA_CSV):
        print(f"\nERROR: falta {UEFA_CSV}")
        print("Corre: python parser_openfootball_v2.py ~/champions-league")
        sys.exit(1)

    mapeo = cargar_mapeo()
    u = pd.read_csv(UEFA_CSV)
    u["date"] = pd.to_datetime(u["date"])

    # Traduce los nombres que tengan equivalente; los demás se quedan con
    # su nombre de openfootball (equipos de ligas que no cubrimos, que
    # igual aportan información sobre el nivel de su rival).
    u["home_team"] = u["home_team"].map(lambda x: mapeo.get(x, f"UEFA:{x}"))
    u["away_team"] = u["away_team"].map(lambda x: mapeo.get(x, f"UEFA:{x}"))

    uefa = u[["date", "home_team", "away_team", "home_score", "away_score"]].copy()
    uefa["competicion"] = "uefa"
    uefa["es_uefa"] = True
    uefa["neutral"] = False
    frames.append(uefa)

    mapeados = sum(1 for x in pd.concat([u["home_team"], u["away_team"]])
                   if not str(x).startswith("UEFA:"))
    total = len(u) * 2
    print(f"  {'UEFA':<16} {len(uefa):>6,} partidos "
          f"({mapeados/total:.0%} de equipos mapeados a ligas conocidas)")

    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("date").reset_index(drop=True)


# ------------------------------------------------------------
# Elo unificado
# ------------------------------------------------------------
def calcular_elo_unificado(df, elo_inicial=1500):
    """Elo cronológico sobre todos los partidos, con K distinto para UEFA."""
    df = df.sort_values("date").reset_index(drop=True)
    elos = {}
    eh_list = np.zeros(len(df))
    ea_list = np.zeros(len(df))

    for i, r in enumerate(df.itertuples()):
        h, a = r.home_team, r.away_team
        eh = elos.get(h, elo_inicial)
        ea = elos.get(a, elo_inicial)
        eh_list[i] = eh
        ea_list[i] = ea

        va = 0 if r.neutral else VENTAJA_LOCAL
        we_h = 1 / (1 + 10 ** (-(eh - ea + va) / 400))
        hs, as_ = r.home_score, r.away_score
        w_h = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)
        K = K_UEFA if r.es_uefa else K_LIGA
        cambio = K * factor_goles(hs - as_) * (w_h - we_h)
        elos[h] = eh + cambio
        elos[a] = ea - cambio

    df = df.copy()
    df["elo_home"] = eh_list
    df["elo_away"] = ea_list
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    return df, elos


def calcular_elo_por_liga(df):
    """Elo aislado por liga (el método actual), para comparar."""
    partes = []
    elos_todos = {}
    for comp in df["competicion"].unique():
        if comp == "uefa":
            continue
        sub = df[df["competicion"] == comp].sort_values("date").reset_index(drop=True)
        elos = {}
        eh, ea = np.zeros(len(sub)), np.zeros(len(sub))
        for i, r in enumerate(sub.itertuples()):
            h, a = r.home_team, r.away_team
            e_h = elos.get(h, 1500); e_a = elos.get(a, 1500)
            eh[i], ea[i] = e_h, e_a
            we = 1 / (1 + 10 ** (-(e_h - e_a + VENTAJA_LOCAL) / 400))
            w = 1.0 if r.home_score > r.away_score else (0.0 if r.home_score < r.away_score else 0.5)
            c = K_LIGA * factor_goles(r.home_score - r.away_score) * (w - we)
            elos[h] = e_h + c; elos[a] = e_a - c
        sub = sub.copy()
        sub["elo_home"] = eh; sub["elo_away"] = ea
        sub["elo_diff"] = sub["elo_home"] - sub["elo_away"]
        partes.append(sub)
        elos_todos.update({f"{comp}:{k}": v for k, v in elos.items()})
    return pd.concat(partes, ignore_index=True), elos_todos


# ------------------------------------------------------------
# Evaluación
# ------------------------------------------------------------
def brier(ev, params, calib, max_goals=10, n_max=3000):
    eps = 1e-6
    logit = lambda p: np.log((p + eps) / (1 - p + eps))
    sub = ev.sample(min(n_max, len(ev)), random_state=1)
    i = np.arange(max_goals + 1)
    errores = []
    for r in sub.itertuples():
        elos_row = {r.home_team: r.elo_home, r.away_team: r.elo_away}
        lh, la = lambdas(params, elos_row, r.home_team, r.away_team,
                         neutral=bool(r.neutral))
        M = np.outer(poisson.pmf(i, lh), poisson.pmf(i, la)); M /= M.sum()
        ph = float(np.tril(M, -1).sum())
        pdw = float(np.trace(M))
        pa = float(np.triu(M, 1).sum())
        cal = calib.predict_proba(np.array([[logit(ph), logit(pdw), logit(pa)]]))[0]
        y = np.zeros(3)
        y[0 if r.home_score > r.away_score else
          (1 if r.home_score == r.away_score else 2)] = 1.0
        errores.append(float(((cal - y) ** 2).sum()))
    return float(np.mean(errores))


def brier_naive(ev):
    ph = (ev["home_score"] > ev["away_score"]).mean()
    pdw = (ev["home_score"] == ev["away_score"]).mean()
    base = np.array([ph, pdw, 1 - ph - pdw])
    err = 0.0
    for r in ev.itertuples():
        y = np.zeros(3)
        y[0 if r.home_score > r.away_score else
          (1 if r.home_score == r.away_score else 2)] = 1.0
        err += float(((base - y) ** 2).sum())
    return err / len(ev)


def main():
    print("Construyendo dataset unificado...\n")
    df = cargar_dataset()
    print(f"\nTOTAL: {len(df):,} partidos "
          f"({df['date'].min().date()} → {df['date'].max().date()})")

    df = df[df["date"].dt.year >= 2013].reset_index(drop=True)

    print("\nCalculando Elo unificado...")
    df_uni, elos_uni = calcular_elo_unificado(df)

    print("Calculando Elo por liga (método actual)...")
    df_liga, _ = calcular_elo_por_liga(df)

    # --- Evaluación 1: partidos domésticos ---
    print(f"\n{'='*66}")
    print("  1) ¿Empeora las predicciones DOMÉSTICAS?")
    print(f"{'='*66}")
    print(f"{'Liga':<16}{'por liga':>11}{'unificado':>12}{'mejora':>10}")

    mejoras = []
    for clave, (_, _, nombre) in LIGAS.items():
        res = {}
        for etiqueta, fuente in (("liga", df_liga), ("uni", df_uni)):
            sub = fuente[fuente["competicion"] == clave].reset_index(drop=True)
            if len(sub) < 800:
                res[etiqueta] = None
                continue
            counts = pd.concat([sub["home_team"], sub["away_team"]]).value_counts()
            solidos = set(counts[counts >= 30].index)
            sub = sub[sub["home_team"].isin(solidos) &
                      sub["away_team"].isin(solidos)].reset_index(drop=True)
            corte = sub["date"].quantile(0.8)
            tr = sub[sub["date"] <= corte].reset_index(drop=True)
            ev = sub[sub["date"] > corte].reset_index(drop=True)
            elos_tr = dict(zip(tr["home_team"], tr["elo_home"]))
            p = ajustar_modelo_elo(tr)
            c = ajustar_calibrador(tr, p, elos_tr)
            res[etiqueta] = brier(ev, p, c)

        if res.get("liga") and res.get("uni"):
            d = res["liga"] - res["uni"]
            mejoras.append(d)
            print(f"{nombre:<16}{res['liga']:>11.4f}{res['uni']:>12.4f}{d:>+10.4f}")

    if mejoras:
        media = np.mean(mejoras)
        print(f"\n  Mejora promedio: {media:+.4f}  "
              f"(gana en {sum(1 for m in mejoras if m>0)}/{len(mejoras)})")
        if media < -0.003:
            print("  ⚠️  El Elo unificado EMPEORA lo doméstico. Costo real.")
        elif media > 0.003:
            print("  ✓ Mejora también lo doméstico.")
        else:
            print("  = Neutral en lo doméstico (no rompe nada).")

    # --- Evaluación 2: partidos UEFA ---
    print(f"\n{'='*66}")
    print("  2) ¿Predice partidos UEFA mejor que el azar?")
    print(f"{'='*66}")

    uefa = df_uni[df_uni["es_uefa"]].reset_index(drop=True)
    conocidos = uefa[~uefa["home_team"].str.startswith("UEFA:") &
                     ~uefa["away_team"].str.startswith("UEFA:")].reset_index(drop=True)
    print(f"Partidos UEFA totales: {len(uefa):,}")
    print(f"Con AMBOS equipos de ligas cubiertas: {len(conocidos):,}")

    if len(conocidos) < 300:
        print("\n  Muestra insuficiente para evaluar.")
        return

    corte = conocidos["date"].quantile(0.8)
    tr_u = conocidos[conocidos["date"] <= corte].reset_index(drop=True)
    ev_u = conocidos[conocidos["date"] > corte].reset_index(drop=True)
    print(f"Train: {len(tr_u):,}  Eval: {len(ev_u):,}")

    # El modelo se entrena con TODOS los partidos domésticos (para tener
    # ataque/defensa de cada equipo) y se evalúa en los UEFA.
    dom = df_uni[~df_uni["es_uefa"] & (df_uni["date"] <= corte)]
    counts = pd.concat([dom["home_team"], dom["away_team"]]).value_counts()
    solidos = set(counts[counts >= 30].index)
    dom = dom[dom["home_team"].isin(solidos) &
              dom["away_team"].isin(solidos)].reset_index(drop=True)

    p_u = ajustar_modelo_elo(dom)
    c_u = ajustar_calibrador(dom, p_u, dict(zip(dom["home_team"], dom["elo_home"])))

    ev_u = ev_u[ev_u["home_team"].isin(solidos) &
                ev_u["away_team"].isin(solidos)].reset_index(drop=True)
    if len(ev_u) < 100:
        print("  Muy pocos partidos UEFA evaluables tras filtrar.")
        return

    b_modelo = brier(ev_u, p_u, c_u)
    b_base = brier_naive(ev_u)
    print(f"\n  Baseline naïve (tasa base UEFA): {b_base:.4f}")
    print(f"  Modelo con Elo unificado:        {b_modelo:.4f}")
    print(f"  VENTAJA: {b_base - b_modelo:+.4f}", end="  ")
    if b_base - b_modelo > 0.02:
        print("→ CALIFICA")
    elif b_base - b_modelo > 0.008:
        print("→ señal moderada")
    elif b_base - b_modelo > 0:
        print("→ marginal")
    else:
        print("→ NO CALIFICA")

    print(f"\n  (evaluado en {len(ev_u)} partidos UEFA)")


if __name__ == "__main__":
    main()
