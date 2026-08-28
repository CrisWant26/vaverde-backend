"""
tendencias.py — Promedios de últimos N partidos (mercados secundarios)
=======================================================================
NO es un modelo predictivo. Son promedios descriptivos de lo que cada
equipo hizo en sus últimos partidos, más el promedio de la liga como
referencia para que el usuario sepa si un número es alto o bajo.

La distinción importa: una predicción tiene que superar un baseline
para publicarse; un promedio observado es un hecho y no necesita
validación. Por eso esta feature sale sin backtest.

Columnas de football-data que se usan (solo ligas "main"):
  HC / AC   córners
  HST / AST tiros a puerta
  HY / AY   tarjetas amarillas

Las ligas "extra" (México, Brasil, Argentina) NO traen estas columnas,
así que para ellas se devuelve None y la app oculta la sección.

Uso desde generate_league.py:
    from tendencias import calcular_tendencias
    trends = calcular_tendencias(df_completo, home, away, n=10)
"""
import pandas as pd

# Métricas que se muestran: (clave_json, col_local, col_visita, etiqueta)
METRICAS = [
    ("corners", "HC", "AC", "Córners"),
    ("shots_on_target", "HST", "AST", "Tiros a puerta"),
    ("yellow_cards", "HY", "AY", "Amarillas"),
]

COLUMNAS_REQUERIDAS = {"HC", "AC", "HST", "AST", "HY", "AY"}


def tiene_estadisticas(df):
    """True si el DataFrame trae las columnas de estadísticas."""
    return COLUMNAS_REQUERIDAS.issubset(set(df.columns))


def _promedios_equipo(df, equipo, n):
    """Promedios a favor y en contra del equipo en sus últimos n partidos.

    Toma los partidos del equipo (como local o visitante) ordenados por
    fecha, se queda con los n más recientes, y promedia cada métrica
    desde la perspectiva del equipo.
    """
    partidos = df[
        (df["home_team"] == equipo) | (df["away_team"] == equipo)
    ].sort_values("date").tail(n)

    if partidos.empty:
        return None

    out = {"partidos": int(len(partidos))}

    for clave, col_h, col_a, _ in METRICAS:
        a_favor, en_contra = [], []
        for r in partidos.itertuples():
            es_local = (r.home_team == equipo)
            v_local = getattr(r, col_h, None)
            v_visita = getattr(r, col_a, None)
            if pd.isna(v_local) or pd.isna(v_visita):
                continue
            a_favor.append(v_local if es_local else v_visita)
            en_contra.append(v_visita if es_local else v_local)

        if a_favor:
            out[f"{clave}_for"] = round(sum(a_favor) / len(a_favor), 1)
            out[f"{clave}_against"] = round(sum(en_contra) / len(en_contra), 1)

    return out


def _promedios_liga(df, temporadas_recientes=2):
    """Promedio de la liga (total por partido) como referencia.

    Usa solo las temporadas recientes: el fútbol cambia y un promedio
    de hace 10 años no sirve de referencia para hoy.
    """
    corte = df["date"].max() - pd.Timedelta(days=365 * temporadas_recientes)
    reciente = df[df["date"] >= corte]
    if reciente.empty:
        reciente = df

    out = {"partidos": int(len(reciente))}
    for clave, col_h, col_a, _ in METRICAS:
        if col_h not in reciente.columns:
            continue
        serie = (pd.to_numeric(reciente[col_h], errors="coerce")
                 + pd.to_numeric(reciente[col_a], errors="coerce")).dropna()
        if len(serie):
            out[f"{clave}_total"] = round(float(serie.mean()), 1)
    return out


def calcular_tendencias(df, home, away, n=10):
    """Devuelve el bloque de tendencias para un partido, o None si la
    liga no trae estadísticas.

    Estructura:
        {
          "n_partidos": 10,
          "home": {"corners_for": 5.4, "corners_against": 4.8, ...},
          "away": {...},
          "liga": {"corners_total": 9.8, ...}
        }
    """
    if not tiene_estadisticas(df):
        return None

    t_home = _promedios_equipo(df, home, n)
    t_away = _promedios_equipo(df, away, n)
    if t_home is None or t_away is None:
        return None

    return {
        "n_partidos": n,
        "home": t_home,
        "away": t_away,
        "liga": _promedios_liga(df),
    }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from adaptador_main import cargar_liga_main, CODIGOS

    clave = sys.argv[1] if len(sys.argv) > 1 else "premier"
    home = sys.argv[2] if len(sys.argv) > 2 else "Arsenal"
    away = sys.argv[3] if len(sys.argv) > 3 else "Chelsea"

    df = cargar_liga_main(CODIGOS[clave], nombre=clave, con_stats=True)
    print(f"Columnas de stats presentes: {tiene_estadisticas(df)}")
    t = calcular_tendencias(df, home, away)
    if t is None:
        print("Esta liga no tiene estadísticas disponibles.")
    else:
        import json
        print(json.dumps(t, indent=2, ensure_ascii=False))
