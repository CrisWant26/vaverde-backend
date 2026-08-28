"""
adaptador_main.py — Carga de ligas "main" de football-data.co.uk
=================================================================
Las ligas grandes europeas NO usan el formato de archivo único
(MEX.csv, BRA.csv) sino archivos POR TEMPORADA con otro esquema:

  URL:      https://www.football-data.co.uk/mmz4281/{temporada}/{codigo}.csv
  Ejemplo:  .../mmz4281/2526/E0.csv   (Premier League 2025-26)

  Columnas: Date, HomeTeam, AwayTeam, FTHG, FTAG, ...
            (vs Date, Home, Away, HG, AG del formato "extra")

Este módulo descarga N temporadas, las concatena y las normaliza al
mismo esquema canónico que consume elo.py / modelo_elo.py.

Tolerante a fallos: si una temporada no existe o falla, la salta y
avisa. Así una liga con menos histórico igual funciona.

Uso desde generate_league.py: cfg con "formato": "main" y "codigo".
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

# Códigos de liga en football-data (formato main)
CODIGOS = {
    "premier":    "E0",   # Inglaterra
    "laliga":     "SP1",  # España
    "seriea":     "I1",   # Italia
    "bundesliga": "D1",   # Alemania
    "ligue1":     "F1",   # Francia
    "eredivisie": "N1",   # Países Bajos
    "primeira":   "P1",   # Portugal
}


def temporada_actual():
    """Año de inicio de la temporada en curso.

    Las ligas europeas van de agosto a mayo. En julio 2026 la temporada
    que viene es 2026-27, así que devuelve 2026. En marzo 2026 la
    temporada en curso empezó en 2025, así que devuelve 2025.
    """
    from datetime import date
    hoy = date.today()
    return hoy.year if hoy.month >= 7 else hoy.year - 1


def temporadas(desde=2013, hasta=None):
    """Genera los códigos de temporada: 2013 -> '1314', 2025 -> '2526'.

    'hasta' es el año de INICIO de la última temporada completa.
    En julio 2026, la última temporada terminada es 2025-26 -> hasta=2025.
    """
    if hasta is None:
        hasta = temporada_actual()
    out = []
    for a in range(desde, hasta + 1):
        out.append(f"{str(a)[-2:]}{str(a + 1)[-2:]}")
    return out


def _parse_fecha(serie):
    """football-data usa dd/mm/yy en temporadas viejas y dd/mm/yyyy en
    recientes. dayfirst=True maneja ambos."""
    return pd.to_datetime(serie, dayfirst=True, errors="coerce")


def cargar_liga_main(codigo, desde=2013, hasta=None, nombre="Liga", verbose=True,
                     con_stats=False):
    """Descarga y concatena todas las temporadas de una liga main.

    Devuelve el DataFrame canónico: date, home_team, away_team,
    home_score, away_score, tournament, neutral.
    """
    frames = []
    fallidas = []

    for temp in temporadas(desde, hasta):
        url = f"https://www.football-data.co.uk/mmz4281/{temp}/{codigo}.csv"
        try:
            raw = pd.read_csv(url, encoding="utf-8-sig", on_bad_lines="skip")
        except Exception as e:
            fallidas.append(temp)
            continue

        # Verifica que traiga las columnas esperadas
        faltantes = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"} - set(raw.columns)
        if faltantes:
            fallidas.append(f"{temp}(cols:{faltantes})")
            continue

        # BLINDAJE: football-data a veces sube el contenido de una liga en
        # el archivo de otra (visto en 2627/SP1.csv, que traía datos de P1).
        # Si la columna Div no corresponde, se descarta el archivo entero.
        if "Div" in raw.columns:
            divs = set(raw["Div"].dropna().unique())
            if divs and divs != {codigo}:
                fallidas.append(f"{temp}(Div={divs}, esperaba {codigo})")
                continue

        df = pd.DataFrame({
            "date": _parse_fecha(raw["Date"]),
            "home_team": raw["HomeTeam"].astype(str).str.strip(),
            "away_team": raw["AwayTeam"].astype(str).str.strip(),
            "home_score": pd.to_numeric(raw["FTHG"], errors="coerce"),
            "away_score": pd.to_numeric(raw["FTAG"], errors="coerce"),
        })
        if con_stats:
            for col in ("HS", "AS", "HST", "AST", "HC", "AC",
                        "HY", "AY", "HR", "AR", "HF", "AF", "Referee"):
                if col in raw.columns:
                    df[col] = raw[col]

        frames.append(df.dropna(subset=["date", "home_score", "away_score"]))

    if verbose and fallidas:
        print(f"  Temporadas omitidas ({len(fallidas)}): {', '.join(map(str, fallidas))}")

    if not frames:
        raise RuntimeError(f"No se pudo cargar ninguna temporada de {codigo}. "
                           f"¿El código de liga es correcto?")

    out = pd.concat(frames, ignore_index=True)
    out["home_score"] = out["home_score"].astype(int)
    out["away_score"] = out["away_score"].astype(int)
    out["tournament"] = nombre
    out["neutral"] = False          # clubes: localía SIEMPRE
    out = out.drop_duplicates(subset=["date", "home_team", "away_team"])
    return out.sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    import sys
    clave = sys.argv[1] if len(sys.argv) > 1 else "premier"
    if clave not in CODIGOS:
        print(f"Liga desconocida: {clave}. Disponibles: {list(CODIGOS)}")
        sys.exit(1)

    print(f"Descargando {clave} ({CODIGOS[clave]})...")
    df = cargar_liga_main(CODIGOS[clave], nombre=clave)
    print(f"  Partidos: {len(df):,}  ({df['date'].min().date()} → {df['date'].max().date()})")

    act = df[df["date"] >= "2025-08-01"]
    equipos = sorted(set(act["home_team"]) | set(act["away_team"]))
    print(f"  Equipos última temporada ({len(equipos)}):")
    for e in equipos:
        print(f"    {e}")
