"""
generar_mapeo.py — Puente entre football-data y openfootball
=============================================================
Para construir un Elo unificado europeo hay que saber que "Ath Bilbao"
(football-data) y "Athletic Club" (openfootball) son el mismo equipo.

Este script genera un CSV de mapeo CANDIDATO. No lo resuelve solo: marca
qué está seguro y qué necesita tu ojo. El trabajo manual se reduce, no
desaparece.

CÓMO ACOTA EL PROBLEMA:
openfootball marca el país de cada equipo, así que para mapear equipos
españoles solo busca entre los (ESP). Eso reduce el espacio de búsqueda
de 367 candidatos a ~40 por liga y sube mucho la precisión.

SALIDA: leagues/mapeo_uefa.csv con columnas
    liga, football_data, openfootball, score, estado
donde estado es:
    AUTO     — score alto, probablemente correcto (revisar por encima)
    REVISAR  — hay candidato pero dudoso (revisar con cuidado)
    SIN_MATCH — no encontró nada (llenar a mano o dejar vacío)

Los equipos que nunca jugaron en Europa se quedan SIN_MATCH y está bien:
simplemente no tendrán puente, pero sí Elo propio de su liga.

Uso:
    python generar_mapeo.py                    # genera el CSV
    python generar_mapeo.py --revisar          # muestra lo pendiente
"""
import os
import sys
import unicodedata
from difflib import SequenceMatcher

import pandas as pd

from adaptador_main import cargar_liga_main

SALIDA = "leagues/mapeo_uefa.csv"
UEFA_CSV = os.path.expanduser("~/uefa_parseado.csv")

# liga -> (código football-data, código de país en openfootball)
LIGAS = {
    "laliga":     ("SP1", "ESP"),
    "premier":    ("E0",  "ENG"),
    "seriea":     ("I1",  "ITA"),
    "bundesliga": ("D1",  "GER"),
    "ligue1":     ("F1",  "FRA"),
    "eredivisie": ("N1",  "NED"),
    "primeira":   ("P1",  "POR"),
}

UMBRAL_AUTO = 0.78
UMBRAL_REVISAR = 0.45

# Abreviaturas que football-data usa y que ninguna heurística adivina
EXPANSIONES = {
    "ath": "athletic atletico",
    "man": "manchester",
    "m'gladbach": "borussia monchengladbach",
    "ein": "eintracht",
    "sp": "sporting",
    "dep": "deportivo",
    "nott'm": "nottingham",
    "for": "fortuna",
    "az": "alkmaar zaanstreek",
    "psv": "philips eindhoven",
    "nec": "nijmegen",
    "rb": "rasenballsport",
    "vfb": "vfb", "vfl": "vfl",
}


def sin_acentos(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def tokens(nombre):
    """Convierte un nombre a un conjunto de palabras comparables."""
    n = sin_acentos(nombre.lower())
    for ch in ".-'/":
        n = n.replace(ch, " ")
    palabras = []
    for p in n.split():
        palabras.extend(EXPANSIONES.get(p, p).split())
    # Ruido societario que no ayuda a distinguir
    ruido = {"fc", "cf", "sc", "ac", "as", "sk", "fk", "sv", "bc", "afc",
             "cfc", "kv", "nk", "hnk", "club", "de", "the", "1", "04", "05"}
    return {p for p in palabras if p not in ruido and len(p) > 1}


def similitud(a, b):
    """Combina similitud de texto con coincidencia de palabras."""
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0

    # Jaccard sobre palabras: lo más informativo
    inter = len(ta & tb)
    union = len(ta | tb)
    jaccard = inter / union if union else 0

    # Si una palabra distintiva coincide exacto, pesa mucho
    bonus = 0.35 if inter >= 1 and min(len(ta), len(tb)) <= 2 else 0

    # Similitud de cadena completa como respaldo
    seq = SequenceMatcher(None,
                          " ".join(sorted(ta)),
                          " ".join(sorted(tb))).ratio()

    return min(jaccard * 0.6 + seq * 0.4 + bonus, 1.0)


def equipos_football_data(codigo, temporadas=5):
    """Equipos de las últimas N temporadas (cubre ascensos/descensos)."""
    df = cargar_liga_main(codigo, desde=2026 - temporadas, nombre=codigo,
                          verbose=False)
    return sorted(set(df["home_team"]) | set(df["away_team"]))


def equipos_openfootball(pais):
    """Equipos de ese país que aparecen en competencias UEFA."""
    if not os.path.exists(UEFA_CSV):
        print(f"ERROR: no existe {UEFA_CSV}")
        print("Corre primero: python parser_openfootball_v2.py ~/champions-league")
        sys.exit(1)
    df = pd.read_csv(UEFA_CSV)
    locales = df[df["pais_home"] == pais][["home_team", "home_raw"]]
    locales.columns = ["norm", "raw"]
    visitas = df[df["pais_away"] == pais][["away_team", "away_raw"]]
    visitas.columns = ["norm", "raw"]
    todos = pd.concat([locales, visitas]).drop_duplicates(subset=["norm"])
    return sorted(zip(todos["norm"], todos["raw"]))


def generar():
    filas = []
    for liga, (cod, pais) in LIGAS.items():
        print(f"\n=== {liga} ({cod} / {pais}) ===")
        try:
            fd = equipos_football_data(cod)
        except Exception as e:
            print(f"  error cargando football-data: {e}")
            continue
        of = equipos_openfootball(pais)
        print(f"  football-data: {len(fd)} equipos | openfootball: {len(of)}")

        for nombre_fd in fd:
            mejor, score = None, 0.0
            for norm, raw in of:
                s = max(similitud(nombre_fd, norm), similitud(nombre_fd, raw))
                if s > score:
                    mejor, score = norm, s

            estado = ("AUTO" if score >= UMBRAL_AUTO
                      else "REVISAR" if score >= UMBRAL_REVISAR
                      else "SIN_MATCH")
            filas.append({
                "liga": liga,
                "football_data": nombre_fd,
                "openfootball": mejor if estado != "SIN_MATCH" else "",
                "score": round(score, 3),
                "estado": estado,
            })

        auto = sum(1 for f in filas if f["liga"] == liga and f["estado"] == "AUTO")
        rev = sum(1 for f in filas if f["liga"] == liga and f["estado"] == "REVISAR")
        sin = sum(1 for f in filas if f["liga"] == liga and f["estado"] == "SIN_MATCH")
        print(f"  AUTO {auto}  |  REVISAR {rev}  |  SIN_MATCH {sin}")

    df = pd.DataFrame(filas)
    os.makedirs("leagues", exist_ok=True)
    df.to_csv(SALIDA, index=False)

    print(f"\n{'='*60}")
    print(f"Guardado en {SALIDA}  ({len(df)} equipos)")
    tot = df["estado"].value_counts()
    for est in ("AUTO", "REVISAR", "SIN_MATCH"):
        print(f"  {est:<10} {tot.get(est, 0)}")
    print("\nSIGUIENTE PASO: abre el CSV y revisa.")
    print("  - Los AUTO: pasada rápida, corrige los que estén mal.")
    print("  - Los REVISAR: uno por uno.")
    print("  - Los SIN_MATCH: llena la columna 'openfootball' si el equipo")
    print("    sí jugó en Europa; si nunca jugó, déjalo vacío (es correcto).")


def revisar():
    if not os.path.exists(SALIDA):
        print(f"No existe {SALIDA}. Corre primero: python generar_mapeo.py")
        return
    df = pd.read_csv(SALIDA).fillna("")
    pend = df[df["estado"] != "AUTO"]
    print(f"Pendientes de revisar: {len(pend)}\n")
    for liga in pend["liga"].unique():
        sub = pend[pend["liga"] == liga]
        print(f"--- {liga} ---")
        for r in sub.itertuples():
            print(f"  [{r.estado:<9}] {r.football_data:<22} → "
                  f"{r.openfootball or '(vacío)':<28} score {r.score}")
        print()


if __name__ == "__main__":
    if "--revisar" in sys.argv:
        revisar()
    else:
        generar()
