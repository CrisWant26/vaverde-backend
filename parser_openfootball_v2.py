"""
parser_openfootball_v2.py — Parser corregido + normalización de nombres
========================================================================
Corrige dos problemas de la versión anterior:

1. BUG DE FECHAS. El año solo aparece en la primera fecha de cada
   bloque; las demás lo heredan. La v1 asumía que si el mes retrocedía
   era cambio de año (dic -> ene), pero los archivos tienen VARIOS
   bloques cronológicos (League, Playoffs, Finals) que reinician en
   septiembre. Resultado: las fechas se corrían hasta 2027.
   FIX: detectar el reinicio de bloque (▪ o encabezado de sección) y
   reiniciar el año al de la temporada del archivo.

2. NOMBRES DUPLICADOS. openfootball no es consistente consigo mismo:
   "AS Monaco" / "AS Monaco FC", "Atalanta" / "Atalanta BC",
   "Aston Villa" / "Aston Villa FC" son el mismo club escrito distinto
   según la temporada.
   FIX: normalización por sufijos/prefijos societarios comunes.

Licencia de los datos: CC0 (dominio público).

Uso:
    python parser_openfootball_v2.py ~/champions-league
"""
import os
import re
import sys
from collections import defaultdict

import pandas as pd

MESES = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
         "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}

RE_FECHA = re.compile(
    r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+"
    r"(?P<mes>[A-Z][a-z]{2})\s+(?P<dia>\d{1,2})"
    r"(?:\s+(?P<anio>\d{4}))?\s*$")

RE_PARTIDO = re.compile(
    r"^\s*(?:\d{1,2}:\d{2}\s+)?"
    r"(?P<home>.+?)\s+\((?P<pais_h>[A-Z]{3})\)\s+"
    r"v\s+"
    r"(?P<away>.+?)\s+\((?P<pais_a>[A-Z]{3})\)\s+"
    r"(?P<hg>\d+)-(?P<ag>\d+)")

# Inicio de un bloque nuevo (ronda/fase). Resetea el seguimiento de año.
RE_BLOQUE = re.compile(r"^\s*[▪»]\s*\S")

# Sufijos y prefijos societarios que openfootball usa de forma inconsistente
SUFIJOS = [
    " FC", " CF", " SC", " AC", " BC", " SK", " FK", " AFC", " CFC",
    " SFP", " PFK", " KV", " SAD", " AS", " SV", " TSV", " VfL", " VfB",
    " BSC", " 1846", " 04", " 05", " 96", " 1899", " 1900", " 1909",
]
PREFIJOS = [
    "FC ", "AFC ", "CF ", "SC ", "AC ", "AS ", "SK ", "FK ", "SV ",
    "VfL ", "VfB ", "BSC ", "TSV ", "PAE ", "PFK ", "KV ", "NK ", "HNK ",
]


def normalizar_nombre(nombre):
    """Reduce variantes del mismo club a una forma canónica.

    "AS Monaco FC" -> "monaco"
    "AS Monaco"    -> "monaco"
    "Atalanta BC"  -> "atalanta"

    NO resuelve todos los casos (los nombres compuestos raros se quedan),
    pero colapsa la mayoría de duplicados por sufijo/prefijo societario.
    """
    n = nombre.strip()

    cambio = True
    while cambio:
        cambio = False
        for suf in SUFIJOS:
            if n.endswith(suf) and len(n) > len(suf) + 2:
                n = n[: -len(suf)].strip()
                cambio = True
        for pre in PREFIJOS:
            if n.startswith(pre) and len(n) > len(pre) + 2:
                n = n[len(pre):].strip()
                cambio = True

    return n.lower().strip()


def anio_inicio_temporada(carpeta):
    """'2025-26' -> 2025"""
    m = re.match(r"^(\d{4})", carpeta)
    return int(m.group(1)) if m else None


def parsear_archivo(path, anio_temporada):
    """Parsea un .txt. anio_temporada es el año de INICIO (ej. 2025 para
    la temporada 2025-26), usado para reconstruir fechas sin año."""
    partidos = []
    anio_actual = anio_temporada
    mes_anterior = None

    with open(path, encoding="utf-8") as f:
        for linea in f:
            # Nuevo bloque: reinicia el seguimiento cronológico
            if RE_BLOQUE.match(linea):
                anio_actual = anio_temporada
                mes_anterior = None
                continue

            m = RE_FECHA.match(linea)
            if m:
                mes = MESES[m.group("mes")]
                dia = int(m.group("dia"))

                if m.group("anio"):
                    anio_actual = int(m.group("anio"))
                else:
                    # Dentro del mismo bloque: si el mes retrocede y
                    # estamos en la segunda mitad de temporada, es el
                    # año siguiente. Nunca más de +1 sobre la temporada.
                    if mes_anterior is not None and mes < mes_anterior:
                        if anio_actual < anio_temporada + 1:
                            anio_actual += 1
                    elif mes <= 7 and anio_actual == anio_temporada:
                        # ene-jul de una temporada que empezó en agosto
                        anio_actual = anio_temporada + 1

                mes_anterior = mes
                try:
                    fecha_actual = pd.Timestamp(year=anio_actual, month=mes, day=dia)
                except ValueError:
                    fecha_actual = None
                continue

            m = RE_PARTIDO.match(linea)
            if m and 'fecha_actual' in dir() and fecha_actual is not None:
                partidos.append({
                    "date": fecha_actual,
                    "home_raw": m.group("home").strip(),
                    "away_raw": m.group("away").strip(),
                    "home_team": normalizar_nombre(m.group("home")),
                    "away_team": normalizar_nombre(m.group("away")),
                    "home_score": int(m.group("hg")),
                    "away_score": int(m.group("ag")),
                    "pais_home": m.group("pais_h"),
                    "pais_away": m.group("pais_a"),
                    "temporada": f"{anio_temporada}-{str(anio_temporada+1)[-2:]}",
                })
    return partidos


def cargar_repo(ruta_repo):
    todos = []
    for carpeta in sorted(os.listdir(ruta_repo)):
        sub = os.path.join(ruta_repo, carpeta)
        if not os.path.isdir(sub) or carpeta.startswith("."):
            continue
        anio = anio_inicio_temporada(carpeta)
        if anio is None:
            continue
        for archivo in sorted(os.listdir(sub)):
            if not archivo.endswith(".txt"):
                continue
            n = parsear_archivo(os.path.join(sub, archivo), anio)
            print(f"  {carpeta}/{archivo}: {len(n)} partidos")
            todos.extend(n)

    df = pd.DataFrame(todos)
    if df.empty:
        return df
    df["tournament"] = "UEFA"
    df["neutral"] = False
    return df.sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    ruta = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/champions-league")
    print(f"Parseando {ruta}...\n")
    df = cargar_repo(ruta)

    if df.empty:
        print("\nNo se parseó nada.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"TOTAL: {len(df):,} partidos")
    print(f"Rango de fechas: {df['date'].min().date()} → {df['date'].max().date()}")

    # Verificación del bug de fechas
    fuera = df[df["date"] > pd.Timestamp("2026-12-31")]
    if len(fuera):
        print(f"\n⚠️  {len(fuera)} partidos con fecha posterior a 2026 "
              f"(el bug de fechas persiste)")
        print(fuera[["date", "home_raw", "away_raw", "temporada"]].head().to_string())
    else:
        print("✓ Sin fechas imposibles — el bug quedó corregido.")

    # Efecto de la normalización
    crudos = set(df["home_raw"]) | set(df["away_raw"])
    normalizados = set(df["home_team"]) | set(df["away_team"])
    print(f"\nEquipos con nombre crudo:     {len(crudos)}")
    print(f"Equipos tras normalizar:      {len(normalizados)}")
    print(f"Duplicados colapsados:        {len(crudos) - len(normalizados)}")

    # Qué variantes se unieron (para revisar que no junte equipos distintos)
    variantes = defaultdict(set)
    for _, r in df.iterrows():
        variantes[r["home_team"]].add(r["home_raw"])
        variantes[r["away_team"]].add(r["away_raw"])
    multi = {k: v for k, v in variantes.items() if len(v) > 1}
    print(f"\nNombres canónicos con varias variantes: {len(multi)}")
    print("(revisa que no haya juntado equipos DISTINTOS)\n")
    for k, v in sorted(multi.items())[:20]:
        print(f"  {k:<28} ← {', '.join(sorted(v))}")

    print(f"\nPartidos por país (puentes por liga):")
    paises = pd.concat([
        df[["pais_home"]].rename(columns={"pais_home": "pais"}),
        df[["pais_away"]].rename(columns={"pais_away": "pais"})])
    print(paises["pais"].value_counts().head(12).to_string())

    salida = os.path.expanduser("~/uefa_parseado.csv")
    df.to_csv(salida, index=False)
    print(f"\nGuardado en {salida}")
