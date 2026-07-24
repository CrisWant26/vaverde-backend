# Grafías exactas de football-data.co.uk

Usar EXACTAMENTE estos nombres en los archivos fixtures_*.csv.
Si un nombre no coincide, el partido no empareja con su resultado.

## Brasileirão (BRA.csv) — temporada 2026, 20 equipos
Athletico-PR      <- Paranaense, con "th"
Atletico-MG       <- Mineiro, sin "h"
Bahia
Botafogo RJ       <- sufijo con ESPACIO
Bragantino
Chapecoense-SC    <- sufijo con GUION
Corinthians
Coritiba
Cruzeiro
Flamengo RJ       <- sufijo con ESPACIO
Fluminense
Gremio
Internacional
Mirassol
Palmeiras
Remo
Santos
Sao Paulo
Vasco
Vitoria

## Liga MX (MEX.csv)
Atl. San Luis, Atlante, Atlas, Club America, Club Leon, Club Tijuana,
Cruz Azul, Guadalajara Chivas, Juarez, Monterrey, Necaxa, Pachuca,
Puebla, Queretaro, Santos Laguna, Tigres UANL, Toluca, UNAM Pumas

## Para regenerar esta lista de cualquier liga:
python3 -c "
import pandas as pd
raw = pd.read_csv('https://www.football-data.co.uk/new/BRA.csv', encoding='utf-8-sig')
raw['Date'] = pd.to_datetime(raw['Date'], format='%d/%m/%Y', errors='coerce')
act = raw[raw['Date'] >= '2026-01-01']
for e in sorted(set(act['Home'].dropna()) | set(act['Away'].dropna())): print(e)
"
