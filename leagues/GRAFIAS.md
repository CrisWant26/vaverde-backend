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

## Primeira Liga (P1) — temporada 2025-26, 18 equipos
AVS, Alverca, Arouca, Benfica, Casa Pia, Estoril, Estrela, Famalicao,
Gil Vicente, Guimaraes, Moreirense, Nacional, Porto, Rio Ave,
Santa Clara, Sp Braga, Sp Lisbon, Tondela
OJO: "Sp Lisbon" = Sporting CP | "Guimaraes" = Vitória SC

## Eredivisie (N1) — temporada 2025-26, 18 equipos
AZ Alkmaar, Ajax, Excelsior, Feyenoord, For Sittard, Go Ahead Eagles,
Groningen, Heerenveen, Heracles, NAC Breda, Nijmegen, PSV Eindhoven,
Sparta Rotterdam, Telstar, Twente, Utrecht, Volendam, Zwolle
OJO: "For Sittard" = Fortuna Sittard | "Nijmegen" = NEC

## LaLiga (SP1) — temporada 2025-26, 20 equipos
Alaves, Ath Bilbao, Ath Madrid, Barcelona, Betis, Celta, Elche, Espanol,
Getafe, Girona, Levante, Mallorca, Osasuna, Oviedo, Real Madrid, Sevilla,
Sociedad, Valencia, Vallecano, Villarreal
OJO: "Ath Bilbao"=Athletic | "Ath Madrid"=Atlético | "Espanol" sin ñ
     "Sociedad"=Real Sociedad | "Vallecano"=Rayo | "Celta"=Celta de Vigo
NOTA: al 13-ago-2026 football-data tiene roto el archivo 2627/SP1.csv
(contiene datos de P1). Verificar con: python adaptador_main.py laliga

## LaLiga (SP1) — temporada 2026-27
Alaves, Ath Bilbao, Ath Madrid, Barcelona, Betis, Celta, Dep. A Coruna,
Elche, Espanol, Getafe, Girona, Levante, Mallorca, Osasuna, Oviedo,
Real Madrid, Santander, Sevilla, Sociedad, Valencia, Vallecano, Villarreal
OJO: "Ath Bilbao"=Athletic | "Ath Madrid"=Atletico | "Espanol" sin enye
     "Sociedad"=Real Sociedad | "Vallecano"=Rayo | "Santander"=Racing
     "Dep. A Coruna"=Deportivo (con punto y espacio)
