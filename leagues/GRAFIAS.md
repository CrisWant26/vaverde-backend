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

## Premier League (E0)
Arsenal, Aston Villa, Bournemouth, Brentford, Brighton, Burnley, Chelsea,
Crystal Palace, Everton, Fulham, Leeds, Liverpool, Man City, Man United,
Newcastle, Nott'm Forest, Sunderland, Tottenham, West Ham, Wolves
OJO: "Nott'm Forest" CON apostrofe | "Man City"/"Man United" abreviados

## Serie A (I1)
Atalanta, Bologna, Cagliari, Como, Cremonese, Fiorentina, Genoa, Inter,
Juventus, Lazio, Lecce, Milan, Napoli, Parma, Pisa, Roma, Sassuolo,
Torino, Udinese, Verona
OJO: "Inter" (no Inter Milan) | "Milan" (no AC Milan) | "Verona" (Hellas)

## Ligue 1 (F1)
Angers, Auxerre, Brest, Le Havre, Le Mans, Lens, Lille, Lorient, Lyon,
Marseille, Metz, Monaco, Nantes, Nice, Paris FC, Paris SG, Rennes,
Strasbourg, Toulouse, Troyes
OJO: "Paris SG" = PSG | "Paris FC" es OTRO equipo distinto

## Bundesliga (D1)
Augsburg, Bayern Munich, Dortmund, Ein Frankfurt, FC Koln, Freiburg,
Hamburg, Heidenheim, Hoffenheim, Leverkusen, M'gladbach, Mainz,
RB Leipzig, St Pauli, Stuttgart, Union Berlin, Werder Bremen, Wolfsburg
OJO: "Ein Frankfurt" | "M'gladbach" con apostrofe | "FC Koln" sin dieresis

## Bundesliga (D1) — temporada 2026-27, ACTUALIZADO 3-sep
Augsburg, Bayern Munich, Dortmund, Ein Frankfurt, Elversberg, FC Koln,
Freiburg, Hamburg, Heidenheim, Hoffenheim, Leverkusen, M'gladbach,
Mainz, Paderborn, RB Leipzig, Schalke 04, St Pauli, Stuttgart,
Union Berlin, Werder Bremen, Wolfsburg
OJO: "Schalke 04" CON el 04 | "Ein Frankfurt"=Eintracht
     "M'gladbach" con apostrofe | "FC Koln" sin dieresis
