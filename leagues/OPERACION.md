# Rutina semanal de jornadas — VaVerde

## Pasos

1. Pedir fixtures (ChatGPT u otra fuente) y VALIDAR grafías contra GRAFIAS.md.
   Verificar que los equipos estén realmente en primera división.

2. Agregar al CSV de cada liga:
   cat >> leagues/fixtures_<liga>.csv << 'FIN'
   YYYY-MM-DD,Local,Visitante
   FIN

3. Correr local para VERIFICAR (no para publicar):
   python leagues/generate_league.py <liga>
   - Que NO haya avisos de "no existe en el histórico"
   - Que el número de predicciones escritas sea el esperado

4. Descartar TODO lo generado (el bot lo regenera en el workflow):
   git checkout -- docs/leagues/ leagues/pending_*.json leagues/history_*.json

5. Commitear SOLO las fixtures:
   git add leagues/fixtures_*.csv
   git commit -m "Fixtures jornada ..."
   git pull --rebase origin main
   git push origin main

6. Actions -> "Actualizar ligas" -> Run workflow

## Regla de oro
Los JSON de docs/ y los history/pending los publica SOLO el bot.
Si tú también los commiteas, hay conflicto de rebase cada semana.
Excepción: si corriges una grafía DENTRO de un pending, ese día sí lo
commiteas explícitamente (ej. el caso "ADO Den Haag" -> "Den Haag").

## Al agregar una liga NUEVA, son TRES cosas (no se te olvide ninguna):
1. Entrada en LEAGUES de leagues/generate_league.py
2. Paso en .github/workflows/ligas.yml  <-- la que más se olvida
3. Case en Competition.swift de la app iOS

## Ligas activas
Internacional (selecciones), Liga MX, Brasileirao, Primeira Liga,
Eredivisie, LaLiga

## Rechazadas por no superar el baseline
Argentina (-0.0030), MLS (-0.0023)

## Pendientes de integrar (ya configuradas en LEAGUES, falta workflow + app)
Premier League, Serie A, Ligue 1, Bundesliga
