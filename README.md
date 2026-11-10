# Dagvoorspelling elektriciteitsverbruik voor scholen (ASHRAE GEP III)

Per school het uurverbruik (kWh) voorspellen voor de komende 24 uur, op basis van weer, kalender
en recent verbruik. We voorspellen het **75e percentiel** in plaats van het gemiddelde, want een
tekort (€0,06/kWh) is drie keer duurder dan een overschot (€0,02/kWh).
Zie het scopingdocument voor de uitleg.

## Data klaarzetten

De data is te groot voor GitHub. Download van de Kaggle-competitie *ASHRAE – Great Energy
Predictor III* deze drie bestanden en zet ze in `data/`:

- `train.csv`
- `building_metadata.csv`
- `weather_train.csv`

Staan ze elders, pas dan de paden aan in `configs/default.yaml`. De bestanden in `data/` worden
niet gecommit (zie `.gitignore`).

## Draaien

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

pytest                          # draait op nagemaakte data, de echte bestanden zijn niet nodig
ruff check . ; ruff format --check .
python -m src.train             # traint, vergelijkt met de baseline, schrijft models/model.joblib
python -m src.predict 2016-12-15   # voorspelt die dag, schrijft outputs/voorspelling_2016-12-15.csv
```

Draai alles **vanuit deze map**, niet vanuit `src/`.

## Wat waar staat

```
configs/default.yaml   paden, sites, features, kosten, de splitdatum. Geen code
ruff.toml              regels voor opmaak en stijl
src/
  config.py            leest de config, controleert dat geen lag korter is dan de horizon
  schema.py            hoe de data eruit hoort te zien (Pandera)
  data.py              inlezen, valideren, defecte meters (lange nulreeksen) verwijderen
  features.py          kalender, lags en voortschrijdende gemiddelden, zonder lekken
  doel.py              log1p-omrekening van het doel
  metrics.py           RMSLE, CV(RMSE), NMBE, pinball, kost van onevenwicht
  train.py             pijplijn bouwen, trainen, vergelijken met de baseline, wegschrijven
  predict.py           model laden, een dag voorspellen
tests/
  conftest.py          nagemaakte ASHRAE-data
  test_data.py         klopt de data? breekt het schema op kapotte data?
  test_features.py     ziet het model de toekomst niet?
  test_metrics.py      de metrieken op getallen die je met de hand narekent
  test_model.py        vorm, seed, dekking, richting, baseline, opslaan, een dag voorspellen
data/                  hier komen de Kaggle-bestanden (niet committen)
notebooks/             verkenning
models/                wordt aangemaakt door src.train
outputs/               wordt aangemaakt door src.predict
```

## Belangrijke keuzes

- **Kwantielregressie.** Het kwantiel volgt uit de kosten: `q = kost_fn / (kost_fn + kost_fp)`.
  Wijzig je de kosten in de config, dan verandert het kwantiel mee.
- **Geen toekomst.** Elke lag is minstens 24 uur oud. Een voorspelling voor de komende 24 uur
  gebruikt dus enkel verbruik dat op dat moment al gemeten was.
- **Tijdsgebonden split.** Alles voor `split.testvanaf` is train, alles erna is test. Nooit
  willekeurig splitsen bij een tijdreeks.
- **Baseline.** Het model moet goedkoper zijn dan "zelfde uur vorige week" (`kostratio ≤ 1`),
  anders schrijft `src.train` het model niet weg.
- **Weer.** De dataset bevat gemeten weer. In productie heb je een voorspelling, dus de
  testscores zijn te optimistisch.
- **Schoolvakanties.** Zet optioneel een csv met kolommen `site_id,datum` in `data/` en vul
  `vakantie_pad` in de config in. Zonder bestand staat `is_vakantie` overal op 0.
- **NMBE.** Positief betekent dat het model gemiddeld te hoog voorspelt. Bij q = 0,75 hoort die
  licht positief te zijn.
