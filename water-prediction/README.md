# Meterwise — Smart Water Consumption Prediction

Predicts daily household water consumption from a handful of easy-to-collect
inputs (household size, temperature, weekend/weekday, garden, season) using a
scikit-learn regression model, served by a FastAPI backend and shown in a
standalone HTML/CSS/JS dashboard.

```
water-prediction/
├── backend/
│   ├── main.py            FastAPI app (endpoints below)
│   ├── model.py           Trains/loads the RandomForestRegressor
│   ├── data_generator.py  Synthetic training data + 30-day history generator
│   └── requirements.txt
├── frontend/
│   └── index.html         Self-contained dashboard (no build step)
└── README.md
```

## 1. Run the backend

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The first request trains the model on synthetic data (a couple of seconds)
and caches it to `backend/model.joblib`. Delete that file any time to force
a retrain, or call `POST /api/retrain`.

Interactive API docs: http://localhost:8000/docs

### Endpoints

| Method | Path            | Description                                      |
|--------|-----------------|---------------------------------------------------|
| GET    | `/api/health`   | Liveness check                                     |
| POST   | `/api/predict`  | Predict L/day from household + weather features    |
| GET    | `/api/history`  | 30-day synthetic reading history for the dashboard  |
| POST   | `/api/retrain`  | Regenerate data and retrain the model in place      |

Example predict request:
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"household_size":4,"temperature_c":28,"is_weekend":true,"has_garden":true,"season":"summer"}'
```

## 2. Run the frontend

`frontend/index.html` is a plain static file — no build step, no npm install.
Just open it, or serve it so `fetch` behaves consistently with the API's CORS setup:

```bash
cd frontend
python -m http.server 5500
```

Then visit http://localhost:5500. It talks to the API at `http://localhost:8000`
by default — change `API_BASE` near the top of the `<script>` block in
`index.html` if your backend runs elsewhere.

## About the model

Real smart-meter datasets are rarely available in a clean, shareable form, so
`data_generator.py` simulates ~2,000 rows from a physically-plausible formula:
a per-person baseline, a temperature effect above ~18°C, a weekend bump, a
garden-irrigation effect on hot days, and a seasonal offset, all with noise
layered on top. The model is a `RandomForestRegressor` wrapped in a
scikit-learn `Pipeline` (one-hot encoding for season) — swap in real
meter/billing data by replacing `generate_dataset()` in `data_generator.py`
with a loader for your CSV/database; nothing else in `model.py` or `main.py`
needs to change as long as the column names match.

## Next steps you may want

- Persist real per-household readings (Postgres/SQLite) instead of synthetic history
- Add authentication if this moves beyond a local demo
- Swap the tariff constant in `main.py` for your utility's real rate schedule
- Retrain on a schedule as new meter data arrives
