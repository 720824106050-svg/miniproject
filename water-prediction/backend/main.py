from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from data_generator import generate_history
from model import load_or_train, predict, train_and_save

app = FastAPI(
    title="Smart Water Consumption Prediction API",
    description="Predicts daily household water consumption from household and weather features.",
    version="1.0.0",
)

# CORS enabled for static frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = load_or_train()


class PredictRequest(BaseModel):
    household_size: int = Field(default=3, ge=1, le=15, description="Number of people in the household")
    temperature_c: float = Field(default=22.0, ge=-20, le=50, description="Average daily temperature in Celsius")
    is_weekend: bool = False
    has_garden: bool = False
    season: Literal["winter", "spring", "summer", "autumn"] = "spring"


class PredictResponse(BaseModel):
    predicted_liters_per_day: float
    predicted_liters_per_month: float
    estimated_cost_per_month: float
    liters_per_person: float


COST_PER_1000_LITERS = 2.5


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict", response_model=PredictResponse)
def predict_consumption(req: PredictRequest):
    liters_per_day = predict(
        pipeline,
        household_size=req.household_size,
        temperature_c=req.temperature_c,
        is_weekend=req.is_weekend,
        has_garden=req.has_garden,
        season=req.season,
    )
    liters_per_month = round(liters_per_day * 30, 1)
    cost_per_month = round((liters_per_month / 1000) * COST_PER_1000_LITERS, 2)
    per_person = round(liters_per_day / req.household_size, 1)

    return PredictResponse(
        predicted_liters_per_day=liters_per_day,
        predicted_liters_per_month=liters_per_month,
        estimated_cost_per_month=cost_per_month,
        liters_per_person=per_person,
    )


@app.get("/api/history")
def history(household_size: int = 3, has_garden: bool = False, days: int = 30):
    if not (1 <= days <= 365):
        raise HTTPException(status_code=400, detail="days must be between 1 and 365")
    df = generate_history(days=days, household_size=household_size, has_garden=int(has_garden))
    return df.to_dict(orient="records")


@app.post("/api/retrain")
def retrain(n_rows: int = 2000):
    """Regenerates the synthetic dataset and retrains the model in place."""
    global pipeline
    try:
        metrics = train_and_save(n_rows=n_rows)
        pipeline = load_or_train()
        return {"message": "Model retrained", "mae": metrics.mae, "r2": metrics.r2, "n_rows": metrics.n_rows}
    except Exception as e:
        return {"message": f"Retrain failed: {e}"}
