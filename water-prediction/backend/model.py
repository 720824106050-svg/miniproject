"""
Trains and serves a regression model that predicts daily household water
consumption (litres/day) from a handful of easy-to-collect features.
"""

import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from data_generator import generate_dataset

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

NUMERIC_FEATURES = ["household_size", "temperature_c", "is_weekend", "has_garden"]
CATEGORICAL_FEATURES = ["season"]


@dataclass
class TrainMetrics:
    mae: float
    r2: float
    n_rows: int


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def train_and_save(n_rows: int = 2000) -> TrainMetrics:
    df = generate_dataset(n_rows=n_rows)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["liters_per_day"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    metrics = TrainMetrics(
        mae=round(float(mean_absolute_error(y_test, preds)), 2),
        r2=round(float(r2_score(y_test, preds)), 3),
        n_rows=n_rows,
    )

    joblib.dump(pipeline, MODEL_PATH)
    return metrics


def fallback_predict(household_size: int, temperature_c: float, is_weekend: bool, has_garden: bool, season: str) -> float:
    baseline_per_person = 115.0
    temp_effect = max(float(temperature_c) - 18.0, 0.0) * 2.2
    weekend_effect = 12.0 if is_weekend else 0.0
    garden_effect = (max(float(temperature_c) - 15.0, 0.0) * 3.5) if has_garden else 0.0
    season_offsets = {"winter": -25.0, "spring": 0.0, "summer": 35.0, "autumn": -5.0}
    season_effect = season_offsets.get(season, 0.0)
    liters = (baseline_per_person * household_size) + temp_effect + weekend_effect + garden_effect + season_effect
    return round(float(max(liters, 40.0)), 1)


def load_or_train() -> Pipeline:
    try:
        if os.path.exists(MODEL_PATH):
            return joblib.load(MODEL_PATH)
        train_and_save()
        return joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Warning: Could not load or train model ({e}). Using fallback prediction.")
        return None


def predict(pipeline: Pipeline, household_size: int, temperature_c: float,
            is_weekend: bool, has_garden: bool, season: str) -> float:
    if pipeline is None:
        return fallback_predict(household_size, temperature_c, is_weekend, has_garden, season)
    try:
        row = pd.DataFrame([{
            "household_size": household_size,
            "temperature_c": temperature_c,
            "is_weekend": int(is_weekend),
            "has_garden": int(has_garden),
            "season": season,
        }])
        pred = pipeline.predict(row)[0]
        return round(float(pred), 1)
    except Exception as e:
        print(f"Prediction failed with model ({e}). Using fallback prediction.")
        return fallback_predict(household_size, temperature_c, is_weekend, has_garden, season)

