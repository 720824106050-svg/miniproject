"""
Generates a realistic synthetic dataset of daily household water consumption.

Real smart-meter datasets are rarely public/clean enough to ship in a demo,
so we simulate one from a physically-plausible formula (baseline per-person
use + temperature/garden/weekend effects + seasonal swing + noise). Swap
`load_real_data()` in for this once real meter readings are available -
the model training code doesn't care where the rows come from.
"""

import numpy as np
import pandas as pd

SEASONS = ["winter", "spring", "summer", "autumn"]
SEASON_OFFSET = {"winter": -25, "spring": 0, "summer": 35, "autumn": -5}


def generate_dataset(n_rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    household_size = rng.integers(1, 7, size=n_rows)  # 1-6 occupants
    season = rng.choice(SEASONS, size=n_rows)
    temperature_c = rng.normal(
        loc=[10 if s == "winter" else 22 if s == "summer" else 16 for s in season],
        scale=5,
    )
    is_weekend = rng.integers(0, 2, size=n_rows)
    has_garden = rng.integers(0, 2, size=n_rows)

    baseline_per_person = 115  # litres/person/day, typical indoor use
    temp_effect = np.clip(temperature_c - 18, 0, None) * 2.2
    weekend_effect = is_weekend * 12
    garden_effect = has_garden * np.clip(temperature_c - 15, 0, None) * 3.5
    season_effect = np.array([SEASON_OFFSET[s] for s in season])
    noise = rng.normal(0, 15, size=n_rows)

    liters_per_day = (
        baseline_per_person * household_size
        + temp_effect
        + weekend_effect
        + garden_effect
        + season_effect
        + noise
    )
    liters_per_day = np.clip(liters_per_day, 40, None)

    return pd.DataFrame(
        {
            "household_size": household_size,
            "temperature_c": np.round(temperature_c, 1),
            "is_weekend": is_weekend,
            "has_garden": has_garden,
            "season": season,
            "liters_per_day": np.round(liters_per_day, 1),
        }
    )


def generate_history(days: int = 30, household_size: int = 3, has_garden: int = 0, seed: int = 7) -> pd.DataFrame:
    """30-ish days of 'recent readings' for a single household, for the dashboard chart."""
    rng = np.random.default_rng(seed)
    today = pd.Timestamp.today().normalize()
    dates = pd.date_range(end=today, periods=days)

    rows = []
    for d in dates:
        month = d.month
        season = (
            "winter" if month in (12, 1, 2)
            else "spring" if month in (3, 4, 5)
            else "summer" if month in (6, 7, 8)
            else "autumn"
        )
        base_temp = {"winter": 10, "spring": 16, "summer": 22, "autumn": 16}[season]
        temperature_c = rng.normal(base_temp, 4)
        is_weekend = int(d.dayofweek >= 5)

        baseline_per_person = 115
        temp_effect = max(temperature_c - 18, 0) * 2.2
        weekend_effect = is_weekend * 12
        garden_effect = has_garden * max(temperature_c - 15, 0) * 3.5
        season_effect = SEASON_OFFSET[season]
        noise = rng.normal(0, 10)

        liters = baseline_per_person * household_size + temp_effect + weekend_effect + garden_effect + season_effect + noise
        rows.append({"date": d.strftime("%Y-%m-%d"), "liters_per_day": round(max(liters, 40), 1)})

    return pd.DataFrame(rows)
