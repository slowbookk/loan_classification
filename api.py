import json
import pickle

import pandas as pd
from fastapi import FastAPI

from src.cleaning import clean_inputs
from src.config import MODELS_DIR
from src.features import add_features


app = FastAPI()

models = {}
for name in ["logistic", "random_forest"]:
    with open(MODELS_DIR / f"{name}_pipeline.pkl", "rb") as file:
        models[name] = pickle.load(file)

schema = json.loads(
    (MODELS_DIR / "input_schema.json").read_text()
)


@app.post("/predict")
def predict(features: dict):
    data = pd.DataFrame([features]).reindex(columns=schema["columns"])
    data = clean_inputs(data)
    data = add_features(data)

    results = {}

    for name, model in models.items():
        probability = float(model.predict_proba(data)[0, 1])

        results[name] = {
            "prediction": int(probability >= 0.5),
            "default_probability": round(probability, 4),
        }

    return results