import json
import pickle

import pandas as pd
from sklearn.pipeline import Pipeline

from src.cleaning import clean_data
from src.config import RAW_CSV, NROWS, MODELS_DIR, REPORTS_DIR
from src.evaluate import evaluate_model
from src.features import add_features
from src.preprocessing import prepare_inputs, build_preprocessor
from src.split_data import separate_target, train_test
from src.train import train_models


def main():
    print("LOAD DATA")
    data = pd.read_csv(RAW_CSV, nrows=NROWS, low_memory=False)

    print("CLEAN DATA")
    data = clean_data(data)
    raw_inputs, y = separate_target(data)

    schema = {
        "columns": raw_inputs.columns.tolist(),
        "numeric": raw_inputs.select_dtypes(include="number").columns.tolist(),
    }

    print("FEATURE ENGINEERING")
    engineered = add_features(raw_inputs)
    X, _, _ = prepare_inputs(engineered)

    print("SPLIT")
    X_train, X_test, y_train, y_test = train_test(X, y)

    print("PREPROCESS")
    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)

    print("TRAIN")
    models = train_models(X_train_processed, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", model),
        ])

        print(f"EVALUATE: {name}")
        results[name] = evaluate_model(pipeline, X_test, y_test)

        with open(MODELS_DIR / f"{name}_pipeline.pkl", "wb") as file:
            pickle.dump(pipeline, file)

    (MODELS_DIR / "input_schema.json").write_text(
        json.dumps(schema, indent=2)
    )
    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps(results, indent=2)
    )


if __name__ == "__main__":
    main()