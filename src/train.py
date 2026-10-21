from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import RANDOM_STATE


def train_models(X_train, y_train):
    models = {
        "logistic": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    for name, model in models.items():
        print(f"TRAIN: {name}")
        model.fit(X_train, y_train)

    return models