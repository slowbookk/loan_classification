import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CAT_COLUMNS = [
    "grade",
    "term",
    "home_ownership",
    "verification_status",
    "addr_state",
]


def prepare_inputs(data):
    X = data.copy()

    cats = [
        column
        for column in CAT_COLUMNS
        if column in X
    ]

    for column in cats:
        values = X[column].astype("string")

        X[column] = values.astype(object).where(
            values.notna(),
            np.nan,
        )

    X["dti"] = X["dti"].clip(upper=50)

    nums = X.select_dtypes(
        include="number"
    ).columns.tolist()

    return X[nums + cats], nums, cats


def build_preprocessor(X_train):
    cats = [col for col in CAT_COLUMNS if col in X_train]
    nums = X_train.select_dtypes(include="number").columns.tolist()
    
    numeric_pipeline = Pipeline([
        (
            "impute",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            ),
        ),
        ("scale", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        (
            "impute",
            SimpleImputer(
                strategy="most_frequent",
                keep_empty_features=True,
            ),
        ),
        (
            "encode",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, nums),
        ("categorical", categorical_pipeline, cats),
    ])