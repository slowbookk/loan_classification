import numpy as np
import pandas as pd


def add_features(data):
    data = data.copy()

    credit_date = pd.to_datetime(
        data["earliest_cr_line"],
        errors="coerce",
    )

    data["cr_year"] = credit_date.dt.year

    data["income_to_loan"] = (
        data["annual_inc"]
        / data["loan_amnt"].replace(0, np.nan)
    )

    data["closed_acc"] = (
        data["total_acc"] - data["open_acc"]
    )

    return data.drop(columns="earliest_cr_line")