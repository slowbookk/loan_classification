
import numpy as np
import pandas as pd
from src.preprocessing import CAT_COLUMNS
DOMAIN_DROP_COLS = [
    # Identifiers and text
    "id", "member_id", "url",
    "desc", "emp_title", "title", "zip_code",

    # Redundant or excluded fields
    "funded_amnt", "funded_amnt_inv",
    "sub_grade", "fico_range_high",
    "installment", "purpose", "initial_list_status",

    # Delinquency fields
    "num_tl_90g_dpd_24m", "num_accts_ever_120_pd",
    "mths_since_last_delinq", "delinq_2yrs",

    # Inquiry fields
    "inq_last_12m", "inq_fi", "mths_since_recent_inq",

    # Recent account-opening fields
    "open_acc_6m", "acc_open_past_24mths",
    "num_tl_op_past_12m",

    # Detailed bankcard fields
    "bc_open_to_buy", "bc_util", "max_bal_bc",
    "num_actv_bc_tl", "num_bc_sats", "num_bc_tl",
    "mths_since_recent_bc", "percent_bc_gt_75",
    "total_bc_limit",

    # Post-loan leakage
    "out_prncp", "out_prncp_inv",
    "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt",
    "next_pymnt_d", "last_credit_pull_d",
    "last_fico_range_high", "last_fico_range_low",
    "debt_settlement_flag",

    # Other exclusions and original target
    "issue_d", "application_type", "loan_status",
]

DATE_COLS = ["earliest_cr_line"]

def clean_inputs(data):
    data = data.copy()

    replacements = {
        "< 1 year": "0",
        "10+ years": "10",
        "n/a": pd.NA,
    }

    for col in ["term", "emp_length"]:
        if col in data:
            values = data[col].astype("string").replace(replacements)
            data[col] = pd.to_numeric(
                values.str.extract(r"(\d+)", expand=False),
                errors="coerce",
            )

    if "earliest_cr_line" in data:
        data["earliest_cr_line"] = pd.to_datetime(
            data["earliest_cr_line"], format="%b-%Y", errors="coerce"
        )

    if "dti" in data:
        data["dti"] = pd.to_numeric(data["dti"], errors="coerce").clip(upper=50)

    for col in CAT_COLUMNS:
        if col in data:
            values = data[col].astype("string")
            data[col] = values.astype(object).where(values.notna(), np.nan)

    return data


def clean_data(data):
    status_mapping = {"Fully Paid": 0, "Charged Off": 1}

    data = data.drop_duplicates()
    data = data.loc[data["loan_status"].isin(status_mapping)].copy()
    data["target"] = data["loan_status"].map(status_mapping)

    data = data.drop(columns=DOMAIN_DROP_COLS, errors="ignore")

    return clean_inputs(data)