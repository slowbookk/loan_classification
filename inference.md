## Key EDA Findings and Modeling Decisions

*The outcome analysis in the original notebook used about 7,100 training loans, with an overall default rate of about 19.2%. The combined EDA notebook loads the CSV sample directly, so its results may differ.*

### 1. Target and evaluation

- The target is imbalanced: about 81% Fully Paid and 19% Charged Off.
- Accuracy alone can hide poor default detection. Use **average precision (PR-AUC)** and **default recall** alongside ROC-AUC and the confusion matrix.
- `class_weight="balanced"` gives the less frequent default class more weight during model training. The prediction threshold should be chosen based on the trade-off between catching defaults and false alarms.

### 2. Numeric features

- In the earlier training analysis, `int_rate` had the strongest positive numeric association with default (correlation about 0.31), followed by `term` (about 0.23).
- `fico_range_low` had a negative association (about −0.14): lower scores were associated with more defaults.
- Recent inquiry and account-opening features had weaker positive associations. This suggests default risk is spread across several features rather than explained by one feature alone.
- `annual_inc`, `dti`, `revol_bal`, and `tot_cur_bal` had strongly right-skewed distributions. Review log transforms or capping based on the training data.
- The IQR check identifies possible extreme values; those are review candidates, not automatic errors or rows to remove.
- `loan_amnt`, `int_rate`, `total_acc`, `open_acc`, and `revol_util` showed milder skew in the original analysis.
- `fico_range_low` had a lower bound in the observed sample. Confirm unusual ranges against domain knowledge before treating values as data errors.

### 3. Categorical features

- `grade` showed a strong relationship with default rate: risk increased across grades.
- `term` also mattered. In the original analysis, 60-month loans had higher default rates than 36-month loans within the compared grades. For example, Grade C was about 30.5% versus 18.5%, and Grade D was about 46.2% versus 29.9%.
- `home_ownership` and `verification_status` showed differences in default rates. In the original training analysis, Rent was about 22.7% versus Mortgage at 15.9%; Verified was about 24.2% versus Not Verified at 14.9%. These are associations, not evidence of cause.
- `addr_state` had many levels, but state-level default rates were generally close to the overall rate, suggesting limited signal on its own.
- `application_type` was almost entirely “Individual” in the original sample, so it was a near-constant feature with limited variation.

### 4. Missing values and low-variation columns

- Missing-value percentages varied substantially across columns. Review columns with high missingness before deciding whether to exclude them.
- Several columns were dominated by zero or one category. Low variation can limit their usefulness, but a rare value may still carry predictive information.
- Keep the decisions from this review in the documented feature-drop list; don’t drop columns automatically based only on a plot.

### 5. Correlation and feature overlap

- The original analysis found 21 numeric feature pairs with absolute correlation of at least 0.70. Examples included overlapping account-opening windows such as `open_il_12m` and `open_il_24m`, and `open_rv_12m` and `open_rv_24m`.
- Some variables were highly redundant, such as funded amounts with `loan_amnt`, and `fico_range_high` with `fico_range_low`. The project’s drop list excludes selected redundant fields.
- `int_rate`, `term`, and `fico_range_low` were related but not interchangeable in the observed sample. Keep them available for modeling unless model evaluation gives a reason to remove one.
- The Grade × term comparison showed that term carried information within grade groups, supporting keeping both features.

### 6. Cleaning and preprocessing decisions

- Keep only loans with final outcomes: Fully Paid and Charged Off; map them to 0 and 1.
- Remove duplicate rows and exclude identifiers, free-text fields, selected redundant columns, and post-loan information that would not be available at application time.
- Convert `term` and `emp_length` into numeric values, and parse date fields needed for feature creation.
- Apply the chosen `dti` cap consistently in training and API input cleaning.
- Fit imputers, scalers, and encoders on training data only; reuse the fitted preprocessing in the saved model pipeline.
- Apply transformations such as `log1p` only to the selected skewed features if they are included in the implemented preprocessing pipeline.