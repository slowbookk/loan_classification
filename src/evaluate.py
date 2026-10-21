from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def evaluate_model(pipeline, X_test, y_test):
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        ).tolist(),
    }

    print(classification_report(
        y_test,
        predictions,
        zero_division=0,
    ))
    print(f"Average precision: {metrics['average_precision']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print("Confusion matrix:", metrics["confusion_matrix"])

    return metrics