import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import FIGURE_DIR


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def save_confusion_matrix(model, X_test, y_test, model_name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=["Rejected", "Approved"],
        cmap="Blues",
        colorbar=False,
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


def save_feature_importance(model, model_name: str, top_n: int = 10) -> None:
    estimator = model.named_steps["model"]
    names = model.named_steps["preprocessor"].get_feature_names_out()

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        return

    order = np.argsort(values)[-top_n:]
    labels = [n.split("__", 1)[-1] for n in names[order]]

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    plt.barh(labels, values[order], color="#264653")
    plt.title(f"Fitur Paling Berpengaruh - {model_name}")
    plt.xlabel("Tingkat pengaruh")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "feature_importance.png", dpi=150)
    plt.close()
