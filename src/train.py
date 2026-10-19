import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import METRICS_PATH, MODEL_PATH, RANDOM_STATE, RAW_DATA_PATH
from src.evaluate import evaluate_model, save_confusion_matrix, save_feature_importance
from src.preprocessing import (
    build_preprocessor,
    clean_data,
    load_data,
    split_features_target,
    split_train_test,
)


def get_candidate_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])


def main() -> None:
    df = clean_data(load_data(RAW_DATA_PATH))
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    print(f"Data latih: {len(X_train)} | Data uji: {len(X_test)}\n")

    results = []
    fitted = {}
    for name, estimator in get_candidate_models().items():
        pipeline = make_pipeline(estimator)
        cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1").mean()
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_test, y_test)
        metrics.update(
            {
                "model": name,
                "train_accuracy": accuracy_score(y_train, pipeline.predict(X_train)),
                "cv_f1": cv_f1,
            }
        )
        results.append(metrics)
        fitted[name] = pipeline

    report = (
        pd.DataFrame(results)
        .set_index("model")[
            ["train_accuracy", "accuracy", "precision", "recall", "f1", "roc_auc", "cv_f1"]
        ]
        .round(4)
        .sort_values("cv_f1", ascending=False)
    )
    report["gap_train_test"] = (report["train_accuracy"] - report["accuracy"]).round(4)

    print(report.to_string())
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(METRICS_PATH)

    best_name = report.index[0]
    best_model = fitted[best_name]
    print(f"\nModel terbaik berdasarkan CV F1 (data latih): {best_name}")

    save_confusion_matrix(best_model, X_test, y_test, best_name)
    save_feature_importance(best_model, best_name)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Model disimpan: {MODEL_PATH}")


if __name__ == "__main__":
    main()
