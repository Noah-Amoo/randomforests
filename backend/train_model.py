from pathlib import Path
from typing import TypedDict

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from generate_data import DATA_PATH, SEED

FEATURE_NAMES = [
    "annual_income",
    "loan_amount",
    "credit_score",
    "monthly_debt_payments",
    "years_employed",
]
TARGET_NAME = "defaulted"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "random_forest.joblib"


class ModelArtifact(TypedDict):
    model: RandomForestClassifier
    feature_names: list[str]


def train_model(
    dataset: pd.DataFrame,
) -> tuple[ModelArtifact, dict[str, float], list[list[int]]]:
    expected_columns = set(FEATURE_NAMES + [TARGET_NAME])
    if set(dataset.columns) != expected_columns or not dataset.columns.is_unique:
        raise ValueError(
            f"Dataset must contain exactly these columns: {FEATURE_NAMES + [TARGET_NAME]}"
        )
    if not all(pd.api.types.is_numeric_dtype(dataset[column]) for column in dataset):
        raise ValueError("Dataset columns must all be numeric.")
    if dataset.isna().any().any() or dataset.isin([float("inf"), -float("inf")]).any().any():
        raise ValueError("Dataset must not contain missing or infinite values.")
    if set(dataset[TARGET_NAME]) != {0, 1}:
        raise ValueError("The defaulted target must contain both 0 and 1, and no other labels.")
    if dataset[TARGET_NAME].value_counts().min() < 2:
        raise ValueError("Each target class needs at least two rows for a stratified split.")

    features = dataset[FEATURE_NAMES]
    target = dataset[TARGET_NAME]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=SEED, stratify=target
    )
    model = RandomForestClassifier(
        n_estimators=100, min_samples_leaf=5, random_state=SEED
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    default_class_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(x_test)[:, default_class_index]
    metrics = {
        "precision": float(precision_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()
    artifact: ModelArtifact = {"model": model, "feature_names": FEATURE_NAMES.copy()}
    return artifact, metrics, matrix


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Run generate_data.py first."
        )
    dataset = pd.read_csv(DATA_PATH)
    artifact, metrics, matrix = train_model(dataset)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    print(f"Trained Random Forest using an 80/20 stratified split (seed {SEED}).")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("Confusion matrix: rows = actual [0, 1], columns = predicted [0, 1]")
    print(f"[[{matrix[0][0]}, {matrix[0][1]}],")
    print(f" [{matrix[1][0]}, {matrix[1][1]}]]")
    print(f"Saved model and feature order to {MODEL_PATH}")
    print("Synthetic-data metrics only; not evidence of real-world lending accuracy.")


if __name__ == "__main__":
    main()
