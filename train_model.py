import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]
CATEGORICAL_FEATURES = ["gender", "smoking_history"]
TARGET = "diabetes"
REQUIRED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]


def train(data_path: Path, model_path: Path) -> None:
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Download the CSV and pass its path with --data."
        )

    data = pd.read_csv(data_path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    features = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    target = data[TARGET]
    if target.isna().any():
        raise ValueError(f"Target column '{TARGET}' contains missing values.")
    if set(target.unique()) != {0, 1}:
        raise ValueError(f"Target column '{TARGET}' must contain both 0 and 1 labels.")

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(random_state=0, max_iter=1000)),
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_test, predictions, zero_division=0):.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Saved pipeline to {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save the diabetes prediction pipeline.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("diabetes_prediction_dataset_export.csv"),
        help="Path to the diabetes dataset CSV.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/diabetes_pipeline.joblib"),
        help="Output path for the trained pipeline.",
    )
    args = parser.parse_args()
    train(args.data, args.model)


if __name__ == "__main__":
    main()