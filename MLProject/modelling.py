import os
import json
import warnings
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

warnings.filterwarnings("ignore")


def load_preprocessed_data(data_dir):
    train_path = os.path.join(data_dir, "train_preprocessed.csv")
    test_path = os.path.join(data_dir, "test_preprocessed.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"File train tidak ditemukan: {train_path}")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"File test tidak ditemukan: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if "target" not in train_df.columns:
        raise ValueError("Kolom 'target' tidak ditemukan pada train_preprocessed.csv")

    if "target" not in test_df.columns:
        raise ValueError("Kolom 'target' tidak ditemukan pada test_preprocessed.csv")

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    return X_train, X_test, y_train, y_test


def setup_mlflow(current_dir):
    mlruns_dir = os.path.join(current_dir, "mlruns")
    os.makedirs(mlruns_dir, exist_ok=True)

    tracking_uri = Path(mlruns_dir).as_uri()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Bank Marketing CI Training")

    mlflow.sklearn.autolog(disable=True)

    print("MLflow Tracking URI:", tracking_uri)


def create_artifacts_dir(current_dir):
    artifacts_dir = os.path.join(current_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    return artifacts_dir


def save_confusion_matrix(y_test, y_pred, artifacts_dir):
    cm = confusion_matrix(y_test, y_pred)

    path = os.path.join(artifacts_dir, "confusion_matrix.png")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["no", "yes"]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return path


def save_classification_report(y_test, y_pred, artifacts_dir):
    path = os.path.join(artifacts_dir, "classification_report.txt")

    report = classification_report(y_test, y_pred)

    with open(path, "w") as file:
        file.write(report)

    return path


def save_roc_curve(y_test, y_proba, artifacts_dir):
    path = os.path.join(artifacts_dir, "roc_curve.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return path


def train_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.join(
        current_dir,
        "bank_preprocessing"
    )

    artifacts_dir = create_artifacts_dir(current_dir)

    X_train, X_test, y_train, y_test = load_preprocessed_data(data_dir)

    print("Data berhasil dimuat.")
    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)

    setup_mlflow(current_dir)

    # Menghapus run ID bawaan dari mlflow run agar script bisa membuat run baru sendiri
    if os.getenv("MLFLOW_RUN_ID"):
        print("Menghapus MLFLOW_RUN_ID dari environment:", os.getenv("MLFLOW_RUN_ID"))
        os.environ.pop("MLFLOW_RUN_ID", None)

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    with mlflow.start_run(run_name="CI_RandomForest_Training"):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)

        mlflow.log_param("model_name", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 150)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("min_samples_split", 5)
        mlflow.log_param("min_samples_leaf", 2)
        mlflow.log_param("max_features", "sqrt")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        confusion_matrix_path = save_confusion_matrix(
            y_test,
            y_pred,
            artifacts_dir
        )

        classification_report_path = save_classification_report(
            y_test,
            y_pred,
            artifacts_dir
        )

        roc_curve_path = save_roc_curve(
            y_test,
            y_proba,
            artifacts_dir
        )

        metrics_path = os.path.join(artifacts_dir, "metrics.json")

        metrics_data = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }

        with open(metrics_path, "w") as file:
            json.dump(metrics_data, file, indent=4)

        mlflow.log_artifact(confusion_matrix_path, artifact_path="evaluation")
        mlflow.log_artifact(classification_report_path, artifact_path="evaluation")
        mlflow.log_artifact(roc_curve_path, artifact_path="evaluation")
        mlflow.log_artifact(metrics_path, artifact_path="evaluation")

        input_example = X_test.head(5)
        signature = infer_signature(X_test, model.predict(X_test))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

        run_id = mlflow.active_run().info.run_id

        run_id_path = os.path.join(current_dir, "run_id.txt")

        with open(run_id_path, "w") as file:
            file.write(run_id)

        print("\nTraining selesai.")
        print("Accuracy :", accuracy)
        print("Precision:", precision)
        print("Recall   :", recall)
        print("F1-score :", f1)
        print("ROC-AUC  :", roc_auc)
        print("MLflow Run ID:", run_id)
        print("Run ID disimpan di:", run_id_path)


if __name__ == "__main__":
    train_model()