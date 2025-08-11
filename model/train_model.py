#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_fscore_support,
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.preprocessing import LabelEncoder
import joblib
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from ensemble import DiFF_RF, DiFF_RF_Wrapper


def calculate_entropy(hex_str):
    if pd.isna(hex_str):
        return 0
    hex_str = str(hex_str)
    if len(hex_str) % 2 != 0:
        hex_str = "0" + hex_str
    bytes_list = [int(hex_str[i : i + 2], 16) for i in range(0, len(hex_str), 2)]
    if not bytes_list:
        return 0
    counts = np.bincount(bytes_list, minlength=256)
    probs = counts[counts > 0] / len(bytes_list)
    return entropy(probs, base=2)


def extract_features(df):
    df["Data (HEX)"] = df["Data (HEX)"].fillna("0")
    df["ID (HEX)"] = df["ID (HEX)"].fillna("0")

    df["ID_int"] = df["ID (HEX)"].apply(lambda x: int(x, 16) if isinstance(x, str) else 0)
    df["time_diff_by_id"] = df.groupby("ID (HEX)")["Timestamp"].diff().fillna(0)

    data_bytes_list = (
        df["Data (HEX)"]
        .apply(lambda x: [int(x[i : i + 2], 16) for i in range(0, min(len(str(x)), 16), 2)])
        .tolist()
    )

    df["payload_sum"] = [sum(bl) for bl in data_bytes_list]
    df["payload_mean"] = [np.mean(bl) if bl else 0 for bl in data_bytes_list]
    df["payload_std"] = [np.std(bl) if len(bl) > 1 else 0 for bl in data_bytes_list]
    df["payload_entropy"] = df["Data (HEX)"].apply(calculate_entropy)

    df["msg_rate"] = 1 / (df["time_diff_by_id"] + 1e-6)

    window_size = 10
    df["rolling_mean_interval"] = df.groupby("ID (HEX)")["time_diff_by_id"].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )
    df["rolling_std_interval"] = (
        df.groupby("ID (HEX)")["time_diff_by_id"]
        .transform(lambda x: x.rolling(window=window_size, min_periods=1).std())
        .fillna(0)
    )

    features = [
        "ID_int",
        "time_diff_by_id",
        "payload_sum",
        "payload_mean",
        "payload_std",
        "payload_entropy",
        "msg_rate",
        "rolling_mean_interval",
        "rolling_std_interval",
    ]

    processed_df = df.copy()

    le = LabelEncoder()
    processed_df["label"] = le.fit_transform(processed_df["Legitimacy"])
    if "Benign" in le.classes_ and le.transform(["Benign"])[0] != 0:
        processed_df["label"] = 1 - processed_df["label"]

    return processed_df, features


def load_and_merge_data(base_csv_path, output_path):
    print("Step 1: Loading and merging data...")

    file_paths = {
        "Body_ECU": os.path.join(base_csv_path, "body.csv"),
        "Chassis_ECU": os.path.join(base_csv_path, "chassis.csv"),
        "PowerTrain_ECU": os.path.join(base_csv_path, "powetrain.csv"),
        "dos_attack": os.path.join(base_csv_path, "dos.csv"),
        "fuzzing_attack": os.path.join(base_csv_path, "fuzzing.csv"),
        "replay_attack": os.path.join(base_csv_path, "replay.csv"),
        "spoofing_attack": os.path.join(base_csv_path, "spoofing.csv"),
    }

    dataframes = []
    for source, path in tqdm(file_paths.items(), desc="Loading CSVs"):
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["Source"] = source
            dataframes.append(df)

    if not dataframes:
        raise FileNotFoundError("No CSV files found in the specified directory.")

    full_df = pd.concat(dataframes, ignore_index=True)
    full_df = full_df.sort_values(by="Timestamp").reset_index(drop=True)

    merged_csv_path = os.path.join(output_path, "data.csv")
    full_df.to_csv(merged_csv_path, index=False)
    print(f"Merged data sorted and saved to {merged_csv_path}")
    return merged_csv_path


def create_data_splits(merged_csv_path, output_path, train_val_ratio=0.7, val_ratio=0.2):
    print("Step 2: Creating robust chronological data splits...")
    df = pd.read_csv(merged_csv_path)

    train_val_end_idx = int(len(df) * train_val_ratio)
    train_val_df = df.iloc[:train_val_end_idx]
    test_df = df.iloc[train_val_end_idx:]

    train_end_idx = int(len(train_val_df) * (1 - val_ratio))
    train_df = train_val_df.iloc[:train_end_idx]
    val_df = train_val_df.iloc[train_end_idx:]

    train_csv_path = os.path.join(output_path, "train.csv")
    val_csv_path = os.path.join(output_path, "validation.csv")
    test_csv_path = os.path.join(output_path, "test.csv")

    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    print(f"Training data: {len(train_df)} samples")
    print(f"Validation data: {len(val_df)} samples")
    print(f"Test data: {len(test_df)} samples")

    return train_csv_path, val_csv_path, test_csv_path


def hyperparameter_search(train_df, val_df, features, n_iter=200):
    print("\nStep 3: Supervised Hyperparameter Search...")

    param_dist = {
        "n_trees": [3, 8, 16, 32, 64, 128, 256],
        "sample_size": [16, 32, 64, 128, 256],
        "alpha": np.logspace(-4, 1, 6),
    }

    X_train_benign = train_df[train_df["label"] == 0][features]

    X_search = pd.concat([X_train_benign, val_df[features]], ignore_index=True)

    y_search = pd.Series([0] * len(X_search))

    split_index = [-1] * len(X_train_benign) + [0] * len(val_df)
    pds = PredefinedSplit(test_fold=split_index)

    estimator = DiFF_RF_Wrapper()

    def validation_scorer(estimator, X, y):
        y_val_true = val_df["label"].values
        scores = estimator.decision_function(X, alpha=estimator.get_params()["alpha"])["collective"]
        return roc_auc_score(y_val_true, scores)

    random_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=validation_scorer,
        cv=pds,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    random_search.fit(X_search.values, y_search.values)

    print(f"Best Hyperparameter ROC AUC Score on Validation: {random_search.best_score_:.4f}")
    print(f"Best params: {random_search.best_params_}")
    return random_search.best_params_


def find_optimal_threshold(model, val_data, features, params):
    print("\nStep 5: Finding optimal threshold on validation set...")
    X_val = val_data[features].values
    y_val = val_data["label"].values

    scores = model.anomaly_score(X_val, alpha=params["alpha"])["collective"]

    thresholds = np.linspace(np.min(scores), np.max(scores), 1000)
    f1_scores = [f1_score(y_val, (scores >= t).astype(int)) for t in thresholds]

    best_f1 = np.max(f1_scores)
    optimal_threshold = thresholds[np.argmax(f1_scores)]

    print(f"Best F1-score on validation set: {best_f1:.4f}")
    print(f"Optimal threshold found: {optimal_threshold:.4f}")
    return optimal_threshold


def plot_confusion_matrix(y_true, y_pred, results_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malicious"],
        yticklabels=["Benign", "Malicious"],
    )
    plt.title("Confusion Matrix (Test Set)")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    cm_path = os.path.join(results_path, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()


def evaluate_accuracy_by_source(test_data_with_preds, results_path):
    print("\n--- Accuracy by Source ---")

    source_accuracy = test_data_with_preds.groupby("Source").apply(
        lambda x: accuracy_score(x["label"], x["predicted_label"]), include_groups=False
    )

    source_accuracy_df = source_accuracy.reset_index(name="accuracy")

    print(source_accuracy_df.to_string(index=False))

    accuracy_by_source_path = os.path.join(results_path, "accuracy_by_source.csv")
    source_accuracy_df.to_csv(accuracy_by_source_path, index=False)
    print(f"\nSource-specific accuracy table saved to {accuracy_by_source_path}")


def evaluate_model(model, test_data, features, results_path, model_params, threshold):
    print("\nStep 6: Evaluating final model on the unseen test set...")

    X_test = test_data[features].values
    y_true = test_data["label"].values

    scores = model.anomaly_score(X_test, alpha=model_params["alpha"])["collective"]

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, lw=2, label=f"Test Set ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) on Test Set")
    plt.legend(loc="lower right")
    roc_path = os.path.join(results_path, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()

    y_pred = (scores >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)

    results = {
        "auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "threshold": threshold,
    }

    metrics_df = pd.DataFrame([results])
    metrics_path = os.path.join(results_path, "test_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    plot_confusion_matrix(y_true, y_pred, results_path)

    test_data_with_preds = test_data.copy()
    test_data_with_preds["predicted_label"] = y_pred

    evaluate_accuracy_by_source(test_data_with_preds, results_path)

    return results


def save_hyperparameters(params, features, results_path):
    print("\nStep 7: Saving hyperparameters and artifacts...")

    hyperparams_payload = {"params": params, "features": features}

    params_path = os.path.join(results_path, "diff_rf_hyperparams.joblib")
    joblib.dump(hyperparams_payload, params_path)
    print(f"Hyperparameters and artifacts saved to {params_path}")


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_CSV_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../csv"))
    RESULTS_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../results"))

    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    try:
        merged_path = load_and_merge_data(BASE_CSV_PATH, BASE_CSV_PATH)

        train_path, val_path, test_path = create_data_splits(merged_path, BASE_CSV_PATH)

        train_df_raw = pd.read_csv(train_path)
        val_df_raw = pd.read_csv(val_path)
        test_df_raw = pd.read_csv(test_path)

        all_ids = train_df_raw["ID (HEX)"].unique()

        train_df, features = extract_features(train_df_raw)
        val_df, _ = extract_features(val_df_raw)
        test_df, _ = extract_features(test_df_raw)

        train_df_benign = train_df[train_df["label"] == 0]
        print(f"\nUsing {len(train_df_benign)} benign samples for model training.")

        best_params = hyperparameter_search(train_df_benign, val_df, features, n_iter=25)

        print("\nStep 4: Training final model with best parameters...")
        final_model = DiFF_RF(
            sample_size=best_params["sample_size"], n_trees=best_params["n_trees"]
        )
        final_model.fit(train_df_benign[features].values, n_jobs=-1)

        optimal_threshold = find_optimal_threshold(final_model, val_df, features, best_params)

        results = evaluate_model(
            final_model, test_df, features, RESULTS_PATH, best_params, optimal_threshold
        )

        print("\n--- Final Test Set Results ---")
        for metric, value in results.items():
            print(f"  {metric.replace('_', ' ').capitalize()}: {value:.4f}")

        save_hyperparameters(best_params, features, RESULTS_PATH)

    except Exception as e:
        import traceback

        traceback.print_exc()
