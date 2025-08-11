#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import can
import numpy as np
import joblib
import os
import argparse
import time
import pandas as pd
from tqdm import tqdm
import sys
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../model")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../model/src")))

from train_model import extract_features
from ensemble import DiFF_RF

ECU_NAME = "IDS_ECU"


def listen_and_train(channel, duration, hyperparams_path, output_model_path):
    print(f"[{ECU_NAME}] Listening for {duration}s to capture benign traffic...")

    bus = can.interface.Bus(channel=channel, interface="socketcan")

    captured_messages = []
    start_time = time.time()

    with tqdm(total=duration, unit="s", desc="Capturing") as pbar:
        while time.time() - start_time < duration:
            msg = bus.recv(timeout=1.0)
            if msg:
                captured_messages.append(
                    {
                        "Timestamp": msg.timestamp,
                        "ID (HEX)": f"{msg.arbitration_id:03X}",
                        "Data (HEX)": msg.data.hex().upper(),
                        "Legitimacy": "Benign",
                    }
                )
            pbar.n = round(time.time() - start_time, 2)
            pbar.refresh()
    bus.shutdown()

    if not captured_messages:
        print(f"[{ECU_NAME}] CRITICAL: No messages captured. Cannot train. Exiting.")
        return

    print(f"\n[{ECU_NAME}] Captured {len(captured_messages)} messages. Training model...")
    df = pd.DataFrame(captured_messages)

    print(f"[{ECU_NAME}] Loading hyperparameters from {hyperparams_path}")
    artifacts = joblib.load(hyperparams_path)
    params = artifacts["params"]
    features_list = artifacts["features"]

    processed_df, _ = extract_features(df)
    X_train = processed_df[features_list].values

    print(f"[{ECU_NAME}] Training DiFF-RF model with params: {params}")
    model = DiFF_RF(sample_size=params["sample_size"], n_trees=params["n_trees"])
    model.fit(X_train, n_jobs=-1)

    print(f"[{ECU_NAME}] Calculating live threshold...")
    scores = model.anomaly_score(X_train, alpha=params.get("alpha", 0.1))["collective"]
    live_threshold = np.percentile(scores, 99.9)
    print(f"[{ECU_NAME}] New threshold set: {live_threshold:.4f}")

    final_model_payload = {
        "model": model,
        "features": features_list,
        "params": params,
        "threshold": live_threshold,
    }
    joblib.dump(final_model_payload, output_model_path)
    print(f"[{ECU_NAME}] Live-trained model saved to {output_model_path}")


class AnomalyDetectionECU:

    def __init__(self, channel, model_path, history_size=50):
        print(f"[{ECU_NAME}] Initializing DETECT mode...")
        self.bus = can.interface.Bus(channel=channel, interface="socketcan")
        self.is_running = True
        self.history_size = history_size

        try:
            artifacts = joblib.load(model_path)
            self.model = artifacts["model"]
            self.features_list = artifacts["features"]
            self.threshold = artifacts["threshold"]
            self.alpha = artifacts["params"].get("alpha", 0.1)
            print(f"[{ECU_NAME}] Model loaded successfully from {model_path}")
            print(f"[{ECU_NAME}] Detection threshold: {self.threshold:.4f}")
        except FileNotFoundError:
            print(f"[{ECU_NAME}] CRITICAL: Model file not found at {model_path}. Exiting.")
            exit(1)

        self.message_history = pd.DataFrame(
            columns=["Timestamp", "ID (HEX)", "Data (HEX)", "Legitimacy"]
        )

    def run(self):
        print(f"[{ECU_NAME}] Starting anomaly detection on {self.bus.channel_info}...")
        for msg in self.bus:
            if not self.is_running:
                break

            new_message_df = pd.DataFrame(
                [
                    {
                        "Timestamp": msg.timestamp,
                        "ID (HEX)": f"{msg.arbitration_id:03X}",
                        "Data (HEX)": msg.data.hex().upper(),
                        "Legitimacy": "Unknown",
                    }
                ]
            )

            self.message_history = pd.concat(
                [self.message_history, new_message_df], ignore_index=True
            )
            if len(self.message_history) > self.history_size:
                self.message_history = self.message_history.iloc[-self.history_size :]
            elif len(self.message_history) < self.history_size:
                continue

            processed_history, _ = extract_features(self.message_history.copy())

            feature_vector = processed_history[self.features_list].iloc[-1:].values

            score = self.model.anomaly_score(feature_vector, alpha=self.alpha)["collective"][0]

            if score >= self.threshold:
                print(
                    f"[{ECU_NAME}] ANOMALY! "
                    f"ID: 0x{msg.arbitration_id:03X}, "
                    f"Data: {msg.data.hex().upper()}"
                )

    def stop(self):
        self.is_running = False
        if self.bus:
            self.bus.shutdown()
        print(f"\n[{ECU_NAME}] Shutting down.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"{ECU_NAME} Simulator")
    parser.add_argument("-c", "--channel", default="vcan0", help="CAN interface to use.")
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=60.0,
    )

    RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    HYPERPARAMS_PATH = os.path.join(RESULTS_DIR, "diff_rf_hyperparams.joblib")
    MODEL_PATH = os.path.join(RESULTS_DIR, "diff_rf_model.joblib")

    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"[{ECU_NAME}] No trained model found at {MODEL_PATH}.")
        if not os.path.exists(HYPERPARAMS_PATH):
            print(f"[{ECU_NAME}] CRITICAL: Hyperparams file not found at {HYPERPARAMS_PATH}.")
            print(f"[{ECU_NAME}] Please run train_model.py first to generate them. Exiting.")
            exit(1)
        listen_and_train(args.channel, args.duration, HYPERPARAMS_PATH, MODEL_PATH)

    ecu = AnomalyDetectionECU(channel=args.channel, model_path=MODEL_PATH)
    try:
        ecu.run()
    except KeyboardInterrupt:
        ecu.stop()
