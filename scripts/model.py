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

from train_model import extract_features, calculate_entropy
from ensemble import DiFF_RF

ECU_NAME = "IDS_ECU"


def listen_and_train(channel, duration, hyperparams_path, output_model_path):
    print(f"[{ECU_NAME}] Entering LISTEN & TRAIN mode.")
    print(
        f"[{ECU_NAME}] Listening for {duration} seconds to capture a baseline of benign traffic..."
    )

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
            elapsed = time.time() - start_time
            pbar.n = round(elapsed, 2)
            pbar.refresh()

    bus.shutdown()

    if not captured_messages:
        print(f"[{ECU_NAME}] CRITICAL: No messages were captured. Cannot train model. Exiting.")
        exit(1)

    print(f"\n[{ECU_NAME}] Captured {len(captured_messages)} messages. Preparing to train model...")
    df = pd.DataFrame(captured_messages)

    print(f"[{ECU_NAME}] Loading hyperparameters from {hyperparams_path}")
    artifacts = joblib.load(hyperparams_path)
    params = artifacts["params"]
    id_map = artifacts["id_map"]
    features_list = artifacts["features"]

    processed_df, _ = extract_features(df, id_map)
    X_train = processed_df[features_list].values

    print(f"[{ECU_NAME}] Training new DiFF-RF model with parameters: {params}")
    model = DiFF_RF(sample_size=params["sample_size"], n_trees=params["n_trees"])
    model.fit(X_train, n_jobs=-1)

    print(f"[{ECU_NAME}] Calculating live threshold...")
    scores = model.anomaly_score(X_train, alpha=params.get("alpha", 0.1))["collective"]
    live_threshold = np.percentile(scores, 99.9)
    print(f"[{ECU_NAME}] New threshold set to: {live_threshold:.4f}")

    final_model_payload = {
        "model": model,
        "id_map": id_map,
        "features": features_list,
        "params": params,
        "threshold": live_threshold,
    }
    joblib.dump(final_model_payload, output_model_path)
    print(f"[{ECU_NAME}] Live-trained model saved successfully to {output_model_path}")


class AnomalyDetectionECU:
    def __init__(self, channel, model_path):
        print(f"[{ECU_NAME}] Initializing DETECT mode...")
        try:
            artifacts = joblib.load(model_path)
            self.model = artifacts["model"]
            self.id_map = artifacts["id_map"]
            self.features_list = artifacts["features"]
            self.threshold = artifacts["threshold"]
            self.alpha = artifacts["params"].get("alpha", 0.1)
            print(f"[{ECU_NAME}] Live-trained model loaded successfully from {model_path}")
            print(f"[{ECU_NAME}] Using detection threshold: {self.threshold:.4f}")
        except FileNotFoundError:
            print(f"[{ECU_NAME}] CRITICAL: Model file not found at {model_path}. Exiting.")
            exit(1)

        self.bus = can.interface.Bus(channel=channel, interface="socketcan")
        self.is_running = True
        self.id_state = {}
        self.window_size = 10

    def create_feature_vector(self, msg: can.Message):
        msg_id_hex = f"{msg.arbitration_id:03X}"
        msg_data_hex = msg.data.hex().upper()
        current_time = msg.timestamp

        if msg_id_hex not in self.id_state:
            self.id_state[msg_id_hex] = {
                "last_time": None,
                "diff_history": deque(maxlen=self.window_size),
            }
        state = self.id_state[msg_id_hex]

        time_diff_by_id = 0.0
        if state["last_time"] is not None:
            time_diff_by_id = current_time - state["last_time"]
        state["diff_history"].append(time_diff_by_id)
        state["last_time"] = current_time

        diff_history = state["diff_history"]
        rolling_mean_interval = np.mean(diff_history) if diff_history else 0.0
        rolling_std_interval = np.std(diff_history) if len(diff_history) > 1 else 0.0

        id_int = self.id_map.get(msg_id_hex, -1)
        data_bytes = list(msg.data)
        payload_sum = sum(data_bytes)
        payload_mean = np.mean(data_bytes) if data_bytes else 0.0
        payload_std = np.std(data_bytes) if len(data_bytes) > 1 else 0.0
        payload_entropy = calculate_entropy(msg_data_hex)
        msg_rate = 1 / (time_diff_by_id + 1e-6)

        feature_dict = {
            "ID_int": id_int,
            "time_diff_by_id": time_diff_by_id,
            "payload_sum": payload_sum,
            "payload_mean": payload_mean,
            "payload_std": payload_std,
            "payload_entropy": payload_entropy,
            "msg_rate": msg_rate,
            "rolling_mean_interval": rolling_mean_interval,
            "rolling_std_interval": rolling_std_interval,
        }
        feature_vector = [feature_dict[feature] for feature in self.features_list]
        return np.array(feature_vector).reshape(1, -1)

    def run(self):
        print(
            f"[{ECU_NAME}] Starting real-time anomaly detection on channel {self.bus.channel_info}..."
        )
        for msg in self.bus:
            if not self.is_running:
                break
            feature_vector = self.create_feature_vector(msg)
            scores = self.model.anomaly_score(feature_vector, alpha=self.alpha)
            collective_score = scores["collective"][0]
            if collective_score >= self.threshold:
                print(
                    f"[{ECU_NAME}] ANOMALY DETECTED! "
                    f"ID: 0x{msg.arbitration_id:03X}, "
                    f"Score: {collective_score:.4f} (Threshold: {self.threshold:.4f}), "
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
        help="Duration (seconds) to train if no model exists.",
    )

    RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    HYPERPARAMS_PATH = os.path.join(RESULTS_DIR, "diff_rf_hyperparams.joblib")
    MODEL_PATH = os.path.join(RESULTS_DIR, "diff_rf_model.joblib")

    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"[{ECU_NAME}] No trained model found at {MODEL_PATH}.")
        if not os.path.exists(HYPERPARAMS_PATH):
            print(f"[{ECU_NAME}] CRITICAL: Hyperparameters file not found at {HYPERPARAMS_PATH}.")
            print(
                f"[{ECU_NAME}] Please run train_model.py first to generate hyperparameters. Exiting."
            )
            exit(1)

        listen_and_train(args.channel, args.duration, HYPERPARAMS_PATH, MODEL_PATH)
    else:
        print(f"[{ECU_NAME}] Found existing model at {MODEL_PATH}. Skipping training phase.")

    ecu = AnomalyDetectionECU(channel=args.channel, model_path=MODEL_PATH)
    try:
        ecu.run()
    except KeyboardInterrupt:
        ecu.stop()
