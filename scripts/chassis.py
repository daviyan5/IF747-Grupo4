import can
import cantools
import time
import threading
import random
import argparse
import os
import csv
from pathlib import Path
import json

ECU_NAME = "Chassis_ECU"
DBC_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../dbc/chassis.dbc")
CAN_CHANNEL = "vcan0"
SLEEP_MIN = 0.05
SLEEP_MAX = 0.5


class ChassisECU:
    def __init__(self, channel, output_file):
        self.db = cantools.database.load_file(DBC_FILE)
        self.bus = can.interface.Bus(channel=channel, interface="socketcan")
        self.is_running = True
        self.accelerator_pos = 0
        self.brake_pos = 0
        self.steering_angle = 0

        self.output_path = Path(output_file)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = self.output_path.open("w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            ["Timestamp", "ID (HEX)", "Message Name", "Data (HEX)", "Decoded Signals", "Legitimacy"]
        )
        print(f"[{ECU_NAME}] Logging sent messages to {self.output_path}")

    def _log_sent_message(self, msg, msg_name, decoded_data):
        self.csv_writer.writerow(
            [
                time.time(),
                f"{msg.arbitration_id:03X}",
                msg_name,
                msg.data.hex().upper(),
                json.dumps(decoded_data),
                "Benign",
            ]
        )

    def send_messages(self):
        print(f"[{ECU_NAME}] Starting sender thread.")
        while self.is_running:
            self.accelerator_pos = (self.accelerator_pos + random.uniform(-5, 5)) % 100
            if self.accelerator_pos < 0:
                self.accelerator_pos = 0
            self.brake_pos = (self.brake_pos + random.uniform(-10, 10)) % 100
            if self.brake_pos < 0:
                self.brake_pos = 0
            self.steering_angle += random.uniform(-15, 15)
            if self.steering_angle > 51.1:
                self.steering_angle = 51.1
            if self.steering_angle < -51.1:
                self.steering_angle = -51.1

            # --- Accelerator message ---
            accel_msg_def = self.db.get_message_by_name("Accelerator_Pedal_Operation")
            accel_data = {"Accelerator_Position": self.accelerator_pos}
            encoded_data = accel_msg_def.encode(accel_data)
            msg = can.Message(arbitration_id=accel_msg_def.frame_id, data=encoded_data)
            self.bus.send(msg)
            self._log_sent_message(msg, "Accelerator_Pedal_Operation", accel_data)

            # --- Brake message ---
            brake_msg_def = self.db.get_message_by_name("Brake_Operation_Indicator")
            brake_data = {"Brake_Percentage": self.brake_pos}
            encoded_data = brake_msg_def.encode(brake_data)
            msg = can.Message(arbitration_id=brake_msg_def.frame_id, data=encoded_data)
            self.bus.send(msg)
            self._log_sent_message(msg, "Brake_Operation_Indicator", brake_data)

            # --- Steering message ---
            steer_msg_def = self.db.get_message_by_name("Steering_Wheel_Operation")
            steer_data = {"Steering_Angle": self.steering_angle}
            encoded_data = steer_msg_def.encode(steer_data)
            msg = can.Message(arbitration_id=steer_msg_def.frame_id, data=encoded_data)
            self.bus.send(msg)
            self._log_sent_message(msg, "Steering_Wheel_Operation", steer_data)

            sleep_time = random.uniform(SLEEP_MIN, SLEEP_MAX)
            time.sleep(sleep_time)

    def receive_messages(self):
        print(f"[{ECU_NAME}] Starting receiver thread (no actions).")
        for msg in self.bus:
            if not self.is_running:
                break

    def run(self):
        sender_thread = threading.Thread(target=self.send_messages)
        receiver_thread = threading.Thread(target=self.receive_messages)
        sender_thread.start()
        receiver_thread.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"[{ECU_NAME}] Shutting down.")
            self.is_running = False
            sender_thread.join()
            self.bus.shutdown()
            receiver_thread.join()
            self.csv_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"{ECU_NAME} Simulator")
    parser.add_argument("-c", "--channel", default=CAN_CHANNEL)
    parser.add_argument("-o", "--output", default=f"../csv/{ECU_NAME.lower()}.csv")
    args = parser.parse_args()
    ecu = ChassisECU(channel=args.channel, output_file=args.output)
    ecu.run()
