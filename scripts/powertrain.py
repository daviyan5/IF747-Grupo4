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

ECU_NAME = "PowerTrain_ECU"
DBC_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../dbc/powertrain.dbc")
CAN_CHANNEL = "vcan0"
SLEEP_MIN = 0.05
SLEEP_MAX = 0.5


class PowerTrainECU:
    def __init__(self, channel, all_dbc_files, output_file):
        self.sending_db = cantools.database.load_file(DBC_FILE)
        self.full_db = cantools.database.Database()
        for file in all_dbc_files:
            self.full_db.add_dbc_file(file)

        self.bus = can.interface.Bus(channel=channel, interface="socketcan")
        self.is_running = True
        self.engine_rpm = 800
        self.vehicle_speed = 0
        self.throttle_pos = 0
        self.brake_output = 0

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
            rpm_change = (
                (self.throttle_pos * 20) - (self.brake_output * 5) - (self.vehicle_speed * 2)
            )
            self.engine_rpm += rpm_change
            if self.engine_rpm < 800:
                self.engine_rpm = 800
            if self.engine_rpm > 6000:
                self.engine_rpm = 6000
            self.vehicle_speed = (self.engine_rpm - 800) / 40
            if self.vehicle_speed < 0:
                self.vehicle_speed = 0

            rpm_msg_def = self.sending_db.get_message_by_name("Engine_RPM_Speed")
            rpm_data = {
                "Engine_RPM": self.engine_rpm,
                "Vehicle_Speed": self.vehicle_speed,
            }
            encoded_data = rpm_msg_def.encode(rpm_data)
            msg = can.Message(arbitration_id=rpm_msg_def.frame_id, data=encoded_data)
            self.bus.send(msg)
            self._log_sent_message(msg, "Engine_RPM_Speed", rpm_data)

            sleep_time = random.uniform(SLEEP_MIN, SLEEP_MAX)
            time.sleep(sleep_time)

    def receive_messages(self):
        print(f"[{ECU_NAME}] Starting receiver thread.")
        for msg in self.bus:
            if not self.is_running:
                break
            try:
                decoded = self.full_db.decode_message(msg.arbitration_id, msg.data)
                message_def = self.full_db.get_message_by_frame_id(msg.arbitration_id)
                message_name = message_def.name

                if message_name == "Accelerator_Pedal_Operation":
                    self.throttle_pos = decoded["Accelerator_Position"]
                elif message_name == "Brake_Operation_Indicator":
                    self.brake_output = decoded["Brake_Percentage"]
            except Exception:
                pass

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
    parser.add_argument("all_dbc_files", nargs="+")
    parser.add_argument("-c", "--channel", default=CAN_CHANNEL)
    parser.add_argument("-o", "--output", default=f"../csv/{ECU_NAME}_log.csv")
    args = parser.parse_args()
    ecu = PowerTrainECU(
        channel=args.channel, all_dbc_files=args.all_dbc_files, output_file=args.output
    )
    ecu.run()
