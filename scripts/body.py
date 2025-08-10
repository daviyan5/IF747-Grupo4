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

ECU_NAME = "Body_ECU"
DBC_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../dbc/body.dbc")
CAN_CHANNEL = "vcan0"
SLEEP_MIN = 0.05
SLEEP_MAX = 0.5


class BodyECU:
    def __init__(self, channel, all_dbc_files, output_file):
        self.sending_db = cantools.database.load_file(DBC_FILE)
        self.full_db = cantools.database.Database()
        for file in all_dbc_files:
            self.full_db.add_dbc_file(file)

        self.bus = can.interface.Bus(channel=channel, interface="socketcan")
        self.is_running = True
        self.left_turn_active = 0
        self.right_turn_active = 0
        self.hazard_active = 0
        self.horn_active = 0
        self.doors_locked = 1

        # CSV Logging Setup
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
            # --- Turn Signal Message ---
            turn_ind_def = self.sending_db.get_message_by_name("Turn_Signal_Indicator")
            turn_data = {
                "Left_Turn_Active": self.left_turn_active,
                "Right_Turn_Active": self.right_turn_active,
                "Hazard_Active": self.hazard_active,
            }
            encoded_data = turn_ind_def.encode(turn_data)
            msg = can.Message(arbitration_id=turn_ind_def.frame_id, data=encoded_data)
            self.bus.send(msg)
            self._log_sent_message(msg, "Turn_Signal_Indicator", turn_data)

            # --- Horn Message ---
            horn_op_def = self.sending_db.get_message_by_name("Horn_Operation")
            horn_data = {"Horn_Active": self.horn_active}
            encoded_data = horn_op_def.encode(horn_data)
            msg = can.Message(arbitration_id=horn_op_def.frame_id, data=encoded_data)
            self.bus.send(msg)
            self._log_sent_message(msg, "Horn_Operation", horn_data)

            # --- Door Status Message ---
            door_stat_def = self.sending_db.get_message_by_name("Door_Status")
            door_data = {
                "Driver_Door_Open": 0,
                "Passenger_Door_Open": 0,
                "Rear_Left_Door_Open": 0,
                "Rear_Right_Door_Open": 0,
                "All_Doors_Locked": self.doors_locked,
            }
            encoded_data = door_stat_def.encode(door_data)
            msg = can.Message(arbitration_id=door_stat_def.frame_id, data=encoded_data)
            self.bus.send(msg)
            self._log_sent_message(msg, "Door_Status", door_data)

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

                if message_name == "Turn_Signal_Hazard_Switch":
                    self.left_turn_active = decoded["Left_Turn"]
                    self.right_turn_active = decoded["Right_Turn"]
                    self.hazard_active = decoded["Hazard"]
                elif message_name == "Horn_Switch":
                    self.horn_active = decoded["Horn_Pressed"]
                elif message_name == "Door_Lock_Control":
                    if decoded["Lock_All"] == 1:
                        self.doors_locked = 1
                    if decoded["Unlock_All"] == 1:
                        self.doors_locked = 0
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
    ecu = BodyECU(channel=args.channel, all_dbc_files=args.all_dbc_files, output_file=args.output)
    ecu.run()
