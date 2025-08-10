import cantools
import csv
import argparse
import json
from pathlib import Path
import os
import time
import can


def log_decoded_can_messages(channel, dbc_files, log_file, bitrate=500000):
    try:
        db = cantools.database.Database()
        for file_path in dbc_files:
            db.add_dbc_file(file_path)
    except FileNotFoundError as e:
        print(f"Error: DBC file not found - {e}")
        return
    except Exception as e:
        print(f"Error loading DBC files: {e}")
        return

    print(f"Starting CAN message logging...")

    csv_headers = [
        "Timestamp",
        "ID (HEX)",
        "Message Name",
        "Data (HEX)",
        "Decoded Signals",
    ]
    output_path = Path(log_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        bus = can.interface.Bus(channel=channel, interface="socketcan")

        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()
            print(f"Logging decoded CAN messages from '{channel}' to '{log_file}'...")
            print("Press Ctrl+C to stop.")

            for msg in bus:
                timestamp = time.time()
                data_for_decode = msg.data

                try:
                    message_def = db.get_message_by_frame_id(msg.arbitration_id)

                except KeyError:
                    message_def = None

                row = {
                    "Timestamp": timestamp,
                    "ID (HEX)": f"0x{msg.arbitration_id:X}",
                    "Data (HEX)": msg.data.hex().upper(),
                }

                if message_def:
                    row["Message Name"] = message_def.name
                    try:
                        if len(data_for_decode) == message_def.length:
                            decoded_signals = db.decode_message(msg.arbitration_id, data_for_decode)
                            row["Decoded Signals"] = json.dumps(decoded_signals)
                        else:
                            row["Decoded Signals"] = "DLC_MISMATCH_FOR_DECODING"
                    except Exception:
                        row["Decoded Signals"] = "DECODING_ERROR"
                else:
                    row["Message Name"] = "Unknown"
                    row["Decoded Signals"] = ""

                writer.writerow(row)

    except OSError as e:
        print(f"Error: {e}. Is the '{channel}' interface up?")
    except KeyboardInterrupt:
        print(f"\nLogging stopped. Data saved to '{log_file}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if "bus" in locals():
            bus.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log and decode CAN bus messages to a CSV file.")
    parser.add_argument("dbc_files", nargs="+")
    parser.add_argument("-c", "--channel", default="vcan0")
    DEFAULT_OUTPUT = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../csv/decoded_can.csv"
    )
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    parser.add_argument("-b", "--bitrate", type=int, default=500000)
    args = parser.parse_args()
    log_decoded_can_messages(args.channel, args.dbc_files, args.output, args.bitrate)
