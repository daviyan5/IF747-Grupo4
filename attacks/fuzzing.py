import can
import time
import random
import argparse
import csv
from pathlib import Path

SLEEP_MIN = 0.05
SLEEP_MAX = 0.5


def fuzzing_attack(channel="vcan0", output_file="fuzzing_log.csv"):
    print(f"Starting Fuzzing attack on {channel}...")
    print(f"Logging sent messages to {output_file}")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with output_path.open("w", newline="") as f, can.interface.Bus(
            channel=channel, interface="socketcan"
        ) as bus:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Timestamp",
                    "ID (HEX)",
                    "Message Name",
                    "Data (HEX)",
                    "Decoded Signals",
                    "Legitimacy",
                ]
            )

            while True:
                random_id = random.randint(0x000, 0x7FF)
                random_dlc = random.randint(0, 8)
                random_data = [random.randint(0, 255) for _ in range(random_dlc)]
                msg = can.Message(
                    arbitration_id=random_id, data=random_data, dlc=random_dlc, is_extended_id=False
                )

                bus.send(msg)
                writer.writerow(
                    [
                        time.time(),
                        f"{msg.arbitration_id:03X}",
                        "Fuzzing_Attack",
                        msg.data.hex().upper(),
                        "{}",
                        "Malign",
                    ]
                )
                print(f"Sent fuzzing message: {msg}", end="\r")
                time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

    except KeyboardInterrupt:
        print("\nFuzzing attack interrupted by user.")
    except can.CanError as e:
        print(f"Error with CAN interface: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a Fuzzing attack on a CAN bus and log sent messages."
    )
    parser.add_argument("-c", "--channel", default="vcan0", help="CAN interface name")
    parser.add_argument(
        "-o", "--output", default="../csv/fuzzing_attack_log.csv", help="Output CSV log file"
    )
    args = parser.parse_args()

    fuzzing_attack(channel=args.channel, output_file=args.output)
