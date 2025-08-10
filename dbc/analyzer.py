#!/usr/bin/env python3

import cantools
import argparse
import sys
from typing import Dict, Any
from colorama import init, Fore, Style

init(autoreset=True)


def format_bytes(size):
    if size == 1:
        return "1 byte"
    return f"{size} bytes"


def format_bit_range(signal):
    start = signal.start
    length = signal.length
    byte_order = signal.byte_order

    if byte_order == "little_endian":
        end = start + length - 1
        return f"Bit {start} to {end} (Little Endian)"
    else:
        return f"Bit {start}, Length {length} (Big Endian)"


def format_value_range(signal):
    if signal.minimum is not None and signal.maximum is not None:
        return f"[{signal.minimum} to {signal.maximum}]"
    elif signal.length == 1:
        return "[0 to 1]"
    else:
        max_val = (2**signal.length) - 1
        if signal.is_signed:
            min_val = -(2 ** (signal.length - 1))
            max_val = (2 ** (signal.length - 1)) - 1
            return f"[{min_val} to {max_val}]"
        return f"[0 to {max_val}]"


def print_signal_info(signal, indent="    "):
    print(f"{indent}{Fore.CYAN}Signal: {Style.BRIGHT}{signal.name}{Style.RESET_ALL}")

    print(f"{indent}  • Bit Position: {format_bit_range(signal)}")
    print(f"{indent}  • Bit Length: {signal.length} bits")

    data_type = "Signed" if signal.is_signed else "Unsigned"
    print(f"{indent}  • Data Type: {data_type}")

    if signal.scale != 1 or signal.offset != 0:
        print(f"{indent}  • Scale Factor: {signal.scale}")
        print(f"{indent}  • Offset: {signal.offset}")
        print(
            f"{indent}  • Formula: physical_value = (raw_value * {signal.scale}) + {signal.offset}"
        )

    print(f"{indent}  • Value Range: {format_value_range(signal)}")

    if signal.unit:
        print(f"{indent}  • Unit: {signal.unit}")

    if signal.receivers:
        receivers = ", ".join(signal.receivers)
        print(f"{indent}  • Receivers: {receivers}")

    if signal.choices:
        print(f"{indent}  • Value Definitions:")
        for value, name in sorted(signal.choices.items()):
            print(f"{indent}      {value}: {name}")

    if signal.comment:
        print(f"{indent}  • Description: {signal.comment}")

    print()


def print_message_info(message):
    print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Message: {Style.BRIGHT}{message.name}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")

    print(f"  • CAN ID: 0x{message.frame_id:03X} (decimal: {message.frame_id})")
    print(f"  • DLC: {format_bytes(message.length)}")
    print(f"  • Extended ID: {message.is_extended_frame}")

    if message.senders:
        senders = ", ".join(message.senders)
        print(f"  • Senders: {senders}")

    if hasattr(message, "cycle_time") and message.cycle_time:
        print(f"  • Cycle Time: {message.cycle_time} ms")

    if message.comment:
        print(f"  • Description: {message.comment}")

    if message.signals:
        print(f"\n  {Fore.MAGENTA}Signals ({len(message.signals)} total):{Style.RESET_ALL}")
        for signal in message.signals:
            print_signal_info(signal)
    else:
        print(f"  {Fore.RED}No signals defined{Style.RESET_ALL}")

    if message.signals:
        print(f"  {Fore.MAGENTA}Byte Layout:{Style.RESET_ALL}")
        print_byte_layout(message)


def print_byte_layout(message):
    byte_array = [["." for _ in range(8)] for _ in range(message.length)]

    signal_chars = {}
    char_index = 0

    for signal in message.signals:
        if signal.name:
            char = signal.name[0].upper()
        else:
            char = str(char_index % 10)
        signal_chars[signal.name] = char
        char_index += 1

        start_bit = signal.start
        length = signal.length

        if signal.byte_order == "little_endian":
            for i in range(length):
                bit_pos = start_bit + i
                byte_num = bit_pos // 8
                bit_num = bit_pos % 8
                if byte_num < message.length:
                    byte_array[byte_num][bit_num] = char
        else:
            byte_num = start_bit // 8
            bit_num = start_bit % 8
            if byte_num < message.length:
                byte_array[byte_num][bit_num] = char

    print("    Byte:  ", end="")
    for i in range(message.length):
        print(f"  {i:2d}  ", end="")
    print()

    print("    Bits:  ", end="")
    for i in range(message.length):
        print(" 76543210", end="")
    print()

    print("           ", end="")
    for i in range(message.length):
        print(" [", end="")
        for bit in reversed(byte_array[i]):
            print(bit, end="")
        print("]", end="")
    print()

    print("\n    Legend:")
    for signal in message.signals:
        char = signal_chars[signal.name]
        print(f"      {char} = {signal.name}")


def analyze_dbc_file(dbc_file):
    try:
        db = cantools.database.load_file(dbc_file)

        print(f"\n{Fore.BLUE}{'#'*60}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}# DBC File Analysis{Style.RESET_ALL}")
        print(f"{Fore.BLUE}{'#'*60}{Style.RESET_ALL}")
        print(f"\nFile: {Style.BRIGHT}{dbc_file}{Style.RESET_ALL}")

        if hasattr(db, "version"):
            print(f"Version: {db.version}")

        if db.nodes:
            print(f"\n{Fore.YELLOW}ECUs/Nodes:{Style.RESET_ALL}")
            for node in db.nodes:
                print(f"  • {node.name}")
                if node.comment:
                    print(f"    Description: {node.comment}")

        print(f"\n{Fore.YELLOW}Statistics:{Style.RESET_ALL}")
        print(f"  • Total Messages: {len(db.messages)}")
        total_signals = sum(len(msg.signals) for msg in db.messages)
        print(f"  • Total Signals: {total_signals}")

        if db.messages:
            min_id = min(msg.frame_id for msg in db.messages)
            max_id = max(msg.frame_id for msg in db.messages)
            print(f"  • ID Range: 0x{min_id:03X} to 0x{max_id:03X}")

        print(f"\n{Fore.YELLOW}Messages:{Style.RESET_ALL}")

        sorted_messages = sorted(db.messages, key=lambda x: x.frame_id)

        for message in sorted_messages:
            print_message_info(message)

        print(f"\n{Fore.BLUE}{'#'*60}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}# Analysis Complete{Style.RESET_ALL}")
        print(f"{Fore.BLUE}{'#'*60}{Style.RESET_ALL}")

        if hasattr(db, "value_tables") and db.value_tables:
            print(f"\n{Fore.YELLOW}Global Value Tables:{Style.RESET_ALL}")
            for name, table in db.value_tables.items():
                print(f"  {name}:")
                for value, description in sorted(table.items()):
                    print(f"    {value}: {description}")

        return True

    except FileNotFoundError:
        print(f"{Fore.RED}Error: File '{dbc_file}' not found{Style.RESET_ALL}")
        return False
    except Exception as e:
        print(f"{Fore.RED}Error loading DBC file: {e}{Style.RESET_ALL}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Analyze DBC files and display detailed information about CAN messages and signals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                %(prog)s chassis.dbc              # Basic analysis
                %(prog)s *.dbc                    # Analyze multiple files
                """,
    )

    parser.add_argument("dbc_file", nargs="+")
    args = parser.parse_args()

    success = True
    for dbc_file in args.dbc_file:
        if len(args.dbc_file) > 1:
            print(f"\n{Fore.CYAN}{'*'*60}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Processing: {dbc_file}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'*'*60}{Style.RESET_ALL}")

        if not analyze_dbc_file(dbc_file):
            success = False

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
