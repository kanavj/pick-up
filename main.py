import argparse
import csv
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path

from src.detection import DetectionConfig, NoteEvent, calibrate, detect_notes, load_wav


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect overlapping guitar notes in a WAV file."
    )
    parser.add_argument("wav", type=Path)
    parser.add_argument(
        "--calibration", type=Path, help="Directory of isolated notes, e.g. 'C 3.wav'"
    )
    parser.add_argument("--min-amplitude", type=float, default=0.001)
    parser.add_argument(
        "--mode",
        choices=("polyphonic", "monophonic"),
        default="polyphonic",
        help="Polyphonic harmonic fitting (default) or monophonic periodicity tracking",
    )
    parser.add_argument(
        "--output", type=Path, help="Write .json or .csv (default: JSON to stdout)"
    )
    args = parser.parse_args()
    if args.output and args.output.suffix.lower() not in {".json", ".csv"}:
        parser.error("--output must end in .json or .csv")
    try:
        config = DetectionConfig(min_amplitude=args.min_amplitude, mode=args.mode)
        if args.mode == "monophonic" and args.calibration:
            parser.error("--calibration is only supported in polyphonic mode")
        profiles = calibrate(args.calibration) if args.calibration else None
        samples, rate = load_wav(args.wav)
        rows = [
            asdict(event) for event in detect_notes(samples, rate, config, profiles)
        ]
        if args.output and args.output.suffix.lower() == ".csv":
            with args.output.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=[f.name for f in fields(NoteEvent)]
                )
                writer.writeheader()
                writer.writerows(rows)
        elif args.output:
            args.output.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
        else:
            json.dump(rows, sys.stdout, indent=2, allow_nan=False)
            print()
    except (OSError, ValueError) as error:
        parser.exit(2, f"pick-up: {error}\n")


if __name__ == "__main__":
    main()
