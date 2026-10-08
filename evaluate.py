"""Run score-backed excerpt tests; optionally export WAV/reference pairs."""

import argparse
import csv
import hashlib
import json
import re
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from src.detection import DetectionConfig, analyze_frames, load_wav, track_notes
from src.evaluation import PitchPoint, case_references, read_midi, score_events, score_frames

DEFAULT_MANIFEST = Path(__file__).parent / "tests/fixtures/meteor_eyes.json"


def run_suite(
    manifest_path: Path,
    output_dir: Path | None = None,
    selected: list[str] | None = None,
) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported evaluation manifest version")
    source_dir = (manifest_path.parent / manifest["source_directory"]).resolve()
    for filename, expected_hash in manifest["sha256"].items():
        with (source_dir / filename).open("rb") as stream:
            actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Reference asset changed: {filename}; regenerate/review the fixture")
    midi = read_midi(source_dir / manifest["midi"])
    ids = {case["id"] for case in manifest["cases"]}
    if selected and set(selected) - ids:
        raise ValueError(f"Unknown cases: {sorted(set(selected) - ids)}")
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    audio_cache = {}
    results = []
    for case in manifest["cases"]:
        if selected and case["id"] not in selected:
            continue
        if not re.fullmatch(r"[a-z0-9-]+", case["id"]):
            raise ValueError("Case ids must contain only lowercase letters, numbers and hyphens")
        if case["audio"] not in audio_cache:
            audio_cache[case["audio"]] = load_wav(source_dir / case["audio"])
        audio, rate = audio_cache[case["audio"]]
        start = midi.tempo_map.seconds_at(case["start_tick"])
        end = midi.tempo_map.seconds_at(case["end_tick"])
        offset = case["audio_offset_s"]
        context = manifest["context_s"]
        if not 0 <= start < end or not np.isfinite(offset) or context < 0:
            raise ValueError(f"Invalid excerpt boundaries for {case['id']}")
        first = max(0, int(np.floor((start + offset - context) * rate)))
        last = min(len(audio), int(np.ceil((end + offset + context) * rate)))
        if first >= last or start + offset < 0 or end + offset > len(audio) / rate:
            raise ValueError(f"Score interval is outside the audio for {case['id']}")
        excerpt = audio[first:last]
        clip_score_start = first / rate - offset
        config = DetectionConfig(**case["detector"])
        local_frames = analyze_frames(excerpt, rate, config)
        local_events = track_notes(local_frames, len(excerpt) / rate, config)
        frames = [replace(frame, time=frame.time + clip_score_start) for frame in local_frames]
        events = [
            replace(event, start_s=event.start_s + clip_score_start, end_s=event.end_s + clip_score_start)
            for event in local_events
        ]
        references = case_references(case, midi)
        event_metrics = score_events(references, events, start, end, manifest["onset_tolerance_s"])
        frame_metrics = score_frames(references, frames, start, end, manifest["pitch_tolerance_cents"])
        primary = event_metrics if case["metric_family"] == "events" else frame_metrics
        checks = {
            metric: primary[metric] >= target for metric, target in case["targets"].items()
        }
        result = {
            "id": case["id"], "bars": case["bars"], "mode": config.mode,
            "reference_source": case["reference_source"], "techniques": case["techniques"],
            "metric_family": case["metric_family"],
            "score_start_s": start, "score_end_s": end, "audio_offset_s": offset,
            "sample_start": first, "sample_end": last, "sample_rate": rate,
            "reference_notes": len(references),
            "dead_notes_excluded": sum("dead_note" in note.tags for note in references),
            "events": event_metrics, "frames": frame_metrics,
            "targets": case["targets"], "checks": checks, "passed": all(checks.values()),
            "technique_classification": "not implemented; GPX markings are reference labels only",
        }
        results.append(result)
        if output_dir:
            raw_rate, raw = wavfile.read(source_dir / case["audio"], mmap=True)
            if raw_rate != rate:
                raise ValueError("Audio changed during extraction")
            wavfile.write(output_dir / f"{case['id']}.wav", rate, raw[first:last])
            relative_references = [
                replace(
                    note, start_s=note.start_s - clip_score_start,
                    end_s=note.end_s - clip_score_start,
                    pitch_curve=tuple(PitchPoint(point.time_s - clip_score_start, point.midi) for point in note.pitch_curve),
                )
                for note in references
            ]
            payload = {
                "source": case["audio"], "reference_source": case["reference_source"],
                "sample_start": first, "sample_end": last, "sample_rate": rate,
                "score_window_s": [start - clip_score_start, end - clip_score_start],
                "notes": [asdict(note) for note in relative_references],
                "detected": [asdict(note) for note in local_events],
            }
            (output_dir / f"{case['id']}.reference.json").write_text(
                json.dumps(payload, indent=2, allow_nan=False) + "\n"
            )
    report = {
        "schema_version": 1,
        "provenance": manifest["provenance"],
        "cases": results,
        "passed": sum(case["passed"] for case in results),
        "total": len(results),
        "all_passed": all(case["passed"] for case in results),
    }
    if output_dir:
        (output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        with (output_dir / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "case", "reference", "primary_metric", "precision", "recall", "f1",
                "onset_mae_ms", "frame_recall_50c", "mono_pitch_mae_cents", "passed",
            ])
            writer.writeheader()
            for case in results:
                primary = case[case["metric_family"]]
                writer.writerow({
                    "case": case["id"], "reference": case["reference_source"],
                    "primary_metric": case["metric_family"],
                    "precision": primary["precision"], "recall": primary["recall"], "f1": primary["f1"],
                    "onset_mae_ms": case["events"]["onset_mae_ms"],
                    "frame_recall_50c": case["frames"]["recall"],
                    "mono_pitch_mae_cents": case["frames"]["monophonic_pitch_mae_cents"],
                    "passed": case["passed"],
                })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, help="Export exact PCM excerpts, reference JSON, and metrics")
    parser.add_argument("--case", action="append", dest="cases", help="Run only this case (repeatable)")
    parser.add_argument("--check", action="store_true", help="Exit 1 if any declared accuracy target is missed")
    args = parser.parse_args()
    try:
        report = run_suite(args.manifest, args.output_dir, args.cases)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"evaluation: {error}\n")
    for case in report["cases"]:
        primary = case[case["metric_family"]]
        print(
            f"{case['id']}: precision={primary['precision']:.3f} recall={primary['recall']:.3f} "
            f"f1={primary['f1']:.3f} {'PASS' if case['passed'] else 'FAIL'}"
        )
    print(f"{report['passed']}/{report['total']} cases meet their declared targets")
    if args.check and not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
