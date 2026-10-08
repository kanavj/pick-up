"""Reference-aware evaluation of rendered guitar excerpts."""

from bisect import bisect_right
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import mido
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.detection import ANALYSIS_RATE, HOP, FrameNote, NoteEvent


@dataclass(frozen=True)
class PitchPoint:
    time_s: float
    midi: float


@dataclass(frozen=True)
class ReferenceNote:
    midi: int
    start_s: float
    end_s: float
    start_tick: int
    end_tick: int
    track: int
    channel: int
    velocity: int
    pitch_curve: tuple[PitchPoint, ...] = ()
    tags: tuple[str, ...] = ()
    string: int | None = None
    fret: int | None = None

    def pitch_at(self, time_s: float) -> float:
        times = [point.time_s for point in self.pitch_curve]
        index = bisect_right(times, time_s) - 1
        return self.pitch_curve[index].midi if index >= 0 else float(self.midi)


@dataclass(frozen=True)
class TempoMap:
    ticks_per_beat: int
    ticks: tuple[int, ...]
    seconds: tuple[float, ...]
    tempos: tuple[int, ...]

    def seconds_at(self, tick: int) -> float:
        if tick < 0:
            raise ValueError("MIDI ticks must be nonnegative")
        index = bisect_right(self.ticks, tick) - 1
        return self.seconds[index] + mido.tick2second(
            tick - self.ticks[index], self.ticks_per_beat, self.tempos[index]
        )


@dataclass(frozen=True)
class MidiReference:
    notes: tuple[ReferenceNote, ...]
    tempo_map: TempoMap
    track_names: tuple[str, ...]


def read_midi(filename: str | Path) -> MidiReference:
    """Read SMF 0/1, global tempos, note pairs and channel pitch-bend sensitivity."""
    midi = mido.MidiFile(filename)
    if midi.type == 2 or midi.ticks_per_beat <= 0:
        raise ValueError("Only synchronous, PPQN MIDI type 0/1 is supported")
    events = []
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        for index, message in enumerate(track):
            tick += message.time
            events.append((tick, track_index, index, message))
    events.sort(key=lambda event: event[:3])
    tempos = {0: 500000}
    for tick, _, _, message in events:
        if message.type == "set_tempo":
            if message.tempo <= 0:
                raise ValueError("MIDI tempo must be positive")
            tempos[tick] = message.tempo
    ticks = sorted(tempos)
    seconds = [0.0]
    for previous, current in zip(ticks, ticks[1:]):
        seconds.append(
            seconds[-1] + mido.tick2second(
                current - previous, midi.ticks_per_beat, tempos[previous]
            )
        )
    tempo_map = TempoMap(
        midi.ticks_per_beat, tuple(ticks), tuple(seconds),
        tuple(tempos[tick] for tick in ticks),
    )
    active: dict[tuple[int, int, int], deque[tuple[int, int]]] = defaultdict(deque)
    notes = []
    rpn = [[127, 127] for _ in range(16)]
    ranges = [[2, 0] for _ in range(16)]
    wheel = [0] * 16
    bends: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for tick, track, _, message in events:
        if message.is_meta:
            continue
        channel = message.channel
        if message.type == "note_on" and message.velocity > 0:
            active[(track, channel, message.note)].append((tick, message.velocity))
        elif message.type == "note_off" or (
            message.type == "note_on" and message.velocity == 0
        ):
            key = (track, channel, message.note)
            if not active[key]:
                raise ValueError(f"Unmatched MIDI note-off at tick {tick}: {key}")
            start, velocity = active[key].popleft()
            if tick <= start:
                raise ValueError(f"Nonpositive MIDI note duration at tick {tick}")
            notes.append(ReferenceNote(
                message.note, tempo_map.seconds_at(start), tempo_map.seconds_at(tick),
                start, tick, track, channel, velocity,
            ))
        elif message.type == "pitchwheel":
            wheel[channel] = message.pitch
            bends[channel].append((
                tick, message.pitch / 8192 * (ranges[channel][0] + ranges[channel][1] / 100),
            ))
        elif message.type == "control_change":
            control, value = message.control, message.value
            if control == 64 and value >= 64:
                raise ValueError("Sustain-pedal timing is not supported by this evaluator")
            if control in (101, 100):
                rpn[channel][control == 100] = value
            elif control in (6, 38) and rpn[channel] == [0, 0]:
                ranges[channel][control == 38] = value
                bends[channel].append((
                    tick, wheel[channel] / 8192 * (ranges[channel][0] + ranges[channel][1] / 100),
                ))
            elif control in (6, 38, 96, 97) and rpn[channel] != [127, 127]:
                raise ValueError(f"Unsupported MIDI registered parameter: {rpn[channel]}")
    if any(active.values()):
        raise ValueError("MIDI contains unterminated notes")
    result = []
    for note in notes:
        channel_bends = bends[note.channel]
        before = [offset for tick, offset in channel_bends if tick <= note.start_tick]
        curve = [PitchPoint(note.start_s, note.midi + (before[-1] if before else 0.0))]
        curve.extend(
            PitchPoint(tempo_map.seconds_at(tick), note.midi + offset)
            for tick, offset in channel_bends if note.start_tick < tick < note.end_tick
        )
        result.append(replace(note, pitch_curve=tuple(curve)))
    return MidiReference(
        tuple(sorted(result, key=lambda note: (note.start_s, note.midi, note.track))),
        tempo_map, tuple(track.name for track in midi.tracks),
    )


def case_references(case: dict, midi: MidiReference) -> list[ReferenceNote]:
    """Attach GPX semantics; use the actual MIDI performance wherever available."""
    result = []
    for annotation in case["score_notes"]:
        if case["reference_source"] == "midi":
            candidates = [
                note for note in midi.notes
                if note.track == case["midi_track"] and note.midi == annotation["midi"]
                and abs(note.start_tick - annotation["start_tick"]) <= 35
            ]
            if len(candidates) != 1:
                raise ValueError(f"Ambiguous or missing MIDI/GPX note in {case['id']}: {annotation}")
            note = candidates[0]
        elif case["reference_source"] == "gpx":
            if annotation["bend_points"]:
                raise ValueError("GPX-only bend timing requires an exported MIDI reference")
            start, end = annotation["start_tick"], annotation["end_tick"]
            note = ReferenceNote(
                annotation["midi"], midi.tempo_map.seconds_at(start),
                midi.tempo_map.seconds_at(end), start, end, case["score_track"], -1, 0,
            )
        else:
            raise ValueError(f"Unknown reference source: {case['reference_source']}")
        result.append(replace(
            note, tags=tuple(annotation["tags"]),
            string=annotation["string"], fret=annotation["fret"],
        ))
    return sorted(result, key=lambda note: (note.start_s, note.midi))


def _assignment(cost: np.ndarray, limit: float) -> list[tuple[int, int]]:
    if not cost.size:
        return []
    valid = cost <= limit
    rows, columns = linear_sum_assignment(np.where(valid, cost, 1e9))
    return [(int(row), int(column)) for row, column in zip(rows, columns) if valid[row, column]]


def _rates(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else float(fn == 0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def score_events(
    references: list[ReferenceNote],
    detections: list[NoteEvent],
    start_s: float,
    end_s: float,
    onset_tolerance_s: float = 0.075,
) -> dict:
    if not 0 <= start_s < end_s or onset_tolerance_s <= 0:
        raise ValueError("Invalid scoring interval or onset tolerance")
    expected = [
        note for note in references
        if start_s <= note.start_s < end_s and "dead_note" not in note.tags
    ]
    # Two strings at the same pitch/onset are one acoustically observable event.
    unique = {}
    for note in expected:
        key = (note.start_s, note.midi)
        if key not in unique or note.end_s > unique[key].end_s:
            unique[key] = note
    expected = list(unique.values())
    predicted = [
        note for note in detections
        if start_s - onset_tolerance_s <= note.start_s < end_s + onset_tolerance_s
    ]
    cost = np.full((len(expected), len(predicted)), np.inf)
    for i, reference in enumerate(expected):
        for j, detected in enumerate(predicted):
            if abs(detected.midi - reference.pitch_at(reference.start_s)) <= 0.5:
                cost[i, j] = abs(detected.start_s - reference.start_s)
    matches = _assignment(cost, onset_tolerance_s)
    matched_ref = {i for i, _ in matches}
    matched_pred = {j for _, j in matches}
    extras = [
        asdict(note) for j, note in enumerate(predicted)
        if j not in matched_pred and start_s <= note.start_s < end_s
    ]
    missed = [asdict(note) for i, note in enumerate(expected) if i not in matched_ref]
    onset_errors = [predicted[j].start_s - expected[i].start_s for i, j in matches]
    # Offsets at excerpt edges are censored, not artificial performance errors.
    offset_errors = [
        predicted[j].end_s - expected[i].end_s for i, j in matches
        if expected[i].end_s < end_s - onset_tolerance_s
    ]
    return {
        **_rates(len(matches), len(extras), len(missed)),
        "onset_mae_ms": float(np.mean(np.abs(onset_errors)) * 1000) if onset_errors else None,
        "onset_bias_ms": float(np.mean(onset_errors) * 1000) if onset_errors else None,
        "offset_mae_ms": float(np.mean(np.abs(offset_errors)) * 1000) if offset_errors else None,
        "offsets_scored": len(offset_errors),
        "missed": missed, "extra": extras,
    }


def score_frames(
    references: list[ReferenceNote],
    frames: list[FrameNote],
    start_s: float,
    end_s: float,
    pitch_tolerance_cents: float = 50.0,
) -> dict:
    if not 0 <= start_s < end_s or pitch_tolerance_cents <= 0:
        raise ValueError("Invalid scoring interval or pitch tolerance")
    hop = HOP / ANALYSIS_RATE
    by_time: dict[int, list[float]] = defaultdict(list)
    for frame in frames:
        index = round((frame.time - start_s) / hop)
        by_time[index].append(float(69 + 12 * np.log2(frame.frequency_hz / 440)))
    tp = fp = fn = 0
    errors = []
    for index, time in enumerate(np.arange(start_s, end_s, hop)):
        expected = sorted({
            note.pitch_at(float(time)) for note in references
            if note.start_s <= time < note.end_s and "dead_note" not in note.tags
        })
        predicted = by_time.get(index, [])
        cost = np.abs(np.subtract.outer(expected, predicted)) * 100
        matches = _assignment(cost, pitch_tolerance_cents)
        tp += len(matches)
        fp += len(predicted) - len(matches)
        fn += len(expected) - len(matches)
        if len(expected) == len(predicted) == 1:
            errors.append(float(cost[0, 0]))
    return {
        **_rates(tp, fp, fn),
        "monophonic_pitch_mae_cents": float(np.mean(errors)) if errors else None,
        "monophonic_frames_scored": len(errors),
        "pitch_tolerance_cents": pitch_tolerance_cents,
    }
