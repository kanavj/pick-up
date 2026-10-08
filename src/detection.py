"""Offline, harmonic-aware guitar note detection."""

import re
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.fft import rfft
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import nnls
from scipy.signal import correlate, find_peaks, resample_poly

from src.note import Note

FloatArray = NDArray[np.float64]
HARMONICS = 10
ANALYSIS_RATE = 22050
WINDOW = 4096
HOP = 220


@dataclass(frozen=True)
class DetectionConfig:
    min_midi: int = 35
    max_midi: int = 88
    min_amplitude: float = 0.001
    relative_threshold: float = 0.04
    release_ratio: float = 0.5
    min_duration: float = 0.18
    max_gap: float = 0.12
    harmonic_tolerance_cents: float = 35.0
    mode: Literal["polyphonic", "monophonic"] = "polyphonic"

    def __post_init__(self) -> None:
        if self.mode not in ("polyphonic", "monophonic"):
            raise ValueError("mode must be 'polyphonic' or 'monophonic'")
        if not 0 <= self.min_midi <= self.max_midi <= 127:
            raise ValueError("MIDI range must satisfy 0 <= min <= max <= 127")
        for name in ("min_amplitude", "min_duration", "harmonic_tolerance_cents"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("relative_threshold", "release_ratio"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not np.isfinite(self.max_gap) or self.max_gap < 0:
            raise ValueError("max_gap must be finite and nonnegative")


@dataclass(frozen=True)
class FrameNote:
    time: float
    midi: int
    frequency_hz: float
    amplitude: float
    rms_amplitude: float
    onset: bool = False


@dataclass(frozen=True)
class NoteEvent:
    note: str
    midi: int
    start_s: float
    end_s: float
    duration_s: float
    frequency_hz: float
    cents: float
    amplitude: float
    peak_amplitude: float
    rms_amplitude: float
    dbfs: float


def load_wav(filename: str | Path) -> tuple[FloatArray, int]:
    """Read mono/stereo PCM or floating WAV without changing its gain."""
    rate, raw = wavfile.read(filename)
    if raw.dtype == np.uint8:
        samples = (raw.astype(np.float64) - 128.0) / 128.0
    elif np.issubdtype(raw.dtype, np.signedinteger):
        # scipy left-justifies 24-bit PCM in int32.
        samples = raw.astype(np.float64) / float(-np.iinfo(raw.dtype).min)
    elif np.issubdtype(raw.dtype, np.floating):
        samples = raw.astype(np.float64)
    else:
        raise ValueError(f"Unsupported WAV sample format: {raw.dtype}")
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    _validate_signal(samples, rate)
    return samples, rate


def _validate_signal(samples: FloatArray, rate: int) -> None:
    if not isinstance(rate, (int, np.integer)) or rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("samples must be a finite, mono, one-dimensional signal")


def _prepare(samples: FloatArray, rate: int) -> FloatArray:
    _validate_signal(samples, rate)
    if samples.size == 0 or rate == ANALYSIS_RATE:
        return samples
    divisor = gcd(rate, ANALYSIS_RATE)
    return resample_poly(samples, ANALYSIS_RATE // divisor, rate // divisor)


def _spectra(samples: FloatArray, window_size: int = WINDOW):
    window = np.hanning(window_size)
    padded = np.pad(samples, (window_size // 2, window_size // 2))
    for center in range(0, len(samples), HOP):
        frame = padded[center : center + window_size]
        spectrum = np.abs(rfft((frame - frame.mean()) * window, window_size * 4))
        yield center / ANALYSIS_RATE, spectrum * (2.0 / window.sum())


def _peaks(
    spectrum: FloatArray, window_size: int = WINDOW
) -> tuple[FloatArray, FloatArray]:
    indices, _ = find_peaks(
        spectrum, height=max(0.00002, float(spectrum.max()) * 0.008), distance=3
    )
    indices = indices[(indices > 0) & (indices < len(spectrum) - 1)]
    logs = np.log(np.maximum(spectrum, np.finfo(float).tiny))
    left, middle, right = logs[indices - 1], logs[indices], logs[indices + 1]
    shift = 0.5 * (left - right) / (left - 2 * middle + right)
    frequencies = (indices + shift) * ANALYSIS_RATE / (window_size * 4)
    amplitudes = np.exp(middle - 0.25 * (left - right) * shift)
    keep = (frequencies >= 20) & (frequencies <= 6000)
    for index in range(len(frequencies)):
        nearby = (
            np.abs(frequencies - frequencies[index]) < 3 * ANALYSIS_RATE / window_size
        )
        if np.any(amplitudes[nearby] > amplitudes[index] * 8):
            keep[index] = False
    return frequencies[keep], amplitudes[keep]


def _frequency(midi: int) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def _matches(frequencies: FloatArray, fundamental: float, tolerance: float):
    targets = fundamental * np.arange(1, HARMONICS + 1)
    errors = np.abs(1200 * np.log2(frequencies[:, None] / targets))
    indices = errors.argmin(axis=0)
    matched = errors[indices, np.arange(HARMONICS)] <= tolerance
    return indices, matched


def calibrate(directory: str | Path) -> dict[int, FloatArray]:
    """Learn partial/first-partial ratios from files named e.g. 'C 3.wav'.

    An optional '__label' suffix can identify string, fret and take.
    Other filenames (including chord recordings) are deliberately excluded.
    Calibration measures timbre only; it does not restrict the detectable notes.
    """
    profiles: dict[int, FloatArray] = {}
    for path in sorted(Path(directory).glob("*.wav")):
        match = re.fullmatch(r"([A-G])([#b]?)\s*(-?\d+)(?:__.+)?", path.stem)
        if match is None:
            continue
        letter, accidental, octave = match.groups()
        pitch = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[letter]
        pitch += {"": 0, "#": 1, "b": -1}[accidental]
        midi = (int(octave) + 1) * 12 + pitch
        if not 0 <= midi <= 127:
            raise ValueError(f"Invalid calibration pitch: {path}")
        samples, rate = load_wav(path)
        observations = []
        for _, spectrum in _spectra(_prepare(samples, rate)):
            frequencies, amplitudes = _peaks(spectrum)
            if not len(frequencies):
                continue
            indices, matched = _matches(frequencies, _frequency(midi), 50)
            if not matched[0] or amplitudes[indices[0]] < 0.001:
                continue
            fundamental = frequencies[indices[0]]
            indices, matched = _matches(frequencies, fundamental, 45)
            partials = np.where(matched, amplitudes[indices], 0)
            observations.append(partials / partials[0])
        if not observations:
            raise ValueError(f"No usable pitched frames in calibration file: {path}")
        chunks = np.array_split(np.asarray(observations), min(3, len(observations)))
        shapes = np.array([np.median(chunk, axis=0) for chunk in chunks])
        profiles[midi] = (
            np.concatenate((profiles[midi], shapes)) if midi in profiles else shapes
        )
    if not profiles:
        raise ValueError(f"No isolated-note WAV files found in {directory}")
    return profiles


def analyze_frames(
    samples: FloatArray,
    sample_rate: int,
    config: DetectionConfig | None = None,
    profiles: dict[int, FloatArray] | None = None,
) -> list[FrameNote]:
    config = config or DetectionConfig()
    if config.mode == "monophonic":
        if profiles is not None:
            raise ValueError("Monophonic detection does not use calibration profiles")
        return _analyze_monophonic_frames(samples, sample_rate, config)
    profiles = {} if profiles is None else profiles
    for profile in profiles.values():
        if (
            profile.ndim != 2
            or profile.shape[1] != HARMONICS
            or profile.shape[0] == 0
            or not np.isfinite(profile).all()
            or np.any(profile < 0)
            or np.any(profile[:, 0] != 1)
        ):
            raise ValueError(
                "Profiles must contain rows of ten nonnegative ratios starting at 1"
            )
    result = []
    fallback = (1 / np.arange(1, HARMONICS + 1, dtype=float))[None, :]
    shapes_by_pitch = {}
    for midi in range(config.min_midi, config.max_midi + 1):
        if midi in profiles:
            shapes_by_pitch[midi] = profiles[midi]
        elif profiles:
            nearest = sorted(profiles, key=lambda pitch: abs(pitch - midi))[:2]
            shapes_by_pitch[midi] = np.mean(
                [np.mean(profiles[pitch], axis=0) for pitch in nearest], axis=0
            )[None, :]
        else:
            shapes_by_pitch[midi] = fallback
    for time, spectrum in _spectra(_prepare(samples, sample_rate)):
        frequencies, amplitudes = _peaks(spectrum)
        if (
            not len(frequencies)
            or amplitudes.max() < config.min_amplitude * config.release_ratio
        ):
            continue
        candidates: dict[int, int] = {}
        for index, frequency in enumerate(frequencies):
            midi = round(69 + 12 * np.log2(frequency / 440))
            if config.min_midi <= midi <= config.max_midi:
                if amplitudes[index] < config.min_amplitude * config.release_ratio:
                    continue
                indices, matched = _matches(
                    frequencies, frequency, config.harmonic_tolerance_cents
                )
                strong = matched & (amplitudes[indices] >= amplitudes[index] * 0.08)
                lower = frequencies[:index]
                ratios = frequency / lower
                harmonics = np.maximum(1, np.round(ratios))
                explained = (
                    (harmonics >= 2)
                    & (harmonics <= HARMONICS)
                    & (
                        np.abs(1200 * np.log2(ratios / harmonics))
                        < config.harmonic_tolerance_cents
                    )
                    & (amplitudes[:index] >= config.min_amplitude)
                )
                if np.any(explained) and np.count_nonzero(strong) < 3:
                    continue
                if (
                    midi not in candidates
                    or amplitudes[index] > amplitudes[candidates[midi]]
                ):
                    candidates[midi] = index
        if not candidates:
            continue
        pitches = list(candidates)
        columns = []
        for midi in pitches:
            shapes = shapes_by_pitch[midi]
            _, matched = _matches(
                frequencies,
                frequencies[candidates[midi]],
                config.harmonic_tolerance_cents,
            )
            if not profiles and np.count_nonzero(matched) == 1:
                shapes = np.eye(1, HARMONICS)
            columns.extend((midi, shape) for shape in shapes)
        matrix = np.zeros((len(frequencies) + len(columns), len(columns)))
        estimates = []
        for column, (midi, shape) in enumerate(columns):
            fundamental = frequencies[candidates[midi]]
            indices, matched = _matches(
                frequencies, fundamental, config.harmonic_tolerance_cents
            )
            matrix[indices[matched], column] = shape[matched]
            # Missing predicted partials penalize subharmonic explanations.
            matrix[len(frequencies) + column, column] = np.linalg.norm(shape[~matched])
            low = matched & (np.arange(HARMONICS) < 3)
            partial_frequencies = (
                frequencies[indices[low]] / np.arange(1, HARMONICS + 1)[low]
            )
            estimates.append(float(np.median(partial_frequencies)))
        coefficients, _ = nnls(matrix, np.pad(amplitudes, (0, len(columns))))
        # A weak residual at a known lower note's harmonic is not enough
        # evidence for an extra, uncalibrated note.
        totals = {
            midi: sum(
                coefficients[i] for i, (pitch, _) in enumerate(columns) if pitch == midi
            )
            for midi in pitches
        }
        rejected = set()
        for midi in pitches:
            if midi in profiles or totals[midi] >= amplitudes[candidates[midi]] * 0.5:
                continue
            for lower in pitches:
                if (
                    lower >= midi
                    or totals[lower] < config.min_amplitude * config.release_ratio
                ):
                    continue
                ratio = frequencies[candidates[midi]] / frequencies[candidates[lower]]
                harmonic = round(ratio)
                if (
                    2 <= harmonic <= HARMONICS
                    and abs(1200 * np.log2(ratio / harmonic))
                    < config.harmonic_tolerance_cents
                ):
                    rejected.add(midi)
                    break
        if rejected:
            keep = [i for i, (midi, _) in enumerate(columns) if midi not in rejected]
            coefficients[:] = 0
            if keep:
                coefficients[keep], _ = nnls(
                    matrix[:, keep], np.pad(amplitudes, (0, len(columns)))
                )
        fitted = {}
        for midi in pitches:
            selected = [i for i, (pitch, _) in enumerate(columns) if pitch == midi]
            amplitude = float(coefficients[selected].sum())
            partials = sum(
                (coefficients[i] * columns[i][1] for i in selected), np.zeros(HARMONICS)
            )
            fitted[midi] = (
                amplitude,
                float(np.linalg.norm(partials) / np.sqrt(2)),
                estimates[selected[0]],
            )
        threshold = max(
            config.min_amplitude * config.release_ratio,
            max(value[0] for value in fitted.values())
            * config.relative_threshold
            * config.release_ratio,
        )
        for midi, (amplitude, rms, frequency) in fitted.items():
            if amplitude >= threshold:
                result.append(
                    FrameNote(
                        time,
                        midi,
                        frequency,
                        float(amplitude),
                        rms,
                    )
                )
    return result


def _attack_times(signal: FloatArray, config: DetectionConfig) -> FloatArray:
    if not signal.size:
        return np.array([], dtype=float)
    padded = np.pad(signal, (0, (-len(signal)) % HOP))
    rms = np.sqrt(np.mean(padded.reshape(-1, HOP) ** 2, axis=1))
    rms = gaussian_filter1d(rms, 1)
    previous = np.pad(rms, (3, 0))[: len(rms)]
    rise = np.maximum(rms - previous, 0)
    fraction = rise / np.maximum(rms, np.finfo(float).tiny)
    score = rise * fraction
    peaks, _ = find_peaks(
        score,
        height=config.min_amplitude * 0.5,
        distance=max(1, round(0.18 * ANALYSIS_RATE / HOP)),
    )
    peaks = peaks[fraction[peaks] >= 0.5]
    times = peaks.astype(float) * HOP / ANALYSIS_RATE
    if rms[0] >= config.min_amplitude * config.release_ratio:
        times = np.r_[0.0, times[times >= 0.18]]
    return times


def _analyze_monophonic_frames(
    samples: FloatArray, sample_rate: int, config: DetectionConfig
) -> list[FrameNote]:
    signal = _prepare(samples, sample_rate)
    attacks = _attack_times(signal, config)
    window_size = max(2048, 4 * int(ANALYSIS_RATE / _frequency(config.min_midi)))
    if window_size % 2:
        window_size += 1
    padded = np.pad(signal, (window_size // 2, window_size // 2))
    min_lag = max(
        2, int(ANALYSIS_RATE / (_frequency(config.max_midi) * 2 ** (0.5 / 12)))
    )
    max_lag = min(
        window_size // 2,
        int(np.ceil(ANALYSIS_RATE / (_frequency(config.min_midi) / 2 ** (0.5 / 12)))),
    )
    comparison_size = window_size - max_lag
    result = []
    last_frequency: float | None = None
    last_time = -np.inf
    consumed_attack = -1
    for index, (time, spectrum) in enumerate(_spectra(signal, window_size)):
        attack_index = int(np.searchsorted(attacks, time + 0.04)) - 1
        recent_attack = bool(attack_index >= 0 and time - attacks[attack_index] <= 0.15)
        new_attack = recent_attack and attack_index > consumed_attack
        if time - last_time > config.max_gap + HOP / ANALYSIS_RATE:
            last_frequency = None
        if last_frequency is None and not recent_attack:
            continue
        frame = padded[index * HOP : index * HOP + window_size]
        if np.sqrt(np.mean(frame**2)) < config.min_amplitude * config.release_ratio:
            continue
        centered = frame - frame.mean()
        energy = np.concatenate(([0.0], np.cumsum(centered**2)))
        correlation = correlate(
            centered, centered[:comparison_size], mode="valid", method="fft"
        )
        difference = np.maximum(
            energy[comparison_size]
            + energy[comparison_size:]
            - energy[: max_lag + 1]
            - 2 * correlation,
            0,
        )
        cumulative = np.cumsum(difference[1:])
        normalized = np.ones_like(difference)
        np.divide(
            difference[1:] * np.arange(1, max_lag + 1),
            cumulative,
            out=normalized[1:],
            where=cumulative > np.finfo(float).tiny,
        )
        valleys, _ = find_peaks(-normalized)
        candidates = valleys[(valleys >= min_lag) & (normalized[valleys] < 0.15)]
        if not len(candidates):
            continue
        lag = int(candidates[0])
        if last_frequency is not None and not new_attack:
            continuity = valleys[
                (valleys >= min_lag)
                & (np.abs(12 * np.log2(ANALYSIS_RATE / valleys / last_frequency)) <= 2)
                & (normalized[valleys] < 0.3)
            ]
            if len(continuity):
                lag = int(continuity[np.argmin(normalized[continuity])])
        left, middle, right = difference[lag - 1 : lag + 2]
        curvature = left - 2 * middle + right
        shift = 0.5 * (left - right) / curvature if curvature > 0 else 0.0
        frequency = ANALYSIS_RATE / (lag + shift)
        if (
            last_frequency is not None
            and not new_attack
            and abs(12 * np.log2(frequency / last_frequency)) > 2
        ):
            continue
        midi = round(69 + 12 * np.log2(frequency / 440))
        if not config.min_midi <= midi <= config.max_midi:
            continue
        frequencies, amplitudes = _peaks(spectrum, window_size)
        if not len(frequencies):
            continue
        indices, matched = _matches(
            frequencies, frequency, config.harmonic_tolerance_cents
        )
        if not matched[0]:
            continue
        amplitude = float(amplitudes[indices[0]])
        if amplitude < config.min_amplitude * config.release_ratio:
            continue
        partials = np.where(matched, amplitudes[indices], 0.0)
        low = matched & (np.arange(HARMONICS) < 3)
        frequency = float(
            np.median(frequencies[indices[low]] / np.arange(1, HARMONICS + 1)[low])
        )
        # Spectral refinement can cross a semitone boundary near a bend.
        midi = round(69 + 12 * np.log2(frequency / 440))
        if config.min_midi <= midi <= config.max_midi:
            onset = new_attack
            result.append(
                FrameNote(
                    time,
                    midi,
                    frequency,
                    amplitude,
                    float(np.linalg.norm(partials) / np.sqrt(2)),
                    onset,
                )
            )
            if onset:
                consumed_attack = attack_index
            last_frequency, last_time = frequency, time
    return result


def track_notes(
    frames: list[FrameNote], duration: float, config: DetectionConfig | None = None
) -> list[NoteEvent]:
    config = config or DetectionConfig()
    if not np.isfinite(duration) or duration < 0:
        raise ValueError("duration must be finite and nonnegative")
    by_pitch: dict[int, list[FrameNote]] = {}
    frame_max: dict[float, float] = {}
    for frame in frames:
        by_pitch.setdefault(frame.midi, []).append(frame)
        frame_max[frame.time] = max(frame_max.get(frame.time, 0), frame.amplitude)
    events = []

    def finish(group: list[FrameNote]) -> None:
        audible = [
            f
            for f in group
            if f.amplitude
            >= max(config.min_amplitude, frame_max[f.time] * config.relative_threshold)
        ]
        if not audible:
            return
        if len(audible) * HOP / ANALYSIS_RATE < config.min_duration:
            return
        start = max(0.0, audible[0].time - HOP / ANALYSIS_RATE / 2)
        end = min(duration, group[-1].time + HOP / ANALYSIS_RATE / 2)
        if end - start < config.min_duration:
            return
        stable = [f for f in group if f.time >= audible[0].time]
        frequency = float(np.median([f.frequency_hz for f in audible]))
        amplitude = float(np.median([f.amplitude for f in stable]))
        midi = group[0].midi
        note = Note(NoteNumber=midi % 12, Octave=midi // 12 - 1)
        events.append(
            NoteEvent(
                note.name,
                midi,
                start,
                end,
                end - start,
                frequency,
                float(1200 * np.log2(frequency / _frequency(midi))),
                amplitude,
                max(f.amplitude for f in stable),
                float(np.median([f.rms_amplitude for f in stable])),
                float(20 * np.log10(amplitude)),
            )
        )

    streams = [frames] if config.mode == "monophonic" else by_pitch.values()
    for observations in streams:
        group: list[FrameNote] = []
        for frame in sorted(observations, key=lambda f: f.time):
            if group and (
                frame.time - group[-1].time > config.max_gap + HOP / ANALYSIS_RATE
                or frame.midi != group[-1].midi
                or frame.onset
            ):
                finish(group)
                group = []
            group.append(frame)
        if group:
            finish(group)
    return sorted(events, key=lambda event: (event.start_s, event.midi))


def detect_notes(
    samples: FloatArray,
    sample_rate: int,
    config: DetectionConfig | None = None,
    profiles: dict[int, FloatArray] | None = None,
) -> list[NoteEvent]:
    frames = analyze_frames(samples, sample_rate, config, profiles)
    return track_notes(frames, len(samples) / sample_rate, config)
