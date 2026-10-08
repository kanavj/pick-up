import tempfile
import unittest
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from src.detection import (
    DetectionConfig,
    analyze_frames,
    calibrate,
    detect_notes,
    load_wav,
    track_notes,
)

ROOT = Path(__file__).resolve().parents[1]
RATE = 22050


def tone(
    midi, cents=0.0, amplitude=0.04, start=0.3, end=1.3, duration=1.8, harmonics=10
):
    time = np.arange(round(duration * RATE)) / RATE
    frequency = 440 * 2 ** ((midi - 69) / 12 + cents / 1200)
    signal = sum(
        amplitude / harmonic * np.sin(2 * np.pi * frequency * harmonic * time)
        for harmonic in range(1, harmonics + 1)
    )
    return signal * ((time >= start) & (time < end))


class SyntheticTests(unittest.TestCase):
    def test_seven_string_low_range(self):
        self.assertEqual(DetectionConfig().min_midi, 35)
        self.assertEqual(DetectionConfig().max_midi, 88)
        for midi in range(35, 40):
            with self.subTest(midi=midi):
                events = detect_notes(tone(midi, cents=-12), RATE)
                self.assertEqual([event.midi for event in events], [midi])
                self.assertAlmostEqual(events[0].cents, -12, delta=2)
                self.assertAlmostEqual(events[0].start_s, 0.3, delta=0.09)
                self.assertAlmostEqual(events[0].end_s, 1.3, delta=0.09)

    def test_signed_cents_timing_and_amplitude(self):
        for cents in (-17, 23):
            with self.subTest(cents=cents):
                events = detect_notes(tone(69, cents), RATE)
                self.assertEqual([e.note for e in events], ["A4"])
                event = events[0]
                self.assertAlmostEqual(event.cents, cents, delta=2)
                self.assertAlmostEqual(event.start_s, 0.3, delta=0.08)
                self.assertAlmostEqual(event.end_s, 1.3, delta=0.08)
                self.assertAlmostEqual(event.amplitude, 0.04, delta=0.004)
                self.assertAlmostEqual(event.duration_s, event.end_s - event.start_s)
                self.assertAlmostEqual(event.dbfs, 20 * np.log10(event.amplitude))
                self.assertGreaterEqual(event.peak_amplitude, event.amplitude)
                self.assertTrue(
                    all(np.isfinite(v) for k, v in asdict(event).items() if k != "note")
                )

    def test_pure_sine(self):
        events = detect_notes(tone(69, amplitude=0.1, harmonics=1), RATE)
        self.assertEqual([e.note for e in events], ["A4"])
        self.assertAlmostEqual(events[0].amplitude, 0.1, delta=0.01)
        self.assertAlmostEqual(events[0].rms_amplitude, 0.1 / np.sqrt(2), delta=0.01)

    def test_uncalibrated_polyphony_with_independent_boundaries(self):
        signal = tone(50, start=0.3, end=1.1)
        signal += tone(57, start=0.5, end=1.3)
        signal += tone(65, start=0.7, end=1.5)
        events = detect_notes(signal, RATE)
        self.assertEqual({e.midi for e in events}, {50, 57, 65})
        self.assertEqual(len(events), 3)
        for event, start, end in zip(
            sorted(events, key=lambda e: e.midi), (0.3, 0.5, 0.7), (1.1, 1.3, 1.5)
        ):
            self.assertAlmostEqual(event.start_s, start, delta=0.09)
            self.assertAlmostEqual(event.end_s, end, delta=0.09)
            self.assertAlmostEqual(event.cents, 0, delta=3)

    def test_repeated_pitch_after_silence(self):
        signal = tone(69, start=0.2, end=0.6) + tone(69, start=1.0, end=1.4)
        events = detect_notes(signal, RATE)
        self.assertEqual([e.note for e in events], ["A4", "A4"])
        self.assertLess(events[0].end_s, events[1].start_s)

    def test_silence_noise_short_and_subthreshold(self):
        signals = [
            np.array([], dtype=float),
            np.zeros(RATE),
            np.zeros(1),
            np.random.default_rng(5).normal(0, 0.0001, RATE),
            tone(69, amplitude=0.0001),
            tone(69, start=0.3, end=0.33),
        ]
        for signal in signals:
            with self.subTest(length=len(signal)):
                self.assertEqual(detect_notes(signal, RATE), [])

    def test_file_boundaries_and_resampling(self):
        signal = tone(64, start=0, end=1.8)
        events = detect_notes(signal, RATE)
        self.assertEqual(len(events), 1)
        self.assertGreaterEqual(events[0].start_s, 0)
        self.assertLessEqual(events[0].end_s, 1.8)
        from scipy.signal import resample_poly

        for rate in (16000, 44100, 48000):
            resampled = resample_poly(signal, rate, RATE)
            event = detect_notes(resampled, rate)[0]
            self.assertEqual(event.midi, 64)
            self.assertAlmostEqual(event.cents, 0, delta=2)

    def test_invalid_inputs(self):
        for signal, rate in (
            (np.array([np.nan]), RATE),
            (np.zeros((2, 2)), RATE),
            (np.zeros(2), 0),
        ):
            with self.assertRaises(ValueError):
                detect_notes(signal, rate)
        for kwargs in (
            {"min_amplitude": 0},
            {"min_duration": float("nan")},
            {"release_ratio": 2},
            {"max_gap": -1},
            {"min_midi": 90, "max_midi": 40},
            {"mode": "invalid"},
        ):
            with self.assertRaises(ValueError):
                DetectionConfig(**kwargs)


class MonophonicTests(unittest.TestCase):
    def setUp(self):
        self.config = DetectionConfig(mode="monophonic")

    def test_range_cents_amplitude_and_boundaries(self):
        for midi in (35, 40, 48, 55, 64, 76, 88):
            for cents in (-23, 17):
                with self.subTest(midi=midi, cents=cents):
                    events = detect_notes(tone(midi, cents=cents), RATE, self.config)
                    self.assertEqual([event.midi for event in events], [midi])
                    self.assertAlmostEqual(events[0].cents, cents, delta=2)
                    self.assertAlmostEqual(events[0].amplitude, 0.04, delta=0.004)
                    self.assertAlmostEqual(events[0].start_s, 0.3, delta=0.05)
                    self.assertAlmostEqual(events[0].end_s, 1.3, delta=0.05)

    def test_louder_harmonics_do_not_become_notes(self):
        time = np.arange(2 * RATE) / RATE
        for midi in (35, 48, 69):
            with self.subTest(midi=midi):
                frequency = 440 * 2 ** ((midi - 69) / 12)
                signal = sum(
                    amplitude * np.sin(2 * np.pi * frequency * harmonic * time)
                    for harmonic, amplitude in enumerate((0.02, 0.16, 0.1, 0.03), 1)
                ) * ((time >= 0.3) & (time < 1.3))
                events = detect_notes(signal, RATE, self.config)
                self.assertEqual([event.midi for event in events], [midi])
                self.assertAlmostEqual(events[0].amplitude, 0.02, delta=0.004)

    def test_replucks_without_silence(self):
        time = np.arange(2 * RATE) / RATE
        envelope = np.where(
            time < 0.8,
            0.08 * np.exp(-4 * np.maximum(time - 0.2, 0)),
            0.08 * np.exp(-4 * (time - 0.8)),
        ) * ((time >= 0.2) & (time < 1.5))
        events = detect_notes(
            envelope * np.sin(2 * np.pi * 440 * time), RATE, self.config
        )
        self.assertEqual([event.note for event in events], ["A4", "A4"])
        self.assertAlmostEqual(events[1].start_s, 0.8, delta=0.05)
        self.assertLessEqual(events[0].end_s, events[1].start_s)

    def test_decay_does_not_start_an_octave_note(self):
        time = np.arange(2 * RATE) / RATE
        envelope = (time >= 0.3) & (time < 1.5)
        frequency = 261.625565
        fundamental = 0.04 * np.exp(-5 * np.maximum(time - 0.3, 0))
        signal = envelope * (
            fundamental * np.sin(2 * np.pi * frequency * time)
            + 0.08 * np.sin(4 * np.pi * frequency * time)
        )
        events = detect_notes(signal, RATE, self.config)
        self.assertEqual([event.note for event in events], ["C4"])

    def test_frame_and_event_output_is_monophonic(self):
        signal = tone(60, start=0.2, end=0.7) + tone(64, start=0.9, end=1.4)
        frames = analyze_frames(signal, RATE, self.config)
        self.assertEqual(len(frames), len({frame.time for frame in frames}))
        self.assertEqual(sum(frame.onset for frame in frames), 2)
        events = track_notes(frames, len(signal) / RATE, self.config)
        self.assertEqual([event.note for event in events], ["C4", "E4"])
        self.assertLessEqual(events[0].end_s, events[1].start_s)

    def test_silence_noise_dc_and_short_signals(self):
        for signal in (
            np.array([], dtype=float),
            np.zeros(1),
            np.zeros(RATE),
            np.ones(RATE) * 0.1,
            np.eye(1, RATE, RATE // 2).ravel(),
            np.random.default_rng(7).normal(0, 0.01, RATE),
            tone(69, amplitude=0.0001),
            tone(69, start=0.3, end=0.33),
        ):
            with self.subTest(length=len(signal)):
                self.assertEqual(detect_notes(signal, RATE, self.config), [])
        with self.assertRaisesRegex(ValueError, "does not use calibration"):
            detect_notes(np.zeros(RATE), RATE, self.config, profiles={})

    def test_file_edges_and_resampling(self):
        from scipy.signal import resample_poly

        signal = tone(35, start=0, end=1.8)
        for rate in (16000, 44100, 48000):
            with self.subTest(rate=rate):
                events = detect_notes(
                    resample_poly(signal, rate, RATE), rate, self.config
                )
                self.assertEqual([event.note for event in events], ["B1"])
                self.assertGreaterEqual(events[0].start_s, 0)
                self.assertLessEqual(events[0].end_s, 1.8)
                self.assertAlmostEqual(events[0].cents, 0, delta=2)

    def test_all_35_recorded_notes_in_order(self):
        signal, rate = load_wav(ROOT / "test_files/0 3 5 7 12.wav")
        events = detect_notes(signal, rate, self.config)
        expected = [
            root + fret
            for root in (35, 40, 45, 50, 55, 59, 64)
            for fret in (0, 3, 5, 7, 12)
        ]
        self.assertEqual([event.midi for event in events], expected)
        self.assertTrue(all(abs(event.cents) < 15 for event in events))
        self.assertTrue(
            all(left.end_s <= right.start_s for left, right in pairwise(events))
        )

    def test_original_isolated_recordings(self):
        for name in ("G 2", "C 3", "G 3", "C 4", "E 4", "G 4"):
            with self.subTest(recording=name):
                signal, rate = load_wav(ROOT / "test_files" / f"{name}.wav")
                events = detect_notes(signal, rate, self.config)
                self.assertEqual(
                    [event.note for event in events], [name.replace(" ", "")]
                )


class WavTests(unittest.TestCase):
    def test_pcm_float_and_stereo_scaling(self):
        cases = [
            (np.array([0, 128, 255], dtype=np.uint8), [-1, 0, 127 / 128]),
            (np.array([-32768, 0, 16384], dtype=np.int16), [-1, 0, 0.5]),
            (np.array([-2147483648, 0, 1073741824], dtype=np.int32), [-1, 0, 0.5]),
            (np.array([-0.25, 0, 0.5], dtype=np.float32), [-0.25, 0, 0.5]),
            (
                np.array([[1000, 3000], [-1000, 1000]], dtype=np.int16),
                [2000 / 32768, 0],
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.wav"
            for raw, expected in cases:
                wavfile.write(path, RATE, raw)
                signal, rate = load_wav(path)
                self.assertEqual(rate, RATE)
                np.testing.assert_allclose(signal, expected)

    def test_calibration_rejects_missing_data(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No isolated-note"):
                calibrate(directory)
            wavfile.write(Path(directory) / "A4.wav", RATE, np.zeros(RATE))
            with self.assertRaisesRegex(ValueError, "No usable pitched"):
                calibrate(directory)

    def test_multiple_takes_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            for label in ("A4__G-string_fret14_take1", "A4__B-string_fret10_take2"):
                wavfile.write(Path(directory) / f"{label}.wav", RATE, tone(69))
            profiles = calibrate(directory)
            self.assertEqual(set(profiles), {69})
            self.assertEqual(profiles[69].shape, (6, 10))


class RecordingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profiles = calibrate(ROOT / "test_files")

    def test_seven_string_open_b(self):
        signal, rate = load_wav(ROOT / "test_files/0 3 5 7 12.wav")
        events = detect_notes(signal[: 3 * rate], rate, profiles=self.profiles)
        low_b = [event for event in events if event.midi == 35]
        self.assertEqual(len(low_b), 1)
        self.assertEqual(low_b[0].note, "B1")
        self.assertGreater(low_b[0].duration_s, 1)
        self.assertAlmostEqual(low_b[0].frequency_hz, 61.74, delta=0.5)

    def test_all_six_isolated_recordings(self):
        for name in ("G 2", "C 3", "G 3", "C 4", "E 4", "G 4"):
            with self.subTest(recording=name):
                signal, rate = load_wav(ROOT / "test_files" / f"{name}.wav")
                events = detect_notes(signal, rate, profiles=self.profiles)
                self.assertEqual(
                    [event.note for event in events], [name.replace(" ", "")]
                )
                self.assertLess(abs(events[0].cents), 20)

    def test_chord_sequence_and_simultaneous_recall(self):
        signal, rate = load_wav(ROOT / "test_files/C major.wav")
        events = detect_notes(signal, rate, profiles=self.profiles)
        expected = ["G2", "C3", "G3", "C4", "E4", "G4"]
        picked = [event for event in events if event.start_s < 11]
        self.assertEqual([event.note for event in picked], expected)
        # Approximate attack regions measured from this recording, not
        # independent hand-labelled note-off ground truth.
        for event, start in zip(picked, (1.5, 3.0, 4.5, 6.15, 7.75, 9.4)):
            self.assertAlmostEqual(event.start_s, start, delta=0.12)
        sounding = {
            event.note for event in events if event.start_s <= 12.1 < event.end_s
        }
        self.assertTrue(set(expected).issubset(sounding))
        extras = {event.note for event in events if event.start_s >= 11} - set(expected)
        self.assertLessEqual(
            len(extras), 1, f"Unexpected extra chord pitches: {extras}"
        )

    def test_calibration_does_not_limit_pitch_vocabulary(self):
        for midi in (45, 57, 66, 72):
            with self.subTest(midi=midi):
                events = detect_notes(tone(midi), RATE, profiles=self.profiles)
                self.assertIn(midi, [event.midi for event in events])


if __name__ == "__main__":
    unittest.main()
