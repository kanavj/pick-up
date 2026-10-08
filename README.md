# pick-up

Offline guitar note and polyphonic transcription, using NumPy and SciPy.
Two pitch detectors share audio loading, event output, and note tracking:
monophonic analysis for single-note passages, and polyphonic analysis for chords.

## Run

Use the project's Python >=3.14 environment (`uv sync` for a new checkout):

```sh
uv run python main.py "test_files/0 3 5 7 12.wav" --mode monophonic --output single-notes.csv
uv run python main.py "test_files/C major.wav" --calibration test_files
uv run python main.py "test_files/C major.wav" --calibration test_files --output notes.csv
uv run python main.py recording.wav --output notes.json
uv run python -m unittest discover -s tests -v
```

In an existing environment, `.venv/bin/python` can replace `uv run python`.
No new dependencies are required. The first code cell in [STFT.ipynb](STFT.ipynb)
is a standalone analysis with a table, waveform, note timeline, and frame-level
cents plot. It now defaults to the seven-string recording in monophonic mode;
change its `recording` and `DetectionConfig(mode=...)` for chord analysis.
The original exploratory analysis is preserved below it.

`--mode polyphonic` remains the CLI/API default. `--mode monophonic` uses
periodicity, attack evidence and pitch continuity; it does not need calibration
or an expected-note sequence. Combining it with `--calibration` is an error,
not a silently ignored option.

In polyphonic mode, `--calibration` learns harmonic amplitude profiles from isolated-note WAVs.
It **does not whitelist pitches**: unrecorded pitches use profiles interpolated
from the nearest two recorded pitches. Without calibration, the detector uses
a generic harmonic template. Both detectors cover B1 through E6 by default,
including the low B string of a standard seven-string guitar (MIDI 35,
approximately 61.74 Hz).

## Output

Each JSON/CSV row contains:

| Field | Meaning |
| --- | --- |
| `note`, `midi` | Nearest equal-tempered note, with A4 = 440 Hz |
| `start_s`, `end_s`, `duration_s` | Seconds from the start of the recording |
| `frequency_hz` | Median estimated fundamental during above-threshold frames |
| `cents` | Signed deviation from that note: positive sharp, negative flat |
| `amplitude` | Median estimated fundamental sinusoid **peak** amplitude during the event |
| `peak_amplitude` | Maximum estimated fundamental amplitude during the event |
| `rms_amplitude` | Median estimated RMS of the note's first ten partials |
| `dbfs` | `20 * log10(amplitude)`, using full-scale sinusoid peak = 1 |

Amplitude is relative to digital full scale, **not SPL, velocity, or perceptual
loudness**. Polyphonic amplitudes/RMS are model estimates, not separated audio
measurements; calibration and phase interference affect them. WAV loading
preserves gain, handles unsigned 8-bit, signed PCM (including scipy's
left-justified 24-bit), floating-point WAVs, and averages stereo channels.
Monophonic amplitude/RMS come from measured spectral partials, not a fitted
guitar template; these are still estimates and can include nearby ringing tones.

For frame-level frequency, amplitude and modeled RMS, use
`analyze_frames()` from [src/detection.py](src/detection.py).
`detect_notes()` wraps frame analysis and event tracking; `track_notes()` can
also regroup existing frame results without repeating spectral analysis.
Frame rows additionally carry `onset`, marking an accepted monophonic attack.
Reusing frames with `track_notes()` requires the same mode/configuration.

## Algorithm and defaults

### Polyphonic (unchanged)

- Resample to 22,050 Hz; centered 4,096-sample Hann windows with 220-sample hops
  (186 ms windows, approximately 10 ms hops). Fourfold FFT padding and
  log-parabolic peak interpolation refine pitch estimates, not true resolution.
- Match up to ten partials per candidate and jointly fit nonnegative harmonic
  templates. Penalize missing partials, reject sidelobes, and suppress weak
  residual harmonics that lack independent note evidence.
- Calibration learns early/middle/late spectral shapes, accommodating decay.
  Calibrated pitches retain their measured templates when their fundamentals
  overlap another note's harmonics.
- Default onset amplitude: **0.001 (-60 dBFS)**, also at least **4%** of the
  strongest fitted note in that frame. Release threshold is half the onset
  threshold. The absolute threshold is adjustable with `--min-amplitude`.
### Monophonic

- Use 2,048-sample windows (93 ms at 22,050 Hz; enlarged for configured ranges
  below B1), with the same approximately 10 ms hop.
- Estimate pitch with a YIN-style cumulative mean normalized difference
  function. Select the first periodicity valley below 0.15; allow an existing
  pitch to continue at a valley below 0.3 to resist octave jumps during decay.
- Refine frequency and measure amplitude from spectral partials.
- Detect attacks from a smoothed short-window RMS rise. A new note needs
  a recent attack; without one, follow nearby pitch movement (up to two
  semitones per frame) but reject sudden harmonic jumps. Accepted attacks
  split repeated plucks even without silence.
- No guitar templates, fretboard labels, or expected notes enter inference.

### Shared timing and limitations

- Require **180 ms** of above-onset-threshold evidence; bridge gaps up to
  **120 ms**. Other settings, including duration, MIDI range and harmonic
  tolerance (35 cents), are available through `DetectionConfig`.

An end time means the note became undetectable at the configured threshold,
not when the player released or muted the string. Timing is window-limited:
the 10 ms hop does **not** imply 10 ms onset/offset accuracy. Synthetic tests
check boundaries within 50 ms for monophonic and 80-90 ms for polyphonic
detection, with signed tuning within 2-3 cents.
Fast notes shorter than the duration threshold are deliberately ignored.
Polyphonic replucks without a below-threshold gap may merge; decays can split.
Monophonic attack detection has a 180 ms minimum peak spacing. It can miss
very soft attacks, large legato jumps, and pitch after a long unvoiced gap.
Gradual bends can be followed in frame results, but semitone event segmentation
is not yet a bend/vibrato/technique classifier. Monophonic mode is not for chords.
Stereo with opposite-polarity channels can cancel when downmixed.

## Results on the supplied recordings

### Seven-string fretboard recording

For `0 3 5 7 12.wav`, the monophonic detector produces **35 events matching
all 35 played notes in order, with no extra events**. The previous polyphonic
baseline produced 86 events without calibration, or 58 with the six isolated
reference recordings. The polyphonic detector has not been retuned to force
one-note output.

| String | Frets 0, 3, 5, 7, 12 |
| --- | --- |
| Low B | B1, D2, E2, F#2, B2 |
| Low E | E2, G2, A2, B2, E3 |
| A | A2, C3, D3, E3, A3 |
| D | D3, F3, G3, A3, D4 |
| G | G3, A#3, C4, D4, G4 |
| B | B3, D4, E4, F#4, B4 |
| High E | E4, G4, A4, B4, E5 |

This sequence is used only for regression evaluation, not to guide detection.
Both modes still detect the six original isolated recordings correctly.

### C-major chord

With `--calibration test_files` and the default thresholds:

- Each of the six isolated recordings produces its expected note, with no
  additional events. Estimated tuning is roughly -2 to -11 cents.
- The confirmed `335553` voicing is **G2, C3, G3, C4, E4, G4**. The picked
  notes appear in that order, starting around **1.47, 2.99, 4.48, 6.13, 7.73,
  and 9.39 seconds**.
- All six pitches are detected together at 12.1 seconds in the final strum.
- A **D4 harmonic false positive** remains during the chord, and several
  decaying notes split into later events. This is not error-free transcription.

These are **in-sample tuning/regression results**, not a held-out accuracy
score. Monophonic thresholds were tuned on the fretboard recording, while the
six isolated files are calibration material for polyphonic mode. There
is no independently annotated note-off ground truth. The tests additionally
cover uncalibrated synthetic polyphony, unrecorded pitches with calibration,
positive/negative cents, amplitude, resampling, repeated notes, silence, noise,
short events, input validation, and WAV scaling. Monophonic tests additionally
cover much louder overtones, octave takeover during decay, repeated plucks
without silence, and exactly one pitch per frame with non-overlapping events.

Missing fundamentals, very quiet notes under loud harmonics, octave-related
chords, bends, different playing techniques and uncalibrated timbres remain
difficult. In particular, weak uncalibrated octave notes may be suppressed as
harmonics. More samples should improve the templates, but cannot guarantee
separation of every overlapping harmonic.

## Recording more samples

Keep mic position, interface gain, tuning, and pickup settings fixed. Disable
automatic gain/noise suppression if possible. Record about 0.5 seconds of
silence, one clean isolated pluck, and 2-4 seconds of decay. Start with frets
**0, 3, 5, 7, 12 on every string**, then fill gaps; the same pitch on different
strings has different timbre. Capture soft/medium/firm plucks separately.

Calibration filenames must begin with the sounding note/octave:

```text
E2.wav
C 3.wav
A3__G-string_fret2_medium_take1.wav
A3__D-string_fret7_soft_take2.wav
F#4__high-E-string_fret2_take1.wav
```

The optional `__label` suffix is freeform. Multiple files for the same pitch
contribute additional templates. Chord files such as `C major.wav` are excluded
from calibration by their names. Empty calibration directories or named notes
without a usable fundamental raise explicit errors.

Keep **separate takes for evaluation**, not just calibration: record single
notes, dyads (especially octaves/fifths), and several chords, picked separately
then strummed. Note tuning, fret/string, approximate attack times and deliberate
mute times. This will allow meaningful held-out pitch and timing measurements.
