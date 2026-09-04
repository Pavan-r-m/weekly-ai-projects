# Markov Melody Composer

A procedural music generator that composes an original melody (plus a chord
accompaniment) using a **weighted Markov chain** grounded in music-theory
rules, then exports it as a standard MIDI file. It also ships a piano-roll
visualizer so you can *see* the composition without opening a DAW.

## Why this is interesting

Most "AI generates music" demos are black-box neural nets trained on large
corpora. This project goes the other direction: it builds the transition
probabilities **by hand from music theory** (stepwise motion is likely,
big leaps are rare, phrases gravitate back to the tonic) rather than
learning them from data. It's a clean, inspectable example of how a
first-order Markov chain over *scale degrees* (not raw pitches) guarantees
every note stays diatonic while still producing melodies that sound
different every run. It's the same modeling idea behind more advanced
generative-music systems, just fully transparent and dependency-light.

## Tech stack & key concepts

- **Markov chain** — first-order, weighted random walk over scale-degree
  states, with hand-tuned transition weights (stepwise bias, leap
  rarity, tonic-pull).
- **Music theory** — 6 scales (major, minor, major/minor pentatonic,
  blues, dorian), diatonic triad harmonization, 4 common chord
  progressions (I–IV–V–I, I–vi–IV–V, etc.).
- **Weighted rhythm sampling** — independent random note-duration choice
  (16th through half notes) weighted like typical melodic rhythm.
- **`mido`** — for constructing and writing General MIDI files (two
  tracks: piano melody + string accompaniment).
- **`matplotlib`** — for rendering a piano-roll plot (time vs. pitch) from
  the generated MIDI.

No API key, internet connection, or training data required — everything
is generated procedurally at run time.

## Installation

```bash
pip install -r requirements.txt
```

## How to run it

Generate a melody:

```bash
python melody_composer.py --key C --scale major --bars 16 --tempo 100 --seed 42 --output melody.mid
```

Other examples:

```bash
# A-minor pentatonic riff, 8 bars
python melody_composer.py --key A --scale minor_pentatonic --bars 8 --output blues_riff.mid

# F# blues, short 4-bar phrase
python melody_composer.py --key F# --scale blues --bars 4 --output sharp.mid
```

Visualize any generated (or other) MIDI file as a piano roll:

```bash
python visualize_melody.py --input melody.mid --output melody_pianoroll.png
```

### CLI options (`melody_composer.py`)

| Flag | Default | Description |
|---|---|---|
| `--key` | `C` | Root note (e.g. `C`, `D#`, `F#`) |
| `--scale` | `major` | `major`, `minor`, `major_pentatonic`, `minor_pentatonic`, `blues`, `dorian` |
| `--bars` | `16` | Number of 4/4 bars to generate |
| `--tempo` | `100` | Tempo in BPM |
| `--octave` | `4` | Base octave for the melody |
| `--seed` | none | Random seed, for reproducible output |
| `--output` | `melody.mid` | Output MIDI file path |

## Example output

```
$ python melody_composer.py --key C --scale major --bars 16 --tempo 100 --seed 42 --output melody.mid
Markov Melody Composer
=======================
Key:           C major
Tempo:         100 BPM
Bars:          16
Notes written: 79 melody notes, 48 chord tones
Chord loop:    I - IV - V - I
Saved to:      melody.mid
```

Running the visualizer on that file produces a piano-roll PNG showing the
melody (blue) sitting on top of the chord accompaniment (orange), with
clear stepwise motion and occasional leaps.

## How it works

1. **Scale setup** — the chosen key and scale are converted into a list
   of semitone offsets from the root (e.g. C major → `[0,2,4,5,7,9,11]`).
   The Markov chain's states are *indices into this list* (scale degrees),
   not raw MIDI pitches, so every sampled note is guaranteed to be in key.

2. **Transition weights** — for the current scale degree, each candidate
   next degree gets a weight based on: how far away it is (1 step is
   weighted highest, 2–3 is a rarer "leap", anything further is rarer
   still), plus a small bonus if moving toward it brings the melody
   closer to the tonic (degree 0). This produces natural-sounding
   phrases: mostly stepwise, occasionally leaping, tending to resolve.

3. **Random walk** — starting at the tonic, the model samples one degree
   at a time via `random.choices` weighted by the transition function
   above. At the start of each bar there's also a chance to "reset"
   toward the tonic, mimicking how real melodic phrases often begin near
   the tonic.

4. **Rhythm** — independently of pitch, each note's duration is sampled
   from a weighted set of common note values (16th, 8th, quarter, dotted
   quarter, half), biased toward quarters and eighths.

5. **Chords** — a 4-degree progression (e.g. I–IV–V–I) is picked at
   random and cycled one chord per bar. Each chord is built as a triad
   (major/minor/diminished, chosen by diatonic scale degree) rooted an
   octave below the melody.

6. **MIDI export** — `mido` builds two tracks (melody + chords), converts
   beat-based timing into MIDI delta-ticks, and writes a standard `.mid`
   file playable in any DAW, media player, or MIDI-capable app.

7. **Visualization** — `visualize_melody.py` re-parses the MIDI file,
   reconstructs `(start_time, duration, pitch)` for every note by pairing
   `note_on`/`note_off` events, and draws each as a horizontal bar on a
   time-vs-pitch axis (a "piano roll"), colored by track.
