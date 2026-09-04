"""
Markov Melody Composer
=======================
A procedural music generator that uses a weighted Markov chain to compose
an original melody (with chord accompaniment) and renders it to a
standard MIDI file you can play in any DAW, GarageBand, VLC, etc.

Why Markov chains for music?
-----------------------------
A Markov chain models a sequence where the next state depends only on the
current state. Melodies are a great fit: a human melody-writer rarely
picks the next note completely at random -- it tends to be "close" to the
previous note (stepwise motion), occasionally leaps, and gravitates back
toward the tonic (the "home" note of the key) especially at phrase
endings. We encode those tendencies directly into a transition-probability
matrix built from music-theory rules (not scraped training data), then
sample a random walk through it. This is the same core idea used by much
more sophisticated generative-music AI systems, just with hand-designed
(rather than learned) transition weights.

No API key or internet connection required.

Usage
-----
    python melody_composer.py --key C --scale major --bars 16 --tempo 100 \
        --output my_song.mid

    python melody_composer.py --key A --scale minor_pentatonic --bars 8 \
        --seed 42 --output blues_riff.mid
"""

import argparse
import random
from dataclasses import dataclass, field

try:
    import mido
    from mido import Message, MidiFile, MidiTrack, MetaMessage
except ImportError:
    raise SystemExit(
        "Missing dependency 'mido'. Install with: pip install -r requirements.txt"
    )

# ---------------------------------------------------------------------------
# Music theory building blocks
# ---------------------------------------------------------------------------

# Semitone offsets from the tonic (root note) for each supported scale.
SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "major_pentatonic": [0, 2, 4, 7, 9],
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
}

# Diatonic chord qualities (as scale-degree triads) used for a simple
# accompaniment progression. Index = scale degree (0-based).
MAJOR_CHORD_QUALITIES = ["maj", "min", "min", "maj", "maj", "min", "dim"]
MINOR_CHORD_QUALITIES = ["min", "dim", "maj", "min", "min", "maj", "maj"]

CHORD_INTERVALS = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Common, pleasant-sounding progressions expressed as scale-degree indices
# (0 = I, 3 = IV, 4 = V, 5 = vi ...). Picked at random per song.
PROGRESSIONS = [
    [0, 3, 4, 0],       # I - IV - V - I
    [0, 5, 3, 4],       # I - vi - IV - V
    [5, 3, 0, 4],       # vi - IV - I - V
    [0, 4, 5, 3],       # I - V - vi - IV
]


def note_name_to_number(name: str, octave: int = 4) -> int:
    """Convert e.g. 'C' or 'F#' + octave into a MIDI note number."""
    name = name.strip().upper().replace("B#", "C").replace("FLAT", "")
    if name not in NOTE_NAMES:
        raise ValueError(f"Unknown note name: {name}")
    return NOTE_NAMES.index(name) + (octave + 1) * 12


# ---------------------------------------------------------------------------
# Markov transition model for melodic motion
# ---------------------------------------------------------------------------

@dataclass
class MarkovMelodyModel:
    """
    States are *scale-degree indices* (0..len(scale)-1), not raw semitones.
    Working in scale-degree space guarantees every generated note is
    diatonic (musically "in key") no matter how the walk moves.

    Transition weights encode classic melodic-motion tendencies:
      - Strongly prefer small steps (+-1 scale degree): stepwise motion.
      - Occasionally allow a leap (+-2 or +-3): keeps things interesting.
      - Mild gravitational pull back toward scale degree 0 (the tonic),
        which grows stronger the further away we've wandered.
    """
    num_degrees: int
    step_weight: float = 6.0
    leap_weight: float = 2.0
    same_weight: float = 1.0
    tonic_pull: float = 0.15

    def transition_weights(self, current_degree: int) -> list:
        weights = []
        for candidate in range(self.num_degrees):
            distance = candidate - current_degree
            abs_dist = abs(distance)
            if abs_dist == 0:
                w = self.same_weight
            elif abs_dist == 1:
                w = self.step_weight
            elif abs_dist in (2, 3):
                w = self.leap_weight
            else:
                w = self.leap_weight * 0.3  # big leaps are rare

            # Pull toward the tonic (degree 0): the farther the candidate
            # is from home, the more we discount it; moving toward 0 gets
            # a small bonus proportional to how far we currently are.
            pull_bonus = 0.0
            if current_degree != 0:
                moving_toward_tonic = abs(current_degree) > abs(current_degree + distance)
                if moving_toward_tonic:
                    pull_bonus = self.tonic_pull * abs(current_degree)
            weights.append(max(w + pull_bonus, 0.01))
        return weights

    def next_degree(self, current_degree: int, rng: random.Random) -> int:
        weights = self.transition_weights(current_degree)
        return rng.choices(range(self.num_degrees), weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Rhythm generation (independent weighted-random choice per note)
# ---------------------------------------------------------------------------

# Duration given in MIDI ticks-per-beat multiples; here expressed as
# fractions of a quarter note. Weighted so quarters/eighths dominate,
# like most real melodies.
RHYTHM_CHOICES = [
    (0.25, 3),   # sixteenth note
    (0.5, 6),    # eighth note
    (1.0, 8),    # quarter note
    (1.5, 2),    # dotted quarter
    (2.0, 2),    # half note
]


def random_duration(rng: random.Random) -> float:
    values = [v for v, _ in RHYTHM_CHOICES]
    weights = [w for _, w in RHYTHM_CHOICES]
    return rng.choices(values, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Composition: turn the Markov walk + rhythm into concrete note events
# ---------------------------------------------------------------------------

@dataclass
class NoteEvent:
    pitch: int          # MIDI note number
    start_beat: float   # position, in quarter-note beats, from song start
    duration: float      # length, in quarter-note beats


@dataclass
class Song:
    key: str
    scale_name: str
    tempo_bpm: int
    melody: list = field(default_factory=list)   # list[NoteEvent]
    chords: list = field(default_factory=list)   # list[NoteEvent] (chord tones)
    progression_degrees: list = field(default_factory=list)


def compose_melody(key: str, scale_name: str, bars: int, tempo_bpm: int,
                    seed: int | None = None, octave: int = 4) -> Song:
    rng = random.Random(seed)
    scale = SCALES[scale_name]
    root_midi = note_name_to_number(key, octave)
    model = MarkovMelodyModel(num_degrees=len(scale))

    is_minor_family = scale_name in ("minor", "minor_pentatonic", "dorian", "blues")
    chord_qualities = MINOR_CHORD_QUALITIES if is_minor_family else MAJOR_CHORD_QUALITIES
    # For pentatonic/blues scales (fewer than 7 degrees) we still borrow the
    # 7-degree quality table conceptually, but chords are built on the
    # nearest full diatonic scale for that key so the accompaniment stays
    # harmonically sensible.
    full_scale = SCALES["minor"] if is_minor_family else SCALES["major"]

    progression = rng.choice(PROGRESSIONS)

    song = Song(key=key, scale_name=scale_name, tempo_bpm=tempo_bpm,
                progression_degrees=progression)

    beats_per_bar = 4.0
    total_beats = bars * beats_per_bar

    # --- Melody: Markov walk over scale degrees, one phrase per bar ---
    current_degree = 0
    t = 0.0
    while t < total_beats:
        # At the start of every bar, nudge the walk gently toward the
        # tonic so phrases feel resolved (classic "start/end near home").
        bar_position = t % beats_per_bar
        if bar_position == 0 and rng.random() < 0.4:
            current_degree = 0

        current_degree = model.next_degree(current_degree, rng)
        # wrap into a comfortable one-octave range with occasional octave jump
        octave_shift = 12 if rng.random() < 0.08 else 0
        pitch = root_midi + scale[current_degree % len(scale)] + octave_shift

        duration = random_duration(rng)
        duration = min(duration, total_beats - t)  # don't overshoot the end
        song.melody.append(NoteEvent(pitch=pitch, start_beat=t, duration=duration))
        t += duration

    # --- Chords: one triad per bar, cycling through the chosen progression ---
    for bar in range(bars):
        degree = progression[bar % len(progression)]
        quality = chord_qualities[degree % len(chord_qualities)]
        chord_root = root_midi - 12 + full_scale[degree % len(full_scale)]  # one octave below melody
        for interval in CHORD_INTERVALS[quality]:
            song.chords.append(NoteEvent(
                pitch=chord_root + interval,
                start_beat=bar * beats_per_bar,
                duration=beats_per_bar,
            ))

    return song


# ---------------------------------------------------------------------------
# MIDI rendering
# ---------------------------------------------------------------------------

def song_to_midi(song: Song, output_path: str, ticks_per_beat: int = 480) -> None:
    midi = MidiFile(ticks_per_beat=ticks_per_beat)

    melody_track = MidiTrack()
    chord_track = MidiTrack()
    midi.tracks.append(melody_track)
    midi.tracks.append(chord_track)

    tempo_meta = mido.bpm2tempo(song.tempo_bpm)
    melody_track.append(MetaMessage("set_tempo", tempo=tempo_meta, time=0))
    melody_track.append(MetaMessage("track_name", name="Markov Melody", time=0))
    chord_track.append(MetaMessage("track_name", name="Accompaniment", time=0))

    melody_track.append(Message("program_change", program=0, time=0))   # Acoustic Grand Piano
    chord_track.append(Message("program_change", program=48, time=0))   # Strings ensemble

    def beats_to_ticks(beats: float) -> int:
        return int(round(beats * ticks_per_beat))

    def write_track(track: MidiTrack, events: list, channel: int, velocity: int):
        # MIDI events need delta-times, so sort by start and emit
        # note_on/note_off pairs sorted by absolute tick.
        abs_events = []
        for ev in events:
            start_tick = beats_to_ticks(ev.start_beat)
            end_tick = beats_to_ticks(ev.start_beat + ev.duration)
            abs_events.append((start_tick, "on", ev.pitch))
            abs_events.append((end_tick, "off", ev.pitch))
        abs_events.sort(key=lambda x: (x[0], x[1] == "on"))

        last_tick = 0
        for tick, kind, pitch in abs_events:
            delta = tick - last_tick
            last_tick = tick
            msg_type = "note_on" if kind == "on" else "note_off"
            vel = velocity if kind == "on" else 0
            track.append(Message(msg_type, note=pitch, velocity=vel,
                                  time=delta, channel=channel))

    write_track(melody_track, song.melody, channel=0, velocity=90)
    write_track(chord_track, song.chords, channel=1, velocity=55)

    midi.save(output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate an original melody with a Markov chain and export it as MIDI.")
    parser.add_argument("--key", default="C", help="Root note, e.g. C, D#, F (default: C)")
    parser.add_argument("--scale", default="major", choices=list(SCALES.keys()), help="Scale to compose in (default: major)")
    parser.add_argument("--bars", type=int, default=16, help="Number of 4/4 bars to generate (default: 16)")
    parser.add_argument("--tempo", type=int, default=100, help="Tempo in BPM (default: 100)")
    parser.add_argument("--octave", type=int, default=4, help="Base octave for the melody (default: 4)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible output")
    parser.add_argument("--output", default="melody.mid", help="Output MIDI file path (default: melody.mid)")
    args = parser.parse_args()

    song = compose_melody(
        key=args.key,
        scale_name=args.scale,
        bars=args.bars,
        tempo_bpm=args.tempo,
        seed=args.seed,
        octave=args.octave,
    )
    song_to_midi(song, args.output)

    degree_names = ["I", "ii", "iii", "IV", "V", "vi", "vii"]
    prog_str = " - ".join(degree_names[d] for d in song.progression_degrees)

    print("Markov Melody Composer")
    print("=======================")
    print(f"Key:           {args.key} {args.scale}")
    print(f"Tempo:         {args.tempo} BPM")
    print(f"Bars:          {args.bars}")
    print(f"Notes written: {len(song.melody)} melody notes, {len(song.chords)} chord tones")
    print(f"Chord loop:    {prog_str}")
    print(f"Saved to:      {args.output}")


if __name__ == "__main__":
    main()
