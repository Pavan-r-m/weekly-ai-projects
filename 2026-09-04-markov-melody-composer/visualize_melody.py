"""
Piano-Roll Visualizer
======================
Reads a MIDI file (e.g. one produced by melody_composer.py) and renders a
piano-roll plot: time on the x-axis, pitch on the y-axis, one horizontal
bar per note. This is a quick, dependency-light way to "see" what the
Markov chain composed without needing a DAW or audio playback.

Usage
-----
    python visualize_melody.py --input melody.mid --output melody_pianoroll.png
"""

import argparse

try:
    import mido
except ImportError:
    raise SystemExit("Missing dependency 'mido'. Install with: pip install -r requirements.txt")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ImportError:
    raise SystemExit("Missing dependency 'matplotlib'. Install with: pip install -r requirements.txt")

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

TRACK_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def midi_note_name(n: int) -> str:
    return f"{NOTE_NAMES[n % 12]}{n // 12 - 1}"


def extract_notes(midi_path: str):
    """Return {track_index: [(start_beat, duration_beats, pitch), ...]}."""
    midi = mido.MidiFile(midi_path)
    ticks_per_beat = midi.ticks_per_beat

    tracks_notes = {}
    for i, track in enumerate(midi.tracks):
        abs_tick = 0
        active = {}  # pitch -> start_tick
        notes = []
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = abs_tick
            elif msg.type in ("note_off",) or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in active:
                    start_tick = active.pop(msg.note)
                    start_beat = start_tick / ticks_per_beat
                    dur_beat = (abs_tick - start_tick) / ticks_per_beat
                    notes.append((start_beat, dur_beat, msg.note))
        if notes:
            tracks_notes[i] = notes
    return tracks_notes


def plot_piano_roll(tracks_notes: dict, output_path: str, title: str = "Markov Melody - Piano Roll"):
    fig, ax = plt.subplots(figsize=(14, 6))

    all_pitches = [p for notes in tracks_notes.values() for _, _, p in notes]
    if not all_pitches:
        raise ValueError("No notes found in MIDI file.")
    min_pitch, max_pitch = min(all_pitches) - 2, max(all_pitches) + 2

    for track_idx, notes in tracks_notes.items():
        color = TRACK_COLORS[track_idx % len(TRACK_COLORS)]
        label = f"Track {track_idx}"
        first = True
        for start_beat, dur_beat, pitch in notes:
            rect = patches.Rectangle(
                (start_beat, pitch - 0.4), max(dur_beat, 0.05), 0.8,
                facecolor=color, edgecolor="black", linewidth=0.3, alpha=0.85,
                label=label if first else None,
            )
            ax.add_patch(rect)
            first = False

    max_beat = max(s + d for notes in tracks_notes.values() for s, d, _ in notes)
    ax.set_xlim(0, max_beat + 1)
    ax.set_ylim(min_pitch, max_pitch)

    # Label the y-axis with note names at each C for readability
    y_ticks = [p for p in range(min_pitch, max_pitch + 1) if p % 12 == 0]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([midi_note_name(p) for p in y_ticks])

    ax.set_xlabel("Time (beats)")
    ax.set_ylabel("Pitch")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved piano-roll image to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Render a piano-roll PNG from a MIDI file.")
    parser.add_argument("--input", default="melody.mid", help="Path to input MIDI file")
    parser.add_argument("--output", default="melody_pianoroll.png", help="Path to output PNG")
    args = parser.parse_args()

    tracks_notes = extract_notes(args.input)
    plot_piano_roll(tracks_notes, args.output)


if __name__ == "__main__":
    main()
