#!/usr/bin/env python3
"""Generate deterministic 16 kHz mono WAV fixtures for the voice-loop harness.

home-ops#584. Standard library only: wave, struct, math. No numpy, no external
audio, no randomness that could drift between runs (the "noise" is a fixed LCG).

Fixtures (16 kHz, 16-bit PCM, mono):
  question_en.wav  1.8 s: 0.3 s lead-in silence + 1.5 s speech-like signal
  question_es.wav  1.8 s: same shape, different f0 / rhythm / formants
  barge_in.wav     2.5 s: 1.2 s silence + 1.3 s speech

"Speech-like" means a glottal harmonic stack with formant weighting, syllable
envelopes and consonant bursts: enough energy modulation for an energy VAD to
see a turn start and end, which is all the harness needs. It is deliberately
not intelligible; the fake ASR returns a fixed transcript.

Usage:
  python3 tests/voice/make_fixtures.py [--out DIR]
"""

import argparse
import math
import os
import struct
import wave

RATE = 16000
TWO_PI = 2.0 * math.pi


class Lcg:
    """Tiny deterministic pseudo-noise source (no `random` dependency)."""

    def __init__(self, seed=12345):
        self.state = seed & 0xFFFFFFFF

    def next_float(self):
        # Numerical Recipes LCG, mapped to [-1, 1).
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state / 2147483648.0) - 1.0


def envelope(pos, length):
    """Raised-cosine syllable envelope in [0, 1]."""
    if length <= 0:
        return 0.0
    x = pos / float(length)
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return math.sin(math.pi * x) ** 2


def speech(n_samples, f0, formants, syllables, seed, peak=0.20):
    """Render a speech-like signal: harmonic stack, syllable envelope, bursts.

    f0        base fundamental in Hz
    formants  [(center_hz, bandwidth_hz), ...] used to weight harmonics
    syllables [(dur_s, gap_s, f0_scale, amp_scale), ...] repeated to fill
    """
    lcg = Lcg(seed)
    out = []
    phase = 0.0
    harmonics = list(range(1, 14))
    idx = 0
    syl_phase = 0.0
    syl_pos = 0
    syl_len = int(syllables[0][0] * RATE)
    gap_len = int(syllables[0][1] * RATE)
    in_gap = False
    gap_pos = 0

    for n in range(n_samples):
        dur, gap, f0_scale, amp_scale = syllables[idx % len(syllables)]

        if in_gap:
            env = 0.0
            gap_pos += 1
            if gap_pos >= gap_len:
                in_gap = False
                gap_pos = 0
                syl_pos = 0
                syl_phase = 0.0
                idx += 1
                dur, gap, f0_scale, amp_scale = syllables[idx % len(syllables)]
                syl_len = int(dur * RATE)
                gap_len = int(gap * RATE)
        else:
            env = envelope(syl_pos, syl_len) * amp_scale
            syl_pos += 1
            if syl_pos >= syl_len:
                in_gap = True
                gap_pos = 0

        # Intonation: a gentle arc inside each syllable.
        intonation = 1.0 + 0.05 * math.sin(TWO_PI * 0.8 * syl_phase)
        f = f0 * f0_scale * intonation
        phase += TWO_PI * f / RATE
        if phase > TWO_PI:
            phase -= TWO_PI
        syl_phase += 1.0 / RATE

        sample = 0.0
        for h in harmonics:
            fh = f * h
            if fh > RATE * 0.45:
                break
            gain = 0.0
            for center, bw in formants:
                gain += math.exp(-((fh - center) ** 2) / (2.0 * bw * bw))
            sample += (0.55 / h) * gain * math.sin(h * phase)

        # Consonant-ish burst at syllable onsets: fricative noise, fast decay.
        if syl_pos < int(0.04 * RATE) and not in_gap:
            sample += 0.5 * lcg.next_float() * (1.0 - syl_pos / (0.04 * RATE))

        out.append(peak * env * sample)

    # Normalise to a predictable peak so the fixture amplitude is stable.
    peak_abs = max((abs(v) for v in out), default=0.0)
    if peak_abs > 0.0:
        scale = peak / peak_abs
        out = [v * scale for v in out]
    return out


def silence(n_samples):
    return [0.0] * n_samples


def write_wav(path, samples, rate=RATE):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    frames = bytearray()
    for v in samples:
        clamped = max(-1.0, min(1.0, v))
        frames += struct.pack("<h", int(round(clamped * 32767.0)))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return len(frames) + 44


# A four-syllable word-ish pattern: stressed, weak, weak, stressed-short.
EN_SYLLABLES = [
    (0.19, 0.05, 1.00, 1.00),
    (0.13, 0.04, 1.06, 0.75),
    (0.12, 0.04, 0.97, 0.70),
    (0.17, 0.09, 1.03, 0.95),
]
ES_SYLLABLES = [
    (0.15, 0.04, 1.00, 0.95),
    (0.11, 0.03, 1.09, 0.80),
    (0.14, 0.03, 1.04, 0.90),
    (0.10, 0.05, 0.99, 0.70),
    (0.13, 0.10, 1.07, 0.85),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "fixtures"),
                    help="output directory (default: tests/voice/fixtures)")
    args = ap.parse_args()

    rate = RATE
    files = []

    lead = int(0.3 * rate)
    speech_len = int(1.5 * rate)
    tail = int(0.8 * rate)  # > vad_min_silence_ms so end-of-speech fires before EOF
    en = (silence(lead) +
          speech(speech_len, 118.0, [(520.0, 130.0), (1180.0, 200.0), (2600.0, 320.0)],
                 EN_SYLLABLES, seed=20251101) + silence(tail))
    files.append(("question_en.wav", en))

    es = (silence(lead) +
          speech(speech_len, 132.0, [(610.0, 120.0), (1440.0, 180.0), (2480.0, 300.0)],
                 ES_SYLLABLES, seed=20251102) + silence(tail))
    files.append(("question_es.wav", es))

    barge = (silence(int(1.2 * rate)) +
             speech(int(1.3 * rate), 126.0,
                    [(540.0, 130.0), (1300.0, 190.0), (2500.0, 310.0)],
                    EN_SYLLABLES, seed=20251103))
    files.append(("barge_in.wav", barge))

    print(f"writing fixtures to {args.out} ({rate} Hz mono 16-bit)")
    for name, samples in files:
        path = os.path.join(args.out, name)
        size = write_wav(path, samples, rate)
        print(f"  {name:18s} {len(samples) / rate:5.2f} s  {size:7d} bytes")


if __name__ == "__main__":
    main()
