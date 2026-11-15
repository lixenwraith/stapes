#!/usr/bin/env python
"""The noise bank train.py --channel draws from: two minutes of white noise as band frames, both planes, each band's
mean one, so a band keeps its own frame-to-frame spread however it is shaped, and the band shape of the real room a
phone heard, in dB about its mean; a frame whose longest window runs past either end is left out.
make_noise.py <quiet.wav> <out.npz>"""
import sys
import wave

import numpy as np

import ripmodels


def inner(e):
    """the frames whose windows lie within the recording, every one if it is too short to have such frames"""
    edge = -(-ripmodels.AHEAD // ripmodels.HOP)
    return e[edge:len(e) - edge] if len(e) > 2 * edge else e


def main():
    quiet, path = sys.argv[1], sys.argv[2]
    with wave.open(quiet) as f:
        sr = f.getframerate()
        x = np.frombuffer(f.readframes(f.getnframes()), "<i2").astype(float) / 32768
    rate = ripmodels.RATE
    x = np.interp(np.arange(int(len(x) * rate / sr)) * sr / rate, np.arange(len(x)), x)
    room = inner(ripmodels.band_energy(x)).mean(axis=0)
    room_db = 10 * np.log10(room + 1e-14)  # a floor under digital silence
    print(f"the room: mean {room_db.mean():.1f} dB, from {room_db.min():.1f} to {room_db.max():.1f} over the bands, a full-scale sine 0")
    rng = np.random.default_rng(1)
    white = inner(ripmodels.band_energy(rng.standard_normal(120 * rate)))
    white /= white.mean(axis=0, keepdims=True)
    np.savez(path, white=white.astype(np.float32), room=(room_db - room_db.mean()).astype(np.float32))
    print(f"{path}: {len(white)} frames of {white.shape[1]} values, layout {ripmodels.LAYOUT}; band spread {white.std(axis=0).min():.2f} to {white.std(axis=0).max():.2f} of its mean")


main()
