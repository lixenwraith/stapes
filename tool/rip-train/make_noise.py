#!/usr/bin/env python
"""The noise bank train.py --channel draws from: two minutes of white noise as mel
frames, each band's mean one, so a band keeps its own frame-to-frame spread however
it is shaped; and the band shape of the real room a phone heard, in dB about its mean.
make_noise.py <quiet.wav> <out.npz>"""
import sys, wave
import numpy as np

RATE, N, HOP, BANDS = 22050, 2048, 256, 229


def weights():
    mel = lambda h: 2595 * np.log10(1 + h / 700)
    hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    edges = hz(np.linspace(mel(30), mel(RATE / 2), BANDS + 2)) * N / RATE
    bins = np.arange(N // 2 + 1)
    w = np.zeros((BANDS, N // 2 + 1))
    for b in range(BANDS):
        l, c, r = edges[b], edges[b + 1], edges[b + 2]
        w[b] = np.where(bins < l, 0, np.where(bins <= c, (bins - l) / max(c - l, 1e-9), np.where(bins <= r, (r - bins) / max(r - c, 1e-9), 0)))
    return w


def frames(x, w):
    win = np.hanning(N)
    out = []
    for i in range(0, len(x) - N, HOP):
        m = np.abs(np.fft.rfft(x[i:i + N] * win)) / (N / 4)
        out.append(w @ (m * m))
    return np.array(out)


def main():
    quiet, path = sys.argv[1], sys.argv[2]
    with wave.open(quiet) as f:
        sr = f.getframerate()
        x = np.frombuffer(f.readframes(f.getnframes()), "<i2").astype(float) / 32768
    x = np.interp(np.arange(int(len(x) * RATE / sr)) * sr / RATE, np.arange(len(x)), x)
    w = weights()
    room = frames(x, w).mean(axis=0)
    # at the rip's rate with a floor under an empty low band's bins
    room_db = 10 * np.log10(room + 1e-14)
    print(f"the room: mean {room_db.mean():.1f} dB, from {room_db.min():.1f} to {room_db.max():.1f} over the bands, full scale 0")
    rng = np.random.default_rng(1)
    white = frames(rng.standard_normal(120 * RATE), w)
    means = white.mean(axis=0, keepdims=True)
    # a band under one bin has no energy of its own: it borrows the next band's frames
    for b in range(BANDS):
        if means[0, b] <= 0:
            white[:, b] = white[:, b + 1]
            means[0, b] = means[0, b + 1]
    white /= means
    np.savez(path, white=white.astype(np.float32), room=(room_db - room_db.mean()).astype(np.float32))
    print(f"{path}: {len(white)} frames; band spread {white.std(axis=0).min():.2f} to {white.std(axis=0).max():.2f} of its mean")


main()
