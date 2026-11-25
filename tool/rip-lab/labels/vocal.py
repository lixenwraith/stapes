#!/usr/bin/env python3
"""vocal.py: a solo voice's frames, each one's pitch by YIN, its periodicity and its level, which place.py places
notes on, placefit.py fits by and datasets/vocalsing.py measures. ../README.md says what HAND's two bounds are."""
import wave
import numpy as np
SR, HOP, W, TMAX, TMIN = 22050, 256, 1024, 330, 18
CENTRE = W / 2 / SR  # a frame's pitch is its window's, so its time is the window's centre, two hops after its start
HAND = dict(dip=.12, top=-38.)

def read(p):
    w = wave.open(p); sr = w.getframerate(); ch = w.getnchannels(); sw = w.getsampwidth()
    x = np.frombuffer(w.readframes(w.getnframes()), {2: '<i2', 3: None, 4: '<i4'}[sw]).astype(np.float64) / 2 ** (8 * sw - 1)
    x = x.reshape(-1, ch).mean(1)
    if sr == 44100:
        x = np.convolve(x, [.25, .5, .25], 'same')[::2]
    elif sr != SR:
        n = int(len(x) * SR / sr)
        x = np.interp(np.arange(n) * sr / SR, np.arange(len(x)), x)
    return x

def lowpass(x, fc, taps=201):
    n = np.arange(taps) - (taps - 1) / 2
    h = np.sinc(2 * fc / SR * n) * np.hamming(taps)
    return np.convolve(x, h / h.sum(), 'same')

def frames(x, lp=0, dip=.12):
    """each frame's pitch in Hz, its aperiodicity (the difference function at the lag taken) and its level"""
    n = max((len(x) - W - TMAX) // HOP, 0)
    f0, ap = np.zeros(n), np.ones(n)
    z = lowpass(x, lp) if lp else x
    L, fft, lags = W + TMAX, 2048, np.arange(TMAX + 1)
    for a in range(0, n, 512):
        b = min(a + 512, n)
        y = np.stack([z[i * HOP:i * HOP + L] for i in range(a, b)])
        head = np.zeros_like(y); head[:, :W] = y[:, :W]
        c = np.fft.irfft(np.conj(np.fft.rfft(head, fft)) * np.fft.rfft(y, fft), fft)[:, :TMAX + 1]
        sq = np.cumsum(np.concatenate([np.zeros((len(y), 1)), y ** 2], 1), 1)
        r0, rt = sq[:, W][:, None], sq[:, W + lags] - sq[:, lags]
        d = np.maximum(r0 + rt - 2 * c, 0)
        dp = np.ones_like(d)
        dp[:, 1:] = d[:, 1:] * np.arange(1, TMAX + 1) / np.maximum(np.cumsum(d[:, 1:], 1), 1e-12)
        # the first lag whose dip is deep enough, or the deepest; then down to the bottom of its dip
        v, rows = dp[:, TMIN:TMAX], np.arange(len(y))
        deep = v < dip
        t = TMIN + np.where(deep.any(1), deep.argmax(1), v.argmin(1))
        while True:
            down = (t + 1 < TMAX) & (dp[rows, np.minimum(t + 1, TMAX)] < dp[rows, t])
            if not down.any():
                break
            t = t + down
        aa, bb, cc = dp[rows, t - 1], dp[rows, t], dp[rows, t + 1]
        den = aa - 2 * bb + cc
        f0[a:b] = SR / (t + np.where(den > 1e-9, .5 * (aa - cc) / np.where(den > 1e-9, den, 1), 0))
        ap[a:b] = bb
    e = np.concatenate([[0], np.cumsum(x ** 2)])
    i = np.arange(n) * HOP
    return f0, ap, np.sqrt(np.maximum(e[i + W] - e[i], 0) / W)
