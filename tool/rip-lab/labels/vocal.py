#!/usr/bin/env python3
"""vocal.py: a take's frames (pitch by YIN, periodicity and level), which place.py reads, and a free tracker that
cuts notes from them, which no label uses: trackscore.py scores it and trackfit.py fits its BOUNDS (bounds.json
beside this file replaces them; VOCAL_BOUNDS=hand keeps the hand-set ones). ../README.md names each bound."""
import json, os, wave
import numpy as np
SR, HOP, W, TMAX, TMIN = 22050, 256, 1024, 330, 18
CENTRE = W / 2 / SR  # a frame's pitch is its window's, so its time is the window's centre, two hops after its start
HAND = dict(lp=0, dip=.12, ap_max=.25, top=-38., move=8., gap=3, minlen=6, join=10, slip_len=.25, slip_far=7,
            blip_len=0., blip_near=1, slide_len=0., slide_share=.5)
BOUNDS, COVER_MIN = dict(HAND), 0.0   # COVER_MIN: the least cover a recording keeps its place in a list with
FITTED = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bounds.json')
ASKED = os.environ.get('VOCAL_BOUNDS', '')   # hand: as before any fit; a JSON object: those bounds over the hand-set ones
if ASKED.startswith('{'):
    BOUNDS.update(json.loads(ASKED))
elif os.path.exists(FITTED) and ASKED != 'hand':
    BOUNDS.update(json.load(open(FITTED))['bounds'])
    COVER_MIN = json.load(open(FITTED)).get('cover_min', 0.0)

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

def blips(notes, q, join, blip_len, blip_near, slide_len, slide_share):
    """notes as [start s, length s, midi, first frame, frames] less the swings, slides and scoops written as notes"""
    frame = HOP / SR
    touch = lambda a, b: b[3] - (a[3] + a[4]) <= join
    crossed = lambda note, by: (np.median(q[note[3] + note[4] - 3:note[3] + note[4]]) - np.median(q[note[3]:note[3] + 3])) / by
    while True:
        for i in sorted(range(len(notes)), key=lambda i: notes[i][4]):      # the shortest first
            p, c, x = notes[i - 1] if i else None, notes[i], notes[i + 1] if i + 1 < len(notes) else None
            before, after = p is not None and touch(p, c), x is not None and touch(c, x)
            if before and after and c[1] < blip_len and p[2] == x[2] and 0 < abs(c[2] - p[2]) <= blip_near:
                notes[i - 1:i + 2] = [[p[0], x[0] + x[1] - p[0], p[2], p[3], x[3] + x[4] - p[3]]]
                break
            if c[1] >= slide_len or not after or c[2] == x[2]:
                continue
            if before and min(p[2], x[2]) < c[2] < max(p[2], x[2]) and crossed(c, x[2] - p[2]) >= slide_share:
                notes[i - 1:i + 1] = [[p[0], c[0] + c[1] - p[0], p[2], p[3], c[3] + c[4] - p[3]]]
                break
            if not before and crossed(c, x[2] - c[2]) >= slide_share:
                notes[i:i + 2] = [[c[0], x[0] + x[1] - c[0], x[2], c[3], x[3] + x[4] - c[3]]]
                break
        else:
            return notes

def cut(f0, ap, rms, top=-38., move=8., gap=3, minlen=6, ap_max=.25, join=10, blip_len=0., blip_near=1, slide_len=0., slide_share=.5, **_):
    """(start s, length s, midi) of a solo voice from its frames"""
    if not len(f0):
        return []
    voiced = (ap < ap_max) & (rms > rms.max() * 10 ** (top / 20))
    p = 69 + 12 * np.log2(np.maximum(f0, 1e-6) / 440)
    q = p.copy()
    if voiced.any():                                           # octave glitches and the like: the median of the voiced frames two either way
        pad = np.concatenate([[np.nan] * 2, np.where(voiced, p, np.nan), [np.nan] * 2])
        around = np.stack([pad[k:k + len(p)] for k in range(5)], 1)
        q[voiced] = np.nanmedian(around[voiced], 1)
    v = voiced.copy()
    t = 0
    while t < len(v):                                          # short gaps inside a note
        if not v[t]:
            e = t
            while e < len(v) and not v[e]:
                e += 1
            if 0 < t and e < len(v) and e - t <= gap and abs(q[e] - q[t - 1]) < 1:
                v[t:e] = True
                q[t:e] = q[t - 1]
            t = e
        else:
            t += 1
    # the singer's own pitch standard, so a steady flat or sharp doesn't round to the next semitone
    if v.any():
        ang = 2 * np.pi * q[v]
        q = q - np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) / (2 * np.pi)
    out = []
    n = len(v)
    t = 0
    frame = HOP / SR
    while t < n:
        if not v[t]:
            t += 1
            continue
        e = t
        while e < n and v[e]:
            e += 1
        seg = q[t:e]
        # a path over semitones paying each frame its distance and each move a cost: a vibrato
        # around a pitch stays on it, where a step that holds pays for itself
        lo = int(np.floor(seg.min())); hi = int(np.ceil(seg.max())) + 1
        states = np.arange(lo, hi)
        cost = np.minimum(np.abs(seg[:, None] - states[None, :]), 3.0)
        best = cost[0].copy()
        back = np.zeros((len(seg), len(states)), int)
        for i in range(1, len(seg)):
            m = int(np.argmin(best))
            stay = best
            jump = best[m] + move
            take = jump < stay
            back[i] = np.where(take, m, np.arange(len(states)))
            best = np.where(take, jump, stay) + cost[i]
        path = np.zeros(len(seg), int)
        path[-1] = int(np.argmin(best))
        for i in range(len(seg) - 1, 0, -1):
            path[i - 1] = back[i, path[i]]
        i = 0
        while i < len(seg):
            j = i + 1
            while j < len(seg) and path[j] == path[i]:
                j += 1
            if j - i >= minlen:
                out.append(((t + i) * frame + CENTRE, (j - i) * frame, int(states[path[i]]), t + i, j - i))
            i = j
        t = e
    merged = []
    for a, d, m, at, n in out:                                 # a breath of a gap inside a held pitch
        if merged and merged[-1][2] == m and a - merged[-1][0] - merged[-1][1] <= join * frame:
            merged[-1] = [merged[-1][0], a + d - merged[-1][0], m, merged[-1][3], at + n - merged[-1][3]]
        else:
            merged.append([a, d, m, at, n])
    if blip_len or slide_len:
        merged = blips(merged, q, join, blip_len, blip_near, slide_len, slide_share)
    return [(a, d, m) for a, d, m, _, _ in merged]

def slips(ns, slip_len=.25, slip_far=7, **_):
    """the notes less the tracker's slips, and how many those were"""
    out = []
    for i, (a, d, m) in enumerate(ns):
        p = ns[i - 1][2] if i else m
        q = ns[i + 1][2] if i + 1 < len(ns) else m
        if not (d < slip_len and abs(m - p) > slip_far and abs(m - q) > slip_far):
            out.append((a, d, m))
    return out, len(ns) - len(out)

def sung(rms):
    """the frames a voice sounds in: within 20 dB of the recording's loud level, the 95th percentile of its frames'"""
    return rms > np.percentile(rms, 95) / 10 if len(rms) else np.zeros(0, bool)

def track(x, **bounds):
    """(the notes of a solo voice as (start s, length s, midi), how many slips were left out, the cover: the share of
    the sung frames a note holds), under BOUNDS or bounds"""
    b = {**BOUNDS, **bounds}
    f0, ap, rms = frames(x, b['lp'], b['dip'])
    ns, slipped = slips(cut(f0, ap, rms, **b), **b)
    held, loud = np.zeros(len(rms), bool), sung(rms)
    for a, d, _ in ns:
        i = int(round((a - CENTRE) * SR / HOP))
        held[i:i + int(round(d * SR / HOP))] = True
    return ns, slipped, float((held & loud).sum() / max(loud.sum(), 1))
