#!/usr/bin/env python3
"""place.py: the notes of a take of one of VocalSet's exercises placed on its sound, and the test a take is refused
by; imported by label.py and fitted by placefit.py. Every bound is in TRIED as first tried; place.json beside this
file, once placefit.py has written it, replaces them, and RULED is laid over that. ../README.md has the method."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vocal, shapes
FRAME = vocal.HOP / vocal.SR
TRIED = dict(lp=1300, ap_max=.5, smooth=0., fold=0., split=0, ramp=0., reach=3.5, stub=0., arrive=.5, arrive_share=1., settle=0, near=.5, far=1.5, stray=1., gap=0., even=0.)
BOUNDS = dict(TRIED)
FITTED = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'place.json')
# Bounds ruled on, laid over the fitted ones; placefit.py fits and reports without them. Both refuse more than the
# fit's rule took: a take with a squeezed note, and one with a shorter silence inside a note (a restart).
RULED = dict(gap=.3, even=1 / 3)
if os.path.exists(FITTED) and os.environ.get('PLACE_BOUNDS') != 'tried':
    BOUNDS.update(json.load(open(FITTED))['bounds'])
    BOUNDS.update(RULED)
# what the sheet gives each note's length, the others' being one: a scale ends on a quarter after its eighths
LENGTHS = {'scales': [1] * 16 + [2]}

def pitch(f0, ap, rms, ap_max):
    """a take's frames from vocal.frames: each one's pitch as a MIDI number on A440's grid, and whether it is voiced"""
    return 69 + 12 * np.log2(np.maximum(f0, 1e-6) / 440), (ap < ap_max) & (rms > rms.max() * 10 ** (vocal.HAND['top'] / 20))

def centre(p, voiced, half):
    """p with each voiced frame's pitch replaced by the median of the voiced frames within `half` frames of it, in its
    own run of sound (a gap of two frames breaks no run): a vibrato's middle; a slide, being one way, stays as it is"""
    out, idx = p.copy(), np.flatnonzero(voiced)
    if half < 1 or not len(idx):
        return out
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 3) + 1):
        h = min(half, len(run) - 1)
        if h >= 1:
            out[run] = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(p[run], h, mode='reflect'), 2 * h + 1), axis=1)
    return out

def settled(x):
    """where a note's pitch settles: the median of the half of its frames, one after another, that moves least. A
    slide out of the note, a scoop into it and the turn at its edge are the note's and are not its pitch."""
    w = max(3, len(x) // 2)
    if len(x) <= w:
        return float(np.median(x))
    runs = np.lib.stride_tricks.sliding_window_view(x, w)
    return float(np.median(runs[int(np.argmin(runs.std(axis=1)))]))

def cut(x, up, ramp=0):
    """where a run of pitch parts into two levels, the second the higher one when `up` and the lower when not, with a
    straight slide of up to `ramp` frames between them: the frame the second level starts at (the slide's end, a
    slide being the first note's), three frames at least in each level, by the least squared distance of the run
    from the two levels and the slide; or nothing where no such parting has its levels that way round. Pitch more
    than three semitones from the run's median counts as three."""
    n = len(x)
    if n < 6:
        return None
    y = np.clip(x - np.median(x), -3.0, 3.0)
    i = np.arange(n)
    s, q, w = (np.concatenate([[0.0], np.cumsum(v)]) for v in (y, y * y, i * y))      # sums over the frames before each one
    best = (np.inf, None)
    for L in range(0, int(ramp) + 1):
        t1 = np.arange(3, n - 2 - L)                        # the slide runs from t1 to t1 + L
        if not len(t1):
            break
        t2 = t1 + L
        a, b = s[t1] / t1, (s[n] - s[t2]) / (n - t2)
        err = (q[t1] - t1 * a * a) + (q[n] - q[t2] - (n - t2) * b * b)
        if L:
            sy, siy = s[t2] - s[t1], w[t2] - w[t1]
            err = err + (q[t2] - q[t1]) - 2 * (a * sy + (b - a) / L * (siy - (t1 - 0.5) * sy)) + L * a * a + a * (b - a) * L + (b - a) ** 2 * (4 * L * L - 1) / (12 * L)
        ok = (b > a) if up else (b < a)
        if ok.any():
            j = int(np.argmin(np.where(ok, err, np.inf)))
            if err[j] < best[0]:
                best = (float(err[j]), int(t2[j]))
    return best[1]

def place(p, voiced, rms, steps, smooth=0., fold=0., split=0, ramp=0., reach=3.5, stub=0., arrive=.5, arrive_share=1., settle=0, **_):
    """(the tonic, each note's (first frame, frame past its last), the take's standard in semitones over A440's
    grid, the pitch as placed: p, and with `fold` each frame read an octave from its note's key moved onto it), or
    nothing for a take with fewer voiced frames than notes"""
    rel = np.concatenate([[0], np.cumsum(steps)])
    idx = np.flatnonzero(voiced)
    if len(idx) < len(rel):
        return None
    half = int(round(min(smooth, 0.5 * len(idx) * FRAME / len(rel)) / FRAME / 2))
    c = centre(p, voiced, half)
    ang = 2 * np.pi * c[idx]
    grid = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) / (2 * np.pi)
    q, n = c[idx] - grid, len(rel)
    tonics = np.arange(int(np.floor(q.min() - rel.max())) - 1, int(np.ceil(q.max() - rel.min())) + 2)
    cost = np.abs(q[:, None, None] - (tonics[None, :, None] + rel[None, None, :]))                    # frame, tonic, note
    if fold:
        cost = np.minimum(cost, np.abs(cost - 12) + fold)  # one octave over or under, no more
    cost = np.minimum(cost, 3.0)
    d = np.full((len(tonics), n), np.inf)
    d[:, 0] = cost[0, :, 0]
    backs = np.zeros((len(q), len(tonics), n), bool)      # true where the path came from the note before
    edge = np.full((len(tonics), 1), np.inf)
    for t in range(1, len(q)):
        prev = np.concatenate([edge, d[:, :-1]], 1)
        backs[t] = prev < d
        d = np.minimum(d, prev) + cost[t]
    k = int(np.argmin(d[:, -1]))
    tonic, back = int(tonics[k]), backs[:, k, :]
    i, bounds, given = n - 1, [[0, 0] for _ in range(n)], np.zeros(len(q), int)
    bounds[i][1] = idx[-1] + 1
    for t in range(len(q) - 1, -1, -1):
        given[t] = i
        if t and back[t, i]:
            bounds[i][0] = idx[t]
            i -= 1
            bounds[i][1] = idx[t - 1] + 1
    bounds[0][0] = idx[0]
    if fold:                                               # the octave the sheet says, in the pitch as placed
        d = p[idx] - grid - (tonic + rel[given])
        p = p.copy()
        p[idx] -= np.where(np.abs(np.abs(d) - 12) + fold < np.minimum(np.abs(d), 3.0), 12 * np.sign(d), 0.0)
        c = centre(p, voiced, half)
    # the standard, from the notes placed, within half a semitone of A440's grid
    off = float(np.median(p[idx] - (tonic + rel[given])))
    move = int(np.floor(off + 0.5))
    tonic, off = tonic + move, off - move
    keys = tonic + rel
    for i in range(n):                                     # a note is its own sound: a breath's frames beside it are not
        a, b = bounds[i]
        mine = np.flatnonzero(voiced[a:b] & (np.abs(p[a:b] - off - keys[i]) <= reach))
        if len(mine) and stub:                             # nor is a scrap of sound a longer silence parts from the rest
            runs = np.split(mine, np.flatnonzero(np.diff(mine) > 1) + 1)
            still = lambda r, s: int((~voiced[a + r[-1] + 1:a + s[0]]).sum())      # the frames with no pitch between two runs
            while len(runs) > 1 and len(runs[0]) * FRAME < stub and still(runs[0], runs[1]) > len(runs[0]):
                runs.pop(0)
            while len(runs) > 1 and len(runs[-1]) * FRAME < stub and still(runs[-2], runs[-1]) > len(runs[-1]):
                runs.pop()
            mine = np.concatenate(runs)
        if len(mine):
            bounds[i] = [a + int(mine[0]), a + int(mine[-1]) + 1]
    for i in range(1, n):
        a, b = bounds[i][0], bounds[i][1]
        if bounds[i - 1][1] < a - 2:                       # silence before it: it starts with its first sound
            continue
        if keys[i] == keys[i - 1]:                         # the same key again: where the level between the two is lowest
            lo, hi = bounds[i - 1][0] + 3, b - 3
            if hi > lo:
                a = lo + int(np.argmin(rms[lo:hi]))
        elif abs(keys[i] - keys[i - 1]) <= split:          # a small step: where the two notes' own levels part
            lo = bounds[i - 1][0]
            v = np.flatnonzero(voiced[lo:b])
            at = cut(p[lo:b][v], keys[i] > keys[i - 1], round(ramp / FRAME)) if len(v) else None
            if at is not None:
                a = lo + int(v[at])
        else:
            near = min(arrive, arrive_share * abs(keys[i] - keys[i - 1]))      # a small step arrives nearer
            own = c[a:b][voiced[a:b]]
            goal = settled(own) if settle and len(own) >= 4 else off + keys[i]
            here = np.flatnonzero(voiced[a:b] & (np.abs(c[a:b] - goal) <= near))
            if len(here) and here[0] < (b - a) * 0.6:
                a += int(here[0])
        bounds[i - 1][1], bounds[i][0] = a, a
    return tonic, bounds, off, p

def refused(p, voiced, tonic, steps, bounds, off, exercise='', near=.5, far=1.5, stray=1., gap=0., even=0., **_):
    """what a placed take is refused for, or nothing"""
    keys = tonic + np.concatenate([[0], np.cumsum(steps)])
    want = LENGTHS.get(exercise, [1] * len(keys))
    strays, frames, lengths = 0, 0, []
    for (a, b), k, w in zip(bounds, keys, want):
        v = p[a:b][voiced[a:b]] - off
        if b - a < 3 or not len(v):
            return 'a note too short'
        if abs(settled(v) - k) > near:
            return 'a note off its key'
        if gap and max(len(run) for run in ''.join('v' if x else '.' for x in voiced[a:b]).split('v')) * FRAME > gap:
            return 'a silence inside a note'
        strays, frames = strays + int((np.abs(v - k) > far).sum()), frames + len(v)
        lengths.append((b - a) / w)
    if strays > stray * frames:
        return 'sound that is not its notes'
    if even and min(lengths) < even * np.median(lengths):
        return 'a note far shorter than the others'
    return ''

def take(x, exercise, **bounds):
    """a take's notes placed and judged under BOUNDS or bounds: (the notes as (start s, length s, midi), or none; what
    it was refused for, or nothing; the tonic; its standard in cents; its frames' seconds and Hz as placed, 0 where
    unvoiced; the voiced frames the fold moved an octave)"""
    b = {**BOUNDS, **bounds}
    f0, ap, rms = vocal.frames(x, b['lp'], vocal.HAND['dip'])
    p, voiced = pitch(f0, ap, rms, b['ap_max'])
    track = (np.arange(len(f0)) * FRAME + vocal.CENTRE, np.where(voiced, f0, 0.0))
    got = place(p, voiced, rms, shapes.SHAPE[exercise], **b)
    if got is None:
        return None, 'too little sound', 0, 0.0, track, 0
    tonic, spans, off, placed = got
    folded = int((voiced & (np.abs(placed - p) > 6)).sum())
    p = placed
    track = (track[0], np.where(voiced, 440 * 2 ** ((p - 69) / 12), 0.0))
    why = refused(p, voiced, tonic, shapes.SHAPE[exercise], spans, off, exercise, **b)
    keys = tonic + np.concatenate([[0], np.cumsum(shapes.SHAPE[exercise])])
    notes = [(a * FRAME + vocal.CENTRE, (e - a) * FRAME, int(k)) for (a, e), k in zip(spans, keys)]
    return (None if why else notes), why, tonic, 100 * off, track, folded
