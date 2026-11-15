#!/usr/bin/env python3
"""datacheck.py <corpus dir>...: a shallow check of a corpus's labels against its sound, on a sample of loops (SAMPLE,
160; GROUP=1 splits a corpus by where each recording came from): the frames' layout, file sizes, finite frames, labels
within their loop and the piano's keys, doubled (frame, pitch) labels, loops with no label, notes and hits in a loop's
own quiet, and the lag in 11.6 ms frames at which the sharp plane rises most after a labelled start, hit or beat, with
how far over the median it stands. A frame is peak-normalized, so a loop reads the same at any scale."""
import json
import os
import sys

import numpy as np

import ripmodels

K, LAGS = int(os.environ.get('SAMPLE', 160)), range(-8, 9)
HEAD = f'{"corpus":26s} loops sampled  size nan range  doubled  empty | notes: lag lift  in silence | hits: lag lift  in silence | beats: lag lift'


def load(p):
    with open(p) as f:
        return json.load(f)


def frames(p, n):
    """the first n band frames of a loop's file, both planes"""
    return np.fromfile(p, dtype='<f4', count=n * ripmodels.VALUES, offset=8).reshape(n, ripmodels.VALUES)


def source(d, i):
    s = load(f'{d}/loop_{i:03d}.json').get('source', 'generated'); t = s.split('/')[0]
    return 'egmd' if t.startswith('drummer') else 'vocadito' if 'vocadito' in s.lower() else t


def sampled(n):
    """the loops a corpus of n is read on, spread over it"""
    return sorted(set(int(i * n / min(K, n)) for i in range(min(K, n))))


def check(d, only=None, g=None):
    """a corpus's counts over its sampled loops, those of the group only when given (g, each loop's group)"""
    c = load(d + '/corpus.json')
    if c.get('layout') != ripmodels.LAYOUT:
        raise SystemExit(ripmodels.refused(f'{d}: its manifest', c.get('layout')))
    n = c['loops']
    nj = sum(1 for f in os.listdir(d) if f.startswith('loop_') and f.endswith('.json')); nb = sum(1 for f in os.listdir(d) if f.endswith('.bands'))
    idx = sampled(n); big = 0
    bad = dict(size=0, nan=0, rng=0); dbl = tot = empty = 0; notes_sil = notes_n = hits_sil = hits_n = 0
    out = past = fq = fn = 0
    acc = {k: np.zeros(len(LAGS)) for k in ('note', 'hit', 'beat')}; cnt = dict(note=0, hit=0, beat=0); why = []
    for i in (idx if only is None else [j for j in idx if g[j] == only]):
        b = f'{d}/loop_{i:03d}'; lab = load(b + '.json'); fr = lab['frames']; ev = lab['events']
        if os.path.getsize(b + '.bands') != 8 + 4 * fr * ripmodels.VALUES: bad['size'] += 1; why.append(f'loop {i}: frames size'); continue
        if fr > 60000: big += 1; fr = 60000
        m = frames(b + '.bands', fr)
        if not np.isfinite(m).all(): bad['nan'] += 1; why.append(f'loop {i}: not finite')
        silent, m = not m.any(), m[:, ripmodels.BANDS:]  # the sharp plane hears a strike in its frame, the fine plane's low bands over seconds
        nt = [(e['frame'], e['end'], e['midi']) for e in ev if e['drum'] < 0 and e['midi'] > 0]; ht = [e['frame'] for e in ev if e['drum'] >= 0]
        r = sum(1 for f, e, k in nt if not (0 <= f < lab['frames'] and f <= e))
        out += sum(1 for f, e, k in nt if not ripmodels.LOW <= k < ripmodels.LOW + ripmodels.PITCHES); past += sum(1 for f, e, k in nt if e > lab['frames'] + 2)
        r += sum(1 for e in ev if not (-1 <= e['drum'] < len(c['drums']) and -1 <= e.get('family', -1) < len(c.get('families', [])) and -1 <= e.get('wave', -1) < len(c.get('waves', []))))
        r += sum(1 for f in ht if not 0 <= f < lab['frames'])
        if r: bad['rng'] += 1; why.append(f'loop {i}: {r} labels out of range')
        tot += len(nt); dbl += len(nt) - len(set((f, k) for f, e, k in nt)); empty += not ev
        loud = m.mean(1); lo, hi = np.percentile(loud, 1), np.percentile(loud, 99); quiet = lo + 0.08 * (hi - lo)
        flux = np.maximum(m[1:] - m[:-1], 0).sum(1); flux = np.concatenate([[0], flux]); med = np.median(flux) + 1e-6

        def at(frames_, key):
            f = np.array([x for x in frames_ if 8 <= x < fr - 12], dtype=int)
            if len(f) == 0: return
            acc[key] += np.array([flux[f + l].mean() for l in LAGS]) / med * len(f); cnt[key] += len(f)
        at([f for f, e, k in nt], 'note'); at(ht, 'hit'); at(lab.get('beats', []), 'beat')
        if silent or hi - lo > 0.01:  # a loop of one steady sound has no quiet of its own to tell
            fq += len(loud) if silent else int((loud <= quiet).sum()); fn += len(loud)
            for f, e, k in nt:
                if f < fr - 6: notes_n += 1; notes_sil += silent or loud[f:f + 6].max() <= quiet
            for f in ht:
                if f < fr - 4: hits_n += 1; hits_sil += silent or loud[f:f + 4].max() <= quiet
    lags = {k: (LAGS[int((acc[k] / cnt[k]).argmax())], float((acc[k] / cnt[k]).max())) if cnt[k] else None for k in acc}
    return dict(loops=n, sampled=len(idx), json=nj, bands=nb, big=big, bad=bad, doubled=(dbl, tot), empty=empty, outside=out, past=past,
                quiet=(fq, fn), notes=(notes_sil, notes_n), hits=(hits_sil, hits_n), lags=lags, why=why)


def report(name, v):
    """a corpus's line and the line under it of what it found"""
    lag = lambda key: '   -     -' if v['lags'][key] is None else '{:+4d} {:5.1f}'.format(*v['lags'][key])
    pct = lambda a, b: f'{100 * a / b:5.1f}%' if b else '     -'
    bad, tot = v['bad'], v['doubled'][1]
    line = (f'{name[:26]:26s} {v["loops"]:6d} {v["sampled"]:4d}   {bad["size"]:3d} {bad["nan"]:3d} {bad["rng"]:4d}   {pct(*v["doubled"])}  {pct(v["empty"], v["sampled"])} |'
            f'      {lag("note")}  {pct(*v["notes"])}    |     {lag("hit")}  {pct(*v["hits"])}    |     {lag("beat")}')
    why = [f"{100 * v['quiet'][0] / max(v['quiet'][1], 1):.0f}% of all frames are quiet; {100 * v['outside'] / max(tot, 1):.2f}% of notes outside the piano's keys, "
           f"{100 * v['past'] / max(tot, 1):.2f}% end after their loop"] + v['why']
    n = v['loops']
    extra = ([f'{v["json"]} json and {v["bands"]} frame files for {n} loops'] if v['json'] != n or v['bands'] != n else []) + why[:4] + ([f'{v["big"]} loops read to 60,000 frames'] if v['big'] else [])
    return [line] + (['      ' + '; '.join(extra)] if extra else [])


def main(argv):
    print(HEAD)
    jobs = []
    for d in argv:
        d = os.path.expanduser(d)
        if os.environ.get('GROUP'):
            g = {i: source(d, i) for i in sampled(load(d + '/corpus.json')['loops'])}
            jobs += [(d, k, g) for k in sorted(set(g.values()))]
        else:
            jobs.append((d, None, {}))
    for d, only, g in jobs:
        print('\n'.join(report(os.path.basename(d) + ('' if only is None else ': ' + only), check(d, only, g))))


if __name__ == '__main__':
    main(sys.argv[1:])
