#!/usr/bin/env python3
"""datacheck.py <corpus dir>...: a shallow check of a corpus's labels against its sound, on a sample of loops
(SAMPLE, 160; GROUP=1 splits a corpus by where each recording came from): file sizes, finite frames, each loop's
level (a recording read at the wrong scale sits far under the rest), labels within their loop and the piano's
keys, the pitches a model is built for, doubled (frame, pitch) labels, loops with no label, notes and hits in silence, and the lag in 11.6 ms
frames at which the sound rises most after a labelled start, hit or beat, with how far over the median it stands."""
import json, os, sys, numpy as np
K = int(os.environ.get('SAMPLE', 160)); LAGS = range(-8, 9)
def mel(p, frames, bands): return np.fromfile(p, dtype='<f4', count=frames * bands, offset=8).reshape(frames, bands)
print(f'{"corpus":26s} loops sampled  size nan range  doubled  empty | notes: lag lift  in silence | hits: lag lift  in silence | beats: lag lift')
def source(d, i):
    s = json.load(open(f'{d}/loop_{i:03d}.json')).get('source', 'generated'); t = s.split('/')[0]
    return 'egmd' if t.startswith('drummer') else 'vocadito' if 'vocadito' in s.lower() else t
jobs = []
for d in sys.argv[1:]:
    d = os.path.expanduser(d); n = json.load(open(d + '/corpus.json'))['loops']
    if os.environ.get('GROUP'):
        ix = sorted(set(int(i * n / min(K, n)) for i in range(min(K, n)))); g = {i: source(d, i) for i in ix}
        jobs += [(d, k, g) for k in sorted(set(g.values()))]
    else: jobs.append((d, None, {}))
for d, only, g in jobs:
    grp = lambda j: g[j]
    c = json.load(open(d + '/corpus.json')); n = c['loops']; bands = c['mel_bands']
    nj = sum(1 for f in os.listdir(d) if f.startswith('loop_') and f.endswith('.json')); nm = sum(1 for f in os.listdir(d) if f.endswith('.mel'))
    idx = sorted(set(int(i * n / min(K, n)) for i in range(min(K, n)))); big = 0
    bad = dict(size=0, nan=0, rng=0); dbl = tot = empty = 0; notes_sil = notes_n = hits_sil = hits_n = 0
    out = past = fq = fn = 0
    peaks = []
    acc = {k: np.zeros(len(LAGS)) for k in ('note', 'hit', 'beat')}; cnt = dict(note=0, hit=0, beat=0); why = []
    for i in (idx if only is None else [j for j in idx if grp(j) == only]):
        b = f'{d}/loop_{i:03d}'; lab = json.load(open(b + '.json')); fr = lab['frames']; ev = lab['events']
        if os.path.getsize(b + '.mel') != 8 + 4 * fr * bands: bad['size'] += 1; why.append(f'loop {i}: mel size'); continue
        if fr > 60000: big += 1; fr = 60000
        m = mel(b + '.mel', fr, bands)
        peaks.append(float(m.max()))
        if not np.isfinite(m).all(): bad['nan'] += 1; why.append(f'loop {i}: not finite')
        nt = [(e['frame'], e['end'], e['midi']) for e in ev if e['drum'] < 0 and e['midi'] > 0]; ht = [e['frame'] for e in ev if e['drum'] >= 0]
        r = sum(1 for f, e, k in nt if not (0 <= f < lab['frames'] and f <= e))
        out += sum(1 for f, e, k in nt if not 21 <= k <= 108); past += sum(1 for f, e, k in nt if e > lab['frames'] + 2)
        r += sum(1 for e in ev if not (-1 <= e['drum'] < len(c['drums']) and -1 <= e.get('family', -1) < len(c.get('families', [])) + 0 and -1 <= e.get('wave', -1) < len(c.get('waves', []))))
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
        if hi - lo > 0.01:
            fq += int((loud <= quiet).sum()); fn += len(loud)
            for f, e, k in nt:
                if f < fr - 6: notes_n += 1; notes_sil += loud[f:f + 6].max() <= quiet
            for f in ht:
                if f < fr - 4: hits_n += 1; hits_sil += loud[f:f + 4].max() <= quiet
    def lag(key):
        if not cnt[key]: return '   -     -'
        v = acc[key] / cnt[key]; j = int(v.argmax()); return f'{LAGS[j]:+4d} {v[j]:5.1f}'
    pct = lambda a, b: f'{100 * a / b:5.1f}%' if b else '     -'
    name = os.path.basename(d) + ('' if only is None else ': ' + only)
    print(f'{name[:26]:26s} {n:6d} {len(idx):4d}   {bad["size"]:3d} {bad["nan"]:3d} {bad["rng"]:4d}   {pct(dbl, tot)}  {pct(empty, len(idx))} |      {lag("note")}  {pct(notes_sil, notes_n)}    |     {lag("hit")}  {pct(hits_sil, hits_n)}    |     {lag("beat")}')
    pk = float(np.median(peaks)) if peaks else 0.0
    why = [f'loudest value a loop {pk:.2f} (median), {sum(1 for x in peaks if x < pk / 10)} loops under a tenth of it; {100 * fq / max(fn, 1):.0f}% of all frames are quiet by the same measure; {100 * out / max(tot, 1):.2f}% of notes outside the piano's keys, {100 * past / max(tot, 1):.2f}% end after their loop'] + why
    extra = ([f'{nj} json and {nm} mel files for {n} loops'] if nj != n or nm != n else []) + why[:4] + ([f'{big} loops read to 60,000 frames'] if big else [])
    if extra: print('      ' + '; '.join(extra))
