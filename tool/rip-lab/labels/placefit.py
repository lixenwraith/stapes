#!/usr/bin/env python3
"""placefit.py <truth root> [write]: place.py's bounds fitted on known takes of VocalSet's exercises (lines.py's
three draws) and checked on another seed and on voices left out; a value is taken only if its gain holds
over a thousand redraws of the takes. With `write` the bounds go to place.json. placefit.py <truth root> reach
<corpus dir>...: how far a placed start sits from its known start, into each corpus.json. ../README.md has both."""
import collections, csv, glob, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vocal, shapes, place

PLACING = collections.OrderedDict(lp=[0, 2500, 1800, 1300, 1000, 800], ap_max=[.25, .35, .5, .65, .8], smooth=[0., .1, .15, .2, .25, .3],
                                  fold=[0., .25, .5, 1., 2.], split=[0, 1, 2, 12], ramp=[0., .05, .1, .2, .3], reach=[2.5, 3.5, 5., 99.], settle=[0, 1], arrive=[.2, .3, .5, .7, 1., 99.],
                                  arrive_share=[.15, .25, .35, .5, 1.])
TEST = collections.OrderedDict(near=[.5, .7, .9, 1., 1.1, 1.2, 1.5], stub=[0., .03, .06, .1, .15], gap=[0., .1, .15, .2, .3, .5],
                               far=[1.5, 2., 2.5], stray=[.05, .1, .15, .2, .3, .5, 1.], even=[0., .2, .3, .4, .5, .6])

class Draw:
    def __init__(self, root, name):
        folder, self.name, self.takes, self.frames, self.held = f'{root}/engine-shapes-{name}', name, [], {}, {}
        for r in csv.DictReader(open(f'{folder}/lines.tsv'), delimiter='\t'):
            notes = [(float(l['start']), float(l['end']), int(l['key'])) for l in csv.DictReader(open(f"{folder}/{r['name']}.tsv"), delimiter='\t') if l['kind'] == 'note']
            self.takes.append(dict(name=r['name'], exercise=r['exercise'], wrong=r['wrong'], off=int(r['off_note']) >= 0, half=abs(float(r['tuning'])) >= 45,
                                   notes=notes, x=vocal.read(f"{folder}/{r['name']}.wav")))
        self.right = np.array([not t['wrong'] for t in self.takes])

    def placed(self, b):
        """every take placed under b's low-pass, periodicity bound, smoothing, fold and arrival: (the pitch as placed,
        voiced, the placing)"""
        key = (b['lp'], b['ap_max'], b['smooth'], b['fold'], b['split'], b['ramp'], b['reach'], b['stub'], b['settle'], b['arrive'], b['arrive_share'])
        if key not in self.held:
            if b['lp'] not in self.frames:
                self.frames[b['lp']] = [vocal.frames(t['x'], b['lp'], vocal.HAND['dip']) for t in self.takes]
            out = []
            for t, (f0, ap, rms) in zip(self.takes, self.frames[b['lp']]):
                p, voiced = place.pitch(f0, ap, rms, b['ap_max'])
                got = place.place(p, voiced, rms, shapes.SHAPE[t['exercise']], **b)
                out.append((got[3] if got else p, voiced, got[:3] if got else None))
            self.held[key] = out
        return self.held[key]

    def labels(self, b):
        """a row a take: sounding frames, those given the right key, notes, starts within 50 ms, within 100 ms, whether
        it passes the test, and the starts' misses in seconds; the first five are a right take's alone"""
        rows, misses = [], []
        for t, (p, voiced, got) in zip(self.takes, self.placed(b)):
            row, miss = [0, 0, 0, 0, 0, 0], []
            if got is not None:
                tonic, spans, off = got
                steps = shapes.SHAPE[t['exercise']]
                row[5] = int(not place.refused(p, voiced, tonic, steps, spans, off, t['exercise'], **b))
                if not t['wrong']:
                    keys = tonic + np.concatenate([[0], np.cumsum(steps)])
                    grid = np.zeros(len(p), int)
                    for (a, e), k in zip(spans, keys):
                        grid[a:e] = k
                    best = (0, 0)
                    for shift in ((-1, 0, 1) if t['half'] else (0,)):
                        sound = right = 0
                        for s, e, k in t['notes']:
                            i, j = int((s + 0.05 - vocal.CENTRE) / place.FRAME) + 1, int((e - 0.05 - vocal.CENTRE) / place.FRAME)
                            sound, right = sound + max(j - i, 0), right + int((grid[i:j] == k + shift).sum())
                        best = max(best, (right, sound))
                    miss = [a * place.FRAME + vocal.CENTRE - s for (a, e), (s, _, _) in zip(spans, t['notes'])]
                    row[:5] = [best[1], best[0], len(miss), sum(1 for m in miss if abs(m) <= 0.05), sum(1 for m in miss if abs(m) <= 0.1)]
            rows.append(row)
            misses.append(miss)
        return np.array(rows, float), misses

def placing(rows, pick=None):
    """the placing's score over the right takes: the mean of the sounding time on the right key and the starts within 50 ms"""
    r = rows if pick is None else rows[pick]
    s = r.sum(0)
    return (s[1] / max(s[0], 1) + s[3] / max(s[2], 1)) / 2

def test(rows, right, pick=None):
    """the test's score: the mean of the right takes that pass and the wrong takes refused"""
    r, w = rows[right], rows[~right]
    if pick is not None:
        r, w = r[pick[0]], w[pick[1]]
    return (r[:, 5].mean() + (1 - w[:, 5]).mean()) / 2

def fit(draw, grid, b, score, strata, rng, t0):
    """a coordinate search over grid from b: (the bounds, what moved)"""
    held, moved = draw.labels(b)[0], collections.OrderedDict()
    for p in range(3):
        took = False
        for name, values in grid.items():
            tried = {v: (held if v == b[name] else draw.labels({**b, name: v})[0]) for v in values}
            top = max(score(tried[v]) for v in values)
            equal = [v for v in values if score(tried[v]) >= top - 1e-12]
            best = b[name] if b[name] in equal else equal[(len(equal) - 1) // 2]     # of equals the middle one; the bound in hand if it is one
            line = f'pass {p + 1}, {name}: ' + ', '.join(f'{v:g} {score(tried[v]):.4f}' + ('*' if v == b[name] else '') for v in values)
            if best != b[name]:
                draws = []
                for _ in range(1000):
                    pick = [rng.choice(s, len(s)) for s in strata]
                    pick = pick[0] if len(pick) == 1 else pick
                    draws.append(score(tried[best], pick) - score(held, pick))
                g, low = score(tried[best]) - score(held), float(np.percentile(draws, 2.5))
                line += f' | {best:g} gains {g:+.4f}, {low:+.4f} at the 2.5th percentile: ' + ('taken' if low > 0 else 'not told apart, stays')
                if low > 0:
                    moved[name] = (moved[name][0] if name in moved else b[name], best)
                    b, held, took = {**b, name: best}, tried[best], True
            else:
                line += ' | stays'
            print(line, f'[{time.time() - t0:.0f} s]', flush=True)
        if not took:
            break
    return b, collections.OrderedDict((k, v) for k, v in moved.items() if v[0] != v[1])

def report(draw, b, title):
    rows, misses = draw.labels(b)
    takes, right = draw.takes, draw.right
    off = np.array([t['off'] for t in takes])
    passed = rows[:, 5] > 0
    m = np.array([x for i, ms in enumerate(misses) if right[i] and passed[i] for x in ms])
    s = rows[right & passed].sum(0)
    wrong = collections.Counter(t['wrong'] for t in takes if t['wrong'])
    let = collections.Counter(t['wrong'] for t, ok in zip(takes, passed) if t['wrong'] and ok)
    why = collections.Counter()
    for t, (p, voiced, got) in zip(takes, draw.placed(b)):
        if not t['wrong']:
            why[place.refused(p, voiced, got[0], shapes.SHAPE[t['exercise']], got[1], got[2], t['exercise'], **b) if got else 'too little sound'] += 1
    by = collections.Counter((t['exercise'], bool(ok)) for t, ok in zip(takes, passed) if not t['wrong'])
    pairs = collections.Counter()                           # two neighbours with no silence between, by the step between them
    for t, ok, (p, voiced, got) in zip(takes, passed, draw.placed(b)):
        if t['wrong'] or not ok:
            continue
        steps = shapes.SHAPE[t['exercise']]
        want = place.LENGTHS.get(t['exercise'], [1] * (len(steps) + 1))
        ln = [(e - a) / w for (a, e), w in zip(got[1], want)]
        for i, st in enumerate(steps):
            if got[1][i][1] >= got[1][i + 1][0] - 2:
                kind = 'a semitone' if abs(st) == 1 else 'a tone' if abs(st) == 2 else 'further'
                pairs[kind, 'pairs'] += 1
                pairs[kind, 'two to one'] += max(ln[i], ln[i + 1]) >= 2 * min(ln[i], ln[i + 1])
    print(f'  {title}: the placing {placing(rows[right]):.4f}, the test {test(rows, right):.4f}')
    print(f'    right takes that pass: {int(passed[right].sum())} of {int(right.sum())}; with a note sung off {int(passed[right & off].sum())} of {int((right & off).sum())},',
          f'without {int(passed[right & ~off].sum())} of {int((right & ~off).sum())}; by exercise', {e: f'{by[e, True]} of {by[e, True] + by[e, False]}' for e in sorted(shapes.SHAPE)})
    print('    a right take is refused for:', {k: v for k, v in why.items() if k} or 'nothing')
    print('    wrong takes let through:', {k: f'{let[k]} of {wrong[k]}' for k in sorted(wrong)})
    if len(m):
        print(f'    on the right takes that pass: {s[1] / max(s[0], 1):.4f} of the sounding time on the right key; a start within 50 ms in {s[3] / max(s[2], 1):.3f}',
              f'of {int(s[2])} notes, within 100 ms {s[4] / max(s[2], 1):.3f}; a start\'s miss, median {1000 * np.median(m):+.0f} ms, median size {1000 * np.median(np.abs(m)):.0f} ms')
    print('    of two joined neighbours, the longer twice the shorter or more: ' + ', '.join(
        f'{k} apart {pairs[k, "two to one"]} of {pairs[k, "pairs"]} ({pairs[k, "two to one"] / max(pairs[k, "pairs"], 1):.3f})' for k in ('a semitone', 'a tone', 'further')))
    return dict(placing=round(float(placing(rows[right])), 4), test=round(float(test(rows, right)), 4), right_pass=f'{int(passed[right].sum())} of {int(right.sum())}')

REACH = .95   # the share of placed starts whose known start the reach holds: a high one, not the worst

def reach(root):
    """how far a placed start sits from its known start, the REACH quantile over the starts of the right takes that
    pass on the three draws, placed under place.py's bounds as a label is: in frames, rounded up; and in seconds,
    of all of them and of each draw, with their count"""
    each = {}
    for name in ('fit', 'check', 'out'):
        d = Draw(root, name)
        rows, misses = d.labels(place.BOUNDS)
        each[name] = np.abs([m for i, ms in enumerate(misses) if d.right[i] and rows[i, 5] > 0 for m in ms])
    every = np.concatenate(list(each.values()))
    q = float(np.quantile(every, REACH))
    return int(np.ceil(q / place.FRAME)), q, {k: float(np.quantile(v, REACH)) for k, v in each.items()}, len(every)

def stretches(d, r):
    """a corpus's placed starts and the stretches they make under reach r, those that touch one (train.py's)"""
    starts = runs = 0
    for f in sorted(glob.glob(f'{d}/loop_*.json')):
        label = json.load(open(f))
        at = sorted(e['frame'] for e in label['events'] if e['drum'] < 0) if label.get('placed') else []
        starts, runs = starts + len(at), runs + (1 + sum(b - a > 2 * r + 3 for a, b in zip(at, at[1:])) if at else 0)
    return starts, runs

if __name__ == '__main__' and sys.argv[2:3] == ['reach']:
    r, q, by, n = reach(os.path.abspath(sys.argv[1]))
    print(f'reach {r} frames: {1000 * q:.0f} ms, the {REACH:g} quantile of how far {n} placed starts of the right takes that pass',
          'sit from their known ones under place.py\'s bounds (' + ', '.join(f'{k} {1000 * v:.0f} ms' for k, v in by.items()) + ')', flush=True)
    for d in sys.argv[3:]:
        m = json.load(open(f'{d}/corpus.json'))
        m['reach'] = r
        with open(f'{d}/corpus.json.new', 'w') as f:
            json.dump(m, f, separators=(',', ':'))
        os.replace(f'{d}/corpus.json.new', f'{d}/corpus.json')
        print(f'{d}: its %d placed starts make %d stretches; reach {r} written to its corpus.json' % stretches(d, r), flush=True)
elif __name__ == '__main__':
    root, t0, rng = os.path.abspath(sys.argv[1]), time.time(), np.random.default_rng(1)
    draw = Draw(root, 'fit')
    right = draw.right
    print(f'the fit draw: {len(draw.takes)} takes, {int(right.sum())} right ({sum(1 for t in draw.takes if t["off"])} with a note sung off),',
          f'wrong: {dict(collections.Counter(t["wrong"] for t in draw.takes if t["wrong"]))}; as first tried {place.TRIED}', flush=True)
    b = dict(place.TRIED)
    print('the placing, on the right takes:', flush=True)
    b, moved1 = fit(draw, PLACING, b, lambda rows, pick=None: placing(rows[right], pick), [np.arange(int(right.sum()))], rng, t0)
    print('the test, on every take:', flush=True)
    b, moved2 = fit(draw, TEST, b, lambda rows, pick=None: test(rows, right, pick), [np.arange(int(right.sum())), np.arange(int((~right).sum()))], rng, t0)
    moved = {**moved1, **moved2}
    print('fitted:', b, '| moved:', {k: f'{v[0]:g} to {v[1]:g}' for k, v in moved.items()}, flush=True)
    summary = {}
    for name in ('fit', 'check', 'out'):
        d = draw if name == 'fit' else Draw(root, name)
        print(f'\n== the {name} draw ({len(d.takes)} takes)')
        first = report(d, dict(place.TRIED), 'as first tried')
        summary[name] = dict(first_tried=first, fitted=report(d, b, 'fitted'))
        print(f'[{time.time() - t0:.0f} s]', flush=True)
    if len(sys.argv) > 2 and sys.argv[2] == 'write':
        json.dump(dict(bounds=b, first_tried=place.TRIED, moved={k: list(v) for k, v in moved.items()}, scores=summary,
                       fitted=f'by placefit.py on {os.path.basename(root)} at {time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}'), open(place.FITTED, 'w'), indent=1)
        print('wrote', place.FITTED)
