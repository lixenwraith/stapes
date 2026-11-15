#!/usr/bin/env python3
"""trackfit.py <truth root> [write]: the free tracker's bounds (vocal.py) fitted on lines whose notes are known
(lines.py's six draws under <truth root>), a bound at a time, a value taken only if its gain holds over a thousand
redraws, then checked on another seed and on voices left out. With `write` the bounds go to bounds.json beside
vocal.py. ../README.md has the procedure."""
import collections, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vocal, trackscore as ts

GRID = collections.OrderedDict(
    lp=[0, 2500, 1800, 1300, 1000, 800, 600], dip=[.08, .12, .16, .2, .25], ap_max=[.15, .2, .25, .3, .4, .5, .65, .8],
    top=[-50., -44., -38., -32., -26.], move=[3., 4., 6., 8., 11., 15.], gap=[1, 3, 5, 8, 12], minlen=[3, 4, 6, 8, 11],
    join=[0, 5, 10, 16], slip_len=[0., .15, .25, .4, .6], slip_far=[5, 7, 9, 12],
    blip_len=[0., .1, .15, .2, .3, .45], blip_near=[1, 2], slide_len=[0., .1, .15, .2, .3, .45], slide_share=[.3, .5, .7])

class Draw:
    """a draw's lines, with each line's frames kept by the two bounds that shape them"""
    def __init__(self, root, name):
        self.name = name
        self.lines = ts.load(f'{root}/engine-{name}') + ts.load(f'{root}/nsynth-{name}')
        self.kinds, self.frames = [l['kind'] for l in self.lines], {}
    def run(self, b, offsets=False):
        key = (b['lp'], b['dip'])
        if key not in self.frames:
            self.frames[key] = [vocal.frames(l['x'], *key) for l in self.lines]
        rows, offs = [], []
        for l, fr in zip(self.lines, self.frames[key]):
            c, o = ts.components(l, vocal.slips(vocal.cut(*fr, **b), **b)[0])
            rows.append([c[k] for k in ts.SUMS])
            offs.append(o)
        rows = np.array(rows, float)
        return (rows, offs) if offsets else rows
    def score(self, rows):
        return ts.score(rows, self.kinds)

def gain(draw, held, tried, rng, n=1000):
    """how much better `tried` scores than `held`, and that gain's 2.5th percentile over the lines drawn again"""
    idx = {k: np.array([i for i, x in enumerate(draw.kinds) if x == k]) for k in sorted(set(draw.kinds))}
    obj = lambda rows, pick: np.mean([ts.figures(rows[pick[k]])['score'] for k in idx])
    draws = []
    for _ in range(n):
        pick = {k: rng.choice(v, len(v)) for k, v in idx.items()}
        draws.append(obj(tried, pick) - obj(held, pick))
    return obj(tried, idx) - obj(held, idx), float(np.percentile(draws, 2.5))

if __name__ == '__main__':
    root, t0, rng = os.path.abspath(sys.argv[1]), time.time(), np.random.default_rng(1)
    fit = Draw(root, 'fit')
    b = dict(vocal.HAND)
    held = fit.run(b)
    print(f'the fit draw: {len(fit.lines)} lines, {sum(len(l["notes"]) for l in fit.lines)} notes, {sum(l["seconds"] for l in fit.lines) / 60:.1f} minutes;',
          f'by hand {b} scores {fit.score(held):.4f}', flush=True)
    moved = collections.OrderedDict()
    for p in range(3):
        took = False
        for name, grid in GRID.items():
            tried = {v: (held if v == b[name] else fit.run({**b, name: v})) for v in grid}
            best = max(grid, key=lambda v: fit.score(tried[v]))
            line = f'pass {p + 1}, {name}: ' + ', '.join(f'{v:g} {fit.score(tried[v]):.4f}' + ('*' if v == b[name] else '') for v in grid)
            if best != b[name]:
                g, low = gain(fit, held, tried[best], rng)
                line += f' | {best:g} gains {g:+.4f}, {low:+.4f} at the 2.5th percentile: ' + ('taken' if low > 0 else 'not told apart, stays')
                if low > 0:
                    moved[name] = (moved[name][0] if name in moved else b[name], best)
                    b[name], held, took = best, tried[best], True
            else:
                line += ' | stays'
            print(line, f'[{time.time() - t0:.0f} s]', flush=True)
        if not took:
            break
    moved = collections.OrderedDict((k, v) for k, v in moved.items() if v[0] != v[1])
    print('fitted:', b, '| moved:', {k: f'{v[0]:g} to {v[1]:g}' for k, v in moved.items()}, flush=True)
    summary = {}
    for name in ('fit', 'check', 'out'):
        d = fit if name == 'fit' else Draw(root, name)
        hand, fitted = d.run(dict(vocal.HAND), True), d.run(b, True)
        g, low = gain(d, hand[0], fitted[0], rng)
        summary[name] = dict(by_hand=round(d.score(hand[0]), 4), fitted=round(d.score(fitted[0]), 4))
        print(f'\n== the {name} draw: by hand {d.score(hand[0]):.4f}, fitted {d.score(fitted[0]):.4f} ({g:+.4f}, {low:+.4f} at the 2.5th percentile)')
        ts.report(f'{name}, the bounds by hand', d.lines, *hand)
        ts.report(f'{name}, the bounds fitted', d.lines, *fitted)
        for k, (was, now) in moved.items():
            back = d.run({**b, k: was})
            g, low = gain(d, back, fitted[0], rng)
            print(f'  {k} put back to {was:g} alone scores {d.score(back):.4f}: at {now:g} it gives {g:+.4f} ({low:+.4f} at the 2.5th percentile)')
        print(f'[{time.time() - t0:.0f} s]', flush=True)
    if len(sys.argv) > 2 and sys.argv[2] == 'write':
        json.dump(dict(bounds=b, by_hand=vocal.HAND, moved={k: list(v) for k, v in moved.items()}, scores=summary,
                       fitted=f'by trackfit.py on {os.path.basename(root)} at {time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}'),
                  open(vocal.FITTED, 'w'), indent=1)
        print('wrote', vocal.FITTED)
