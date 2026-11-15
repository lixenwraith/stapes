#!/usr/bin/env python3
"""shapes.py <vocal-list.csv>...: the notes VocalSet's sheet gives each exercise (SHAPE, which place.py places), and
a list's labels counted against them, which says something of labels a tracker cut and nothing of placed ones."""
import collections, csv, os, sys
UP = [2, 2, 1, 2, 2, 2, 1, 2]
SHAPE = {'arpeggios': [4, 3, 5, -5, -3, -4], 'scales': UP + [-s for s in reversed(UP)], 'long_tones': [12, 5]}

def keys(path):
    """the keys of the notes a MIDI file starts, in order"""
    b = open(path, 'rb').read()
    t, i, out, status = b[22:], 0, [], None
    while i < len(t):
        while t[i] & 0x80:
            i += 1
        i += 1
        if t[i] == 0xff:
            i += 3 + t[i + 2]
            continue
        if t[i] & 0x80:
            status = t[i]
            i += 1
        if status & 0xf0 == 0x90 and t[i + 1] > 0:
            out.append(t[i])
        i += 2
    return out

def table(per):
    """per: {(exercise, technique): [(the keys of a take's notes, its seconds)]}; prints the table and returns each
    exercise's (takes judged, exact count, steps right, more, fewer)"""
    print('  exercise    technique     takes  notes  exact  steps  median   more  fewer  long takes')
    tot = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
    for (c, t), v in sorted(per.items()):
        mid = sorted(d for _, d in v)[len(v) // 2]
        short = [k for k, d in v if d <= 1.6 * mid]
        ns = sorted(len(k) for k in short)
        if c not in SHAPE:
            print(f'  {c:11s} {t:13s} {len(v):5d}      -      -      -   {ns[len(ns) // 2]:5d}      -      -  {len(v) - len(short):4d}')
            continue
        want = len(SHAPE[c]) + 1
        got = [sum(1 for k in short if f(k)) for f in (lambda k: len(k) == want, lambda k: [b - a for a, b in zip(k, k[1:])] == SHAPE[c],
                                                       lambda k: len(k) > want, lambda k: len(k) < want)]
        print(f'  {c:11s} {t:13s} {len(v):5d}  {want:5d}   {got[0] / len(short):.2f}   {got[1] / len(short):.2f}   {ns[len(ns) // 2]:5d}   '
              f'{got[2] / len(short):.2f}   {got[3] / len(short):.2f}  {len(v) - len(short):4d}')
        for i, x in enumerate([len(short)] + got):
            tot[c][i] += x
    for c in sorted(tot):
        a, e, s, m, f = tot[c]
        print(f'  {c}: {e} of {a} takes hold their {len(SHAPE[c]) + 1} notes ({100 * e / a:.0f}%), {s} with every step right ({100 * s / a:.0f}%); {m} hold more, {f} fewer')
    return dict(tot)

if __name__ == '__main__':
    for lst in sys.argv[1:]:
        here = os.path.dirname(os.path.abspath(lst))
        per = collections.defaultdict(list)
        for r in csv.DictReader(open(lst, newline='')):
            _, _, singer, context, tech, name = r['midi_filename'].split('/')
            per[context, tech].append((keys(os.path.join(here, r['midi_filename'])), float(r['duration'])))
        print(lst)
        table(per)
