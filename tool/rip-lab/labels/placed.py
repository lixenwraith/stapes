#!/usr/bin/env python3
"""placed.py <labels dir>: VocalSet's placed labels read back from what label.py wrote there (the table of takes, the
list, the MIDI files and pitch tracks), with no sound read: tonics against the takes' names, notes against their own
pitch, the singers, the fold and squeezed neighbours. Run after every run of label.py; ../README.md has each count."""
import collections, csv, os, re, sys
import numpy as np
NAMES = 'C C# D D# E F F# G G# A A# B'.split()
REASONS = ['a note off its key', 'a note too short', 'a silence inside a note', 'sound that is not its notes', 'a note far shorter than the others',
           'too little sound', 'a note outside the 88 keys']

def notes(path):
    """(start s, end s, key) of every note of a MIDI file label.py wrote: 480 ticks a quarter at 120 a minute"""
    t = open(path, 'rb').read()[22:]
    i, tick, on, out = 0, 0, {}, []
    while i < len(t):
        d = 0
        while True:
            d = d << 7 | t[i] & 127
            i += 1
            if t[i - 1] < 128:
                break
        tick += d
        if t[i] == 0xff:
            i += 3 + t[i + 2]
            continue
        kind, k = t[i] & 0xf0, t[i + 1]
        i += 3
        if kind == 0x90:
            on[k] = tick
        else:
            out.append((on.pop(k) / 960, tick / 960, k))
    return sorted(out)

def settled(x):
    """where a note's pitch settles: the median of the half of its frames, one after another, that moves least"""
    w = max(3, len(x) // 2)
    if len(x) <= w:
        return float(np.median(x))
    runs = np.lib.stride_tricks.sliding_window_view(x, w)
    return float(np.median(runs[int(np.argmin(runs.std(axis=1)))]))

def named(rec):
    """the key a take's name gives, as a pitch class, or nothing"""
    m = re.search(r'_(c|f)(_|$)', os.path.basename(rec)[:-4])
    return {'c': 0, 'f': 5}[m.group(1)] if m else None

if __name__ == '__main__':
    O = os.path.abspath(sys.argv[1])
    rows = [r for r in csv.DictReader(open(f'{O}/vocal-placed.tsv'), delimiter='\t') if r['exercise'] != 'excerpts' and r['listed'] != 'a copy of another recording']
    per = collections.defaultdict(list)
    for r in rows:
        per[r['exercise'], r['technique']].append(r)
    seen = [w for w in REASONS if any(r['listed'] == w for r in rows)]
    print('1. takes tried, listed and refused\n   exercise    technique     tried listed  ' + '  '.join(seen))
    for (c, t), v in sorted(per.items()):
        why = collections.Counter(r['listed'] for r in v)
        print(f'   {c:11s} {t:13s} {len(v):5d} {why["listed"]:6d}  ' + '  '.join(f'{why[w]:{len(w)}d}' for w in seen))
    why = collections.Counter(r['listed'] for r in rows)
    print(f'   {"all":25s} {len(rows):5d} {why["listed"]:6d}  ' + '  '.join(f'{why[w]:{len(w)}d}' for w in seen))

    print('2. the tonic against the key in the take\'s name: the named key, a semitone from it, elsewhere')
    odd = []
    for what, pick in (('listed', lambda r: r['listed'] == 'listed'), ('refused', lambda r: r['listed'] != 'listed')):
        for c in ('arpeggios', 'scales'):
            got = collections.Counter()
            for r in rows:
                k = named(r['recording'])
                if k is None or r['exercise'] != c or not pick(r) or r['listed'] == 'too little sound':
                    continue
                d = (int(r['tonic']) - k + 6) % 12 - 6
                got['same' if d == 0 else 'semitone' if abs(d) == 1 else 'elsewhere'] += 1
                if abs(d) > 1 and what == 'listed':
                    odd.append(f"{r['recording']} ({NAMES[int(r['tonic']) % 12]})")
            n = max(sum(got.values()), 1)
            print(f'   {what:8s} {c:10s} {sum(got.values()):5d} takes: {got["same"]:4d} ({got["same"] / n:.2f})  {got["semitone"]:4d} ({got["semitone"] / n:.2f})  {got["elsewhere"]:4d} ({got["elsewhere"] / n:.2f})')
    if odd:
        print('   listed with a tonic elsewhere:', ', '.join(odd[:12]) + (f' and {len(odd) - 12} more' if len(odd) > 12 else ''))

    placed = {r['recording']: r for r in rows}
    note, spot, takes = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    pairs = collections.Counter()
    for r in csv.DictReader(open(f'{O}/vocal-list.csv', newline='')):
        p = placed[r['audio_filename'].split('data_by_singer/')[1]]
        c, t, off = p['exercise'], p['technique'], float(p['cents']) / 100
        tr = np.loadtxt(f"{O}/{r['pitch']}", delimiter=',', skiprows=1)
        at, v = tr[:, 0], tr[:, 1] > 0
        m = 69 + 12 * np.log2(np.maximum(tr[:, 1], 1e-6) / 440) - off
        ns = notes(f"{O}/{r['midi_filename']}")
        ln = np.array([e - s for s, e, _ in ns])
        mid = float(np.median(ln[:-1])) if len(ln) > 3 else float(np.median(ln))
        short = 0
        for i, (s, e, k) in enumerate(ns):
            x = m[(at >= s) & (at < e) & v] - k
            if len(x):
                row = (settled(x), float((np.abs(x) <= 0.5).mean()), (e - s) / mid)
                note[c, t].append(row)
                spot[c, 'fast' if 'fast' in t else 'slow', i].append(float(np.median(x)))
                short += row[2] < 0.25
        takes[c, t].append(short > 0)
        sheet = [(e - s) / (2 if c == 'scales' and i == len(ns) - 1 else 1) for i, (s, e, _) in enumerate(ns)]
        speed = 'fast' if 'fast' in t else 'slow'
        for i in range(len(ns) - 1):
            if ns[i + 1][0] - ns[i][1] < 0.03:
                step = abs(ns[i + 1][2] - ns[i][2])
                kind = 'a semitone' if step == 1 else 'a tone' if step == 2 else 'further'
                pairs[speed, kind, 'pairs'] += 1
                pairs[speed, kind, 'two to one'] += max(sheet[i], sheet[i + 1]) >= 2 * min(sheet[i], sheet[i + 1])
    def line(v):
        d, s, l = (np.array([x[i] for x in v]) for i in range(3))
        return f'{len(v):6d}  {np.mean(np.abs(d) <= .25):.3f}  {np.mean(np.abs(d) <= .5):.3f}   {np.mean(s):.3f}   {np.mean(l < .5):.3f}  {np.mean(l < .25):.3f}'
    print('3. the notes of the listed takes against the take\'s own pitch\n   exercise    technique      notes  <=.25   <=.5  in key  <half  <quarter  takes with a note under a quarter of their median note')
    for k in sorted(note):
        print(f'   {k[0]:11s} {k[1]:13s} ' + line(note[k]) + f'   {sum(takes[k]):4d} of {len(takes[k])}')
    print(f'   {"all":25s} ' + line([x for v in note.values() for x in v]) + f'   {sum(sum(v) for v in takes.values()):4d} of {sum(len(v) for v in takes.values())}')
    print('4. by the note\'s place in the bar, the median of the notes\' median pitch less their key, cents')
    for c in ('arpeggios', 'scales', 'long_tones'):
        for sp in ('fast', 'slow'):
            if (c, sp, 0) in spot:
                print(f'   {c:10s} {sp}: ' + ' '.join(f'{100 * np.median(spot[c, sp, i]):+.0f}' for i in range(17) if (c, sp, i) in spot))
    by = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        s = r['recording'].split('/')[0]
        by[s][0] += 1
        by[s][1] += r['listed'] == 'listed'
    print('5. by singer, the takes listed of those tried: ' + ', '.join(f'{s} {b}/{a}' for s, (a, b) in sorted(by.items(), key=lambda kv: (kv[0][0], int(re.sub(r'\D', '', kv[0]))))))
    for sex in 'fm':
        a, b = (sum(v[i] for s, v in by.items() if s[0] == sex) for i in (0, 1))
        print(f'   {"women" if sex == "f" else "men"}: {b} of {a} ({b / max(a, 1):.2f})')
    if 'folded' in rows[0]:
        frames = {r['audio_filename'].split('data_by_singer/')[1]: sum(1 for l in open(f"{O}/{r['pitch']}") if not l.endswith(',0.000\n')) - 1
                  for r in csv.DictReader(open(f'{O}/vocal-list.csv', newline=''))}
        got = collections.Counter()
        for r in rows:
            if r['listed'] == 'listed' and int(r['folded']):
                got[r['recording'][0], 'some'] += 1
                got[r['recording'][0], 'a tenth'] += int(r['folded']) >= 0.1 * frames[r['recording']]
        print('6. listed takes holding frames the fold moved an octave: ' + ', '.join(
            f'{"women" if x == "f" else "men"} {got[x, "some"]} ({got[x, "a tenth"]} with a tenth of their voiced frames or more)' for x in 'fm'))
    print('7. of two joined neighbours, the longer twice the shorter or more: ' + '; '.join(
        f'{sp} takes, ' + ', '.join(f'{k} apart {pairs[sp, k, "two to one"]} of {pairs[sp, k, "pairs"]} ({pairs[sp, k, "two to one"] / max(pairs[sp, k, "pairs"], 1):.3f})'
                                    for k in ('a semitone', 'a tone', 'further')) for sp in ('slow', 'fast')))
