#!/usr/bin/env python3
"""vocalsing.py frames <labels dir> <cache.pkl> | vocalsing.py <standard|spread|octave|vibrato> <labels dir>
[<cache.pkl>]: how VocalSet's arpeggios, scales and long tones are sung (each singer's pitch standard, the notes a
period's measure reads an octave off, the vibrato), measured from the placed labels and the recordings for the
sets' issues. Writes nothing but the cache; ../README.md says how each is measured."""
import collections, csv, os, pickle, re, sys, wave
import numpy as np
LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LAB + '/labels')
import vocal, place, shapes, lines, placed
V = os.path.join(os.path.expanduser(os.environ.get('VOCALSET', '~/data/rip/vocalset')), 'data_by_singer', '')
ORDER = lambda s: (s[0], int(re.sub(r'\D', '', s)))

def tried(O):
    return {r['recording']: r for r in csv.DictReader(open(f'{O}/vocal-placed.tsv'), delimiter='\t')
            if r['exercise'] != 'excerpts' and r['listed'] != 'a copy of another recording'}

def frames(O, cache):
    out = {}
    for rec in tried(O):
        f0, ap, rms = vocal.frames(vocal.read(V + rec), place.BOUNDS['lp'], vocal.HAND['dip'])
        out[rec] = (f0.astype(np.float32), ap.astype(np.float32), rms.astype(np.float32))
    pickle.dump(dict(lp=place.BOUNDS['lp'], takes=out), open(cache, 'wb'))
    print(len(out), 'takes kept in', cache)

def placings(O, cache, bounds):
    """(recording, its row, pitch, voiced, the placing or nothing) for every tried take, under bounds"""
    held = pickle.load(open(cache, 'rb'))
    assert held['lp'] == bounds['lp'], 'the cache was made under another low-pass'
    rows = tried(O)
    for rec, (f0, ap, rms) in sorted(held['takes'].items()):
        p, voiced = place.pitch(f0.astype(float), ap.astype(float), rms.astype(float), bounds['ap_max'])
        yield rec, rows[rec], p, voiced, place.place(p, voiced, rms.astype(float), shapes.SHAPE[rows[rec]['exercise']], **bounds)

def standard(O):
    rows, agree, by = tried(O), [], collections.defaultdict(list)
    for r in csv.DictReader(open(f'{O}/vocal-list.csv', newline='')):
        rec = r['audio_filename'].split('data_by_singer/')[1]
        s, e, k = placed.notes(f"{O}/{r['midi_filename']}")[0]
        with wave.open(os.path.join(O, r['audio_filename'])) as w:
            sr = w.getframerate()
            x = np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(np.float64) / 32768
        a, b = s + 0.25 * (e - s), s + 0.75 * (e - s)
        if b - a < 0.06:
            continue
        width = min(0.1, b - a)
        reads = sorted(m for m in (lines.sound_pitch(x[int(t * sr):int((t + width) * sr)], sr) for t in np.linspace(a, b - width, 5)) if m > 0)
        if not reads:
            continue
        heard = reads[len(reads) // 2]
        d = heard - (k + float(rows[rec]['cents']) / 100)
        agree.append(d - 12 * round(d / 12))
        key = placed.named(rec)
        v = heard - (0 if key is None else key)             # a take with no key in its name is on C, by the sheet
        by[rec.split('/')[0]].append((100 * (v - 12 * round(v / 12)), rec))
    a = np.array(agree)
    print(f'{len(a)} listed takes read. The second reading of the first note less the label\'s word for it (its key and the take\'s standard): median {100 * np.median(a):+.0f} cents,',
          f'within 25 cents in {np.mean(np.abs(a) <= .25):.3f}, within 50 in {np.mean(np.abs(a) <= .5):.3f}, over a semitone in {np.mean(np.abs(a) > 1):.3f}')
    print('by singer, the first note over the key the take\'s name gives (C where it gives none), cents: median, the middle half, the takes over a semitone and a half up, the highest of them')
    for s in sorted(by, key=ORDER):
        v = np.array([x[0] for x in by[s]])
        top = max(by[s])
        print(f'   {s:9s} {len(v):4d} takes  {np.median(v):+5.0f}  ({np.percentile(v, 25):+5.0f} to {np.percentile(v, 75):+5.0f})  {int((v > 150).sum()):3d}' + (f'  {top[1]} at {top[0]:+.0f}' if top[0] > 150 else ''))
    med = sorted((float(np.median([x[0] for x in v])), s) for s, v in by.items())
    print(f'singers\' medians from {med[0][0]:+.0f} ({med[0][1]}) to {med[-1][0]:+.0f} ({med[-1][1]}); 35 cents or more sharp: {", ".join(s for m, s in med if m >= 35)}; 35 or more flat: {", ".join(s for m, s in med if m <= -35)}')

def spread(O):
    rows, per, notes = tried(O), [], []
    for r in csv.DictReader(open(f'{O}/vocal-list.csv', newline='')):
        row = rows[r['audio_filename'].split('data_by_singer/')[1]]
        if 'fast' in row['technique']:
            continue
        tr = np.loadtxt(f"{O}/{r['pitch']}", delimiter=',', skiprows=1)
        at, v = tr[:, 0], tr[:, 1] > 0
        m = 69 + 12 * np.log2(np.maximum(tr[:, 1], 1e-6) / 440) - float(row['cents']) / 100
        mine = []
        for s, e, k in placed.notes(f"{O}/{r['midi_filename']}"):
            x = m[(at >= s) & (at < e) & v] - k
            if len(x) >= 12:
                mine.append(float(np.median(x[len(x) // 4:len(x) - len(x) // 4])))
        notes += mine
        if len(mine) >= 3:
            per.append(float(np.std(mine)))
    a, sd = np.abs(np.array(notes)), np.array(per)
    print(f'{len(a)} notes of {len(sd)} listed takes that are not fast ones: the settled middle of a note less its key and the take\'s standard is within 10 cents in {np.mean(a <= .1):.2f},',
          f'20 in {np.mean(a <= .2):.2f}, 30 in {np.mean(a <= .3):.2f}, 50 in {np.mean(a <= .5):.2f}')
    print(f'a take\'s own spread, semitones: median {np.median(sd):.3f}; the lowest twentieth under {np.percentile(sd, 5):.3f}, a quarter under {np.percentile(sd, 25):.3f}, three quarters under {np.percentile(sd, 75):.3f}, the highest twentieth over {np.percentile(sd, 95):.3f}')

def octave(O, cache):
    b = {**place.BOUNDS, 'fold': 0.}
    takes, lv, shown, n, second = collections.Counter(), collections.defaultdict(list), [], 0, collections.Counter()
    for rec, row, p, voiced, got in placings(O, cache, b):
        n += 1
        if got is None:
            continue
        tonic, spans, off, _ = got
        steps = shapes.SHAPE[row['exercise']]
        if place.refused(p, voiced, tonic, steps, spans, off, row['exercise'], **b) != 'a note off its key':
            continue
        keys = tonic + np.concatenate([[0], np.cumsum(steps)])
        devs = [float(np.median(p[a:e][voiced[a:e]] - off - k)) if voiced[a:e].any() else 0.0 for (a, e), k in zip(spans, keys)]
        worst = max(devs, key=abs)
        if abs(worst) < 9:
            continue
        sex = 'men' if rec[0] == 'm' else 'women'
        takes[sex, 'over' if worst > 0 else 'under'] += 1
        takes['by', sex, row['exercise'], row['technique']] += 1
        if len(shown) < 3 and worst > 0:
            i = int(np.argmax(devs))
            shown.append(f'{rec} (its note {i + 1}, {440 * 2 ** ((keys[i] + off - 69) / 12):.0f} Hz)')
        x = vocal.read(V + rec)
        for (a, e), k, d in zip(spans, keys, devs):
            if (e - a) * place.FRAME < 0.12 or not (d > 9 or abs(d) < 0.5):
                continue
            seg = x[int((a + (e - a) / 4) * vocal.HOP):int((e - (e - a) / 4) * vocal.HOP)]
            if len(seg) < 1024:
                continue
            size = 1 << 15
            mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), size))
            f = 440 * 2 ** ((k + off - 69) / 12)
            h = [20 * np.log10(mag[int(f * i * 0.96 * size / vocal.SR):int(f * i * 1.04 * size / vocal.SR) + 1].max() + 1e-12) for i in (1, 2, 3, 4)]
            lv['over' if d > 9 else 'right'].append((h[1] - h[0], h[1] - h[2], h[1] - h[3], f))
            if d > 9:                                       # what the frames do inside the note, and a second reader's word
                y = p[a:e][voiced[a:e]] - off - k
                lo, hi = float((np.abs(y) < 1.5).mean()), float((np.abs(y - 12) < 1.5).mean())
                reads = sorted(m for m in (lines.sound_pitch(seg[i:i + int(0.1 * vocal.SR)], vocal.SR) for i in np.linspace(0, max(len(seg) - int(0.1 * vocal.SR), 0), 5).astype(int)) if m > 0)
                heard = reads[len(reads) // 2] - (k + off) if reads else None
                second['flips'] += lo >= 0.1 and hi >= 0.1
                second['notes'] += 1
                second['unread' if heard is None else 'over' if abs(heard - 12) < 1.5 else 'on' if abs(heard) < 1.5 else 'elsewhere'] += 1
    total = sum(v for k, v in takes.items() if k[0] in ('men', 'women'))
    print(f'of {n} takes tried, {total} are refused for a note read 9 semitones or more from its key: men\'s {takes["men", "over"]} over and {takes["men", "under"]} under, women\'s {takes["women", "over"]} over and {takes["women", "under"]} under')
    print('most by technique:', ', '.join(f'{k[1]} {k[2]} {k[3]} {v}' for k, v in sorted(((k, v) for k, v in takes.items() if k[0] == 'by'), key=lambda kv: -kv[1])[:8]))
    print('for example:', '; '.join(shown))
    for k in ('over', 'right'):
        v = np.array(lv[k])
        print(f'notes read {"an octave over" if k == "over" else "within half a semitone, in the same takes"}: {len(v)}; the second harmonic over the first, dB: median {np.median(v[:, 0]):+.0f}, the middle half {np.percentile(v[:, 0], 25):+.0f} to {np.percentile(v[:, 0], 75):+.0f};',
              f'over the third {np.median(v[:, 1]):+.0f} ({np.percentile(v[:, 1], 25):+.0f} to {np.percentile(v[:, 1], 75):+.0f}); over the fourth {np.median(v[:, 2]):+.0f};',
              f'twice the note\'s pitch: {2 * np.percentile(v[:, 3], 5):.0f} to {2 * np.percentile(v[:, 3], 95):.0f} Hz for nine in ten')
    print(f'of the {second["notes"]} notes read an octave over: in {second["flips"]} the period\'s measure holds both octaves, a tenth of the frames or more each;',
          f'the reading by harmonics (lines.sound_pitch, on the note\'s middle) says the octave over for {second["over"]}, the key for {second["on"]}, elsewhere for {second["elsewhere"]}, nothing for {second["unread"]}')

def vibrato(O, cache):
    b = dict(place.BOUNDS)
    ext, rate, notes = collections.defaultdict(list), collections.defaultdict(list), []
    for rec, row, p, voiced, got in placings(O, cache, b):
        if got is None:
            continue
        tonic, spans, off, p = got
        if place.refused(p, voiced, tonic, shapes.SHAPE[row['exercise']], spans, off, row['exercise'], **b):
            continue
        e, r = [], []
        for a, z in spans:
            if (z - a) * place.FRAME < 0.4:
                continue
            x = p[a + (z - a) // 5:z - (z - a) // 5][voiced[a + (z - a) // 5:z - (z - a) // 5]]
            if len(x) < 20:
                continue
            x = x - np.median(x)
            x = x[np.abs(x) < 4]                            # a frame misread by an octave is no vibrato
            if len(x) < 20:
                continue
            e.append((np.percentile(x, 90) - np.percentile(x, 10)) / 2 / 0.951)
            r.append(int((np.diff(np.sign(x)) != 0).sum()) / 2 / (len(x) * place.FRAME))
            notes.append((e[-1], r[-1]))
        if e:
            ext[row['exercise'], row['technique']].append(float(np.median(e)))
            rate[row['exercise'], row['technique']].append(float(np.median(r)))
    print('exercise    technique      takes  extent in cents either way: the median take, three quarters under, a twentieth over, the widest | Hz')
    for k in sorted(ext):
        v = 100 * np.array(ext[k])
        print(f'{k[0]:11s} {k[1]:13s} {len(v):5d}   {np.median(v):5.0f} {np.percentile(v, 75):5.0f} {np.percentile(v, 95):5.0f} {v.max():5.0f} | {np.median(rate[k]):.1f}')
    v = 100 * np.array([x for l in ext.values() for x in l])
    print(f'all {len(v)} takes with a held note: the median take {np.median(v):.0f} cents either way, three quarters under {np.percentile(v, 75):.0f}, nine tenths under {np.percentile(v, 90):.0f},',
          f'a hundredth over {np.percentile(v, 99):.0f}, the widest {v.max():.0f}; over 100 cents: {np.mean(v > 100):.3f} of takes; 80 or more: {np.mean(v >= 80):.3f}')
    n = np.array(notes)
    big = n[n[:, 0] > 0.5]
    print(f'the rate, on the {len(big)} held notes whose extent is over 50 cents: median {np.median(big[:, 1]):.1f} Hz, a twentieth under {np.percentile(big[:, 1], 5):.1f}, a twentieth over {np.percentile(big[:, 1], 95):.1f}')

if __name__ == '__main__':
    what, O = sys.argv[1], os.path.abspath(sys.argv[2])
    if what == 'frames':
        frames(O, sys.argv[3])
    elif what in ('standard', 'spread'):
        {'standard': standard, 'spread': spread}[what](O)
    else:
        {'octave': octave, 'vibrato': vibrato}[what](O, sys.argv[3])
