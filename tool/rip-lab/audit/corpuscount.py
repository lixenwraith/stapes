#!/usr/bin/env python3
"""corpuscount.py <corpus dir> [nowav]: a rendered corpus counted from its labels and its WAV files alone, with no
code of stapes or of the trainer: its loops by kind and meter, the notes each key starts, the overlaps' instruments,
intervals and velocities, and each WAV's level against the level a loop is laid at. Exits 1 on what may not be (a
note outside its loop or the 88 keys, a beat in an overlap, a silent or clipped loop). The README says what it counts."""
import collections, glob, json, math, os, sys, wave
import numpy as np
D = sys.argv[1].rstrip('/'); WAVS = 'nowav' not in sys.argv[2:]
QUIET, LOUD, PEAK, HOP = -32.0, -14.0, -1.0, 256 / 22050
labs = {int(os.path.basename(p)[5:-5]): json.load(open(p)) for p in sorted(glob.glob(D + '/loop_*.json'))}
bad = collections.defaultdict(list)
kinds = collections.Counter(l.get('kind', '?') for l in labs.values())
keys = collections.Counter(); regs = collections.Counter(); low = high = total = 0; fam = collections.Counter(); recorded = engine = 0; vels = []
meters = collections.Counter(); pickup = 0; parts = collections.Counter(); insts = collections.Counter(); most = collections.Counter(); soft = again = ov_notes = 0; rel = collections.Counter()
for i, l in labs.items():
    ev = l['events']; notes = [e for e in ev if e['drum'] < 0 and e.get('midi', 0) > 0]
    for e in ev:
        if e['frame'] < 0 or e['frame'] >= l['frames'] or e['end'] < e['frame']: bad['an event out of its loop or ending before it starts'].append(f'loop {i}')
    for e in notes:
        k = e['midi']; keys[k] += 1; total += 1; vels.append(e['vel']); fam[e.get('family', -1)] += 1
        if e.get('voice'): engine += 1
        else: recorded += 1
        if not 21 <= k <= 108: bad['a note outside the 88 keys'].append(f'loop {i} midi {k}')
        regs['A0 to G#1' if k < 33 else 'A1 to B3' if k < 60 else 'C4 to C6' if k < 85 else 'C#6 to C8'] += 1
        low += e.get('part') == 'bass' and k < 33; high += e.get('part') in ('melody', 'arpeggio') and k > 96
        if not 0 < e['vel'] <= 1: bad['a velocity outside (0, 1]'].append(f'loop {i}')
    if l.get('kind') == 'overlap':
        if l.get('beats') or l.get('downbeats') or l.get('chords') or l.get('bpm'): bad['an overlap with a beat, a bar, a chord or a tempo'].append(f'loop {i}')
        insts[len({e['instr'] for e in notes})] += 1; ov_notes += len(notes); soft += sum(e['vel'] <= 10 ** (-15 / 20) for e in notes)
        ring = []; top = 0
        for e in sorted(notes, key=lambda e: e['frame']):
            ring = [r for r in ring if r['end'] > e['frame']]
            again += any(r['instr'] == e['instr'] and r['midi'] == e['midi'] for r in ring)
            for r in ring:
                if abs(r['midi'] - e['midi']) <= 12: rel[abs(r['midi'] - e['midi'])] += 1
            ring.append(e); top = max(top, len(ring))
        most[top] += 1
    elif l.get('kind') == 'arranged':
        if not l.get('beats'): bad['an arranged loop with no beat'].append(f'loop {i}')
        meters[l.get('meter') or '4/4'] += 1; pickup += bool(l.get('downbeats')) and l['downbeats'][0] > 0
        parts[len({e.get('part') for e in ev})] += 1
print(f'{len(labs)} loops: ' + ', '.join(f'{n} {k}' for k, n in sorted(kinds.items())) + f'; {total} notes, {recorded} on recorded instruments and {engine} on the engine\'s voices')
fewest = min(range(21, 109), key=lambda k: keys[k])
print(f'keys: the fewest notes a key of the 88 starts is {keys[fewest]} (midi {fewest}); outside the 88: {sum(n for k, n in keys.items() if not 21 <= k <= 108)}; ' + ', '.join(f'{r} {100 * regs[r] / max(total, 1):.1f}%' for r in ('A0 to G#1', 'A1 to B3', 'C4 to C6', 'C#6 to C8')))
print(f'bass notes under A1: {low}; melody and arpeggio notes over C7: {high}; velocities from {min(vels):.3f} to {max(vels):.3f}; notes of family unknown or none: {fam[8] + fam[-1]} of {total}')
arr = max(kinds.get('arranged', 0), 1)
print('arranged, by meter: ' + ', '.join(f'{m} {n} ({100 * n / arr:.0f}%)' for m, n in meters.most_common()) + f'; starting part-way into a bar: {pickup} ({100 * pickup / arr:.1f}%)')
print('arranged, by parts: ' + ', '.join(f'{k} parts {n}' for k, n in sorted(parts.items())))
ov = max(kinds.get('overlap', 0), 1)
print(f'overlaps, by instruments: ' + ', '.join(f'{k}: {n}' for k, n in sorted(insts.items())) + f'; one alone in {100 * insts[1] / ov:.0f}%')
print('overlaps, by the most notes sounding at once: ' + ', '.join(f'{k}: {n}' for k, n in sorted(most.items())))
print(f'overlaps: {ov_notes} notes, {soft} of them 15 dB down or more, {again} struck again while they ring; a note against each one sounding within an octave: a unison, third, fifth or octave {sum(rel[k] for k in (0, 3, 4, 7, 12)) / 5:.0f} times an interval, any other {sum(rel[k] for k in (1, 2, 5, 6, 8, 9, 10, 11)) / 8:.0f} times')
if WAVS:
    lv, pk, silent, off, short = [], [], 0, [], 0
    for i, l in labs.items():
        p = f'{D}/loop_{i:03d}.wav'
        if not os.path.exists(p): bad['a loop with no WAV'].append(f'loop {i}'); continue
        with wave.open(p) as w:
            sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes(); x = np.frombuffer(w.readframes(n), '<i2').reshape(-1, ch).mean(1) / 32768
        wn = min(int(0.5 * sr), len(x)); e = np.concatenate([[0], np.cumsum(x * x)]); a = np.arange(0, len(x) - wn + 1, max(wn // 2, 1))
        r = math.sqrt(((e[a + wn] - e[a]) / wn).max()); peak = float(np.abs(x).max())
        if r <= 1e-4: silent += 1; bad['a silent loop'].append(f'loop {i}'); continue
        lv.append(20 * math.log10(r)); pk.append(20 * math.log10(peak))
        if not QUIET - 0.5 <= lv[-1] <= LOUD + 0.5: off.append((i, round(lv[-1], 1)))
        if pk[-1] > PEAK + 0.1: bad['a peak over -1 dBFS'].append(f'loop {i} at {pk[-1]:.2f}')
        if abs(n / sr - l['frames'] * HOP) > 2 * HOP: short += 1; bad['a WAV whose length is not its label\'s frames'].append(f'loop {i}: {n / sr:.3f} s, {l["frames"]} frames')
    lv = np.array(lv)
    print(f'WAVs: loudest half second from {lv.min():.1f} to {lv.max():.1f} dBFS, median {np.median(lv):.1f}; by thirds of -32 to -14: {int((lv < -26).sum())}, {int(((lv >= -26) & (lv < -20)).sum())}, {int((lv >= -20).sum())}; outside -32.5 to -13.5: {len(off)} {off[:6]}; peaks up to {max(pk):.2f} dBFS; silent {silent}')
    for i, v in off: bad['a level outside -32 to -14 dBFS'].append(f'loop {i} at {v}')
for k, v in bad.items(): print(f'MAY NOT BE, {k}: {len(v)}; first: ' + ' | '.join(v[:4]))
print('nothing that may not be' if not bad else f'{sum(map(len, bad.values()))} things that may not be')
sys.exit(1 if bad else 0)
