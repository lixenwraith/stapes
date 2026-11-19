#!/usr/bin/env python3
"""corpuscount.py <corpus dir> [nowav]: a corpus counted from its labels, its WAV files and its frames' headers
alone, with no code of stapes or of the trainer: its loops by kind and meter, the notes each key starts, the overlaps'
instruments, intervals and velocities, a kit's strikes by piece, its genes' cover and its places, a recording's hits,
lag and the hits its label leaves out, and each WAV's level against the level a loop is laid at. Exits 1 on what may
not be; the README says what it counts and what may not be."""
import collections, glob, json, math, os, struct, sys, wave
import numpy as np
D = sys.argv[1].rstrip('/'); WAVS = 'nowav' not in sys.argv[2:]
QUIET, LOUD, PEAK, HOP = -32.0, -14.0, -1.0, 256 / 22050
BANDS, PLACES = 315, range(2, 33)   # the frames' bands, and the counts of places they are cut into evenly
labs = {int(os.path.basename(p)[5:-5]): json.load(open(p)) for p in sorted(glob.glob(D + '/loop_*.json'))}
bad = collections.defaultdict(list)
kinds = collections.Counter(l.get('kind', 'recorded' if l.get('source') else '?') for l in labs.values())
skip = set(open(D + '/skip.txt').read().split()) if os.path.exists(D + '/skip.txt') else set()
hits = collections.Counter(); unasked = 0; lags = []; left = collections.Counter(); frames = 0; gone = collections.Counter(); gone_keys = collections.Counter()
table = [r.split('\t') for r in open(D + '/left.tsv').read().split('\n')[1:] if r] if os.path.exists(D + '/left.tsv') else []
keys = collections.Counter(); regs = collections.Counter(); low = high = total = 0; fam = collections.Counter(); recorded = engine = 0; vels = []
meters = collections.Counter(); pickup = 0; parts = collections.Counter(); insts = collections.Counter(); most = collections.Counter(); soft = again = ov_notes = 0; rel = collections.Counter()
man = json.load(open(D + '/corpus.json')) if os.path.exists(D + '/corpus.json') else None
sizes = collections.Counter(); struck = []; by_size = collections.defaultdict(list); genes = []; bands = []; crowded = pairs = 0; shared = collections.Counter()
for i, l in labs.items():
    ev = l['events']; notes = [e for e in ev if e['drum'] < 0 and e.get('midi', 0) > 0]
    for e in ev:
        if e['frame'] < 0 or e['frame'] >= l['frames'] or e['end'] < e['frame']: bad['an event out of its loop or ending before it starts'].append(f'loop {i}')
        unasked += e['vel'] == -1
        if e['drum'] >= 0 or e.get('midi', 0) == 0: hits[('kick', 'snare', 'hihat', 'a strike of no line')[e['drum']]] += 1
    if l.get('source'):  # a recording: its hours, the lag its hits were moved by or why it is left out, which skip.txt must hold
        frames += l['frames']
        if 'lag' in l: lags.append(l['lag'])
        if 'skip' in l: left[l['skip']] += 1
        if 'skip' in l and ('lag' in l or f'loop_{i:03d}' not in skip): bad['a loop left out by its label that states a lag, or that skip.txt lacks'].append(f'loop {i}')
        for h in l.get('left', []):  # a hit its label leaves out, which only a loop put on its sound has
            gone[h['why']] += 1; gone_keys[h['key']] += 1
            if 'lag' not in l or not 0 <= h['frame'] < l['frames']: bad['a hit left out of a label that states no lag, or out of its loop'].append(f'loop {i}')
    for e in notes:
        k = e['midi']; keys[k] += 1; total += 1; vels += [e['vel']] * (e['vel'] != -1); fam[e.get('family', -1)] += 1
        if e.get('voice'): engine += 1
        else: recorded += 1
        if not 21 <= k <= 108: bad['a note outside the 88 keys'].append(f'loop {i} midi {k}')
        regs['A0 to G#1' if k < 33 else 'A1 to B3' if k < 60 else 'C4 to C6' if k < 85 else 'C#6 to C8'] += 1
        low += e.get('part') == 'bass' and k < 33; high += e.get('part') in ('melody', 'arpeggio') and k > 96
        if not 0 < e['vel'] <= 1 and e['vel'] != -1: bad['a velocity outside (0, 1] that is not -1, which asks nothing'].append(f'loop {i}')
    kit, strikes = l.get('kit') or [], [e for e in ev if 'piece' in e]
    if strikes and not kit: bad['a strike of a loop with no kit'].append(f'loop {i}')
    if kit:  # a kit drawn from the struck genome: each piece's genes and band once, each strike its piece's number
        sizes[len(kit)] += 1; count = collections.Counter(e['piece'] for e in strikes); at = collections.defaultdict(set)
        if not 2 <= len(kit) <= 10: bad['a kit of fewer than two pieces or more than ten'].append(f'loop {i}')
        for p in kit:
            genes.append(p['genes']); bands.append(p['band'])
            if len(p['genes']) != len((man or {}).get('struck_genes') or p['genes']) or not all(0 <= g <= 1 for g in p['genes']) or not 0 <= p['band'] < BANDS:
                bad['a piece whose genes are not the manifest\'s, each in [0, 1], or whose band is not one of the 315'].append(f'loop {i}')
        for e in strikes:
            if not 0 <= e['piece'] < len(kit): bad['a strike of a piece its kit lacks'].append(f'loop {i} piece {e["piece"]}'); continue
            if e['drum'] != -1 or e.get('midi', 0) != 0 or e.get('instr'): bad['a strike that names a drum, a key or an instrument'].append(f'loop {i}')
            at[e['frame']].add(e['piece'])
        for n in range(len(kit)):
            struck.append(count[n]); by_size[len(kit)].append(count[n])
            if not count[n]: bad['a piece of a kit that strikes nothing'].append(f'loop {i} piece {n}')
        for ps in at.values():  # every two pieces struck on one frame: whether their bands fall in one place
            crowded += len(ps) > 1; ps = sorted(ps)
            for a in range(len(ps)):
                for b in ps[a + 1:]:
                    pairs += 1
                    for n in PLACES: shared[n] += kit[ps[a]]['band'] * n // BANDS == kit[b]['band'] * n // BANDS
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
print(f'bass notes under A1: {low}; melody and arpeggio notes over C7: {high}; velocities from {min(vels, default=0):.3f} to {max(vels, default=0):.3f} over {len(vels)} notes; notes of family unknown or none: {fam[8] + fam[-1]} of {total}')
arr = max(kinds.get('arranged', 0), 1)
print('arranged, by meter: ' + ', '.join(f'{m} {n} ({100 * n / arr:.0f}%)' for m, n in meters.most_common()) + f'; starting part-way into a bar: {pickup} ({100 * pickup / arr:.1f}%)')
print('arranged, by parts: ' + ', '.join(f'{k} parts {n}' for k, n in sorted(parts.items())))
ov = max(kinds.get('overlap', 0), 1)
print(f'overlaps, by instruments: ' + ', '.join(f'{k}: {n}' for k, n in sorted(insts.items())) + f'; one alone in {100 * insts[1] / ov:.0f}%')
print('overlaps, by the most notes sounding at once: ' + ', '.join(f'{k}: {n}' for k, n in sorted(most.items())))
print(f'overlaps: {ov_notes} notes, {soft} of them 15 dB down or more, {again} struck again while they ring; a note against each one sounding within an octave: a unison, third, fifth or octave {sum(rel[k] for k in (0, 3, 4, 7, 12)) / 5:.0f} times an interval, any other {sum(rel[k] for k in (1, 2, 5, 6, 8, 9, 10, 11)) / 8:.0f} times')
if sizes:
    names = (man or {}).get('struck_genes') or [f'gene {k}' for k in range(len(genes[0]))]
    if not (man or {}).get('struck_genes') and man is not None: bad['a corpus of kits whose manifest names no struck genes'].append('corpus.json')
    print(f'kits: {sum(sizes.values())} loops, by size: ' + ', '.join(f'{k} {n}' for k, n in sorted(sizes.items())) + f'; {len(struck)} pieces, {sum(struck)} strikes; '
          f'strikes a piece: fewest {min(struck)}, median {np.median(struck):.0f}, most {max(struck)}; pieces of one strike: {sum(c == 1 for c in struck)}')
    print('strikes a piece by kit size, the median and the fewest: ' + ', '.join(f'{k}: {np.median(v):.0f}, {min(v)}' for k, v in sorted(by_size.items())))
    g = np.array(genes); tenths = lambda v: np.histogram(v, 10, (0, 1))[0]
    print(f'each gene over its range, its {len(g)} pieces by tenths, the fewest and the most a tenth holds, and the least and most value: ' + '; '.join(
        f'{names[k]} {tenths(g[:, k]).min()} to {tenths(g[:, k]).max()} ({g[:, k].min():.2f} to {g[:, k].max():.2f})' for k in range(g.shape[1])))
    print('bands by tenths of the 315: ' + ' '.join(str(c) for c in np.histogram(bands, 10, (0, BANDS))[0]))
    print(f'strikes of two pieces or more on one frame: {crowded} frames, {pairs} pairs of pieces; of the pairs, the share whose pieces\' bands fall in one place, the 315 cut evenly into n places: '
          + ', '.join(f'{n} {100 * shared[n] / max(pairs, 1):.0f}%' for n in PLACES))
if kinds['recorded']:
    print(f'recorded: {frames * HOP / 3600:.2f} hours; hits: ' + (', '.join(f'{n} {k}' for k, n in sorted(hits.items())) or 'none') + f'; events of velocity -1, which ask nothing: {unasked}')
    print(f'hits put on their sound in {len(lags)} loops' + (f', by a lag from {1000 * min(lags):+.1f} to {1000 * max(lags):+.1f} ms, median {1000 * float(np.median(lags)):+.1f}' if lags else '') + f'; left out: {sum(left.values())}' + ''.join(f', {n} for {k}' for k, n in sorted(left.items())) + f'; skip.txt names {len(skip)} loops')
    print(f'hits left out of their labels: {sum(gone.values())}' + ''.join(f', {n} for {k}' for k, n in sorted(gone.items())) + '; by key: ' + (', '.join(f'{k} {n}' for k, n in sorted(gone_keys.items())) or 'none') + f'; left.tsv rows: {len(table)}')
if sorted((r[0], int(r[3]), int(r[4]), int(r[5]), r[7]) for r in table) != sorted((f'loop_{i:03d}', h['frame'], h['key'], h['drum'], h['why']) for i, l in labs.items() for h in l.get('left', [])):
    bad["left.tsv's rows that are not the hits its labels leave out"].append(f'{len(table)} rows')
if WAVS:
    lv, pk, silent, off, short, framed = [], [], 0, [], 0, 0
    for i, l in labs.items():
        p = f'{D}/loop_{i:03d}.wav'
        if not os.path.exists(p):  # a corpus of frames alone: the loop is as long as its frames' header says, and whole
            b = p[:-4] + '.bands'; head = struct.unpack('<II', open(b, 'rb').read(8)) if os.path.exists(b) and os.path.getsize(b) >= 8 else (-1, 0)
            if head[0] != l['frames'] or os.path.getsize(b) != 8 + 4 * head[0] * head[1]: bad['a loop with no WAV whose frames are not its label\'s, or not whole'].append(f'loop {i}')
            framed += 1; continue
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
    if framed: print(f'loops with no WAV, counted from their labels and their frames\' headers: {framed}')
    if len(lv): print(f'WAVs: loudest half second from {lv.min():.1f} to {lv.max():.1f} dBFS, median {np.median(lv):.1f}; by thirds of -32 to -14: {int((lv < -26).sum())}, {int(((lv >= -26) & (lv < -20)).sum())}, {int((lv >= -20).sum())}; outside -32.5 to -13.5: {len(off)} {off[:6]}; peaks up to {max(pk):.2f} dBFS; silent {silent}')
    for i, v in off: bad['a level outside -32 to -14 dBFS'].append(f'loop {i} at {v}')
for k, v in bad.items(): print(f'MAY NOT BE, {k}: {len(v)}; first: ' + ' | '.join(v[:4]))
print('nothing that may not be' if not bad else f'{sum(map(len, bad.values()))} things that may not be')
sys.exit(1 if bad else 0)
