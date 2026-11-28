#!/usr/bin/env python3
"""joinloops.py <out dir> <seed> <corpus dir or truth list>...: recordings of two sections, each two arranged loops laid end
to end as one recording with one label, in four conditions of 40 pairs drawn by random.Random(seed): tempo, the second
loop's tempo 15% or more off the first's, alike in meter and key; meter, its meter another, alike in tempo and key; key,
its scale's tones another, alike in tempo and meter; alike, alike in all three. Alike in tempo is the labels' bpm within
5%, in key the same tonic and scale. The README says what a pair's label holds and what is written."""
import json, os, random, sys, wave

RATE, HOP, PAIRS = 22050, 256, 40   # the labels' frames are of HOP samples at RATE
SCALES = ([0, 1, 3, 5, 7, 8, 10], [0, 2, 3, 5, 7, 8, 10], [0, 2, 3, 5, 7, 8, 11], [0, 2, 3, 5, 7, 9, 10], [0, 3, 5, 7, 10],
          [0, 2, 4, 5, 7, 9, 11])   # audio's ScaleID order: phrygian, minor, harmonic minor, dorian, minor pentatonic, major
NAMES = ('phrygian', 'minor', 'harmonic_minor', 'dorian', 'minor_pent', 'major')
TONICS = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
FIELDS = {'bpm', 'root', 'scale', 'bars', 'meter', 'frames', 'tuning', 'events', 'kind', 'beats', 'downbeats', 'chords'}


def loops(src):
    """every arranged loop a corpus directory or a truth list names, with its WAV of four seconds or more"""
    if os.path.isdir(src):
        names = sorted(f for f in os.listdir(src) if f.startswith('loop_') and f.endswith('.json'))
        pairs = [(os.path.join(src, f[:-5] + '.wav'), os.path.join(src, f)) for f in names]
    else:
        pairs = [tuple(l.split('\t')[:2]) for l in open(src).read().splitlines() if l]
    for w, p in pairs:
        lab = json.load(open(p)) if p.endswith('.json') else {}
        if lab.get('kind') != 'arranged' or not lab.get('beats') or not os.path.exists(w):
            continue
        with wave.open(w) as f:
            if f.getnframes() < 4 * f.getframerate():
                continue
        if set(lab) - FIELDS:
            sys.exit(f'{p}: a field this tool does not join: {sorted(set(lab) - FIELDS)}')
        yield dict(wav=w, label=p, lab=lab, bpm=lab['bpm'], meter=lab.get('meter') or '4/4', key=(lab['root'] % 12, lab['scale']),
                   tones=frozenset((lab['root'] + i) % 12 for i in SCALES[lab['scale']]))


def off(a, b): return max(a['bpm'], b['bpm']) / min(a['bpm'], b['bpm']) - 1


CONDITIONS = (
    ('tempo', lambda a, b: off(a, b) >= 0.15 and a['meter'] == b['meter'] and a['key'] == b['key']),
    ('meter', lambda a, b: off(a, b) <= 0.05 and a['meter'] != b['meter'] and a['key'] == b['key']),
    ('key', lambda a, b: off(a, b) <= 0.05 and a['meter'] == b['meter'] and a['tones'] != b['tones']),
    ('alike', lambda a, b: off(a, b) <= 0.05 and a['meter'] == b['meter'] and a['key'] == b['key']),
)


def draw(pop, rng):
    """each condition's pairs, no loop twice within one, then its fit half: the candidates in the population's order,
    shuffled, each taken unless a loop of it is taken already"""
    out = []
    for name, ok in CONDITIONS:
        cand = [(a, b) for a in pop for b in pop if a is not b and ok(a, b)]
        rng.shuffle(cand)
        taken, used = [], set()
        for a, b in cand:
            if a['label'] not in used and b['label'] not in used and len(taken) < PAIRS:
                taken.append((a, b))
                used |= {a['label'], b['label']}
        if len(taken) < PAIRS:
            sys.exit(f'{name}: {len(taken)} pairs of loops not used twice, of {len(cand)} candidates; {PAIRS} are drawn')
        fit = set(rng.sample(range(PAIRS), PAIRS // 2))
        out += [(name, 'fit' if k in fit else 'read', a, b) for k, (a, b) in enumerate(taken)]
    return out


def join(a, b, stem):
    """a's sound, padded to its label's whole frames, then b's; one label: a's fields but for the frames and bars, which
    add, a's events ending at its end, b's events, beats, downbeats and chords moved by a's frames, and sections"""
    sound = []
    for x in (a, b):
        with wave.open(x['wav']) as f:
            sound.append((f.getparams(), f.readframes(f.getnframes())))
    (pa, sa), (pb, sb) = sound
    if pa[:3] != pb[:3] or pa.framerate % RATE:
        sys.exit(f'{a["wav"]}, {b["wav"]}: channels, width or rate differ, or the rate is no multiple of {RATE}')
    la, lb = a['lab'], b['lab']
    hop, f1 = HOP * pa.framerate // RATE, la['frames']
    if not pa.nframes <= f1 * hop < pa.nframes + hop:
        sys.exit(f'{a["label"]}: {la["frames"]} frames, its WAV {pa.nframes} samples')
    pad = bytes((f1 * hop - pa.nframes) * pa.nchannels * pa.sampwidth)
    with wave.open(stem + '.wav', 'wb') as f:
        f.setparams(pa)
        f.writeframes(sa + pad + sb)
    lab = dict(la, frames=f1 + lb['frames'], bars=la['bars'] + lb['bars'])
    lab['events'] = [dict(e, end=min(e['end'], f1)) for e in la['events']] + [dict(e, frame=e['frame'] + f1, end=e['end'] + f1) for e in lb['events']]
    for k in ('beats', 'downbeats'):
        lab[k] = la.get(k, []) + [f + f1 for f in lb.get(k, [])]
    lab['chords'] = la.get('chords', []) + [dict(c, frame=c['frame'] + f1) for c in lb.get('chords', [])]
    lab['sections'] = [dict(frame=at, bpm=x['bpm'], meter=x['meter'], root=x['lab']['root'], scale=x['lab']['scale']) for at, x in ((0, a), (f1, b))]
    with open(stem + '.json', 'w') as f:
        json.dump(lab, f, separators=(',', ':'))
    return f1


def key(x): return f'{TONICS[x["key"][0]]} {NAMES[x["key"][1]]}'


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    out, seed = sys.argv[1], int(sys.argv[2])
    pop = [x for src in sys.argv[3:] for x in loops(src)]
    os.makedirs(out, exist_ok=True)
    with open(f'{out}/list.tsv', 'w') as lst, open(f'{out}/pairs.tsv', 'w') as rec:
        rec.write('n\tcondition\thalf\tboundary\tfirst\tsecond\tbpm\tmeter\tkey\ttuning\n')
        for n, (name, half, a, b) in enumerate(draw(pop, random.Random(seed))):
            stem = f'{out}/loop_{n:03d}'
            f1 = join(a, b, stem)
            lst.write(f'{stem}.wav\t{stem}.json\t{name}-{half}\ttimed\n')
            rec.write(f'{n}\t{name}\t{half}\t{f1}\t{a["label"]}\t{b["label"]}\t{a["bpm"]} {b["bpm"]}\t{a["meter"]} {b["meter"]}\t'
                      f'{key(a)}, {key(b)}\t{a["lab"]["tuning"]:+.1f} {b["lab"]["tuning"]:+.1f}\n')
    print(f'{len(pop)} arranged loops of four seconds or more; {len(CONDITIONS) * PAIRS} recordings of two sections in {out}, '
          f'seed {seed}: list.tsv (a condition and its half a line), pairs.tsv')


main()
