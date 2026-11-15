#!/usr/bin/env python3
"""label.py <out dir>: VocalSet's singing as MIDI files, pitch tracks and a list for `corpus <dir> from <out
dir>/vocal-list.csv`: a take's notes are its exercise's, as the set's sheet writes them, placed on its sound by
place.py; a take the placing refuses is left out and counted in vocal-placed.tsv. VOCALSET names the set's folder.
../README.md has the rest."""
import collections, csv, hashlib, os, shutil, struct, sys, wave
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # vocal.py, place.py and shapes.py are beside this file, under python -I too
import vocal, place, shapes
H = os.path.join(os.path.expanduser(os.environ.get('VOCALSET', '~/data/rip/vocalset')), '')
V = H + 'data_by_singer/'
O = os.path.abspath(sys.argv[1])
SKIP = {'vocal_fry', 'inhaled', 'spoken', 'trill', 'trillo'}
TEST = {l.strip() for l in open(H + 'test_singers_technique.txt') if l.strip()}
def vlq(n):
    b = [n & 127]
    while n > 127:
        n >>= 7
        b.insert(0, (n & 127) | 128)
    return bytes(b)
def write(path, notes, ppq=480, tempo=500000):
    ev = []
    for t, d, m in notes:
        ev.append((round(t * 2 * ppq), 0x90, m, 80))
        ev.append((round((t + d) * 2 * ppq), 0x80, m, 0))
    ev.sort(key=lambda e: (e[0], e[1] == 0x90))
    tr = bytearray(b'\x00\xff\x51\x03' + tempo.to_bytes(3, 'big'))
    now = 0
    for tick, kind, m, v in ev:
        tr += vlq(tick - now) + bytes([kind, m, v])
        now = tick
    tr += b'\x00\xff\x2f\x00'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(b'MThd' + struct.pack('>IHHH', 6, 0, 1, ppq) + b'MTrk' + struct.pack('>I', len(tr)) + tr)
rel = lambda p: os.path.relpath(p, O)
rows, placed, count, secs, left = [], [], 0, 0.0, collections.Counter()
for own in ('midi', 'pitch'):                            # this tool's own, written afresh a run
    shutil.rmtree(f'{O}/vocalset/{own}', ignore_errors=True)
print('the placing\'s bounds:', place.BOUNDS, '| from', 'place.json, as placefit.py fitted them' if os.path.exists(place.FITTED) else 'place.py, as first tried',
      '| ruled, over the fit:', place.RULED or 'nothing', flush=True)
files, first = [], {}
for d, _, fs in sorted(os.walk(V)):
    for f in sorted(fs):
        p = os.path.join(d, f)
        if f.endswith('.wav') and os.path.relpath(p, V).split('/')[2] not in SKIP:
            h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
            files.append((p, h))
            if h not in first or (len(p), p) < (len(first[h]), first[h]):
                first[h] = p
tried = collections.Counter()
for p, h in files:
    r = os.path.relpath(p, V)
    singer, exercise, tech = r.split('/')[:3]
    split = 'test' if singer in TEST else 'train'
    with wave.open(p) as w:
        long = w.getnframes() / w.getframerate()
    folded = 0
    if first[h] != p:
        why, notes, tonic, cents = 'a copy of another recording', None, 0, 0.0
    elif exercise not in shapes.SHAPE:
        why, notes, tonic, cents = 'an excerpt', None, 0, 0.0
    else:
        tried[exercise] += 1
        notes, why, tonic, cents, (at, hz), folded = place.take(vocal.read(p), exercise)
        if notes and not all(21 <= n[2] <= 108 for n in notes):
            notes, why = None, 'a note outside the 88 keys'
    placed.append((r, split, long, exercise, tech, tonic, cents, len(notes or []), why or 'listed', folded))
    if why:
        left[why] += 1
        continue
    mid, track = f'{O}/vocalset/midi/{r[:-4]}.mid', f'{O}/vocalset/pitch/{r[:-4]}.csv'
    write(mid, notes)
    os.makedirs(os.path.dirname(track), exist_ok=True)
    with open(track, 'w') as f:
        f.write('seconds,hz\n' + ''.join(f'{t:.5f},{v:.3f}\n' for t, v in zip(at, hz)))
    rows.append((rel(p), rel(mid), split, long, 'voice', rel(track), 'measured', singer, 'placed'))  # a singer's takes share a unit
    count += len(notes); secs += long
print('vocalset', len(rows), 'takes listed,', count, 'notes,', round(secs / 3600, 2), 'hours; left out:', dict(left.most_common()), flush=True)
listed = collections.Counter(c[3] for c in placed if c[8] == 'listed')
print('by exercise, the takes listed of those tried:', {e: f'{listed[e]} of {tried[e]}' for e in sorted(tried)}, flush=True)
with open(f'{O}/vocal-placed.tsv', 'w') as f:
    f.write('recording\tsplit\tseconds\texercise\ttechnique\ttonic\tcents\tnotes\tlisted\tfolded\n' + ''.join('%s\t%s\t%.3f\t%s\t%s\t%d\t%.1f\t%d\t%s\t%d\n' % c for c in placed))
with open(f'{O}/vocal-list.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['audio_filename', 'midi_filename', 'split', 'duration', 'family', 'pitch', 'velocity', 'unit', 'starts'])
    w.writerows(rows)
tr = sum(1 for r in rows if r[2] == 'train')
print('list', len(rows), 'rows,', tr, 'train,', round(sum(r[3] for r in rows if r[2] == 'train') / 3600, 2), 'train hours')
