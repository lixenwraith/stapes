#!/usr/bin/env python3
"""guitar.py <out dir>: GuitarSet's notes as MIDI files, a string a channel, and a list for `corpus <dir> from <out
dir>/guitar-list.csv` that names each recording's JAMS file as its pitch track and its beats; from the set's JAMS
files alone (GUITARSET names the set's folder). ../README.md has the rules a note is written by."""
import csv, json, math, os, struct, sys, wave
G = os.path.join(os.path.expanduser(os.environ.get('GUITARSET', '~/data/rip/guitarset')), '')
O = os.path.abspath(sys.argv[1])
LOW, HIGH, VEL, TICKS = 21, 108, 80, 44100
SPLIT = {'00': 'train', '01': 'train', '02': 'train', '03': 'train', '04': 'val', '05': 'test'}
def vlq(n):
    b = [n & 127]
    while n > 127:
        n >>= 7
        b.insert(0, (n & 127) | 128)
    return bytes(b)
def observations(data):
    """an annotation's data as (time, duration, value), written as a list of observations or as a list a field"""
    if isinstance(data, list):
        return [(o['time'], o['duration'], o['value']) for o in data]
    return list(zip(data['time'], data['duration'], data['value']))
def write(path, notes):
    """notes as (start s, end s, key, channel); at one tick a note-off goes before a note-on"""
    ev = []
    for a, b, m, ch in notes:
        on = round(a * TICKS)
        ev += [(on, 1, 0x90 | ch, m, VEL), (max(round(b * TICKS), on + 1), 0, 0x80 | ch, m, 0)]
    tr = bytearray(b'\x00\xff\x51\x03' + (500000).to_bytes(3, 'big'))
    now = 0
    for tick, _, status, m, v in sorted(ev):
        tr += vlq(tick - now) + bytes([status, m, v])
        now = tick
    tr += b'\x00\xff\x2f\x00'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(b'MThd' + struct.pack('>IHHH', 6, 0, 1, TICKS // 2) + b'MTrk' + struct.pack('>I', len(tr)) + tr)
rel = lambda p: os.path.relpath(p, O)
rows, count = [], dict(notes=0, outside=0, cut=0, overlap=0, short=0, nopitch=0, nobeats=0, beats=0, lines=0, nowav=0)
keys, off = [], []
for name in sorted(os.listdir(G + 'annotation')):
    if not name.endswith('.jams'):
        continue
    clip, jams = name[:-5], G + 'annotation/' + name
    wav = f'{G}audio/{clip}_mic.wav'
    if not os.path.exists(wav):
        count['nowav'] += 1
        continue
    jam = json.load(open(jams))
    strings, contours, beats = {}, {}, []
    for a in jam['annotations']:
        src = a['annotation_metadata'].get('data_source') or ''
        if a['namespace'] == 'note_midi':
            strings.setdefault(src or str(len(strings)), []).extend(observations(a['data']))
        elif a['namespace'] == 'pitch_contour':
            contours.setdefault(src or str(len(contours)), []).extend(t for t, _, _ in observations(a['data']))
        elif a['namespace'] == 'beat_position' and not beats:
            beats = [(t, v['position']) for t, _, v in observations(a['data'])]
    assert len(strings) <= 9, f'{name}: {len(strings)} strings, and channel 10 is the drums'
    notes = []
    for ch, src in enumerate(sorted(strings)):
        line = sorted((t, t + d, math.floor(v + 0.5), v) for t, d, v in strings[src])
        for i, (a, b, m, v) in enumerate(line):
            later = [x[0] for x in line[i + 1:] if x[0] < b]
            count['overlap'] += bool(later)
            same = [x[0] for x in line[i + 1:] if x[0] < b and x[2] == m]
            if same:
                b, count['cut'] = min(same), count['cut'] + 1
            if not LOW <= m <= HIGH:
                count['outside'] += 1
                continue
            count['short'] += b - a < 1 / TICKS
            notes.append((a, b, m, ch))
            keys.append(m)
            off.append(abs(v - m))
    if not notes:
        continue
    mid = f'{O}/guitarset/midi/{clip}.mid'
    write(mid, notes)
    # stapes refuses a pitch track whose seconds do not ascend and a JAMS file with no beat: such a cell is left empty
    pitch = any(contours.values()) and all(c == sorted(c) for c in contours.values())
    count['nopitch'] += not pitch
    count['nobeats'] += not beats
    count['beats'] += len(beats)
    count['lines'] += sum(1 for _, p in beats if p == 1)
    count['notes'] += len(notes)
    with wave.open(wav) as w:
        secs = w.getnframes() / w.getframerate()
    rows.append((rel(wav), rel(mid), SPLIT[clip[:2]], secs, 'guitar', rel(jams) if pitch else '', rel(jams) if beats else '', 'measured'))
with open(f'{O}/guitar-list.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['audio_filename', 'midi_filename', 'split', 'duration', 'family', 'pitch', 'beats', 'velocity'])
    w.writerows(rows)
off.sort()
print('guitarset', len(rows), 'recordings', count['notes'], 'notes', round(sum(r[3] for r in rows) / 3600, 2), 'hours;',
      ', '.join(f'{s} {sum(1 for r in rows if r[2] == s)} ({round(sum(r[3] for r in rows if r[2] == s) / 3600, 2)} h)' for s in ('train', 'val', 'test')))
print('keys MIDI', min(keys), 'to', max(keys), ';', count['outside'], 'notes outside the 88 keys left out; a pitch sits a median',
      round(off[len(off) // 2], 3), 'of a semitone from its key, the furthest', round(off[-1], 3), ',', sum(1 for x in off if x > 0.4), 'past 0.4')
print(count['overlap'], 'notes run past the start of a later note on their string,', count['cut'], 'of them on its key and so ended there;',
      count['short'], 'notes shorter than a tick;', count['beats'], 'beats,', count['lines'], 'of them bar lines;',
      count['nopitch'], 'recordings with no pitch track named,', count['nobeats'], 'with no beats named,', count['nowav'], 'annotations with no recording')
