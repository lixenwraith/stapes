#!/usr/bin/env python3
"""guitarcount.py <guitar-list.csv>: GuitarSet's written labels recounted from the set's own JAMS files, with no code
of labels/guitar.py or of audio: every row's paths, split by player and duration, and the MIDI file's notes against
the JAMS file's note_midi observations, string by string. Exits 1 if a note, a path, a split or a duration differs.
The set's folder is $GUITARSET (default ~/data/rip/guitarset). The README says what counts as a difference."""
import csv, json, math, os, struct, sys, wave
LIST = os.path.abspath(sys.argv[1]); HERE = os.path.dirname(LIST)
SET = os.path.realpath(os.path.expanduser(os.environ.get('GUITARSET', '~/data/rip/guitarset')))
SPLIT = dict(zip('00 01 02 03 04 05'.split(), 'train train train train val test'.split()))
TICK = 1 / 44100

def midi_notes(path):
    """(channel, key, start s, end s, velocity) of every note of a MIDI file, by its own tempo and division"""
    d = open(path, 'rb').read()
    assert d[:4] == b'MThd', path
    fmt, ntrk, div = struct.unpack('>HHH', d[8:14])
    assert div < 0x8000, 'SMPTE time'
    pos, events = 14, []
    for _ in range(ntrk):
        assert d[pos:pos + 4] == b'MTrk', path
        end = pos + 8 + struct.unpack('>I', d[pos + 4:pos + 8])[0]; pos += 8; tick = 0; status = 0
        while pos < end:
            delta = 0
            while True:
                b = d[pos]; pos += 1; delta = delta << 7 | b & 127
                if b < 128: break
            tick += delta
            if d[pos] >= 128: status = d[pos]; pos += 1
            if status == 0xff:
                kind = d[pos]; n = d[pos + 1]; body = d[pos + 2:pos + 2 + n]; pos += 2 + n
                assert n < 128, 'a long meta event'
                if kind == 0x51: events.append((tick, 'tempo', int.from_bytes(body, 'big'), 0, 0))
            elif status in (0xf0, 0xf7):
                n = d[pos]; pos += 1 + n
            else:
                hi = status & 0xf0; n = 1 if hi in (0xc0, 0xd0) else 2; a = d[pos]; b = d[pos + 1] if n == 2 else 0; pos += n
                if hi == 0x90 and b > 0: events.append((tick, 'on', status & 15, a, b))
                elif hi == 0x80 or hi == 0x90: events.append((tick, 'off', status & 15, a, 0))
    events.sort(key=lambda e: (e[0], {'tempo': 0, 'off': 1, 'on': 2}[e[1]]))
    sec, last, tempo, open_, notes = 0.0, 0, 500000, {}, []
    for tick, kind, a, b, c in events:
        sec += (tick - last) * tempo / 1e6 / div; last = tick
        if kind == 'tempo': tempo = a
        elif kind == 'on': open_.setdefault((a, b), []).append((sec, c))
        elif open_.get((a, b)):
            s, v = open_[(a, b)].pop(0); notes.append((a, b, s, sec, v))
    assert not any(open_.values()), f'{path}: a note never ends'
    return notes, fmt, ntrk, div

def rows_of(data):
    return [(o['time'], o['duration'], o['value']) for o in data] if isinstance(data, list) else list(zip(data['time'], data['duration'], data['value']))

tot = dict(rows=0, notes=0, midi=0, beats=0, lines=0, runs=0, cut=0, exact_half=0)
bad = {}; keys = set(); vels = set(); hours = {}; clips = {}; gap = 0.0; seen = set()
def note(kind, what):
    bad.setdefault(kind, []).append(what)
for r in csv.DictReader(open(LIST)):
    tot['rows'] += 1
    p = {k: os.path.realpath(os.path.join(HERE, r[k])) for k in ('audio_filename', 'midi_filename', 'pitch', 'beats') if r[k]}
    for k, v in p.items():
        inside = v.startswith(os.path.realpath(HERE) + os.sep) if k == 'midi_filename' else v.startswith((SET + '/audio/', SET + '/annotation/'))
        if not inside or not os.path.exists(v) or os.path.isabs(r[k]): note('path', f'{r[k]} ({k})')
    clip = os.path.basename(p['midi_filename'])[:-4]
    if clip in seen: note('listed twice', clip)
    seen.add(clip)
    if r['split'] != SPLIT.get(clip[:2]): note('split', f'{clip}: {r["split"]}')
    if r['family'] != 'guitar': note('family', f'{clip}: {r["family"]}')
    with wave.open(p['audio_filename']) as w: secs = w.getnframes() / w.getframerate()
    if abs(secs - float(r['duration'])) > 1e-6: note('duration', f'{clip}: {r["duration"]} listed, {secs} in the WAV')
    hours[r['split']] = hours.get(r['split'], 0) + secs / 3600; clips[r['split']] = clips.get(r['split'], 0) + 1
    jam = json.load(open(os.path.join(SET, 'annotation', clip + '.jams')))
    if p.get('pitch') != os.path.join(SET, 'annotation', clip + '.jams') or p.get('beats') != p.get('pitch'): note('pitch or beats cell', clip)
    strings = {}
    for a in jam['annotations']:
        if a['namespace'] == 'note_midi':
            strings.setdefault(str(a['annotation_metadata'].get('data_source')), []).extend(rows_of(a['data']))
        elif a['namespace'] == 'beat_position':
            b = rows_of(a['data']); tot['beats'] += len(b); tot['lines'] += sum(1 for _, _, v in b if v['position'] == 1)
    if len(strings) != 6: note('strings', f'{clip}: {len(strings)}')
    want = []
    for s in sorted(strings):
        line = sorted(strings[s])
        for i, (t, d, v) in enumerate(line):
            k = int(math.floor(v + 0.5)); tot['exact_half'] += v + 0.5 == math.floor(v + 0.5)
            nxt = [u for u, _, x in line[i + 1:] if u < t + d and int(math.floor(x + 0.5)) == k]
            want.append((sorted(strings).index(s), k, t, t + d, min(nxt) if nxt else None)); tot['runs'] += bool(nxt)
    got, fmt, ntrk, div = midi_notes(p['midi_filename'])
    if (fmt, ntrk, div) != (0, 1, 22050): note('midi header', f'{clip}: format {fmt}, {ntrk} tracks, {div} ticks a quarter')
    tot['notes'] += len(want); tot['midi'] += len(got)
    keys.update(k for _, k, _, _, _ in want); vels.update(v for *_, v in got)
    if len(want) != len(got): note('count', f'{clip}: {len(want)} in the JAMS, {len(got)} in the MIDI'); continue
    for (c, k, t, e, cut), (gc, gk, gs, ge, _) in zip(sorted(want), sorted(got)):
        if (c, k) != (gc, gk) or abs(gs - t) > TICK: note('start or key', f'{clip}: string {c} key {k} at {t:.4f}, MIDI string {gc} key {gk} at {gs:.4f}'); continue
        gap = max(gap, abs(gs - t))
        if abs(ge - e) <= TICK or (e - t < TICK and ge - gs <= 2 * TICK): gap = max(gap, min(abs(ge - e), TICK)); continue
        if cut is not None and abs(ge - cut) <= TICK: tot['cut'] += 1; continue
        note('end', f'{clip}: string {c} key {k} at {t:.4f} ends {e:.4f}, MIDI {ge:.4f}')
print(f'{tot["rows"]} rows; {tot["notes"]} notes in the JAMS files, {tot["midi"]} in the MIDI files; keys {min(keys)} to {max(keys)}; velocities written: {sorted(vels)}')
print('; '.join(f'{s} {clips.get(s, 0)} ({hours.get(s, 0):.2f} h)' for s in ('train', 'val', 'test')), f'; {sum(hours.values()):.2f} hours in all')
print(f'{tot["beats"]} beats, {tot["lines"]} of them bar lines; {tot["runs"]} notes run into the same string\'s next note of their key, {tot["cut"]} of them ended a whole tick early for it; {tot["exact_half"]} pitches exactly a half from two keys')
print(f'largest gap of a matched start or end: {gap * 1e6:.1f} microseconds (a tick is {TICK * 1e6:.1f})')
for kind, what in bad.items(): print(f'DIFFERS, {kind}: {len(what)}; first: ' + ' | '.join(what[:3]))
print('no difference' if not bad else f'{sum(map(len, bad.values()))} differences')
sys.exit(1 if bad else 0)
