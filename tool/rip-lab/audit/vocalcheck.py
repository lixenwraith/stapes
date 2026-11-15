#!/usr/bin/env python3
"""vocalcheck.py <vocal-list.csv> [sample]: VocalSet's written labels checked with no code of labels/label.py: every
row (paths, split, family, duration, copies), the sung time a note covers, a second pitch reading and, where label.py
wrote its table of takes beside the list, every recording accounted for, each listed take's keys against the set's
sheet, each note read from the sound and each held note's octave. Exits 1 if a row differs. The set's folder is
$VOCALSET (default ~/data/rip/vocalset). The README has each part; vocal_sheet.py imports its readers."""
import csv, hashlib, math, os, random, struct, sys, wave
import numpy as np
SET = os.path.realpath(os.path.expanduser(os.environ.get('VOCALSET', '~/data/rip/vocalset')))
CLEAR = 0.8  # the autocorrelation at its best lag from which a frame's pitch counts as clear

def midi_notes(path):
    """(key, start s, end s) of every note of a MIDI file, by its own tempo and division"""
    d = open(path, 'rb').read(); fmt, ntrk, div = struct.unpack('>HHH', d[8:14]); pos, ev = 14, []
    for _ in range(ntrk):
        end = pos + 8 + struct.unpack('>I', d[pos + 4:pos + 8])[0]; pos += 8; tick = 0; status = 0
        while pos < end:
            delta = 0
            while True:
                b = d[pos]; pos += 1; delta = delta << 7 | b & 127
                if b < 128: break
            tick += delta
            if d[pos] >= 128: status = d[pos]; pos += 1
            if status == 0xff:
                kind, n = d[pos], d[pos + 1]
                if kind == 0x51: ev.append((tick, 0, int.from_bytes(d[pos + 2:pos + 2 + n], 'big'), 0))
                pos += 2 + n
            elif status in (0xf0, 0xf7): pos += 1 + d[pos]
            else:
                hi = status & 0xf0; n = 1 if hi in (0xc0, 0xd0) else 2
                if hi == 0x90 and d[pos + 1] > 0: ev.append((tick, 2, d[pos], 1))
                elif hi in (0x80, 0x90): ev.append((tick, 1, d[pos], 0))
                pos += n
    ev.sort(); sec, last, tempo, on, out = 0.0, 0, 500000, {}, []
    for tick, kind, a, b in ev:
        sec += (tick - last) * tempo / 1e6 / div; last = tick
        if kind == 0: tempo = a
        elif kind == 2: on.setdefault(a, []).append(sec)
        elif on.get(a): out.append((a, on[a].pop(0), sec))
    return sorted(out, key=lambda n: n[1])

def read(p):
    with wave.open(p) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), '<i2').reshape(-1, w.getnchannels()).mean(1) / 32768
        return x, w.getframerate(), w.getnframes() / w.getframerate()

def sung(x, sr):
    """the centre time of each 46 ms frame and whether it lies within 20 dB of the recording's loud level"""
    n, h = int(0.0464 * sr), int(0.0232 * sr); e = np.concatenate([[0], np.cumsum(x * x)]); a = np.arange(0, len(x) - n, h)
    db = 10 * np.log10((e[a + n] - e[a]) / n + 1e-12)
    return (a + n / 2) / sr, db > np.percentile(db, 95) - 20

def pitch(x, sr, times):
    """a pitch in MIDI and how clear it is, at each time: the best lag of the autocorrelation of 93 ms under 1 kHz"""
    k = max(int(round(sr / 11025)), 1)
    f = np.fft.rfftfreq(8192, 1 / sr); taps = np.fft.irfft(np.where(f < 1000, 1.0, 0.0), 8192); taps = np.roll(taps, 256)[:512] * np.hanning(512)
    y = np.convolve(x, taps, 'same')[::k]; r = sr / k; n = int(0.093 * r); lo, hi = int(r / 1000), int(r / 65); out = []
    for t in times:
        a = int(t * r) - n // 2
        if a < 0 or a + n + hi > len(y): out.append((0.0, 0.0)); continue
        s = y[a:a + n + hi]; head = s[:n]; e0 = float(head @ head)
        if e0 <= 0: out.append((0.0, 0.0)); continue
        c = np.array([head @ s[l:l + n] for l in range(lo, hi)]); en = np.array([float(s[l:l + n] @ s[l:l + n]) for l in range(lo, hi)])
        q = c / np.sqrt(e0 * np.maximum(en, 1e-20)); i = int(np.argmax(q))
        first = np.flatnonzero(q > 0.9 * q[i]); i = int(first[0]) if len(first) else i   # the shortest lag near the best: no octave under
        while i + 1 < len(q) and q[i + 1] > q[i]: i += 1
        out.append((69 + 12 * math.log2(r / (lo + i) / 440), float(q[i])))
    return np.array(out).reshape(-1, 2)

def covered(times, ns):
    """whether a note sounds at each time"""
    inside = np.zeros(len(times), bool)
    for _, a, b in ns: inside |= (times >= a) & (times <= b)
    return inside

def against(p, tl, ns):
    """each note's read pitch less its key where three clear frames or more lie in it, and the recording's own median of them"""
    d = []
    for key, a, b in ns:
        m = (p[:, 1] >= CLEAR) & (tl >= a) & (tl <= b)
        d.append(float(np.median(p[m, 0])) - key if m.sum() >= 3 else None)
    got = [v for v in d if v is not None]
    return d, (float(np.median(got)) if len(got) >= 3 else None)

def blips(ns, short=0.25, held=0.4, gap=0.05):
    """the notes shorter than short that touch a note of held or more on each side: a slide written as a note"""
    return [(k, a, b) for i, (k, a, b) in enumerate(ns) if 0 < i < len(ns) - 1 and b - a < short and a - ns[i - 1][2] < gap and ns[i + 1][1] - b < gap
            and ns[i - 1][2] - ns[i - 1][1] >= held and ns[i + 1][2] - ns[i + 1][1] >= held]

def leaps(ns):
    """the notes nine semitones or more from the note before and the note after"""
    return [(k, a, b) for i, (k, a, b) in enumerate(ns) if 0 < i < len(ns) - 1 and abs(k - ns[i - 1][0]) >= 9 and abs(k - ns[i + 1][0]) >= 9]

def flips(ns, short=0.25, gap=0.05):
    """the short notes between two notes of one key a tone or less away, touching both: a vibrato written as notes"""
    return [(k, a, b) for i, (k, a, b) in enumerate(ns) if 0 < i < len(ns) - 1 and b - a < short and ns[i - 1][0] == ns[i + 1][0] and 0 < abs(k - ns[i - 1][0]) <= 2
            and a - ns[i - 1][2] < gap and ns[i + 1][1] - b < gap]

SHAPE = {'arpeggios': [0, 4, 7, 12, 7, 4, 0], 'scales': [0, 2, 4, 5, 7, 9, 11, 12, 14, 12, 11, 9, 7, 5, 4, 2, 0], 'long_tones': [0, 12, 17]}   # the set's sheet of its exercises, by the audit's reading
ORDER = ['on the key', 'an octave off, on the key', 'half a semitone to one and a half off', 'another pitch', 'no clear pitch']
SHORT = "a note under a third of the take's median"

def placed(HERE, rows, note, log):
    """label.py's table against the set, the list and the sheet, and every listed note against the sound"""
    table = {r['recording']: r for r in csv.DictReader(open(HERE + '/vocal-placed.tsv'), delimiter='\t')}
    root = SET + '/data_by_singer'; out = {}; total = 0
    for d, _, files in os.walk(root):
        for f in files:
            if f.endswith('.wav'):
                total += 1; rel = os.path.relpath(os.path.join(d, f), root)
                if rel not in table: k = '/'.join(rel.split(os.sep)[1:3]); out[k] = out.get(k, 0) + 1
    for rel in table:
        if not os.path.exists(os.path.join(root, rel)): note('a table row with no recording', rel)
    listed = {r['recording'] for r in table.values() if r['listed'] == 'listed'}
    inlist = {os.path.relpath(os.path.realpath(os.path.join(HERE, r['audio_filename'])), root) for r in rows}
    for rel in sorted(listed ^ inlist): note('the list and the table disagree', rel)
    why = {}
    for r in table.values(): why[r['listed']] = why.get(r['listed'], 0) + 1
    print(f"the table: {len(table)} of the set's {total} recordings; " + ', '.join(f'{k} {v}' for k, v in sorted(why.items(), key=lambda kv: -kv[1])) + '; not in it, by folder: ' + (', '.join(f'{k} {v}' for k, v in sorted(out.items(), key=lambda kv: -kv[1])) or 'none'))
    def share(key):
        g = {}
        for r in table.values():
            if r['exercise'] in SHAPE: a = g.setdefault(key(r), [0, 0]); a[0] += r['listed'] == 'listed'; a[1] += 1
        return ', '.join(f'{k} {a}/{b}' for k, (a, b) in sorted(g.items()))
    print('listed of those tried, by exercise: ' + share(lambda r: r['exercise'])); print('   by technique: ' + share(lambda r: r['technique'])); print('   by singer: ' + share(lambda r: r['recording'].split('/')[0]))
    cls = {}; takes = {'another pitch': set(), 'no clear pitch': set(), SHORT: set()}; flagged = []; octs = {}; least = 9.0
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # the audit's folder, which python -I leaves off the path
    from corpusheard import shares
    for r in rows:
        wav = os.path.realpath(os.path.join(HERE, r['audio_filename'])); rel = os.path.relpath(wav, root); t = table.get(rel)
        if t is None or t['exercise'] not in SHAPE: continue
        ns = midi_notes(os.path.join(HERE, r['midi_filename'])); keys = [k for k, _, _ in ns]; want = SHAPE[t['exercise']]
        if len(keys) != len(want) or any(k - keys[0] != w for k, w in zip(keys, want)): note("a listed take that is not its exercise's shape", f'{rel}: {keys}')
        elif keys[0] != int(t['tonic']): note('a tonic that is not the first note', rel)
        if not r.get('pitch') or not os.path.exists(os.path.join(HERE, r['pitch'])): note('no pitch track', rel)
        lens = [b - a for _, a, b in ns]; med = float(np.median(lens)) if lens else 0.0
        if any(l < med / 3 for l in lens): takes[SHORT].add(rel)
        if med > 0: least = min(least, min(lens) / med)
        x, sr, _ = read(wav); std = float(t['cents'] or 0) / 100; speed = 'fast' if t['technique'].startswith('fast') else 'slow'
        o = [0, 0, 0]   # held notes read, those an octave under their label, those an octave over: by where their power lies, which no period's measure decides
        for key, a, b in ns:
            if b - a < 0.25: continue
            f = 440 * 2 ** ((key + std - 69) / 12); h = shares(x, sr, a + 0.05, min(b - 0.02, a + 0.45), f)
            if h is None: continue
            o[0] += 1
            if h[1] > h[0]: o[1] += 1                        # the partials only the octave under has outweigh the label's
            else:
                h2 = shares(x, sr, a + 0.05, min(b - 0.02, a + 0.45), 2 * f)
                if h2 is not None and h2[1] < 0.03 * (h2[0] + h2[1]): o[2] += 1   # nothing on the label's odd partials
        g = octs.setdefault(rel.split(os.sep)[0], [0, 0, 0]); g[0] += o[0]; g[1] += o[1]; g[2] += o[2]
        if o[0] >= 3 and max(o[1], o[2]) >= 0.5 * o[0]: note('a take an octave from its sound', f'{rel}: {o[1]} under and {o[2]} over of {o[0]} held notes')
        spans = [np.arange(a + 0.03, b - 0.03, 0.0232) if b - a > 0.1 else np.array([(a + b) / 2]) for _, a, b in ns]
        q = pitch(x, sr, np.concatenate(spans)) if spans else np.zeros((0, 2)); at = 0
        for i, ((key, a, b), sp) in enumerate(zip(ns, spans)):
            v = q[at:at + len(sp)]; at += len(sp); ok = v[:, 1] >= CLEAR; d = None
            if ok.sum() < 2 or ok.mean() < 0.3: c = 'no clear pitch'
            else:
                d = float(np.median(v[ok, 0])) - key - std; f = (d + 6) % 12 - 6
                c = 'another pitch' if abs(f) >= 1.5 else 'half a semitone to one and a half off' if abs(f) >= 0.5 else 'an octave off, on the key' if abs(d - f) > 0.5 else 'on the key'
            g = cls.setdefault((speed, t['exercise']), {}); g[c] = g.get(c, 0) + 1
            if c in takes:
                takes[c].add(rel); flagged.append((rel, i, f'{a:.3f}', f'{b:.3f}', key, c, '' if d is None else f'{d:+.2f}', f'{ok.mean():.2f}'))
    print("every listed note against a second reading of its sound (the take's own standard taken off, an octave folded), by fast or slow takes and exercise:")
    for k in sorted(cls):
        n = sum(cls[k].values()); print(f'   {k[0]} {k[1]}: {n} notes; ' + ', '.join(f'{c} {cls[k].get(c, 0) / n:.1%}' for c in ORDER))
    n = sum(sum(g.values()) for g in cls.values()); tot = {c: sum(g.get(c, 0) for g in cls.values()) for c in ORDER}
    print(f'   all: {n} notes; ' + ', '.join(f'{c} {tot[c]} ({tot[c] / max(n, 1):.1%})' for c in ORDER))
    print(f'of {len(rows)} listed takes, those holding ' + '; '.join(f'{k}: {len(v)}' for k, v in takes.items()) + f"; the shortest note of any take is {least:.3f} of its take's median")
    n = max(sum(g[0] for g in octs.values()), 1); worst = max(octs.items(), key=lambda kv: (kv[1][1] + kv[1][2]) / max(kv[1][0], 1), default=('none', [1, 0, 0]))
    print(f'octaves, by where a held note\'s power lies: {n} notes of 0.25 s or longer; an octave under their label {sum(g[1] for g in octs.values()) / n:.2%}, an octave over it {sum(g[2] for g in octs.values()) / n:.2%}; the singer with most, {worst[0]}: {(worst[1][1] + worst[1][2]) / max(worst[1][0], 1):.1%}')
    with open(log, 'w') as f:
        f.write('recording\tnote\tstart\tend\tkey\twhat\tread less key\tclear share\n'); f.writelines('\t'.join(map(str, row)) + '\n' for row in flagged)
    print(f'the notes on another pitch or with no clear pitch are listed in {log}')

def main():
    LIST =os.path.abspath(sys.argv[1]); HERE = os.path.dirname(LIST); N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    TEST = {l.strip() for l in open(SET + '/test_singers_technique.txt') if l.strip()}
    rows = list(csv.DictReader(open(LIST))); bad = {}; cover = []; hours = {}; clips = {}; held = {}; shape = dict(far=0, flip=0, notes=0); blip = []
    def note(kind, what): bad.setdefault(kind, []).append(what)
    for r in rows:
        wav, mid = (os.path.realpath(os.path.join(HERE, r[k])) for k in ('audio_filename', 'midi_filename'))
        if not wav.startswith(SET + '/data_by_singer/') or not mid.startswith(os.path.realpath(HERE) + os.sep) or not os.path.exists(wav) or not os.path.exists(mid) or os.path.isabs(r['audio_filename']):
            note('path', r['audio_filename']); continue
        singer = os.path.relpath(wav, SET + '/data_by_singer').split(os.sep)[0]
        if r['split'] != ('test' if singer in TEST else 'train'): note('split', f'{singer}: {r["split"]}')
        if r.get('family') != 'voice': note('family', r['audio_filename'])
        digest = hashlib.sha256(open(wav, 'rb').read()).hexdigest()
        if digest in held: note('held twice', f'{os.path.basename(wav)} is {os.path.basename(held[digest])}')
        held.setdefault(digest, wav)
        x, sr, secs = read(wav)
        if abs(secs - float(r['duration'])) > 1e-6: note('duration', f'{r["audio_filename"]}: {r["duration"]} listed, {secs} in the WAV')
        hours[r['split']] = hours.get(r['split'], 0) + secs / 3600; clips[r['split']] = clips.get(r['split'], 0) + 1
        ns = midi_notes(mid); t, loud = sung(x, sr); inside = covered(t, ns)
        blip += [f'{os.path.basename(wav)} at {a:.2f} s' for k, a, b in blips(ns)]
        shape['far'] += len(leaps(ns)); shape['flip'] += len(flips(ns)); shape['notes'] += len(ns)
        cover.append((float((inside & loud).sum() / max(loud.sum(), 1)), float(loud.sum() * 0.0232), singer, wav, mid, len(ns)))
    c = np.array([v[0] for v in cover]); secs = np.array([v[1] for v in cover])
    print(f'{len(rows)} rows; ' + '; '.join(f'{s} {clips.get(s, 0)} ({hours.get(s, 0):.2f} h)' for s in ('train', 'test')) + f'; {sum(v[5] for v in cover)} notes; singers held out: {sorted(TEST)}')
    print(f'sung time a note covers: median {np.median(c):.2f} a recording; under 0.3 in {int((c < 0.3).sum())} recordings ({secs[c < 0.3].sum() / 60:.0f} min of sung time), under 0.5 in {int((c < 0.5).sum())} ({secs[c < 0.5].sum() / 60:.0f} min); of {secs.sum() / 3600:.2f} h of sung time in all, {(secs * (1 - c)).sum() / 3600:.2f} h has no note')
    by = {}
    for v in cover:
        if v[0] < 0.3: by[v[2]] = by.get(v[2], 0) + 1
    print('recordings under 0.3 by singer: ' + ', '.join(f'{s} {n}' for s, n in sorted(by.items(), key=lambda kv: -kv[1])))
    table = os.path.exists(HERE + '/vocal-placed.tsv')   # with it a take's notes are its sheet's, so a short note is one squeezed, counted; without it, a slide written as a note
    if not table: bad.update({'a short note between two held ones': blip} if blip else {})
    print(f'of {shape["notes"]} notes: {len(blip)} are short notes between two held ones' + (f' ({"; ".join(blip[:3])})' if blip and table else '') + f', {shape["flip"]} short notes between two of one key a tone or less away, {shape["far"]} nine semitones or more from both neighbours')
    rng = random.Random(1115); low = [v for v in cover if v[0] < 0.5]; rest = [v for v in cover if v[0] >= 0.5]
    for name, pool in (('least covered', low), ('the rest', rest)):
        pick = rng.sample(pool, min(N, len(pool))); clear, agree, octave, other, total, uncovered_clear, uncovered = 0, 0, 0, 0, 0, 0, 0
        for cov, _, singer, wav, mid, _ in pick:
            x, sr, _ = read(wav); t, loud = sung(x, sr); tl = t[loud]; p = pitch(x, sr, tl); ns = midi_notes(mid)
            ok = p[:, 1] >= CLEAR; clear += ok.sum(); total += len(p); inside = covered(tl, ns)
            uncovered += (~inside).sum(); uncovered_clear += (ok & ~inside).sum()
            d, off = against(p, tl, ns)
            if off is None: continue
            for v in d:
                if v is None: continue
                e = abs(v - off)
                if e < 0.5: agree += 1
                elif abs(e - 12) < 0.5: octave += 1
                else: other += 1
        n = max(agree + octave + other, 1)
        print(f'{name}, {len(pick)} recordings: {clear / max(total, 1):.0%} of the sung time has a clear pitch by the second reading; of the sung time no note covers, {uncovered_clear / max(uncovered, 1):.0%} has one; of {agree + octave + other} notes read, the key agrees with it for {agree / n:.0%}, sits an octave off for {octave / n:.0%}, elsewhere for {other / n:.0%}')
    lab = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'corpora', 'labels'))   # the lab's labels: their doubtful notes go to the audit's logs, any other list's beside it
    if table: placed(HERE, rows, note, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'vocalcheck-notes.tsv') if os.path.realpath(HERE) == lab else HERE + '/vocalcheck-notes.tsv')
    for kind, what in bad.items(): print(f'DIFFERS, {kind}: {len(what)}; first: ' + ' | '.join(what[:3]))
    print('nothing differs' if not bad else f'{sum(map(len, bad.values()))} things differ')
    sys.exit(1 if bad else 0)

if __name__ == '__main__':
    main()
