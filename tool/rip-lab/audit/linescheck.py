#!/usr/bin/env python3
"""linescheck.py <draw dir>...: the known lines that sing VocalSet's exercises (labels/lines.py) checked
against themselves with none of that tool's code: each truth's keys against the shape the set's sheet writes and
against what its list says is wrong with the take, and each note's pitch read from the WAV twice, by a comb about the
pitch the truth gives and by a difference function. Exits 1 if a truth lacks its shape or a draw's files were written
while it was read. The README says what each reading can and cannot tell."""
import collections, csv, os, sys, time, wave
import numpy as np
SHAPE = {'arpeggios': [0, 4, 7, 12, 7, 4, 0], 'scales': [0, 2, 4, 5, 7, 9, 11, 12, 14, 12, 11, 9, 7, 5, 4, 2, 0], 'long_tones': [0, 12, 17]}

def same(keys, shape): return len(keys) == len(shape) and all(k - keys[0] == s - shape[0] for k, s in zip(keys, shape))

def shaped(keys, shape, wrong):
    """whether a truth's keys are the exercise's shape with what its list says is wrong"""
    if not wrong: return same(keys, shape)
    if wrong == 'a note missing': return any(same(keys, shape[:i] + shape[i + 1:]) for i in range(len(shape)))
    if wrong == 'a bar twice': return same(keys, shape + shape)
    if wrong == 'a restart': return any(same(keys, shape[:k] + shape) for k in range(1, len(shape)))
    if wrong == 'a wrong note':
        if len(keys) != len(shape): return False
        for j in (0, 1):   # the key the take is in, by a note that is not the wrong one
            d = [k - keys[j] - (s - shape[j]) for k, s in zip(keys, shape)]
            if sum(v != 0 for v in d) == 1 and 2 <= abs(next(v for v in d if v)) <= 4: return True
    return False

def read(path):
    with wave.open(path) as w: return np.frombuffer(w.readframes(w.getnframes()), '<i2').reshape(-1, w.getnchannels()).mean(1) / 32768, w.getframerate()

def period(seg, sr, lo=60.0, hi=1100.0):
    """a span's pitch in MIDI and how periodic it is: the first deep dip of its difference function, each lag's mean over the lags before it taken out"""
    tmax = int(sr / lo); tmin = int(sr / hi); w = len(seg) - tmax
    if w < tmax // 2: return None
    d = np.array([np.sum((seg[:w] - seg[t:t + w]) ** 2) for t in range(1, tmax + 1)]); c = d * np.arange(1, tmax + 1) / np.maximum(np.cumsum(d), 1e-20)
    c[:tmin] = 2; under = np.flatnonzero(c < 0.15); t = int(under[0]) if len(under) else int(np.argmin(c))
    while t + 1 < tmax and c[t + 1] < c[t]: t += 1
    if 0 < t < tmax - 1:
        a, b, e = c[t - 1], c[t], c[t + 1]; x = 0.5 * (a - e) / (a - 2 * b + e) if a - 2 * b + e else 0.0
    else: x = 0.0
    return 69 + 12 * np.log2(sr / (t + 1 + x) / 440), float(c[t])

def comb(seg, sr, f, span=150, step=10):
    """the semitones from f at which a comb of ten partials gathers most of a span's spectrum, within span cents either way: blind to the octave, sure of the pitch within it"""
    n = 1 << int(np.ceil(np.log2(len(seg))) + 2); m = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n)); best, at = -1.0, 0
    for c in range(-span, span + 1, step):
        k = np.arange(1, 11) * f * 2 ** (c / 1200) * n / sr; k = k[k < len(m) - 1]; i = k.astype(int); v = float((m[i] * (1 - (k - i)) + m[i + 1] * (k - i)).sum())
        if v > best: best, at = v, c
    return at / 100

def main():
    bad = 0
    for D in sys.argv[1:]:
        D = D.rstrip('/'); began = time.time(); rows = list(csv.DictReader(open(D + '/lines.tsv'), delimiter='\t'))
        kinds = collections.Counter(); notes = secs = 0; unshaped = []; R = []
        for r in rows:
            ex, wrong = r['exercise'], r['wrong']; kinds[(ex, wrong or 'right')] += 1
            ns = [l for l in csv.DictReader(open(f'{D}/{r["name"]}.tsv'), delimiter='\t') if l['kind'] == 'note']; keys = [int(l['key']) for l in ns]; notes += len(ns)
            if not shaped(keys, SHAPE[ex], wrong): unshaped.append(f'{r["name"]} ({ex}, {wrong or "right"}): {keys}')
            x, sr = read(f'{D}/{r["name"]}.wav'); secs += len(x) / sr; off, by, tun = int(r['off_note']), float(r['off_by']), float(r['tuning']) / 100
            vib = max(float(r['vibrato_cents']), float(r.get('wide_cents') or 0)); chest = float(r.get('chest_hz') or 0) > 0
            for i, l in enumerate(ns):
                a, b = float(l['start']), float(l['end']); held = (b - a) * (float(l['at']) if float(l['bend']) else 1.0); t0, t1 = a + 0.4 * held, a + held - 0.005
                sung = float(l['off']) if l.get('off') not in (None, '') else (by if i == off else 0.0)   # the semitones the truth says the note is sung off its key
                # the comb's reading less the pitch the truth gives: under a vibrato the middle of its swing, where the held part holds a whole cycle
                hz = float(r.get('wide_hz') or 0) or float(r['vibrato_hz'] or 0); ws = np.arange(t0, t1 - 0.06 + 1e-9, 0.03); cd = None
                if len(ws) and not (vib and (hz <= 0 or t1 - t0 < 1 / hz)):
                    v = np.array([comb(x[int(t * sr):int((t + 0.06) * sr)], sr, 440 * 2 ** ((int(l['key']) + tun + sung - 69) / 12)) for t in ws])
                    cd = float((np.percentile(v, 10) + np.percentile(v, 90)) / 2) if vib else float(np.median(v))
                if t1 - t0 < 0.045: R.append((r['name'], i, vib, i == off, None, None, chest, cd)); continue
                ts = np.arange(t0, max(t1 - 0.045, t0) + 1e-9, 0.02); ts = ts[np.linspace(0, len(ts) - 1, min(len(ts), 12)).astype(int)]   # twelve spans at most, spread over the held part
                got = [period(x[int(t * sr):int((t + 0.045) * sr) + int(sr / 60)], sr) for t in ts]
                got = [g for g in got if g and g[1] < 0.3]
                if not got: R.append((r['name'], i, vib, i == off, None, None, chest, cd)); continue
                p = float(np.median([g[0] for g in got])); want = int(l['key']) + tun
                R.append((r['name'], i, vib, i == off, p - want - sung, p - want, chest, cd))
        newest = max(os.path.getmtime(os.path.join(D, f)) for f in os.listdir(D))
        print(f'== {D}: {len(rows)} lines, {notes} notes, {secs / 60:.1f} min; its newest file written {time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(newest))}' + ('; CHANGED WHILE READ, so nothing below holds' if newest > began else ''))
        bad += newest > began
        for ex in SHAPE: print(f'   {ex}: {sum(v for (e, _), v in kinds.items() if e == ex)} takes; ' + ', '.join(f'{w} {kinds[(ex, w)]}' for w in ('right', 'a note missing', 'a bar twice', 'a restart', 'a wrong note')))
        print(f'   by what is wrong: ' + ', '.join(f'{w} {sum(v for (_, k), v in kinds.items() if k == w)}' for w in ('right', 'a note missing', 'a bar twice', 'a restart', 'a wrong note')) + f'; right takes with a note sung off: {sum(1 for r in rows if not r["wrong"] and int(r["off_note"]) >= 0)}')
        print(f'   truths that do not have the shape their list says: {len(unshaped)}', *unshaped[:12], sep='\n      ')
        bad += len(unshaped)
        for name, sel in (('notes of lines with no vibrato and no chest resonance, the one sung far off aside', lambda q: q[2] == 0 and not q[3] and not q[6]), ('notes of lines with a vibrato and no chest resonance, the one sung far off aside', lambda q: q[2] > 0 and not q[3] and not q[6]),
                          ('notes of lines with a chest resonance, the one sung far off aside', lambda q: q[6] and not q[3]), ('the notes sung far off, against key, tuning and the distance the truth gives', lambda q: q[3])):
            s = [q for q in R if sel(q)]; d = np.array([q[4] for q in s if q[4] is not None])
            if not len(s): continue
            line = f'   {name}: {len(s)}; no held part long or periodic enough to read {len(s) - len(d)}'
            if len(d): line += f'; read {len(d)}: within 0.3 of a semitone {np.mean(np.abs(d) <= 0.3):.1%}, within 0.5 {np.mean(np.abs(d) <= 0.5):.1%}, an octave off {np.mean(np.abs(np.abs(d) - 12) <= 0.5):.1%}, a median {np.median(np.abs(d)) * 100:.0f} cents'
            print(line)
            k = np.array([q[7] for q in s if q[7] is not None])
            if len(k): print(f'      by a comb at the pitch the truth gives (blind to the octave; a vibrato read at the middle of its swing over a whole cycle or more): read {len(k)}: within 0.3 of a semitone {np.mean(np.abs(k) <= 0.3):.1%}, within 0.5 {np.mean(np.abs(k) <= 0.5):.1%}, a median {np.median(np.abs(k)) * 100:.0f} cents')
            far = [f'{q[0]} note {q[1]} {q[4]:+.2f}' for q in s if q[4] is not None and abs(q[4]) > 0.5]
            if far: print(f'      over half a semitone off: {len(far)}:', '; '.join(far[:10]))
        o = np.array([q[5] for q in R if q[3] and q[5] is not None])
        if len(o): print(f'   the notes sung off sit a median {np.median(np.abs(o)):.2f} of a semitone from their key at the line\'s tuning ({np.abs(o).min():.2f} to {np.abs(o).max():.2f})')
    sys.exit(1 if bad else 0)

if __name__ == '__main__': main()
