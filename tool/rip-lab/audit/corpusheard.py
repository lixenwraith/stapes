#!/usr/bin/env python3
"""corpusheard.py <corpus dir> [NSynth's audio dir]: whether a rendered corpus's notes sound as its labels say, from
the labels and the WAV files alone: what the genes of the engine's voices switch on, each note's rise at its own
partials, where a note that sounds alone has its power, what rings on after a label ends a note, the recorded notes
let go before their recording speaks, and how empty the overlaps are. It prints figures and judges nothing: run it
before a sample of a corpus is cut, so the ear is asked only what this cannot tell. The README has each measure."""
import collections, json, os, sys, wave
import numpy as np
HOP = 256 / 22050
GENES = [('unison', 1, 3, 0), ('detune', 1, 1.02, 0), ('sub', 0, 1, 0), ('fm_ratio', 0, 8, 0), ('fm_index', 0, 8, 0), ('fm_decay', 0, 2, 0), ('bright', 0, 1, 0), ('ring', 0.05, 4, 1),
         ('partials', 4, 48, 0), ('strike_pos', 0.05, 0.5, 0), ('tilt', 0.5, 3, 0), ('inharm', 1e-5, 1e-2, 1), ('inharm_track', 0, 2, 0), ('decay_time', 0.05, 4, 1), ('decay_track', 0, 1.5, 0),
         ('damp_hz', 200, 8000, 1), ('pair', 0, 10, 0), ('pair_level', 0, 0.6, 0), ('pair_slow', 0.1, 1, 0), ('hammer_hz', 500, 8000, 1), ('knock', 0, 0.5, 0), ('cutoff', 50, 10000, 1),
         ('key_track', 0, 8, 0), ('env_amount', 0, 5000, 0), ('env_decay', 0, 1, 0), ('q', 0, 4, 0), ('attack', 0.001, 1, 1), ('decay', 0.005, 2, 1), ('sustain', 0, 1, 0), ('release', 0.005, 3, 1),
         ('drive', 0, 4, 0), ('level', 0.1, 2, 0)]   # audio's voiceGenes: name, floor, top, read on a log scale
OFF = ('sub', 'shut', 'fm', 'slow')   # what a voice does that takes a note's sound off its label
WHAT = (('sub', 'a sine an octave under the note at 0.05 or more'), ('sub15', 'at 0.15 or more'), ('filter', 'a filter'), ('shut', 'its cutoff at rest under the note\'s own fundamental'),
        ('shut-open', 'and still under it with the filter\'s envelope open'), ('fm', 'FM at an index of 0.3 or more on a ratio that is no whole number'), ('slow', 'an attack longer than the note'),
        ('any', 'any of: the sine under, the cutoff under the note, that FM, that attack'), ('r05', 'a release of 0.5 s or more'), ('r1', 'of 1 s or more'), ('r2', 'of 2 s or more'),
        ('r5x', 'a release five times the note\'s labelled length or more'))

def voice(g):
    """a label's genes as the engine reads them back"""
    return {n: (lo * (hi / lo) ** x if log else lo + x * (hi - lo)) for (n, lo, hi, log), x in zip(GENES, g)}

def hz(e, tun): return 440 * 2 ** ((e['midi'] + tun - 69) / 12)

def length(e): return (max(e['end'], e['frame'] + 1) - e['frame']) * HOP

def state(e, tun, waves):
    """what a note's voice does to it, by its genes alone"""
    v = voice(e['voice']); f = hz(e, tun); w = waves[e['wave']]; osc = w in ('sine', 'saw', 'square'); s = set()
    if v['sub'] >= 0.05: s.add('sub')
    if v['sub'] >= 0.15: s.add('sub15')
    if v['q'] > 0:
        s.add('filter')
        if v['cutoff'] + v['key_track'] * f < f: s.add('shut')
        if v['cutoff'] + v['key_track'] * f + v['env_amount'] * e['vel'] < f: s.add('shut-open')
    if osc and v['fm_index'] >= 0.3 and v['fm_ratio'] > 0 and abs(v['fm_ratio'] - round(v['fm_ratio'])) > 0.03: s.add('fm')
    if osc and v['attack'] > length(e): s.add('slow')
    if s & set(OFF): s.add('any')
    for k, r in (('r05', 0.5), ('r1', 1), ('r2', 2)):
        if v['release'] >= r: s.add(k)
    if v['release'] >= 5 * length(e): s.add('r5x')
    return w, s

def read(path):
    with wave.open(path) as w: return np.frombuffer(w.readframes(w.getnframes()), '<i2').reshape(-1, w.getnchannels()).mean(1) / 32768, w.getframerate()

HANN = {}
def own(x, sr, t, f, n):
    """the power at a note's first four partials in n samples from t"""
    a = int(t * sr)
    if a < 0 or a + n > len(x): return None
    if n not in HANN: HANN[n] = np.hanning(n)
    h = np.arange(1, 5); h = h[f * h < 0.45 * sr]
    return float((np.abs(np.exp(-2j * np.pi * f / sr * np.outer(h, np.arange(n))) @ (x[a:a + n] * HANN[n])) ** 2).sum() / n ** 2)

def shares(x, sr, a, b, f):
    """of a span's power under 6.6 times f: the share on the label's partials 1 to 6, on the octave under's own (a half, one and a half, ...) and elsewhere"""
    seg = x[int(a * sr):int(b * sr)]; T = len(seg) / sr
    if T < max(0.12, 9.1 / f): return None
    n = 1 << int(np.ceil(np.log2(len(seg))) + 1); p = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n)) ** 2; fr = np.fft.rfftfreq(n, 1 / sr)
    tot = p[(fr >= 30) & (fr < 6.6 * f)].sum()
    if tot <= 0: return None
    band = lambda c, k: p[np.abs(fr - c) <= min(0.22 * f, max(0.03 * k * f, 2.0 / T))].sum()
    h = sum(band(k * f, k) for k in range(1, 7)); u = sum(band((k - 0.5) * f, k) for k in range(1, 7) if (k - 0.5) * f >= 30)
    return h / tot, u / tot, max(tot - h - u, 0) / tot

def db(a, b): return 10 * np.log10((a + 1e-16) / (b + 1e-16))

def main():
    D = sys.argv[1].rstrip('/') + '/'; NS = sys.argv[2].rstrip('/') + '/' if len(sys.argv) > 2 else None
    C = json.load(open(D + 'corpus.json')); waves = C['waves']; N = C['loops']
    files = collections.defaultdict(dict); speaks = {}
    if NS:
        for f in os.listdir(NS):
            if f.endswith('.wav'): ins, p, _ = f[:-4].rsplit('-', 2); files[ins].setdefault(int(p), f)
    def speak(ins, midi):
        """seconds the recording nearest a key takes to come within 6 dB of its loudest 50 ms"""
        ps = files.get(ins)
        if not ps: return None
        f = ps[min(ps, key=lambda q: abs(q - midi))]
        if f not in speaks:
            x, sr = read(NS + f); n = int(0.05 * sr); e = np.concatenate([[0], np.cumsum(x * x)]); a = np.arange(0, min(len(x), 3 * sr) - n, n // 5); r = np.sqrt((e[a + n] - e[a]) / n + 1e-12)
            speaks[f] = float(a[np.argmax(r >= r.max() * 0.5)] / sr)
        return speaks[f]
    census = collections.Counter(); bywave = collections.defaultdict(collections.Counter); inloops = collections.Counter(); engloops = 0
    R = []; A = []; ring = []; early = []; gaps = collections.defaultdict(list)
    for i in range(N):
        lab = json.load(open(f'{D}loop_{i:03d}.json')); tun = lab.get('tuning', 0) / 100; kind = lab.get('kind', '?'); x, sr = read(f'{D}loop_{i:03d}.wav')
        notes = sorted((e for e in lab['events'] if e['drum'] < 0 and e.get('midi', 0) > 0), key=lambda e: e['frame']); hits = [e['frame'] for e in lab['events'] if e['drum'] >= 0]
        st = {id(e): state(e, tun, waves) for e in notes if e.get('voice')}; engloops += bool(st); seen = set()
        for w, s in st.values():
            census['notes'] += 1; bywave[w]['notes'] += 1; seen |= s
            for k in s: census[k] += 1; bywave[w][k] += 1
        for k in seen: inloops[k] += 1
        got = []; top = 0.0
        for e in notes:
            f = hz(e, tun); n = int(max(0.06, 4 / f) * sr); t0 = e['frame'] * HOP; t1 = t0 + length(e); rec = not e.get('voice'); fam = e['instr'].rsplit('_', 2)[0] if rec else 'engine'
            after = [v for v in (own(x, sr, t0 + d, f, n) for d in (0.01, 0.04, 0.08, 0.14) if t0 + d < max(t1, t0 + 0.05)) if v is not None]
            if not after: continue
            a = max(after); top = max(top, a); before = own(x, sr, t0 - 0.005 - n / sr, f, n) or 0.0
            same = any(o is not e and o['midi'] == e['midi'] and o['frame'] <= e['frame'] < o['end'] for o in notes)   # its own key already sounds
            got.append((e, a, before, same, rec, fam))
            if not any(o is not e and o['midi'] == e['midi'] and e['frame'] - 26 < o['frame'] < e['end'] + 130 for o in notes):   # its key is not struck again within 1.5 s
                later = own(x, sr, t1 + 0.5, f, n)
                if later is not None: ring.append((rec, kind, db(later, a)))
            if rec and NS:
                s = speak(e['instr'], e['midi'])
                if s is not None: early.append((kind, fam, length(e), s))
            if t1 - t0 < 0.2 or any(o is not e and o['frame'] < e['end'] and o['end'] > e['frame'] for o in notes) or any(e['frame'] - 3 <= h < e['end'] for h in hits): continue
            r = shares(x, sr, t0 + 0.05, min(t1, t0 + 0.45), f)
            if r is None: continue
            a0, b0 = int(max(t0 - 0.12, 0) * sr), int(max(t0 - 0.02, 0) * sr); a1, b1 = int((t0 + 0.05) * sr), int(min(t1, t0 + 0.45) * sr)
            if b0 - a0 > 0.05 * sr and np.mean(x[a0:b0] ** 2) > np.mean(x[a1:b1] ** 2) * 10 ** -1.5: continue   # what sounded before it still rings within 15 dB of it
            A.append((rec, fam if rec else st[id(e)][0], st[id(e)][1] if not rec else set(), r))
        for e, a, before, same, rec, fam in got: R.append((kind, rec, fam, length(e), e['vel'], db(a, before), db(a, top), same))
        if kind == 'overlap' and notes:
            cov = np.zeros(lab['frames'], bool)
            for e in notes: cov[e['frame']:max(e['end'], e['frame'] + 1)] = True
            gaps[min(len({e['instr'] for e in notes}), 4)].append((1 - cov.mean(), lab['frames'] * HOP, len(notes)))
    n = max(census['notes'], 1); print(f'{census["notes"]} notes on the engine\'s voices, in {engloops} of {N} loops, by the genes their labels carry:')
    for k, what in WHAT: print(f'   {what}: {census[k]} ({census[k] / n:.0%}), in {inloops[k]} loops')
    print('   by wave (notes; the sine under; the cutoff under the note; that FM; that attack; any of them; a release of 0.5 s or more):')
    for w in waves:
        c = bywave[w]; m = max(c['notes'], 1); print(f'      {w:7s} {c["notes"]:6d}  {c["sub"] / m:4.0%} {c["shut"] / m:4.0%} {c["fm"] / m:4.0%} {c["slow"] / m:4.0%} {c["any"] / m:4.0%} {c["r05"] / m:4.0%}')
    def rise(name, sel):
        s = [r for r in R if sel(r)]; m = max(len(s), 1)
        print(f'   {name}: {len(s)} notes, {sum(r[5] < 6 and not r[7] for r in s) / m:.0%} with no rise of 6 dB at their own partials, {sum(r[6] < -40 for r in s) / m:.0%} more than 40 dB under their loop\'s strongest note')
    print(f'{len(R)} notes, each at its onset: the power at its own first four partials against just before it')
    rise('all', lambda r: True); rise('arranged', lambda r: r[0] == 'arranged'); rise('overlaps', lambda r: r[0] == 'overlap'); rise('on recorded instruments', lambda r: r[1]); rise('on the engine\'s voices', lambda r: not r[1])
    for lo, hi in ((0, 0.15), (0.15, 0.3), (0.3, 0.6), (0.6, 99)): rise(f'held {lo} to {hi} s' if hi < 99 else f'held {lo} s or more', lambda r, lo=lo, hi=hi: lo <= r[3] < hi)
    for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 1.01)): rise(f'velocity over {lo} to {min(hi, 1)}', lambda r, lo=lo, hi=hi: lo < r[4] <= hi)
    for f, _ in collections.Counter(r[2] for r in R if r[1]).most_common(): rise(f'recorded, {f}', lambda r, f=f: r[1] and r[2] == f)
    def where(name, sel):
        r = np.array([a[3] for a in A if sel(a)])
        if len(r) < 5: print(f'   {name}: {len(r)} notes'); return
        print(f'   {name}: {len(r)} notes; the median share on the label\'s partials {np.median(r[:, 0]):.2f}; under half on them {np.mean(r[:, 0] < 0.5):.0%}; the octave under\'s own outweigh them {np.mean(r[:, 1] > r[:, 0]):.0%}; elsewhere outweighs them {np.mean(r[:, 2] > r[:, 0]):.0%}')
    print(f'{len(A)} notes that sound alone and enter cleanly (0.2 s or longer, no other note or hit with them, nothing within 15 dB just before), by where their power is from 50 ms after the onset:')
    where('recorded', lambda a: a[0]); where('engine', lambda a: not a[0])
    for w in waves: where(f'engine, {w}', lambda a, w=w: not a[0] and a[1] == w)
    where('engine, none of the four by its genes', lambda a: not a[0] and 'any' not in a[2])
    for k, what in WHAT[:7]:
        if k != 'filter': where(f'engine, {what}', lambda a, k=k: not a[0] and k in a[2])
    for f, _ in collections.Counter(a[1] for a in A if a[0]).most_common(): where(f'recorded, {f}', lambda a, f=f: a[0] and a[1] == f)
    print(f'{len(ring)} notes whose key is not struck again within 1.5 s: their own partials half a second after the label ends them, against where they stood while it held them')
    for name, sel in (('engine', lambda r: not r[0]), ('recorded', lambda r: r[0]), ('engine, arranged', lambda r: not r[0] and r[1] == 'arranged'), ('recorded, arranged', lambda r: r[0] and r[1] == 'arranged')):
        d = np.array([r[2] for r in ring if sel(r)])
        if len(d): print(f'   {name}: {len(d)} notes; a median {np.median(d):+.0f} dB; within 6 dB {np.mean(d >= -6):.0%}, within 12 dB {np.mean(d >= -12):.0%}, within 20 dB {np.mean(d >= -20):.0%}')
    if early:
        e = np.array([(a, b) for _, _, a, b in early]); fams = np.array([f for _, f, _, _ in early]); kinds = np.array([k for k, _, _, _ in early])
        print(f'{len(e)} notes on recorded instruments, against the seconds their recording takes to come within 6 dB of its loudest 50 ms:')
        for name, m in (('all', np.ones(len(e), bool)), ('arranged', kinds == 'arranged'), ('overlaps', kinds == 'overlap')):
            if m.sum(): print(f'   {name}: {m.sum()} notes, held a median {np.median(e[m, 0]):.2f} s; let go before that {np.mean(e[m, 0] < e[m, 1]):.0%}')
        for f, c in collections.Counter(fams).most_common():
            m = fams == f; print(f'   {f}: {c} notes; the recording takes a median {np.median(e[m, 1]):.2f} s; let go before that {np.mean(e[m, 0] < e[m, 1]):.0%}')
    print('the overlaps, by instruments (loops; the median share of a loop with no note sounding by its label; its median seconds; its median notes):')
    for k in sorted(gaps):
        g = np.array(gaps[k]); print(f'   {k}{" or more" if k == 4 else ""} instrument{"" if k == 1 else "s"}: {len(g)} loops, {np.median(g[:, 0]):.0%} with no note, {np.median(g[:, 1]):.1f} s, {np.median(g[:, 2]):.0f} notes')

if __name__ == '__main__': main()
