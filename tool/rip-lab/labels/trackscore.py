#!/usr/bin/env python3
"""trackscore.py <truth dir>...: the free tracker (vocal.py, under its bounds as they stand) scored on lines whose
notes are known (lines.py lays them): by time, by note, by onset, and two faults of its own, a line read a semitone
off and a slide written as a note. A kind of line scores the mean of the time said right and the onsets' F.
trackfit.py imports it; ../README.md has what each count leaves out."""
import collections, csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vocal

COLLAR, NEAR, FAR = 0.05, 0.05, 0.15
SUMS = ('sound', 'right', 'octave', 'elsewhere', 'none', 'quiet', 'false', 'noise', 'noisy', 'n_right', 'n_octave', 'n_elsewhere', 'n_none', 'truth', 'tracked', 'found',
        'up1', 'down1', 'glides', 'slid', 'passing')

def load(folder):
    """a folder's lines: name, kind, the tags to report by, the samples as the tracker reads them, the truth"""
    out = []
    for r in csv.DictReader(open(f'{folder}/lines.tsv'), delimiter='\t'):
        step = (44100 * 60 // (int(r['bpm']) * 4)) / 44100
        notes, noises = [], []
        for l in csv.DictReader(open(f"{folder}/{r['name']}.tsv"), delimiter='\t'):
            s, e = float(l['start']), float(l['end'])
            if l['kind'] == 'noise':
                noises.append((s, e))
                continue
            scoop = step if int(l['scoop']) else 0.0
            glide = (s + scoop) + float(l['at']) * (e - s - scoop) if float(l['bend']) else e
            notes.append([s, e, int(l['key']), scoop, glide])
        # a voice's release rings on after a note that a rest follows: neither a note nor silence is right there
        release = float(r.get('release') or 0)
        notes = [(*n, release if i + 1 == len(notes) or notes[i + 1][0] > n[1] + 1e-6 else 0.0) for i, n in enumerate(notes)]
        keys = sorted(n[2] for n in notes)
        mid, vib = keys[len(keys) // 2], float(r['vibrato_cents'])
        tags = [r['kind'], 'low, median key under 50' if mid < 50 else 'high, median key over 67' if mid > 67 else 'middle']
        if r['kind'] == 'engine':
            tags.append('no vibrato' if vib == 0 else 'vibrato to 50 cents' if vib <= 50 else 'vibrato over 50 cents')
            tags.append('vowel ' + (r['voice'].split()[1]))
            tuning = float(r.get('tuning') or 0)
            if tuning:
                tags.append('tuned within 25 cents' if abs(tuning) <= 25 else 'tuned 25 to 45 cents off' if abs(tuning) <= 45 else 'tuned 45 to 50 cents off')
            rung = float(r.get('ring_db') or 0)
            tags.append('no ring' if rung == 0 else 'ring to 22 dB' if rung <= 22 else 'ring over 22 dB')
            if rung > 22 and vib > 50 and mid < 56:
                tags.append('ring over 22, vibrato over 50, median key under 56')
        x = vocal.read(f"{folder}/{r['name']}.wav")
        out.append(dict(name=r['name'], kind=r['kind'], tags=tags, x=x, seconds=len(x) / vocal.SR, notes=notes, noises=noises,
                        peak=float(r['peak_db'])))
    return out

def components(line, tracked):
    """one line's counts under the notes tracked for it"""
    n = int(line['seconds'] * 100) + 2
    at = lambda t: min(max(int(round(t * 100)), 0), n)
    gt, gn, keep, din = np.zeros(n, int), np.zeros(n, int), np.ones(n, bool), np.zeros(n, bool)
    for s, e, k, scoop, glide, tail in line['notes']:
        gt[at(s):at(e)] = k
        for a, b in ((s - COLLAR, s + scoop + COLLAR), (glide - COLLAR if glide < e else e - COLLAR, e + tail + COLLAR)):
            keep[at(a):at(b)] = False
    for a, d, k in tracked:
        gn[at(a):at(a + d)] = k
    for s, e in line['noises']:
        din[at(s):at(e)] = True
    sound, quiet = keep & (gt > 0), keep & (gt == 0)
    wrong = sound & (gn > 0) & (gn != gt)
    c = dict(sound=sound.sum(), right=(sound & (gn == gt)).sum(), octave=(wrong & ((gn - gt) % 12 == 0)).sum(),
             elsewhere=(wrong & ((gn - gt) % 12 != 0)).sum(), none=(sound & (gn == 0)).sum(), quiet=quiet.sum(), false=(quiet & (gn > 0)).sum(),
             noise=(quiet & din).sum(), noisy=(quiet & din & (gn > 0)).sum(), n_right=0, n_octave=0, n_elsewhere=0, n_none=0,
             up1=(sound & (gn == gt + 1)).sum(), down1=(sound & (gn == gt - 1)).sum(), glides=0, slid=0, passing=0)
    # a slide between two held notes is no note: the notes tracked to start inside a glide on neither of its two keys
    for (s, e, k, scoop, glide, tail), nxt in zip(line['notes'], line['notes'][1:]):
        if glide < e:
            inside = [m for a, d, m in tracked if glide < a < e and m not in (k, nxt[2])]
            c['glides'], c['slid'], c['passing'] = c['glides'] + 1, c['slid'] + bool(inside), c['passing'] + len(inside)
    used, offs = set(), []
    for s, e, k, scoop, glide, tail in line['notes']:
        vals, counts = np.unique(gn[at(s):max(at(e), at(s) + 1)], return_counts=True)
        top = int(vals[counts.argmax()])
        c['n_none' if top == 0 else 'n_right' if top == k else 'n_octave' if (top - k) % 12 == 0 else 'n_elsewhere'] += 1
        # how far a start is from the true one: from the nearer end of a scoop, nothing inside it
        far = lambda a: max(s - a, a - s - scoop, 0.0) * (1 if a >= s else -1)
        near = [(abs(far(a)), i) for i, (a, d, m) in enumerate(tracked) if i not in used and m == k and abs(far(a)) <= FAR]
        if near:
            gap, i = min(near)
            used.add(i)
            offs.append(far(tracked[i][0]))
    c.update(truth=len(line['notes']), tracked=len(tracked), found=sum(1 for o in offs if abs(o) <= NEAR))
    return {k: int(v) for k, v in c.items()}, offs

def figures(rows):
    """a set of lines' counts (an array, a column a name of SUMS) as the shares reported"""
    s = dict(zip(SUMS, np.asarray(rows, float).sum(0)))
    p = s['found'] / s['tracked'] if s['tracked'] else 0.0
    r = s['found'] / s['truth'] if s['truth'] else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    said = (s['right'] + s['quiet'] - s['false']) / max(s['sound'] + s['quiet'], 1)
    return dict(score=(said + f) / 2, said=said, cover=1 - s['none'] / max(s['sound'], 1), right=s['right'] / max(s['sound'], 1),
                octave=s['octave'] / max(s['sound'], 1), elsewhere=s['elsewhere'] / max(s['sound'], 1), false=s['false'] / max(s['quiet'], 1),
                noisy=s['noisy'] / max(s['noise'], 1), n_right=s['n_right'] / max(s['truth'], 1), n_octave=s['n_octave'] / max(s['truth'], 1),
                n_elsewhere=s['n_elsewhere'] / max(s['truth'], 1), n_none=s['n_none'] / max(s['truth'], 1), precision=p, recall=r, f=f,
                notes=int(s['truth']), tracked=int(s['tracked']), up1=s['up1'] / max(s['sound'], 1), down1=s['down1'] / max(s['sound'], 1),
                glides=int(s['glides']), slid=s['slid'] / max(s['glides'], 1), passing=int(s['passing']))

def score(rows, kinds):
    """the score over kinds: the mean of each kind's own"""
    rows = np.asarray(rows, float)
    return float(np.mean([figures(rows[[i for i, k in enumerate(kinds) if k == kind]])['score'] for kind in sorted(set(kinds))]))

def run(lines, **bounds):
    """every line tracked and counted: (the counts as rows of SUMS, each line's onset offsets)"""
    rows, offs = [], []
    for l in lines:
        c, o = components(l, vocal.track(l['x'], **bounds)[0])
        rows.append([c[k] for k in SUMS])
        offs.append(o)
    return rows, offs

def report(title, lines, rows, offs):
    print(title)
    groups = collections.OrderedDict()
    for i, l in enumerate(lines):
        for t in l['tags']:
            groups.setdefault(t, []).append(i)
    print('  %-26s %5s %6s %6s %6s %6s %6s %6s %6s %6s | %6s %6s %6s %6s | %5s %5s %5s %7s %7s | %5s %5s | %6s %5s' % (
        '', 'lines', 'score', 'said', 'cover', 'right', 'octave', 'else', 'false', 'noisy', 'nright', 'noct', 'nelse', 'nnone', 'P', 'R', 'F', 'off ms', '|off|',
        'up1', 'down1', 'glides', 'slid'))
    for t, idx in groups.items():
        f = figures([rows[i] for i in idx])
        o = np.array([x for i in idx for x in offs[i]]) * 1000
        print('  %-26s %5d %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f | %6.3f %6.3f %6.3f %6.3f | %5.3f %5.3f %5.3f %+7.1f %7.1f | %5.3f %5.3f | %6d %5.3f' % (
            t, len(idx), f['score'], f['said'], f['cover'], f['right'], f['octave'], f['elsewhere'], f['false'], f['noisy'],
            f['n_right'], f['n_octave'], f['n_elsewhere'], f['n_none'], f['precision'], f['recall'], f['f'],
            np.median(o) if len(o) else 0, np.median(np.abs(o)) if len(o) else 0, f['up1'], f['down1'], f['glides'], f['slid']))
    print('  the score over kinds: %.4f' % score(rows, [l['kind'] for l in lines]))

if __name__ == '__main__':
    print('bounds:', vocal.BOUNDS, '| from', 'bounds.json' if os.path.exists(vocal.FITTED) else 'vocal.py, as set by hand')
    print('columns: said = time said right; cover, right, octave, else = shares of the time a note sounds; false = of the silent time, noisy = of',
          'the noises\' time, written into; nright .. nnone = true notes by the key holding most of them; P, R, F = onsets; off = tracked start less true, median and median size;',
          'up1, down1 = sounding time a semitone over, under; glides, slid = glides between held notes, the share with a passing note written in')
    for folder in sys.argv[1:]:
        lines = load(folder)
        report(f'{folder}: {len(lines)} lines, {sum(len(l["notes"]) for l in lines)} notes, {sum(l["seconds"] for l in lines) / 60:.1f} minutes', lines, *run(lines))
