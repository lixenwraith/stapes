"""stringrise.py <guitar list csv> <out tsv>: where each string's channel rises against the start GuitarSet's annotation
gives a note on it: the greatest rise (60 ms after over 60 ms before) within 0.3 s either side, between the note's
neighbours on the string. A row a recording: split, notes, the share with a clear rise (15 dB), their lag in ms at one,
five and nine in ten (the measure reads some 30 ms early), the shares over 50 and 100 ms late and over 100 ms early, and
silent100, the share over 100 ms late on a string 20 dB under its level after the rise where the note is said to start."""
import csv, json, os, sys, wave
import numpy as np
WIN, GRID, REACH, CLEAR, QUIET = 0.06, 0.005, 0.3, 15.0, 20.0

def lags(hexwav, jams):
    """each annotated note's lag in seconds, its rise in dB and how far under the level after that rise its string
    sits in the 60 ms after the annotated start, on its own string's channel"""
    w = wave.open(hexwav); sr, ch = w.getframerate(), w.getnchannels()
    x = np.frombuffer(w.readframes(w.getnframes()), '<i2').reshape(-1, ch).astype(np.float64) / 32768
    n, g, out = int(WIN * sr), int(GRID * sr), []
    for s, an in enumerate([a for a in json.load(open(jams))['annotations'] if a['namespace'] == 'note_midi']):
        d = an['data']; st = sorted(r['time'] for r in d) if isinstance(d, list) else sorted(d['time'])
        c = np.concatenate([[0], np.cumsum(x[:, s] ** 2)]); at = np.arange(n, len(x) - n, g); tt = at / sr
        rise = 10 * np.log10((c[at + n] - c[at]) / n + 1e-12) - 10 * np.log10((c[at] - c[at - n]) / n + 1e-12)
        for i, t in enumerate(st):
            lo = max(t - REACH, st[i - 1] + 0.03 if i else 0); hi = min(t + REACH, st[i + 1] - 0.03 if i + 1 < len(st) else 1e9)
            m = (tt >= lo) & (tt <= hi)
            if not m.any(): continue
            k = np.argmax(rise[m]); b = at[m][k]; i = min(max(int(t * sr), 0), len(x) - n)
            out.append((tt[m][k] - t, rise[m][k], 10 * np.log10((c[b + n] - c[b]) / n + 1e-12) - 10 * np.log10((c[i + n] - c[i]) / n + 1e-12)))
    return np.array(out).reshape(-1, 3)

if __name__ == '__main__':
    lst, dst = sys.argv[1:3]; base = os.path.dirname(os.path.abspath(lst)); rows = list(csv.DictReader(open(lst)))
    with open(dst, 'w') as f:
        f.write('recording\tsplit\tnotes\tclear\tlag10\tlag50\tlag90\tlate50\tlate100\tearly100\tsilent100\n')
        for r in rows:
            name = os.path.basename(r['pitch'])[:-5]; a = lags(os.path.join(base, r['strings']), os.path.join(base, r['pitch']))
            total = max(len(a), 1); a = a[a[:, 1] >= CLEAR] if len(a) else a; c = a[:, 0] * 1000
            if not len(c): f.write(f'{name}\t{r["split"]}\t{total}\t0\t\t\t\t\t\t\t\n'); continue
            p = np.percentile(c, (10, 50, 90))
            f.write(f'{name}\t{r["split"]}\t{total}\t{len(c) / total:.3f}\t{p[0]:.0f}\t{p[1]:.0f}\t{p[2]:.0f}\t{np.mean(c > 50):.3f}\t{np.mean(c > 100):.3f}\t{np.mean(c < -100):.3f}\t{np.mean((c > 100) & (a[:, 2] >= QUIET)):.3f}\n'); f.flush()
