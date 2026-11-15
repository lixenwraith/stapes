#!/usr/bin/env python3
"""wavscan.py <folder> <out.tsv>: every WAV under a folder by its own header and samples, a line a file: format,
seconds, peak, RMS, DC offset and the samples at full scale (a run of three or more is a clipped peak). It prints
the totals: formats, peaks, quiet files, clipped ones and those that do not read. A set is looked at with it before
anything is read from it. Reads only."""
import collections, math, os, struct, sys
import numpy as np

def read(path):
    """(rate, channels, bytes a sample, float?, the samples in full scale as frames by channels)"""
    b = open(path, 'rb').read()
    if b[:4] != b'RIFF' or b[8:12] != b'WAVE':
        raise ValueError('not a RIFF WAVE file')
    i, fmt, data = 12, None, None
    while i + 8 <= len(b) and data is None:
        tag, n = b[i:i + 4], struct.unpack('<I', b[i + 4:i + 8])[0]
        if tag == b'fmt ':
            fmt = b[i + 8:i + 8 + n]
        elif tag == b'data':
            data = b[i + 8:i + 8 + n]
        i += 8 + n + (n & 1)
    if fmt is None or data is None:
        raise ValueError('no fmt chunk or no data chunk')
    kind, ch, rate, _, _, bits = struct.unpack('<HHIIHH', fmt[:16])
    if kind == 0xfffe and len(fmt) >= 26:
        kind = struct.unpack('<H', fmt[24:26])[0]
    w = bits // 8
    data = data[:len(data) // (w * ch) * (w * ch)]
    if kind == 3 and w in (4, 8):
        x = np.frombuffer(data, '<f4' if w == 4 else '<f8').astype(np.float64)
    elif kind == 1 and w == 3:
        raw = np.frombuffer(data, np.uint8).reshape(-1, 3).astype(np.int32)
        x = (raw[:, 0] | raw[:, 1] << 8 | ((raw[:, 2] ^ 0x80) - 0x80) << 16) / 2.0 ** 23
    elif kind == 1 and w == 1:
        x = (np.frombuffer(data, np.uint8).astype(np.float64) - 128) / 128
    elif kind == 1 and w in (2, 4):
        x = np.frombuffer(data, '<i2' if w == 2 else '<i4').astype(np.float64) / 2.0 ** (bits - 1)
    else:
        raise ValueError(f'format tag {kind} at {bits} bits')
    return rate, ch, w, kind == 3, x.reshape(-1, ch)

def run(mask):
    """the longest run of true in a 1-d mask"""
    if not mask.any():
        return 0
    edge = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return int((edge[1::2] - edge[::2]).max())

db = lambda v: 20 * math.log10(v) if v > 0 else -math.inf
root, out = os.path.abspath(sys.argv[1]), sys.argv[2]
formats, peaks, bad, rows, secs = collections.Counter(), [], [], [], 0.0
for d, _, fs in sorted(os.walk(root)):
    for f in sorted(fs):
        if not f.lower().endswith('.wav'):
            continue
        rel = os.path.relpath(os.path.join(d, f), root)
        try:
            rate, ch, w, fl, x = read(os.path.join(d, f))
        except Exception as e:
            bad.append((rel, str(e)))
            continue
        if not len(x):
            bad.append((rel, 'no samples'))
            continue
        top = 1.0 if fl else 1 - 2.0 ** (1 - 8 * w)
        full = np.abs(x) >= top
        peak, rms = float(np.abs(x).max()), float(np.sqrt((x ** 2).mean()))
        rows.append((rel, rate, ch, w, 'float' if fl else 'int', len(x) / rate, db(peak), db(rms), float(x.mean()),
                     int(full.sum()), max(run(full[:, c]) for c in range(ch))))
        formats[rate, ch, w, 'float' if fl else 'int'] += 1
        peaks.append(db(peak))
        secs += len(x) / rate
with open(out, 'w') as f:
    f.write('path\trate\tchannels\tbytes\tkind\tseconds\tpeak_db\trms_db\tmean\tfull_scale\tlongest_run\n')
    for r in rows:
        f.write('%s\t%d\t%d\t%d\t%s\t%.3f\t%.2f\t%.2f\t%.6f\t%d\t%d\n' % r)
peaks.sort()
q = lambda p: peaks[min(int(p * len(peaks)), len(peaks) - 1)]
print(len(rows), 'WAV files read,', round(secs / 3600, 2), 'hours;', len(bad), 'do not read', bad[:3])
print('formats (rate, channels, bytes a sample, kind):', dict(formats))
if peaks:
    print(f'peaks in dB of full scale: lowest {peaks[0]:.1f}, a twentieth under {q(.05):.1f}, median {q(.5):.1f}, highest {peaks[-1]:.1f};',
          sum(1 for p in peaks if p < -30), 'files peak under -30 dB,', sum(1 for p in peaks if p < -40), 'under -40 dB')
    clip = sorted((r for r in rows if r[10] >= 3), key=lambda r: -r[10])
    print(sum(1 for r in rows if r[9]), 'files hold a sample at full scale;', len(clip), 'hold a run of three or more:',
          [(r[0], r[10]) for r in clip[:5]])
    off = sorted(rows, key=lambda r: -abs(r[8]))[:3]
    print('the largest means (a DC offset), of full scale:', [(r[0], round(r[8], 5)) for r in off])
    print('the quietest:', [(r[0], round(r[6], 1)) for r in sorted(rows, key=lambda r: r[6])[:5]])
