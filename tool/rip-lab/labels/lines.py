#!/usr/bin/env python3
"""lines.py <out dir> <engine|nsynth> <seed> <count> <fit|out> [tuned|shapes] | lines.py <out dir> verify: lines of
one voice whose notes are known, each a WAV and its truth, for scoring a label procedure: the engine's,
rendered by lines/ (ENGINELINES names its binary), among them takes of VocalSet's exercises sung right and wrong,
and NSynth's vocal notes laid end to end (NSYNTH names its audio folder). ../README.md says what is drawn."""
import csv, itertools, json, math, os, random, struct, subprocess, sys, wave
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # shapes.py is beside this file, under python -I too
import shapes

SR, LENGTH, LOW, HIGH = 44100, 12.0, 36, 86
LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NSYNTH = os.path.join(os.path.expanduser(os.environ.get('NSYNTH', '~/data/rip/nsynth/nsynth-train/audio')), '')
STEPS = [(0, 8), (1, 11), (-1, 11), (2, 15), (-2, 15), (3, 5), (-3, 5), (4, 5), (-4, 5), (5, 4), (-5, 4), (7, 3.5), (-7, 3.5), (12, 2.5), (-12, 2.5)]

def sound_pitch(seg, sr):
    """a held sound's pitch as a MIDI number, apart from vocal.py's own measure; 0 for none. The pitch whose first
    ten harmonics hold the most of the spectrum, each weighed 0.84 of the one under it (subharmonic summation), so a
    weak fundamental or a strong third harmonic does not move it an octave or a twelfth"""
    if not len(seg) or not seg.any():
        return 0.0
    seg = seg - seg.mean()
    size = 1 << 16
    mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), size))
    f = 55 * 2 ** (np.arange(0, 5600, 10) / 1200)
    score = np.zeros(len(f))
    for h in range(1, 11):
        at = f * h * size / sr
        ok = at < len(mag) - 1
        lo = at[ok].astype(int)
        score[ok] += 0.84 ** (h - 1) * (mag[lo] * (1 - (at[ok] - lo)) + mag[lo + 1] * (at[ok] - lo))
    i = min(max(int(np.argmax(score)), 1), len(f) - 2)
    a, c, d = score[i - 1], score[i], score[i + 1]
    fine = i + (0.5 * (a - d) / (a - 2 * c + d) if a - 2 * c + d < 0 else 0)
    return 69 + 12 * math.log2(55 * 2 ** (fine / 120) / 440)

def envelope(x, sr, ms=10):
    """the RMS of x in windows of ms"""
    w = max(int(sr * ms / 1000), 1)
    n = len(x) // w
    return np.sqrt((x[:n * w].reshape(n, w) ** 2).mean(1)) if n else np.zeros(0)

def resample(x, sr, to):
    n = int(round(len(x) * to / sr))
    spec, out = np.fft.rfft(x), np.zeros(n // 2 + 1, complex)
    k = min(len(spec), len(out))
    out[:k] = spec[:k]
    return np.fft.irfft(out, n) * (n / len(x))

def ring(x, db):
    """the band from 1.5 to 4 kHz raised by db, where a trained voice's ring holds most of its energy"""
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return np.fft.irfft(np.fft.rfft(x) * 10 ** (db * np.clip((f - 1200) / 300, 0, 1) * np.clip((4500 - f) / 500, 0, 1) / 20), len(x))

def warp(x, cents):
    """x played so that each of its samples sounds cents (a value a sample) higher: (the sound, the place in it of
    each sample of x and of x's end, in samples). Four-point interpolation; a pitch moved a semitone or two."""
    at = np.concatenate([[0.0], np.cumsum(2.0 ** (-np.asarray(cents, float) / 1200))])
    src = np.interp(np.arange(int(at[-1])), at[:-1], np.arange(len(x)))
    i = np.clip(src.astype(int), 1, len(x) - 3)
    f = src - i
    a, b, c, d = x[i - 1], x[i], x[i + 1], x[i + 2]
    return b + 0.5 * f * (c - a + f * (2 * a - 5 * b + 4 * c - d + f * (3 * (b - c) + d - a))), at

def sung(notes, step, offs, n, wide):
    """the cents each of a rendered take's n samples is moved by: every note its own distance off its key, a slide
    ending on the pitch the next note is sung at (a note with no slide lets go 5 ms before the next), and a vibrato
    (cents either way, Hz, phase) wider than the engine's where the take has one"""
    c, start = np.full(n, 100.0 * offs[-1]), 0
    for i, (pos, dur, key, vel, scoop, bend, at) in enumerate(notes[:-1]):
        nxt = min(int(notes[i + 1][0] * step * SR), n)
        c[start:nxt] = 100 * offs[i]
        a = int(((pos + 1 + at * (dur - 1)) if scoop else (pos + at * dur)) * step * SR) if bend else nxt - int(0.005 * SR)
        a = min(max(a, start), nxt)
        c[a:nxt] = np.linspace(100 * offs[i], 100 * offs[i + 1], nxt - a, endpoint=False)
        start = nxt
    if wide:
        c += wide[0] * np.sin(2 * np.pi * wide[1] * np.arange(n) / SR + wide[2])
    return c

def chest(x, hz, db, width):
    """a resonance: the spectrum raised by db at hz, falling off as a bell `width` octaves wide"""
    f = np.maximum(np.fft.rfftfreq(len(x), 1 / SR), 1.0)
    return np.fft.irfft(np.fft.rfft(x) * 10 ** (db * np.exp(-0.5 * (np.log2(f / hz) / width) ** 2) / 20), len(x))

def noise(nrng, n, lo, hi):
    """n samples of noise between lo and hi Hz under a raised cosine, at an RMS of one"""
    spec = np.fft.rfft(nrng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    spec[(f < lo) | (f > hi)] = 0
    y = np.fft.irfft(spec, n) * np.hanning(n)
    return y / (np.sqrt((y ** 2).mean()) + 1e-12)

def phrase(rng, step, pick):
    """a line's notes on the step grid as [pos, dur, key, vel, scoop, bend, at]"""
    total = int(math.ceil(LENGTH / step / 16)) * 16
    notes, pos, key = [], rng.randrange(0, 3), pick(None, 0)
    while pos < total - 1:
        dur = max(1, min(round(math.exp(rng.uniform(math.log(0.1), math.log(2.5))) / step), 64, total - pos))
        notes.append([pos, dur, key, 10 ** (rng.uniform(-26, 0) / 20), 0, 0.0, 0.0])
        pos += dur
        if rng.random() < 0.3:
            pos += max(1, round(math.exp(rng.uniform(math.log(0.15), math.log(1.0))) / step))
        key = pick(key, rng.choices([s for s, _ in STEPS], [w for _, w in STEPS])[0])
    return notes

def ornaments(rng, notes, step, voice=False):
    """glides into the next note and scoops from below, on the engine's notes; a voice slides into a short note too"""
    for i, n in enumerate(notes):
        nxt = notes[i + 1] if i + 1 < len(notes) else None
        joined = nxt is not None and nxt[0] == n[0] + n[1]
        after = i > 0 and notes[i - 1][0] + notes[i - 1][1] == n[0]
        # a singer slides into most notes that follow another, in a twentieth to a quarter of a second, and scoops
        # into half of those that follow silence
        if joined and nxt[2] != n[2] and abs(nxt[2] - n[2]) <= 7 and n[1] * step >= (0.1 if voice else 0.2) and rng.random() < 0.7:
            n[5], n[6] = float(nxt[2] - n[2]), 1 - min(rng.uniform(0.03 if voice else 0.05, 0.25), 0.6 * n[1] * step) / (n[1] * step)
        if not after and n[1] >= 2 and step <= 0.15 and n[2] - 3 >= LOW and rng.random() < 0.5:
            n[4] = rng.choice([1, 2, 3])

def exercise(rng):
    """a take of one of the sheet's exercises on the step grid: its name, its tempo, the notes sung as phrase gives
    them, and what is wrong with the take, if anything: a note missing, a bar sung twice, a restart after two to four
    notes, or a wrong note two to four semitones from the one written"""
    kind = rng.choices(['arpeggios', 'scales', 'long_tones'], [43, 43, 14])[0]
    fast = kind != 'long_tones' and rng.random() < 0.3
    secs = (rng.uniform(2.0, 5.0) if kind == 'long_tones' else rng.uniform(0.14, 0.24) if fast and kind == 'arpeggios'
            else rng.uniform(0.12, 0.2) if fast else rng.uniform(0.5, 1.2) if kind == 'arpeggios' else rng.uniform(0.35, 0.7))
    k = next(k for k in (32, 16, 8, 4, 2, 1) if 72 <= round(15 * k / secs) <= 190)      # steps a note
    bpm = round(15 * k / secs)
    step = (SR * 60 // (bpm * 4)) / SR
    rel = [0] + list(itertools.accumulate(shapes.SHAPE[kind]))
    tonic = rng.randint(40, min(HIGH - 2 - max(rel), 67))
    sung = [tonic + r for r in rel]
    wrong = rng.choices(['', 'a note missing', 'a bar twice', 'a restart', 'a wrong note'], [60, 10, 10, 10, 10])[0]
    whole = len(sung)
    if wrong == 'a note missing':
        del sung[rng.randrange(1, len(sung) - 1)]
    elif wrong == 'a wrong note':
        j = rng.randrange(1, len(sung) - 1)
        sung[j] += rng.choice([d for d in (-4, -3, -2, 2, 3, 4) if sung[j] + d not in (sung[j - 1], sung[j + 1])])
    parts = [sung, list(sung)] if wrong == 'a bar twice' else [sung[:rng.randint(2, min(4, len(sung) - 1))], sung] if wrong == 'a restart' else [sung]
    detached = fast and k >= 2 and rng.random() < 0.5
    notes, pos = [], rng.randrange(0, 3)
    urng = random.Random(rng.random())                                                  # a draw of its own for the lengths
    for part in parts:
        for i, key in enumerate(part):
            slot = 2 * k if kind == 'scales' and i == len(part) - 1 and len(part) >= whole - 1 else k    # a scale ends on a quarter
            if k >= 4 and kind != 'long_tones':                                         # no two notes of a slow take are one length
                slot += urng.choice([-1, 0, 0, 1])
            dur = max(1, round(slot * rng.uniform(0.5, 0.85))) if detached else slot
            notes.append([pos, dur, key, 10 ** (rng.uniform(-6, 0) / 20), 0, 0.0, 0.0])
            pos += slot
            if kind == 'long_tones' and i < len(part) - 1 and rng.random() < 0.8:      # a breath between two long tones
                pos += max(1, round(rng.uniform(0.3, 1.5) / step))
        pos += max(1, round(rng.uniform(0.5, 1.5) / step))                              # and before starting again
    return kind, bpm, step, notes, wrong

def finish(rng, nrng, x, truth, peaks):
    """the noises of a recording laid beside a line's notes, and its level: (int16 samples, noise rows, peak dB)"""
    on = np.zeros(len(x), bool)
    for s, e, *_ in truth:
        on[int(s * SR):int(e * SR)] = True
    level = math.sqrt(float((x[on] ** 2).mean())) if on.any() else 1.0
    rows, edges = [], [(0.0, truth[0][0])] + [(a[1], b[0]) for a, b in zip(truth, truth[1:])] + [(truth[-1][1], len(x) / SR)]
    for a, b in edges:
        if b - a >= 0.35 and rng.random() < 0.5:           # a breath in a rest
            n = rng.uniform(0.15, min(0.5, b - a - 0.1))
            at = rng.uniform(a + 0.05, b - 0.05 - n)
            x[int(at * SR):int(at * SR) + int(n * SR)] += noise(nrng, int(n * SR), 500, 6000) * level * 10 ** (rng.uniform(-26, -10) / 20)
            rows.append((at, at + n))
        if b - a >= 0.1 and b < len(x) / SR - 0.05 and rng.random() < 0.3:   # a consonant before a note from silence
            n = rng.uniform(0.04, min(0.09, b - a))
            i = int((b - n) * SR)
            if i >= 0:
                x[i:i + int(n * SR)] += noise(nrng, int(n * SR), 2500, 12000) * level * 10 ** (rng.uniform(-20, -6) / 20)
                rows.append((b - n, b))
    peak = rng.choice(peaks)
    x = x / (np.abs(x).max() + 1e-12) * 10 ** (peak / 20)
    return np.round(x * 32767).astype('<i2'), sorted(rows), peak

def write(out, name, x, truth, noises, offs=None):
    with wave.open(f'{out}/{name}.wav', 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(x.tobytes())
    with open(f'{out}/{name}.tsv', 'w') as f:
        f.write('kind\tstart\tend\tkey\tvel\tscoop\tbend\tat\toff\n')
        for i, (s, e, k, v, sc, b, at) in enumerate(truth):
            f.write(f'note\t{s:.6f}\t{e:.6f}\t{k}\t{v:.3f}\t{sc}\t{b:g}\t{at:.3f}\t{offs[i] if offs else 0:.3f}\n')
        for s, e in noises:
            f.write(f'noise\t{s:.6f}\t{e:.6f}\t0\t0\t0\t0\t0\t0\n')

def engine(out, seed, count, which, peaks, mode=''):
    spec, lines = [], []
    for i in range(count):
        rng = random.Random(f'engine-{which}-{seed}-{i}')
        kind, wrong = '', ''
        if mode == 'shapes':
            kind, bpm, step, notes, wrong = exercise(rng)
        else:
            bpm = rng.randint(72, 190)
            step = (SR * 60 // (bpm * 4)) / SR
            centre = rng.randint(LOW + 9, HIGH - 9)
            lo, hi = centre - 9, centre + 9
            def pick(key, iv, rng=rng, lo=lo, hi=hi):
                if key is None:
                    return rng.randint(lo, hi)
                return key + iv if lo <= key + iv <= hi else key - iv if lo <= key - iv <= hi else key
            notes = phrase(rng, step, pick)
        ornaments(rng, notes, step, mode == 'shapes')
        voice = dict(Wave='saw' if which == 'fit' else 'square', Vowel=rng.choice(['a', 'e', 'i', 'o', 'u', '']),
                     Attack=rng.uniform(0.005, 0.06), Decay=rng.uniform(0.05, 0.3), Sustain=rng.uniform(0.6, 1.0),
                     Release=rng.uniform(0.01, 0.04) if mode == 'shapes' else rng.uniform(0.03, 0.15))
        if rng.random() < 0.7:
            voice.update(VibratoHz=rng.uniform(4.5, 7.0), VibratoCents=rng.uniform(20, 100), VibratoDelay=rng.uniform(0, 0.3))
        if rng.random() < 0.5:
            voice.update(Cutoff=rng.uniform(1500, 6000), Q=0.7)
        events = []
        for pos, dur, key, vel, scoop, bend, at in notes:
            if scoop:
                events.append(dict(Pos=pos, Dur=1, MIDI=key - scoop, Vel=vel, Bend=float(scoop), At=0.0))
                events.append(dict(Pos=pos + 1, Dur=dur - 1, MIDI=key, Vel=vel, Bend=bend, At=at))
            else:
                events.append(dict(Pos=pos, Dur=dur, MIDI=key, Vel=vel, Bend=bend, At=at))
        name = f'engine{i:04d}'
        # a singer's own pitch standard: a draw of its own, so an untuned draw's lines are what they were
        trng = random.Random(f'tuning-{which}-{seed}-{i}')
        tuning = 0.0 if not mode else trng.choice([-50.0, 50.0]) if trng.random() < 0.125 else trng.uniform(-50, 50)
        # one note of three in ten right takes is sung most of a semitone off its key and is the note meant all the
        # same
        off, offs, spread, wide, res = (-1, 0.0), None, 0.0, None, None
        if mode == 'shapes':
            if not wrong and trng.random() < 0.3:
                off = (trng.randrange(len(notes)), trng.uniform(0.5, 0.9) * (-1 if tuning > 0 else 1))
            # how the take is sung: each note its own distance off its key, the take's spread a draw; centred by
            # the notes' lengths, so the take's standard is the tuning drawn
            srng = random.Random(f'sung-{which}-{seed}-{i}')
            spread = math.exp(srng.uniform(math.log(0.07), math.log(0.40)))
            offs = [next(d for d in iter(lambda: srng.gauss(0, spread), None) if abs(d) <= 0.5) for _ in notes]
            order = sorted(range(len(notes)), key=lambda j: offs[j])
            half, run = sum(n[1] for n in notes) / 2, 0
            for j in order:
                run += notes[j][1]
                if run >= half:
                    offs = [d - offs[j] for d in offs]
                    break
            if off[0] >= 0:
                offs[off[0]] = off[1]
            if srng.random() < 0.15:
                wide = (srng.uniform(100, 170), srng.uniform(4.3, 7.7), srng.uniform(0, 2 * math.pi))
                for k in ('VibratoHz', 'VibratoCents', 'VibratoDelay'):
                    voice.pop(k, None)
            if srng.random() < 0.25:
                res = (math.exp(srng.uniform(math.log(330), math.log(1050))), srng.uniform(14, 28), srng.uniform(0.1, 0.25))
        spec.append(dict(Name=name, BPM=bpm, Tuning=tuning, Voice=voice, Notes=events))
        lines.append((name, rng, bpm, step, voice, notes, tuning, kind, wrong, off, offs, spread, wide, res))
    json.dump(spec, open(f'{out}/spec.json', 'w'))
    subprocess.run([os.environ['ENGINELINES'], f'{out}/spec.json', out], check=True, stdout=open(f'{out}/render.log', 'w'))
    rows = []
    for i, (name, rng, bpm, step, voice, notes, tuning, kind, wrong, off, offs, spread, wide, res) in enumerate(lines):
        raw = open(f'{out}/{name}.raw.wav', 'rb').read()
        x = np.frombuffer(raw[44:], '<f4').astype(np.float64)
        os.remove(f'{out}/{name}.raw.wav')
        truth = [(pos * step, (pos + dur) * step, key, vel, scoop, bend, at) for pos, dur, key, vel, scoop, bend, at in notes]
        if offs:                                            # the take as it is sung, and where its notes now are
            x, place = warp(x, sung(notes, step, offs, len(x), wide))
            there = lambda t: float(place[min(int(round(t * SR)), len(place) - 1)]) / SR
            truth = [(there(s), there(e), *rest) for s, e, *rest in truth]
        if res:
            x = chest(x, *res)
        rung = rng.uniform(10, 35) if rng.random() < 0.5 else 0.0
        if rung:
            x = ring(x, rung)
        y, noises, peak = finish(rng, np.random.default_rng([seed, i, 1]), x, truth, peaks)
        write(out, name, y, truth, noises, offs)
        rows.append((name, 'engine', f"{voice['Wave']} {voice['Vowel'] or 'none'}", bpm, round(voice.get('VibratoCents', 0)), round(voice.get('VibratoHz', 0), 2),
                     round(voice.get('Cutoff', 0)), round(peak, 1), round(len(y) / SR, 2), len(notes), round(voice['Release'], 3), round(rung, 1), round(tuning, 1), kind, wrong, off[0], round(off[1], 3),
                     round(spread, 3), round(wide[0]) if wide else 0, round(wide[1], 2) if wide else 0,
                     round(res[0]) if res else 0, round(res[1], 1) if res else 0, round(res[2], 3) if res else 0))
    return rows

def nsynth_notes(out, which):
    """the vocal_acoustic notes on disk an instrument, each checked against its label: {instrument: {key: [file]}}"""
    table, left, held = {}, [], 0
    for f in sorted(os.listdir(NSYNTH)):
        if not f.startswith('vocal_acoustic'):
            continue
        held += 1
        inst, key, vel = f[:-4].rsplit('-', 2)
        with wave.open(NSYNTH + f) as w:
            sr = w.getframerate()
            x = np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(np.float64) / 32768
        # a note is laid from its first sample, so its pitch has to be its label's from a tenth of a second on
        reads = [sound_pitch(x[int(t * sr):int((t + 0.1) * sr)], sr) for t in (0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0)]
        heard = max(reads, key=lambda r: abs(r - int(key)))
        if not LOW <= int(key) <= HIGH or min(reads) <= 0 or abs(heard - int(key)) > 0.6 or np.abs(x).max() < 0.01:
            left.append((f, int(key), round(heard, 2), round(float(np.abs(x).max()), 4)))
            continue
        table.setdefault(inst, {}).setdefault(int(key), []).append(f)
    with open(f'{out}/notes-left-out.tsv', 'w') as f:
        f.write('file\tlabel\theard\tpeak\n' + ''.join('%s\t%d\t%s\t%s\n' % r for r in left))
    names = sorted(n for n, keys in table.items() if len(keys) >= 5)
    random.Random(0).shuffle(names)
    cut = len(names) * 2 // 3
    mine = sorted(names[:cut] if which == 'fit' else names[cut:])
    print(f'NSynth held {held} vocal_acoustic training notes when this ran; {len(left)} left out for a label their sound does not hold,',
          f'a silence or a key out of reach; {len(names)} instruments with five keys or more, {len(mine)} of them {which}: {" ".join(n[15:] for n in mine)}')
    return {n: table[n] for n in mine}

def nsynth(out, seed, count, which, peaks):
    table = nsynth_notes(out, which)
    names, rows = sorted(table), []
    for i in range(count):
        rng = random.Random(f'nsynth-{which}-{seed}-{i}')
        inst = names[i % len(names)]
        keys = sorted(table[inst])
        bpm = rng.randint(72, 190)
        step = (SR * 60 // (bpm * 4)) / SR
        def pick(key, iv, rng=rng, keys=keys):
            if key is None:
                return rng.choice(keys)
            k = min(keys, key=lambda p: (abs(p - (key + iv)), p))
            if k == key and iv:
                j = keys.index(key) + (1 if iv > 0 else -1)
                k = keys[j] if 0 <= j < len(keys) else key
            return k
        notes = phrase(rng, step, pick)
        sr = 16000
        x = np.zeros(int((notes[-1][0] + notes[-1][1]) * step * sr) + sr)
        truth = []
        for pos, dur, key, vel, *_ in notes:
            with wave.open(NSYNTH + rng.choice(table[inst][key])) as w:
                y = np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(np.float64) / 32768
            n = min(int(dur * step * sr), len(y))
            y = y[:n].copy()
            fade = min(int(0.03 * sr), n // 2)
            y[n - fade:] *= np.cos(np.linspace(0, math.pi / 2, fade)) ** 2
            env = envelope(y, sr, 50)
            y *= vel / (env.max() + 1e-12)
            fine = envelope(y, sr, 10)
            rise = int(np.argmax(fine >= 0.1 * fine.max())) * 0.01
            at = int(pos * step * sr)
            x[at:at + n] += y
            truth.append((pos * step + rise, pos * step + n / sr - fade / sr / 2, key, vel, 0, 0.0, 0.0))
        name = f'nsynth{i:04d}'
        y, noises, peak = finish(rng, np.random.default_rng([seed, i, 2]), resample(x, sr, SR), truth, peaks)
        write(out, name, y, truth, noises)
        rows.append((name, 'nsynth', inst[15:], bpm, 0, 0, 0, round(peak, 1), round(len(y) / SR, 2), len(notes), 0, 0, 0, '', '', -1, 0, 0, 0, 0, 0, 0, 0))
    return rows

def verify(out, rows):
    """each line read back apart from vocal.py: a plain note's pitch at its middle, and the sound at its start"""
    checked, near, off, mute, worst = 0, 0, 0, 0, []
    for r in rows:
        sway = max(r[4], r[18]) / 100       # a vibrato's reach in semitones: a short window reads anywhere within it
        with wave.open(f'{out}/{r[0]}.wav') as w:
            x = np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(np.float64) / 32768
        if r[11]:
            x = ring(x, -r[11])     # a ring moves no pitch, but a reading by harmonics is surer with it taken off again
        if r[20]:
            x = chest(x, r[20], -r[21], r[22])      # and so is a chest's resonance
        for at, l in enumerate(csv.DictReader(open(f'{out}/{r[0]}.tsv'), delimiter='\t')):
            s, e, k = float(l['start']), float(l['end']), int(l['key']) + float(l.get('off') or 0)
            if l['kind'] != 'note' or float(l['bend']) or int(l['scoop']) or e - s < 0.2:
                continue
            checked += 1
            mid = [sound_pitch(x[int(t * SR):int(t * SR) + int(0.1 * SR)], SR) for t in np.linspace(s + 0.25 * (e - s), s + 0.75 * (e - s) - 0.1, 5)]
            mid = sorted(m for m in mid if m > 0)
            heard = mid[len(mid) // 2] - r[12] / 100 if mid else 0.0   # a tuned line sounds its tuning off its keys
            near += 0.5 + sway < abs(heard - k) <= 1 + sway
            if abs(heard - k) > 1 + sway:
                off += 1
                worst.append((r[0], round(s, 2), k, round(heard, 2)))
            seg = envelope(x[int(s * SR):int(e * SR)], SR)
            mute += not len(seg) or seg[:8].max() < 0.1 * seg.max()
    print(f'read back: {checked} plain notes of 0.2 s or more; {near} read between a half and a whole semitone from their key (past the vibrato\'s reach),',
          f'{off} more than a semitone {worst[:6]}; {mute} have no sound in their first 80 ms')

if __name__ == '__main__':
    if sys.argv[2] == 'verify':     # lines.py <out dir> verify: the lines of a folder read back again
        rows = [(r['name'], r['kind'], r['voice'], 0, float(r['vibrato_cents']), 0, 0, 0, 0, 0, 0, float(r['ring_db']), float(r.get('tuning') or 0), r.get('exercise', ''), r.get('wrong', ''),
                 int(r.get('off_note') or -1), float(r.get('off_by') or 0), float(r.get('spread') or 0), float(r.get('wide_cents') or 0), float(r.get('wide_hz') or 0),
                 float(r.get('chest_hz') or 0), float(r.get('chest_db') or 0), float(r.get('chest_width') or 0))
                for r in csv.DictReader(open(f'{sys.argv[1]}/lines.tsv'), delimiter='\t')]
        sys.exit(verify(os.path.abspath(sys.argv[1]), rows))
    out, kind, seed, count, which = os.path.abspath(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    mode = sys.argv[6] if len(sys.argv) > 6 else ''
    assert kind in ('engine', 'nsynth') and which in ('fit', 'out') and mode in ('', 'tuned', 'shapes') and not (mode and kind == 'nsynth')
    os.makedirs(out, exist_ok=True)
    peaks = [float(r['peak_db']) for r in csv.DictReader(open(f'{LAB}/datasets/logs/vocalset-wavscan.tsv'), delimiter='\t')]
    rows = engine(out, seed, count, which, peaks, mode) if kind == 'engine' else nsynth(out, seed, count, which, peaks)
    with open(f'{out}/lines.tsv', 'w') as f:
        f.write('name\tkind\tvoice\tbpm\tvibrato_cents\tvibrato_hz\tcutoff\tpeak_db\tseconds\tnotes\trelease\tring_db\ttuning\texercise\twrong\toff_note\toff_by\tspread\twide_cents\twide_hz\tchest_hz\tchest_db\tchest_width\n')
        for r in rows:
            f.write('\t'.join(str(v) for v in r) + '\n')
    print(f'{len(rows)} {kind} lines ({which} voices, seed {seed}) in {out}: {sum(r[9] for r in rows)} notes, {sum(r[8] for r in rows) / 60:.1f} minutes')
    verify(out, rows)
