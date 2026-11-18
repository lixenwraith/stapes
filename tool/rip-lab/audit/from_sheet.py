#!/usr/bin/env python3
"""from_sheet.py <corpus dir> <out dir> <picks.tsv>: a listening page of a corpus made from recordings: for each pick a
stretch of the recording its label names and, on the same timeline and never mixed, what the label says as plain
sounds made for the page, with a picture of the label (notes as bars, hits in lanes, beats, bar lines) and one
question. A pick is a line: loop (with its folder, for another corpus's), start, end, question, tab-separated, then
optionally spans to shade (a-b,a-b, the clip's seconds). Needs each corpus's line beside it (<corpus>.txt) and ffmpeg."""
import csv, html, json, math, os, shutil, sys, tempfile, wave
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sheetkit import RATE, name, write, player, page, resample, envelope
D = sys.argv[1].rstrip('/'); OUT = os.path.abspath(sys.argv[2]); PICKS = sys.argv[3]
HOP = 256 / 22050; LOUD = -18.0   # a frame of the labels in seconds; the dBFS a clip's loudest half second is set to
LANES = ('kick', 'snare', 'hi-hat', 'no line')
KNOWN = {}

def about(d):
    """a corpus's list folder (a label's source is read from it), its list's rows by source, and what its skip.txt names"""
    if d not in KNOWN:
        words = open(d + '.txt').readline().split(); lst = os.path.abspath(words[words.index('from') + 1])
        KNOWN[d] = (os.path.dirname(lst), {r['audio_filename']: r for r in csv.DictReader(open(lst))},
                    set(open(d + '/skip.txt').read().split()) if os.path.exists(d + '/skip.txt') else set())
    return KNOWN[d]

def stretch(path, a, b):
    """seconds a to b of a WAV file as one channel at the page's rate"""
    with wave.open(path) as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        w.setpos(min(int(a * sr), w.getnframes())); raw = w.readframes(int((b - a) * sr))
    if sw == 3:
        u = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32); x = u[:, 0] | u[:, 1] << 8 | u[:, 2] << 16; x = np.where(x >= 1 << 23, x - (1 << 24), x) / (1 << 23)
    else:
        x = np.frombuffer(raw, {2: '<i2', 4: '<i4'}[sw]) / float(1 << (8 * sw - 1))
    return resample(x.reshape(-1, ch).mean(1), sr)

def hit(e): return e['drum'] >= 0 or e.get('midi', 0) == 0
def lane(e): return min(e['drum'], 2) if e['drum'] >= 0 else 3

def sounds(lab, a, secs, n):
    """the label's events of the stretch as sound: a note four partials at its key for as long as the label holds it,
    a kick a low thump, a snare a burst of noise, a hi-hat a short hiss, a strike of no line a higher thump (most are toms)"""
    y = np.zeros(n); rng = np.random.default_rng(1); bend = 2 ** (lab.get('tuning', 0) / 1200)
    for e in lab['events']:
        t = e['frame'] * HOP - a; v = e['vel'] if e['vel'] > 0 else 0.7
        if hit(e):
            if not 0 <= t < secs: continue
            k = np.arange(int((0.09, 0.12, 0.04, 0.14)[lane(e)] * RATE)) / RATE
            s = (np.sin(2 * np.pi * 70 * k) * np.exp(-k / 0.03), rng.standard_normal(len(k)) * 0.6 * np.exp(-k / 0.035),
                 np.diff(rng.standard_normal(len(k)) * 0.35 * np.exp(-k / 0.01), prepend=0), np.sin(2 * np.pi * 150 * k) * 0.9 * np.exp(-k / 0.045))[lane(e)]
        else:
            end = min(max(e['end'], e['frame'] + 1) * HOP - a, secs); t = max(t, 0.0); held = int((end - t) * RATE)
            if held < 8: continue
            f = 440 * 2 ** ((e['midi'] - 69) / 12) * bend; k = np.arange(held) / RATE
            s = sum(np.sin(2 * np.pi * f * h * k) / h for h in range(1, 5) if f * h < 0.45 * RATE)
            if isinstance(s, int): continue
            s = s * (0.3 + 0.7 * np.exp(-k / 0.3))  # a struck start that settles to a held tone, so a chord's notes are heard one by one
            ramp = min(int(0.008 * RATE), held // 2); s[:ramp] *= np.linspace(0, 1, ramp); s[held - ramp:] *= np.linspace(1, 0, ramp)
        i = int(t * RATE); s = s[:max(n - i, 0)]; y[i:i + len(s)] += s * v
    return y

def loud(x):
    """a sound set to the page's loudness by its loudest half second, turned down if its peak would clip"""
    n = min(RATE // 2, len(x)); e = np.concatenate([[0], np.cumsum(x * x)]); at = np.arange(0, len(x) - n + 1, max(n // 2, 1))
    lv = 20 * math.log10(max(math.sqrt(((e[at + n] - e[at]) / n).max()), 1e-9)); y = x * 10 ** ((LOUD - lv) / 20)
    return y * min(1.0, 0.89 / max(np.abs(y).max(), 1e-9)), lv

def picture(lab, a, secs, shade, rec):
    """the stretch's picture: on top the recording's waveform with the label drawn over it (an upright line where a note
    or a hit starts, a short tick where a note ends), under it the label's notes as bars and its hits in lanes"""
    ev =[e for e in lab['events'] if e['frame'] * HOP < a + secs and max(e['end'], e['frame'] + 1) * HOP > a]
    notes = [e for e in ev if not hit(e)]; hits = [e for e in ev if hit(e) and a <= e['frame'] * HOP < a + secs]; lanes = sorted({lane(e) for e in hits})
    lo, hi = (min(e['midi'] for e in notes) - 2, max(e['midi'] for e in notes) + 2) if notes else (0, -1)
    rh = max(3.0, min(9.0, 300 / (hi - lo + 1))) if notes else 0; left, wtop, wh, pw = 52, 14, 110, 658; top = wtop + wh + 10; base = top + (hi - lo + 1) * rh; low = base + len(lanes) * 14 + 4; W = left + pw + 10
    X = lambda s: left + min(max(s, 0), secs) / secs * pw; Y = lambda k: top + (hi - k) * rh + rh / 2; T = lambda f: f * HOP - a
    o = [f'<svg viewBox="0 0 {W} {low + 30:.0f}" data-left="{left}" data-pw="{pw}" role="img" aria-label="What the label says: notes, hits, beats and bar lines">']
    for s0, s1 in shade: o.append(f'<rect class="bare" x="{X(s0):.1f}" y="{wtop - 4}" width="{X(s1) - X(s0):.1f}" height="{low - wtop + 4:.1f}"><title>{s0:.2f} to {s1:.2f} s: the stretch the question asks about</title></rect>')
    mn, mx = envelope(rec, int(pw)); peak = max(float(np.abs(rec).max()), 1e-9); mid = wtop + wh / 2; xs = left + (np.arange(len(mn)) + 0.5) * pw / len(mn)
    edge = lambda v: ' '.join(f'{x:.1f},{mid - u / peak * wh / 2:.1f}' for x, u in v)
    o.append(f'<line class="mid" x1="{left}" x2="{left + pw}" y1="{mid:.1f}" y2="{mid:.1f}"/><polygon class="wave" points="{edge(zip(xs, mx))} {edge(zip(xs[::-1], mn[::-1]))}"><title>the recording</title></polygon>'
             f'<text class="ax" x="{left - 6}" y="{mid + 3:.1f}" text-anchor="end">sound</text>')
    for k in range(lo, hi + 1):
        if k % 12 == 0: o.append(f'<line class="grid" x1="{left}" x2="{left + pw}" y1="{Y(k):.1f}" y2="{Y(k):.1f}"/><text class="ax" x="{left - 6}" y="{Y(k) + 3:.1f}" text-anchor="end">{name(k)}</text>')
    for f in lab.get('beats', []):
        if 0 <= T(f) <= secs: o.append(f'<line class="grid" x1="{X(T(f)):.1f}" x2="{X(T(f)):.1f}" y1="{top}" y2="{low}"/>')
    for f in lab.get('downbeats', []):
        if 0 <= T(f) <= secs: o.append(f'<line class="bar" x1="{X(T(f)):.1f}" x2="{X(T(f)):.1f}" y1="{top - 4}" y2="{low}"/>')
    for e in notes:  # the label over the waveform: where each note starts, and a short tick where it ends
        t0, t1 = T(e['frame']), T(max(e['end'], e['frame'] + 1))
        if t0 >= 0: o.append(f'<line x1="{X(t0):.1f}" x2="{X(t0):.1f}" y1="{wtop}" y2="{wtop + wh}" style="stroke:var(--note);stroke-width:1"/>')
        else: o.append(f'<line x1="{left}" x2="{left}" y1="{wtop}" y2="{wtop + wh}" style="stroke:var(--note);stroke-width:1.4;stroke-dasharray:4 3"/>')  # held from before the clip
        if t1 <= secs: o.append(f'<line x1="{X(t1):.1f}" x2="{X(t1):.1f}" y1="{wtop + wh * 0.72:.1f}" y2="{wtop + wh}" style="stroke:var(--off);stroke-width:1.4"/>')
    for e in hits: o.append(f'<line x1="{X(T(e["frame"])):.1f}" x2="{X(T(e["frame"])):.1f}" y1="{wtop}" y2="{wtop + wh}" style="stroke:var(--p{(0, 2, 1, 4)[lane(e)]});stroke-width:1"/>')
    for s in range(int(secs) + 1): o.append(f'<text class="ax" x="{X(s):.1f}" y="{low + 20:.1f}" text-anchor="middle">{s} s</text>')
    for e in notes:
        t0, t1 = T(e['frame']), T(max(e['end'], e['frame'] + 1)); said = f'velocity {e["vel"]:.2f}' if e['vel'] > 0 else 'no velocity stated'
        o.append(f'<rect class="note" x="{X(t0):.1f}" y="{Y(e["midi"]) - rh / 2 + 0.5:.1f}" width="{max(X(t1) - X(t0), 1.5):.1f}" height="{max(rh - 1, 2):.1f}" rx="1.5"><title>{name(e["midi"])}, {t0:.2f} to {t1:.2f} s, {said}</title></rect>')
    for e in hits:
        y = base + 6 + lanes.index(lane(e)) * 14; t0 = T(e['frame'])
        o.append(f'<rect x="{X(t0) - 1.2:.1f}" y="{y:.1f}" width="2.6" height="11" style="fill:var(--p{(0, 2, 1, 4)[lane(e)]})"><title>{LANES[lane(e)]}, {t0:.2f} s</title></rect>')
    for d in lanes: o.append(f'<text class="ax" x="{left - 6}" y="{base + 15 + lanes.index(d) * 14:.1f}" text-anchor="end">{LANES[d]}</text>')
    o.append(f'<line class="head" x1="{left}" x2="{left}" y1="{wtop - 6}" y2="{low + 6:.1f}"/>')
    return ''.join(o) + '</svg>', notes, hits

picks = []
for line in open(PICKS):
    if line.strip() and not line.startswith('#'):
        p = line.rstrip('\n').split('\t'); shade = [tuple(float(v) for v in s.split('-')) for s in p[4].split(',')] if len(p) > 4 and p[4].strip() else []
        d, _, loop = p[0].rpartition('/')  # a loop of another corpus is named with its folder, beside this one or whole
        picks.append((os.path.normpath(os.path.join(os.path.dirname(D), d)) if d else D, int(loop.replace('loop_', '')), float(p[1]), float(p[2]), p[3], shade))
os.makedirs(OUT, exist_ok=True); sections, listing = [], []
with tempfile.TemporaryDirectory() as tmp:
    for n, (d, i, a, b, ask, shade) in enumerate(picks, 1):
        BASE, ROWS, SKIP = about(d)
        lab = json.load(open(f'{d}/loop_{i:03d}.json')); src = lab['source']; path = os.path.normpath(os.path.join(BASE, src)); short = os.path.basename(src)[:-4]
        b = min(b, lab['frames'] * HOP); secs = b - a
        rec, lv = loud(stretch(path, a, b)); secs = len(rec) / RATE; said, _ = loud(sounds(lab, a, secs, len(rec)))
        stem = f'{n:02d}-{short}'; fa, fb = os.path.join(OUT, stem + '.wav'), os.path.join(OUT, stem + '-label.wav'); write(fa, rec); write(fb, said)
        pic, notes, hits = picture(lab, a, secs, shade, rec); row = ROWS.get(src, {})
        key = ('<p class="small key">Over the waveform:'
               + ('<span style="background:var(--note)"></span>the label starts a note<span style="background:var(--off)"></span>the label ends a note: nothing is played there, its tone stops dead where a guitar\'s fades. A dashed line at the left edge: a note already held when the clip starts' if notes else '')
               + ('<span style="background:var(--p0)"></span>kick<span style="background:var(--p2)"></span>snare<span style="background:var(--p1)"></span>hi-hat<span style="background:var(--p4)"></span>a strike of no line' if hits else '') + '</p>')
        of = ', '.join(v for v in (row.get('kit_name'), row.get('style'), f'{row["bpm"]} bpm' if row.get('bpm') else '') if v)
        counts = '; '.join(v for v in ([f'{len(notes)} notes, {sum(e["vel"] > 0 for e in notes)} with a velocity stated'] if notes else [])
                           + [f'{sum(lane(e) == d for e in hits)} {"strikes of no line" if d == 3 else LANES[d]}' for d in sorted({lane(e) for e in hits})])
        moved = f' The label\'s hits and beats were moved {1000 * lab["lag"]:+.1f} ms to sit on the sound.' if 'lag' in lab else ''
        out = ' This recording is kept out of training for now.' if f'loop_{i:03d}' in SKIP else ''
        facts = (f'Seconds {a:g} to {b:g} of a recording of {lab["frames"] * HOP:.1f} s. In this stretch the label holds {counts or "nothing"}.{moved}{out} '
                 f'The recording\'s loudest half second here is {lv:.1f} dBFS; on this page both sounds are set to {LOUD:.0f} dBFS by their loudest half second.')
        sections.append(f'<section data-secs="{secs:.3f}"><h2>{n}. {html.escape(short)}</h2><p class="tag">{html.escape(of or "a recording and its label")}</p>'
                        f'<p class="ask"><b>The question:</b> {html.escape(ask)}</p>{pic}{key}'
                        f'<div class="tr"><button data-src="0">The recording</button><button data-src="1">The label, as sounds made for this page</button><span class="time">0.0 s</span></div>{player(fa, tmp)}{player(fb, tmp)}'
                        f'<p class="small">{facts}</p><p class="small"><code>{html.escape(path.replace(os.path.expanduser("~"), "~"))}</code>, loop {i} of {html.escape(os.path.basename(d))}</p></section>')
        listing.append(f'{n:2d}. loop {i}, {short}: {a:g} to {b:g} s; {counts}; {lv:.1f} dBFS -> {stem}.wav, {stem}-label.wav')
intro = ('<p>Stretches of recordings beside what their labels say. A clip has two sounds on one timeline, never mixed: the recording, and the label as plain sounds made for this page '
         '(a tone for a note, held for as long as the label holds it; a low thump for a kick; a burst of noise for a snare; a short hiss for a hi-hat; a higher thump for a strike the label gives to no drum). '
         'Press one, then the other: each goes on from where the other was. The line on the picture is where you are, and a click or a drag on the picture moves it. Every time in a question is the clip\'s own, as its picture shows it.</p>'
         '<p><b>Speed:</b> the row below stays at the top of the page. Slower lowers the pitch, as a tape does, so that timing stays exact; Keep pitch corrects the pitch, and can smear or double a sharp hit.</p>'
         '<p class="small"><b>The picture:</b> on top is the recording\'s waveform with the label drawn over it: an upright line where a note or a hit starts (a hit in its drum\'s colour), a short orange tick where a note ends. Under it, a bar is a note of the label, as long as the label holds it; a mark in a lane is a hit, each drum in a lane of its own; thin upright lines are the label\'s beats and the stronger ones its bar lines; a shaded stretch is the one a question asks about.</p>')
open(os.path.join(OUT, 'sheet.html'), 'w').write(page('Recordings beside their labels', f'{html.escape(os.path.basename(D))}, {len(picks)} clip{"" if len(picks) == 1 else "s"}', intro, sections))
if os.path.abspath(PICKS) != os.path.join(OUT, 'picks.tsv'): shutil.copy(PICKS, os.path.join(OUT, 'picks.tsv'))
open(os.path.join(OUT, 'files.txt'), 'w').write('\n'.join(listing) + '\n')
print('\n'.join(listing)); print(f'sheet.html: {os.path.getsize(os.path.join(OUT, "sheet.html")) / 1e6:.1f} MB, {len(picks)} clips')
