#!/usr/bin/env python3
"""loop_sheet.py <corpus dir> <out dir> [picks.tsv] [overlaps]: a listening page of a rendered corpus's loops beside
their labels: each loop as rendered and its label's notes as plain tones on one timeline, with a picture of the label,
a kit a lane and a tone a piece. With no picks it draws seven arranged loops of different meters by a seed, overlaps
four overlaps more; a picks file names loops (number, tab, the one question). Run corpusheard.py first: the page asks
only whether a loop sounds like music. Needs each loop's WAV and ffmpeg; the README says how the page is read."""
import html, json, math, os, random, sys, tempfile, wave
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sheetkit import RATE, name, write, player, page, resample
from corpusheard import voice, own, hz, length, db, band_hz, third
args = sys.argv[1:]; OVERLAPS = 'overlaps' in args; args = [a for a in args if a != 'overlaps']
D = args[0].rstrip('/'); OUT = os.path.abspath(args[1]); PICKS = args[2] if len(args) > 2 else None
HOP = 256 / 22050; FAMILIES = 'piano guitar voice bass strings winds synth drums unknown'.split(); DRUMS = ['kick', 'snare', 'hihat']
LOUD = -18.0   # dBFS a clip's loudest half second is set to on the page
LANE = 11      # a drum's or a piece's lane under the notes, in the picture's units

def loop(i):
    lab = json.load(open(f'{D}/loop_{i:03d}.json'))
    with wave.open(f'{D}/loop_{i:03d}.wav') as w:
        sr, ch = w.getframerate(), w.getnchannels(); x = np.frombuffer(w.readframes(w.getnframes()), '<i2').reshape(-1, ch).mean(1) / 32768
    return lab, x, sr

def level(x, sr):
    n = min(int(0.5 * sr), len(x)); e = np.concatenate([[0], np.cumsum(x * x)]); a = np.arange(0, len(x) - n + 1, max(n // 2, 1))
    return 20 * math.log10(max(math.sqrt(((e[a + n] - e[a]) / n).max()), 1e-9)), 20 * math.log10(max(float(np.abs(x).max()), 1e-9))

def most(notes):
    ring, top = [], 0
    for e in sorted(notes, key=lambda e: e['frame']):
        ring = [r for r in ring if r['end'] > e['frame']]; ring.append(e); top = max(top, len(ring))
    return top

def release(e):
    """seconds a note rings after its label ends it, where the label carries its voice: the voice's release"""
    return min(voice(e['voice'])['release'], 3.0) if e.get('voice') else 0.0

def faint(vel): return max(0.06, min(1.0, 1 + 20 * math.log10(max(vel, 1e-3)) / 36))   # a bar's opacity: full at velocity 1, half at 18 dB under it

def ranks(lab):
    """each piece of a loop's kit by its place in the order of the bands, lowest first, and of equals the first numbered"""
    kit = lab.get('kit') or []; order = sorted(range(len(kit)), key=lambda k: (kit[k]['band'], k)); return {k: r for r, k in enumerate(order)}

def struck(r): return 45 + 4 * r   # the key a piece's strikes sound at on the page: A2 for the lowest band, a major third a place up

def tones(lab, n):
    """the label's events as sound: a note four partials at its key and the loop's tuning, fading over its voice's release; a drum a thump, a burst or a tick;
    a kit's strike a short tone of its piece's own, lower the lower its band, struck and gone in a tenth of a second"""
    y = np.zeros(n); rng = np.random.default_rng(1); bend = 2 ** (lab.get('tuning', 0) / 1200); rank = ranks(lab)
    for e in lab['events']:
        i = int(e['frame'] * HOP * RATE)
        if e.get('piece') in rank:
            k = np.arange(int(0.1 * RATE)) / RATE; f = 440 * 2 ** ((struck(rank[e['piece']]) - 69) / 12)
            s = (np.sin(2 * np.pi * f * k) + 0.3 * np.sin(2 * np.pi * 4 * f * k)) * np.exp(-k / 0.025) * np.minimum(k / 0.001, 1)
        elif e['drum'] >= 0:
            m = int((0.09, 0.12, 0.04)[min(e['drum'], 2)] * RATE); k = np.arange(m) / RATE
            s = (np.sin(2 * np.pi * 70 * k) if e['drum'] == 0 else rng.standard_normal(m) * (0.6 if e['drum'] == 1 else 0.35)) * np.exp(-k / ((0.03, 0.035, 0.01)[min(e['drum'], 2)]))
            if e['drum'] == 2: s = np.diff(s, prepend=0)
        else:
            held = int(max(e['end'], e['frame'] + 1) * HOP * RATE) - i; tail = max(int(release(e) * RATE), int(0.008 * RATE))
            if held < 8: continue
            f = 440 * 2 ** ((e['midi'] - 69) / 12) * bend; k = np.arange(held + tail) / RATE
            s = sum(np.sin(2 * np.pi * f * h * k) / h for h in range(1, 5) if f * h < 0.45 * RATE)
            if isinstance(s, int): continue
            ramp = min(int(0.008 * RATE), held // 2); s[:ramp] *= np.linspace(0, 1, ramp); s[held:] *= np.linspace(1, 0, tail)
        s = s[:max(n - i, 0)]; y[i:i + len(s)] += s * e['vel']
    return y

def picture(lab, secs, parts):
    notes = [e for e in lab['events'] if e['drum'] < 0 and e.get('midi', 0) > 0]; drums = sorted({e['drum'] for e in lab['events'] if e['drum'] >= 0})
    kit = lab.get('kit') or []; rank = ranks(lab); drums += [('piece', k) for k in sorted(rank, key=lambda k: -rank[k])]   # a lane a piece, the highest band on top
    lo, hi = (min(e['midi'] for e in notes) - 2, max(e['midi'] for e in notes) + 2) if notes else (58, 62)
    rh = max(3.0, min(9.0, 300 / (hi - lo + 1))); left, top, pw = 46, 14, 664; base = top + (hi - lo + 1) * rh; lanes = len(drums) * LANE; h = base + lanes + 34; W = left + pw + 10
    X = lambda s: left + s / secs * pw; Y = lambda k: top + (hi - k) * rh + rh / 2
    o = [f'<svg viewBox="0 0 {W} {h:.0f}" data-left="{left}" data-pw="{pw}" role="img" aria-label="The label\'s notes, beats and bar lines">']
    for k in range(lo, hi + 1):
        if k % 12 == 0: o.append(f'<line class="grid" x1="{left}" x2="{left + pw}" y1="{Y(k):.1f}" y2="{Y(k):.1f}"/><text class="ax" x="{left - 6}" y="{Y(k) + 3:.1f}" text-anchor="end">{name(k)}</text>')
    for f in lab.get('beats', []): o.append(f'<line class="grid" x1="{X(f * HOP):.1f}" x2="{X(f * HOP):.1f}" y1="{top}" y2="{base + lanes}"/>')
    for f in lab.get('downbeats', []): o.append(f'<line class="bar" x1="{X(f * HOP):.1f}" x2="{X(f * HOP):.1f}" y1="{top - 4}" y2="{base + lanes}"/>')
    for s in range(int(secs) + 1): o.append(f'<text class="ax" x="{X(s):.1f}" y="{base + lanes + 24}" text-anchor="middle">{s} s</text>')
    for e in notes:
        a, b = e['frame'] * HOP, max(e['end'], e['frame'] + 1) * HOP; r = min(b + release(e), secs); c = f'var(--p{parts[(e.get("part"), e.get("instr"))] % 8})'; op = faint(e['vel'])
        if r - b > 0.05: o.append(f'<rect x="{X(b):.1f}" y="{Y(e["midi"]) - max(rh / 6, 0.6):.1f}" width="{X(r) - X(b):.1f}" height="{max(rh / 3, 1.2):.1f}" style="fill:{c};opacity:{op * 0.45:.2f}"/>')
        o.append(f'<rect x="{X(a):.1f}" y="{Y(e["midi"]) - rh / 2 + 0.5:.1f}" width="{max(X(b) - X(a), 1.5):.1f}" height="{max(rh - 1, 2):.1f}" rx="1.5" style="fill:{c};opacity:{op:.2f}"><title>{name(e["midi"])}, {a:.2f} to {b:.2f} s' + (f', ringing to {r:.2f} s' if r - b > 0.05 else '') + f', velocity {e["vel"]:.2f} ({20 * math.log10(max(e["vel"], 1e-3)):+.0f} dB), {html.escape(str(e.get("part")))} on {html.escape(str(e.get("instr")))}</title></rect>')
    lane = lambda d: DRUMS[min(d, 2)] if isinstance(d, int) else f'{d[1]} {hzs(band_hz(kit[d[1]]["band"]))}'
    for e in lab['events']:
        d = ('piece', e['piece']) if e.get('piece') in rank else e['drum'] if e['drum'] >= 0 else None
        if d is not None:
            y = base + 4 + drums.index(d) * LANE; x = X(e['frame'] * HOP); what = f'piece {d[1]}, band {kit[d[1]]["band"]} ({hzs(band_hz(kit[d[1]]["band"]))}Hz), struck on the page at {name(struck(rank[d[1]]))}' if d in drums and not isinstance(d, int) else lane(d)
            o.append(f'<rect x="{x:.1f}" y="{y}" width="2.5" height="7" style="fill:var(--p7);opacity:{faint(e["vel"]):.2f}"><title>{what}, {e["frame"] * HOP:.2f} s, velocity {e["vel"]:.2f}</title></rect>')
    for d in drums: o.append(f'<text class="ax" x="{left - 6}" y="{base + 11 + drums.index(d) * LANE}" text-anchor="end">{lane(d)}</text>')
    o.append(f'<line class="head" x1="{left}" x2="{left}" y1="{top - 6}" y2="{base + lanes + 8}"/>')
    return ''.join(o) + '</svg>'

def hzs(f): return f'{f / 1000:.1f}k' if f >= 1000 else f'{f:.0f}'

def kitchips(lab, x, sr):
    """each piece of a loop's kit, lowest band first: its band, the key it strikes on the page, its strikes and how many rise 6 dB about its band"""
    kit = lab.get('kit') or []; rank = ranks(lab); out = []
    for k in sorted(rank, key=rank.get):
        f = band_hz(kit[k]['band']); w = int(max(0.03, 4 / f) * sr); ups = []
        for e in lab['events']:
            if e.get('piece') == k:
                t0 = e['frame'] * HOP; a = [v for v in (third(x, sr, t0 + d, f, w) for d in (0, 0.01, 0.02)) if v is not None]
                if a: ups.append(db(max(a), third(x, sr, t0 - 0.005 - w / sr, f, w) or 0.0) >= 6)
        out.append(f'piece {k} ({hzs(f)}Hz) at {name(struck(rank[k]))}: {len(ups)} strike{"" if len(ups) == 1 else "s"}, {sum(ups)} rising')
    return out

def shown(lab, x, sr, es):
    """each of a part's notes at its onset: whether its own first four partials stand 6 dB over what was there just before, and their power"""
    tun = lab.get('tuning', 0) / 100; out = []
    for e in es:
        f = hz(e, tun); w = int(max(0.06, 4 / f) * sr); t0 = e['frame'] * HOP; t1 = t0 + length(e)
        after = [v for v in (own(x, sr, t0 + d, f, w) for d in (0.01, 0.04, 0.08, 0.14) if t0 + d < max(t1, t0 + 0.05)) if v is not None]
        if after: out.append((db(max(after), own(x, sr, t0 - 0.005 - w / sr, f, w) or 0.0) >= 6, max(after)))
    return out

def loud(x):
    """a sound set to the page's loudness by its loudest half second, turned down if its peak would clip"""
    lv, _ = level(x, RATE); y = x * 10 ** ((LOUD - lv) / 20); return y * min(1.0, 0.89 / max(np.abs(y).max(), 1e-9))

ids = sorted(int(os.path.basename(p)[5:-5]) for p in os.listdir(D) if p.startswith('loop_') and p.endswith('.json') and os.path.exists(f'{D}/{p[:-5]}.wav'))
chosen = []
if PICKS:
    for line in open(PICKS):
        if line.strip() and not line.startswith('#'):
            i, _, ask = line.rstrip('\n').partition('\t'); chosen.append((int(i), ask or None))
else:
    rng = random.Random(1115); order = ids[:]; rng.shuffle(order); seen_m, seen_b, arr, ov = set(), set(), [], []
    for want_short in (True, False):
        for i in order:
            lab = json.load(open(f'{D}/loop_{i:03d}.json')); secs = lab['frames'] * HOP
            if want_short and secs > 10: continue
            if lab.get('kind') == 'arranged' and len(arr) < 7 and (lab.get('meter') or '4/4') not in seen_m:
                seen_m.add(lab.get('meter') or '4/4'); arr.append(i)
            elif OVERLAPS and lab.get('kind') == 'overlap' and len(ov) < 4:
                k = len({e['instr'] for e in lab['events']}); b = 0 if k == 1 else 1 if k <= 3 else 2 if k <= 6 else 3
                if b not in seen_b: seen_b.add(b); ov.append(i)
    chosen = [(i, None) for i in arr + ov]

os.makedirs(OUT, exist_ok=True); sections, listing = [], []
with tempfile.TemporaryDirectory() as tmp:
    for n, (i, ask) in enumerate(chosen, 1):
        lab, x, sr = loop(i); secs = len(x) / sr; lv, pk = level(x, sr)
        rec = loud(resample(x, sr)); said = loud(tones(lab, len(rec)))
        stem = f'{n:02d}-loop_{i:03d}'; fa, fb = os.path.join(OUT, stem + '.wav'), os.path.join(OUT, stem + '-label.wav'); write(fa, rec); write(fb, said)
        notes = [e for e in lab['events'] if e['drum'] < 0 and e.get('midi', 0) > 0]
        parts = {}
        for e in lab['events']:
            if e['drum'] < 0 and 'piece' not in e: parts.setdefault((e.get('part'), e.get('instr')), len(parts))
        kind = lab.get('kind', 'a loop')
        if kind == 'arranged':
            head = f'arranged, {lab.get("meter") or "4/4"} at {lab["bpm"]}' + (', starting part-way into a bar' if lab.get('downbeats') and lab['downbeats'][0] > 0 else '') + (f', drums on a kit of {len(lab["kit"])}' if lab.get('kit') else '')
        else:
            head = f'{kind}, {len(parts)} instrument{"" if len(parts) == 1 else "s"}, no meter' if kind == 'overlap' else kind
        chips = []; got = {key: shown(lab, x, sr, [e for e in notes if (e.get('part'), e.get('instr')) == key]) for key in parts}; top = max((a for g in got.values() for _, a in g), default=1e-16)
        for (p, ins), k in parts.items():
            es = [e for e in notes if (e.get('part'), e.get('instr')) == (p, ins)]; g = got[(p, ins)]; eng = any(e.get('voice') for e in es); rel = max((release(e) for e in es), default=0)
            under = -float(np.median([db(a, top) for _, a in g])) if g else 0.0
            chips.append(f'<span style="background:var(--p{k % 8})"></span>{html.escape(str(p))} on {html.escape(str(ins))}' + (f' (the engine\'s, family {FAMILIES[max(es[0].get("family", 8), 0)]}' + (f', ringing {rel:.1f} s after a note is let go' if rel > 0.3 else '') + ')' if eng else ' (recorded)')
                         + f': {len(es)} note{"" if len(es) == 1 else "s"}, {sum(u for u, _ in g)} of {len(g)} showing at their start, a median {under:.0f} dB under the loop\'s strongest note')
        chips = '; '.join(chips)
        if any(e['drum'] >= 0 for e in lab['events']): chips += '; <span style="background:var(--p7)"></span>drums'
        if lab.get('kit'): chips += f'; <span style="background:var(--p7)"></span>a kit of {len(lab["kit"])} struck sounds, the lowest band first: ' + '; '.join(kitchips(lab, x, sr))
        facts = f'{secs:.1f} s. {len(notes)} notes, at most {most(notes)} held at once by the label, velocities from {min((e["vel"] for e in notes), default=0):.2f} to {max((e["vel"] for e in notes), default=0):.2f}. As rendered its loudest half second is {lv:.1f} dBFS and its peak {pk:.1f} dBFS; on this page both sounds are set to {LOUD:.0f} dBFS by their loudest half second. Tuning {lab.get("tuning", 0):+.0f} cents.'
        q = f'<p class="ask"><b>The question:</b> {html.escape(ask)}</p>' if ask else ''
        sections.append(f'<section data-secs="{secs:.3f}"><h2>{n}. loop {i}</h2><p class="tag">{html.escape(head)}</p>{q}{picture(lab, secs, parts)}'
                        f'<div class="tr"><button data-src="0">The loop</button><button data-src="1">The label\'s notes</button><span class="time">0.0 s</span></div>{player(fa, tmp)}{player(fb, tmp)}'
                        f'<p class="key small">{chips}</p><p class="small">{facts}</p><p class="small"><code>{html.escape(D.replace(os.path.expanduser("~"), "~"))}/loop_{i:03d}.wav</code></p></section>')
        listing.append(f'{n:2d}. loop {i}: {head}; {secs:.1f} s, {len(notes)} notes, at most {most(notes)} at once, {lv:.1f} dBFS, peak {pk:.1f} -> {stem}.wav, {stem}-label.wav')
kits = any(json.load(open(f'{D}/loop_{i:03d}.json')).get('kit') for i, _ in chosen)
asks = '' if PICKS else '<p><b>The one question, for each loop:</b> does it sound like music a recording could hold? If not, say what is off and when (the loop\'s number and a time).</p>'
intro = (f'<p>Loops of the corpus as it was rendered. Whether each loop holds the notes its label says was counted before this page was cut, and the count is written under each clip, so you are not asked to match them. A clip has two sounds on one timeline: the loop, and its label\'s notes as plain tones made for this page, there only to show what the label says if a loop puzzles you. Press one, then the other: each goes on from where the other was. The line on the picture is where you are, and a click or a drag on the picture moves it.</p>{asks}'
         '<p class="small"><b>The picture:</b> a bar is a note of the label, in its part\'s colour, as faint as it is soft (half strength at 18 dB under full). A thin tail after a bar is the voice ringing on after the note is let go, which the label counts as the voice and not as the note. Thin upright lines are the label\'s beats and the stronger ones its bar lines.</p>'
         + ('<p class="small"><b>A kit:</b> where a loop\'s drums are a kit of struck sounds, each piece has a lane under the notes, the piece whose band is highest on top, named by its number and its band. In the label\'s sound each piece strikes a short tone of its own, the lowest band\'s at A2 and each next a major third up, so a piece is told by its tone; under the clip each piece names its tone, its strikes and how many rise 6 dB about its band.</p>' if kits else ''))
open(os.path.join(OUT, 'sheet.html'), 'w').write(page('A corpus heard before it trains', f'{html.escape(os.path.basename(D))}, {len(chosen)} of its {len(ids)} loops', intro, sections))
open(os.path.join(OUT, 'files.txt'), 'w').write('\n'.join(listing) + '\n')
print('\n'.join(listing)); print(f'sheet.html: {os.path.getsize(os.path.join(OUT, "sheet.html")) / 1e6:.1f} MB, {len(chosen)} loops')
