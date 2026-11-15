"""sheetkit.py: what the listening pages share: a clip is one timeline with its sounds on it, each going on from
where the other was, a playhead on the picture and no player that mixes them. write, resample and player make a
sound; STYLE and SCRIPT are the page's; a section carries data-secs, its picture's svg data-left and data-pw (the
plot's left edge and width in its own units) and a line of class head, its buttons data-src (the audio each plays)."""
import base64, os, subprocess, wave
import numpy as np
RATE = 22050
NAMES = 'C C# D D# E F F# G G# A A# B'.split()
name = lambda m: f'{NAMES[m % 12]}{m // 12 - 1}'

def resample(x, sr, rate=RATE):
    """x at rate, band-limited: its spectrum cut or padded to the new length"""
    n = int(round(len(x) * rate / sr))
    if n == len(x) or len(x) < 2: return np.asarray(x, float)
    return np.fft.irfft(np.fft.rfft(x)[:n // 2 + 1], n) * (n / len(x))

def runs(mask, t, least):
    """the spans of time over which mask holds for least seconds or more"""
    out, a = [], None
    for i, v in enumerate(list(mask) + [False]):
        if v and a is None: a = i
        elif not v and a is not None:
            if t[i - 1] - t[a] >= least: out.append((float(t[a]), float(t[i - 1]), a, i))
            a = None
    return out

def write(path, y):
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE); w.writeframes((np.clip(y, -1, 1) * 32767).astype('<i2').tobytes())

def player(path, tmp):
    """a sound of a clip's timeline: an audio element with no controls of its own, an MP3 copy where ffmpeg makes one"""
    mp3 = os.path.join(tmp, os.path.basename(path)[:-4] + '.mp3')
    done = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', path, '-codec:a', 'libmp3lame', '-b:a', '56k', mp3]).returncode == 0
    data, kind = (open(mp3, 'rb').read(), 'mpeg') if done else (open(path, 'rb').read(), 'wav')
    return f'<audio preload="auto" src="data:audio/{kind};base64,{base64.b64encode(data).decode()}"></audio>'

STYLE = '''
:root{--bg:#faf9f6;--fg:#1c1b19;--mut:#66635c;--card:#fff;--line:#e2dfd7;--note:#3567c9;--off:#c9760f;--dot:#1c1b19;--bare:#d6453f;--sung:#8f8b82;--ask:#f1eee6;
--p0:#3567c9;--p1:#2c8a57;--p2:#b0531f;--p3:#7a4fb5;--p4:#b03a6e;--p5:#1f8a9a;--p6:#8a7a1f;--p7:#56606e}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#eceae4;--mut:#a5a299;--card:#1d1d1b;--line:#35342f;--note:#7da4f4;--off:#f0a746;--dot:#eceae4;--bare:#ff7a70;--sung:#77746b;--ask:#2a2925;
--p0:#7da4f4;--p1:#6fcf97;--p2:#f0a46a;--p3:#b79af0;--p4:#f08ab4;--p5:#6fd0de;--p6:#d8c868;--p7:#a9b3c2}}
body{background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif;margin:0;padding:16px}main{max-width:780px;margin:0 auto}h1{font-size:1.5rem;margin:.2em 0}h2{font-size:1.1rem;margin:0 0 .2em}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:18px 0}.tag{color:var(--bare);font-weight:600;margin:.1em 0 .5em}.small{font-size:.85rem;color:var(--mut);overflow-wrap:anywhere}
.ask{background:var(--ask);border-radius:8px;padding:10px 12px;margin:.4em 0}
svg{width:100%;height:auto;display:block;margin:.5em 0;cursor:pointer;touch-action:none}audio{display:none}ul{margin:.3em 0;padding-left:1.2em}
.tr{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:.2em 0 .6em}button{font:inherit;padding:9px 16px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.on{background:var(--note);border-color:var(--note);color:#fff}button:disabled{opacity:.55;cursor:default}.time{font-variant-numeric:tabular-nums;color:var(--mut);margin-left:4px}.head{stroke:var(--bare);stroke-width:1.6}
.grid{stroke:var(--line);stroke-width:1}.bar{stroke:var(--mut);stroke-width:1}.ax{fill:var(--mut);font:10px system-ui,sans-serif}.note{fill:var(--note)}.note.off{fill:var(--off)}.dot{fill:var(--dot)}.bare{fill:var(--bare);opacity:.22}.sung{fill:var(--sung)}
.key span{display:inline-block;width:12px;height:10px;border-radius:2px;margin:0 4px 0 12px;vertical-align:baseline}
'''

SCRIPT = '''
document.querySelectorAll('section[data-secs]').forEach(function (sec) {
  var secs = +sec.dataset.secs, svg = sec.querySelector('svg'), head = svg.querySelector('.head'), read = sec.querySelector('.time');
  var auds = [].slice.call(sec.querySelectorAll('audio')), btns = [].slice.call(sec.querySelectorAll('button[data-src]'));
  var L = +svg.dataset.left, PW = +svg.dataset.pw, t = 0, cur = null;
  function show() { var x = L + Math.min(t / secs, 1) * PW; head.setAttribute('x1', x); head.setAttribute('x2', x); read.textContent = t.toFixed(1) + ' s'; }
  function stop() { auds.forEach(function (a) { a.pause(); }); btns.forEach(function (b) { b.classList.remove('on'); }); cur = null; }
  function tick() { if (!cur) return; t = cur.currentTime; show(); requestAnimationFrame(tick); }
  auds.forEach(function (a) {
    a.addEventListener('timeupdate', function () { if (cur === a) { t = a.currentTime; show(); } });
    a.addEventListener('ended', function () { if (cur === a) { stop(); t = 0; show(); } }); });
  btns.forEach(function (b) { b.addEventListener('click', function () {
    var a = auds[+b.dataset.src]; if (cur) t = cur.currentTime;
    if (cur === a) { stop(); show(); return; }
    stop(); if (t >= secs - 0.05) t = 0;
    a.currentTime = t; a.play(); cur = a; b.classList.add('on'); show(); requestAnimationFrame(tick); }); });
  function seek(e) { var r = svg.getBoundingClientRect(), x = (e.clientX - r.left) / r.width * svg.viewBox.baseVal.width;
    t = Math.min(Math.max((x - L) / PW, 0), 0.999) * secs; if (cur) cur.currentTime = t; show(); }
  svg.addEventListener('pointerdown', function (e) { seek(e); svg.setPointerCapture(e.pointerId); svg.onpointermove = seek; });
  svg.addEventListener('pointerup', function () { svg.onpointermove = null; });
  show();
});
'''

def page(title, sub, intro, sections):
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>'
            f'<style>{STYLE}</style></head><body><main><h1>{title}</h1><p class="small">{sub}</p>{intro}{"".join(sections)}</main><script>{SCRIPT}</script></body></html>')
