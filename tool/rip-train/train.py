"""Train the rip's front end on a corpus stapes exported (`corpus <dir>`), and
export the weights lixenwraith/transformer loads:

    python3 train.py --corpus corpus/ --out ripmodel.bin --steps 2000
    python3 train.py --corpus corpus/ --out ripnotes.bin --pitchnet ...   # the notes model

Each loop's log-mel frames (loop_NNN.mel) are the input; its events
(loop_NNN.json) become per-frame targets: an onset within a frame of any event,
each drum's hits, the pitches sounding, whether any sounds, the wave and voice
genes of the highest note sounding, each pitch within a frame of its strikes and
of its releases, and how hard each was struck. Training crops random windows of
frames and builds their targets a window at a time, so memory holds the events
rather than a row per frame.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from itertools import accumulate

import numpy as np
import torch
import torch.nn.functional as F

import ripmodel

# frames read before a window for PCEN's average to settle, over three of its time constants
WARM = 128


class MelFile:
    """A loop's log-mel frames, read from its file as sliced: the frames of 40000
    loops are 8 GB, which the page cache holds rather than the process."""

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            self.frames, self.bands = struct.unpack("<II", f.read(8))

    def __len__(self) -> int:
        return self.frames

    def __getitem__(self, sl: slice) -> np.ndarray:
        start, stop, _ = sl.indices(self.frames)
        n = max(stop - start, 0)
        return np.fromfile(self.path, dtype="<f4", count=n * self.bands, offset=8 + 4 * start * self.bands).reshape(n, self.bands)


class Label:
    """A loop's events as arrays, its per-frame targets built a window at a time
    (window), so memory holds the events and not a row per frame: a corpus of
    recordings runs to tens of millions of frames."""

    def __init__(self, label: dict, frames: int, manifest: dict):
        ev = label["events"]
        self.frames, self.manifest = frames, manifest
        self.t = np.array([e["frame"] for e in ev], np.int32)
        self.end = np.array([e["end"] for e in ev], np.int32)
        self.drum = np.array([e["drum"] for e in ev], np.int8)
        self.midi = np.array([e["midi"] for e in ev], np.int16)
        self.vel = np.array([e.get("vel", -1) for e in ev], np.float32)
        self.wave = np.array([e.get("wave", -1) if e.get("voice") else -1 for e in ev], np.int8)
        voices, index = [], {}  # a loop's notes share a few voices
        self.voice = np.full(len(ev), -1, np.int32)
        for i, e in enumerate(ev):
            if e.get("voice"):
                self.voice[i] = index.setdefault(tuple(e["voice"]), len(index))
                if self.voice[i] == len(voices):
                    voices.append(e["voice"])
        self.voices = np.array(voices, np.float32).reshape(-1, manifest["voice_genes"])
        self.order = np.argsort(self.t, kind="stable")
        self.sorted = self.t[self.order]
        # how far past its frame an event's targets reach
        self.reach = int((np.maximum(self.end, self.t + 2) - self.t).max()) + 1 if len(ev) else 0

    def hits(self, a: int, b: int):
        """the drum events in frames a to b, as (frame, drum)"""
        i = self.order[np.searchsorted(self.sorted, a):np.searchsorted(self.sorted, b)]
        i = i[self.drum[i] >= 0]
        return zip(self.t[i].tolist(), self.drum[i].tolist())

    def window(self, a: int, b: int, with_drums: bool = True) -> dict:
        """Per-frame targets for frames a to b from the events; masks say where a
        target counts. The yes-or-no ones are bytes until batch; the strikes and
        releases are lists of (frame, pitch) from a, each strike's velocity beside
        it, -1 where the corpus does not say."""
        m, frames, w = self.manifest, self.frames, b - a
        pitches, min_midi = m["pitches"], m["min_midi"]
        onset = np.zeros(w, np.uint8)
        drums = np.zeros((w, 3), np.uint8)
        pitch = np.zeros((w, pitches), np.uint8)
        wave = np.full(w, -1, np.int64)
        voice = np.zeros((w, m["voice_genes"]), np.float32)
        top = np.full(w, -1, np.int64)  # the highest midi sounding, whose voice the frame reports
        who = np.full(w, -1, np.int64)  # the event that is
        strikes, releases, vels = [], [], []

        def span(x, y):  # frames x to y of the loop, in the window
            x, y = max(x, a), min(y, b)
            return slice(x - a, max(y, x) - a)

        lo, hi = np.searchsorted(self.sorted, a - self.reach), np.searchsorted(self.sorted, b + 1, "right")
        for i in np.sort(self.order[lo:hi]):  # in the label's order, which the voice a frame reports keeps
            t = int(self.t[i])
            if t < 0 or t >= frames or (self.drum[i] >= 0 and not with_drums):
                continue
            onset[span(max(t - 1, 0), min(t + 2, frames))] = 1
            if self.drum[i] >= 0:
                drums[span(max(t - 1, 0), min(t + 2, frames)), self.drum[i]] = 1
                continue
            p, midi = int(self.midi[i]) - min_midi, int(self.midi[i])
            end = min(max(int(self.end[i]), t + 1), frames)
            if 0 <= p < pitches:
                pitch[span(t, end), p] = 1
                strikes.append((t - a, p))
                vels.append(self.vel[i])
                if self.end[i] < frames:
                    releases.append((int(self.end[i]) - a, p))
            if self.wave[i] >= 0:
                seg = span(t, end)
                up = midi > top[seg]
                top[seg][up], who[seg][up] = midi, i
        heard = who >= 0
        wave[heard], voice[heard] = self.wave[who[heard]], self.voices[self.voice[who[heard]]]
        return {"onset": onset, "drums": drums, "pitch": pitch, "voicing": (pitch.sum(1) > 0).astype(np.uint8),
                "wave": wave, "voice": voice, "voiced": (top >= 0).astype(np.uint8),
                "strikes": np.array(strikes, np.int32).reshape(-1, 2), "vels": np.array(vels, np.float32),
                "releases": np.array(releases, np.int32).reshape(-1, 2)}


def load_corpus(d: str):
    with open(os.path.join(d, "corpus.json")) as f:
        manifest = json.load(f)
    loops = []
    skip = set(open(os.path.join(d, "skip.txt")).read().split()) if os.path.exists(os.path.join(d, "skip.txt")) else set()
    for i in range(manifest["loops"]):
        if f"loop_{i:03d}" in skip:
            continue
        base = os.path.join(d, f"loop_{i:03d}")
        mel = MelFile(base + ".mel")
        with open(base + ".json") as f:
            label = json.load(f)
        loops.append((mel, Label(label, len(mel), manifest)))
    return manifest, loops


class Channel:
    """What a small speaker, a room and a phone do to frames, drawn per window: the
    bands under a low corner and over a high one cut, a ripple across them, and a
    steady noise floor added, in power. A corpus's noise sits 40 to 65 dB under the
    loop; a phone's floor reached 7 to 15 dB under the music, which cost 8b half its recall."""

    def __init__(self, path: str):
        d = np.load(path)
        self.white, self.room = d["white"], d["room"].astype(np.float64)
        mel = lambda h: 2595 * np.log10(1 + h / 700)
        points = np.linspace(mel(30), mel(11025), ripmodel.MEL_BANDS + 2)[1:-1]
        self.points, self.hz = points, 700 * (10 ** (points / 2595) - 1)

    def draw(self, n: int, rng: random.Random):
        """the gain per band and the noise over n frames, in the energy a stored frame holds"""
        octave = np.log2(self.hz)
        low, high = 60 * (400 / 60) ** rng.random(), 3000 * (10000 / 3000) ** rng.random()
        cut = -rng.random() * 24 * np.clip((np.log2(low) - octave) / 1.5, 0, 1) - rng.random() * 24 * np.clip(octave - np.log2(high), 0, 1)
        knots = np.linspace(self.points[0], self.points[-1], 9)
        depth = rng.random() * 6
        ripple = np.interp(self.points, knots, [(rng.random() * 2 - 1) * depth for _ in knots])
        shape = rng.choice(["room", "room", "pink", "brown", "flat"])
        floor = {"room": self.room, "pink": -3 * (octave - octave.mean()), "brown": -6 * (octave - octave.mean()), "flat": np.zeros_like(octave)}[shape]
        floor = floor + np.interp(self.points, knots, [(rng.random() * 2 - 1) * 4 for _ in knots]) + (-78 + rng.random() * 28)
        at = rng.randrange(len(self.white) - n + 1)
        # a stored frame is log1p of a hundred times the energy
        return 10 ** ((cut + ripple) / 10), 100 * 10 ** (floor / 10) * self.white[at:at + n]


# A real hit counts when its new power tops all else sounding by this, in dB, over its best ten bands
HEARD = 0.0


def stands_out(d: np.ndarray, rest: np.ndarray, i: int) -> bool:
    """whether the hit at frame i of the drums d stands out of the rest sounding: what
    it adds counts, not the ring of the hits before it, which is why rest takes that ring"""
    ring = d[max(i - 4, 0):i - 1].mean(0) if i >= 2 else 0.0
    ratio = np.maximum(d[i:i + 3].max(0) - ring, 1e-12) / (rest[i:i + 3].mean(0) + ring + 1e-12)
    return np.sort(10 * np.log10(ratio))[-10:].mean() >= HEARD


_power = {}


def power(mel) -> float:
    """a recording's mean energy per band and frame, read once"""
    if mel.path not in _power:
        _power[mel.path] = float(np.expm1(mel[:len(mel)].astype(np.float64)).mean())
    return _power[mel.path]


def targets(mel, lab: Label, start: int, window: int, with_drums: bool = True) -> dict:
    """the targets of the window of `window` frames from start, the frames past the loop's end masked out"""
    b = min(start + window, len(mel))
    t, pad = lab.window(start, b, with_drums), window - (b - start)

    def take(a, fill=0):
        return np.concatenate([a, np.full((pad,) + a.shape[1:], fill, a.dtype)]) if pad else a
    out = {k: take(t[k]) for k in ("onset", "drums", "pitch", "voicing", "voice", "voiced")}
    out["wave"] = take(t["wave"], -1)
    shape = (window, t["pitch"].shape[1])
    strike, release, velocity, heard = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8), np.zeros(shape, np.float32), np.zeros(shape, np.uint8)
    for (f, p), v in zip(t["strikes"], t["vels"]):
        near = slice(max(f - 1, 0), max(min(f + 2, window), 0))
        strike[near, p] = 1
        if v >= 0:
            velocity[near, p], heard[near, p] = v, 1
    for f, p in t["releases"]:
        release[max(f - 1, 0):max(min(f + 2, window), 0), p] = 1
    out.update(strike=strike, release=release, velocity=velocity, heard=heard)
    out["valid"] = np.concatenate([np.ones(window - pad, np.float32), np.zeros(pad, np.float32)])
    return out


# Shuffled windows: chunks of a second, each two clips summed, spliced into a window
CHUNK = 86


def clip(mel, rng):
    """a chunk's energy from a random place in a recording, zeros past its end, and where it starts"""
    start = rng.randrange(max(len(mel) - CHUNK, 0) + 1)
    e = np.expm1(mel[start:start + CHUNK].astype(np.float64))
    return (np.concatenate([e, np.zeros((CHUNK - len(e), e.shape[1]))]) if len(e) < CHUNK else e), start


def shuffled(pick, window: int, channels: int, rng, channel=None):
    """a window of summed chunks: notes keep their labels, a drum hit counts where it
    stands out of the other clip, and what the frame's voice is goes unasked"""
    total = window + WARM if channels > 1 else window
    through, noise = channel.draw(total, rng) if channel is not None else (1.0, np.zeros((total, ripmodel.MEL_BANDS)))
    es, outs = [], []
    for at in range(0, total, CHUNK):
        (ma, la), (mb, lb) = pick(), pick()
        ea, sa = clip(ma, rng)
        eb, sb = clip(mb, rng)
        eb = eb * 10 ** ((rng.random() * 12 - 6) / 10) * power(ma) / max(power(mb), 1e-9)
        ta, tb = targets(ma, la, sa, CHUNK, False), targets(mb, lb, sb, CHUNK, False)
        out = {k: np.maximum(ta[k], tb[k]) for k in ("onset", "drums", "pitch", "voicing", "strike", "release", "velocity", "heard", "valid")}
        out.update(wave=np.full(CHUNK, -1, np.int64), voice=np.zeros_like(ta["voice"]), voiced=np.zeros_like(ta["voiced"]))
        hiss = noise[at:at + CHUNK]
        hiss = np.concatenate([hiss, np.zeros((CHUNK - len(hiss), hiss.shape[1]))]) if len(hiss) < CHUNK else hiss
        for d, rest, lab, s in ((ea * through, eb * through + hiss, la, sa), (eb * through, ea * through + hiss, lb, sb)):
            for f, drum in lab.hits(s, s + CHUNK):
                if stands_out(d, rest, f - s):
                    near = slice(max(f - s - 1, 0), min(f - s + 2, CHUNK))
                    out["onset"][near] = 1
                    out["drums"][near, drum] = 1
        es.append((ea + eb) * through + hiss)
        outs.append(out)
    e = np.concatenate(es)[:total]
    if rng.random() < RING:
        ringing(e, rng)
    first = total - window
    out = {k: np.concatenate([o[k] for o in outs])[first:total] for k in outs[0]}
    return ripmodel.features(np.log1p(e).astype(np.float32), channels)[first:], out


def sample(mel, lab: Label, start: int, window: int, channels: int, over=None, rng=None, channel=None):
    """the window of `window` frames from start: its features and targets, the
    frames past the loop's end padded with zeros and masked out; over is a recording
    of drums alone and its power against the loop's in dB, laid over the frames, its
    hits that can be heard among the targets; channel is what a speaker and a phone do to them"""
    out = targets(mel, lab, start, window)
    b = min(start + window, len(mel))
    pad = window - (b - start)

    def take(a, fill=0):
        return np.concatenate([a, np.full((pad,) + a.shape[1:], fill, a.dtype)]) if pad else a
    a = max(start - WARM, 0) if channels > 1 else start
    # uncorrelated sounds add in power, and a stored frame is log1p of a hundred times it
    e = np.expm1(take(mel[a:b]).astype(np.float64))
    if rng is not None and rng.random() < RING:
        ringing(e, rng)
    loud = e[:len(e) - pad].mean() if pad < len(e) else 0.0
    through = 1.0
    if channel is not None:
        through, noise = channel.draw(len(e), rng)
        e = e * through + noise
    if over is not None:
        omel, olab, level = over
        first, n = start - a, min(len(omel), len(e))
        at, ofrom = rng.randrange(len(e) - n + 1), rng.randrange(len(omel) - n + 1)
        d = np.expm1(omel[ofrom:ofrom + n].astype(np.float64)) * through * 10 ** (level / 10) * loud / max(power(omel), 1e-9)
        rest = e[at:at + n].copy()
        e[at:at + n] += d
        for f, drum in olab.hits(ofrom, ofrom + n):
            i = f - ofrom
            w = i + at - first
            if -2 < w < window + 1 and stands_out(d, rest, i):
                near = slice(max(w - 1, 0), max(min(w + 2, window), 0))
                out["onset"][near] = 1
                out["drums"][near, drum] = 1
    return ripmodel.features(np.log1p(e).astype(np.float32), channels)[start - a:], out


# A body's ring: what a guitar's air cavity or a room's mode adds at every attack, at one pitch whatever the note, its own harmonics absent. No note is labelled
RING = 0.0  # the share of windows that have one
RING_HZ, RING_TAU, RING_DB = (70.0, 160.0), (0.05, 0.3), (-25.0, -8.0)  # its pitch, decay in seconds, and level against the attack's rise in energy


def ringing(e, rng):
    """adds a ring to frames of energy, in place: three bands, a window's main lobe"""
    mel = lambda f: 2595 * math.log10(1 + f / 700)
    hz = math.exp(rng.uniform(math.log(RING_HZ[0]), math.log(RING_HZ[1])))
    k = int(round((mel(hz) - mel(30.0)) / (mel(11025.0) - mel(30.0)) * (ripmodel.MEL_BANDS + 1))) - 1
    total = e.sum(1)
    kick = np.maximum(np.diff(total, prepend=total[0]), 0)
    keep, y = math.exp(-(256 / 22050) / rng.uniform(*RING_TAU)), 0.0
    out = np.empty(len(e))
    for t, v in enumerate(kick):
        y = v + keep * y
        out[t] = y
    out *= 10 ** (rng.uniform(*RING_DB) / 10)
    for d, w in ((-1, 0.35), (0, 1.0), (1, 0.35)):
        e[:, k + d] += w * out


def stack(samples):
    mels, tgs = zip(*samples)
    out = {k: torch.from_numpy(np.stack([t[k] for t in tgs])) for k in tgs[0]}
    return torch.from_numpy(np.stack(mels)), {k: v if k == "wave" else v.float() for k, v in out.items()}


def batch(draw, window: int, size: int, rng: random.Random, channels: int, channel=None, share: float = 0.0, pick=None, shuffle: float = 0.0):
    """size random windows of `window` frames from the loops draw() picks, a share through the channel, a share shuffled"""
    samples = []
    for _ in range(size):
        heard = channel if channel is not None and rng.random() < share else None
        if pick is not None and rng.random() < shuffle:
            samples.append(shuffled(pick, window, channels, rng, heard))
            continue
        mel, lab, over = draw()
        samples.append(sample(mel, lab, rng.randrange(max(len(mel) - window, 0) + 1), window, channels, over, rng, heard))
    return stack(samples)


def validation(loops, window: int, channels: int):
    """every held-out loop cut into windows of `window` frames, fixed, so the held-out loss is comparable step to step"""
    mels, tgs = [], []
    for mel, lab in loops:
        for start in range(0, max(len(mel) - 1, 1), window):
            m, tg = stack([sample(mel, lab, start, window, channels)])
            mels.append(m)
            tgs.append(tg)
    return mels, tgs


def evaluate(model: ripmodel.RipModel, mels, tgs, device: str) -> dict:
    """the mean of the losses over the held-out windows, the model's weights untouched"""
    model.eval()
    sums: dict = {}
    with torch.no_grad():
        for mel, t in zip(mels, tgs):
            l = losses(model, mel.to(device), {k: v.to(device) for k, v in t.items()})
            for k, v in l.items():
                sums[k] = sums.get(k, 0.0) + v.item()
    model.train()
    return {k: v / max(len(mels), 1) for k, v in sums.items()}


def losses(model: ripmodel.RipModel, mel: torch.Tensor, t: dict) -> dict:
    y = model(mel)
    cfg = model.cfg
    valid = t["valid"]
    def bce(name, target, pos_weight):
        logits = y[..., cfg.slice(name)]
        target = target if target.dim() == 3 else target[..., None]
        l = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(pos_weight), reduction="none")
        return (l.mean(-1) * valid).sum() / valid.sum().clamp(min=1)
    out = {
        "onset": bce("onset", t["onset"], 8.0),
        "drums": bce("drums", t["drums"], 8.0),
        "pitch": bce("pitch", t["pitch"], 12.0),
        "voicing": bce("voicing", t["voicing"], 2.0),
        "strike": bce("strike", t["strike"], 20.0),
        "release": bce("release", t["release"], 20.0),
    }
    heard = t["heard"] * valid[..., None]
    velocity = torch.sigmoid(y[..., cfg.slice("velocity")])
    out["velocity"] = ((velocity - t["velocity"]) ** 2 * heard).sum() / heard.sum().clamp(min=1)
    voiced = t["voiced"] * valid
    n = voiced.sum().clamp(min=1)
    wave_logits = y[..., cfg.slice("wave")].reshape(-1, ripmodel.WAVES)
    wave_target = t["wave"].reshape(-1).clamp(min=0)
    ce = F.cross_entropy(wave_logits, wave_target, reduction="none").reshape(voiced.shape)
    out["wave"] = (ce * voiced).sum() / n
    voice = torch.sigmoid(y[..., cfg.slice("voice")])
    out["voice"] = ((voice - t["voice"]) ** 2).mean(-1).mul(voiced).sum() / n
    out["total"] = (out["onset"] + out["drums"] + out["pitch"] + 0.5 * out["voicing"] + out["strike"] + out["release"]
                    + 2.0 * out["velocity"] + 0.5 * out["wave"] + 2.0 * out["voice"])
    return out


def shape(cfg) -> str:
    """a model's sizes for a message; the output heads are ripmodel's, never a flag's"""
    return ", ".join(f"{k} {v}" for k, v in cfg.items() if k != "outputs") if cfg else "an unrecorded shape"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="directory stapes' corpus verb wrote")
    ap.add_argument("--mix", default="", help="another corpus and the share of windows drawn from it, dir:share, such as recordings with their MIDI files (corpus <dir> from <list>)")
    ap.add_argument("--over", default="", help="a corpus of drums alone and the share of windows that lay it over a loop with no drums of its own, dir:share, at -12 to +6 dB of its power against the loop's: real kits under the corpus's lines")
    ap.add_argument("--channel", default="", help="a noise bank and the share of windows heard through a small speaker, a room and a phone, file.npz:share")
    ap.add_argument("--out", required=True, help="weight file to write, the root's ripmodel.bin for stapes to hear with")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--window", type=int, default=512, help="frames per training window (11.6 ms each)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ffn", type=int, default=1024)
    ap.add_argument("--towers", default="", help="stacks of their own after the shared input, layers:heads each in the heads' order, comma-separated: 2:2,4:7 gives onset and drums two layers and the rest four")
    ap.add_argument("--convs", default="", help="convolutions before the input layer, channels x frames x features / pool each, comma-separated: 16x3x3/1,16x3x3/2,32x3x3/2")
    ap.add_argument("--attention-window", type=int, default=64, help="frames attended either side")
    ap.add_argument("--pcen", action="store_true", help="a second input channel, each band's PCEN")
    ap.add_argument("--ring", type=float, default=0.0, help="the share of windows with a body's ring at every attack, at one pitch whatever the note")
    ap.add_argument("--dilations", default="", help="PitchNet's long convolutions, their frames apart, comma-separated: 3,9,27 hears half a second either way")
    ap.add_argument("--sung", default="", help="a corpus of solo singing and the share of a shuffled window's chunks drawn from it, dir:share")
    ap.add_argument("--shuffle", type=float, default=0.0, help="the share of windows spliced from chunks of a second, each two clips summed")
    ap.add_argument("--extra", default="", help="more corpora and the share of windows drawn from each, dir:share, comma-separated")
    ap.add_argument("--pitchnet", action="store_true", help="the notes model: convolutions only over the harmonic stack, the same weights at every pitch; --out is then ripnotes.bin, which stapes reads beside ripmodel.bin")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shard-seed", type=int, default=0, help="offsets the stream the windows are drawn from, so a round's workers see different loops; the starting weights stay --seed's")
    ap.add_argument("--checkpoint", default="", help="write a .pt every --every steps, and resume from it when present")
    ap.add_argument("--init", default="", help="start from this .pt's model weights, without its optimizer state or step: a training round's common start; a --checkpoint present wins")
    ap.add_argument("--every", type=int, default=200, help="steps between checkpoints and held-out scores")
    ap.add_argument("--holdout", type=int, default=-1, help="loops held out of training and scored every --every steps; the step that scores best is what --out gets (default a tenth, none under five loops)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    global RING
    RING = args.ring

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed + (args.shard_seed << 32))  # a stream per seed and shard; shard 0 is the seed's own
    convs = [tuple(int(v) for v in c.replace("/", "x").split("x")) for c in args.convs.split(",") if c]
    if any(len(c) != 4 for c in convs):
        raise SystemExit(f"--convs {args.convs!r}: want channels x frames x features / pool, comma-separated")
    towers = [tuple(int(v) for v in t.split(":")) for t in args.towers.split(",") if t]
    if any(len(t) != 2 for t in towers) or towers and sum(h for _, h in towers) != len(ripmodel.HEADS):
        raise SystemExit(f"--towers {args.towers!r}: want layers:heads each, the heads adding up to {len(ripmodel.HEADS)}")
    heard = dict(inputs=ripmodel.STACK_INPUTS, channels=ripmodel.STACK_CHANNELS) if args.pitchnet else dict(channels=2 if args.pcen else 1)
    cfg = ripmodel.Config(**heard, convs=convs, dim=args.dim, layers=args.layers, heads=args.heads, ffn=args.ffn, window=args.attention_window, towers=towers,
                          dilations=tuple(int(d) for d in filter(None, args.dilations.split(","))))
    planes = min(cfg.channels, 2)  # what the data loader makes; the model stacks them
    resume = bool(args.checkpoint) and os.path.exists(args.checkpoint)
    start = args.checkpoint if resume else args.init
    ck = torch.load(start, map_location=args.device) if start else None
    if ck is not None and "cfg" in ck:
        ck["cfg"] = {"channels": 1, "dilations": (), **ck["cfg"]}  # a checkpoint from before those has one and none
    # refused before the corpus loads: load_state_dict names only a tensor's size,
    # and takes another head count or window without a word
    if ck is not None and ck.get("cfg") != asdict(cfg):
        raise SystemExit(f"{start} holds a model of {shape(ck.get('cfg'))}, and this run builds {shape(asdict(cfg))}: pass its shape, or another file")
    manifest, loops = load_corpus(args.corpus)
    holdout = args.holdout if args.holdout >= 0 else (len(loops) // 10 if len(loops) >= 5 else 0)
    held = loops[len(loops) - holdout:] if holdout else []
    loops = loops[:len(loops) - holdout] if holdout else loops
    val_mels, val_tgs = validation(held, args.window, planes) if held else ([], [])
    assert manifest["mel_bands"] == ripmodel.MEL_BANDS and manifest["pitches"] == ripmodel.PITCHES
    assert manifest["voice_genes"] == ripmodel.VOICE_GENES and len(manifest["waves"]) == ripmodel.WAVES
    mixed, share = [], 0.0
    if args.mix:
        d, _, sh = args.mix.rpartition(":")
        mixed_manifest, mixed = load_corpus(d)
        share = float(sh)
        assert mixed_manifest["mel_bands"] == ripmodel.MEL_BANDS and mixed_manifest["pitches"] == ripmodel.PITCHES
    drums, over_share, drumless = [], 0.0, []
    if args.over:
        d, _, sh = args.over.rpartition(":")
        over_manifest, drums = load_corpus(d)
        over_share = float(sh)
        assert over_manifest["mel_bands"] == ripmodel.MEL_BANDS
        drumless = [(m, l) for m, l in loops if not (l.drum >= 0).any()]
    model = (ripmodel.PitchNet if args.pitchnet else ripmodel.RipModel)(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    step, best_loss, best_step, best_state = 0, float("inf"), 0, None
    if resume:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        best_loss, best_step, best_state = ck.get("best", (best_loss, best_step, best_state))
        print(f"resumed at step {step}")
    elif ck is not None:
        model.load_state_dict(ck["model"])
        print(f"started from {args.init}")
    print(f"{sum(p.numel() for p in model.parameters())} weights, {len(loops)} loops, {sum(len(m) for m, _ in loops)} frames, {len(held)} held out in {len(val_mels)} windows")
    if mixed:
        print(f"{len(mixed)} loops of {sum(len(m) for m, _ in mixed)} frames mixed in, {share:.0%} of the windows")
    if drums:
        print(f"{len(drums)} recordings of {sum(len(m) for m, _ in drums)} frames laid over {len(drumless)} loops with no drums, {over_share:.0%} of the windows")
    t0 = time.time()

    def score() -> None:
        nonlocal best_loss, best_step, best_state
        if not val_mels:
            return
        v = evaluate(model, val_mels, val_tgs, args.device)
        # the decoders read the onset, drum, pitch, voicing, strike and release
        # heads; the wave, voice and velocity heads say how a note is played and
        # memorize the corpus's, so they do not pick the step exported
        v["decoders"] = v["onset"] + v["drums"] + v["pitch"] + 0.5 * v["voicing"] + v["strike"] + v["release"]
        parts = " ".join(f"{k} {x:.3f}" for k, x in v.items() if k not in ("total", "decoders"))
        mark = ""
        if v["decoders"] < best_loss:
            best_loss, best_step = v["decoders"], step
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            mark = " (best)"
        print(f"step {step} held-out {v['total']:.3f}, decoders {v['decoders']:.3f} ({parts}){mark}", flush=True)

    channel, channel_share = None, 0.0
    if args.channel:
        f, _, sh = args.channel.rpartition(":")
        channel, channel_share = Channel(f), float(sh)
        print(f"{channel_share:.0%} of the windows through a speaker, a room and a phone, from {f}")

    # recordings run from seconds to minutes, and each second is worth the same
    by_mixed = list(accumulate(len(m) for m, _ in mixed))
    by_drums = list(accumulate(len(m) for m, _ in drums))

    extras = []
    for item in filter(None, args.extra.split(",")):
        d, _, sh = item.rpartition(":")
        _, loops_x = load_corpus(d)
        extras.append((float(sh), loops_x, list(accumulate(len(m) for m, _ in loops_x))))
        print(f"{len(loops_x)} loops of {sum(len(m) for m, _ in loops_x)} frames from {d}, {float(sh):.0%} of the windows")

    def draw():
        """a loop, and the drums to lay over it or None"""
        for sh, xs, by in extras:
            if rng.random() < sh:
                return (*rng.choices(xs, cum_weights=by)[0], None)
        if drums and drumless and rng.random() < over_share:
            omel, olab = rng.choices(drums, cum_weights=by_drums)[0]
            return (*rng.choice(drumless), (omel, olab, rng.random() * 18 - 12))
        if mixed and rng.random() < share:
            return (*rng.choices(mixed, cum_weights=by_mixed)[0], None)
        return (*rng.choice(loops), None)

    sung, sung_share, by_sung = [], 0.0, []
    if args.sung:
        d, _, sh = args.sung.rpartition(":")
        _, sung = load_corpus(d)
        sung_share, by_sung = float(sh), list(accumulate(len(m) for m, _ in sung))
        print(f"{len(sung)} sung recordings of {sum(len(m) for m, _ in sung)} frames from {d}, {sung_share:.0%} of the chunks")

    def pick():
        """a recording for a shuffled window's chunk: singing, a loop, or drums alone by length"""
        if sung and rng.random() < sung_share:
            return rng.choices(sung, cum_weights=by_sung)[0]
        if drums and rng.random() < 0.3:
            return rng.choices(drums, cum_weights=by_drums)[0]
        return rng.choice(loops)

    # the next batch is built while the device works on this one; only that thread draws from rng
    pool = ThreadPoolExecutor(1)
    ahead = pool.submit(batch, draw, args.window, args.batch, rng, planes, channel, channel_share, pick, args.shuffle)
    while step < args.steps:
        mel, t = ahead.result()
        ahead = pool.submit(batch, draw, args.window, args.batch, rng, planes, channel, channel_share, pick, args.shuffle)
        mel, t = mel.to(args.device), {k: v.to(args.device) for k, v in t.items()}
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * step / args.steps))  # cosine to zero
        for g in opt.param_groups:
            g["lr"] = lr
        l = losses(model, mel, t)
        opt.zero_grad()
        l["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 20 == 0 or step == args.steps:
            parts = " ".join(f"{k} {v.item():.3f}" for k, v in l.items() if k != "total")
            print(f"step {step} loss {l['total'].item():.3f} ({parts}) {time.time() - t0:.0f}s", flush=True)
        if step % args.every == 0 or step == args.steps:
            score()
            if args.checkpoint:
                # the shape rides along for average.py and the check above, the heads and
                # the window showing in no tensor; the best step, so a resumed run, or one
                # run again once done, still exports it
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": asdict(cfg),
                            "best": (best_loss, best_step, best_state)}, args.checkpoint)
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"best held-out decoders' loss {best_loss:.3f} at step {best_step}; exporting that")
    model.eval()
    ripmodel.export(model.cpu(), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
