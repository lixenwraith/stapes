"""Train the rip's front end on a corpus stapes exported (`corpus <dir>`), and
export the weights lixenwraith/transformer loads:

    python3 train.py --corpus corpus/ --out ripmodel.bin --steps 2000
    python3 train.py --corpus corpus/ --out ripnotes.bin --pitchnet ...   # the notes model

Each loop's log-mel frames (loop_NNN.mel) are the input; its events
(loop_NNN.json) become per-frame targets: an onset within a frame of any event,
each drum's hits, the pitches sounding, whether any sounds, the wave and voice
genes of the highest note sounding, the family playing it, each pitch within a
frame of its strikes and of its releases, how hard each was struck, where each
pitch an engine voice plays sits from its cell's centre, and what plays each
pitch one note alone sounds in: its family, its wave and its genes. Training crops
random windows of frames and builds their targets a window at a time, so memory
holds the events rather than a row per frame.
"""
from __future__ import annotations

import argparse
import array
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


GLIDE_SPAN = 50.0  # the cents a recorded note's pitch track spans before its readings count as gliding: half a cell

# the genes of every voice the labels' notes are played by, a loop's voices in turn: a cell's target is a voice's row here
GENES = array.array("f")


def genes() -> np.ndarray:
    """GENES as a row a voice"""
    return np.frombuffer(GENES, np.float32).reshape(-1, ripmodel.VOICE_GENES)


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
        self.family = np.array([e.get("family", -1) for e in ev], np.int8)
        index = {}  # a loop's notes share a few voices
        self.voice = np.full(len(ev), -1, np.int32)  # the voice's row of GENES
        for i, e in enumerate(ev):
            if e.get("voice"):
                key = tuple(e["voice"])
                if key not in index:
                    index[key] = len(GENES) // manifest["voice_genes"]
                    GENES.extend(key)
                self.voice[i] = index[key]
        # the name a note's cells are asked: an engine voice far from the bank's is a synth, a recorded note of no family has none
        unknown, synth = ripmodel.FAMILIES.index("unknown"), ripmodel.FAMILIES.index("synth")
        self.name = np.where(self.family == unknown, np.where(self.voice >= 0, synth, -1), self.family).astype(np.int8)
        # a note that moves carries its cents from its pitch's cell centre at each frame; one that holds sits at the tuning.
        # A recording's note lists them where its pitch track places it, null, read as nan, at a frame it does not
        self.tuning = float(label.get("tuning", 0.0))
        lists = [np.asarray(e.get("cents") or [], np.float32) for e in ev]
        self.cents_at = np.array([at if len(c) else -1 for c, at in zip(lists, accumulate(map(len, lists), initial=0))], np.int64)
        self.cents = np.concatenate(lists) if lists else np.zeros(0, np.float32)
        self.moves = np.zeros(len(self.cents), bool)  # a glide's frames: where the cents differ from the frame before or after
        for i, (c, at) in enumerate(zip(lists, self.cents_at)):
            heard = c[~np.isnan(c)]
            if self.voice[i] < 0 and len(heard):  # a recording's: no track holds a note still, so every frame of one that spans GLIDE_SPAN
                self.moves[at:at + len(c)] = heard.max() - heard.min() >= GLIDE_SPAN
            elif len(c) > 1:
                d = np.abs(np.diff(c)) > 0  # false beside a frame that is not placed
                self.moves[at + 1:at + len(c)] |= d
                self.moves[at:at + len(c) - 1] |= d
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
        it, -1 where the corpus does not say. A note that moves sounds, is struck
        and is let go in the cell nearest its frequency at each frame. cents is
        each cell's offset in semitones from the sound nearest it within one, asked
        where an engine voice plays that sound, moving where it is on a glide. A
        cell one event alone sounds in is that event's (note), and says its name,
        wave and voice; one two sound in says neither's, being the sound of both."""
        m, frames, w = self.manifest, self.frames, b - a
        pitches, min_midi = m["pitches"], m["min_midi"]
        onset = np.zeros(w, np.uint8)
        drums = np.zeros((w, 3), np.uint8)
        pitch = np.zeros((w, pitches), np.uint8)
        wave = np.full(w, -1, np.int64)
        voice = np.zeros((w, m["voice_genes"]), np.float32)
        top = np.full(w, -1, np.int64)  # the highest midi sounding, whose voice the frame reports
        who = np.full(w, -1, np.int64)  # the event that is
        high = np.full(w, -1, np.int64)  # the same over every event, a hit at midi 0 on its frame, whose family the frame reports
        whose = np.full(w, -1, np.int64)
        strikes, releases, vels = [], [], []
        cents = np.zeros((w, pitches), np.float32)
        near = np.full((w, pitches), np.inf, np.float32)  # how far from a cell's centre the sound it reports sits
        asked, moving = np.zeros((w, pitches), np.uint8), np.zeros((w, pitches), np.uint8)
        sole = np.full((w, pitches), -1, np.int32)  # the event sounding in a cell, -2 where a second does

        def span(x, y):  # frames x to y of the loop, in the window
            x, y = max(x, a), min(y, b)
            return slice(x - a, max(y, x) - a)

        lo, hi = np.searchsorted(self.sorted, a - self.reach), np.searchsorted(self.sorted, b + 1, "right")
        for i in np.sort(self.order[lo:hi]):  # in the label's order, which the voice a frame reports keeps
            t = int(self.t[i])
            if t < 0 or t >= frames or (self.drum[i] >= 0 and not with_drums):
                continue
            onset[span(max(t - 1, 0), min(t + 2, frames))] = 1
            p, midi = int(self.midi[i]) - min_midi, int(self.midi[i])
            end = min(max(int(self.end[i]), t + 1), frames)
            seg = span(t, end)
            up = midi > high[seg]
            high[seg][up], whose[seg][up] = midi, i
            if self.drum[i] >= 0:
                drums[span(max(t - 1, 0), min(t + 2, frames)), self.drum[i]] = 1
                continue
            x, at = max(t, a), int(self.cents_at[i])
            rows = np.arange(x - a, max(min(end, b), x) - a)  # the note's frames in the window
            first = last = p
            placed = np.full(len(rows), self.voice[i] >= 0)  # an engine voice plays where the label says; a recording only where it lists its cents
            if at < 0:
                off, glides = np.full(len(rows), p + self.tuning / 100, np.float32), np.zeros(len(rows), bool)
                cell = np.full(len(rows), p)
            else:
                c, glides = self.cents[at + x - t:at + x - t + len(rows)], self.moves[at + x - t:at + x - t + len(rows)]
                placed = ~np.isnan(c)  # a frame that is not placed sounds in the note's own cell
                off = p + np.nan_to_num(c) / 100
                cell = np.rint(off).astype(np.int64)
                first, last = (int(np.rint(p + np.nan_to_num(self.cents[at + k]) / 100)) for k in (0, end - t - 1))
            ok = (cell >= 0) & (cell < pitches)
            r, c = rows[ok], cell[ok]
            pitch[r, c], sole[r, c] = 1, np.where(sole[r, c] == -1, i, -2)
            if 0 <= first < pitches:
                strikes.append((t - a, first))
                vels.append(self.vel[i])
            if 0 <= last < pitches and self.end[i] < frames:
                releases.append((int(self.end[i]) - a, last))
            if 0 <= p < pitches:
                near[rows[~placed], p], asked[rows[~placed], p] = 0, 0  # where it sits in its cell is not known
            for beside in (-1, 0, 1):
                cell = np.rint(off).astype(np.int64) + beside
                o = off - cell
                ok = placed & (np.abs(o) < 1) & (cell >= 0) & (cell < pitches)
                ok[ok] = np.abs(o[ok]) < near[rows[ok], cell[ok]]  # the nearest sound is the one a cell reports
                r, c = rows[ok], cell[ok]
                near[r, c], cents[r, c], asked[r, c], moving[r, c] = np.abs(o[ok]), o[ok], 1, glides[ok]
            if self.wave[i] >= 0:
                up = midi > top[seg]
                top[seg][up], who[seg][up] = midi, i
        heard = who >= 0
        wave[heard], voice[heard] = self.wave[who[heard]], genes()[self.voice[who[heard]]]
        family = np.full(w, -1, np.int64)
        family[whose >= 0] = self.family[whose[whose >= 0]]
        one = sole >= 0

        def of(a):  # an event's value at the cells it alone sounds in, -1 at the others
            out = np.full(sole.shape, -1, a.dtype)
            out[one] = a[sole[one]]
            return out
        name = of(self.name)
        if FAMILY_HOLDOUT is not None:  # the family held out is named to no head
            family[family == FAMILY_HOLDOUT], name[name == FAMILY_HOLDOUT] = -1, -1
        return {"onset": onset, "drums": drums, "pitch": pitch, "voicing": (pitch.sum(1) > 0).astype(np.uint8),
                "wave": wave, "voice": voice, "voiced": (top >= 0).astype(np.uint8), "family": family,
                "strikes": np.array(strikes, np.int32).reshape(-1, 2), "vels": np.array(vels, np.float32),
                "releases": np.array(releases, np.int32).reshape(-1, 2), "cents": cents, "asked": asked, "moving": moving,
                "note": np.where(one, sole, -1), "pitchfamily": name, "pitchwave": of(self.wave), "pitchvoice": of(self.voice)}


# a family's index in ripmodel.FAMILIES whose notes the run names to no head, so a calibration run never hears its name
FAMILY_HOLDOUT = None


def load_corpus(d: str):
    with open(os.path.join(d, "corpus.json")) as f:
        manifest = json.load(f)
    if manifest.get("families", list(ripmodel.FAMILIES)) != list(ripmodel.FAMILIES):
        raise SystemExit(f"{d}: families {manifest['families']}, the model's {list(ripmodel.FAMILIES)}")
    loops, unnamed = [], 0
    skip = set(open(os.path.join(d, "skip.txt")).read().split()) if os.path.exists(os.path.join(d, "skip.txt")) else set()
    for i in range(manifest["loops"]):
        if f"loop_{i:03d}" in skip:
            continue
        base = os.path.join(d, f"loop_{i:03d}")
        mel = MelFile(base + ".mel")
        with open(base + ".json") as f:
            label = json.load(f)
        lab = Label(label, len(mel), manifest)
        unnamed += int((lab.name == FAMILY_HOLDOUT).sum()) if FAMILY_HOLDOUT is not None else 0
        loops.append((mel, lab))
    if unnamed:
        print(f"{unnamed} notes of {d} are {ripmodel.FAMILIES[FAMILY_HOLDOUT]}'s, held out: no head is told their family")
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
        if rng.random() < DIP:  # a room's or a phone's dip under the melody: 6 to 20 dB, 0.5 to 1.5 octaves, centred 100 to 500 Hz
            centre, half, deep = np.log2(100) + rng.random() * np.log2(5), 0.25 + rng.random() * 0.5, 6 + rng.random() * 14
            ripple = ripple - deep * 0.5 * (1 + np.cos(np.pi * np.clip((octave - centre) / half, -1, 1)))
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
    out = {k: take(t[k]) for k in ("onset", "drums", "pitch", "voicing", "voice", "voiced", "cents", "asked", "moving")}
    out.update({k: take(t[k], -1) for k in ("wave", "family", "note", "pitchfamily", "pitchwave", "pitchvoice")})
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
    stands out of the other clip, what the frame's voice and family are goes unasked,
    and a cell says what plays it where one clip alone sounds in it"""
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
        out.update(wave=np.full(CHUNK, -1, np.int64), family=np.full(CHUNK, -1, np.int64), voice=np.zeros_like(ta["voice"]), voiced=np.zeros_like(ta["voiced"]))
        # a cell is asked its cents where one clip alone sounds within a semitone of it
        mine = (ta["asked"] > 0) & (tb["asked"] == 0) & (tb["pitch"] == 0)
        asked = mine | (tb["asked"] > 0) & (ta["asked"] == 0) & (ta["pitch"] == 0)
        out.update(asked=asked.astype(np.uint8), cents=np.where(mine, ta["cents"], tb["cents"]) * asked, moving=np.where(mine, ta["moving"], tb["moving"]) * asked)
        out.update({k: np.where(tb["pitch"] == 0, ta[k], np.where(ta["pitch"] == 0, tb[k], -1)) for k in ("pitchfamily", "pitchwave", "pitchvoice")})
        out["note"] = np.full_like(ta["note"], -1)  # a chunk's notes are no one loop's
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


# Bent windows: a tempo read along a slow curve, so no grid or steady pulse is there to learn
BEND_RATE, BEND_SWAY = 0.18, 12  # the window's tempo up to this share either way, swaying this many frames


def bent(mel, lab, window: int, channels: int, over, rng, heard):
    """a window read along a curve over the frames of a longer or shorter one: the features interpolated, the targets nearest"""
    rate = math.exp(rng.uniform(-BEND_RATE, BEND_RATE))
    sway, period, phase = rng.uniform(0, BEND_SWAY), rng.uniform(150, 500), rng.uniform(0, 2 * math.pi)
    n = int(window * rate + 2 * BEND_SWAY) + 2
    x, out = sample(mel, lab, rng.randrange(max(len(mel) - n, 0) + 1), n, channels, over, rng, heard)
    t = np.arange(window)
    g = np.clip(t * rate + BEND_SWAY + sway * np.sin(2 * math.pi * t / period + phase), 0, n - 1.001)
    lo = g.astype(np.int64)
    f = (g - lo)[:, None].astype(np.float32)
    near = np.rint(g).astype(np.int64)
    return x[lo] * (1 - f) + x[lo + 1] * f, {k: v[near] for k, v in out.items()}


def stack(samples):
    mels, tgs = zip(*samples)
    out = {k: torch.from_numpy(np.stack([t[k] for t in tgs])) for k in tgs[0]}
    return torch.from_numpy(np.stack(mels)), {k: v if v.dtype in (torch.int8, torch.int32, torch.int64) else v.float() for k, v in out.items()}


def batch(draw, window: int, size: int, rng: random.Random, channels: int, channel=None, share: float = 0.0, pick=None, shuffle: float = 0.0, bend: float = 0.0):
    """size random windows of `window` frames from the loops draw() picks, a share through the channel, a share shuffled, a share of the rest bent"""
    samples = []
    for _ in range(size):
        heard = channel if channel is not None and rng.random() < share else None
        if pick is not None and rng.random() < shuffle:
            samples.append(shuffled(pick, window, channels, rng, heard))
            continue
        mel, lab, over = draw()
        if rng.random() < bend:
            samples.append(bent(mel, lab, window, channels, over, rng, heard))
            continue
        samples.append(sample(mel, lab, rng.randrange(max(len(mel) - window, 0) + 1), window, channels, over, rng, heard))
    return stack(samples)


def validation(loops, window: int, channels: int):
    """every held-out loop cut into windows of `window` frames, fixed, so the held-out loss is comparable step to step;
    and their notes, a column a note in the windows' `note`: its name, its wave and its voice's row, -1 where its label says none"""
    mels, tgs, notes, first = [], [], [np.zeros((3, 0), np.int64)], 0
    for mel, lab in loops:
        for start in range(0, max(len(mel) - 1, 1), window):
            m, tg = stack([sample(mel, lab, start, window, channels)])
            tg["note"] = torch.where(tg["note"] >= 0, tg["note"] + first, tg["note"])
            mels.append(m)
            tgs.append(tg)
        notes.append(np.stack([lab.name, lab.wave, lab.voice]).astype(np.int64))
        first += len(lab.t)
    return mels, tgs, torch.from_numpy(np.concatenate(notes, 1))


# A notes model's strikes as the decoder reads them: a peak over the bar a strike starts a note at, the higher bar
# under E3, right when a labelled strike of its pitch lies within STRIKE_REACH frames
STRIKE_BARS, STRIKE_LOW, STRIKE_REACH = (0.7, 0.9), 52, 4


def struck(logits: torch.Tensor, strike: torch.Tensor, valid: torch.Tensor, low: int) -> torch.Tensor:
    """a window's peaks right, peaks, labelled strikes heard and labelled strikes, from [frames, pitches] each"""
    p = torch.sigmoid(logits)
    bar = torch.where(torch.arange(p.shape[1], device=p.device) < low, STRIKE_BARS[1], STRIKE_BARS[0])
    peak = (p >= bar) & (p >= F.max_pool1d(p.T[None], 5, 1, 2)[0].T) & (valid[:, None] > 0)
    first = (strike > 0) & (F.pad(strike[:-1], (0, 0, 1, 0)) == 0)  # a mark's first frame; the strike is the next
    onset = F.pad(first[:-1], (0, 0, 1, 0))
    near = lambda x: F.max_pool1d(x.float().T[None], 2 * STRIKE_REACH + 1, 1, STRIKE_REACH)[0].T > 0
    return torch.stack([(peak & near(onset)).sum(), peak.sum(), (onset & near(peak)).sum(), onset.sum()])


# the readings a cents head is scored over: of a cell its sound is on a glide in, of one it holds its pitch in, and of the
# cell beside a sound; and of a sounding cell the label does not place, where only the error stated can be told
CENTS_KINDS = ("gliding", "steady", "beside", "unplaced")


# A note is played by what its cells say, each by its salience, over the frames a cell is among the
# VOICE_CELLS most salient: those lixenwraith/audio keeps the say of (RipVoiceCells)
VOICE_CELLS = 12
SURE = 0.95  # the share of a held-out family's notes whose surest name stays under the measure


def cells(y: torch.Tensor, cfg, name: str) -> torch.Tensor:
    """a per-pitch head's scores, [..., pitches, values]"""
    return y[..., cfg.slice(name)].unflatten(-1, (-1, ripmodel.PITCHES)).transpose(-1, -2)


def said(y: torch.Tensor, cfg):
    """what a notes model says plays each cell, [..., pitches, each family's share, each wave's, the genes],
    and the weight a cell's say has in its note's"""
    v = torch.cat([cells(y, cfg, "pitchfamily").softmax(-1), cells(y, cfg, "pitchwave").softmax(-1), torch.sigmoid(cells(y, cfg, "pitchvoice"))], -1)
    p = y[..., cfg.slice("pitch")]
    return v, torch.sigmoid(p) * (p >= p.topk(VOICE_CELLS, -1).values[..., -1:])


def noted(heard: torch.Tensor, notes: torch.Tensor) -> dict:
    """the notes of fixed windows as they were heard against their labels: the share named their family, over
    all, by family and over those under another note; the share given their wave; their genes' RMS miss; and of
    a family held out, the share its notes' surest name reaches, SURE of them under it and at the median, with
    the share of the other notes past it and of those the share named right"""
    name, wave, voice = notes
    names, waves = ripmodel.NAMES, ripmodel.WAVES
    w = heard[:, -2]
    share, to = heard[:, :names] / w.clamp(min=1e-9)[:, None], heard[:, names:names + waves].argmax(1)
    top, named = share.max(1)
    asked, right = (w > 0) & (name >= 0) & (name != (-1 if FAMILY_HOLDOUT is None else FAMILY_HOLDOUT)), named == name
    rate = lambda at: (right[at].float().mean().item(), int(at.sum()))
    known = (w > 0) & (voice >= 0)
    miss = (heard[known, names + waves:-2] / w[known, None] - torch.from_numpy(genes()[voice[known].numpy()])) ** 2
    out = {"families": rate(asked), "by": [(ripmodel.FAMILIES[k], *rate(asked & (name == k))) for k in range(names) if (asked & (name == k)).any()],
           "under": rate(asked & (heard[:, -1] > 0.5 * w)), "waves": (to == wave)[(w > 0) & (wave >= 0)].float().mean().item(),
           "genes": miss.mean(1).sqrt().mean().item()}
    held = (w > 0) & (name == FAMILY_HOLDOUT) if FAMILY_HOLDOUT is not None else torch.zeros(0, dtype=torch.bool)
    if held.any():
        reach = top[held].quantile(SURE).item()
        out["held"] = (int(held.sum()), reach, top[held].median().item(), (top[asked] > reach).float().mean().item(), rate(asked & (top > reach))[0])
    return out


def evaluate(model: ripmodel.RipModel, mels, tgs, device: str, low: int | None = None, notes: torch.Tensor | None = None) -> dict:
    """the mean of the losses over the held-out windows, the model's weights untouched;
    with low, the first pitch over the strikes' low bar, the strikes' F1 beside them,
    and what a cents head reads: over the readings the labels place and each kind,
    the median miss and the median error it states, in cents, and the share within
    the stated; nan where a label does not say; with notes, the windows' notes
    (validation), what a model that says what plays each pitch heard them as (noted)"""
    model.eval()
    sums: dict = {}
    counts = torch.zeros(4, dtype=torch.long, device=device)
    read = []  # each asked reading's miss and stated error in cents, and its kind
    voiced = notes is not None and notes.shape[1] > 0 and "pitchfamily" in dict(model.cfg.outputs)
    # a note's cells' say summed by their weights, then the weights and those of the cells under a higher one sounding
    heard = torch.zeros(notes.shape[1] if voiced else 0, ripmodel.NAMES + ripmodel.WAVES + ripmodel.VOICE_GENES + 2, device=device)
    with torch.no_grad():
        for mel, t in zip(mels, tgs):
            mel, t = mel.to(device), {k: v.to(device) for k, v in t.items()}
            l = losses(model, mel, t)
            for k, v in l.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            if low is not None:
                out = model(mel)
                y = out[..., model.cfg.slice("strike")]
                for i in range(len(y)):
                    counts += struck(y[i], t["strike"][i], t["valid"][i], low)
                if "cents" in dict(model.cfg.outputs):
                    place, spread = out[..., model.cfg.slice("cents")].chunk(2, -1)
                    placed = t["asked"] > 0
                    ask = (placed | (t["pitch"] > 0)) & (t["valid"][..., None] > 0)
                    kind = torch.where(t["pitch"] > 0, 1 - t["moving"], torch.full_like(t["moving"], 2))
                    kind = torch.where(placed, kind, torch.full_like(kind, 3))
                    miss = torch.where(placed, 100 * (place - t["cents"]).abs(), torch.full_like(place, float("nan")))
                    read.append(torch.stack([miss[ask], 100 * spread.exp()[ask], kind[ask]]).cpu())
                if voiced:
                    v, w = said(out, model.cfg)
                    at = (t["note"] >= 0) & (t["valid"][..., None] > 0)
                    cell = torch.arange(ripmodel.PITCHES, device=device)
                    under = cell < (t["pitch"] * cell).amax(-1, keepdim=True)
                    w = w[at]
                    heard.index_add_(0, t["note"][at].long(), torch.cat([v[at] * w[:, None], w[:, None], (w * under[at])[:, None]], 1))
    model.train()
    out = {k: v / max(len(mels), 1) for k, v in sums.items()}
    if voiced:
        out["voices"] = noted(heard.cpu(), notes)
    if read and sum(r.shape[1] for r in read):
        miss, stated, kind = torch.cat(read, 1)
        out["read"] = [(name, miss[at].median().item(), stated[at].median().item(),
                        (miss[at] <= stated[at]).float().mean().item() if name != CENTS_KINDS[3] else float("nan"))
                       for name, at in [("cents", kind < 3)] + [(k, kind == i) for i, k in enumerate(CENTS_KINDS)] if at.any()]
    if low is not None:
        right, peaks, heard, given = (float(c) for c in counts)
        precision, recall = right / max(peaks, 1), heard / max(given, 1)
        out["struck"] = 2 * precision * recall / max(precision + recall, 1e-9)
    return out


STRIKE_POS = 20.0  # a strike's weight against the frames without one
DIP = 0.0  # the share of the channel's windows given a dip under the melody, as phones across a room measured 15 dB at 160-320 Hz
STRIKE_AGAIN = 1.0  # and a strike of a pitch still sounding, the most missed of all, by this much more
CENTS_GRAIN = 0.001  # a label's cents are written to a tenth: a miss under it, in semitones, says no more
EVEN_GLIDES = False  # the gliding readings half the cents loss wherever a batch holds them, a few in a hundred of its readings
SPREAD = 0.2  # the stated error's weight in the total: its own row of weights learns as fast as any, the layers under it less from it
PLAYS = 0.25  # what plays a cell against what plays the frame: its heads teach the channels the decoders read, and at the whole weight the strikes are heard worse


def losses(model: ripmodel.RipModel, mel: torch.Tensor, t: dict) -> dict:
    y = model(mel)
    cfg = model.cfg
    valid = t["valid"]
    def bce(name, target, pos_weight):
        logits = y[..., cfg.slice(name)]
        target = target if target.dim() == 3 else target[..., None]
        l = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.as_tensor(pos_weight, device=logits.device), reduction="none")
        return (l.mean(-1) * valid).sum() / valid.sum().clamp(min=1)
    out = {
        "onset": bce("onset", t["onset"], 8.0),
        "drums": bce("drums", t["drums"], 8.0),
        "pitch": bce("pitch", t["pitch"], 12.0),
        "voicing": bce("voicing", t["voicing"], 2.0),
        "strike": bce("strike", t["strike"], STRIKE_POS * (1 + (STRIKE_AGAIN - 1) * F.pad(t["pitch"][:, :-3], (0, 0, 3, 0)))),
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
    if "family" in dict(cfg.outputs):  # the frame model's alone
        asked = (t["family"] >= 0) * valid
        logits = y[..., cfg.slice("family")].reshape(-1, len(ripmodel.FAMILIES))
        ce = F.cross_entropy(logits, t["family"].reshape(-1).clamp(min=0), reduction="none").reshape(asked.shape)
        out["family"] = (ce * asked).sum() / asked.sum().clamp(min=1)
    if "cents" in dict(cfg.outputs):  # the notes model's alone
        place, spread = y[..., cfg.slice("cents")].chunk(2, -1)
        asked = t["asked"] * valid[..., None]
        n = asked.sum().clamp(min=1)
        miss = (place - t["cents"]).abs()
        # the error the head states is a Laplace's scale fitted to the miss as it
        # stands: the miss is held fixed, so stating an error moves no offset
        fit = miss.detach().clamp(min=CENTS_GRAIN) * torch.exp(-spread) + spread
        glide = asked * t["moving"]
        if EVEN_GLIDES and 0 < glide.sum() < asked.sum():  # a glide's readings weigh as much as the rest
            rest = asked - glide
            out["cents"], out["spread"] = ((x * glide).sum() / glide.sum() / 2 + (x * rest).sum() / rest.sum() / 2 for x in (miss, fit))
        else:
            out["cents"], out["spread"] = (miss * asked).sum() / n, (fit * asked).sum() / n
    if "pitchfamily" in dict(cfg.outputs):  # what plays each cell, asked where one note alone sounds in it
        for name in ("pitchfamily", "pitchwave"):
            at = (t[name] >= 0) & (valid[..., None] > 0)
            out[name] = F.cross_entropy(cells(y, cfg, name)[at], t[name][at].long(), reduction="sum") / at.sum().clamp(min=1)
        at = (t["pitchvoice"] >= 0) & (valid[..., None] > 0)
        want = torch.from_numpy(genes()[t["pitchvoice"][at].cpu().numpy()]).to(y.device)
        out["pitchvoice"] = ((torch.sigmoid(cells(y, cfg, "pitchvoice")[at]) - want) ** 2).mean(-1).sum() / at.sum().clamp(min=1)
    # what plays a cell is weighed as what plays the frame is, by PLAYS of it
    out["total"] = (out["onset"] + out["drums"] + out["pitch"] + 0.5 * out["voicing"] + out["strike"] + out["release"]
                    + 2.0 * out["velocity"] + 0.5 * out["wave"] + 2.0 * out["voice"] + 0.5 * out.get("family", 0.0)
                    + 2.0 * out.get("cents", 0.0) + SPREAD * out.get("spread", 0.0)
                    + PLAYS * (0.5 * out.get("pitchfamily", 0.0) + 0.5 * out.get("pitchwave", 0.0) + 2.0 * out.get("pitchvoice", 0.0)))
    return out


def shape(cfg) -> str:
    """a model's sizes and heads for a message; the heads are ripmodel's, never a flag's"""
    return ", ".join(f"{k} {v}" if k != "outputs" else "outputs " + "+".join(h for h, _ in v) for k, v in cfg.items()) if cfg else "an unrecorded shape"


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
    ap.add_argument("--towers", default="", help="stacks of their own after the shared input, layers:heads each in the heads' order, comma-separated: 2:2,4:8 gives onset and drums two layers and the rest four")
    ap.add_argument("--convs", default="", help="convolutions before the input layer, channels x frames x features / pool each, comma-separated: 16x3x3/1,16x3x3/2,32x3x3/2")
    ap.add_argument("--attention-window", type=int, default=64, help="frames attended either side")
    ap.add_argument("--pcen", action="store_true", help="a second input channel, each band's PCEN")
    ap.add_argument("--strike-weight", type=float, default=20.0, help="a strike's weight in its head's loss against a frame without one")
    ap.add_argument("--again-weight", type=float, default=1.0, help="how much more a strike of a pitch still sounding weighs than another")
    ap.add_argument("--bend", type=float, default=0.0, help="the share of windows read along a tempo that sways, so no grid or steady pulse is there to learn")
    ap.add_argument("--ring", type=float, default=0.0, help="the share of windows with a body's ring at every attack, at one pitch whatever the note")
    ap.add_argument("--dilations", default="", help="PitchNet's long convolutions, their frames apart, comma-separated: 3,9,27 hears half a second either way")
    ap.add_argument("--cell", type=int, default=32, help="PitchNet's channels at each cell, its width; the weight file carries it")
    ap.add_argument("--even-glides", action="store_true", help="the cents head's gliding readings weigh as much as the rest in its loss wherever a batch holds both")
    ap.add_argument("--dip", type=float, default=0.0, help="the share of the channel's windows given a dip of 6 to 20 dB centred 100 to 500 Hz, as a phone across a room hears")
    ap.add_argument("--sung", default="", help="a corpus of solo singing and the share of a shuffled window's chunks drawn from it, dir:share")
    ap.add_argument("--shuffle", type=float, default=0.0, help="the share of windows spliced from chunks of a second, each two clips summed")
    ap.add_argument("--extra", default="", help="more corpora and the share of windows drawn from each, dir:share, comma-separated")
    ap.add_argument("--pitchnet", action="store_true", help="the notes model: convolutions only over the harmonic stack, the same weights at every pitch; --out is then ripnotes.bin, which stapes reads beside ripmodel.bin")
    ap.add_argument("--family-holdout", choices=ripmodel.FAMILIES, help="a family no head is told the name of: its notes are trained on as sound and scored as notes, and a notes model's lines print the share their surest name reaches, the measure of where a name stops being sure, which its weight file carries")
    ap.add_argument("--sure", type=float, default=0.0, help="the share a family must pass over a note to name it, as a --family-holdout run of the same recipe measured it, for a notes model's weight file to carry; without it the start's is kept, and with neither no family names a note")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shard-seed", type=int, default=0, help="offsets the stream the windows are drawn from, so a round's workers see different loops; the starting weights stay --seed's")
    ap.add_argument("--checkpoint", default="", help="write a .pt every --every steps, and resume from it when present")
    ap.add_argument("--init", default="", help="start from this .pt's model weights, without its optimizer state or step: a training round's common start; a --checkpoint present wins")
    ap.add_argument("--every", type=int, default=200, help="steps between checkpoints and the held-out and validation scores")
    ap.add_argument("--holdout", type=int, default=-1, help="loops held out of training and scored every --every steps; the step that scores best is what --out gets (default a tenth, none under five loops)")
    ap.add_argument("--validate", default="", help="a corpus of recordings (corpus <dir> from <list>) cut into fixed windows and scored every --every steps; the step that scores best on it is what --out gets, in place of the held-out loops' best: by the decoders' loss, a notes model by its strikes' F1 at the decoder's bars")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    global RING, STRIKE_POS, STRIKE_AGAIN, DIP, FAMILY_HOLDOUT, EVEN_GLIDES
    RING, STRIKE_POS, STRIKE_AGAIN, DIP, EVEN_GLIDES = args.ring, args.strike_weight, args.again_weight, args.dip, args.even_glides
    FAMILY_HOLDOUT = ripmodel.FAMILIES.index(args.family_holdout) if args.family_holdout else None

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed + (args.shard_seed << 32))  # a stream per seed and shard; shard 0 is the seed's own
    convs = [tuple(int(v) for v in c.replace("/", "x").split("x")) for c in args.convs.split(",") if c]
    if any(len(c) != 4 for c in convs):
        raise SystemExit(f"--convs {args.convs!r}: want channels x frames x features / pool, comma-separated")
    towers = [tuple(int(v) for v in t.split(":")) for t in args.towers.split(",") if t]
    if any(len(t) != 2 for t in towers) or towers and sum(h for _, h in towers) != len(ripmodel.HEADS):
        raise SystemExit(f"--towers {args.towers!r}: want layers:heads each, the heads adding up to {len(ripmodel.HEADS)}")
    heard = dict(inputs=ripmodel.STACK_INPUTS, channels=ripmodel.STACK_CHANNELS, outputs=ripmodel.PITCHNET_HEADS) if args.pitchnet else dict(channels=2 if args.pcen else 1)
    cfg = ripmodel.Config(**heard, convs=convs, dim=args.dim, layers=args.layers, heads=args.heads, ffn=args.ffn, window=args.attention_window, towers=towers,
                          dilations=tuple(int(d) for d in filter(None, args.dilations.split(","))), cell=args.cell)
    planes = min(cfg.channels, 2)  # what the data loader makes; the model stacks them
    resume = bool(args.checkpoint) and os.path.exists(args.checkpoint)
    start = args.checkpoint if resume else args.init
    ck = torch.load(start, map_location=args.device) if start else None
    if ck is not None and "cfg" in ck:
        ck["cfg"] = {**asdict(ripmodel.Config()), **ck["cfg"]}  # a setting newer than the checkpoint takes the default it was built with
    # a start from a model of the heads before the last added, such as the family:
    # its weights carry over, the new heads' rows of the output layer start fresh
    had = [tuple(h) for h in ck["cfg"]["outputs"]] if ck is not None and not resume and "cfg" in ck else []
    grown = 0 < len(had) < len(cfg.outputs) and had == list(cfg.outputs)[:len(had)] and \
        {k: v for k, v in ck["cfg"].items() if k != "outputs"} == {k: v for k, v in asdict(cfg).items() if k != "outputs"}
    # refused before the corpus loads: load_state_dict names only a tensor's size,
    # and takes another head count or window without a word
    if ck is not None and ck.get("cfg") != asdict(cfg) and not grown:
        raise SystemExit(f"{start} holds a model of {shape(ck.get('cfg'))}, and this run builds {shape(asdict(cfg))}: pass its shape, or another file")
    manifest, loops = load_corpus(args.corpus)
    holdout = args.holdout if args.holdout >= 0 else (len(loops) // 10 if len(loops) >= 5 else 0)
    held = loops[len(loops) - holdout:] if holdout else []
    loops = loops[:len(loops) - holdout] if holdout else loops
    val_mels, val_tgs, val_notes = validation(held, args.window, planes)
    assert manifest["mel_bands"] == ripmodel.MEL_BANDS and manifest["pitches"] == ripmodel.PITCHES
    assert manifest["voice_genes"] == ripmodel.VOICE_GENES and len(manifest["waves"]) == ripmodel.WAVES
    recorded, check_mels, check_tgs, check_notes = [], [], [], None
    if args.validate:
        check_manifest, recorded = load_corpus(args.validate)
        assert check_manifest["mel_bands"] == ripmodel.MEL_BANDS and check_manifest["pitches"] == ripmodel.PITCHES
        if not recorded:
            raise SystemExit(f"--validate {args.validate}: no recordings to score")
        check_mels, check_tgs, check_notes = validation(recorded, args.window, planes)
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
    validated = (float("inf"), 0, None, None)  # the best validation loss, its step, its weights and the held-out loss there
    if resume:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        best_loss, best_step, best_state = ck.get("best", (best_loss, best_step, best_state))
        validated = ck.get("validated", validated)
        print(f"resumed at step {step}")
    elif grown:
        state = model.state_dict()
        for k, v in ck["model"].items():
            if state[k].shape == v.shape:
                state[k] = v
            elif v.dim() and state[k].shape[1:] == v.shape[1:] and len(state[k]) > len(v):
                state[k][:len(v)] = v  # the output layer: the rows of the heads it had
            else:
                raise SystemExit(f"{start}: {k} is {tuple(v.shape)}, this run's {tuple(state[k].shape)}")
        model.load_state_dict(state)
        print(f"started from {args.init}, its heads but {', '.join(h for h, _ in cfg.outputs[len(had):])}, which start fresh")
    elif ck is not None:
        model.load_state_dict(ck["model"])
        print(f"started from {args.init}")
    if args.sure and hasattr(model, "sure"):
        model.sure.fill_(args.sure)
    print(f"{sum(p.numel() for p in model.parameters())} weights, {len(loops)} loops, {sum(len(m) for m, _ in loops)} frames, {len(held)} held out in {len(val_mels)} windows")
    if recorded:
        print(f"{len(recorded)} recordings of {sum(len(m) for m, _ in recorded)} frames to validate on in {len(check_mels)} windows, from {args.validate}")
    if mixed:
        print(f"{len(mixed)} loops of {sum(len(m) for m, _ in mixed)} frames mixed in, {share:.0%} of the windows")
    if drums:
        print(f"{len(drums)} recordings of {sum(len(m) for m, _ in drums)} frames laid over {len(drumless)} loops with no drums, {over_share:.0%} of the windows")
    t0 = time.time()

    low = STRIKE_LOW - manifest["min_midi"] if args.pitchnet else None

    def scored(name: str, mels, tgs, notes):
        """the decoders' loss over fixed windows, a notes model's strike F1, and the line that reports them;
        a family held out sets the model's measure by its notes there"""
        v = evaluate(model, mels, tgs, args.device, low, notes)
        # the decoders read the onset, drum, pitch, voicing, strike and release
        # heads; the wave, voice, velocity and family heads say how a note is
        # played and by what, and memorize the corpus's, so they do not pick the step
        v["decoders"] = v["onset"] + v["drums"] + v["pitch"] + 0.5 * v["voicing"] + v["strike"] + v["release"]
        parts = " ".join(f"{k} {x:.3f}" for k, x in v.items() if k not in ("total", "decoders", "struck", "read", "voices"))
        f1 = f", strikes' F1 {v['struck']:.3f}" if "struck" in v else ""
        # the cents head: each kind of reading's median miss, the median error stated for it, and the share within the stated
        read = "".join(f"{'; ' if i else ', '}{k} " + (f"{stated:.1f} stated" if miss != miss else f"{miss:.1f} off, {stated:.1f} stated, {within:.0%} within")
                       for i, (k, miss, stated, within) in enumerate(v.get("read", [])))
        voices = ""
        if "voices" in v:  # what plays each note: the share named right, by family and under another note, their waves and genes
            n = v["voices"]
            by = ", ".join(f"{k} {x:.3f} of {count}" for k, x, count in n["by"])
            voices = f"; families {n['families'][0]:.3f} of {n['families'][1]} notes ({by}), {n['under'][0]:.3f} of the {n['under'][1]} under another, waves {n['waves']:.3f}, genes {n['genes']:.3f} off"
            if "held" in n:
                count, reach, median, past, right = n["held"]
                model.sure.fill_(reach)
                voices += f"; {args.family_holdout}'s {count} notes, held out, reach {reach:.3f}, {median:.3f} at the median: {past:.0%} of the others' notes pass it, {right:.0%} of them right"
        return v["decoders"], v.get("struck"), f"step {step} {name} {v['total']:.3f}, decoders {v['decoders']:.3f}{f1}{read}{voices} ({parts})"

    def weights() -> dict:
        return {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}

    def score() -> None:
        nonlocal best_loss, best_step, best_state, validated
        held_loss = None
        if val_mels:
            held_loss, _, line = scored("held-out", val_mels, val_tgs, val_notes)
            if held_loss < best_loss:
                best_loss, best_step, best_state = held_loss, step, weights()
                line += " (best)"
            print(line, flush=True)
        if check_mels:
            # a notes model's loss on recordings is mostly its onset head, which no decoder
            # reads from it, and ranks rounds unlike their note scores; its strikes' F1 does
            loss, f1, line = scored("validation", check_mels, check_tgs, check_notes)
            key = loss if f1 is None else -f1
            if key < validated[0]:
                validated = (key, step, weights(), held_loss)
                line += " (best)"
            print(line, flush=True)

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
    ahead = pool.submit(batch, draw, args.window, args.batch, rng, planes, channel, channel_share, pick, args.shuffle, args.bend)
    while step < args.steps:
        mel, t = ahead.result()
        ahead = pool.submit(batch, draw, args.window, args.batch, rng, planes, channel, channel_share, pick, args.shuffle, args.bend)
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
                # the window showing in no tensor; the best held-out and validation steps,
                # so a resumed run, or one run again once done, still exports its pick
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": asdict(cfg),
                            "best": (best_loss, best_step, best_state), "validated": validated}, args.checkpoint)
    if args.validate and validated[2] is not None:
        model.load_state_dict(validated[2])
        beside = f", held-out {validated[3]:.3f} there" if validated[3] is not None else ""
        what = f"strikes' F1 {-validated[0]:.3f}" if args.pitchnet else f"decoders' loss {validated[0]:.3f}"
        print(f"best validation {what} at step {validated[1]}{beside}; exporting that")
    elif best_state is not None:
        model.load_state_dict(best_state)
        print(f"best held-out decoders' loss {best_loss:.3f} at step {best_step}; exporting that")
    if args.sure and hasattr(model, "sure"):
        model.sure.fill_(args.sure)  # over the step exported, whatever its run measured
    model.eval()
    ripmodel.export(model.cpu(), args.out)
    sure = ""
    if hasattr(model, "sure"):
        sure = f", a family naming a note past a share of {model.sure.item():.3f}" if model.sure > 0 else ", no family naming a note: no run measured where a name stops being sure (--family-holdout, --sure)"
    print(f"wrote {args.out}{sure}")


if __name__ == "__main__":
    main()
