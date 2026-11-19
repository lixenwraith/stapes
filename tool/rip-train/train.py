"""Train the rip's front end on a corpus stapes exported (`corpus <dir>`) and export the weights
lixenwraith/transformer loads: python3 train.py --corpus corpus/ --out ripframe.bin --steps 2000,
and with --pitchnet the notes model's ripnotes.bin. A loop's band frames (loop_NNN.bands) are the
input and its label (loop_NNN.json) the per-frame targets doc/rip-training.md §5 lists, built for
random windows as they are drawn, so memory holds the events rather than a row per frame."""
from __future__ import annotations

import argparse
import array
import copy
import gc
import json
import math
import os
import random
import struct
import time
from dataclasses import asdict
from itertools import accumulate

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

import ripmodels

# frames read before a window for PCEN's average to settle, over three of its time constants
WARM = 128


class FrameFile:
    """A loop's band frames, read from its file as sliced: a corpus's frames are
    gigabytes, which the page cache holds rather than the process."""

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            self.frames, self.values = struct.unpack("<II", f.read(8))

    def __len__(self) -> int:
        return self.frames

    def __getitem__(self, sl: slice) -> np.ndarray:
        start, stop, _ = sl.indices(self.frames)
        n = max(stop - start, 0)
        return np.fromfile(self.path, dtype="<f4", count=n * self.values, offset=8 + 4 * start * self.values).reshape(n, self.values)


GLIDE_SPAN = 50.0  # the cents a recorded note's pitch track spans before its readings count as gliding: half a cell

# the genes of every voice the labels' notes are played by, a loop's voices in turn: a cell's target is a voice's row here
GENES = array.array("f")


def genes() -> np.ndarray:
    """GENES as a row a voice"""
    return np.frombuffer(GENES, np.float32).reshape(-1, ripmodels.VOICE_GENES)


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
        self.vel = np.array([e.get("vel", -1) for e in ev], np.float32)  # -1 where the recording's list says its velocities say nothing
        self.placed = bool(label.get("placed"))  # its notes' starts placed on the recording by an alignment, not heard struck
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
        # the name a note's cells are asked: an instrument's, a recording's or a sampled note's family as its set gives it;
        # a note the engine rendered has a wave and genes and no instrument, and is asked none (its wave and genes it is)
        unknown = ripmodels.FAMILIES.index("unknown")
        self.name = np.where((self.voice >= 0) | (self.family == unknown), -1, self.family).astype(np.int8)
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
        # A loop's beats and bar lines are those its label lists. A recording's are asked from the first one's mark to the
        # last one's: past them its list says nothing. A rendered loop's, where its tempo and its start put them, are asked
        # over all its frames but the last two, which a beat at its end, past the list, would mark. Notes overlapping in no
        # meter list none and are asked none
        self.beats, self.lines = (np.unique(np.asarray(label.get(k, []), np.int64)) for k in ("beats", "downbeats"))
        self.metered = (int(self.beats[0]) - 1, int(self.beats[-1]) + 2) if len(self.beats) else (0, 0)
        if label.get("bars") and len(self.beats):
            self.metered = (0, frames - 2)
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
        """Per-frame targets for frames a to b, as doc/rip-training.md §5 lists them; masks say
        where a target counts. The yes-or-no ones are bytes until batch; the strikes and
        releases are lists of (frame, pitch) from a, each strike's velocity beside it, -1
        where the corpus does not say. A beat and a bar line mark a frame either side."""
        m, frames, w = self.manifest, self.frames, b - a
        pitches, min_midi = ripmodels.PITCHES, ripmodels.LOW
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

        beat, downbeat, metered = np.zeros(w, np.uint8), np.zeros(w, np.uint8), np.zeros(w, np.uint8)
        metered[span(*self.metered)] = 1
        for marks, at in ((beat, self.beats), (downbeat, self.lines)):
            for t in at[np.searchsorted(at, a - 1):np.searchsorted(at, b, "right")].tolist():
                marks[span(max(t - 1, 0), min(t + 2, frames))] = 1

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
            r0, r1 = x - a, max(min(end, b), x) - a  # the note's frames in the window
            first = last = p
            placed = self.voice[i] >= 0  # an engine voice plays where the label says; a recording only where it lists its cents
            if at < 0:  # it holds its pitch: one cell and one offset over its frames, a column of each target
                held, off = slice(r0, r1), np.float32(p + self.tuning / 100)
                if 0 <= p < pitches:
                    pitch[held, p], sole[held, p] = 1, np.where(sole[held, p] == -1, i, -2)
                    if not placed:
                        near[held, p], asked[held, p] = 0, 0  # where it sits in its cell is not known
                for cell in range(int(np.rint(off)) - 1, int(np.rint(off)) + 2):
                    o = float(off) - cell
                    if placed and abs(o) < 1 and 0 <= cell < pitches:
                        nearer = np.float64(abs(o)) < near[held, cell]  # the nearest sound is the one a cell reports; a bare float compares in float32
                        near[held, cell][nearer], cents[held, cell][nearer], asked[held, cell][nearer], moving[held, cell][nearer] = abs(o), o, 1, 0
            else:
                rows = np.arange(r0, r1)
                c, glides = self.cents[at + x - t:at + x - t + len(rows)], self.moves[at + x - t:at + x - t + len(rows)]
                placed = ~np.isnan(c)  # a frame that is not placed sounds in the note's own cell
                off = p + np.nan_to_num(c) / 100
                cell = np.rint(off).astype(np.int64)
                first, last = (int(np.rint(p + np.nan_to_num(self.cents[at + k]) / 100)) for k in (0, end - t - 1))
                ok = (cell >= 0) & (cell < pitches)
                r, c = rows[ok], cell[ok]
                pitch[r, c], sole[r, c] = 1, np.where(sole[r, c] == -1, i, -2)
                if 0 <= p < pitches:
                    near[rows[~placed], p], asked[rows[~placed], p] = 0, 0
                # the cells under, at and over each frame's offset in one pass: no two of a frame's are one cell
                rows, cell = np.tile(rows, 3), np.concatenate([cell - 1, cell, cell + 1])
                o = np.tile(off, 3) - cell
                ok = np.tile(placed, 3) & (np.abs(o) < 1) & (cell >= 0) & (cell < pitches)
                ok[ok] = np.abs(o[ok]) < near[rows[ok], cell[ok]]
                r, c = rows[ok], cell[ok]
                near[r, c], cents[r, c], asked[r, c], moving[r, c] = np.abs(o[ok]), o[ok], 1, np.tile(glides, 3)[ok]
            if 0 <= first < pitches:
                strikes.append((t - a, first))
                vels.append(self.vel[i])
            if 0 <= last < pitches and self.end[i] < frames:
                releases.append((int(self.end[i]) - a, last))
            if self.wave[i] >= 0:
                up = midi > top[seg]
                top[seg][up], who[seg][up] = midi, i
        heard = who >= 0
        wave[heard], voice[heard] = self.wave[who[heard]], genes()[self.voice[who[heard]]]
        family = np.full(w, -1, np.int64)  # a frame's, by the rule its highest note's cells are asked by: no engine voice's
        family[whose >= 0] = self.name[whose[whose >= 0]]
        one = sole >= 0

        def of(a):  # an event's value at the cells it alone sounds in, -1 at the others
            out = np.full(sole.shape, -1, a.dtype)
            out[one] = a[sole[one]]
            return out
        name = of(self.name)
        if FAMILY_HOLDOUT is not None:  # the family held out is named to no head
            family[family == FAMILY_HOLDOUT], name[name == FAMILY_HOLDOUT] = -1, -1
        return {"onset": onset, "drums": drums, "pitch": pitch, "voicing": (pitch.sum(1) > 0).astype(np.uint8),
                "wave": wave, "voice": voice, "voiced": (top >= 0).astype(np.uint8), "family": family, "beat": beat, "downbeat": downbeat, "metered": metered,
                "strikes": np.array(strikes, np.int32).reshape(-1, 2), "vels": np.array(vels, np.float32),
                "releases": np.array(releases, np.int32).reshape(-1, 2), "cents": cents, "asked": asked, "moving": moving,
                "note": np.where(one, sole, -1), "pitchfamily": name, "pitchwave": of(self.wave), "pitchvoice": of(self.voice)}


# a family's index in ripmodels.FAMILIES whose notes the run names to no head, so a calibration run never hears its name
FAMILY_HOLDOUT = None
FAMILY_UNASKED = False  # a frame model's family head grown and asked nothing: no term in the loss, its rows zero


def load_corpus(d: str):
    """a corpus's manifest and its loops, each its frames and its label; a corpus of frames of another layout, or of
    none, is refused by name"""
    with open(os.path.join(d, "corpus.json")) as f:
        manifest = json.load(f)
    if manifest.get("layout") != ripmodels.LAYOUT:
        raise SystemExit(ripmodels.refused(f"{d}: its manifest", manifest.get("layout")) + ": export it again")
    if manifest.get("families", list(ripmodels.FAMILIES)) != list(ripmodels.FAMILIES):
        raise SystemExit(f"{d}: families {manifest['families']}, the model's {list(ripmodels.FAMILIES)}")
    loops, unnamed = [], 0
    skip = set(open(os.path.join(d, "skip.txt")).read().split()) if os.path.exists(os.path.join(d, "skip.txt")) else set()
    for i in range(manifest["loops"]):
        if f"loop_{i:03d}" in skip:
            continue
        base = os.path.join(d, f"loop_{i:03d}")
        fr = FrameFile(base + ".bands")
        if fr.values != ripmodels.VALUES:
            raise SystemExit(f"{fr.path}: {fr.values} values a frame, layout {ripmodels.LAYOUT}'s {ripmodels.VALUES}")
        with open(base + ".json") as f:
            label = json.load(f)
        lab = Label(label, len(fr), manifest)
        unnamed += int((lab.name == FAMILY_HOLDOUT).sum()) if FAMILY_HOLDOUT is not None else 0
        loops.append((fr, lab))
    if unnamed:
        print(f"{unnamed} notes of {d} are {ripmodels.FAMILIES[FAMILY_HOLDOUT]}'s, held out: no head is told their family")
    return manifest, loops


class Channel:
    """What a small speaker, a room and a phone do to frames, drawn per window: the
    bands under a low corner and over a high one cut, a ripple across them, and a
    steady noise floor added, in power. A corpus's noise sits 40 to 65 dB under the
    loop, a phone's far nearer the music."""

    def __init__(self, path: str):
        d = np.load(path)
        self.white, self.room = d["white"], d["room"].astype(np.float64)
        if self.white.shape[1:] != (ripmodels.VALUES,) or self.room.shape != (ripmodels.VALUES,):
            raise SystemExit(f"{path}: a bank of {self.white.shape[1:]} values a frame, layout {ripmodels.LAYOUT}'s {ripmodels.VALUES}: make it again (make_noise.py)")
        # each band of both planes, its ripple and floor drawn over the mel scale
        self.hz = np.tile(ripmodels.band_hz(np.arange(ripmodels.BANDS)), 2)
        self.points = 2595 * np.log10(1 + self.hz / 700)

    def draw(self, n: int, rng: random.Random):
        """the gain per band and the noise over n frames, in the energy a stored frame holds"""
        octave = np.log2(self.hz)
        low, high = 60 * (400 / 60) ** rng.random(), 3000 * (10000 / 3000) ** rng.random()
        cut = -rng.random() * 24 * np.clip((np.log2(low) - octave) / 1.5, 0, 1) - rng.random() * 24 * np.clip(octave - np.log2(high), 0, 1)
        knots = np.linspace(self.points.min(), self.points.max(), 9)
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


# A real hit counts when its new power tops all else sounding by this, in dB, over its best ten bands of the sharp plane
HEARD = 0.0


def stands_out(d: np.ndarray, rest: np.ndarray, i: int) -> bool:
    """whether the hit at frame i of the drums d stands out of the rest sounding: what
    it adds counts, not the ring of the hits before it, which is why rest takes that ring.
    The sharp plane's bands judge it, which hear a hit when it comes: the fine plane's
    low bands spread it over their windows, seconds long"""
    d, rest = d[:, ripmodels.BANDS:], rest[:, ripmodels.BANDS:]
    ring = d[max(i - 4, 0):i - 1].mean(0) if i >= 2 else 0.0
    ratio = np.maximum(d[i:i + 3].max(0) - ring, 1e-12) / (rest[i:i + 3].mean(0) + ring + 1e-12)
    return np.sort(10 * np.log10(ratio))[-10:].mean() >= HEARD


_power = {}


def power(fr) -> float:
    """a recording's mean energy per band and frame, read once"""
    if fr.path not in _power:
        _power[fr.path] = float(np.expm1(fr[:len(fr)].astype(np.float64)).mean())
    return _power[fr.path]


def targets(fr, lab: Label, start: int, window: int, with_drums: bool = True) -> dict:
    """the targets of the window of `window` frames from start, the frames past the loop's end masked out"""
    b = min(start + window, len(fr))
    t, pad = lab.window(start, b, with_drums), window - (b - start)

    def take(a, fill=0):
        return np.concatenate([a, np.full((pad,) + a.shape[1:], fill, a.dtype)]) if pad else a
    out = {k: take(t[k]) for k in ("onset", "drums", "pitch", "voicing", "voice", "voiced", "cents", "asked", "moving", "beat", "downbeat", "metered")}
    out.update({k: take(t[k], -1) for k in ("wave", "family", "note", "pitchfamily", "pitchwave", "pitchvoice")})
    shape = (window, t["pitch"].shape[1])
    strike, release, velocity, heard = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8), np.zeros(shape, np.float32), np.zeros(shape, np.uint8)
    for (f, p), v in zip(t["strikes"], t["vels"]):
        near = slice(max(f - 1, 0), max(min(f + 2, window), 0))
        strike[near, p] = not lab.placed  # a placed start misses its strike by frames: the head is asked none of it
        if v >= 0:
            velocity[near, p], heard[near, p] = v, 1
    for f, p in t["releases"]:
        release[max(f - 1, 0):max(min(f + 2, window), 0), p] = 1
    out.update(strike=strike, release=release, velocity=velocity, heard=heard, placed=take(np.full(b - start, lab.placed, np.uint8)))
    out["valid"] = np.concatenate([np.ones(window - pad, np.float32), np.zeros(pad, np.float32)])
    return out


# Shuffled windows: chunks of a second, each two clips summed, spliced into a window
CHUNK = 86


def clip(fr, rng):
    """a chunk's energy from a random place in a recording, zeros past its end, and where it starts"""
    start = rng.randrange(max(len(fr) - CHUNK, 0) + 1)
    e = np.expm1(fr[start:start + CHUNK].astype(np.float64))
    return (np.concatenate([e, np.zeros((CHUNK - len(e), e.shape[1]))]) if len(e) < CHUNK else e), start


def shuffled(pick, window: int, planes: int, rng, channel=None):
    """a window of summed chunks: notes keep their labels, a drum hit counts where it
    stands out of the other clip, what the frame's voice and family are and where its
    beats fall go unasked, and a cell says what plays it where one clip alone sounds in it"""
    total = window + WARM if planes == 4 else window
    through, noise = channel.draw(total, rng) if channel is not None else (1.0, np.zeros((total, ripmodels.VALUES)))
    es, outs = [], []
    for at in range(0, total, CHUNK):
        (ma, la), (mb, lb) = pick(rng), pick(rng)
        ea, sa = clip(ma, rng)
        eb, sb = clip(mb, rng)
        eb = eb * 10 ** ((rng.random() * 12 - 6) / 10) * power(ma) / max(power(mb), 1e-9)
        ta, tb = targets(ma, la, sa, CHUNK, False), targets(mb, lb, sb, CHUNK, False)
        out = {k: np.maximum(ta[k], tb[k]) for k in ("onset", "drums", "pitch", "voicing", "strike", "release", "velocity", "heard", "valid", "placed")}
        out.update(wave=np.full(CHUNK, -1, np.int64), family=np.full(CHUNK, -1, np.int64), voice=np.zeros_like(ta["voice"]), voiced=np.zeros_like(ta["voiced"]))
        out.update({k: np.zeros(CHUNK, np.uint8) for k in ("beat", "downbeat", "metered")})  # a splice of two clips has neither's grid
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
    return ripmodels.features(np.log1p(e).astype(np.float32), planes)[first:], out


def sample(fr, lab: Label, start: int, window: int, planes: int, over=None, rng=None, channel=None):
    """the window of `window` frames from start: its features and targets, the
    frames past the loop's end padded with zeros and masked out; over is a recording
    of drums alone and its power against the loop's in dB, laid over the frames, its
    hits that can be heard among the targets and its window asked no beat, the drums
    keeping another grid; channel is what a speaker and a phone do to them"""
    out = targets(fr, lab, start, window)
    b = min(start + window, len(fr))
    pad = window - (b - start)

    def take(a, fill=0):
        return np.concatenate([a, np.full((pad,) + a.shape[1:], fill, a.dtype)]) if pad else a
    a = max(start - WARM, 0) if planes == 4 else start
    # uncorrelated sounds add in power, and a stored frame is log1p of a hundred times it
    e = np.expm1(take(fr[a:b]).astype(np.float64))
    if rng is not None and rng.random() < RING:
        ringing(e, rng)
    loud = e[:len(e) - pad].mean() if pad < len(e) else 0.0
    through = 1.0
    if channel is not None:
        through, noise = channel.draw(len(e), rng)
        e = e * through + noise
    if over is not None:
        ofr, olab, level = over
        out["metered"][:] = 0
        first, n = start - a, min(len(ofr), len(e))
        at, ofrom = rng.randrange(len(e) - n + 1), rng.randrange(len(ofr) - n + 1)
        d = np.expm1(ofr[ofrom:ofrom + n].astype(np.float64)) * through * 10 ** (level / 10) * loud / max(power(ofr), 1e-9)
        rest = e[at:at + n].copy()
        e[at:at + n] += d
        for f, drum in olab.hits(ofrom, ofrom + n):
            i = f - ofrom
            w = i + at - first
            if -2 < w < window + 1 and stands_out(d, rest, i):
                near = slice(max(w - 1, 0), max(min(w + 2, window), 0))
                out["onset"][near] = 1
                out["drums"][near, drum] = 1
    return ripmodels.features(np.log1p(e).astype(np.float32), planes)[start - a:], out


# A body's ring: what a guitar's air cavity or a room's mode adds at every attack, at one pitch whatever the note, its own harmonics absent. No note is labelled
RING = 0.0  # the share of windows that have one
RING_HZ, RING_TAU, RING_DB = (70.0, 160.0), (0.05, 0.3), (-25.0, -8.0)  # its pitch, decay in seconds, and level against the attack's rise in energy


def ringing(e, rng):
    """adds a ring to frames of energy, in place: in each band as a steady tone of its pitch reads there, against the
    attack's rise in the sharp plane, which hears it when it comes, each band's energy over the times the plane
    counts a tone there, so a low note's counts once"""
    hz = math.exp(rng.uniform(math.log(RING_HZ[0]), math.log(RING_HZ[1])))
    total = (e[:, ripmodels.BANDS:] / ripmodels.sharp_counts()).sum(1)
    kick = np.maximum(np.diff(total, prepend=total[0]), 0)
    keep, y = math.exp(-(256 / 22050) / rng.uniform(*RING_TAU)), 0.0
    out = np.empty(len(e))
    for t, v in enumerate(kick):
        y = v + keep * y
        out[t] = y
    out *= 10 ** (rng.uniform(*RING_DB) / 10)
    e += out[:, None] * ripmodels.sine_response(hz)


# Bent windows: a tempo read along a slow curve, so no grid or steady pulse is there to learn
BEND_RATE, BEND_SWAY = 0.18, 12  # the window's tempo up to this share either way, swaying this many frames


def bent(fr, lab, window: int, planes: int, over, rng, heard):
    """a window read along a curve over the frames of a longer or shorter one: the features interpolated, the targets nearest"""
    rate = math.exp(rng.uniform(-BEND_RATE, BEND_RATE))
    sway, period, phase = rng.uniform(0, BEND_SWAY), rng.uniform(150, 500), rng.uniform(0, 2 * math.pi)
    n = int(window * rate + 2 * BEND_SWAY) + 2
    x, out = sample(fr, lab, rng.randrange(max(len(fr) - n, 0) + 1), n, planes, over, rng, heard)
    t = np.arange(window)
    g = np.clip(t * rate + BEND_SWAY + sway * np.sin(2 * math.pi * t / period + phase), 0, n - 1.001)
    lo = g.astype(np.int64)
    f = (g - lo)[:, None].astype(np.float32)
    near = np.rint(g).astype(np.int64)
    return x[lo] * (1 - f) + x[lo + 1] * f, {k: v[near] for k, v in out.items()}


def stack(samples):
    frs, tgs = zip(*samples)
    out = {k: torch.from_numpy(np.stack([t[k] for t in tgs])) for k in tgs[0]}
    return torch.from_numpy(np.stack(frs)), {k: v if v.dtype in (torch.int8, torch.int32, torch.int64) else v.float() for k, v in out.items()}


def batch(draw, window: int, size: int, rng: random.Random, planes: int, channel=None, share: float = 0.0, pick=None, shuffle: float = 0.0, bend: float = 0.0):
    """size random windows of `window` frames from the loops draw(rng) picks, a share through the channel, a share shuffled, a share of the rest bent"""
    samples = []
    for _ in range(size):
        heard = channel if channel is not None and rng.random() < share else None
        if pick is not None and rng.random() < shuffle:
            samples.append(shuffled(pick, window, planes, rng, heard))
            continue
        fr, lab, over = draw(rng)
        if rng.random() < bend:
            samples.append(bent(fr, lab, window, planes, over, rng, heard))
            continue
        samples.append(sample(fr, lab, rng.randrange(max(len(fr) - window, 0) + 1), window, planes, over, rng, heard))
    return stack(samples)


def stream(seed: int, shard: int, k: int) -> random.Random:
    """what batch k of a run draws from: a stream of its seed, its shard and k alone, each under 2**32 but k"""
    return random.Random(seed + (shard << 32) + (k << 64))


class Batches:
    """a run's batches by index, as a builder forked from the run makes them: build(rng) over batch k's stream"""

    def __init__(self, build, seed: int, shard: int):
        self.build, self.seed, self.shard = build, seed, shard

    def __getitem__(self, k: int):
        return self.build(stream(self.seed, self.shard, k))


def validation(loops, window: int, planes: int):
    """every held-out loop cut into windows of `window` frames, fixed, so the held-out loss is comparable step to step,
    each window naming its loop; and their notes, a column a note in the windows' `note`: its name, its wave and its
    voice's row, -1 where its label says none"""
    frs, tgs, notes, first = [], [], [np.zeros((3, 0), np.int64)], 0
    for i, (fr, lab) in enumerate(loops):
        for start in range(0, max(len(fr) - 1, 1), window):
            m, tg = stack([sample(fr, lab, start, window, planes)])
            tg["note"] = torch.where(tg["note"] >= 0, tg["note"] + first, tg["note"])
            tg["loop"] = torch.tensor([i])  # a reading's noise counts a loop's windows as one draw (apart)
            frs.append(m)
            tgs.append(tg)
        notes.append(np.stack([lab.name, lab.wave, lab.voice]).astype(np.int64))
        first += len(lab.t)
    return frs, tgs, torch.from_numpy(np.concatenate(notes, 1))


# A notes model's strikes as the decoder reads them: a peak over the bar a strike starts a note at, the higher bar
# under E3, right when a labelled strike of its pitch lies within STRIKE_REACH frames
STRIKE_BARS, STRIKE_LOW, STRIKE_REACH = (0.7, 0.9), 52, 4
BEAT_BAR = 0.5  # a frame model's beat and downbeat are posteriors: a peak is one the head calls likelier than not


def struck(logits: torch.Tensor, strike: torch.Tensor, valid: torch.Tensor, bar) -> torch.Tensor:
    """a window's peaks right, peaks, labelled strikes heard and labelled strikes, from [frames, pitches]
    each, a peak one at bar or over, a pitch's own or one for all: a beat's and a bar line's are counted so too"""
    p = torch.sigmoid(logits)
    peak = (p >= bar) & (p >= F.max_pool1d(p.T[None], 5, 1, 2)[0].T) & (valid[:, None] > 0)
    first = (strike > 0) & (F.pad(strike[:-1], (0, 0, 1, 0)) == 0)  # a mark's first frame; the strike is the next
    onset = F.pad(first[:-1], (0, 0, 1, 0))
    near = lambda x: F.max_pool1d(x.float().T[None], 2 * STRIKE_REACH + 1, 1, STRIKE_REACH)[0].T > 0
    return torch.stack([(peak & near(onset)).sum(), peak.sum(), (onset & near(peak)).sum(), onset.sum()])


def fscore(counts) -> float:
    """the F1 of struck's four counts"""
    right, peaks, heard, given = (float(c) for c in counts)
    precision, recall = right / max(peaks, 1), heard / max(given, 1)
    return 2 * precision * recall / max(precision + recall, 1e-9)


# the readings a cents head is scored over: of a cell its sound is on a glide in, of one it holds its pitch in, and of the
# cell beside a sound; and of a sounding cell the label does not place, where only the error stated can be told
CENTS_KINDS = ("gliding", "steady", "beside", "unplaced")


# A note is played by what its cells say, each by its salience, over the frames a cell is among the
# VOICE_CELLS most salient: those lixenwraith/audio keeps the say of (RipVoiceCells)
VOICE_CELLS = 12
SURE = 0.95  # the share of a held-out family's notes whose surest name stays under the measure


def cells(y: torch.Tensor, cfg, name: str) -> torch.Tensor:
    """a per-pitch head's scores, [..., pitches, values]"""
    return y[..., cfg.slice(name)].unflatten(-1, (-1, cfg.pitches)).transpose(-1, -2)


def said(y: torch.Tensor, cfg):
    """what a notes model says plays each cell, [..., pitches, each family's share, each wave's, the genes],
    and the weight a cell's say has in its note's"""
    v = torch.cat([cells(y, cfg, "pitchfamily").softmax(-1), cells(y, cfg, "pitchwave").softmax(-1), torch.sigmoid(cells(y, cfg, "pitchvoice"))], -1)
    p = y[..., cfg.slice("pitch")]
    return v, torch.sigmoid(p) * (p >= p.topk(VOICE_CELLS, -1).values[..., -1:])


def noted(heard: torch.Tensor, notes: torch.Tensor, prior: torch.Tensor | None = None) -> dict:
    """the notes of fixed windows as they were heard against their labels: the share named their family, over
    all, by family and over those under another note; the share given their wave; their genes' RMS miss; of
    a family held out, the share its notes' surest name reaches, SURE of them under it and at the median, with
    the share of the other notes past it and of those the share named right; and under prior, each family's share
    of the cells asked a name, the share named right with each name's say over its share there, over all and by family"""
    name, wave, voice = notes
    names, waves = ripmodels.NAMES, ripmodels.WAVES
    w = heard[:, -2]
    share, to = heard[:, :names] / w.clamp(min=1e-9)[:, None], heard[:, names:names + waves].argmax(1)
    top, named = share.max(1)
    asked, right = (w > 0) & (name >= 0) & (name != (-1 if FAMILY_HOLDOUT is None else FAMILY_HOLDOUT)), named == name
    rate = lambda at: (right[at].float().mean().item(), int(at.sum()))
    known = (w > 0) & (voice >= 0)
    miss = (heard[known, names + waves:-2] / w[known, None] - torch.from_numpy(genes()[voice[known].numpy()])) ** 2
    out = {"families": rate(asked), "by": [(ripmodels.FAMILIES[k], *rate(asked & (name == k))) for k in range(names) if (asked & (name == k)).any()],
           "under": rate(asked & (heard[:, -1] > 0.5 * w)), "waves": (to == wave)[(w > 0) & (wave >= 0)].float().mean().item(),
           "genes": miss.mean(1).sqrt().mean().item()}
    if prior is not None:  # a name the prior never holds is given to no note
        even = torch.where(prior > 0, share / prior.float().clamp(min=1e-12), share.new_zeros(())).argmax(1) == name
        got = lambda at: even[at].float().mean().item()
        out["even"] = (got(asked), [(ripmodels.FAMILIES[k], got(asked & (name == k))) for k in range(names) if (asked & (name == k)).any()])
    held = (w > 0) & (name == FAMILY_HOLDOUT) if FAMILY_HOLDOUT is not None else torch.zeros(0, dtype=torch.bool)
    if held.any():
        reach = top[held].quantile(SURE).item()
        out["held"] = (int(held.sum()), reach, top[held].median().item(), (top[asked] > reach).float().mean().item(), rate(asked & (top > reach))[0])
    return out


def shares(tgs) -> torch.Tensor | None:
    """the share each family holds of the cells fixed windows ask a name of, the names' prior; None where none is asked"""
    n = torch.zeros(ripmodels.NAMES, dtype=torch.float64)
    for t in tgs:
        n += torch.bincount(t["pitchfamily"][(t["pitchfamily"] >= 0) & (t["valid"][..., None] > 0)].long(), minlength=ripmodels.NAMES)
    return n / n.sum() if n.sum() else None


def evaluate(model: ripmodels.FrameModel, frs, tgs, device: str, low: int | None = None, notes: torch.Tensor | None = None,
             heads: tuple | None = None, prior: torch.Tensor | None = None) -> dict:
    """the mean of the losses over fixed windows, the weights untouched, and in windows each window's with its loop and
    the sum of heads, the decoders' unless given, the rate's reading (Rate); with low, the first pitch over the strikes'
    low bar, their F1 and a cents head's reads by kind, the median miss, error stated and share within it; with notes,
    the windows' (validation), what plays each as heard (noted), under prior the names' loss a cell beside the prior's;
    of a frame model with the heads the beats' and bar lines' F1 at their peaks, None and no loss where no frame is asked"""
    model.eval()
    sums: dict = {}
    each, loops = {}, []
    counts = torch.zeros(4, dtype=torch.long, device=device)
    bars = torch.where(torch.arange(model.cfg.pitches, device=device) < low, STRIKE_BARS[1], STRIKE_BARS[0]) if low is not None else None
    timed = "beat" in dict(model.cfg.outputs)
    beats, metered = torch.zeros(2, 4, dtype=torch.long, device=device), 0.0  # the two heads' peaks counted, and the frames asked
    read = []  # each asked reading's miss and stated error in cents, and its kind
    voiced = notes is not None and notes.shape[1] > 0 and "pitchfamily" in dict(model.cfg.outputs)
    # a note's cells' say summed by their weights, then the weights and those of the cells under a higher one sounding
    heard = torch.zeros(notes.shape[1] if voiced else 0, ripmodels.NAMES + ripmodels.WAVES + ripmodels.VOICE_GENES + 2, device=device)
    named = torch.zeros(1 + ripmodels.NAMES, dtype=torch.float64, device=device)  # the names' loss summed over its cells, and the cells by name
    with torch.no_grad():
        for fr, t in zip(frs, tgs):
            fr, t = fr.to(device), {k: v.to(device) for k, v in t.items()}
            l = losses(model, fr, t)
            for k, v in l.items():
                x = v.item()
                sums[k] = sums.get(k, 0.0) + x
                each.setdefault(k, []).append(x)
            loops.append(int(t["loop"][0]) if "loop" in t else len(loops))
            if low is not None or timed:
                out = model(fr)
            if timed:
                on = t["valid"] * t["metered"]
                metered += on.sum().item()
                for k, name in enumerate(("beat", "downbeat")):
                    for i in range(len(out)):
                        beats[k] += struck(out[i][:, model.cfg.slice(name)], t[name][i][:, None], on[i], BEAT_BAR)
            if low is not None:
                y = out[..., model.cfg.slice("strike")]
                for i in range(len(y)):
                    counts += struck(y[i], t["strike"][i], t["valid"][i] * (1 - t["placed"][i]), bars)
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
                    cell = torch.arange(model.cfg.pitches, device=device)
                    under = cell < (t["pitch"] * cell).amax(-1, keepdim=True)
                    w = w[at]
                    heard.index_add_(0, t["note"][at].long(), torch.cat([v[at] * w[:, None], w[:, None], (w * under[at])[:, None]], 1))
                    at = (t["pitchfamily"] >= 0) & (t["valid"][..., None] > 0)
                    named[0] += F.cross_entropy(cells(out, model.cfg, "pitchfamily")[at], t["pitchfamily"][at].long(), reduction="sum")
                    named[1:] += torch.bincount(t["pitchfamily"][at].long(), minlength=ripmodels.NAMES)
    model.train()
    out = {k: v / max(len(frs), 1) for k, v in sums.items()}
    if voiced:
        out["voices"] = noted(heard.cpu(), notes, prior)
        if prior is not None and named[1:].sum() > 0:  # a mean over the cells, where the loss above is one over windows, an empty one nothing
            count = named[1:].cpu()
            out["voices"]["cells"] = (named[0].item() / count.sum().item(), -(count[count > 0] * prior[count > 0].log()).sum().item() / count.sum().item())
    if read and sum(r.shape[1] for r in read):
        miss, stated, kind = torch.cat(read, 1)
        out["read"] = [(name, miss[at].median().item(), stated[at].median().item(),
                        (miss[at] <= stated[at]).float().mean().item() if name != CENTS_KINDS[3] else float("nan"))
                       for name, at in [("cents", kind < 3)] + [(k, kind == i) for i, k in enumerate(CENTS_KINDS)] if at.any()]
    if low is not None:
        out["struck"] = fscore(counts)
    if timed:  # a corpus that lists no beat asks neither head, and a mean over no frame would read as a head never wrong
        out["beats"] = tuple(fscore(c) for c in beats) if metered else None
        if not metered:
            out.pop("beat", None), out.pop("downbeat", None)
    w = {k: torch.tensor(x, dtype=torch.float64) for k, x in each.items() if k in out}
    if w:
        w["decoders"], w["loop"] = sum(f * w[k] for k, f in heads or DECODERS if k in w), torch.tensor(loops)
        out["windows"] = w
    return out


STRIKE_POS = 20.0  # a strike's weight against the frames without one
DIP = 0.0  # the share of the channel's windows given a dip under the melody, as a phone across a room hears one
STRIKE_AGAIN = 1.0  # and a strike of a pitch still sounding, the most missed of all, by this much more
CENTS_GRAIN = 0.001  # a label's cents are written to a tenth: a miss under it, in semitones, says no more
EVEN_GLIDES = False  # the gliding readings half the cents loss wherever a batch holds them, a few in a hundred of its readings
SPREAD = 0.2  # the stated error's weight in the total: its own row of weights learns as fast as any, the layers under it less from it
PLAYS = 0.25  # what plays a cell against what plays the frame: its heads teach the channels the decoders read, and at the whole weight the strikes are heard worse


# The rate a pass that carries on follows, doc/rip-training.md §8, each constant by the procedure written there
NOISE = 2.0  # a difference between two readings counts past this many standard errors of it
ALPHA = 0.5 * math.erfc(NOISE / math.sqrt(2))  # the share of differences by chance alone past the noise, one way
STALL = 3  # readings without a gain that stall a figure showing none, and passes in a row without one before a wide pass
FALL = 0.5  # what a fall back leaves of the fine rate, and a raise that pays gives back
# the heads the decoders read, and the weight of each in the decoders' loss
DECODERS = (("onset", 1.0), ("drums", 1.0), ("pitch", 1.0), ("voicing", 0.5), ("strike", 1.0), ("release", 1.0), ("beat", 1.0), ("downbeat", 1.0))
# of those, the heads audio reads from each model of a pair, a notes model's and the frame model's beside it: a pass that carries on is read by its model's
READ = {True: ("pitch", "strike", "release"), False: ("onset", "drums", "beat", "downbeat")}


def apart(new, old, loops) -> tuple:
    """how far reading new's figure is over old's on the same windows, and the noise of that: NOISE standard errors
    of the mean difference, a loop's windows one draw, as they share its sound"""
    d = np.asarray(new, np.float64) - np.asarray(old, np.float64)
    _, g = np.unique(np.asarray(loops), return_inverse=True)
    n, per, mean = np.bincount(g), np.bincount(g, d), d.mean()
    if len(n) < 2:
        return mean, math.inf
    return mean, NOISE * math.sqrt(len(n) / (len(n) - 1) * ((per - n * mean) ** 2).sum()) / len(d)


def stall(d: float, noise: float) -> int:
    """the readings without a gain that stall a figure whose first STALL read d on the reading kept: STALL where they
    show no gain, else as many as a figure gaining at their pace takes to gain two noises, one to pass and one more
    against chance, as STALL is at a noise a reading"""
    return STALL if d >= 0 else int(min(2 * STALL * noise / -d, 1e9)) + 1


def better(old: dict, new: dict, heads=DECODERS) -> tuple:
    """whether reading new is better than old, each window's figures: the decoders' loss under old's past its noise,
    and none of its heads over its own past its noise; with how far the loss is over old's, its noise, and the heads
    past theirs"""
    d, noise = apart(new["decoders"], old["decoders"], new["loop"])
    fell = [h for h, _ in heads if h in new and h in old and np.greater(*apart(new[h], old[h], new["loop"]))]
    return d < -noise and not fell, d, noise, fell


def rows(model) -> dict:
    """each head's own weights, as (layer, rows) of the layer that outputs it: a frame model's output layer, or its
    tower's; a notes model's notes convolution for a head of each pitch, its frame layer for a head of the frame"""
    cfg, out = model.cfg, {}
    if isinstance(model, ripmodels.PitchNet):
        sizes = dict(cfg.outputs)
        layers = [(model.notes, ripmodels.cell_heads(cfg)), (model.frame, [(k, sizes[k]) for k in ripmodels.FRAME_HEADS])]
    elif cfg.towers:
        tops = list(accumulate((h for _, h in cfg.towers), initial=0))
        layers = [(tower.out, cfg.outputs[a:b]) for tower, a, b in zip(model.towers, tops, tops[1:])]
    else:
        layers = [(model.out, cfg.outputs)]
    for layer, heads in layers:
        at = 0
        for k, n in heads:
            out[k], at = (layer, slice(at, at + n)), at + n
    return out


def probe(model) -> callable:
    """holds every weight of a model but the rows of the heads a probe trains: a notes model's cell heads that say
    what plays each pitch, a frame model's beat and downbeat. No other weight takes a gradient, theirs is masked
    out of the output layer that holds them among the other heads, and the function returned puts the other rows
    back after a step, since AdamW's decay moves a row with no gradient"""
    for p in model.parameters():
        p.requires_grad_(False)
    probed = ("pitchfamily", "pitchwave", "pitchvoice") if isinstance(model, ripmodels.PitchNet) else ("beat", "downbeat")
    heads = rows(model)
    layer = heads[probed[0]][0]
    mask = torch.zeros(len(layer.bias), device=layer.weight.device)
    for k in probed:
        mask[heads[k][1]] = 1
    held = {t: t.detach().clone() for t in (layer.weight, layer.bias)}
    for t in held:
        t.requires_grad_(True)
        t.register_hook(lambda g: g * mask.view(-1, *[1] * (g.dim() - 1)))

    def hold():
        with torch.no_grad():
            for t, was in held.items():
                t[mask == 0] = was[mask == 0]
    return hold


def unask_family(model) -> None:
    """a frame model's family head asked nothing: its rows of the output layer zero, which no term of the loss moves
    nor AdamW's decay, so every frame states even shares and the file says nothing there"""
    at = model.cfg.slice("family")
    with torch.no_grad():
        model.out.weight[at], model.out.bias[at] = 0, 0


def losses(model: ripmodels.FrameModel, fr: torch.Tensor, t: dict) -> dict:
    y = model(fr)
    cfg = model.cfg
    valid = t["valid"]
    def bce(name, target, pos_weight, asked=valid):
        logits = y[..., cfg.slice(name)]
        target = target if target.dim() == 3 else target[..., None]
        l = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.as_tensor(pos_weight, device=logits.device), reduction="none")
        return (l.mean(-1) * asked).sum() / asked.sum().clamp(min=1)
    out = {
        "onset": bce("onset", t["onset"], 8.0),
        "drums": bce("drums", t["drums"], 8.0),
        "pitch": bce("pitch", t["pitch"], 12.0),
        "voicing": bce("voicing", t["voicing"], 2.0),
        "strike": bce("strike", t["strike"], STRIKE_POS * (1 + (STRIKE_AGAIN - 1) * F.pad(t["pitch"][:, :-3], (0, 0, 3, 0))), valid * (1 - t["placed"])),
        "release": bce("release", t["release"], 20.0),
    }
    heard = t["heard"] * valid[..., None]
    velocity = torch.sigmoid(y[..., cfg.slice("velocity")])
    out["velocity"] = ((velocity - t["velocity"]) ** 2 * heard).sum() / heard.sum().clamp(min=1)
    voiced = t["voiced"] * valid
    n = voiced.sum().clamp(min=1)
    wave_logits = y[..., cfg.slice("wave")].reshape(-1, ripmodels.WAVES)
    wave_target = t["wave"].reshape(-1).clamp(min=0)
    ce = F.cross_entropy(wave_logits, wave_target, reduction="none").reshape(voiced.shape)
    out["wave"] = (ce * voiced).sum() / n
    voice = torch.sigmoid(y[..., cfg.slice("voice")])
    out["voice"] = ((voice - t["voice"]) ** 2).mean(-1).mul(voiced).sum() / n
    if "family" in dict(cfg.outputs) and not FAMILY_UNASKED:  # the frame model's alone
        asked = (t["family"] >= 0) * valid
        logits = y[..., cfg.slice("family")].reshape(-1, len(ripmodels.FAMILIES))
        ce = F.cross_entropy(logits, t["family"].reshape(-1).clamp(min=0), reduction="none").reshape(asked.shape)
        out["family"] = (ce * asked).sum() / asked.sum().clamp(min=1)
    if "beat" in dict(cfg.outputs):  # a beat weighs as a frame without one: audio reads each head's level as a posterior
        out["beat"] = bce("beat", t["beat"], 1.0, t["metered"] * valid)
        out["downbeat"] = bce("downbeat", t["downbeat"], 1.0, t["metered"] * valid)
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
                    + out.get("beat", 0.0) + out.get("downbeat", 0.0)
                    + 2.0 * out.get("cents", 0.0) + SPREAD * out.get("spread", 0.0)
                    + PLAYS * (0.5 * out.get("pitchfamily", 0.0) + 0.5 * out.get("pitchwave", 0.0) + 2.0 * out.get("pitchvoice", 0.0)))
    return out


def shape(cfg) -> str:
    """a model's sizes and heads for a message; the heads are ripmodels', never a flag's"""
    return ", ".join(f"{k} {v}" if k != "outputs" else "outputs " + "+".join(h for h, _ in v) for k, v in cfg.items()) if cfg else "an unrecorded shape"


def whose(name: str) -> str:
    return name + ("'" if name.endswith("s") else "'s")


class Rate:
    """The rate of a pass that carries on (--carry), by the held-out readings, doc/rip-training.md §8: fine while the
    decoders gain, up by FALL at each gain kept and down by it at each fall back; raised to --lr once the readings
    without a gain stall it (stall), a raise kept unless it ends over where it began past the noise. A head stalled
    while the decoders gain is raised alone, its rows' step scaled; a wide pass is a raise as long as the pass."""

    CARRIED = ("fine", "quiet")  # from pass to pass; the rest is a pass's own, which a resumed run reads back

    def __init__(self, model, opt, top: float, steps: int, every: int, wide: int, heads=DECODERS):
        self.model, self.opt, self.top, self.steps, self.every, self.wide, self.sum = model, opt, top, steps, every, wide, heads
        self.heads = rows(model)
        self.fine, self.quiet = top, 0  # a model no pass trained starts fine at --lr
        self.start = self.ref = self.kept = self.swing = None
        self.count, self.due, self.gained, self.own = 0, STALL, False, {}
        self.moving, self.undo, self.put, self.said = [], None, [], ""

    def state(self) -> dict:
        return {k: getattr(self, k) for k in self.CARRIED + ("start", "ref", "kept", "swing", "count", "due", "gained", "own")}

    def load(self, saved: dict | None, whole: bool = False) -> str:
        """what a checkpoint carries, all of it to a run resumed (whole); from a checkpoint of none the fine rate
        starts a fall under --lr, as a good model's does"""
        if saved is None:
            self.fine = self.top * FALL
            return f"no rate carried, fine {self.fine:.1e}, a fall under --lr"
        for k, v in saved.items():
            if whole or k in self.CARRIED:
                setattr(self, k, v)
        self.fine = min(self.fine, self.top)
        return f"fine {self.fine:.1e} carried, " + self.passes()

    def snap(self, step: int) -> tuple:
        return {k: t.detach().clone() for k, t in self.model.state_dict().items()}, copy.deepcopy(self.opt.state_dict()), step

    def curve(self, s: dict, step: int) -> float:
        """a raise's rate at step: --lr at its first, down half a cosine to the fine rate at its end"""
        return self.fine + (self.top - self.fine) * 0.5 * (1 + math.cos(math.pi * (step - s["at"]) / (s["end"] - s["at"])))

    def rate(self, step: int) -> float:
        s = self.swing
        return self.curve(s, step) if s is not None and s["at"] <= step < s["end"] else self.fine

    def at(self, step: int) -> float:
        """the rate for step, and each raised head's rows as they stand, for scale to move at the head's own rate"""
        lr, self.moving = self.rate(step), []
        for h, o in self.own.items():
            if o["swing"] is not None and lr > 0:
                layer, at = self.heads[h]
                self.moving.append((layer, at, [p.detach()[at].clone() for p in (layer.weight, layer.bias)], self.curve(o["swing"], step) / lr))
        return lr

    def scale(self) -> None:
        """each raised head's rows moved at its own rate: AdamW's step is the rate times what the rate leaves alone"""
        with torch.no_grad():
            for layer, at, was, k in self.moving:
                for p, x in zip((layer.weight, layer.bias), was):
                    p[at] = x + k * (p[at] - x)

    def rows_of(self, h: str) -> dict:
        """a head's rows and the optimizer's state of them"""
        layer, at = self.heads[h]
        out = {}
        for name in ("weight", "bias"):
            p = getattr(layer, name)
            out[name] = p.detach()[at].clone()
            out.update({f"{name}.{k}": v[at].clone() for k, v in self.opt.state.get(p, {}).items() if torch.is_tensor(v) and v.shape == p.shape})
        return out

    @staticmethod
    def told(d: float, noise: float, fell: list, against: str) -> str:
        """a difference of the decoders' loss as a line says it, and the heads past their noise"""
        return f"{d:+.3f} on {against}, noise {noise:.3f}" + (f", over past their own noise: {', '.join(fell)}" if fell else "")

    def keep(self, w: dict, step: int) -> None:
        self.ref, self.kept, self.count, self.gained = w, self.snap(step), 0, True

    def stalled(self) -> bool:
        """whether the readings without a gain have stalled the decoders' loss: as many as their first STALL said"""
        return self.count >= max(self.due, STALL)

    def reset(self, w: dict) -> None:
        """each head followed afresh from reading w: no head asked nothing there"""
        self.count, self.own = 0, {h: {"ref": w[h], "count": 0, "swing": None} for h in self.heads if h in w and bool(w[h].any())}

    def back(self, kept: tuple, w: dict) -> None:
        """the weights and optimizer's state kept, put back at settle, and the heads followed from their reading w"""
        self.undo = kept
        self.reset(w)

    def read(self, step: int, w: dict) -> str:
        """reading w at step, evaluate's windows: what the rate was and what the rule does now, for the held-out line;
        the weights a fall back puts back wait for settle, after the validation line has read the ones read here"""
        if self.start is None:
            return self.begin(w)
        s, a, b = self.swing, self.rate((step - 1) // self.every * self.every), self.rate(step - 1)
        self.said = f"; rate {a:.1e}" + (f" to {b:.1e}" if f"{a:.1e}" != f"{b:.1e}" else "")
        if s is None:
            self.said += " fine"
            text = f"{self.said}: " + self.follow(step, w)
        else:
            k, n = (-(-(x - s["at"]) // self.every) for x in (step, s["end"]))
            self.said += " wide" if s["kind"] == "wide" else " raised"
            text = f"{self.said}: reading {k} of {n}" + (", " + self.judge(s, w) if step >= s["end"] else "")
        if step >= self.steps:
            self.quiet = 0 if self.gained or s is not None and s["kind"] == "wide" else self.quiet + 1
        return text

    def begin(self, w: dict) -> str:
        """the start, read before a step: kept, and swung from by a wide pass"""
        self.start, self.ref, self.kept, self.gained = w, w, self.snap(0), False
        self.reset(w)
        if self.quiet < self.wide:
            self.said = f"; rate {self.fine:.1e} fine"
            return f"; the start, read before a step: rate {self.fine:.1e} fine"
        self.swing = {"kind": "wide", "at": 0, "end": self.steps, "began": w, "before": self.kept}
        self.said = f"; rate {self.top:.1e} wide"
        return f"; the start, read before a step: rate {self.top:.1e} wide, down to fine {self.fine:.1e} over the pass, after {self.passes()}"

    def follow(self, step: int, w: dict) -> str:
        """a reading at the fine rate: a gain, the decoders' loss under the kept one's past its noise, is kept; one over
        it past its noise falls back to it; as many readings without a gain as the first STALL of them say (stall)
        raise the rate, or fall it back where it is --lr already; then each head on its own"""
        d, noise = apart(w["decoders"], self.ref["decoders"], w["loop"])
        told = self.told(d, noise, [], f"the reading kept at step {self.kept[2]}")
        if d < -noise:
            self.keep(w, step)
            return f"gained, {told}" + self.follow_heads(step, w)
        if d > noise:
            self.fine *= FALL
            self.back(self.kept, self.ref)
            return f"{told}: fallen back to step {self.kept[2]}'s weights, fine {self.fine:.1e}"
        self.count += 1
        if self.count == STALL:
            self.due = stall(d, noise)
        stalled = self.stalled()
        told += f", {self.count} without a gain" + (f" of the {self.due} the gain they show takes" if self.count >= STALL and not stalled else "")
        if stalled and self.fine >= self.top:
            self.fine *= FALL
            self.count = 0
            return f"{told}: fallen back to fine {self.fine:.1e}, as --lr is the most a raise reaches"
        if stalled and step + STALL * self.every <= self.steps:
            self.swing = {"kind": "raise", "at": step, "end": step + STALL * self.every, "began": w, "before": self.snap(step)}
            self.reset(w)
            return f"{told}: raised to {self.top:.1e}, down to fine over {STALL} readings"
        return told + (", too few readings left to raise" if stalled else "") + self.follow_heads(step, w)

    def judge(self, s: dict, w: dict) -> str:
        """a raise or a wide pass at its end. Better than the reading kept, a wide pass's start: a gain kept. Over where
        it began past the noise, or a head past its own: back to the weights before it, and a raise's fine rate falls.
        Else kept as no gain, the reading kept moving with the weights"""
        self.swing = None
        wide = s["kind"] == "wide"
        good, d, noise, fell = better(s["began"] if wide else self.ref, w, self.sum)
        told = self.told(d, noise, fell, "the start" if wide else f"the reading kept at step {self.kept[2]}")
        if good:
            self.keep(w, s["end"])
            self.reset(w)
            self.fine = min(self.fine / FALL, self.top)
            return f"{'better' if wide else 'paid'}, {told}: {'ending on it' if wide else 'kept'}, fine risen to {self.fine:.1e}"
        _, up, slack, fell = better(s["began"], w, self.sum)
        if up > slack or fell:
            self.back(s["before"], s["began"])
            if wide:
                return f"worse, {told}: ending on the start"
            self.fine *= FALL
            return (f"not paid, {told}, and {self.told(up, slack, fell, 'where it began')}: "
                    f"fallen back to step {s['at']}'s weights, fine {self.fine:.1e}")
        self.ref, self.kept = w, self.snap(s["end"])
        self.reset(w)
        if wide:
            return f"no better, {told}, within its noise: ending on it"
        return f"not paid, {told}, and {self.told(up, slack, [], 'where it began')}, within its noise: kept, the reading kept now this one"

    def follow_heads(self, step: int, w: dict) -> str:
        """each head on its own: one STALL readings without a gain past its own noise while the decoders' loss has not
        stalled is raised alone, its rows at a raise's rate; at the raise's end its rows go back to before it where its
        figure ended over where it began past its noise, and its reading kept moves with them where they stay"""
        text = ""
        for h, o in self.own.items():
            d, noise = apart(w[h], o["ref"], w["loop"])
            s = o["swing"]
            if s is not None and step < s["end"]:
                a, b = self.curve(s, step - self.every), self.curve(s, step - 1)
                text += f"; {whose(h)} rows rate {a:.1e} to {b:.1e} raised: reading {(step - s['at']) // self.every} of {STALL}"
            elif s is not None:
                o["swing"], o["count"] = None, 0
                if d < -noise:
                    o["ref"] = w[h]
                    text += f"; {whose(h)} rows raised: reading {STALL} of {STALL}, paid, {d:+.3f} on its kept, noise {noise:.3f}"
                elif np.greater(*apart(w[h], s["began"], w["loop"])):
                    self.put.append((h, s["before"]))
                    text += f"; {whose(h)} rows raised: reading {STALL} of {STALL}, not paid and over where it began past its noise: fallen back to step {s['at']}'s rows"
                else:
                    o["ref"] = w[h]
                    text += f"; {whose(h)} rows raised: reading {STALL} of {STALL}, not paid, within its noise of where it began"
            elif d < -noise:
                o["ref"], o["count"] = w[h], 0
            else:
                o["count"] += 1
                if o["count"] >= STALL and not self.stalled() and self.fine < self.top and step + STALL * self.every <= self.steps:
                    o["swing"] = {"at": step, "end": step + STALL * self.every, "began": w[h], "before": self.rows_of(h)}
                    text += (f"; {h}: {d:+.3f} on its kept, noise {noise:.3f}, {o['count']} without a gain while the decoders' loss "
                             f"has not stalled: its rows raised to {self.top:.1e}, down to fine over {STALL} readings")
        return text

    def settle(self) -> None:
        """the fall backs the last reading called for, after the validation line has read the weights it read"""
        if self.undo is not None:
            self.model.load_state_dict(self.undo[0])
            self.opt.load_state_dict(copy.deepcopy(self.undo[1]))  # the optimizer steps its state in place
            self.undo = None
        with torch.no_grad():
            for h, saved in self.put:
                layer, at = self.heads[h]
                for name in ("weight", "bias"):
                    p = getattr(layer, name)
                    p[at] = saved[name]
                    for k, v in self.opt.state.get(p, {}).items():
                        if f"{name}.{k}" in saved:
                            v[at] = saved[f"{name}.{k}"]
        self.put = []

    def passes(self) -> str:
        return f"{self.quiet} pass{'es' if self.quiet != 1 else ''} in a row without a gain kept, a wide one after {self.wide}"

    def close(self) -> str:
        """what the pass carries to the next"""
        return f"the rate carries fine {self.fine:.1e} on, " + ("a gain kept this pass" if self.gained else self.passes())


def main() -> None:
    global RING, STRIKE_POS, STRIKE_AGAIN, DIP, FAMILY_HOLDOUT, EVEN_GLIDES, PLAYS, FAMILY_UNASKED
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="directory stapes' corpus verb wrote")
    ap.add_argument("--mix", default="", help="another corpus and the share of windows drawn from it, dir:share, such as recordings with their MIDI files (corpus <dir> from <list>)")
    ap.add_argument("--over", default="", help="a corpus of drums alone and the share of windows that lay it over a loop with no drums of its own, dir:share, at -12 to +6 dB of its power against the loop's: real kits under the corpus's lines")
    ap.add_argument("--channel", default="", help="a noise bank and the share of windows heard through a small speaker, a room and a phone, file.npz:share")
    ap.add_argument("--out", required=True, help="weight file to write, the root's ripframe.bin for stapes to hear with")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--window", type=int, default=512, help="frames per training window (11.6 ms each)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ffn", type=int, default=1024)
    ap.add_argument("--towers", default="", help="stacks of their own after the shared input, layers:heads each in the heads' order, comma-separated: 2:2,4:10 gives onset and drums two layers and the rest four")
    ap.add_argument("--convs", default="", help="convolutions before the input layer, channels x frames x features / pool each, comma-separated: 16x3x3/1,16x3x3/2,32x3x3/2")
    ap.add_argument("--attention-window", type=int, default=64, help="frames attended either side")
    ap.add_argument("--pcen", action="store_true", help="two more input planes, each band's PCEN and their mean, the fine plane's and the sharp one's; a notes model hears all four")
    ap.add_argument("--strike-weight", type=float, default=20.0, help="a strike's weight in its head's loss against a frame without one")
    ap.add_argument("--again-weight", type=float, default=1.0, help="how much more a strike of a pitch still sounding weighs than another")
    ap.add_argument("--bend", type=float, default=0.0, help="the share of windows read along a tempo that sways, so no grid or steady pulse is there to learn")
    ap.add_argument("--ring", type=float, default=0.0, help="the share of windows with a body's ring at every attack, at one pitch whatever the note")
    ap.add_argument("--dilations", default="", help="PitchNet's long convolutions, their frames apart, comma-separated: 3,9,27 hears half a second either way")
    ap.add_argument("--cell", type=int, default=32, help="PitchNet's channels at each cell, its width; the weight file carries it")
    ap.add_argument("--family-unasked", action="store_true", help="a frame model's family head grown and asked nothing: no term for it in the loss and its rows of the output layer zero from the start, so every frame's shares are even and the file says nothing there")
    ap.add_argument("--plays", type=float, default=PLAYS, help="the weight of the heads of what plays each pitch in the total, against what plays the frame: a quarter by default")
    ap.add_argument("--probe", action="store_true", help="a notes model's heads of what plays each pitch, or a frame model's beat and downbeat heads, trained alone from --init's weights, every other weight held as it was: a notes model carries no cents head, so its file rips as its start, and a frame model's other heads say what its start's did; the last step is exported")
    ap.add_argument("--even-glides", action="store_true", help="the cents head's gliding readings weigh as much as the rest in its loss wherever a batch holds both")
    ap.add_argument("--dip", type=float, default=0.0, help="the share of the channel's windows given a dip of 6 to 20 dB centred 100 to 500 Hz, as a phone across a room hears")
    ap.add_argument("--sung", default="", help="a corpus of solo singing and the share of a shuffled window's chunks drawn from it, dir:share")
    ap.add_argument("--shuffle", type=float, default=0.0, help="the share of windows spliced from chunks of a second, each two clips summed")
    ap.add_argument("--extra", default="", help="more corpora, dir:share, comma-separated: each draws its share of the windows the ones before it left")
    ap.add_argument("--pitchnet", action="store_true", help="the notes model: convolutions only over the harmonic stack, the same weights at every pitch; --out is then ripnotes.bin, which stapes reads beside ripframe.bin")
    ap.add_argument("--family-holdout", choices=ripmodels.FAMILIES, help="a family no head is told the name of: its notes are trained on as sound and scored as notes, and a notes model's lines print the share their surest name reaches, the measure of where a name stops being sure, which its weight file carries")
    ap.add_argument("--sure", type=float, default=0.0, help="the share a family must pass over a note to name it, as a --family-holdout run of the same recipe measured it, for a notes model's weight file to carry; without it the start's is kept, and with neither no family names a note")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shard-seed", type=int, default=0, help="moves the streams the batches are drawn from, so a round's workers see different windows; the starting weights stay --seed's")
    ap.add_argument("--builders", type=int, default=1, help="processes forked from the run that build its batches beside the device, each at most two batches ahead: batch k is drawn "
                    "from a stream of --seed, --shard-seed and k alone, so the batches are the same whatever builds them and a run resumed at step k draws those the unbroken run drew")
    ap.add_argument("--checkpoint", default="", help="write a .pt every --every steps, and resume from it when present")
    ap.add_argument("--init", default="", help="start from this .pt's model weights, without its optimizer state (--carry carries it) or step: a training round's common start; a --checkpoint present wins")
    ap.add_argument("--carry", action="store_true", help="a pass that carries on from --init's checkpoint, or starts a chain without one: its optimizer's state and its rate's carried as well as its weights "
                    "(a checkpoint of weights alone starts the optimizer fresh), the held-out loops read before a step, and the rate following their readings by doc/rip-training.md §8 rather than "
                    "running a cosine from --lr: fine while the decoders gain, raised to --lr after 3 readings without a gain, fallen back when a raise does not pay")
    ap.add_argument("--wide", type=int, help="with --carry, a pass swings wide, from --lr down to the fine rate over the pass, after this many passes in a row kept no gain, "
                    "as its checkpoint counts them (3 by default); 0 swings this one. It ends on whichever is better, its start or its end")
    ap.add_argument("--every", type=int, default=200, help="steps between checkpoints and the held-out and validation scores")
    ap.add_argument("--holdout", type=int, default=-1, help="loops held out of training and scored every --every steps; the step that scores best is what --out gets (default a tenth, none under five loops)")
    ap.add_argument("--validate", default="", help="a corpus of recordings (corpus <dir> from <list>) cut into fixed windows and scored every --every steps; the step that scores best on it is what --out gets, in place of the held-out loops' best: by the decoders' loss, a notes model by its strikes' F1 at the decoder's bars")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    RING, STRIKE_POS, STRIKE_AGAIN, DIP, EVEN_GLIDES, PLAYS, FAMILY_UNASKED = args.ring, args.strike_weight, args.again_weight, args.dip, args.even_glides, args.plays, args.family_unasked
    FAMILY_HOLDOUT = ripmodels.FAMILIES.index(args.family_holdout) if args.family_holdout else None

    torch.manual_seed(args.seed)
    if not (0 <= args.seed < 1 << 32 and 0 <= args.shard_seed < 1 << 32):
        raise SystemExit(f"--seed {args.seed} --shard-seed {args.shard_seed}: each from 0 to 2**32-1, or two runs' batches could share a stream")
    if args.builders < 1:
        raise SystemExit(f"--builders {args.builders}: at least one")
    convs = [tuple(int(v) for v in c.replace("/", "x").split("x")) for c in args.convs.split(",") if c]
    if any(len(c) != 4 for c in convs):
        raise SystemExit(f"--convs {args.convs!r}: want channels x frames x features / pool, comma-separated")
    towers = [tuple(int(v) for v in t.split(":")) for t in args.towers.split(",") if t]
    if any(len(t) != 2 for t in towers) or towers and sum(h for _, h in towers) != len(ripmodels.HEADS):
        raise SystemExit(f"--towers {args.towers!r}: want layers:heads each, the heads adding up to {len(ripmodels.HEADS)}")
    if args.family_unasked and (args.pitchnet or towers):
        raise SystemExit("--family-unasked: the family head is a frame model's, on one output layer")
    if args.probe and towers and not args.pitchnet:
        raise SystemExit("--probe with --towers: a model in towers lays its heads over several output layers, and a probe holds one")
    if args.carry and args.probe:
        raise SystemExit("--carry with --probe: a probe holds the weights the rate moves")
    if args.wide is not None and (not args.carry or args.wide < 0):
        raise SystemExit(f"--wide {args.wide}: the passes before a wide one, 0 or more, of a pass that carries on (--carry)")
    heads = [h for h in ripmodels.PITCHNET_HEADS if h[0] != "cents" or not args.probe]  # a probe's file rips as its start
    heard = dict(inputs=ripmodels.STACK_BINS + 1, channels=ripmodels.STACK_CHANNELS, outputs=heads) if args.pitchnet else \
        dict(channels=4 if args.pcen else 2, outputs=list(ripmodels.HEADS))
    cfg = ripmodels.Config(**heard, convs=convs, dim=args.dim, layers=args.layers, heads=args.heads, ffn=args.ffn, window=args.attention_window, towers=towers,
                          dilations=tuple(int(d) for d in filter(None, args.dilations.split(","))), cell=args.cell)
    planes = ripmodels.planes(cfg)  # what the data loader makes; a notes model stacks them
    resume = bool(args.checkpoint) and os.path.exists(args.checkpoint)
    start = args.checkpoint if resume else args.init
    ck = torch.load(start, map_location=args.device) if start else None
    if ck is not None and ck.get("layout") != ripmodels.LAYOUT:  # before its shape is read: another layout's is no start at all
        raise SystemExit(ripmodels.refused(start, ck.get("layout")))
    if resume and args.carry != ("rate" in ck):
        raise SystemExit(f"{start} was written {'with' if 'rate' in ck else 'without'} --carry: resume it as it began")
    had = None
    if ck is not None and "cfg" in ck:
        had = ripmodels.shape(ck["cfg"])  # a setting newer than the checkpoint takes the value it was built with
        had.outputs = [tuple(h) for h in had.outputs]
        ck["cfg"] = asdict(had)
    # a start from a model of the heads before the last added, such as the beat and the downbeat: its weights carry
    # over, a new head's rows of the output layer fresh
    grown = False
    if had is not None and not resume and asdict(had) != asdict(cfg):
        rest = lambda c: {k: v for k, v in asdict(c).items() if k != "outputs"}
        grown = rest(had) == rest(cfg) and had.outputs == cfg.outputs[:len(had.outputs)]
    # refused before the corpus loads: load_state_dict names only a tensor's size,
    # and takes another head count or window without a word
    if ck is not None and ck.get("cfg") != asdict(cfg) and not grown:
        raise SystemExit(f"{start} holds a model of {shape(ck.get('cfg'))}, and this run builds {shape(asdict(cfg))}: pass its shape, or another file")
    manifest, loops = load_corpus(args.corpus)
    holdout = args.holdout if args.holdout >= 0 else (len(loops) // 10 if len(loops) >= 5 else 0)
    held = loops[len(loops) - holdout:] if holdout else []
    loops = loops[:len(loops) - holdout] if holdout else loops
    if args.carry and len(held) < 2:
        raise SystemExit(f"--carry: the rate follows the held-out loops' readings, and a reading's noise takes two loops at least, not {len(held)} (--holdout)")
    val_frs, val_tgs, val_notes = validation(held, args.window, planes)
    prior = shares(val_tgs) if args.pitchnet else None
    # a pass that carries on is read by the heads audio reads from its model in a pair, any other run by them all
    heads = tuple((h, f) for h, f in DECODERS if h in READ[args.pitchnet]) if args.carry else DECODERS
    assert manifest["voice_genes"] == ripmodels.VOICE_GENES and len(manifest["waves"]) == ripmodels.WAVES
    recorded, check_frs, check_tgs, check_notes = [], [], [], None
    if args.validate:
        _, recorded = load_corpus(args.validate)
        if not recorded:
            raise SystemExit(f"--validate {args.validate}: no recordings to score")
        check_frs, check_tgs, check_notes = validation(recorded, args.window, planes)
    mixed, share = [], 0.0
    if args.mix:
        d, _, sh = args.mix.rpartition(":")
        _, mixed = load_corpus(d)
        share = float(sh)
    drums, over_share, drumless = [], 0.0, []
    if args.over:
        d, _, sh = args.over.rpartition(":")
        _, drums = load_corpus(d)
        over_share = float(sh)
        drumless = [(m, l) for m, l in loops if not (l.drum >= 0).any()]
    model = (ripmodels.PitchNet if args.pitchnet else ripmodels.FrameModel)(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    step, best_loss, best_step, best_state = 0, float("inf"), 0, None
    validated = (float("inf"), 0, None, None)  # the best validation loss, its step, its weights and the held-out loss there
    rate = Rate(model, opt, args.lr, args.steps, args.every, STALL if args.wide is None else args.wide, heads) if args.carry else None
    if resume:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        best_loss, best_step, best_state = ck.get("best", (best_loss, best_step, best_state))
        validated = ck.get("validated", validated)
        if rate:
            rate.load(ck["rate"], whole=True)
        print(f"resumed at step {step}")
    elif grown:
        try:
            model.load_state_dict(ripmodels.carried(model, had, ck["model"]))
        except ValueError as err:
            raise SystemExit(f"{start}: {err}")
        print(f"started from {args.init}, its heads but {', '.join(h for h, _ in cfg.outputs[len(had.outputs):])}, which start fresh")
        if rate:
            print(f"carried on, the optimizer fresh as the heads grew: {rate.load(ck.get('rate'))}")
    elif ck is not None:
        model.load_state_dict(ck["model"])
        print(f"started from {args.init}")
        if rate:
            if "opt" in ck:
                opt.load_state_dict(ck["opt"])
            print(f"carried on, {'its optimizer' if 'opt' in ck else 'weights alone, the optimizer fresh'}: {rate.load(ck.get('rate'))}")
    if args.sure and hasattr(model, "sure"):
        model.sure.fill_(args.sure)
    if FAMILY_UNASKED:
        unask_family(model)
        print("the family head is asked nothing: its rows zero, every frame's shares even")
    hold = probe(model) if args.probe else None
    print(f"{sum(p.numel() for p in model.parameters())} weights, {len(loops)} loops, {sum(len(m) for m, _ in loops)} frames, {len(held)} held out in {len(val_frs)} windows")
    if rate:
        print(f"read by its {', '.join(h for h, _ in heads)}, the heads audio reads from a {'notes' if args.pitchnet else 'frame'} model in a pair")
    if recorded:
        print(f"{len(recorded)} recordings of {sum(len(m) for m, _ in recorded)} frames to validate on in {len(check_frs)} windows, from {args.validate}")
    if mixed:
        print(f"{len(mixed)} loops of {sum(len(m) for m, _ in mixed)} frames mixed in, {share:.0%} of the windows")
    if drums:
        print(f"{len(drums)} recordings of {sum(len(m) for m, _ in drums)} frames laid over {len(drumless)} loops with no drums, {over_share:.0%} of the windows")
    t0 = time.time()

    low = STRIKE_LOW - ripmodels.LOW if args.pitchnet else None

    def scored(name: str, frs, tgs, notes, said=None):
        """the decoders' loss over fixed windows, a notes model's strike F1, and the line that reports them, a frame
        model's with its beats' and bar lines' F1, and what said says of them before the parts, which end the line;
        a family held out sets the model's measure by its notes there"""
        v = evaluate(model, frs, tgs, args.device, low, notes, heads, prior)
        # the wave, voice, velocity and family heads say how a note is played and by
        # what, and memorize the corpus's, so they do not pick the step
        v["decoders"] = sum(f * v.get(k, 0.0) for k, f in heads)
        parts = " ".join(f"{k} {x:.3f}" for k, x in v.items() if k not in ("total", "decoders", "struck", "read", "voices", "beats", "windows"))
        f1 = f", strikes' F1 {v['struck']:.3f}" if "struck" in v else ""
        if "beats" in v:
            f1 += ", beats' F1 {:.3f}, bar lines' {:.3f}".format(*v["beats"]) if v["beats"] else ", no beat or bar line asked"
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
            if "cells" in n:  # the names' loss a cell beside the held-out shares' alone, and the notes named with those shares divided out
                by = ", ".join(f"{k} {x:.3f}" for k, x in n["even"][1])
                voices += f"; names {n['cells'][0]:.3f} a cell, {n['cells'][1]:.3f} by their shares alone, {n['even'][0]:.3f} of the notes with the shares divided out ({by})"
        return v, f"step {step} {name} {v['total']:.3f}, decoders {v['decoders']:.3f}{f1}{read}{voices}{said(v) if said else ''} ({parts})"

    def weights() -> dict:
        return {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}

    def score() -> None:
        nonlocal best_loss, best_step, best_state, validated
        held_loss = None
        if val_frs:
            v, line = scored("held-out", val_frs, val_tgs, val_notes, rate and (lambda v: rate.read(step, v["windows"])))
            held_loss = v["decoders"]
            if held_loss < best_loss:
                best_loss, best_step, best_state = held_loss, step, weights()
                line += " (best)"
            print(line, flush=True)
        if check_frs:
            # a notes model's loss on recordings is mostly its onset head, which no decoder
            # reads from it, and ranks rounds unlike their note scores; its strikes' F1 does
            v, line = scored("validation", check_frs, check_tgs, check_notes, rate and (lambda v: rate.said))
            key = v["decoders"] if v.get("struck") is None else -v["struck"]
            if key < validated[0]:
                validated = (key, step, weights(), held_loss)
                line += " (best)"
            print(line, flush=True)
        if rate:
            rate.settle()

    channel, channel_share = None, 0.0
    if args.channel:
        f, _, sh = args.channel.rpartition(":")
        channel, channel_share = Channel(f), float(sh)
        print(f"{channel_share:.0%} of the windows through a speaker, a room and a phone, from {f}")

    # recordings run from seconds to minutes, and each second is worth the same
    by_mixed = list(accumulate(len(m) for m, _ in mixed))
    by_drums = list(accumulate(len(m) for m, _ in drums))

    extras, left = [], 1.0
    for item in filter(None, args.extra.split(",")):
        d, _, sh = item.rpartition(":")
        _, loops_x = load_corpus(d)
        extras.append((float(sh), loops_x, list(accumulate(len(m) for m, _ in loops_x))))
        print(f"{len(loops_x)} loops of {sum(len(m) for m, _ in loops_x)} frames from {d}, {float(sh) * left:.1%} of the windows")
        left *= 1 - float(sh)

    def draw(rng):
        """a loop, and the drums to lay over it or None"""
        for sh, xs, by in extras:
            if rng.random() < sh:
                return (*rng.choices(xs, cum_weights=by)[0], None)
        if drums and drumless and rng.random() < over_share:
            ofr, olab = rng.choices(drums, cum_weights=by_drums)[0]
            return (*rng.choice(drumless), (ofr, olab, rng.random() * 18 - 12))
        if mixed and rng.random() < share:
            return (*rng.choices(mixed, cum_weights=by_mixed)[0], None)
        return (*rng.choice(loops), None)

    sung, sung_share, by_sung = [], 0.0, []
    if args.sung:
        d, _, sh = args.sung.rpartition(":")
        _, sung = load_corpus(d)
        sung_share, by_sung = float(sh), list(accumulate(len(m) for m, _ in sung))
        print(f"{len(sung)} sung recordings of {sum(len(m) for m, _ in sung)} frames from {d}, {sung_share:.0%} of the chunks")

    def pick(rng):
        """a recording for a shuffled window's chunk: singing, a loop, or drums alone by length"""
        if sung and rng.random() < sung_share:
            return rng.choices(sung, cum_weights=by_sung)[0]
        if drums and rng.random() < 0.3:
            return rng.choices(drums, cum_weights=by_drums)[0]
        return rng.choice(loops)

    if rate and step == 0:  # the start read on this pass's held-out loops, which every reading after is judged against
        score()
    # Processes, since a batch is Python holding the interpreter's lock; forked, they share the corpora's pages until a
    # write copies one, and the collector, frozen while they fork, writes to no header of theirs
    build = lambda rng: batch(draw, args.window, args.batch, rng, planes, channel, channel_share, pick, args.shuffle, args.bend)
    loader = torch.utils.data.DataLoader(Batches(build, args.seed, args.shard_seed), batch_size=None, sampler=range(step, args.steps),
                                         num_workers=args.builders, prefetch_factor=2, multiprocessing_context="fork")
    gc.freeze()
    batches = iter(loader)
    gc.unfreeze()
    for fr, t in batches:  # in index order, batch k at step k
        fr, t = fr.to(args.device), {k: v.to(args.device) for k, v in t.items()}
        lr = rate.at(step) if rate else args.lr * 0.5 * (1 + math.cos(math.pi * step / args.steps))  # the readings' rate, or a cosine to zero
        for g in opt.param_groups:
            g["lr"] = lr
        l = losses(model, fr, t)
        opt.zero_grad()
        l["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if hold:
            hold()
        if rate:
            rate.scale()
        step += 1
        if step % 20 == 0 or step == args.steps:
            parts = " ".join(f"{k} {v.item():.3f}" for k, v in l.items() if k != "total")
            print(f"step {step} loss {l['total'].item():.3f} ({parts}) {time.time() - t0:.0f}s", flush=True)
        if step % args.every == 0 or step == args.steps:
            score()
            if args.checkpoint:
                # the layout and the shape ride along for average.py and the checks above, the
                # frames heard, the heads and the window showing in no tensor; the best held-out
                # and validation steps, so a resumed run, or one run again once done, still exports its pick; the rate's
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "layout": ripmodels.LAYOUT, "cfg": asdict(cfg),
                            "best": (best_loss, best_step, best_state), "validated": validated, **({"rate": rate.state()} if rate else {})}, args.checkpoint)
    if rate:
        print(rate.close())
    if args.probe:
        print(f"a probe: exporting its last step, {step}")
    elif args.validate and validated[2] is not None:
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
    ripmodels.export(model.cpu(), args.out)
    sure = ""
    if hasattr(model, "sure"):
        sure = f", a family naming a note past a share of {model.sure.item():.3f}" if model.sure > 0 else ", no family naming a note: no run measured where a name stops being sure (--family-holdout, --sure)"
    print(f"wrote {args.out}{sure}")


if __name__ == "__main__":
    main()
