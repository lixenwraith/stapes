"""The rip's models as PyTorch builds them, the twin of lixenwraith/audio/ripmodels and lixenwraith/transformer: the
frame model, FrameModel, convolutions over the frames as an image, a linear input, pre-norm attention blocks with ALiBi
biases over a window of frames, exact GELU feed-forwards, a final norm and one output layer whose slices are the heads;
the notes model, PitchNet, below; band_frames(), audio's BandFrames; and export(), the weight file
lixenwraith/transformer loads, its tensor names and order."""
from __future__ import annotations

import functools
import math
import struct
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# the frames the models hear, lixenwraith/audio's band frames (BandFrames) of the layout RipLayout names, which a
# corpus manifest, a checkpoint and a weight file state: BANDS bands three a semitone, band k at MIDI BAND_LOWEST + k/3,
# a stored frame the fine plane's then the sharp plane's
LAYOUT, LAYOUT_NAME = 1, "315 bands three a semitone from G#0, a fine plane and a sharp one"
BANDS, BAND_LOWEST = 315, 20
VALUES = 2 * BANDS
INPUTS = BANDS + 1  # a plane of the models' input: its bands over the plane's level, and the level
LOW, PITCHES = 21, 88  # the pitches the models tell apart, the piano's keys from A0 (audio's RipPiano)
DRUMS = 3
WAVES = 5
VOICE_GENES = 32
FAMILIES = ("piano", "guitar", "voice", "bass", "strings", "winds", "synth", "drums", "unknown")  # RipFamilies, in its order
NAMES = len(FAMILIES) - 1  # the families a note may be named: unknown is the absence of a name, the last

# the frame model's heads, each added after the last; struck says that something was struck, of no line or pitch
HEADS = [("onset", 1), ("drums", DRUMS), ("pitch", PITCHES), ("voicing", 1), ("wave", WAVES), ("voice", VOICE_GENES), ("strike", PITCHES),
         ("release", PITCHES), ("velocity", PITCHES), ("family", len(FAMILIES)), ("beat", 1), ("downbeat", 1), ("struck", 1)]


def refused(what: str, layout) -> str:
    """why frames or weights of another layout, or of none, are refused, by this one's name"""
    of = "no frame layout" if layout is None else f"frame layout {layout:g}"
    return f"{what} states {of}, and these models hear layout {LAYOUT} ({LAYOUT_NAME})"


# audio's rip rate, hop and window, its levels of the rate halved and the band whose bins fall three a semitone at
# level 0 (ripBandLevels, ripLevelBand), a level's transform twice its window, and Resample's low-pass taps
RATE, HOP, N = 22050, 256, 2048
LEVELS, LEVEL_BAND, PAD, TAPS = 6, 158, 2, 31
AHEAD = ((N // 2 + TAPS // 2) << (LEVELS - 1)) - TAPS // 2  # the samples past a frame's centre its longest window reads, RipBandsAhead


def band_hz(k):
    """the frequency of band k, a fraction between bands"""
    return 440 * 2 ** ((BAND_LOWEST + np.asarray(k, np.float64) / 3 - 69) / 12)


def band_level(k: int) -> int:
    """the level the fine plane hears band k at: each level down serves the 36 bands under the one above"""
    return min(max((LEVEL_BAND - k + 35) // 36, 0), LEVELS - 1)


def _hann() -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / N)


def _power(x: np.ndarray) -> np.ndarray:
    """the power spectrum of windows, a row each, through a transform PAD times their length"""
    X = np.fft.rfft(x, N * PAD)
    return X.real ** 2 + X.imag ** 2


@functools.cache
def _bands() -> tuple:
    """each band of the fine plane then the sharp one, as (level, first bin, weights): a triangle from the band
    under to the band over, a bin either side at least, scaled so a unit cosine at its centre reads one"""
    out = []
    for k, level in [(k, band_level(k)) for k in range(BANDS)] + [(k, 0) for k in range(BANDS)]:
        rate = RATE / (1 << level)
        bin, c = rate / (N * PAD), float(band_hz(k))
        lo, hi = max(c - float(band_hz(k - 1)), bin), max(float(band_hz(k + 1)) - c, bin)
        first = math.ceil((c - lo) / bin)
        f = np.arange(first, min(int((c + hi) / bin), N * PAD // 2) + 1) * bin
        w = np.where(f <= c, (f - c + lo) / lo, (c + hi - f) / hi)
        unit = _power(np.cos(2 * np.pi * c * (np.arange(N) - N // 2) / rate) * _hann())
        out.append((level, first, w / (w @ unit[first:first + len(w)])))
    return tuple(out)


def _halve(x: np.ndarray) -> np.ndarray:
    """x through Resample's low-pass at half its rate, every other sample, as audio's resampler gives it of a whole
    recording: each tap within the recording, the sum over all of them"""
    t = np.arange(TAPS)
    cutoff, w = 0.225, 0.5 - 0.5 * np.cos(2 * np.pi * t / (TAPS - 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.where(t == TAPS // 2, 2 * cutoff * w, np.sin(2 * np.pi * cutoff * (t - TAPS // 2)) / (np.pi * (t - TAPS // 2)) * w)
    padded, acc, total = np.concatenate([np.zeros(TAPS // 2), x, np.zeros(TAPS // 2)]), np.zeros(len(x)), 0.0
    for k, c in enumerate(h):  # in audio's order, so the samples are its own
        acc += c * padded[k:k + len(x)]
        total += c
    return (acc / total)[:2 * (len(x) // 2):2]


def band_energy(x: np.ndarray) -> np.ndarray:
    """BandStream's energies of a recording at RATE, a row a frame on the rip's grid, frame t centred on sample
    t*HOP: each band under its weights, the fine plane's at its level and the sharp plane's at level 0"""
    levels, bands, hann = [np.asarray(x, np.float64)], _bands(), _hann()
    for _ in range(1, LEVELS):
        levels.append(_halve(levels[-1]))
    frames, size = -(-len(x) // HOP), N * PAD // 2 + 1
    out = np.zeros((frames, VALUES))
    for j, y in enumerate(levels):
        at = [i for i, (level, _, _) in enumerate(bands) if level == j]
        weights = np.zeros((len(at), size))
        for row, i in enumerate(at):
            _, first, w = bands[i]
            weights[row, first:first + len(w)] = w
        padded = np.concatenate([np.zeros(N), y, np.zeros(N)])  # a window past either end reads silence
        for t in range(0, frames, 512):
            starts = (np.arange(t, min(t + 512, frames)) * HOP >> j) - N // 2 + N
            out[t:t + len(starts), at] = _power(padded[starts[:, None] + np.arange(N)] * hann) @ weights.T
    return out


def band_frames(x: np.ndarray) -> np.ndarray:
    """audio's BandFrames of a recording at RATE: each band's energy as log1p of a hundred times it, the recording's
    peak at one"""
    peak = float(np.abs(x).max()) if len(x) else 0.0
    return np.log1p(100 * band_energy(x) / (peak * peak or 1.0)).astype(np.float32)


def sine_response(hz: float) -> np.ndarray:
    """the energy each band of both planes reads of a steady unit cosine at hz, one at a band's centre"""
    bands, power, out = _bands(), {}, np.zeros(VALUES)
    for i, (level, first, w) in enumerate(bands):
        if level not in power:
            power[level] = _power(np.cos(2 * np.pi * hz * (np.arange(N) - N // 2) / (RATE / (1 << level))) * _hann())
        out[i] = w @ power[level][first:first + len(w)]
    return out


@functools.cache
def sharp_counts() -> np.ndarray:
    """how many times the sharp plane's bands, summed, count a steady cosine at each band's centre: about twenty at
    A0, where their triangles share bins, near one at the top"""
    return np.array([sine_response(float(band_hz(k)))[BANDS:].sum() for k in range(BANDS)])


# the energy 80 dB under a full-scale sine's, where the model's logarithm floors
FLOOR = 1e-8

# PCEN, as lixenwraith/audio/ripmodels' constants
FRAME_SEC = HOP / RATE
PCEN_TIME, PCEN_GAIN, PCEN_BIAS, PCEN_POWER, PCEN_EPS = 0.4, 0.98, 2.0, 0.5, 1e-6


# Harmonic input: each of the four planes a channel for each harmonic, the bands shifted so a pitch's harmonics lie at
# its own bin, bin k band k, from the semitone under A0 to the one over C8, zero past the bands
HARMONICS = (0.5, 1, 2, 3, 4, 5, 6, 7)
STACK_PLANES = 4
STACK_CHANNELS = STACK_PLANES * len(HARMONICS)
STACK_SHIFTS = tuple(round(36 * math.log2(h)) for h in HARMONICS)
STACK_BINS = 3 * (PITCHES + 2)  # three a semitone, a cell either side of the pitches


def features(frames: np.ndarray, planes: int = 2) -> np.ndarray:
    """The models' input from band frames as the corpus stores them, log1p(100 e): for the fine plane and then the
    sharp one, each band's energy in dB over FLOOR less the plane's mean, and the mean, over 80 dB, so the bands are
    level-free and a plane's loudness is one input; with four planes, each band's PCEN and their mean, the fine
    plane's and then the sharp one's; as lixenwraith/audio/ripmodels' features computes them"""
    e = np.expm1(frames.astype(np.float64)) / 100
    out = []
    for plane in (e[:, :BANDS], e[:, BANDS:]):
        db = 10 * np.log10(plane + FLOOR)
        level = db.mean(axis=-1, keepdims=True)
        out += [(db - level) / 80, (level + 80) / 80]
    if planes == 4:
        smooth = 1 - math.exp(-FRAME_SEC / PCEN_TIME)
        # a[t] = (1 - s) a[t-1] + s e[t] from a[0] = e[0], as one running sum over growing weights
        grow = (1 - smooth) ** -np.arange(len(e), dtype=np.float64)
        run = e * grow[:, None]
        run[0] = e[0] / smooth
        avg = smooth * np.cumsum(run, axis=0) / grow[:, None]
        p = (e / (PCEN_EPS + avg) ** PCEN_GAIN + PCEN_BIAS) ** PCEN_POWER - PCEN_BIAS ** PCEN_POWER
        for plane in (p[:, :BANDS], p[:, BANDS:]):
            out += [plane, plane.mean(axis=-1, keepdims=True)]
    return np.concatenate(out, axis=-1).astype(np.float32)


@dataclass
class Config:
    inputs: int = INPUTS
    channels: int = 2  # of inputs each, a frame laying them end to end: a frame model's planes, two or four
    # (channels, frames, features, pool) each: a kernel centred on each frame
    # and feature, GELU, then the features max-pooled; none feeds the frames on
    convs: list = field(default_factory=list)
    dim: int = 256
    heads: int = 8
    layers: int = 4
    ffn: int = 1024
    window: int = 64
    # (layers, heads) each: a stack of its own over the shared input computing
    # the next heads; none is one stack of `layers` computing all of them
    towers: list = field(default_factory=list)
    outputs: list = field(default_factory=lambda: list(HEADS))
    # the frames apart of each convolution over the cells after the near ones, PitchNet's alone
    dilations: tuple = ()
    cell: int = 32  # PitchNet's channels at each cell, its width

    @property
    def pitches(self) -> int:
        return dict(self.outputs)["pitch"]

    @property
    def features(self) -> int:
        """what the input layer takes a frame as"""
        n, ch = self.inputs, self.channels
        for out, _, _, pool in self.convs:
            n, ch = n // pool, out
        return n * ch

    @property
    def out_size(self) -> int:
        return sum(n for _, n in self.outputs)

    def slice(self, name: str) -> slice:
        at = 0
        for h, n in self.outputs:
            if h == name:
                return slice(at, at + n)
            at += n
        raise KeyError(name)


def shape(saved: dict) -> Config:
    """the Config a checkpoint recorded: a setting newer than the checkpoint takes the value it was built with"""
    return Config(**{**asdict(Config()), **saved})


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.q = nn.Linear(cfg.dim, cfg.dim)
        self.k = nn.Linear(cfg.dim, cfg.dim)
        self.v = nn.Linear(cfg.dim, cfg.dim)
        self.o = nn.Linear(cfg.dim, cfg.dim)
        self.ln2 = nn.LayerNorm(cfg.dim)
        self.ff1 = nn.Linear(cfg.dim, cfg.ffn)
        self.ff2 = nn.Linear(cfg.ffn, cfg.dim)
        # head h leans on nearer frames by 2^(-8h/H) per frame of distance
        slopes = torch.tensor([2.0 ** (-8.0 * (h + 1) / cfg.heads) for h in range(cfg.heads)])
        self.register_buffer("slopes", slopes, persistent=False)

    def attend(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.cfg.heads, D // self.cfg.heads
        q = self.q(x).view(B, T, H, dh).transpose(1, 2)  # B H T dh
        k = self.k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.v(x).view(B, T, H, dh).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(dh)  # B H T T
        pos = torch.arange(T, device=x.device)
        dist = (pos[:, None] - pos[None, :]).abs()  # T T
        scores = scores - self.slopes[None, :, None, None] * dist[None, None].to(scores.dtype)
        scores = scores.masked_fill((dist > self.cfg.window)[None, None], float("-inf"))
        w = scores.softmax(-1)
        out = (w @ v).transpose(1, 2).reshape(B, T, D)
        return self.o(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attend(self.ln1(x))
        return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))


class Tower(nn.Module):
    def __init__(self, cfg: Config, layers: int, size: int):
        super().__init__()
        self.layers = nn.ModuleList(Block(cfg) for _ in range(layers))
        self.ln = nn.LayerNorm(cfg.dim)
        self.out = nn.Linear(cfg.dim, size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.out(self.ln(x))


class FrameModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        chans = [cfg.channels] + [c[0] for c in cfg.convs]
        self.convs = nn.ModuleList(nn.Conv2d(chans[i], out, (t, f), padding=(t // 2, f // 2)) for i, (out, t, f, _) in enumerate(cfg.convs))
        self.input = nn.Linear(cfg.features, cfg.dim)
        if cfg.towers:
            sizes, at = [], 0
            for _, heads in cfg.towers:
                sizes.append(sum(n for _, n in cfg.outputs[at:at + heads]))
                at += heads
            self.towers = nn.ModuleList(Tower(cfg, layers, size) for (layers, _), size in zip(cfg.towers, sizes))
        else:
            self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
            self.ln = nn.LayerNorm(cfg.dim)
            self.out = nn.Linear(cfg.dim, cfg.out_size)

    def forward(self, planes: torch.Tensor) -> torch.Tensor:
        """planes: B T channels*inputs → B T out_size, raw scores; the heads are slices."""
        x = planes
        if self.cfg.convs:
            x = x.view(x.shape[0], x.shape[1], self.cfg.channels, self.cfg.inputs).transpose(1, 2)  # B C T F
            for conv, (_, _, _, pool) in zip(self.convs, self.cfg.convs):
                x = F.gelu(conv(x))
                if pool > 1:
                    x = F.max_pool2d(x, (1, pool))
            x = x.permute(0, 2, 1, 3).flatten(2)  # B T channels*features, as Go lays a frame
        x = self.input(x)
        if self.cfg.towers:
            return torch.cat([t(x) for t in self.towers], -1)
        for layer in self.layers:
            x = layer(x)
        return self.out(self.ln(x))


# --- the weight file lixenwraith/transformer loads ---

# version 6: a word for the model's kind, 0 a transformer and 1 a cell net, after the version; the constants a file
# states by name after its heads; the bins a cell net's regions pool after its long convolutions
MAGIC = b"VIFT"
VERSION, KIND_CELLS = 6, 1
LAYOUT_CONSTANT = "layout"  # the constant that states the frames a model hears, lixenwraith/audio/ripmodels.Layout


def _fnv64a(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def tensors(model: FrameModel):
    """Every weight by the name lixenwraith/transformer gives it, in its order."""
    sd = model.state_dict()
    out = []
    for i in range(len(model.cfg.convs)):
        out += [(f"convs.{i}.weight", sd[f"convs.{i}.weight"]), (f"convs.{i}.bias", sd[f"convs.{i}.bias"])]
    out += [("in.weight", sd["input.weight"]), ("in.bias", sd["input.bias"])]
    stacks = [(f"towers.{t}.", layers) for t, (layers, _) in enumerate(model.cfg.towers)] or [("", model.cfg.layers)]
    for pt, layers in stacks:
        for i in range(layers):
            p = f"{pt}layers.{i}."
            for name in ("ln1", "q", "k", "v", "o", "ln2", "ff1", "ff2"):
                out.append((p + name + ".weight", sd[p + name + ".weight"]))
                out.append((p + name + ".bias", sd[p + name + ".bias"]))
        out += [(pt + n, sd[pt + n]) for n in ("ln.weight", "ln.bias", "out.weight", "out.bias")]
    return out


class PitchNet(nn.Module):
    """Convolutions only: the same weights at every pitch, each pitch's heads read from its own
    neighbourhood on the harmonic stack, the frame's heads from what the frequencies pool to."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg, self.bins = cfg, STACK_BINS
        self.reduce = nn.Conv2d(STACK_CHANNELS, 8, 1)
        self.wide = nn.Conv2d(8, 16, (3, 39), padding=(1, 19))  # thirteen semitones
        self.cells = nn.Conv2d(16, cfg.cell, (7, 3), stride=(1, 3), padding=(3, 0))  # to one cell a semitone
        self.near = nn.Conv2d(cfg.cell, cfg.cell, (7, 3), padding=(3, 1))
        self.long = nn.ModuleList(nn.Conv2d(cfg.cell, cfg.cell, 3, padding=(d, 1), dilation=(d, 1)) for d in cfg.dilations)
        self.notes = nn.Conv2d(cfg.cell, sum(n for _, n in cell_heads(cfg)), 1)  # a channel for each value a cell head reads at a pitch
        self.frame1 = nn.Conv1d(16 * 16 + STACK_PLANES, 64, 5, padding=2)  # the regions, then each plane's scalar
        self.frame2 = nn.Conv1d(64, 64, 9, padding=8, dilation=2)
        sizes = dict(cfg.outputs)
        self.frame = nn.Linear(64, sum(n for _, n in frame_heads(cfg)))
        if "pitchfamily" in sizes:  # the share a family must reach over a note to name it; zero until a run measures one
            self.register_buffer("sure", torch.zeros(()))

    def stack(self, x: torch.Tensor) -> torch.Tensor:
        """B T STACK_PLANES*INPUTS -> B T STACK_CHANNELS*(bins+1): each plane by harmonic, bin k band k shifted, each
        channel ending in the plane's scalar, as audio's stacked lays them"""
        chans, lead = [], -STACK_SHIFTS[0]
        for plane in x.chunk(STACK_PLANES, -1):
            bands = F.pad(plane[..., :BANDS], (lead, self.bins + STACK_SHIFTS[-1] - BANDS))  # zero past the bands
            for sh in STACK_SHIFTS:
                chans += [bands[..., lead + sh:lead + sh + self.bins], plane[..., BANDS:]]
        return torch.cat(chans, -1)

    def forward(self, planes: torch.Tensor) -> torch.Tensor:
        B, T = planes.shape[:2]
        x = self.stack(planes).view(B, T, STACK_CHANNELS, self.bins + 1).transpose(1, 2)  # B C T F
        scalars = x[:, ::len(HARMONICS), :, self.bins]  # each plane's, its first channel's last
        x = F.gelu(self.reduce(x[..., :self.bins]))
        w = F.gelu(self.wide(x))
        c = F.gelu(self.cells(F.pad(w, (1, 0))))  # cell j is centred on bin 3j, the semitone under A0 + j
        c = F.gelu(self.near(c) + c)
        for layer in self.long:
            c = F.gelu(layer(c) + c)
        n = self.notes(c)[..., 1:self.cfg.pitches + 1]  # B channels T pitches
        f = F.adaptive_avg_pool2d(w, (T, 16)).transpose(2, 3).flatten(1, 2)  # B 16*16 T
        f = torch.cat([f, scalars], 1)
        f = F.gelu(self.frame1(f))
        f = self.frame(F.gelu(self.frame2(f) + f).transpose(1, 2))  # B T frame heads
        at, part = 0, {}
        for k, size in frame_heads(self.cfg):
            part[k], at = f[..., at:at + size], at + size
        at = 0
        for k, per in cell_heads(self.cfg):  # a head of several values a pitch lays each value's pitches in turn
            part[k], at = n[:, at:at + per].transpose(1, 2).flatten(2), at + per
        return torch.cat([part[k] for k, _ in self.cfg.outputs], -1)


# PitchNet's file lists its heads as a cell net's: the per-pitch ones in the notes convolution's order, then the frame's
# the per-pitch heads and the values each reads at a pitch; a new head adds a row.
# cents is a pitch's offset from its cell's centre in semitones, then the log of the error it states;
# the three after it say what plays the pitch: a score for each family a note may be named, one for each wave, and the voice's genes
CELL_HEADS = (("pitch", 1), ("strike", 1), ("release", 1), ("velocity", 1), ("cents", 2),
              ("pitchfamily", NAMES), ("pitchwave", WAVES), ("pitchvoice", VOICE_GENES))
FRAME_HEADS = ("onset", "drums", "voicing", "wave", "voice", "struck")


# PitchNet's heads: those of the frame model's in its order but the frame's family, which names one note of the frame,
# then the per-pitch heads the frame model has none of, then struck, added after them
PITCHNET_HEADS = [h for h in HEADS if h[0] in dict(CELL_HEADS) or h[0] in FRAME_HEADS and h[0] != "struck"] + \
    [(k, n * PITCHES) for k, n in CELL_HEADS if k not in dict(HEADS)] + [("struck", 1)]


def cell_heads(cfg: Config) -> list:
    """the per-pitch heads a model of this shape has, in the notes convolution's order: one trained before a head has none of it"""
    return [(k, n) for k, n in CELL_HEADS if k in dict(cfg.outputs)]


def frame_heads(cfg: Config) -> list:
    """the frame heads a notes model of this shape has and their sizes, in its frame layer's order"""
    sizes = dict(cfg.outputs)
    return [(k, sizes[k]) for k in FRAME_HEADS if k in sizes]


def carried(model: nn.Module, had: Config, state: dict) -> dict:
    """model's weights with those of a model of the shape had, whose heads are the first of model's: every weight
    they share whole, and of an output layer, or of a notes model's notes convolution, the rows of the heads it had,
    a new head's rows fresh; ValueError where one does not grow into the other"""
    cfg, out = model.cfg, model.state_dict()
    if [tuple(h) for h in had.outputs] != [tuple(h) for h in cfg.outputs[:len(had.outputs)]]:
        raise ValueError(f"heads {'+'.join(k for k, _ in had.outputs)} are not the first of {'+'.join(k for k, _ in cfg.outputs)}")
    for k, v in state.items():
        if out[k].shape == v.shape:
            out[k] = v
        elif v.dim() and out[k].shape[1:] == v.shape[1:] and len(out[k]) > len(v):
            out[k][:len(v)] = v  # the rows of the heads it had, a head's rows after the last's
        else:
            raise ValueError(f"{k} is {tuple(v.shape)}, this run's {tuple(out[k].shape)}")
    return out


def planes(cfg: Config) -> int:
    """the planes of features a model hears: a frame model its channels, two or four, and a notes model all four"""
    return STACK_PLANES if cfg.channels == STACK_CHANNELS else cfg.channels


def build(cfg: Config) -> nn.Module:
    """the model a checkpoint's shape is: PitchNet when it reads the harmonic stack"""
    return (PitchNet if cfg.channels == STACK_CHANNELS else FrameModel)(cfg)


def _nears(model: PitchNet) -> list:
    """the convolutions over the cells in turn, one module or a list of them"""
    return list(model.near) if isinstance(model.near, nn.ModuleList) else [model.near]


def _cells_header(model: PitchNet) -> bytes:
    """a cell net's sizes, in lixenwraith/transformer's Cells order, from the layers themselves"""
    wide, cells, nears, frame1, frame2 = model.wide, model.cells, _nears(model), model.frame1, model.frame2
    (wt, wf), (ct, cf), (nt, nf), ft, at = wide.kernel_size, cells.kernel_size, nears[0].kernel_size, frame1.kernel_size[0], frame2.kernel_size[0]
    dil, stride = frame2.dilation[0], cells.stride[1]
    # the Go side centres every kernel and starts cell j at bin stride*j, as forward pads
    assert wide.padding == (wt // 2, wf // 2) and cells.padding == (ct // 2, 0) and cf // 2 == 1 and cells.stride[0] == 1
    assert all(n.kernel_size == (nt, nf) and n.padding == (nt // 2, nf // 2) for n in nears)
    assert frame1.padding == (ft // 2,) and frame2.padding == (dil * (at // 2),)
    sizes, heads, bins, pitches = dict(model.cfg.outputs), cell_heads(model.cfg), model.bins, model.cfg.pitches
    assert all(sizes[k] == n * pitches for k, n in heads) and model.notes.out_channels == sum(n for _, n in heads)
    scalars = tuple(p * len(HARMONICS) * (bins + 1) + bins for p in range(STACK_PLANES))  # each plane's first channel's last
    regions = (frame1.in_channels - len(scalars)) // wide.out_channels
    assert regions * wide.out_channels + len(scalars) == frame1.in_channels
    v = (bins, model.reduce.out_channels, wide.out_channels, wt, wf, cells.out_channels, ct, cf, stride, len(nears), nt, nf,
         1, pitches, len(heads), regions, frame1.out_channels, ft, at, dil, len(scalars), *scalars,
         len(model.long), *(layer.dilation[0] for layer in model.long), 0, 0)  # the regions pool every bin
    return struct.pack(f"<{len(v)}I", *v)


def export(model: nn.Module, path: str) -> None:
    """the weight file, which states the layout of the frames the model hears"""
    cfg = model.cfg
    body = bytearray(MAGIC)
    if isinstance(model, PitchNet):
        body += struct.pack("<4I", VERSION, KIND_CELLS, cfg.inputs, cfg.channels) + _cells_header(model)
        sizes = dict(cfg.outputs)
        heads = [(k, sizes[k]) for k, _ in cell_heads(cfg)] + frame_heads(cfg)
        layers = [("reduce", model.reduce), ("wide", model.wide), ("cells", model.cells)]
        layers += [(f"near.{i}", n) for i, n in enumerate(_nears(model))]
        layers += [(f"long.{i}", n) for i, n in enumerate(model.long)]
        layers += [("cellout", model.notes), ("frame1", model.frame1), ("frame2", model.frame2), ("frameout", model.frame)]
        ts = [(f"{name}.{p}", getattr(m, p)) for name, m in layers for p in ("weight", "bias")]
        if hasattr(model, "sure"):  # the measure is a frame head of no weights, its bias the measure: every frame states it
            heads.append(("sure", 1))
            ts[-2:] = [("frameout.weight", torch.cat([model.frame.weight, torch.zeros(1, model.frame.in_features)])),
                       ("frameout.bias", torch.cat([model.frame.bias, model.sure[None]]))]
    else:
        body += struct.pack("<9I", VERSION, 0, cfg.inputs, cfg.channels, cfg.dim, cfg.heads, cfg.layers, cfg.ffn, cfg.window)
        body += struct.pack("<I", len(cfg.convs))
        for c in cfg.convs:
            body += struct.pack("<4I", *c)
        body += struct.pack("<I", len(cfg.towers))
        for t in cfg.towers:
            body += struct.pack("<2I", *t)
        heads, ts = cfg.outputs, tensors(model)
    body += struct.pack("<I", len(heads))
    for name, size in heads:
        b = name.encode()
        body += struct.pack("<H", len(b)) + b + struct.pack("<I", size)
    body += struct.pack("<IH", 1, len(LAYOUT_CONSTANT)) + LAYOUT_CONSTANT.encode() + struct.pack("<f", LAYOUT)
    body += struct.pack("<I", len(ts))
    for name, t in ts:
        b = name.encode()
        body += struct.pack("<H", len(b)) + b
        dims = list(t.shape)
        body += struct.pack("<B", len(dims)) + struct.pack(f"<{len(dims)}I", *dims)
        body += t.detach().cpu().contiguous().to(torch.float32).numpy().tobytes()
    body += struct.pack("<Q", _fnv64a(bytes(body)))
    with open(path, "wb") as f:
        f.write(body)
