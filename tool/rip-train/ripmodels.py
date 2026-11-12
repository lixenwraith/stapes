"""The rip's models as PyTorch builds them, the twin of lixenwraith/audio/ripmodels and
lixenwraith/transformer. The frame model, FrameModel: convolutions over the frames as an
image, a linear input, pre-norm attention blocks with ALiBi biases over a window of frames,
exact GELU feed-forwards, a final norm and one output layer whose slices are the heads;
the notes model, PitchNet, below. export() writes the weight file lixenwraith/transformer
loads, its tensor names and order.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# lixenwraith/audio/ripmodels.FrameSpec, which a corpus manifest restates
MEL_BANDS = 229
INPUTS = MEL_BANDS + 1  # the bands over the frame's level, and the level
PITCHES = 69
DRUMS = 3
WAVES = 5
VOICE_GENES = 32
FAMILIES = ("piano", "guitar", "voice", "bass", "strings", "winds", "synth", "drums", "unknown")  # RipFamilies, in its order
NAMES = len(FAMILIES) - 1  # the families a note may be named: unknown is the absence of a name, the last

HEADS = [("onset", 1), ("drums", DRUMS), ("pitch", PITCHES), ("voicing", 1), ("wave", WAVES), ("voice", VOICE_GENES), ("strike", PITCHES),
         ("release", PITCHES), ("velocity", PITCHES), ("family", len(FAMILIES)), ("beat", 1), ("downbeat", 1)]


# the energy 80 dB under a full-scale sine's, where the model's logarithm floors
FLOOR = 1e-8

# PCEN, as lixenwraith/audio/ripmodels' constants
FRAME_SEC = 256 / 22050
PCEN_TIME, PCEN_GAIN, PCEN_BIAS, PCEN_POWER, PCEN_EPS = 0.4, 0.98, 2.0, 0.5, 1e-6


# Harmonic input: both planes on a log-frequency axis of 36 bins an octave, read from the
# mel bands, one channel per harmonic with each pitch's harmonic at the pitch's own bin
HARMONICS = (0.5, 1, 2, 3, 4, 5, 6, 7)
STACK_BINS, STACK_ROOT_MIDI, STACK_CHANNELS = 213, 27, 16
STACK_SHIFTS = tuple(round(36 * math.log2(h)) for h in HARMONICS)
STACK_INPUTS = STACK_BINS + 1


def _stack_axis():
    """each long-axis bin's lower mel band, the upper band's weight, and whether a band covers it"""
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    centers = np.log(hz(mel(30.0) + (mel(11025.0) - mel(30.0)) * np.arange(1, MEL_BANDS + 1) / (MEL_BANDS + 1)))
    low = STACK_SHIFTS[0]
    n = STACK_BINS + STACK_SHIFTS[-1] - low
    f = np.log(440.0) + np.log(2.0) * ((STACK_ROOT_MIDI + (np.arange(n) + low) / 3 - 69) / 12)
    b = np.clip(np.searchsorted(centers, f, side="right") - 1, 0, MEL_BANDS - 2)
    return b, (f - centers[b]) / (centers[b + 1] - centers[b]), (f >= centers[0]) & (f < centers[-1])


def features(mel: np.ndarray, channels: int = 1) -> np.ndarray:
    """The model's input from log-mel frames as the corpus stores them, log1p(100 e):
    each band's energy in dB over FLOOR less the frame's mean, and the mean itself,
    both over 80 dB, so a frame's loudness is one input and the bands are level-free;
    with two channels, each band's PCEN and their mean after them; as
    lixenwraith/audio/ripmodels' features computes them"""
    e = np.expm1(mel.astype(np.float64)) / 100
    db = 10 * np.log10(e + FLOOR)
    level = db.mean(axis=-1, keepdims=True)
    out = [(db - level) / 80, (level + 80) / 80]
    if channels == 2:
        smooth = 1 - math.exp(-FRAME_SEC / PCEN_TIME)
        # a[t] = (1 - s) a[t-1] + s e[t] from a[0] = e[0], as one running sum over growing weights
        grow = (1 - smooth) ** -np.arange(len(e), dtype=np.float64)
        run = e * grow[:, None]
        run[0] = e[0] / smooth
        avg = smooth * np.cumsum(run, axis=0) / grow[:, None]
        p = (e / (PCEN_EPS + avg) ** PCEN_GAIN + PCEN_BIAS) ** PCEN_POWER - PCEN_BIAS ** PCEN_POWER
        out += [p, p.mean(axis=-1, keepdims=True)]
    return np.concatenate(out, axis=-1).astype(np.float32)


@dataclass
class Config:
    inputs: int = INPUTS
    channels: int = 1  # of inputs each, a frame laying them end to end
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

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: B T channels*inputs → B T out_size, raw scores; the heads are slices."""
        x = mel
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

MAGIC = b"VIFT"
VERSION = 3


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
        self.cfg = cfg
        b, w, ok = _stack_axis()
        self.register_buffer("axis_band", torch.from_numpy(b), persistent=False)
        self.register_buffer("axis_weight", torch.from_numpy(w).float(), persistent=False)
        self.register_buffer("axis_ok", torch.from_numpy(ok).float(), persistent=False)
        self.reduce = nn.Conv2d(STACK_CHANNELS, 8, 1)
        self.wide = nn.Conv2d(8, 16, (3, 39), padding=(1, 19))  # thirteen semitones
        self.cells = nn.Conv2d(16, cfg.cell, (7, 3), stride=(1, 3), padding=(3, 0))  # to one cell a semitone
        self.near = nn.Conv2d(cfg.cell, cfg.cell, (7, 3), padding=(3, 1))
        self.long = nn.ModuleList(nn.Conv2d(cfg.cell, cfg.cell, 3, padding=(d, 1), dilation=(d, 1)) for d in cfg.dilations)
        self.notes = nn.Conv2d(cfg.cell, sum(n for _, n in cell_heads(cfg)), 1)  # a channel for each value a cell head reads at a pitch
        self.frame1 = nn.Conv1d(16 * 16 + 2, 64, 5, padding=2)
        self.frame2 = nn.Conv1d(64, 64, 9, padding=8, dilation=2)
        sizes = dict(cfg.outputs)
        self.frame = nn.Linear(64, sum(sizes[k] for k in ("onset", "drums", "voicing", "wave", "voice")))
        if "pitchfamily" in sizes:  # the share a family must reach over a note to name it; zero until a run measures one
            self.register_buffer("sure", torch.zeros(()))

    def stack(self, x: torch.Tensor) -> torch.Tensor:
        """B T 2*(MEL_BANDS+1) -> B T STACK_CHANNELS*STACK_INPUTS: each plane by harmonic, each channel ending in the plane's scalar"""
        chans, w = [], self.axis_weight
        for plane in x.chunk(2, -1):
            bands, scalar = plane[..., :MEL_BANDS], plane[..., MEL_BANDS:]
            long = (bands[..., self.axis_band] * (1 - w) + bands[..., self.axis_band + 1] * w) * self.axis_ok
            for sh in STACK_SHIFTS:
                at = sh - STACK_SHIFTS[0]
                chans += [long[..., at:at + STACK_BINS], scalar]
        return torch.cat(chans, -1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        B, T = mel.shape[:2]
        x = self.stack(mel).view(B, T, STACK_CHANNELS, STACK_INPUTS).transpose(1, 2)  # B C T F
        level, pcen = x[:, 0, :, STACK_BINS], x[:, 8, :, STACK_BINS]
        x = F.gelu(self.reduce(x[..., :STACK_BINS]))
        w = F.gelu(self.wide(x))
        c = F.gelu(self.cells(F.pad(w, (1, 0))))  # cell j is centred on bin 3j, STACK_ROOT_MIDI + j
        c = F.gelu(self.near(c) + c)
        for layer in self.long:
            c = F.gelu(layer(c) + c)
        n = self.notes(c)[..., 1:PITCHES + 1]  # B channels T pitches
        f = F.adaptive_avg_pool2d(w, (T, 16)).transpose(2, 3).flatten(1, 2)  # B 16*16 T
        f = torch.cat([f, level[:, None], pcen[:, None]], 1)
        f = F.gelu(self.frame1(f))
        f = self.frame(F.gelu(self.frame2(f) + f).transpose(1, 2))  # B T frame heads
        sizes = dict(self.cfg.outputs)
        at, part = 0, {}
        for k in ("onset", "drums", "voicing", "wave", "voice"):
            part[k], at = f[..., at:at + sizes[k]], at + sizes[k]
        at = 0
        for k, per in cell_heads(self.cfg):  # a head of several values a pitch lays each value's pitches in turn
            part[k], at = n[:, at:at + per].transpose(1, 2).flatten(2), at + per
        return torch.cat([part[k] for k, _ in self.cfg.outputs], -1)


# PitchNet's file: version 5 names the model's kind, 1 a cell net, whose heads
# are the per-pitch ones in the notes convolution's order, then the frame's
CELLS_VERSION, CELLS_KIND = 5, 1
# the per-pitch heads and the values each reads at a pitch; a new head adds a row.
# cents is a pitch's offset from its cell's centre in semitones, then the log of the error it states;
# the three after it say what plays the pitch: a score for each family a note may be named, one for each wave, and the voice's genes
CELL_HEADS = (("pitch", 1), ("strike", 1), ("release", 1), ("velocity", 1), ("cents", 2),
              ("pitchfamily", NAMES), ("pitchwave", WAVES), ("pitchvoice", VOICE_GENES))
FRAME_HEADS = ("onset", "drums", "voicing", "wave", "voice")
# PitchNet's heads: those of HEADS in its order but the frame's family, which names
# one note of the frame, then the per-pitch heads the frame model has none of
PITCHNET_HEADS = [h for h in HEADS if h[0] in dict(CELL_HEADS) or h[0] in FRAME_HEADS] + \
    [(k, n * PITCHES) for k, n in CELL_HEADS if k not in dict(HEADS)]


def cell_heads(cfg: Config) -> list:
    """the per-pitch heads a model of this shape has, in the notes convolution's order: one trained before a head has none of it"""
    return [(k, n) for k, n in CELL_HEADS if k in dict(cfg.outputs)]


def build(cfg: Config) -> nn.Module:
    """the model a checkpoint's shape is: PitchNet when it reads the harmonic stack"""
    return (PitchNet if cfg.inputs == STACK_INPUTS else FrameModel)(cfg)


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
    sizes, heads = dict(model.cfg.outputs), cell_heads(model.cfg)
    assert all(sizes[k] == n * PITCHES for k, n in heads) and model.notes.out_channels == sum(n for _, n in heads)
    scalars = (STACK_BINS, 8 * STACK_INPUTS + STACK_BINS)  # the level, the PCEN mean: channel 0's and 8's last
    regions = (frame1.in_channels - len(scalars)) // wide.out_channels
    assert regions * wide.out_channels + len(scalars) == frame1.in_channels
    v = (STACK_BINS, model.reduce.out_channels, wide.out_channels, wt, wf, cells.out_channels, ct, cf, stride, len(nears), nt, nf,
         1, PITCHES, len(heads), regions, frame1.out_channels, ft, at, dil, len(scalars), *scalars,
         len(model.long), *(layer.dilation[0] for layer in model.long))
    return struct.pack(f"<{len(v)}I", *v)


def export(model: nn.Module, path: str) -> None:
    """the weight file: version 3 for FrameModel, 5 for PitchNet"""
    cfg = model.cfg
    body = bytearray(MAGIC)
    if isinstance(model, PitchNet):
        body += struct.pack("<4I", CELLS_VERSION, CELLS_KIND, cfg.inputs, cfg.channels) + _cells_header(model)
        sizes = dict(cfg.outputs)
        heads = [(k, sizes[k]) for k in [k for k, _ in cell_heads(cfg)] + list(FRAME_HEADS)]
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
        body += struct.pack("<I", VERSION)
        body += struct.pack("<7I", cfg.inputs, cfg.channels, cfg.dim, cfg.heads, cfg.layers, cfg.ffn, cfg.window)
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
