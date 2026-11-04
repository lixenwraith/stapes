"""The rip's front-end model as PyTorch builds it, the twin of lixenwraith/audio/ripmodel
and lixenwraith/transformer: convolutions over the frames as an image, a linear input,
pre-norm attention blocks with ALiBi biases over a window of frames, exact GELU
feed-forwards, a final norm and one output layer whose slices are the heads.
export() writes the weight file lixenwraith/transformer loads, its tensor names and order.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# lixenwraith/audio/ripmodel.Spec, which a corpus manifest restates
MEL_BANDS = 229
INPUTS = MEL_BANDS + 1  # the bands over the frame's level, and the level
PITCHES = 69
DRUMS = 3
WAVES = 5
VOICE_GENES = 32

HEADS = [("onset", 1), ("drums", DRUMS), ("pitch", PITCHES), ("voicing", 1), ("wave", WAVES), ("voice", VOICE_GENES), ("strike", PITCHES),
         ("release", PITCHES), ("velocity", PITCHES)]


# the energy 80 dB under a full-scale sine's, where the model's logarithm floors
FLOOR = 1e-8

# PCEN, as lixenwraith/audio/ripmodel's constants
FRAME_SEC = 256 / 22050
PCEN_TIME, PCEN_GAIN, PCEN_BIAS, PCEN_POWER, PCEN_EPS = 0.4, 0.98, 2.0, 0.5, 1e-6


def features(mel: np.ndarray, channels: int = 1) -> np.ndarray:
    """The model's input from log-mel frames as the corpus stores them, log1p(100 e):
    each band's energy in dB over FLOOR less the frame's mean, and the mean itself,
    both over 80 dB, so a frame's loudness is one input and the bands are level-free;
    with two channels, each band's PCEN and their mean after them; as
    lixenwraith/audio/ripmodel's features computes them"""
    e = np.expm1(mel.astype(np.float64)) / 100
    db = 10 * np.log10(e + FLOOR)
    level = db.mean(axis=-1, keepdims=True)
    out = [(db - level) / 80, (level + 80) / 80]
    if channels == 2:
        smooth = 1 - math.exp(-FRAME_SEC / PCEN_TIME)
        avg, a = np.empty_like(e), e[0]
        for t, row in enumerate(e):
            a = (1 - smooth) * a + smooth * row
            avg[t] = a
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


class RipModel(nn.Module):
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


def tensors(model: RipModel):
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


def export(model: RipModel, path: str) -> None:
    cfg = model.cfg
    body = bytearray(MAGIC)
    body += struct.pack("<I", VERSION)
    body += struct.pack("<7I", cfg.inputs, cfg.channels, cfg.dim, cfg.heads, cfg.layers, cfg.ffn, cfg.window)
    body += struct.pack("<I", len(cfg.convs))
    for c in cfg.convs:
        body += struct.pack("<4I", *c)
    body += struct.pack("<I", len(cfg.towers))
    for t in cfg.towers:
        body += struct.pack("<2I", *t)
    body += struct.pack("<I", len(cfg.outputs))
    for name, size in cfg.outputs:
        b = name.encode()
        body += struct.pack("<H", len(b)) + b + struct.pack("<I", size)
    ts = tensors(model)
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
