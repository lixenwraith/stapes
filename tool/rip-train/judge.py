"""Fit a judge of the rip's strikes on rows stapes wrote (`rows <dir> from ...`),
and export the file lixenwraith/audio loads beside the notes model's:

    python3 judge.py --rows rows/ --leave-out held --out ripjudge.bin

A rows directory holds, a recording, its strikes' rows (<checksum>.rows, as a
loop's frames are written) and their labels (<checksum>.json): each strike's
stage, a note the decoder reported or a strike it dropped, whether a truth note
answers it, and the installed verdict. A stage's network is fitted on its rows
by one stated procedure, nothing picked by a held-out score; the conditions
left out are only counted, by pitch and onset, under the installed verdicts,
the judge's, and a right verdict on every row.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import ripmodels

MAGIC = b"VIFJ"
VERSION = 1
HIDDEN = (256, 64)
DROPOUT = 0.2  # after the first hidden layer
BATCH = 2048
PASSES = 12
LR, PEAK, DECAY = 1e-3, 2e-3, 1e-4  # AdamW's rate, the one-cycle schedule's peak, the weight decay
SEED = 0
STAGES = ("reported", "dropped")  # the file's order


@dataclass
class Rows:
    """Every strike of the directories read: its row, its stage, its label, the
    installed verdict, and its recording's condition; and each condition's
    recordings, truth notes and the truth notes no strike answers."""

    version: int
    size: int
    notes: int  # the checksum of the notes weights the rows were read through
    x: np.ndarray  # strikes x size, float32
    reported: np.ndarray
    held: np.ndarray
    right: np.ndarray
    kept: np.ndarray
    cond: np.ndarray  # an index into conds
    conds: list[str]
    recordings: np.ndarray  # per condition
    given: np.ndarray
    unanswered: np.ndarray


def load(dirs: list[str]) -> Rows:
    kind: tuple[int, int, int] | None = None
    conds: list[str] = []
    xs: list[np.ndarray] = []
    flags: list[tuple[bool, bool, bool, bool, int]] = []
    recordings: list[int] = []
    given: list[int] = []
    unanswered: list[int] = []
    for d in dirs:
        with open(os.path.join(d, "rows.json")) as f:
            m = json.load(f)
        this = (m["row_version"], m["row_size"], int(m["notes"], 16))
        if kind is not None and this != kind:
            sys.exit(f"{d}: rows of version {this[0]} read through the notes weights {this[2]:016x}, the directories before it version {kind[0]} through {kind[2]:016x}: a judge is fitted on one kind")
        kind = this
        for label in sorted(glob.glob(os.path.join(d, "*.json"))):
            if os.path.basename(label) == "rows.json":
                continue
            with open(label) as f:
                rec = json.load(f)
            with open(label[: -len(".json")] + ".rows", "rb") as f:
                n, size = struct.unpack("<II", f.read(8))
                x = np.fromfile(f, dtype="<f4", count=n * size).reshape(n, size)
            if n != len(rec["strikes"]) or n > 0 and size != this[1]:
                sys.exit(f"{label}: {len(rec['strikes'])} strike(s) beside {n} row(s) of {size}")
            if rec["condition"] not in conds:
                conds.append(rec["condition"])
                recordings.append(0)
                given.append(0)
                unanswered.append(0)
            c = conds.index(rec["condition"])
            recordings[c] += 1
            given[c] += rec["given"]
            unanswered[c] += rec["unanswered"]
            if n > 0:
                xs.append(x)
            flags += [(s["reported"], s.get("held", False), s["right"], s["kept"], c) for s in rec["strikes"]]
    if kind is None or not xs:
        sys.exit("no rows in " + ", ".join(dirs))
    fl = np.array(flags, dtype=np.int64)
    return Rows(kind[0], kind[1], kind[2], np.concatenate(xs), fl[:, 0] == 1, fl[:, 1] == 1, fl[:, 2] == 1, fl[:, 3] == 1, fl[:, 4],
                conds, np.array(recordings), np.array(given), np.array(unanswered))


def network(size: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(size, HIDDEN[0]), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(HIDDEN[0], HIDDEN[1]), nn.ReLU(), nn.Linear(HIDDEN[1], 1))


def fit(x: np.ndarray, right: np.ndarray, device: str) -> tuple[nn.Sequential, np.ndarray, np.ndarray, list[float]]:
    """A stage's network over its rows standardized by their mean and spread, a
    row that never varies left as it is, and each pass's mean loss."""
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    spread = x.std(axis=0, dtype=np.float64).astype(np.float32)
    spread[~(spread > 1e-6)] = 1.0
    xs = torch.from_numpy((x - mean) / spread).to(device)
    ys = torch.from_numpy(right.astype(np.float32)).to(device)
    net = network(x.shape[1]).to(device)
    steps = -(-len(xs) // BATCH)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=PEAK, total_steps=PASSES * steps)
    order = torch.Generator().manual_seed(SEED)
    losses: list[float] = []
    net.train()
    for _ in range(PASSES):
        perm, total = torch.randperm(len(xs), generator=order).to(device), 0.0
        for b in range(steps):
            i = perm[b * BATCH : (b + 1) * BATCH]
            loss = F.binary_cross_entropy_with_logits(net(xs[i]).squeeze(1), ys[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(i)
        losses.append(total / len(xs))
    net.eval()
    return net, mean, spread, losses


def odds(net: nn.Sequential, mean: np.ndarray, spread: np.ndarray, x: np.ndarray, device: str) -> np.ndarray:
    """A stage's odds for each row, float32 throughout as audio computes them."""
    with torch.no_grad():
        out = [torch.sigmoid(net(torch.from_numpy((x[i : i + 65536] - mean) / spread).to(device)).squeeze(1)).cpu().numpy() for i in range(0, len(x), 65536)]
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def count(kept: np.ndarray, right: np.ndarray, given: int) -> str:
    """The count by pitch and onset of the strikes kept against the truth's notes."""
    hits, heard = int((kept & right).sum()), int(kept.sum())
    return f"{hits} heard as given, given {given}, heard {heard}, F1 {2 * hits / max(given + heard, 1):.3f}"


def export(path: str, rows: Rows, stages: list[tuple[nn.Sequential, np.ndarray, np.ndarray]]) -> None:
    """The judge file: a header naming the row and the notes weights, each
    stage's mean, spread and layers as audio reads them, an FNV-1a 64 trailer."""
    body = bytearray(MAGIC + struct.pack("<5IQ", VERSION, rows.version, rows.size, HIDDEN[0], HIDDEN[1], rows.notes))
    for net, mean, spread in stages:
        body += mean.astype("<f4").tobytes() + spread.astype("<f4").tobytes()
        for layer in net:
            if isinstance(layer, nn.Linear):
                body += layer.weight.detach().cpu().numpy().astype("<f4").tobytes() + layer.bias.detach().cpu().numpy().astype("<f4").tobytes()
    body += struct.pack("<Q", ripmodels._fnv64a(bytes(body)))
    with open(path, "wb") as f:
        f.write(body)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", nargs="+", required=True, help="directories stapes' rows verb wrote, all read through one notes model")
    ap.add_argument("--leave-out", nargs="*", default=[], help="conditions the fit leaves out and counts")
    ap.add_argument("--out", default="", help="the judge file to write, the root's ripjudge.bin beside ripnotes.bin; none written without it")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    rows = load(args.rows)
    for name in args.leave_out:
        if name not in rows.conds:
            sys.exit(f"--leave-out {name}: no such condition among {', '.join(rows.conds)}")
    out = np.isin(rows.cond, [rows.conds.index(name) for name in args.leave_out])
    fitted = [name for name in rows.conds if name not in args.leave_out]
    print(f"{len(rows.x)} row(s) of {rows.size} (version {rows.version}) read through the notes weights {rows.notes:016x}; fitting on {', '.join(fitted) or 'nothing'}")
    stages: list[tuple[nn.Sequential, np.ndarray, np.ndarray]] = []
    verdict = rows.held.copy()  # a held note stays, unjudged
    for stage in STAGES:
        of = (rows.reported & ~rows.held) if stage == "reported" else ~rows.reported
        take = of & ~out
        if not take.any():
            sys.exit(f"no {stage} row to fit on")
        net, mean, spread, losses = fit(rows.x[take], rows.right[take], args.device)
        print(f"{stage}: {int(take.sum())} row(s), {rows.right[take].mean():.3f} of them right; loss a pass " + " ".join(f"{v:.3f}" for v in losses))
        verdict[of] = odds(net, mean, spread, rows.x[of], args.device) >= 0.5
        stages.append((net, mean, spread))
    for name in args.leave_out:
        c = rows.conds.index(name)
        of, given = rows.cond == c, int(rows.given[c])
        print(f"{name}: {rows.recordings[c]} recording(s), {given} note(s) given, {rows.unanswered[c]} of them no strike answers; {int(of.sum())} strike(s), {int((of & rows.reported).sum())} reported")
        print("  installed  " + count(rows.kept[of], rows.right[of], given))
        print("  judge      " + count(verdict[of], rows.right[of], given))
        print("  right      " + count((rows.right | rows.held)[of], rows.right[of], given))
    if args.out:
        export(args.out, rows, stages)
        print(f"wrote {args.out}: a judge of layers {HIDDEN[0]} and {HIDDEN[1]} a stage, for the notes weights {rows.notes:016x}")


if __name__ == "__main__":
    main()
