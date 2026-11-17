#!/usr/bin/env python3
"""E-GMD helpers, standard library only; the README says how a list is drawn.
extract PART DST CRC SIZE         inflate one zip member fetched by byte range (its local header and data) to DST,
                                  checking CRC-32 and size (fetch.sh's)
status ROOT LIST                  files on disk per batch of LIST (egmd/batches.tsv)
list ROOT SPLIT HOURS OUT [SEED]  a `corpus <dir> from` list of SPLIT's recordings under ROOT, HOURS of them or all
"""
import csv
import os
import random
import struct
import sys
import zlib
from collections import Counter, defaultdict

LOCAL_HEADER = 0x04034B50


def extract(part: str, dst: str, crc: int, size: int) -> int:
    with open(part, "rb") as f:
        raw = f.read()
    sig, _, flag, method, _, _, _, csize, _, nlen, xlen = struct.unpack("<IHHHHHIIIHH", raw[:30])
    if sig != LOCAL_HEADER or flag & 1 or method not in (0, 8):
        print(f"{part}: not a plain zip member (sig {sig:#x}, flag {flag}, method {method})", file=sys.stderr)
        os.remove(part)
        return 1
    body = raw[30 + nlen + xlen:]
    if csize != 0xFFFFFFFF:
        body = body[:csize]
    try:
        data = zlib.decompress(body, -15) if method == 8 else body
    except zlib.error:
        data = b""
    if zlib.crc32(data) != crc or len(data) != size:
        print(f"{part}: CRC or size mismatch, fetched again on the next run", file=sys.stderr)
        os.remove(part)
        return 1
    tmp = dst + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dst)
    os.remove(part)
    return 0


def status(root: str, listing: str) -> int:
    have: dict[str, list[float]] = defaultdict(lambda: [0, 0, 0.0, 0.0])
    with open(listing) as f:
        for line in f:
            batch, off, end, _, usz, path = line.rstrip("\n").split("\t")
            row = have[batch]
            row[1] += 1
            row[3] += (int(end) - int(off) + 1) / 1e9
            try:
                if os.path.getsize(os.path.join(root, "e-gmd-v1.0.0", path)) == int(usz):
                    row[0] += 1
                    row[2] += (int(end) - int(off) + 1) / 1e9
            except OSError:
                pass
    for batch, (done, total, got, gb) in sorted(have.items()):
        print(f"{batch:4s} {done:5d}/{total:<5d} files  {got:6.2f}/{gb:.2f} GB")
    return 0


def listing(root: str, split: str, hours: str, out: str, seed: int) -> int:
    with open(os.path.join(root, "e-gmd-v1.0.0.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    splits: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        splits[r["id"]].add(r["split"])
    if both := sorted(p for p, s in splits.items() if len(s) > 1):
        print(f"{len(both)} performance(s) are in two splits, as {both[0]}: no list is written", file=sys.stderr)
        return 1
    there = lambda r: all(os.path.exists(os.path.join(root, r[k])) for k in ("audio_filename", "midi_filename"))
    pool = [r for r in rows if r["split"] == split and there(r)]
    if not pool:
        print(f"{root} holds no recording of the {split} split with its MIDI file", file=sys.stderr)
        return 1
    rng = random.Random(seed)
    sec = lambda r: float(r["duration"])
    of: dict[str, list[dict[str, str]]] = defaultdict(list)  # a performance's recordings, a kit each
    for r in pool:
        of[r["id"]].append(r)
    performances = len(of)
    bpms = sorted(int(rs[0]["bpm"]) for rs in of.values())
    edges = [bpms[len(bpms) * k // 4] for k in (1, 2, 3)]  # the performances' tempos in quarters
    band = lambda r: sum(int(r["bpm"]) >= e for e in edges)
    genre = lambda r: r["style"].split("/")[0]
    strata: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    for p in sorted(of):
        rng.shuffle(of[p])
        strata[(of[p][0]["drummer"], genre(of[p][0]), band(of[p][0]))].append(p)
    keys = sorted(strata)
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(strata[k])
    order = [strata[k][i] for i in range(max(map(len, strata.values()))) for k in keys if i < len(strata[k])]
    want = float("inf") if hours == "all" else float(hours) * 3600
    drawn: list[dict[str, str]] = []
    total, full, kits = 0.0, False, Counter({r["kit_name"]: 0.0 for r in pool})
    while not full and any(of.values()):  # a pass gives each performance one kit more, the kit the draw holds least of
        for p in order:
            if not of[p]:
                continue
            r = min(of[p], key=lambda r: kits[r["kit_name"]])
            if full := full or total + sec(r) > want:
                continue
            of[p].remove(r)
            drawn.append(r)
            total, kits[r["kit_name"]] = total + sec(r), kits[r["kit_name"]] + sec(r)
    here = os.path.dirname(os.path.abspath(out))
    path = lambda name: os.path.relpath(os.path.join(os.path.abspath(root), name), here)
    kept = ("drummer", "session", "id", "style", "bpm", "beat_type", "time_signature", "kit_name")
    with open(out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(("audio_filename", "midi_filename", "split", "duration", "beats", "velocity") + kept)
        for r in drawn:
            midi = path(r["midi_filename"])
            w.writerow([path(r["audio_filename"]), midi, r["split"], r["duration"], midi, "none"] + [r[k] for k in kept])
    hrs = lambda rs: sum(map(sec, rs)) / 3600

    def spread(name: str, key, names=None) -> None:
        got, had = defaultdict(list), defaultdict(list)
        for r in drawn:
            got[key(r)].append(r)
        for r in pool:
            had[key(r)].append(r)
        print(f"by {name}, hours drawn of those held: " + ", ".join(f"{names[k] if names else k} {hrs(got[k]):.2f}/{hrs(had[k]):.2f}" for k in sorted(had)))

    print(f"{out}: {len(drawn)} of the {len(pool)} recordings of the {split} split on disk, {hrs(drawn):.2f} of {hrs(pool):.2f} hours, seed {seed}")
    spread("drummer", lambda r: int(r["drummer"].removeprefix("drummer")), {int(d.removeprefix("drummer")): d for d in {r["drummer"] for r in pool}})
    spread("style", genre)
    bounds = [bpms[0]] + edges + [bpms[-1] + 1]
    spread("tempo", band, [f"{bounds[k]}-{bounds[k + 1] - 1} bpm" for k in range(4)])
    held = sorted(kits.values())
    print(f"by kit: {sum(v > 0 for v in held)} of {len(held)} kits drawn, from {held[0] / 3600:.2f} to {held[-1] / 3600:.2f} hours a kit")
    per = Counter(Counter(r["id"] for r in drawn).values())
    print(f"by performance: {sum(per.values())} of {performances} drawn, " + ", ".join(f"{n} on {k} kit(s)" for k, n in sorted(per.items())))
    return 0


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["extract", part, dst, crc, size]:
            sys.exit(extract(part, dst, int(crc), int(size)))
        case ["status", root, batches]:
            sys.exit(status(root, batches))
        case ["list", root, split, hours, out, *seed] if len(seed) < 2:
            sys.exit(listing(root, split, hours, out, int(seed[0]) if seed else 1))
        case _:
            print(__doc__, file=sys.stderr)
            sys.exit(2)
