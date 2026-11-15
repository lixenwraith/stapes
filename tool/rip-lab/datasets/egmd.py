#!/usr/bin/env python3
"""E-GMD helpers for fetch.sh, standard library only.
extract PART DST CRC SIZE  inflate one zip member fetched by byte range (its local
                           header and data) to DST, checking CRC-32 and size
status ROOT LIST           files on disk per batch of LIST (egmd/batches.tsv)
"""
import os
import struct
import sys
import zlib
from collections import defaultdict

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


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["extract", part, dst, crc, size]:
            sys.exit(extract(part, dst, int(crc), int(size)))
        case ["status", root, listing]:
            sys.exit(status(root, listing))
        case _:
            print(__doc__, file=sys.stderr)
            sys.exit(2)
