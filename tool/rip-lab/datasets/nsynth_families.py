#!/usr/bin/env python3
"""nsynth_families.py <audio dir> <out dir> <seed> <family>=<count|all>...: a folder for `corpus ... sampled <out dir>`
holding the notes of some of the instruments in <audio dir>, the `audio` folder of one of NSynth's splits, by family
(an instrument's name up to its source, as NSynth names it: bass, ..., synth_lead, vocal): all of a family's
instruments, or a seeded draw of <count> of them, each with every note it has, by hard link. A line a family, then
what the folder holds."""
import os
import random
import sys
from collections import defaultdict


def instrument(note: str) -> tuple[str, str] | None:
    """A note's instrument and family, by NSynth's name for it: family_source_NNN-pitch-velocity.wav"""
    parts = note.removesuffix(".wav").rsplit("-", 2)
    if not note.endswith(".wav") or len(parts) != 3 or not (parts[1].isdigit() and parts[2].isdigit()):
        return None
    fam = parts[0].rsplit("_", 2)  # from the right: synth_lead's name holds an underscore
    return (parts[0], fam[0]) if len(fam) == 3 else None


def link(audio: str, out: str, seed: int, specs: list[str]) -> int:
    notes: dict[str, list[str]] = defaultdict(list)
    family: dict[str, str] = {}
    for f in os.listdir(audio):
        if named := instrument(f):
            notes[named[0]].append(f)
            family[named[0]] = named[1]
    asked = []
    for spec in specs:
        fam, _, n = spec.partition("=")
        inst = sorted(i for i in notes if family[i] == fam)
        if not inst:
            print(f"{spec}: {audio} holds no {fam!r}; it holds {', '.join(sorted(set(family.values())))}", file=sys.stderr)
            return 2
        if n != "all" and not (n.isdigit() and int(n) > 0):
            print(f"{spec}: want <family>=<count> or <family>=all", file=sys.stderr)
            return 2
        asked.append((fam, n, inst))
    # one stream over the families in the order named, so a draw is its seed's and its place's
    rng = random.Random(seed)
    os.makedirs(out, exist_ok=True)
    for fam, n, inst in asked:
        take = inst if n == "all" else rng.sample(inst, min(int(n), len(inst)))
        for i in take:
            for f in notes[i]:
                if not os.path.exists(d := os.path.join(out, f)):
                    os.link(os.path.join(audio, f), d)
        print(f"{fam}: {len(take)} of {len(inst)} instruments, {sum(len(notes[i]) for i in take)} notes")
    held: dict[str, set[str]] = defaultdict(set)
    count = 0
    for f in os.listdir(out):
        if named := instrument(f):
            held[named[1]].add(named[0])
            count += 1
    print(f"{out}: {sum(map(len, held.values()))} instruments in {len(held)} families, {count} notes")
    return 0


if __name__ == "__main__":
    match sys.argv[1:]:
        case [audio, out, seed, *specs] if specs and seed.isdigit():
            sys.exit(link(audio, out, int(seed), specs))
        case _:
            print(__doc__, file=sys.stderr)
            sys.exit(2)
