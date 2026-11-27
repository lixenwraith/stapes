"""Where the rip stands after a fit: the read-back's logs as one table, a kind of
sound a block and a set a row, what fell under a fitted set said first, and with
it the notes a minute heard where none plays where they rose.

    python3 table.py fit/logs/readback-*.log
"""
from __future__ import annotations

import os
import re
import sys

# a column, and how a condition's line of `tune ... truth` gives its figure before any search
READINGS = (
    ("notes", r"F1 ([0-9.]+) before"),
    ("ends", r"ends ([0-9.]+) before, [0-9.]+ after, of (\d+) note"),
    ("loudness", r"loudness ([0-9.]+) before"),
    ("drums", r"drums F1 ([0-9.]+) before"),
    ("struck", r"struck F1 ([0-9.]+) before"),
    ("by line", r"struck F1 .*?, the lines' hits ([0-9.]+),"),
    ("bars", r"bars ([0-9.]+) before"),
    ("parts", r"parts ([0-9.]+) before"),
    ("key", r"key ([0-9.]+) before"),
    ("chords", r"chords ([0-9.]+) before"),
)
WRONG = r"ends (\d+) wrong"
QUIET = r"struck ([0-9.]+) strike\(s\) a minute before, [0-9.]+ after, the lines' hits ([0-9.]+),"  # where none plays
NOTELESS = r"notes ([0-9.]+) a minute before, [0-9.]+ after, heard where none plays"
# the table's columns and their widths: the readings, the ends right in notes after the ends' share
COLUMNS = (("notes", 10), ("ends", 7), ("ends right", 15), ("loudness", 10), ("drums", 7), ("struck", 9), ("by line", 9), ("bars", 7), ("parts", 7), ("key", 7), ("chords", 8))
FIRST = ("models", "judge", "start")  # the rows no search made, in the order they stand
FELL = 0.005  # a fall under this is not said
ROSE = 0.5  # nor a rise of the notes a minute heard where none plays under this


def read(path: str) -> dict[str, dict[str, float]]:
    """each kind's figures in one log, in the order the log names the kinds"""
    kinds: dict[str, dict[str, float]] = {}
    with open(path) as f:
        for line in f:
            kind, colon, rest = line.rstrip("\n").partition(": ")
            if not colon:
                continue
            if m := re.match(WRONG, rest):
                kinds.setdefault(kind, {})["wrong"] = int(m.group(1))
                continue
            if m := re.match(QUIET, rest):
                kinds.setdefault(kind, {}).update({"struck a minute": float(m.group(1)), "by line a minute": float(m.group(2))})
                continue
            if m := re.match(NOTELESS, rest):
                kinds.setdefault(kind, {})["notes a minute"] = float(m.group(1))
                continue
            for column, pattern in READINGS:
                if m := re.match(pattern, rest):
                    row = kinds.setdefault(kind, {})
                    row[column] = float(m.group(1))
                    if column == "ends":
                        row["matched"] = int(m.group(2))
    return kinds


def table(paths: list[str]) -> str:
    sets = {os.path.basename(p).removeprefix("readback-").removesuffix(".log"): read(p) for p in paths}
    order = [s for s in FIRST if s in sets] + sorted(s for s in sets if s not in FIRST)
    if not order:
        return "no read-back to table\n"
    base = next((s for s in reversed(FIRST) if s in sets), order[0])
    kinds = list(dict.fromkeys(k for s in order for k in sets[s]))
    out = []
    for s in order:
        if s in FIRST:
            continue
        falls = [f"{c} on {k} {sets[base][k][c]:.3f} to {sets[s][k][c]:.3f}" for k in kinds for c, _ in READINGS
                 if c in sets[s].get(k, {}) and c in sets[base].get(k, {}) and sets[base][k][c] - sets[s][k][c] > FELL]
        falls += [f"notes where none plays on {k} {sets[base][k][c]:.1f} to {sets[s][k][c]:.1f} a minute" for k in kinds for c in ("notes a minute",)
                  if c in sets[s].get(k, {}) and c in sets[base].get(k, {}) and sets[s][k][c] - sets[base][k][c] > ROSE]
        out.append(f"fell under {s}, against {base}: {', '.join(falls)}" if falls else f"nothing fell under {s}, against {base}")
    width = max(len(x) for x in kinds + order + ["kind"]) + 2
    out.append("".join(h.ljust(width) for h in ("kind", "set")) + "".join(h.rjust(w) for h, w in COLUMNS))
    for k in kinds:
        for s in order:
            row = sets[s].get(k)
            if row is None:
                continue
            right = f"{row['matched'] - row['wrong']} of {row['matched']}" if "wrong" in row and "matched" in row else "-"
            cells = [f"{row[c]:.3f}" if c in row else f"{row[c + ' a minute']:.1f}/min" if c + " a minute" in row else "-" for c, _ in READINGS]
            cells.insert(2, right)
            out.append(k.ljust(width) + s.ljust(width) + "".join(c.rjust(w) for c, (_, w) in zip(cells, COLUMNS)))
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.stdout.write(table(sys.argv[1:]))
