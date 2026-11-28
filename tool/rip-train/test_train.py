"""The trainer's rules, each a test: python3 -m unittest discover tool/rip-train"""
import array
import contextlib
import hashlib
import io
import itertools
import json
import math
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from unittest import mock

import numpy as np
import torch

import average
import datacheck
import ripmodels
import table
import train

# a corpus of the frames the models hear: its manifest states their layout, and its labels hold every note
MANIFEST = {"layout": ripmodels.LAYOUT, "voice_genes": ripmodels.VOICE_GENES}
PLACED_STARTS = ("stretch", "starts", "pitchstretch", "pitchstarts")  # the targets only a placed start sets
F = {k: i for i, k in enumerate(ripmodels.FAMILIES)}
LOW = ripmodels.LOW  # the piano's lowest key, a label's first cell
CFG = ripmodels.Config(inputs=ripmodels.STACK_BINS + 1, channels=ripmodels.STACK_CHANNELS, outputs=ripmodels.PITCHNET_HEADS, cell=4)


def voice(first: float) -> list:
    """an engine voice's genes, its first gene telling it from another"""
    return [first] + [0.5] * (ripmodels.VOICE_GENES - 1)


def note(frame, end, midi, family, genes=None, wave=-1, **more) -> dict:
    return {"frame": frame, "end": end, "drum": -1, "midi": midi, "vel": 0.8, "family": F[family], "wave": wave, **({"voice": genes} if genes else {}), **more}


def label(events, frames=40, reach=None, **more) -> train.Label:
    """a loop's label, its corpus stating reach for its placed starts"""
    return train.Label({"events": events, **more}, frames, MANIFEST if reach is None else {**MANIFEST, "reach": reach})


def heads_before(head: str) -> list:
    """the frame model's heads before head, as a model trained before it was added has them"""
    return ripmodels.HEADS[:[k for k, _ in ripmodels.HEADS].index(head)]


def hit(frame, drum=-1, vel=0.8, **more) -> dict:
    """something struck at frame: a hit of drum's line, or with drum -1 a strike of no line"""
    return {"frame": frame, "end": frame, "drum": drum, "midi": 0, "vel": vel, "wave": -1, "family": F["drums"], **more}


def rendered(events, frames, bars) -> train.Label:
    """a rendered loop's label, bars of 4/4 held at 100 bpm from frame 0, its beats and bar lines as corpus lists them"""
    beats = [int(k * 51.68 + 0.5) for k in range(bars * 4)]
    return label(events, frames, bpm=100, bars=bars, beats=beats, downbeats=beats[::4])


class Frames:
    """a loop's frames in memory, as train.FrameFile gives them, under a path no other takes, as train.power caches by
    it: an id is taken again once its object is gone"""
    made = itertools.count()

    def __init__(self, n):
        self.path, self.fr = f"memory{next(Frames.made)}", np.zeros((n, ripmodels.VALUES), np.float32)

    def __len__(self):
        return len(self.fr)

    def __getitem__(self, sl):
        return self.fr[sl]


class Said(torch.nn.Module):
    """a notes model that says what a test wrote into its scores"""

    def __init__(self, frames):
        super().__init__()
        self.cfg, self.y = CFG, torch.full((1, frames, CFG.out_size), -9.0)
        self.y[..., CFG.slice("pitchvoice")] = 0.0  # every gene a half
        self.register_buffer("sure", torch.zeros(()))

    def say(self, a, b, midi, salience=9.0, family=None, wave=None, gene=None):
        """over frames a to b the pitch sounds, played by a family on a wave, its voice's first gene given"""
        p = midi - LOW
        self.y[0, a:b, self.cfg.slice("pitch")][:, p] = salience
        for head, value, score in (("pitchfamily", family and F[family], 9.0), ("pitchwave", wave, 9.0), ("pitchvoice", 0, gene)):
            if value is not None and score is not None:
                self.y[0, a:b, self.cfg.slice(head)][:, value * ripmodels.PITCHES + p] = score

    def forward(self, x):
        return self.y[:, :x.shape[1]]


class WhatPlaysEachPitch(unittest.TestCase):
    def tearDown(self):
        train.FAMILY_HOLDOUT = None

    def test_a_cell_says_what_plays_it_where_one_note_alone_sounds(self):
        """A cell one note sounds in is asked what plays it: a recorded note its family and no wave or voice, a note
        the engine rendered its wave and voice and no family, registered or drawn far from the bank, a recorded
        note of no family nothing; a note that moves in each cell it sounds in; and a cell two notes sound in
        nothing, though both sound."""
        bend = [0.0] * 5 + [100.0] * 5
        lab = label([note(0, 10, 60, "piano", voice(0.1), 4), note(0, 10, 64, "strings"), note(0, 10, 67, "unknown", voice(0.2), 1),
                     note(0, 10, 70, "unknown"), note(0, 10, 72, "piano", voice(0.1), 4), note(5, 15, 72, "guitar", voice(0.3), 3),
                     note(20, 30, 50, "bass", voice(0.4), 4, cents=bend)])
        t = lab.window(0, 40)
        cell = lambda key, frame, midi: int(t[key][frame, midi - LOW])
        rows = train.genes()
        for frame, midi, family, wave, first in ((3, 60, None, 4, 0.1), (3, 64, "strings", -1, None), (3, 67, None, 1, 0.2), (3, 70, None, -1, None),
                                                 (3, 72, None, 4, 0.1), (7, 72, None, -1, None), (12, 72, None, 3, 0.3),
                                                 (22, 50, None, 4, 0.4), (27, 51, None, 4, 0.4), (27, 50, None, -1, None)):
            with self.subTest(frame=frame, midi=midi):
                self.assertEqual(cell("pitchfamily", frame, midi), F[family] if family else -1)
                self.assertEqual(cell("pitchwave", frame, midi), wave)
                got = cell("pitchvoice", frame, midi)
                self.assertEqual(None if got < 0 else round(float(rows[got][0]), 3), first)
        self.assertEqual([int(t["pitch"][f, m - LOW]) for f, m in ((3, 70), (7, 72), (27, 51), (27, 50))], [1, 1, 1, 0])

    def test_a_family_held_out_is_named_to_no_head(self):
        """The notes of the family held out stay in the run, their sound, wave and voice taught, but neither the
        cells' head nor the frame's is told their family."""
        train.FAMILY_HOLDOUT = F["guitar"]
        t = label([note(0, 10, 60, "guitar"), note(0, 10, 64, "guitar", voice(0.3), 3), note(0, 20, 48, "bass")]).window(0, 20)
        self.assertEqual([int(t["pitchfamily"][3, m - LOW]) for m in (60, 48)], [-1, F["bass"]])
        self.assertEqual([int(t["family"][f]) for f in (3, 15)], [-1, F["bass"]])
        self.assertEqual([int(t[k][3, 64 - LOW]) for k in ("pitch", "pitchwave")], [1, 3])
        self.assertGreaterEqual(int(t["pitchvoice"][3, 64 - LOW]), 0)

    def test_a_frame_is_asked_a_family_only_where_an_instrument_plays_its_highest_note(self):
        """The frame's family head is asked by the cells' rule: a frame whose highest note the engine rendered asks no
        family, one whose highest note an instrument played asks its family."""
        t = label([note(0, 10, 72, "guitar", voice(0.3), 3), note(0, 20, 60, "piano")]).window(0, 20)
        self.assertEqual([int(t["family"][f]) for f in (3, 15)], [-1, F["piano"]])

    def test_a_note_is_heard_as_its_cells_say_by_their_salience(self):
        """A note is named by its cells' say, each by its salience, over the cells among its frame's VOICE_CELLS most
        salient; the lines give the share named right, by family and under another note, the waves' and the genes'
        miss; and the surest name SURE of a held-out family's notes stay under is the measure. The names are read on
        recorded notes and the waves and genes on the engine's, a loop each, the model saying the same of both."""
        train.FAMILY_HOLDOUT = F["guitar"]
        events = [note(0, 20, 72, "piano"), note(0, 20, 40, "bass"), note(20, 40, 60, "voice")]
        events += [note(40 + 10 * i, 50 + 10 * i, 55, "guitar") for i in range(21)]
        engine = [note(0, 20, 72, "piano", voice(0.2), 4), note(0, 20, 40, "bass", voice(0.6), 3), note(20, 40, 60, "voice", voice(0.8), 1)]
        frames = 256  # one window, the model's scores its frames
        model = Said(frames)
        model.say(0, 10, 72, 9.0, "piano", 4, 9.0)  # salience and each score near one: the gene a whole
        model.say(10, 20, 72, -1.0, "synth", 1, -9.0)  # a quarter as salient: three parts piano to one synth
        model.say(0, 20, 40, 9.0, "piano", 3, 0.0)  # the bass, under the piano, named wrong on its own wave
        model.say(30, 40, 60, 2.0, "voice", 1, 9.0)
        model.say(20, 30, 60, 5.0, "synth", 0, -9.0)  # louder, under thirteen louder pitches no note sounds in: not among the cells kept
        for m in range(80, 93):
            model.say(20, 30, m, 5.0 + m)
        for i in range(21):  # the held-out family's notes, each surer of a piano than the last
            model.say(40 + 10 * i, 50 + 10 * i, 55, 9.0, "piano")
            model.y[0, 40 + 10 * i:50 + 10 * i, CFG.slice("pitchfamily")][:, F["synth"] * ripmodels.PITCHES + 55 - LOW] = 9.0 - 0.5 * i
        frs, tgs, notes = train.validation([(Frames(frames), label(events, frames)), (Frames(frames), label(engine, frames))], frames, 4)
        v = train.evaluate(model, frs, tgs, "cpu", 24, notes)["voices"]
        self.assertEqual((round(v["families"][0], 6), v["families"][1]), (round(2 / 3, 6), 3))
        self.assertEqual(v["by"], [("piano", 1.0, 1), ("voice", 1.0, 1), ("bass", 0.0, 1)])
        self.assertEqual(v["under"], (0.0, 1))
        self.assertAlmostEqual(v["waves"], 1.0)
        # the piano's gene 0.2 heard as 0.79, the bass's 0.6 as 0.5, the voice's 0.8 as 1, each over the genes' RMS
        salience = torch.sigmoid(torch.tensor([9.0, -1.0]))
        piano = float((salience[0] * 1.0 + salience[1] * 0.0) / salience.sum())
        want = np.mean([abs(piano - 0.2), 0.1, 0.2]) / np.sqrt(ripmodels.VOICE_GENES)
        self.assertAlmostEqual(v["genes"], want, places=3)
        count, reach, median, past, right = v["held"]
        shares = sorted(1 / (1 + np.exp(-0.5 * i)) for i in range(21))  # a piano's share against the synth's, the rest nothing
        self.assertEqual(count, 21)
        self.assertAlmostEqual(reach, float(np.quantile(shares, train.SURE)), places=3)
        self.assertAlmostEqual(median, float(np.median(shares)), places=3)
        self.assertEqual((round(past, 6), right), (round(2 / 3, 6), 0.5))  # the voice and the bass, named wrong, are surer than they reach

    def test_a_line_reads_the_names_loss_a_cell_beside_their_shares_and_the_notes_with_the_shares_divided_out(self):
        """Under the names' prior, each family's share of the cells fixed windows ask a name of, a line gives the names'
        loss as a mean over the cells asked, a window that asks none adding nothing, beside the loss of the shares
        alone, their entropy on the cells they were counted on; and the share of the notes named right with each
        name's say over its share, where the commoner name outvoted a note's own."""
        frames = 64
        model = Said(frames)
        model.say(0, 30, 72, 9.0, "piano")
        model.say(0, 10, 60, 9.0)  # a guitar's note the cells give 0.73 to the piano and 0.27 to the guitar
        for family, score in (("piano", 1.0), ("guitar", 0.0)):
            model.y[0, 0:10, CFG.slice("pitchfamily")][:, F[family] * ripmodels.PITCHES + 60 - LOW] = score
        recorded = label([note(0, 30, 72, "piano"), note(0, 10, 60, "guitar")], frames)
        engine = label([note(0, 30, 50, "piano", voice(0.2), 4)], frames)  # asks no name
        frs, tgs, notes = train.validation([(Frames(frames), recorded), (Frames(frames), engine)], frames, 4)
        prior = train.shares(tgs)
        self.assertEqual([round(float(x), 6) for x in prior[[F["piano"], F["guitar"]]]], [0.75, 0.25])
        heard = train.evaluate(model, frs, tgs, "cpu", 24, notes, prior=prior)
        v, guitar = heard["voices"], -math.log(1 / (1 + math.e))
        self.assertAlmostEqual(v["cells"][0], guitar / 4, places=4)
        self.assertAlmostEqual(heard["pitchfamily"], guitar / 8, places=4)  # the loss among the parts, a mean over windows
        self.assertAlmostEqual(v["cells"][1], -(0.75 * math.log(0.75) + 0.25 * math.log(0.25)), places=6)
        self.assertEqual((v["families"], v["by"]), ((0.5, 2), [("piano", 1.0, 1), ("guitar", 0.0, 1)]))
        self.assertEqual(v["even"], (1.0, [("piano", 1.0), ("guitar", 1.0)]))

    def test_the_weight_file_carries_the_heads_and_the_measure(self):
        """The file names the heads of what plays each pitch after the cents, a value a pitch each, states the layout
        of the frames it hears as a constant after the heads, and carries the measure as a frame head whose weights
        are zero and whose bias is the measure."""
        model = ripmodels.PitchNet(CFG)
        model.sure.fill_(0.75)
        with tempfile.TemporaryDirectory() as d:
            ripmodels.export(model, os.path.join(d, "notes.bin"))
            heads, constants, tensors = exported(os.path.join(d, "notes.bin"), 4 + 16 + len(ripmodels._cells_header(model)))
        self.assertEqual(constants, [("layout", float(ripmodels.LAYOUT))])
        p = ripmodels.PITCHES
        self.assertEqual(heads[4:8], [("cents", 2 * p), ("pitchfamily", ripmodels.NAMES * p), ("pitchwave", ripmodels.WAVES * p), ("pitchvoice", ripmodels.VOICE_GENES * p)])
        self.assertEqual(heads[-1], ("sure", 1))
        frame = sum(n for k, n in heads[8:])
        self.assertEqual(tensors["frameout.weight"].shape, (frame, model.frame.in_features))
        self.assertEqual((float(np.abs(tensors["frameout.weight"][-1]).max()), float(tensors["frameout.bias"][-1])), (0.0, 0.75))
        self.assertEqual(tensors["cellout.weight"].shape[0], sum(n for _, n in heads[:8]) // p)

    def test_a_merge_refuses_files_that_disagree_on_having_a_measure(self):
        """A file that never measured holds zero, which is no measure: files that all carry one merge to their mean,
        files of which only some do are refused by name, since the mean would be no run's, --sure gives such a merge
        its measure, and a merge of files with none carries none."""
        with tempfile.TemporaryDirectory() as d:
            files, out = [os.path.join(d, name) for name in ("held.pt", "more.pt", "none.pt")], os.path.join(d, "out.bin")
            for file, sure in zip(files, (0.2, 0.4, 0.0)):
                model = ripmodels.PitchNet(CFG)
                model.sure.fill_(sure)
                torch.save({"model": model.state_dict(), "layout": ripmodels.LAYOUT, "cfg": asdict(CFG)}, file)

            def merged(*argv) -> float:
                with mock.patch.object(sys, "argv", ["average.py", "--out", out, *argv]), contextlib.redirect_stdout(io.StringIO()):
                    average.main()
                with open(out, "rb") as f:
                    return struct.unpack("<f", f.read()[-12:-8])[0]  # the last tensor's last value, under the checksum
            self.assertAlmostEqual(merged(*files[:2]), 0.3, places=6)
            with self.assertRaises(SystemExit) as refused:
                merged(*files)
            self.assertTrue(all(file in str(refused.exception) for file in files), refused.exception)
            self.assertAlmostEqual(merged("--sure", "0.25", *files), 0.25, places=6)
            self.assertEqual(merged(files[2]), 0.0)


def corpus(d, labels, frames=64, sound=None, **manifest):
    """a corpus on disk of loops of silence, or of the frames sound gives a label, a label each; manifest, what its
    manifest states over MANIFEST"""
    with open(os.path.join(d, "corpus.json"), "w") as f:
        json.dump({**MANIFEST, "loops": len(labels), "waves": ["sine", "saw", "square", "string", "modal"], **manifest}, f)
    for i, lab in enumerate(labels):
        with open(os.path.join(d, f"loop_{i:03d}.bands"), "wb") as f:
            f.write(struct.pack("<II", frames, ripmodels.VALUES) + (sound(lab).tobytes() if sound else bytes(4 * frames * ripmodels.VALUES)))
        with open(os.path.join(d, f"loop_{i:03d}.json"), "w") as f:
            json.dump(lab, f)


def run(*flags, steps=1) -> str:
    """steps of train.py at the smallest shape unless flags name another, and what it printed; on the CPU whatever
    the machine has, as every tensor a test builds is there"""
    argv = ["train.py", "--steps", str(steps), "--batch", "1", "--window", "32", "--dim", "8", "--heads", "2", "--layers", "1", "--ffn", "16", "--attention-window", "4", "--device", "cpu", *flags]
    with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()) as said:
        train.main()
    return said.getvalue()


def exported(path: str, shape: int) -> tuple:
    """a weight file's heads, constants and tensors, after the bytes that state its model's shape"""
    with open(path, "rb") as f:
        body = f.read()
    at = shape

    def take(fmt):
        nonlocal at
        v = struct.unpack_from("<" + fmt, body, at)
        at += struct.calcsize("<" + fmt)
        return v if len(v) > 1 else v[0]

    def name():
        nonlocal at
        n = take("H")
        at += n
        return body[at - n:at].decode()

    heads, constants, tensors = [(name(), take("I")) for _ in range(take("I"))], [(name(), take("f")) for _ in range(take("I"))], {}
    for _ in range(take("I")):
        k, dims = name(), [take("I") for _ in range(take("B"))]
        tensors[k] = np.frombuffer(body, "<f4", int(np.prod(dims)), at).reshape(dims)
        at += 4 * int(np.prod(dims))
    return heads, constants, tensors


def handed(*flags, steps=1) -> list:
    """the batches a run hands its device, a step's each: its frames and its targets; the run numbers its voices'
    genes from none, as a process of its own does, since a cell's pitchvoice is a row of them"""
    seen, real = [], train.losses

    def losses(model, x, t):
        if torch.is_grad_enabled():  # a step's, not a score's
            seen.append((x, t))
        return real(model, x, t)
    with mock.patch.object(train, "losses", losses), mock.patch.object(train, "GENES", array.array("f")):
        run(*flags, steps=steps)
    return seen


def same(a: list, b: list) -> bool:
    """whether two runs' batches are equal tensor for tensor"""
    return len(a) == len(b) and all(torch.equal(x, y) and t.keys() == u.keys() and all(torch.equal(t[k], u[k]) for k in t) for (x, t), (y, u) in zip(a, b))


def marks(frames, at) -> list:
    """a target of frames frames marked a frame either side of each frame in at"""
    out = np.zeros(frames, np.uint8)
    for f in at:
        out[max(f - 1, 0):f + 2] = 1
    return out.tolist()


class Heard(torch.nn.Module):
    """a frame model that says what a test wrote into its scores"""

    def __init__(self, frames):
        super().__init__()
        self.cfg = ripmodels.Config()
        self.y = torch.full((1, frames, self.cfg.out_size), -9.0)

    def forward(self, x):
        return self.y[:, :x.shape[1]]


class TheBeatAndTheBarLine(unittest.TestCase):
    def test_a_rendered_loop_is_asked_the_beats_and_lines_its_label_lists(self):
        """A rendered loop's beats and bar lines are those its label lists, wherever its tempo and its start put them,
        a bar line past frame 0 after a pickup, each marked a frame either side; it is asked over all its frames but
        the last two, which a beat at its end, unlisted, would mark; a window takes its share of them."""
        frames, beats, lines = 365, [0, 51, 103, 156, 208, 261, 313], [51, 208]  # 3/4 from a beat's pickup, slowing
        lab = label([], frames, bpm=100, bars=2, meter="3/4", beats=beats[::-1], downbeats=lines)
        t = lab.window(0, frames)
        self.assertEqual(t["beat"].tolist(), marks(frames, beats))
        self.assertEqual(t["downbeat"].tolist(), marks(frames, lines))
        self.assertEqual(t["metered"].tolist(), [1] * (frames - 2) + [0] * 2)
        part = lab.window(52, 155)
        for k in ("beat", "downbeat", "metered"):
            self.assertEqual(part[k].tolist(), t[k][52:155].tolist())

    def test_a_recording_is_asked_the_beats_its_label_lists(self):
        """A recording's beats and bar lines are those its label lists, as its list's beats file gave them, in any
        order, asked from its first beat's mark to its last's and nowhere else; one whose label lists none is asked
        nothing."""
        listed = label([note(0, 10, 60, "guitar")], 200, source="a.wav", beats=[155, 26, 69, 112], downbeats=[69]).window(0, 200)
        self.assertEqual(listed["beat"].tolist(), marks(200, [26, 69, 112, 155]))
        self.assertEqual(listed["downbeat"].tolist(), marks(200, [69]))
        self.assertEqual(listed["metered"].tolist(), [0] * 25 + [1] * (157 - 25) + [0] * (200 - 157))
        none = label([note(0, 10, 60, "guitar")], 200, source="b.wav").window(0, 200)
        self.assertEqual([int(none[k].sum()) for k in ("beat", "downbeat", "metered")], [0, 0, 0])

    def test_a_bent_window_sways_its_beats_with_its_hits(self):
        """A window read along a tempo that sways takes the beat and the bar line nearest along its curve, as it
        takes every target: each stays on the hit it fell on, and the beats stand no steady grid apart."""
        frames, kick, hat = 1654, 0, 2  # eight bars of 4/4 at 100 bpm, a hat on every beat and a kick on every line
        grid = rendered([], frames, 8)
        hits = [{"frame": f, "end": f, "drum": hat, "midi": 0, "vel": 0.8, "wave": -1, "family": F["drums"]} for f in grid.beats.tolist()]
        hits += [{**h, "drum": kick} for h in hits[::4]]
        _, t = train.bent(Frames(frames), rendered(hits, frames, 8), 512, 2, None, random.Random(2), None)
        asked = t["metered"] > 0
        self.assertGreater(int(asked.sum()), 400)
        self.assertEqual(t["beat"][asked].tolist(), t["drums"][asked, hat].tolist())
        self.assertEqual(t["downbeat"][asked].tolist(), t["drums"][asked, kick].tolist())
        apart = np.diff(np.flatnonzero(np.diff(t["beat"].astype(int)) == 1))  # from one mark's first frame to the next's
        self.assertFalse(set(apart.tolist()) <= {51, 52}, apart)

    def test_a_window_with_no_grid_of_its_own_is_asked_no_beat(self):
        """A window spliced of two clips and one real drums are laid under hold hits of another grid, and the padding
        past a loop's end holds none: each asks the beat and the bar line nothing, where a plain window asks its loop's."""
        frames, rng = 414, random.Random(1)  # two bars of 4/4 at 100 bpm
        fr, lab = Frames(frames), rendered([{"frame": 52, "end": 52, "drum": 0, "midi": 0, "vel": 0.8, "wave": -1, "family": F["drums"]}], frames, 2)
        _, spliced = train.shuffled(lambda rng: (fr, lab), 256, 2, rng)
        _, laid = train.sample(fr, lab, 0, 256, 2, (fr, lab, 0.0), rng)
        _, plain = train.sample(fr, lab, 200, 256, 2)
        self.assertEqual([int(t["metered"].sum()) for t in (spliced, laid)], [0, 0])
        self.assertEqual(plain["metered"].tolist(), [1] * (412 - 200) + [0] * (256 - 212))

    def test_the_two_heads_are_taught_as_posteriors(self):
        """Each head's loss is cross-entropy over the frames asked, a beat weighing as a frame without one: a head
        that says everywhere the share of those frames a beat marks scores that share's entropy, and no lower saying
        more or less, which is what audio reads a posterior for. Both count whole in the total."""
        frames = 414  # two bars of 4/4 at 100 bpm in a window of 512: 412 frames asked, 23 in a beat's mark and 5 in a line's
        frs, tgs, _ = train.validation([(Frames(frames), rendered([], frames, 2))], 512, 2)
        model = Heard(512)

        def loss(beat, downbeat):
            model.y[..., model.cfg.slice("beat")], model.y[..., model.cfg.slice("downbeat")] = beat, downbeat
            return {k: v.item() for k, v in train.losses(model, frs[0], tgs[0]).items()}
        entropy = lambda p: -(p * np.log(p) + (1 - p) * np.log(1 - p))
        odds = lambda p: float(np.log(p / (1 - p)))
        at = loss(odds(23 / 412), odds(5 / 412))
        self.assertAlmostEqual(at["beat"], entropy(23 / 412), places=5)
        self.assertAlmostEqual(at["downbeat"], entropy(5 / 412), places=5)
        for nudge in (-0.2, 0.2):
            off = loss(odds(23 / 412) + nudge, odds(5 / 412) + nudge)
            self.assertGreater(off["beat"], at["beat"])
            self.assertGreater(off["downbeat"], at["downbeat"])
            self.assertAlmostEqual(off["total"] - at["total"], off["beat"] - at["beat"] + off["downbeat"] - at["downbeat"], places=5)

    def test_a_line_scores_each_heads_peaks_and_says_where_none_is_asked(self):
        """A held-out or validation line gives each head's F1 at its peaks over a half among the frames asked, a
        peak right within four frames of a beat the labels give, as the strikes' is. Over a corpus that lists no
        beat it says none is asked and gives neither head's loss, which over no frame would read as a perfect one."""
        frames = 414  # two bars of 4/4 at 100 bpm: beats at 0, 52, 103, 155, 207, 258, 310 and 362, lines at 0 and 207
        loop, recording = rendered([], frames, 2), label([note(0, 10, 60, "guitar")], frames, source="a.wav")
        model = Heard(512)
        beat, downbeat = model.cfg.slice("beat").start, model.cfg.slice("downbeat").start
        model.y[0, [0, 54, 103, 155, 180, 258, 310, 413], beat] = 9.0  # six beats heard, one two frames late; one that is none; one past the frames asked
        model.y[0, 362, beat] = -0.1  # and one under a half
        model.y[0, 207, downbeat] = 9.0
        v = train.evaluate(model, *train.validation([(Frames(frames), loop)], 512, 2)[:2], "cpu")
        self.assertAlmostEqual(v["beats"][0], 2 * (6 / 7) * (6 / 8) / (6 / 7 + 6 / 8))
        self.assertAlmostEqual(v["beats"][1], 2 * 1 * (1 / 2) / (1 + 1 / 2))
        none = train.evaluate(model, *train.validation([(Frames(frames), recording)], 512, 2)[:2], "cpu")
        self.assertEqual((none["beats"], "beat" in none, "downbeat" in none, "onset" in none), (None, False, False, True))
        with tempfile.TemporaryDirectory() as d:
            os.mkdir(os.path.join(d, "recordings"))
            corpus(d, [{"bpm": 100, "bars": 1, "beats": [0, 52], "downbeats": [0], "events": []}] * 2)
            corpus(os.path.join(d, "recordings"), [{"source": "a.wav", "events": [note(0, 10, 60, "guitar")]}])
            said = run("--corpus", d, "--out", os.path.join(d, "out.bin"), "--holdout", "1", "--validate", os.path.join(d, "recordings")).splitlines()
        held, validation = (next(line for line in said if line.startswith(f"step 1 {name} ")) for name in ("held-out", "validation"))
        self.assertRegex(held, r", beats' F1 \d\.\d{3}, bar lines' \d\.\d{3}, .*\(.* beat \d\.\d{3} downbeat \d\.\d{3}\)")
        self.assertIn(", no beat or bar line asked, ", validation)
        self.assertNotIn("beat ", re.search(r"\((onset [^)]*)\)", validation)[1])

    def test_init_grows_the_heads_and_carries_every_other_weight(self):
        """A frame model from before a head added later starts a run through --init: the run grows the heads after its
        own, every weight it had carried over, so after a step at a rate of 0 every output it had is the start's to the
        bit, and the weight file lists the heads in order; a notes model has none of what is struck."""
        x = torch.randn(1, 40, 2 * ripmodels.INPUTS, generator=torch.Generator().manual_seed(1))
        head = lambda name, size: struct.pack("<H", len(name)) + name.encode() + struct.pack("<I", size)
        layout = struct.pack("<IH", 1, 6) + b"layout" + struct.pack("<f", ripmodels.LAYOUT)
        for first, grew in (("beat", "beat, downbeat, struck, struckvel"), ("struck", "struck, struckvel")):
            with self.subTest(grown=grew), tempfile.TemporaryDirectory() as d:
                at = lambda name: os.path.join(d, name)
                old = ripmodels.Config(dim=8, heads=2, layers=1, ffn=16, window=4, outputs=heads_before(first))
                start = ripmodels.FrameModel(old).eval()
                corpus(d, [{"bpm": 120, "bars": 1, "events": [note(0, 10, 60, "piano"), hit(20, 0)]}])
                torch.save({"model": start.state_dict(), "layout": ripmodels.LAYOUT, "cfg": asdict(old)}, at("old.pt"))
                said = run("--corpus", d, "--out", at("new.bin"), "--init", at("old.pt"), "--checkpoint", at("new.pt"), "--lr", "0")
                after = torch.load(at("new.pt"))
                with open(at("new.bin"), "rb") as f:
                    body = f.read()
                self.assertIn(f"its heads but {grew}, which start fresh", said)
                self.assertEqual([tuple(h) for h in after["cfg"]["outputs"]], ripmodels.HEADS)
                grown = ripmodels.FrameModel(ripmodels.shape(after["cfg"])).eval()
                grown.load_state_dict(after["model"])
                with torch.no_grad():
                    was, now = start(x), grown(x)
                self.assertEqual(now.shape[-1], ripmodels.Config().out_size)
                self.assertTrue(torch.equal(now[..., :old.out_size], was))
                listed = head("family", len(ripmodels.FAMILIES)) + head("beat", 1) + head("downbeat", 1) + head("struck", 1) + head("struckvel", 1)
                self.assertIn(listed + layout + struct.pack("<I", len(after["model"])), body)
        self.assertFalse({"struck", "struckvel"} & set(dict(ripmodels.PITCHNET_HEADS)))

    def test_init_carries_each_head_by_its_name_wherever_the_start_laid_it(self):
        """A frame model whose heads are this run's in another order, one short and with one this run has no more starts
        a run through --init: each head both have keeps its rows of the output layer to the bit, so says what the start's
        did as near as a row's place in the layer lets a sum round, the line names what starts fresh and what is left
        behind, and a start whose head is of another size is refused."""
        x = torch.randn(1, 40, 2 * ripmodels.INPUTS, generator=torch.Generator().manual_seed(1))
        heads = [h for h in ripmodels.HEADS if h[0] != "downbeat"]
        with tempfile.TemporaryDirectory() as d:
            at = lambda name: os.path.join(d, name)
            had = ripmodels.Config(dim=8, heads=2, layers=1, ffn=16, window=4, outputs=heads[5:] + [("gone", 3)] + heads[:5])
            start = ripmodels.FrameModel(had).eval()
            corpus(d, [{"bpm": 120, "bars": 1, "events": [note(0, 10, 60, "piano"), hit(20, 0)]}])
            torch.save({"model": start.state_dict(), "layout": ripmodels.LAYOUT, "cfg": asdict(had)}, at("old.pt"))
            said = run("--corpus", d, "--out", at("new.bin"), "--init", at("old.pt"), "--checkpoint", at("new.pt"), "--lr", "0")
            self.assertIn("its heads but downbeat, which start fresh, and without its gone, which this run's model has none of", said)
            after = torch.load(at("new.pt"))
            self.assertEqual([tuple(h) for h in after["cfg"]["outputs"]], ripmodels.HEADS)
            grown = ripmodels.FrameModel(ripmodels.shape(after["cfg"])).eval()
            grown.load_state_dict(after["model"])
            with torch.no_grad():
                was, now = start(x), grown(x)
            for k, _ in heads:
                for rows in ("out.weight", "out.bias"):
                    self.assertTrue(torch.equal(after["model"][rows][grown.cfg.slice(k)], start.state_dict()[rows][had.slice(k)]), (k, rows))
                self.assertTrue(torch.allclose(now[..., grown.cfg.slice(k)], was[..., had.slice(k)], rtol=0, atol=1e-6), k)
            wider = ripmodels.Config(**{**asdict(had), "outputs": [("pitch", ripmodels.PITCHES + 1) if k == "pitch" else (k, n) for k, n in had.outputs]})
            torch.save({"model": ripmodels.FrameModel(wider).state_dict(), "layout": ripmodels.LAYOUT, "cfg": asdict(wider)}, at("wider.pt"))
            with self.assertRaises(SystemExit) as err:
                run("--corpus", d, "--out", at("no.bin"), "--init", at("wider.pt"))
            self.assertIn("holds a model of", str(err.exception))

    def test_a_probe_trains_the_heads_it_names_alone_on_a_frame_model_held_as_it_was(self):
        """--probe on a frame model started from one without the heads trained, the beat and the downbeat by itself or
        those it names, trains their rows of the output layer and holds the rest: the file it exports, its last step's,
        holds every other weight of the start's file, its every other output is the start's to the bit, and the two
        heads say something else. A head the model lacks, or heads of two output layers, are refused."""
        shape = dict(dim=8, heads=2, layers=1, ffn=16, window=4)
        x = torch.randn(1, 40, 2 * ripmodels.INPUTS, generator=torch.Generator().manual_seed(1))
        for first, words, trained in (("beat", ["--probe"], {"beat", "downbeat"}), ("struck", ["--probe", "struck,struckvel"], {"struck", "struckvel"})):
            old = ripmodels.Config(**shape, outputs=heads_before(first))
            start = ripmodels.FrameModel(old).eval()
            with tempfile.TemporaryDirectory() as d:
                at = lambda name: os.path.join(d, name)
                corpus(d, [{"bpm": 100, "bars": 1, "events": [note(0, 10, 60, "piano"), hit(20, 0)]}])
                torch.save({"model": start.state_dict(), "layout": ripmodels.LAYOUT, "cfg": asdict(old)}, at("old.pt"))
                run("--corpus", d, "--out", at("start.bin"), "--init", at("old.pt"), "--lr", "0")  # the start's file, the new rows as grown
                said = run("--corpus", d, "--out", at("probed.bin"), "--init", at("old.pt"), *words, "--lr", "0.01", steps=3)
                with self.assertRaisesRegex(SystemExit, "no head tempo in this model"):
                    run("--corpus", d, "--out", at("no.bin"), "--init", at("old.pt"), "--probe", "struck,tempo")
                (heads, _, was), (_, _, now) = exported(at("start.bin"), 48), exported(at("probed.bin"), 48)
            self.assertIn("a probe: exporting its last step, 3", said)
            self.assertEqual(heads, ripmodels.HEADS)
            model = ripmodels.FrameModel(ripmodels.Config(**shape)).eval()
            for k, t in ripmodels.tensors(model):
                t.copy_(torch.from_numpy(now[k].copy()))
            rows = train.rows(model)
            with torch.no_grad():
                said_by, says = start(x), model(x)
            self.assertTrue(torch.equal(says[..., :old.out_size], said_by))
            for k in was:
                moved = np.zeros(len(was[k]), bool)
                if k.startswith("out."):
                    for h in trained:
                        moved[rows[h][1]] = True
                with self.subTest(first=first, tensor=k):
                    self.assertTrue(np.array_equal(now[k][~moved], was[k][~moved]))
                    self.assertFalse(moved.any() and any(np.array_equal(now[k][rows[h][1]], was[k][rows[h][1]]) for h in trained))
        with self.assertRaisesRegex(SystemExit, "lie over several output layers"):
            train.probe(ripmodels.PitchNet(CFG), "onset,pitchfamily")

    def test_a_family_unasked_has_no_term_and_its_rows_stay_zero(self):
        """--family-unasked on a frame model grown from the nine heads before the family: the step's loss has no
        family term, and after three steps of a free body the family's rows of the output layer are zero."""
        old = ripmodels.Config(dim=8, heads=2, layers=1, ffn=16, window=4, outputs=heads_before("family"))
        before = ripmodels.FrameModel(old).state_dict()
        with tempfile.TemporaryDirectory() as d:
            at = lambda name: os.path.join(d, name)
            corpus(d, [{"bpm": 100, "bars": 1, "events": [note(0, 10, 60, "piano")]}])
            torch.save({"model": before, "layout": ripmodels.LAYOUT, "cfg": asdict(old)}, at("old.pt"))
            said = run("--corpus", d, "--out", at("new.bin"), "--init", at("old.pt"), "--checkpoint", at("new.pt"), "--family-unasked", "--lr", "0.01", steps=3)
            after = torch.load(at("new.pt"))["model"]
        self.assertIn("its heads but family, beat, downbeat, struck, struckvel, which start fresh", said)
        last = next(line for line in said.splitlines() if line.startswith("step 3 loss "))
        self.assertIn(" downbeat ", last)
        self.assertNotIn("family", last)
        family = ripmodels.Config().slice("family")
        self.assertEqual([after[k][family].abs().sum().item() for k in ("out.weight", "out.bias")], [0, 0])
        self.assertFalse(all(torch.equal(after[k], v) for k, v in before.items() if not k.startswith("out.")))


class WhatIsStruck(unittest.TestCase):
    """The frame model's struck and struckvel heads, and the hits a corpus's labels leave out"""

    KIT = [{"genes": [0.5] * 13, "band": 120}]  # a kit drawn from the genome, whose pieces name no line

    def test_what_is_struck_is_asked_at_every_frame_of_every_window(self):
        """A hit of a line, a strike of no line and a kit's strike are struck, each marked a frame either side, and no
        note is; the head is asked at every frame a window holds, a kit's loop's and a placed recording's among them,
        a strike weighing as a hit does in the drums head."""
        loop = label([hit(10, 0), hit(30), note(50, 60, 60, "piano", voice(0.1), 1), hit(70, piece=0)], 100, kit=self.KIT)
        sung = label([note(20, 40, 52, "voice"), hit(60)], 100, source="a.wav", placed=True, reach=3)
        t = [train.targets(Frames(100), lab, 0, 128) for lab in (loop, sung)]
        self.assertEqual([x["struck"].tolist() for x in t], [marks(100, at) + [0] * 28 for at in ([10, 30, 70], [60])])
        model = Heard(128)
        model.y = torch.zeros(2, 128, model.cfg.out_size)  # every score a half
        x, batch = train.stack([(np.zeros((128, 1), np.float32), tg) for tg in t])
        self.assertAlmostEqual(train.losses(model, x, batch)["struck"].item(), math.log(2) * (188 + train.STRUCK_POS * 12) / 200, places=5)

    def test_how_hard_is_asked_on_a_strikes_frames_where_its_label_says(self):
        """The struckvel head is asked the velocity of a strike on its frame and one either side, of two there the
        louder's, and nothing where the label gives none; its loss is the squared miss, in a laid or a spliced strike's
        window none."""
        lab = label([hit(10, 0, 0.4), hit(10, 2, 0.6), hit(12, 2, 0.2), hit(30, -1, 0.5), hit(50, 1, -1)], 64)
        t = train.targets(Frames(64), lab, 0, 64)
        self.assertEqual(np.flatnonzero(t["struckask"]).tolist(), [9, 10, 11, 12, 13, 29, 30, 31])
        np.testing.assert_allclose(t["struckvel"][[9, 10, 11, 12, 13, 29, 30, 31]], [0.6] * 3 + [0.2] * 2 + [0.5] * 3)
        model = Heard(64)
        model.y[..., model.cfg.slice("struckvel")] = 0.0  # a half
        frs, tgs, _ = train.validation([(Frames(64), lab)], 64, 2)
        self.assertAlmostEqual(train.losses(model, frs[0], tgs[0])["struckvel"].item(), (3 * 0.1 ** 2 + 2 * 0.3 ** 2) / 8, places=6)

    def test_a_laid_or_spliced_strike_counts_where_it_stands_out_of_a_line_or_of_none(self):
        """A strike of drums laid over a loop, or spliced beside another clip, is struck and an onset where it stands
        out of what else sounds, a line's hit or a strike of no line alike, and only a line's hit is a drum's; one
        under the rest is neither. How hard goes unasked, and a hit its label left out asks nothing where it falls."""
        n, e = train.CHUNK, np.zeros((train.CHUNK, ripmodels.VALUES))
        for f, level in ((20, 100.0), (40, 100.0), (60, 1e-3)):
            e[f:f + 2, ripmodels.BANDS + 150:ripmodels.BANDS + 180] = level
        drums, quiet = Frames(n), Frames(n)
        drums.fr[:], quiet.fr[:] = np.log1p(e), 0.01
        struck, loop = train.Label({"events": [hit(20), hit(40, 1), hit(60)]}, n, MANIFEST, [70]), label([], n)
        _, laid = train.sample(quiet, loop, 0, n, 2, (drums, struck, 0.0), random.Random(1))
        turn = itertools.cycle([(drums, struck), (quiet, loop)])
        _, spliced = train.shuffled(lambda rng: next(turn), n, 2, random.Random(1))
        for t in (laid, spliced):
            self.assertEqual([t[k].tolist() for k in ("struck", "onset", "left")], [marks(n, [20, 40])] * 2 + [marks(n, [70])])
            self.assertEqual(t["drums"].T.tolist(), [[0] * n, marks(n, [40]), [0] * n])
            self.assertFalse(t["struckask"].any())

    def test_a_hit_a_label_left_out_asks_what_hears_a_strike_nothing(self):
        """A hit a corpus's left.tsv lists, which its loop's label leaves out as drawing no sound, asks the onset, drums,
        struck and struckvel heads nothing on its frame and one either side, whatever they say there, plain or bent, and
        every other head what it was; a peak there is none a line reads, and a run says how many frames it covers."""
        with tempfile.TemporaryDirectory() as d:
            corpus(d, [{"events": [hit(8, 0)]}, {"events": [hit(8, 0), hit(32, 0), note(40, 60, 60, "piano")]}])
            with open(os.path.join(d, "left.tsv"), "w") as f:
                f.write("loop\tsource\tsec\tframe\tkey\tdrum\tvel\twhy\trise\n"
                        "loop_001\ta.wav\t0.2\t20\t44\t2\t0.4\tno sound\t1e-05\nloop_001\ta.wav\t0.7\t63\t44\t2\t0.4\tno sound\t1e-05\n")
            with contextlib.redirect_stdout(io.StringIO()) as said:
                _, loops = train.load_corpus(d)
        self.assertEqual(said.getvalue(), f"2 hits of {d} left out of its labels: 5 frames ask what hears a strike nothing\n")
        self.assertEqual([lab.left.tolist() for _, lab in loops], [[], [19, 20, 21, 62, 63]])
        fr, lab, model = Frames(64), loops[1][1], Heard(64)
        frs, tgs, _ = train.validation([(fr, lab)], 64, 2)
        model.y = torch.randn(1, 64, model.cfg.out_size, generator=torch.Generator().manual_seed(1))
        was = {k: v.item() for k, v in train.losses(model, frs[0], tgs[0]).items()}
        for head in ("onset", "drums", "struck", "struckvel"):
            model.y[0, [19, 20, 21, 62, 63], model.cfg.slice(head).start] = 9.0
        self.assertEqual({k: v.item() for k, v in train.losses(model, frs[0], tgs[0]).items()}, was)
        model.y[..., model.cfg.slice("struck")] = -9.0
        model.y[0, [8, 20, 32], model.cfg.slice("struck").start] = 9.0
        self.assertEqual(train.evaluate(model, frs, tgs, "cpu")["hits"][0], 1.0)
        _, bent = train.bent(fr, lab, 32, 2, None, random.Random(2), None)
        self.assertTrue(bent["left"].any())

    def test_a_line_reads_the_heads_peaks_against_what_is_struck_or_a_minute_where_nothing_is(self):
        """A held-out or validation line gives the struck head's F1 at its peaks over the score its weight puts at a
        posterior of a half, a peak right within four frames of something struck; over a corpus where nothing is
        struck, its peaks a minute, and where no label says how hard, no loss of that head's."""
        frames, model = 414, Heard(512)
        loop, recording = label([hit(20, 0), hit(100), hit(200, piece=0), hit(300, 2)], frames, kit=self.KIT), label([note(0, 10, 60, "guitar")], frames, source="a.wav")
        at = model.cfg.slice("struck").start
        model.y[0, [20, 101, 250], at] = 9.0  # two right, one that is none
        model.y[0, 350, at] = math.log(train.STRUCK_BAR / (1 - train.STRUCK_BAR)) - 0.1  # one under the bar
        v, none = (train.evaluate(model, *train.validation([(Frames(frames), lab)], 512, 2)[:2], "cpu") for lab in (loop, recording))
        self.assertAlmostEqual(v["hits"][0], 2 * (2 / 3) * (2 / 4) / (2 / 3 + 2 / 4))
        self.assertEqual((none["hits"][0], "struckvel" in v, "struckvel" in none), (None, True, False))
        self.assertAlmostEqual(none["hits"][1], 3 / frames * 60 * ripmodels.RATE / ripmodels.HOP)
        with tempfile.TemporaryDirectory() as d:
            os.mkdir(os.path.join(d, "recordings"))
            corpus(d, [{"bpm": 100, "bars": 1, "events": [hit(8 * k, k % 3) for k in range(8)]}] * 2)
            corpus(os.path.join(d, "recordings"), [{"source": "a.wav", "events": [note(0, 10, 60, "guitar")]}])
            said = run("--corpus", d, "--out", os.path.join(d, "out.bin"), "--holdout", "1", "--validate", os.path.join(d, "recordings")).splitlines()
        held, validation = (next(line for line in said if line.startswith(f"step 1 {name} ")) for name in ("held-out", "validation"))
        self.assertRegex(held, r", struck's F1 \d\.\d{3} \(.* struck \d\.\d{3} struckvel \d\.\d{3}")
        self.assertRegex(validation, r", nothing struck, struck's peaks \d+\.\d a minute \(.* struck \d\.\d{3}")


def sounding(frames, events) -> np.ndarray:
    """a loop's frames as a corpus stores them, each note's first eight harmonics in the band nearest each in both
    planes, louder over its first two frames"""
    e = np.zeros((frames, ripmodels.VALUES))
    for ev in events:
        for h in range(1, 9):
            band = round(3 * (ev["midi"] + 12 * np.log2(h) - ripmodels.BAND_LOWEST))
            if band < ripmodels.BANDS:
                for at in (band, ripmodels.BANDS + band):
                    e[ev["frame"]:ev["end"], at] += 0.05 / h
                    e[ev["frame"]:ev["frame"] + 2, at] += 0.2 / h
    return np.log1p(100 * e).astype(np.float32)


class ThePitches(unittest.TestCase):
    def test_a_note_takes_its_cell_among_the_pianos_keys(self):
        """A note's cell is its key among the piano's 88, A0 to C8: the lowest and the highest keys are a pitch, a strike
        and a release as any, and a note under A0 or over C8 is in no cell, the notes inside asked as ever."""
        events = [note(2, 10, 21, "piano"), note(4, 12, 108, "piano"), note(0, 20, 60, "piano"), note(6, 14, 20, "piano"), note(6, 14, 109, "piano")]
        t = label(events).window(0, 40)
        pairs = lambda key: [tuple(x) for x in t[key].tolist()]
        self.assertEqual(t["pitch"].shape, (40, 88))
        for midi, at, end in ((21, 2, 10), (108, 4, 12)):
            with self.subTest(midi=midi):
                self.assertEqual(int(t["pitch"][:, midi - LOW].sum()), end - at)
                self.assertIn((at, midi - LOW), pairs("strikes"))
                self.assertIn((end, midi - LOW), pairs("releases"))
        self.assertEqual((int(t["pitch"].sum()), len(pairs("strikes")), len(pairs("releases"))), (8 + 8 + 20, 3, 3))

    def test_a_model_taught_a_note_at_either_end_of_the_pianos_keys_reports_it(self):
        """A notes model trained 150 steps on one loop whose sound is an A0 and a C8, twice each, says both pitches
        sound over their notes and are struck in their cells, and no other pitch: the piano's lowest and highest
        keys are taught and heard as any."""
        events = [note(8, 40, 21, "piano"), note(24, 56, 108, "piano"), note(72, 104, 21, "piano"), note(88, 120, 108, "piano")]
        with tempfile.TemporaryDirectory() as d:
            corpus(d, [{"events": events}], 128, lambda lab: sounding(128, lab["events"]))
            argv = ["train.py", "--corpus", d, "--pitchnet", "--cell", "8", "--out", os.path.join(d, "notes.bin"), "--checkpoint", os.path.join(d, "notes.pt"),
                    "--steps", "150", "--every", "150", "--batch", "1", "--window", "64", "--lr", "0.01", "--device", "cpu"]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                train.main()
            ck = torch.load(os.path.join(d, "notes.pt"))
        model = ripmodels.build(ripmodels.shape(ck["cfg"])).eval()
        model.load_state_dict(ck["model"])
        with torch.no_grad():
            y = model(torch.from_numpy(ripmodels.features(sounding(128, events), 4))[None])[0]
        pitch, strike = (torch.sigmoid(y[:, model.cfg.slice(k)]) for k in ("pitch", "strike"))
        cells = [21 - LOW, 108 - LOW]
        for ev in events:
            self.assertGreater(pitch[(ev["frame"] + ev["end"]) // 2, ev["midi"] - LOW].item(), 0.9)
        self.assertGreater(strike[:, cells].amax(0).min().item(), 0.5)
        others = [c for c in range(model.cfg.pitches) if c not in cells]
        self.assertLess(max(pitch[:, others].max().item(), strike[:, others].max().item()), 0.5)


# a notes model at the shape run() gives, and what lixenwraith/transformer hears of a weight file: the frames, rows of
# float32, through it, each head named after them written as float32 a frame at a time
NOTES = dict(inputs=ripmodels.STACK_BINS + 1, channels=ripmodels.STACK_CHANNELS, cell=4, dim=8, heads=2, layers=1, ffn=16, window=4)
HEARD_GO = """package main

import (
	"encoding/binary"
	"math"
	"os"

	"github.com/lixenwraith/transformer"
)

func main() {
	f, err := os.Open(os.Args[1])
	if err != nil {
		panic(err)
	}
	m, err := transformer.Load(f)
	if err != nil {
		panic(err)
	}
	raw, err := os.ReadFile(os.Args[2])
	if err != nil {
		panic(err)
	}
	n := m.Config().Inputs * m.Config().Channels
	frames := make([][]float32, len(raw)/4/n)
	for t := range frames {
		frames[t] = make([]float32, n)
		for i := range frames[t] {
			frames[t][i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*(t*n+i):]))
		}
	}
	var out []byte
	for _, y := range m.Forward(frames) {
		for _, h := range os.Args[4:] {
			for _, v := range m.Slice(y, h) {
				out = binary.LittleEndian.AppendUint32(out, math.Float32bits(v))
			}
		}
	}
	if err := os.WriteFile(os.Args[3], out, 0o644); err != nil {
		panic(err)
	}
}
"""


class TheLongConvolutions(unittest.TestCase):
    """--init grows a notes model's long convolutions under --dilations from a checkpoint of none or of the first of
    them, a head with them where one grows"""

    EVENTS = [note(4, 40, 45, "piano"), note(30, 90, 64, "guitar"), note(70, 120, 72, "piano"), note(96, 126, 50, "guitar")]

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.at = lambda name: os.path.join(self.dir.name, name)
        corpus(self.dir.name, [{"events": self.EVENTS}], 128, lambda lab: sounding(128, lab["events"]))

    def tearDown(self):
        self.dir.cleanup()

    def start(self, dilations=(), outputs=ripmodels.PITCHNET_HEADS, **widths) -> ripmodels.PitchNet:
        """a notes model of those long convolutions, heads and widths, saved as a run's checkpoint is, start.pt"""
        cfg = ripmodels.Config(**{**NOTES, **widths}, outputs=list(outputs), dilations=dilations)
        torch.manual_seed(1)
        model = ripmodels.PitchNet(cfg).eval()
        torch.save({"model": model.state_dict(), "layout": ripmodels.LAYOUT, "cfg": asdict(cfg)}, self.at("start.pt"))
        return model

    def grow(self, name, *flags, steps=1) -> str:
        """a run from start.pt that grows the long convolutions 3, 9 and 27 frames apart, its checkpoint and file name"""
        return run("--corpus", self.dir.name, "--pitchnet", "--cell", "4", "--dilations", "3,9,27", "--init", self.at("start.pt"),
                   "--checkpoint", self.at(name + ".pt"), "--out", self.at(name + ".bin"), *flags, steps=steps)

    def heard(self, model, x) -> tuple:
        """the cells model's notes convolution reads of the planes x, and its outputs"""
        seen = []
        hook = model.notes.register_forward_pre_hook(lambda _, args: seen.append(args[0]))
        with torch.no_grad():
            y = model.eval()(x)
        hook.remove()
        return seen[0][..., 1:ripmodels.PITCHES + 1], y

    def test_a_start_grows_the_long_convolutions_and_carries_every_weight_it_had(self):
        """From a notes model of no long convolutions, a head short, or of the first of them, the rest grow: every
        weight the start had is carried whole, a grown head's rows after its own, the line says what grew, and the
        frame's heads, which hear no cell, say what the start's did to the bit."""
        x = torch.from_numpy(ripmodels.features(sounding(128, self.EVENTS), 4))[None]
        for had, outputs, grew in (((), ripmodels.PITCHNET_HEADS[:-1], "its heads but pitchvoice, which start fresh, and its long convolutions but those 3, 9, 27"),
                                   ((3,), ripmodels.PITCHNET_HEADS, "its long convolutions but those 9, 27")):
            with self.subTest(had=had):
                start = self.start(had, outputs)
                said = self.grow(f"grown{len(had)}", "--lr", "0")
                self.assertRegex(said, f"started from {re.escape(self.at('start.pt'))}, {grew} frames apart, which start handing its cells on, \\d\\.\\d{{3}} off them\n")
                ck = torch.load(self.at(f"grown{len(had)}.pt"))
                for k, v in start.state_dict().items():
                    self.assertTrue(torch.equal(ck["model"][k][:len(v)] if v.dim() else ck["model"][k], v), k)
                grown = ripmodels.build(ripmodels.shape(ck["cfg"]))
                grown.load_state_dict(ck["model"])
                was, now = (self.heard(m, x)[1] for m in (start, grown))
                for k in ripmodels.FRAME_HEADS:
                    self.assertTrue(torch.equal(now[..., grown.cfg.slice(k)], was[..., start.cfg.slice(k)]), k)

    def test_a_start_with_a_head_this_code_has_no_row_for_leaves_it_behind(self):
        """A notes model trained when its frame layer had a head more, laid after its others, starts a run through
        --init: every head this run has says what the start's did to the bit and the line names the head left behind;
        rows no head of the start accounts for are refused."""
        start = self.start((3, 9, 27))
        state, cfg = dict(start.state_dict()), asdict(start.cfg)
        more = lambda: {k: torch.cat([state[k], torch.ones_like(state[k][:1])]) for k in ("frame.weight", "frame.bias")}
        state.update(more())
        cfg["outputs"] = cfg["outputs"] + [("struck", 1)]
        torch.save({"model": state, "layout": ripmodels.LAYOUT, "cfg": cfg}, self.at("start.pt"))
        said = self.grow("left", "--lr", "0")
        self.assertIn(f"started from {self.at('start.pt')}, without its struck, which this run's model has none of\n", said)
        ck = torch.load(self.at("left.pt"))
        self.assertEqual([tuple(h) for h in ck["cfg"]["outputs"]], ripmodels.PITCHNET_HEADS)
        grown = ripmodels.build(ripmodels.shape(ck["cfg"]))
        grown.load_state_dict(ck["model"])
        x = torch.from_numpy(ripmodels.features(sounding(128, self.EVENTS), 4))[None]
        was, now = (self.heard(m, x)[1] for m in (start, grown))
        self.assertTrue(torch.equal(now, was))
        state.update(more())
        torch.save({"model": state, "layout": ripmodels.LAYOUT, "cfg": cfg}, self.at("start.pt"))
        with self.assertRaises(SystemExit) as err:
            self.grow("odd", "--lr", "0")
        self.assertIn("rows are not those of its heads", str(err.exception))

    def test_a_grown_long_convolution_hands_the_cells_on_nearer_than_one_started_at_zero(self):
        """Each long convolution adds its cells back before a GELU, so one of zero weights hands on gelu(c), three of
        them gelu(gelu(gelu(c))); started as each grows, they hand the notes convolution cells under a fifth as far from
        the start's in the mean square, and the miss the start states is theirs on the frames it was fitted to."""
        start = self.start()
        x = torch.from_numpy(ripmodels.features(sounding(128, self.EVENTS), 4))[None]
        cells = {}
        for name in ("zero", "started"):
            grown = ripmodels.PitchNet(ripmodels.Config(**NOTES, outputs=ripmodels.PITCHNET_HEADS, dilations=(3, 9, 27)))
            grown.load_state_dict(ripmodels.carried(grown, start.cfg, start.state_dict()))
            with torch.no_grad():
                for layer in grown.long:
                    layer.weight.zero_()
                    layer.bias.zero_()
            miss = ripmodels.start_long(grown, 0, x) if name == "started" else None
            cells[name] = self.heard(grown, x)[0]
        was = self.heard(start, x)[0]
        off = {k: (c - was).square().mean().sqrt().item() for k, c in cells.items()}
        self.assertLess(off["started"], off["zero"] / 5, off)
        self.assertAlmostEqual(miss, off["started"], places=5)

    def test_a_grown_model_trains_and_its_checkpoint_resumes_as_the_unbroken_run(self):
        """A grown model takes its steps, the taps its long convolutions start at zero among them, and a run of it
        stopped after its first step and resumed from its checkpoint ends on the weights of the run that was never
        stopped, to the bit."""
        self.start()
        self.grow("whole", "--lr", "0.01", steps=2)
        self.grow("part", "--lr", "0.01", steps=1)
        self.assertIn("resumed at step 1", self.grow("part", "--lr", "0.01", steps=2))
        whole, part = (torch.load(self.at(name + ".pt")) for name in ("whole", "part"))
        self.assertEqual((whole["step"], part["step"]), (2, 2))
        for k, v in whole["model"].items():
            self.assertTrue(torch.equal(part["model"][k], v), k)
        self.assertTrue(all(bool(whole["model"][f"long.{i}.weight"][:, :, 0].ne(0).all()) for i in range(3)))

    def test_a_grown_models_file_is_heard_in_go_as_python_hears_it(self):
        """The file a grown model's run writes, its long convolutions moved off their start by a step, is read by
        lixenwraith/transformer at the version stapes' go.mod names, whose outputs over a corpus's frames of notes are
        Python's forward in float64 within 1e-5 and as much again of each, as its cell net's own test holds them to its
        definition; so is one of other widths over the stack and over thirteen semitones, which the file carries."""
        go = shutil.which("go")
        if go is None:
            self.skipTest("no go here to hear the file with")
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
        mod = subprocess.run([go, "list", "-m", "-f", "{{.Version}} {{.Dir}}", "github.com/lixenwraith/transformer"], cwd=root, capture_output=True, text=True)
        version, _, path = mod.stdout.strip().partition(" ")
        if mod.returncode != 0 or not path:
            self.skipTest(f"the transformer module is not here ({mod.stderr.strip()}): go mod download github.com/lixenwraith/transformer")
        toolchain = subprocess.run([go, "env", "GOVERSION"], cwd=root, capture_output=True, text=True).stdout.strip()
        for name, widths in (("grown", {}), ("wider", {"reduce": 3, "wide": 5})):  # the trainer's widths, and others the file must carry
            with self.subTest(widths=widths):
                self.heard_in_go(go, toolchain, version, path, name, widths)

    def heard_in_go(self, go, toolchain, version, path, name, widths):
        self.start(**widths)
        self.grow(name, "--lr", "0.01", *(f for k, v in widths.items() for f in ("--span" if k == "wide" else "--" + k, str(v))))
        ck = torch.load(self.at(name + ".pt"))
        model = ripmodels.build(ripmodels.shape(ck["cfg"])).eval()
        model.load_state_dict(ck["model"])
        self.assertEqual((model.reduce.out_channels, model.wide.out_channels), (widths.get("reduce", 8), widths.get("wide", 16)))
        x = torch.from_numpy(ripmodels.features(sounding(300, [note(30 * k, 30 * k + 40, 33 + 7 * k, "piano") for k in range(10)]), 4))[None]
        with torch.no_grad():
            stacked, y = model.stack(x)[0], model.double()(x.double())[0]
        heads = [k for k, _ in model.cfg.outputs]
        prog = self.at("heard-" + name)
        os.mkdir(prog)
        with open(os.path.join(prog, "go.mod"), "w") as f:
            f.write(f"module heard\n\ngo {toolchain.removeprefix('go')}\n\nrequire github.com/lixenwraith/transformer {version}\n\n"
                    f"replace github.com/lixenwraith/transformer => {path}\n")
        with open(os.path.join(prog, "main.go"), "w") as f:
            f.write(HEARD_GO)
        stacked.numpy().astype("<f4").tofile(self.at("frames.raw"))
        got = subprocess.run([go, "run", ".", self.at(name + ".bin"), self.at("frames.raw"), self.at("heard.raw"), *heads], cwd=prog,
                             capture_output=True, text=True, env={**os.environ, "GOWORK": "off", "GOFLAGS": "-mod=mod"})
        self.assertEqual(got.returncode, 0, got.stderr)
        heard = torch.from_numpy(np.fromfile(self.at("heard.raw"), "<f4").reshape(len(y), -1))
        self.assertEqual(model.long[2].dilation, (27, 1))
        at = 0
        for k, size in model.cfg.outputs:
            with self.subTest(head=k):
                want = y[:, model.cfg.slice(k)]
                off = (heard[:, at:at + size].double() - want).abs() - 1e-5 * (1 + want.abs())
                self.assertLessEqual(off.max().item(), 0.0)
            at += size


class TheBatches(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """loops of noise, half with drums, a note of them gliding, a corpus of drums and a noise bank, and a run's
        flags that draw a window through each share: shuffled, bent, ringing, through the channel, drums laid over,
        a corpus mixed in, another drawn and singing among the chunks"""
        cls.dir = tempfile.TemporaryDirectory()
        at = lambda name: os.path.join(cls.dir.name, name)
        rng = np.random.default_rng(1)
        noise = lambda lab: np.log1p(rng.random((96, ripmodels.VALUES))).astype(np.float32)
        hit = lambda f, drum: {"frame": f, "end": f, "drum": drum, "midi": 0, "vel": 0.8, "wave": -1, "family": F["drums"]}
        loops = [{"bpm": 120, "bars": 1, "events": [note(4, 40, 60, "piano", voice(0.1), 4), note(20, 70, 67, "guitar", cents=[0.0] * 25 + [50.0] * 25)]
                  + ([hit(8 * k, k % 3) for k in range(12)] if i % 2 else [])} for i in range(6)]
        for name, labels in (("loops", loops), ("drums", [{"events": [hit(6 * k, k % 3) for k in range(16)]}] * 2)):
            os.mkdir(at(name))
            corpus(at(name), labels, 96, noise)
        np.savez(at("noise.npz"), white=(0.5 + rng.random((400, ripmodels.VALUES))).astype(np.float32), room=np.zeros(ripmodels.VALUES, np.float32))
        cls.flags = ["--corpus", at("loops"), "--out", at("out.bin"), "--holdout", "0", "--batch", "2", "--pcen", "--shuffle", "0.5", "--bend", "0.5",
                     "--ring", "0.5", "--channel", at("noise.npz") + ":0.5", "--over", at("drums") + ":0.5", "--mix", at("drums") + ":0.2",
                     "--extra", at("loops") + ":0.3", "--sung", at("loops") + ":0.5"]

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_a_batch_is_the_same_whatever_number_of_builders_builds_it(self):
        """Batch k is drawn from a stream of the run's seed, its shard and k alone, so a run hands its device the same
        tensors built by one builder, two or three."""
        one = handed(*self.flags, steps=6)
        for builders in ("2", "3"):
            with self.subTest(builders=builders):
                self.assertTrue(same(handed(*self.flags, "--builders", builders, steps=6), one))

    def test_a_run_resumed_at_step_k_is_handed_the_batches_the_unbroken_run_was(self):
        """A run stopped at step 3 and resumed from its checkpoint trains steps 4 to 6 on the batches the run that
        never stopped trained them on, not on its first three again."""
        whole = handed(*self.flags, steps=6)
        with tempfile.TemporaryDirectory() as d:
            first = handed(*self.flags, "--checkpoint", os.path.join(d, "run.pt"), steps=3)
            rest = handed(*self.flags, "--checkpoint", os.path.join(d, "run.pt"), "--builders", "2", steps=6)
        self.assertTrue(same(first + rest, whole))
        self.assertFalse(same(rest, whole[:3]))

    def test_two_shards_draw_different_batches(self):
        """Two shards of one seed draw no batch alike."""
        a, b = (handed(*self.flags, "--shard-seed", shard, steps=3) for shard in ("0", "1"))
        self.assertFalse(any(torch.equal(x, y) for (x, _), (y, _) in zip(a, b)))

    def test_a_run_asks_every_other_head_as_a_model_without_the_struck_heads_is_asked(self):
        """Over a run's batches drawn through every share a model grown the struck heads has every other head's loss a
        model without them has, bit for bit, and its total is that and theirs."""
        old = ripmodels.Config(channels=4, dim=8, heads=2, layers=1, ffn=16, window=4, outputs=heads_before("struck"))
        start = ripmodels.FrameModel(old)
        grown = ripmodels.FrameModel(ripmodels.Config(**{**asdict(old), "outputs": list(ripmodels.HEADS)}))
        grown.load_state_dict(ripmodels.carried(grown, old, start.state_dict()))
        for x, t in handed(*self.flags, steps=6):
            with torch.no_grad():
                was, now = train.losses(start, x, t), train.losses(grown, x, t)
            self.assertEqual({k: v.item() for k, v in now.items() if k not in ("struck", "struckvel", "total")}, {k: v.item() for k, v in was.items() if k != "total"})
            self.assertEqual(now["total"].item(), (was["total"] + now["struck"] + 2.0 * now["struckvel"]).item())

    def test_without_a_placed_start_every_target_and_loss_is_as_without_reaches(self):
        """A digest of every other target over windows cut anywhere and padded, of labels whose notes hold their pitch
        or move, cents placed or not, two to a cell or outside the pitches, beside drums, is pinned, a placed start's
        targets unset there and in a run's batches of every kind, whose every loss is bit for bit an onset's and a
        strike's asked at every frame, written out here; a change to what a window holds pins it again. With no kit the
        kit's mask is empty and with no hit left out its, so the digest leaves them, and what is struck, where a drum is."""
        rng, digest = random.Random(5), hashlib.sha1()
        with mock.patch.object(train, "GENES", array.array("f")):
            for _ in range(24):
                frames, events = rng.choice([90, 300, 700]), []
                for _ in range(rng.randrange(5, 40)):
                    t = rng.randrange(-2, frames + 2)
                    if rng.random() < 0.25:
                        events.append({"frame": t, "end": t, "drum": rng.randrange(3), "midi": 0, "vel": 0.8, "wave": -1, "family": F["drums"]})
                        continue
                    e = note(t, t + rng.choice([-1, 1, 5, 30, 120, 400]), rng.choice([15, 21, 52, 60, 60, 61, 108, 115]), rng.choice(ripmodels.FAMILIES),
                             voice(rng.random()) if rng.random() < 0.5 else None, rng.randrange(5))
                    if rng.random() < 0.5:  # its cents a frame, a recording's null where its track is unvoiced
                        e["cents"] = [None if "voice" not in e and rng.random() < 0.2 else rng.uniform(-60, 60) for _ in range(e["end"] - t)]
                    events.append(e)
                lab = label(events, frames, tuning=rng.choice([0.0, 50.0, -50.0, 23.4]))
                for start in rng.sample(range(frames), 3):
                    t = train.targets(Frames(frames), lab, start, rng.choice([86, 512]), rng.random() < 0.5)
                    self.assertEqual((t["struck"].tolist(), int(t["left"].sum())), (t["drums"].max(1).tolist(), 0))
                    for k in sorted(set(t) - {"kit", "left", "struck", "struckvel", "struckask"}):
                        if k in PLACED_STARTS:
                            self.assertFalse(t[k].any(), k)
                            continue
                        digest.update(f"{k} {t[k].dtype} {t[k].shape}".encode() + np.ascontiguousarray(t[k]).tobytes())
        self.assertEqual(digest.hexdigest(), "82e0bd088a3cc46e95c43dc4d99fac8faab52033")

        def marks(logits, marks, stretch, weight, starts, valid):  # every valid frame asked its marks, as the other heads are
            l = torch.nn.functional.binary_cross_entropy_with_logits(logits, marks, pos_weight=weight, reduction="none")
            return (l.mean(-1) * valid).sum() / valid.sum().clamp(min=1)
        torch.manual_seed(4)
        model = ripmodels.FrameModel(ripmodels.Config(channels=4, dim=8, heads=2, layers=1, ffn=16, window=4))
        for x, t in handed(*self.flags, steps=6):
            self.assertFalse(any(t[k].any() for k in PLACED_STARTS))
            got = train.losses(model, x, t)
            with mock.patch.object(train, "rise_loss", marks):
                want = train.losses(model, x, t)
            self.assertEqual(got.keys(), want.keys())
            for k in got:
                self.assertTrue(torch.equal(got[k], want[k]), k)


def played(frames, events) -> np.ndarray:
    """a loop's frames as a corpus stores them: sounding's notes, and each drum hit a burst over its drum's third of
    the bands in both planes, for two frames"""
    e = np.expm1(sounding(frames, [x for x in events if x["drum"] < 0]).astype(np.float64)) / 100
    for x in events:
        if x["drum"] >= 0:
            third = np.arange(x["drum"] * ripmodels.BANDS // 3, (x["drum"] + 1) * ripmodels.BANDS // 3)
            e[x["frame"]:x["frame"] + 2, np.concatenate([third, ripmodels.BANDS + third])] += 0.3
    return np.log1p(100 * e).astype(np.float32)


def kind(d, seed, loops=40, frames=128, families=("piano", "guitar")) -> None:
    """a corpus of one kind on disk, its seed's draw of it: each loop notes one after another, each of seven pitches
    and of a family of families, over a kick and a hat"""
    rng = random.Random(seed)
    labels = []
    for _ in range(loops):
        events, t = [], 2
        while t < frames - 12:
            n = rng.randrange(6, 20)
            events.append(note(t, min(t + n, frames - 1), rng.choice([48, 52, 55, 60, 64, 67, 72]), rng.choice(families)))
            t += n + rng.randrange(6)
        events += [{"frame": f, "end": f, "drum": drum, "midi": 0, "vel": 0.8, "wave": -1, "family": F["drums"]}
                   for drum, every in ((0, 32), (2, 8)) for f in range(rng.randrange(every), frames - 2, every)]
        labels.append({"events": events})
    os.makedirs(d, exist_ok=True)
    corpus(d, labels, frames, lambda lab: played(frames, lab["events"]))


def passed(*flags, steps) -> tuple:
    """a pass of the shape the rate's tests train, a reading every 20 steps over 8 held-out loops: what it printed,
    and each held-out reading's windows in turn, with the weights read"""
    seen, real = [], train.evaluate

    def evaluate(model, *args, **kw):
        v = real(model, *args, **kw)
        seen.append({**v["windows"], "weights": {k: t.detach().clone() for k, t in model.state_dict().items()}})
        return v
    with mock.patch.object(train, "evaluate", evaluate):
        said = run("--batch", "8", "--dim", "16", "--ffn", "32", "--attention-window", "8", "--holdout", "8", "--every", "20", "--lr", "0.01", *flags, steps=steps)
    return said, seen


def line(said: str, step: int) -> str:
    """the held-out line of a step"""
    return next(x for x in said.splitlines() if x.startswith(f"step {step} held-out "))


def stalls(gain, first, due, trials=20000, readings=60) -> float:
    """how often a figure gaining `gain` noises a reading stalls in a stretch from a reading kept: each reading off by
    chance, its difference from another of one standard error, a gain counted past the noise, and a stall once the
    readings without one reach due(d), d what the first `first` of them read on the reading kept"""
    rng = np.random.default_rng(2)
    y = -train.NOISE * gain * np.arange(readings) + rng.normal(0, 1 / math.sqrt(2), (trials, readings))
    ref, count, need, stretches, stalled = y[:, 0].copy(), np.zeros(trials, int), np.full(trials, readings), 0, 0
    for t in range(1, readings):
        gained = ref - y[:, t] > train.NOISE
        count = np.where(gained, 0, count + 1)
        read = count == first
        need[read] = [due(d) for d in (y[:, t] - ref)[read]]
        stall = count >= need
        stretches, stalled = stretches + int((gained | stall).sum()), stalled + int(stall.sum())
        ref, count, need = np.where(gained | stall, y[:, t], ref), np.where(stall, 0, count), np.where(gained | stall, readings, need)
    return stalled / stretches


class Replay:
    """The rate's rule replayed on readings and no model. Its weights are each loop's loss a head a notes model's pass
    is read by; a step moves them by its rate over --lr, down by a gain and about by a jitter (move), or a test lays
    them (put). Rate reads them, keeps them and puts them back as a model's; the optimizer holds no state."""

    LOOPS, HEADS = 100, train.READ[True]

    def __init__(self, fine=None, readings=12, top=1e-4, seed=0, carried=None, wide=None):
        self.rng = np.random.default_rng(seed)
        self.x = {h: torch.zeros(self.LOOPS, dtype=torch.float64) for h in self.HEADS}
        self.state = {}
        with mock.patch.object(train, "rows", lambda model: {}):  # a latent has no head's rows to raise alone
            self.rate = train.Rate(self, self, top, readings, 1, train.STALL if wide is None else wide, tuple((h, 1.0) for h in self.HEADS))
        self.rate.load(carried)
        self.rate.fine = self.rate.fine if fine is None else fine
        self.step, self.rolled = 0, 0
        self.said = [self.read()]

    def state_dict(self) -> dict:
        return self.x

    def load_state_dict(self, state) -> None:
        self.x = {h: v.clone() for h, v in state.items()}

    def read(self) -> str:
        """the weights read at this step, as a held-out line's rate says them, and what the reading called for done"""
        w = {h: v.clone() for h, v in self.x.items()}
        line = self.rate.read(self.step, {**w, "decoders": sum(w.values()), "loop": torch.arange(self.LOOPS)})
        self.rolled += self.step - self.rate.undo[2] if self.rate.undo else 0
        self.rate.settle()
        return line

    def put(self, level: float, at: float) -> str:
        """the next reading, laid: its sum level over the first's, and |at - another's at| the noise between the two"""
        spread = at * math.sqrt(self.LOOPS - 1) / train.NOISE * torch.tensor([1.0, -1.0], dtype=torch.float64).repeat(self.LOOPS // 2)
        self.x = {h: (level + spread) / len(self.HEADS) for h in self.HEADS}
        self.step += 1
        self.said.append(self.read())
        return self.said[-1]

    def move(self, gain: float) -> str:
        """the next reading, drawn: a step at the rule's rate, a fine one gaining `gain` of its reading's noise"""
        r = self.rate.at(self.step) / self.rate.top
        g = gain * train.NOISE / math.sqrt(len(self.HEADS) * self.LOOPS)
        self.x = {h: v - g * r + r * torch.from_numpy(self.rng.normal(0, 1, self.LOOPS)) for h, v in self.x.items()}
        self.step += 1
        self.said.append(self.read())
        return self.said[-1]

    def carried(self) -> dict:
        return {k: getattr(self.rate, k) for k in self.rate.CARRIED}


class TheRate(unittest.TestCase):
    """The rate of a pass that carries on, doc/rip-training.md §8: its constants by their procedures, and its rules
    on drawn and laid readings (Replay) and on corpora of one kind the tests render, from a pass that began a chain."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.at = staticmethod(lambda name: os.path.join(cls.dir.name, name))
        for k in range(2):
            kind(cls.at(f"kind{k}"), k + 1)
        passed("--corpus", cls.at("kind0"), "--out", cls.at("first.bin"), "--checkpoint", cls.at("first.pt"), "--carry", steps=300)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_a_difference_by_chance_alone_crosses_a_readings_noise_at_alpha(self):
        """A reading's noise is NOISE standard errors of the mean difference over the same windows, a loop's windows
        one draw: two models alike but for chance, read on 200 loops of three windows that share their loop's sound,
        differ past it one way ALPHA of the time; counted window by window, as if each were a draw of its own, more
        than three times as often."""
        rng, loops, crossed, naive = np.random.default_rng(1), np.repeat(np.arange(200), 3), 0, 0
        for _ in range(4000):
            d = np.repeat(rng.normal(0, 1, 200), 3) + rng.normal(0, 1, 600)
            mean, noise = train.apart(d, np.zeros(600), loops)
            crossed += mean < -noise
            naive += mean < -train.NOISE * d.std(ddof=1) / math.sqrt(600)
        self.assertAlmostEqual(crossed / 4000, train.ALPHA, delta=0.007)
        self.assertGreater(naive / 4000, 3 * train.ALPHA)

    def test_a_stall_is_the_fewest_readings_a_figure_gaining_one_noise_a_reading_rarely_goes_without_a_gain(self):
        """STALL is the fewest readings without a gain that a figure still gaining one noise a reading, the least a
        reading tells, goes through less often than ALPHA in a stretch from a reading kept (stalls). One fewer stalls
        it more often than ALPHA."""
        self.assertGreater(stalls(1, train.STALL - 1, lambda d: train.STALL - 1), train.ALPHA)
        self.assertLess(stalls(1, train.STALL, lambda d: train.STALL), train.ALPHA)

    def test_a_stalls_count_is_what_the_gain_its_readings_show_takes_to_gain_two_noises(self):
        """A stall's count is read from its first STALL readings without a gain: STALL where they show none on the
        reading kept, else as many as their pace takes to gain two noises (stall). A figure gaining a half of a noise a
        reading stalls at STALL more often than ALPHA and at that count less often; and the rate raises at the third
        reading of three that show no gain, and at the thirteenth where the third shows 0.14 under a noise of 0.3."""
        self.assertEqual([train.stall(d, 1.0) for d in (0.5, 0.0, -0.9, -0.45, -0.22)], [3, 3, 7, 14, 28])
        self.assertGreater(stalls(0.5, train.STALL, lambda d: train.STALL), train.ALPHA)
        self.assertLess(stalls(0.5, train.STALL, lambda d: train.stall(d, train.NOISE)), train.ALPHA)
        none, some = Replay(readings=20), Replay(readings=20)
        for k in (1, 2, 3):
            none.put(0.0, 0.1 * k)
        for k in range(1, 14):
            some.put(-0.14 * min(k, 3) / 3, 0.1 * min(k, 3))
        self.assertIn("3 without a gain: raised to 1.0e-04, down to fine over 3 readings", none.said[3])
        self.assertIn("3 without a gain of the 13 the gain they show takes", some.said[3])
        self.assertIn("13 without a gain: raised to 1.0e-04, down to fine over 3 readings", some.said[13])

    def raised(self, level, at, fine=None) -> Replay:
        """a pass raised at its third reading, three that show no gain, the third 0.3 noises from the reading kept, and
        its raise's end laid at level over that reading and `at` noises from it"""
        r = Replay(fine)
        for a in (0.1, 0.2, 0.3, 0.5, 0.9):
            r.put(0.0, a)
        r.put(level, at)
        return r

    def test_a_raise_kept_moves_the_reading_kept_with_the_weights(self):
        """A raise that does not pay and ends within the noise of where it began is kept, and its end is then the
        reading kept, with its weights: the next fine reading is told against it, a gain past the noise between the
        two, where against the reading kept before the raise the raise's whole way lay in its noise."""
        r = self.raised(-0.05, 1.3)
        self.assertIn("not paid, -0.050 on the reading kept at step 0, noise 1.300, and -0.050 on where it began, noise 1.000, "
                      "within its noise: kept, the reading kept now this one", r.said[6])
        self.assertIn("gained, -0.010 on the reading kept at step 6, noise 0.005", r.put(-0.06, 1.305))

    def test_a_raise_goes_back_only_where_it_ends_over_where_it_began_past_the_noise(self):
        """A raise that ends over where it began within the noise between the two is kept, its fine rate as it was; one
        that ends over it past that noise goes back to the weights before it, its steps lost, and its fine rate falls
        by FALL."""
        kept, back = self.raised(0.9, 1.3), self.raised(1.1, 1.3)
        self.assertIn("+0.900 on where it began, noise 1.000, within its noise: kept", kept.said[6])
        self.assertEqual((kept.rate.fine, kept.rolled, round(float(sum(kept.x.values()).mean()), 9)), (5e-5, 0, 0.9))
        self.assertIn("+1.100 on where it began, noise 1.000, over past their own noise: pitch, strike, release: "
                      "fallen back to step 3's weights, fine 2.5e-05", back.said[6])
        self.assertEqual((back.rate.fine, back.rolled, round(float(sum(back.x.values()).mean()), 9)), (2.5e-5, 3, 0.0))

    def test_a_raise_that_pays_raises_the_fine_rate_and_no_further_than_lr(self):
        """The fine rate rises by FALL where a raise pays, its end under the reading kept past the noise, and never
        over --lr; a gain at the fine rate leaves it where it is."""
        paid, high = self.raised(-1.5, 1.3, 2e-5), self.raised(-1.5, 1.3, 8e-5)
        self.assertIn("paid, -1.500 on the reading kept at step 0, noise 1.300: kept, fine risen to 4.0e-05", paid.said[6])
        self.assertIn("gained, -0.500 on the reading kept at step 6, noise 0.100", paid.put(-2.0, 1.4))
        self.assertEqual((paid.rate.fine, high.rate.fine), (4e-5, 1e-4))

    def test_a_figure_gaining_under_its_noise_keeps_its_fine_rate_from_pass_to_pass(self):
        """Replayed over forty chains of four passes from a checkpoint that carries no rate, twelve drawn readings a
        pass: a figure gaining a quarter of a reading's noise at every reading ends with its fine rate at or over where
        it began in three chains of four and within an octave under it in nine of ten, under a step in twenty rolled
        back; one gaining nothing ends within an octave under it at the median, under a step in ten rolled back."""
        for gain, most in ((0.25, 0.05), (0.0, 0.1)):
            fines, rolled = [], 0
            for chain in range(40):
                carried = None
                for k in range(4):
                    r = Replay(seed=100 * chain + k, carried=carried)
                    for _ in range(12):
                        r.move(gain)
                    carried, rolled = r.carried(), rolled + r.rolled
                fines.append(carried["fine"])
            with self.subTest(gain=gain):
                self.assertGreaterEqual(float(np.quantile(fines, 0.1 if gain else 0.5)), 2.5e-5)
                self.assertGreaterEqual(float(np.mean(np.array(fines) >= 5e-5)), 0.75 if gain else 0.0)
                self.assertLess(rolled / (40 * 4 * 12), most)

    def test_a_pass_that_carries_on_is_read_by_the_heads_audio_reads_from_its_model(self):
        """A pass that carries on sums, as its decoders' loss, the heads audio reads from its model in a pair, which
        pick its step and move its rate: a frame model's onset, drums, beat, bar line and what is struck and how hard,
        here a corpus that lists no beat, a notes model's pitch, strike and release. A run without the word sums every
        decoders' head."""
        flags = ("--corpus", self.at("kind1"), "--out", self.at("read.bin"))
        said, frame = passed(*flags, "--init", self.at("first.pt"), "--carry", steps=20)
        _, plain = passed(*flags, "--init", self.at("first.pt"), steps=20)
        _, notes = passed(*flags, "--pitchnet", "--cell", "4", "--carry", steps=20)
        self.assertIn("read by its onset, drums, beat, downbeat, struck, struckvel, the heads audio reads from a frame model in a pair", said)
        self.assertTrue(torch.equal(frame[0]["decoders"], frame[0]["onset"] + frame[0]["drums"] + frame[0]["struck"] + 2.0 * frame[0]["struckvel"]))
        self.assertTrue(torch.equal(plain[0]["decoders"], sum(f * plain[0][h] for h, f in train.DECODERS if h in plain[0])))
        self.assertTrue(torch.equal(notes[0]["decoders"], notes[0]["pitch"] + notes[0]["strike"] + notes[0]["release"]))

    def test_a_gain_bought_with_a_decoders_head_falling_past_its_noise_is_no_better(self):
        """A reading is better than another when the decoders' loss is under the other's past its noise and no
        decoders' head is over its own past that head's noise: a gain past the noise is better; the same gain bought
        with the strikes falling past theirs is not, nor a gain within the noise."""
        rng, loops = np.random.default_rng(3), torch.arange(20).repeat_interleave(3)
        base = {h: rng.uniform(0.2, 0.4, 60) for h, _ in train.DECODERS}

        def reading(**by):
            w = {h: torch.from_numpy(base[h] + by.get(h, 0.0) + rng.normal(0, 0.01, 60)) for h, _ in train.DECODERS}
            return {**w, "decoders": sum(f * w[h] for h, f in train.DECODERS), "loop": loops}
        old = reading()
        self.assertEqual(train.better(old, reading(onset=-0.1))[::3], (True, []))
        self.assertEqual(train.better(old, reading(onset=-0.15, strike=0.05))[::3], (False, ["strike"]))
        self.assertEqual(train.better(old, reading(onset=-0.002))[::3], (False, []))

    def test_without_carry_the_schedule_is_todays_bit_for_bit(self):
        """A run without the new words is today's: from --init's weights alone, a fresh optimizer though the
        checkpoint holds one, the rate down a cosine from --lr to zero, no reading before a step and no rate in its
        checkpoint. Its weights are, bit for bit, those of that schedule written out here over the batches it was
        handed."""
        with tempfile.TemporaryDirectory() as d:
            at = lambda name: os.path.join(d, name)
            kind(at("kind"), 1, loops=4)
            run("--corpus", at("kind"), "--out", at("start.bin"), "--checkpoint", at("start.pt"), "--lr", "0.01", steps=5)
            seen = handed("--corpus", at("kind"), "--out", at("run.bin"), "--init", at("start.pt"), "--checkpoint", at("run.pt"), "--lr", "0.01", steps=8)
            start, ran = torch.load(at("start.pt")), torch.load(at("run.pt"))
        model = ripmodels.build(ripmodels.shape(start["cfg"]))
        model.load_state_dict(start["model"])
        opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
        for step, (x, t) in enumerate(seen):
            for g in opt.param_groups:
                g["lr"] = 0.01 * 0.5 * (1 + math.cos(math.pi * step / 8))
            loss = train.losses(model, x, t)["total"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        self.assertEqual((len(seen), "opt" in start, sorted(ran)), (8, True, ["best", "cfg", "layout", "model", "opt", "step", "validated"]))
        for k, v in model.state_dict().items():
            with self.subTest(tensor=k):
                self.assertTrue(torch.equal(ran["model"][k], v))

    def test_a_pass_that_carries_on_is_not_worse_at_its_first_reading(self):
        """A pass that carries on from the last one's checkpoint, its optimizer's state and its fine rate with its
        weights, reads its start on its fresh corpus's held-out loops before a step, and its first reading is not worse
        than that past the reading's noise; today's schedule from that checkpoint, its optimizer fresh and its rate
        back at --lr, is. A checkpoint of weights alone still carries on, its optimizer fresh and its fine rate a fall
        under --lr."""
        first = torch.load(self.at("first.pt"))
        torch.save({k: first[k] for k in ("model", "layout", "cfg")}, self.at("alone.pt"))
        flags = ("--corpus", self.at("kind1"), "--out", self.at("next.bin"))
        said, carried = passed(*flags, "--init", self.at("first.pt"), "--checkpoint", self.at("carried.pt"), "--carry", steps=20)
        _, today = passed(*flags, "--init", self.at("first.pt"), "--checkpoint", self.at("today.pt"), steps=100)
        alone, _ = passed(*flags, "--init", self.at("alone.pt"), "--checkpoint", self.at("alone2.pt"), "--carry", steps=20)
        start = carried[0]
        self.assertIn(f"carried on, its optimizer: fine {first['rate']['fine']:.1e} carried", said)
        self.assertIn("the start, read before a step", line(said, 0))
        mean, noise = train.apart(carried[1]["decoders"], start["decoders"], start["loop"])
        self.assertLessEqual(mean, noise)
        mean, noise = train.apart(today[0]["decoders"], start["decoders"], start["loop"])
        self.assertGreater(mean, noise)
        self.assertIn("carried on, weights alone, the optimizer fresh: no rate carried, fine 5.0e-03, a fall under --lr", alone)

    def test_a_reading_at_the_fine_rate_worse_than_the_one_kept_falls_back_to_it(self):
        """A reading at the fine rate whose decoders' loss is over the reading kept past its noise falls back: training
        goes on from the kept weights and optimizer's state, here the start's, the fine rate falls by FALL, and the
        line says so."""
        ck = torch.load(self.at("first.pt"))
        ck["rate"]["fine"] = 0.5
        torch.save(ck, self.at("fast.pt"))
        said, _ = passed("--corpus", self.at("kind1"), "--out", self.at("fast.bin"), "--init", self.at("fast.pt"), "--checkpoint", self.at("fell.pt"),
                         "--every", "10", "--lr", "0.5", "--carry", steps=10)
        fell = torch.load(self.at("fell.pt"))
        self.assertRegex(line(said, 10), r"rate 5\.0e-01 fine: \+[0-9.]+ on the reading kept at step 0, noise [0-9.]+: fallen back to step 0's weights, fine 2\.5e-01 \(onset ")
        self.assertEqual(fell["rate"]["fine"], 0.5 * train.FALL)
        for k, v in ck["model"].items():
            self.assertTrue(torch.equal(fell["model"][k], v), k)
        for i, state in ck["opt"]["state"].items():
            for k, v in state.items():
                self.assertTrue(torch.equal(fell["opt"]["state"][i][k], v), (i, k))

    def test_a_figure_made_to_stall_draws_a_raise_and_a_raise_that_does_not_pay_falls_back(self):
        """A carried fine rate too small to move a weight stalls the decoders' figure: after STALL readings without a
        gain the rate is raised to --lr, down half a cosine to the fine rate over STALL readings; a raise that ends no
        better than the reading kept, and worse than where it began, falls the fine rate back by FALL, and training
        goes on from the weights before it, each line saying so."""
        ck = torch.load(self.at("first.pt"))
        ck["rate"]["fine"] = 1e-30
        torch.save(ck, self.at("still.pt"))
        rates, real = [], train.Rate.at

        def at(rate, step):
            rates.append(real(rate, step))
            return rates[-1]
        with mock.patch.object(train.Rate, "at", at):
            said, seen = passed("--corpus", self.at("kind1"), "--out", self.at("still.bin"), "--init", self.at("still.pt"), "--checkpoint", self.at("stalled.pt"),
                                "--every", "10", "--lr", "0.1", "--carry", steps=70)
        self.assertIn("rate 1.0e-30 fine: +0.000 on the reading kept at step 0, noise 0.000, 3 without a gain: raised to 1.0e-01, down to fine over 3 readings", line(said, 30))
        self.assertIn("raised: reading 1 of 3", line(said, 40))
        self.assertRegex(line(said, 60), r"raised: reading 3 of 3, not paid, .*: fallen back to step 30's weights, fine 5\.0e-31")
        self.assertIn("rate 5.0e-31 fine", line(said, 70))
        raised = [1e-30 + (0.1 - 1e-30) * 0.5 * (1 + math.cos(math.pi * k / 30)) for k in range(30)]
        self.assertEqual(rates, [1e-30] * 30 + raised + [5e-31] * 10)
        self.assertGreater(*train.apart(seen[6]["decoders"], seen[3]["decoders"], seen[3]["loop"]))
        self.assertTrue(torch.equal(seen[7]["decoders"], seen[3]["decoders"]))

    def test_a_head_stalled_while_the_decoders_gain_is_moved_alone_and_the_others_hold(self):
        """A head whose own figure goes STALL readings without a gain past its own noise while the decoders' loss
        gains, here the frame's family where every note's is drawn at random, has its rows raised alone, its step
        scaled to a raise's rate: at the raise's end its rows are not where a run that never moved them has them, and
        every decoders' head is within the noise of its own reading since the raise began of where that run has it."""
        cfg = ripmodels.Config(dim=16, heads=2, layers=1, ffn=32, window=8)
        torch.manual_seed(3)
        with tempfile.TemporaryDirectory() as d:
            at = lambda name: os.path.join(d, name)
            kind(at("kind"), 7, families=("piano", "guitar", "strings", "winds"))
            torch.save({"model": ripmodels.FrameModel(cfg).state_dict(), "layout": ripmodels.LAYOUT, "cfg": asdict(cfg)}, at("fresh.pt"))
            flags = ("--corpus", at("kind"), "--out", at("out.bin"), "--init", at("fresh.pt"), "--carry", "--batch", "2")
            said, moved = passed(*flags, "--checkpoint", at("moved.pt"), steps=200)
            with mock.patch.object(train.Rate, "follow_heads", lambda rate, step, w: ""):
                _, held = passed(*flags, "--checkpoint", at("held.pt"), steps=200)
        began = next(step for step in range(20, 141, 20) if "family: " in line(said, step) and "its rows raised to 1.0e-02" in line(said, step))
        self.assertIn("while the decoders' loss has not stalled", line(said, began))
        end = began + 3 * 20
        self.assertIn("family's rows raised: reading 3 of 3", line(said, end))
        k, b = end // 20, began // 20
        rows = lambda run, i: run[i]["weights"]["out.weight"][cfg.slice("family")]
        self.assertTrue(torch.equal(rows(moved, b), rows(held, b)))
        self.assertFalse(torch.equal(rows(moved, k), rows(held, k)))
        for h in ("onset", "drums", "pitch", "voicing", "strike", "release"):
            with self.subTest(head=h):
                apart, _ = train.apart(moved[k][h], held[k][h], moved[k]["loop"])
                _, noise = train.apart(moved[k][h], moved[b][h], moved[k]["loop"])
                self.assertLessEqual(abs(apart), noise)

    def test_a_wide_pass_that_ends_over_its_start_past_the_noise_ends_on_its_start(self):
        """A pass that swings wide reads its start, then runs from --lr down to the fine rate over the pass; at its end
        over its start past the noise, it ends on the start: its checkpoint holds the start's weights and optimizer
        state bit for bit, and the count of passes without a gain kept starts again. --wide 0 makes this pass wide;
        without the word a pass is wide after STALL passes in a row without a gain kept."""
        first = torch.load(self.at("first.pt"))
        flags = ("--corpus", self.at("kind1"), "--out", self.at("wide.bin"), "--init", self.at("first.pt"), "--every", "10", "--carry")
        said, _ = passed(*flags, "--checkpoint", self.at("wide.pt"), "--wide", "0", "--lr", "0.5", steps=30)
        wide = torch.load(self.at("wide.pt"))
        self.assertIn("rate 5.0e-01 wide, down to fine", line(said, 0))
        self.assertRegex(line(said, 30), r"wide: reading 3 of 3, worse, \+[0-9.]+ on the start, noise [0-9.]+.*: ending on the start \(onset ")
        self.assertIn("best held-out decoders' loss", said)
        self.assertIn("at step 0; exporting that", said)
        for k, v in first["model"].items():
            self.assertTrue(torch.equal(wide["model"][k], v), k)
        for i, state in first["opt"]["state"].items():
            for k, v in state.items():
                self.assertTrue(torch.equal(wide["opt"]["state"][i][k], v), (i, k))
        self.assertEqual(wide["rate"]["quiet"], 0)
        first["rate"]["quiet"] = train.STALL
        torch.save(first, self.at("quiet.pt"))
        said, _ = passed("--corpus", self.at("kind1"), "--out", self.at("wide.bin"), "--init", self.at("quiet.pt"), "--every", "10", "--carry",
                         "--checkpoint", self.at("quiet2.pt"), steps=10)
        self.assertIn(f"wide, down to fine {first['rate']['fine']:.1e} over the pass, after 3 passes in a row without a gain kept", line(said, 0))


class TheLayout(unittest.TestCase):
    """The frames the models hear are one layout, ripmodels.LAYOUT: a corpus's manifest, a checkpoint and a noise bank
    of another, or of none, are refused, and the refusal names this one."""

    def refused(self, err, layout):
        said = str(err.exception)
        self.assertIn(ripmodels.LAYOUT_NAME, said)
        self.assertIn("no frame layout" if layout is None else f"frame layout {layout}", said)

    def test_a_corpus_of_another_layout_or_none_is_refused_by_name(self):
        for layout in (None, 0, 2):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as d:
                corpus(d, [{"events": []}], layout=layout)
                with self.assertRaises(SystemExit) as err:
                    train.load_corpus(d)
                self.refused(err, layout)
                self.assertIn("export it again", str(err.exception))

    def test_frames_of_another_width_are_refused(self):
        """a loop's frames of one plane are not the layout's two"""
        with tempfile.TemporaryDirectory() as d:
            corpus(d, [{"events": []}])
            with open(os.path.join(d, "loop_000.bands"), "wb") as f:
                f.write(struct.pack("<II", 64, ripmodels.BANDS) + bytes(4 * 64 * ripmodels.BANDS))
            with self.assertRaises(SystemExit) as err:
                train.load_corpus(d)
        self.assertIn(f"{ripmodels.BANDS} values a frame, layout {ripmodels.LAYOUT}'s {ripmodels.VALUES}", str(err.exception))

    def test_a_start_of_another_layout_or_none_is_refused_by_name(self):
        """--init, a resumed --checkpoint and a merge each refuse a .pt of another layout, or of none, before they read its shape."""
        cfg = ripmodels.Config(dim=8, heads=2, layers=1, ffn=16, window=4)
        state = ripmodels.FrameModel(cfg).state_dict()
        for layout in (None, 2):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as d:
                at = lambda name: os.path.join(d, name)
                corpus(d, [{"events": [note(0, 10, 60, "piano")]}])
                torch.save({"model": state, "cfg": asdict(cfg), **({} if layout is None else {"layout": layout})}, at("old.pt"))
                for flags in (("--init", at("old.pt")), ("--checkpoint", at("old.pt"))):
                    with self.assertRaises(SystemExit) as err:
                        run("--corpus", d, "--out", at("new.bin"), *flags)
                    self.refused(err, layout)
                with self.assertRaises(SystemExit) as err, mock.patch.object(sys, "argv", ["average.py", "--out", at("mean.bin"), at("old.pt")]):
                    average.main()
                self.refused(err, layout)

    def test_a_noise_bank_of_another_width_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            np.savez(os.path.join(d, "noise.npz"), white=np.ones((10, ripmodels.BANDS), np.float32), room=np.zeros(ripmodels.BANDS, np.float32))
            with self.assertRaises(SystemExit) as err:
                train.Channel(os.path.join(d, "noise.npz"))
        self.assertIn("make_noise.py", str(err.exception))


class TheVelocity(unittest.TestCase):
    def test_a_strike_whose_velocity_says_nothing_asks_the_velocity_head_nothing(self):
        """A note whose label's vel is -1, as corpus writes for a list whose velocity column says none, or under
        measured for a note it cannot tell apart from the others, is struck as any and asks the velocity head
        nothing: its cells' velocity target is unheard, and whatever the head says there moves no loss, while a note
        of a velocity is asked it."""
        events = [note(4, 20, 60, "guitar", vel=-1), note(4, 20, 64, "guitar", vel=0.6)]
        t = train.targets(Frames(40), label(events), 0, 40)
        for midi, heard, velocity in ((60, 0, 0.0), (64, 1, 0.6)):
            with self.subTest(midi=midi):
                p = midi - LOW
                self.assertEqual(t["strike"][3:6, p].tolist(), [1, 1, 1])
                self.assertEqual(t["heard"][3:6, p].tolist(), [heard] * 3)
                self.assertEqual([round(float(v), 6) for v in t["velocity"][3:6, p]], [velocity] * 3)
        self.assertEqual(int(t["heard"].sum()), 3)
        frs, tgs, _ = train.validation([(Frames(40), label(events))], 40, 2)
        model = Heard(40)
        vel = model.cfg.slice("velocity")

        def loss(at, logit):
            model.y[..., vel] = 0.0
            model.y[0, :, vel.start + at - LOW] = logit
            return train.losses(model, frs[0], tgs[0])["velocity"].item()
        self.assertEqual(loss(60, -5.0), loss(60, 5.0))
        self.assertNotEqual(loss(64, -5.0), loss(64, 5.0))


def rises(lab, start, window, at, head="onset", struck=()) -> float:
    """the loss of head a window of lab's loop asks of a model silent but at at's frames of the window, a frame each, or
    for the strike a frame and its MIDI pitch, with struck's strikes, a frame and a pitch each, another clip's in it"""
    x, t = train.stack([train.sample(Frames(lab.frames), lab, start, window, 2)])
    for f, midi in struck:
        t["strike"][0, f - 1:f + 2, midi - LOW] = 1
    model = Heard(window)
    for r in at:
        k, cell = (r, 0) if head == "onset" else (r[0], r[1] - LOW)
        model.y[0, k, model.cfg.slice(head).start + cell] = 9.0
    return train.losses(model, x, t)[head].item()


class ThePlacedStart(unittest.TestCase):
    def test_a_loop_whose_starts_are_placed_marks_each_strike_as_a_stretch_at_its_pitch(self):
        """A loop whose label says its starts are placed, as corpus writes for a list whose starts column says so, marks
        no strike: each start's mark lies at its pitch where a struck loop's strike does, its stretch about it, and a
        shuffled window's chunks of it are placed too, while its frames, cents, velocities and releases are taught as a
        struck loop's are, its starts marked where a struck loop's onsets are."""
        events = [note(4, 20, 60, "guitar", vel=0.6), note(10, 30, 64, "guitar", vel=0.4, cents=[5.0] * 20)]
        struck, placed = (train.targets(Frames(40), label(events, reach=2, placed=p), 0, 40) for p in (False, True))
        self.assertEqual((int(struck["strike"].sum()) > 0, int(placed["strike"].sum())), (True, 0))
        self.assertEqual((struck["placed"].tolist(), placed["placed"].tolist()), ([0] * 40, [1] * 40))
        for k in ("pitch", "voicing", "cents", "asked", "velocity", "heard", "release"):
            np.testing.assert_array_equal(placed[k], struck[k], k)
        np.testing.assert_array_equal(placed["starts"], struck["onset"])
        np.testing.assert_array_equal(placed["pitchstarts"], struck["strike"])
        stretch = np.zeros((40, ripmodels.PITCHES), np.uint8)
        stretch[1:8, 60 - LOW] = stretch[7:14, 64 - LOW] = 2  # each start's reach either side and its mark's frame past it
        np.testing.assert_array_equal(placed["pitchstretch"], stretch)
        self.assertFalse(struck["pitchstretch"].any() or struck["pitchstarts"].any())
        _, spliced = train.shuffled(lambda rng: (Frames(200), label(events, 200, reach=2, placed=True)), 256, 2, random.Random(3))
        self.assertEqual(spliced["placed"].min(), 1)

    def test_a_placed_start_asks_one_rise_within_its_reach_wherever_it_is(self):
        """A placed start asks the onset head no mark but a rise within its stretch, its reach and its mark's frame
        either side: the loss is the same for a rise at any frame of it, more for none, and more still for one just
        outside, which is asked silent; two starts whose stretches touch ask one rise between them; and a stretch the
        window's edge cuts asks none, its start maybe past the edge, nor its frames anything."""
        lab = label([note(20, 40, 60, "voice"), note(60, 70, 62, "voice"), note(69, 80, 64, "voice")], 100, reach=3, placed=True)
        within = [rises(lab, 0, 48, [k]) for k in range(16, 25)]
        self.assertEqual(set(within), {within[0]})
        none, outside = rises(lab, 0, 48, []), [rises(lab, 0, 48, [k]) for k in (15, 25)]
        self.assertGreater(none, within[0])
        self.assertGreater(min(outside), none)
        one, two = rises(lab, 50, 48, [8]), rises(lab, 50, 48, [8, 19])  # stretches 56 to 64 and 65 to 73, touching
        self.assertEqual((one, two, rises(lab, 50, 48, [21])), (one, one, one))
        self.assertGreater(rises(lab, 50, 48, []), one)
        cut = rises(lab, 18, 20, [])  # from frame 18, in the first stretch: frames 0 to 6 hold its rest
        self.assertEqual((rises(lab, 18, 20, [3]), rises(lab, 18, 20, [0, 6])), (cut, cut))
        self.assertAlmostEqual(cut, math.log1p(math.exp(-9.0)), places=6)  # the 13 frames past it, each asked silent

    def test_a_placed_note_asks_one_strike_within_its_reach_at_its_pitch(self):
        """A placed note asks the strike head, at its pitch, no mark but a rise within its stretch, its reach and its
        mark's frame either side: the same loss for a rise at any frame of it, more for none, more still for one just
        outside or at another pitch, each asked silent, and another clip's strike there is asked; a pitch's stretches
        that touch ask one rise, two pitches' two, and one the window cuts none; a rise weighs as a strike would."""
        lab = label([note(20, 40, 60, "voice"), note(60, 70, 62, "voice"), note(69, 80, 62, "voice"), note(60, 75, 64, "voice")], 100, reach=3, placed=True)
        strike = lambda start, window, at, struck=(): rises(lab, start, window, at, "strike", struck)
        within = [strike(0, 48, [(k, 60)]) for k in range(16, 25)]
        self.assertEqual(set(within), {within[0]})
        none, wrong = strike(0, 48, []), [strike(0, 48, [(k, 60)]) for k in (15, 25)] + [strike(0, 48, [(k, 61)]) for k in range(16, 25)]
        self.assertGreater(none, within[0])
        n = 48 * ripmodels.PITCHES - 9  # the cells asked silent, all but the stretch's nine: the stretch weighs one
        self.assertAlmostEqual(none, (n * math.log1p(math.exp(-9.0)) + train.STRIKE_POS * math.log1p(math.exp(9.0))) / (n + 1), places=6)
        self.assertGreater(min(wrong), none)
        self.assertAlmostEqual(min(wrong), max(wrong), places=6)  # one strike heard where none is, wherever
        self.assertGreater(strike(0, 48, [(18, 60)], [(22, 60)]), strike(0, 48, [(22, 60)], [(22, 60)]))
        both = strike(50, 48, [(8, 62), (8, 64)])  # 62's stretches 56 to 64 and 65 to 73, touching, and 64's 56 to 64
        self.assertEqual((strike(50, 48, [(8, 62), (19, 62), (8, 64)]), strike(50, 48, [(21, 62), (8, 64)])), (both, both))
        self.assertGreater(strike(50, 48, [(8, 62)]), both)
        cut = strike(18, 20, [])  # from frame 18, in the first stretch: frames 0 to 6 hold its rest
        self.assertEqual((strike(18, 20, [(3, 60)]), strike(18, 20, [(0, 60), (6, 60)])), (cut, cut))
        self.assertAlmostEqual(cut, math.log1p(math.exp(-9.0)), places=6)  # every cell past it asked silent
        again = label([note(20, 40, 60, "voice"), note(40, 60, 60, "voice")], 100, reach=3, placed=True)  # struck again as it sounds
        with mock.patch.object(train, "STRIKE_AGAIN", 3.0):
            unheard, first, second = (rises(again, 0, 100, at, "strike") for at in ([], [(20, 60)], [(40, 60)]))
        self.assertAlmostEqual((unheard - second) / (unheard - first), 3.0, places=4)

    def test_a_corpus_of_placed_starts_states_its_reach(self):
        """A corpus whose labels' starts are placed is refused where its manifest states no reach, by the line that
        measures one and writes it there; one that states it gives it to each loop."""
        with tempfile.TemporaryDirectory() as d:
            corpus(d, [{"placed": True, "events": [note(4, 20, 60, "voice")]}])
            with self.assertRaises(SystemExit) as err:
                train.load_corpus(d)
            self.assertIn(f"python3 tool/rip-lab/labels/placefit.py <truth root> reach {d}", str(err.exception))
            corpus(d, [{"placed": True, "events": [note(4, 20, 60, "voice")]}], reach=5)
            self.assertEqual(train.load_corpus(d)[1][0][1].reach, 5)

    def test_a_line_reads_the_onset_and_strike_heads_peaks_against_the_placed_starts_within_their_reach(self):
        """Over loops of placed starts a line gives the onset head's F1 at its peaks over a half against them, a peak
        right within the reach of a start and a start heard with a peak within its reach, the reach, and the strike
        head's so at their pitch, its peaks over the strikes' bars, 0.9 under E3 and 0.7 from it; a line over loops of
        none gives none of it."""
        frames = 128
        lab = label([note(20, 40, 60, "voice"), note(60, 80, 62, "voice"), note(80, 90, 48, "voice"), note(100, 120, 64, "voice")], frames, reach=3, placed=True)
        model = Heard(frames)
        onset, strike = model.cfg.slice("onset").start, model.cfg.slice("strike").start - LOW
        model.y[0, [22, 56, 101, 115], onset] = 9.0  # right, 4 frames off, right, none near
        model.y[0, 60, onset] = -0.4  # under a half
        model.y[0, [21, 61, 81, 101], [strike + 60, strike + 61, strike + 48, strike + 64]] = torch.tensor([9.0, 9.0, 1.4, 1.4])  # right, another pitch, under its bar, over it
        v = train.evaluate(model, *train.validation([(Frames(frames), lab)], frames, 2)[:2], "cpu")
        self.assertAlmostEqual(v["starts"][0], 2 * (2 / 4) * (2 / 4) / (2 / 4 + 2 / 4))
        self.assertEqual(v["starts"][1], 3)
        self.assertAlmostEqual(v["starts"][2], 2 * (2 / 3) * (2 / 4) / (2 / 3 + 2 / 4))
        with tempfile.TemporaryDirectory() as d:
            os.mkdir(os.path.join(d, "struck"))
            corpus(d, [{"placed": True, "events": [note(10, 30, 60, "voice")]}] * 2, reach=3)
            corpus(os.path.join(d, "struck"), [{"events": [note(10, 30, 60, "guitar")]}])
            said = run("--corpus", d, "--out", os.path.join(d, "out.bin"), "--holdout", "1", "--validate", os.path.join(d, "struck")).splitlines()
        held, validation = (next(line for line in said if line.startswith(f"step 1 {name} ")) for name in ("held-out", "validation"))
        self.assertRegex(held, r", placed starts' F1 \d\.\d{3} within 3 frames, their strikes' \d\.\d{3} \(")
        self.assertNotIn("placed starts", validation)

    def test_shuffled_and_bent_windows_and_drums_laid_over_carry_the_reaches(self):
        """A shuffled window's chunks of a loop of placed starts keep each start's stretches where they fall in the
        chunk, a bent window takes them nearest along its curve, cut again by its edges, and drums laid over a window
        keep them as they were, a hit inside one marked as a hit is anywhere: the onset's and at its pitch the strike's."""
        lab = label([note(40, 60, 60, "voice")], 86, reach=3, placed=True)  # a chunk long: every clip of it is the whole loop
        _, spliced = train.shuffled(lambda rng: (Frames(86), lab), 256, 2, random.Random(1))
        want = np.zeros(256, np.uint8)
        for at in (0, 86, 172):
            want[at + 36:at + 45] = 2
        self.assertEqual(spliced["stretch"].tolist(), want.tolist())
        _, bent = train.bent(Frames(86), lab, 48, 2, None, random.Random(2), None)
        self.assertTrue(bent["stretch"].any())
        for t in (spliced, bent):
            for k in ("stretch", "starts"):
                np.testing.assert_array_equal(t["pitch" + k], np.where(np.arange(ripmodels.PITCHES) == 60 - LOW, t[k][:, None], 0), k)
        hits = label([{"frame": f, "end": f, "drum": 0, "midi": 0, "vel": 0.8, "wave": -1, "family": F["drums"]} for f in (38, 70)], 86)
        _, plain = train.sample(Frames(86), lab, 0, 86, 2)
        _, laid = train.sample(Frames(86), lab, 0, 86, 2, (Frames(86), hits, 0.0), random.Random(1))
        for k in ("stretch", "pitchstretch", "pitchstarts"):
            np.testing.assert_array_equal(laid[k], plain[k], k)
        self.assertEqual((laid["onset"][37:40].tolist(), int(plain["onset"].sum())), ([1, 1, 1], 0))


class Drawn:
    """a stream whose every draw lies at one share of its range"""

    def __init__(self, share):
        self.share = share

    def uniform(self, a, b):
        return a + self.share * (b - a)


class TheRing(unittest.TestCase):
    def test_a_ring_drawn_at_a_level_reads_that_far_under_the_attacks_rise(self):
        """Frames silent until a steady tone starts, at A0 and at C8: a ring drawn at the bottom, the middle and the top
        of its levels reads that far under the tone's energy where it starts, within the 0.6 dB the sharp plane's
        bands, each over the times it counts a tone there, read a tone at A0 off its energy."""
        for midi in (21, 108):
            for share in (0.0, 0.5, 1.0):
                with self.subTest(midi=midi, share=share):
                    e = np.zeros((40, ripmodels.VALUES))
                    e[20:] = 1e-3 * ripmodels.sine_response(440 * 2 ** ((midi - 69) / 12))
                    before = e.copy()
                    train.ringing(e, Drawn(share))
                    tone = ripmodels.sine_response(math.exp(Drawn(share).uniform(*np.log(train.RING_HZ))))
                    ring = (e - before)[20] @ tone / (tone @ tone)
                    self.assertAlmostEqual(10 * math.log10(ring / 1e-3), Drawn(share).uniform(*train.RING_DB), delta=0.6)


class TheFrames(unittest.TestCase):
    def test_the_frames_features_and_stack_are_audios_own(self):
        """audio's fixture, ripmodels/testdata/frames.bin of the audio module stapes' go.mod names: band_frames,
        features and PitchNet's stack read its signal as audio's code does, to the rounding its own test allows"""
        go = shutil.which("go")
        if go is None:
            self.skipTest("no go here to find the audio module with")
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
        got = subprocess.run([go, "list", "-m", "-f", "{{.Dir}}", "github.com/lixenwraith/audio"], cwd=root, capture_output=True, text=True)
        path = os.path.join(got.stdout.strip(), "ripmodels", "testdata", "frames.bin")
        if got.returncode != 0 or not got.stdout.strip() or not os.path.exists(path):
            self.skipTest(f"the audio module's fixture is not here ({got.stderr.strip() or path}): go mod download github.com/lixenwraith/audio")
        with open(path, "rb") as f:
            body = f.read()
        first, n, values, inputs, stacked = struct.unpack_from("<5I", body)
        self.assertEqual((values, inputs, stacked), (ripmodels.VALUES, 4 * ripmodels.INPUTS, ripmodels.STACK_CHANNELS * (ripmodels.STACK_BINS + 1)))
        v = np.frombuffer(body, "<f4", offset=20).astype(np.float64)
        want = {"frames": v[:n * values].reshape(n, values), "features": v[n * values:n * (values + inputs)].reshape(n, inputs), "stack": v[n * (values + inputs):]}
        s = np.arange(4 * ripmodels.RATE) / ripmodels.RATE  # audio's fixtureSignal
        x = 0.3 * np.sin(2 * np.pi * 27.5 * s + 0.3) + 0.2 * np.sin(2 * np.pi * 29.135 * s) + 0.1 * np.sin(2 * np.pi * 440 * s) * np.exp(-s)
        x = x + np.where(np.mod(s, 0.5) < 0.1, 0.05 * np.sin(2 * np.pi * 3520 * s), 0.0)
        frames = ripmodels.band_frames(x)
        features = ripmodels.features(frames[:first + n], 4)
        with torch.no_grad():
            stack = ripmodels.PitchNet(CFG).stack(torch.from_numpy(features[first:first + 1])[None])[0, 0].numpy()
        for name, mine in (("frames", frames[first:first + n]), ("features", features[first:first + n]), ("stack", stack)):
            with self.subTest(name):
                self.assertEqual(mine.shape, want[name].shape)
                off = np.abs(mine.astype(np.float64) - want[name]) - (1e-6 + 1e-5 * np.abs(want[name]))
                self.assertLessEqual(float(off.max()), 0.0, f"value {np.unravel_index(off.argmax(), off.shape)} off")



def tones(frames, onsets) -> np.ndarray:
    """a recording at the rip's rate of frames frames, a tone of 0.4 s from each onset's frame, a major third over the last"""
    s = np.arange(frames * ripmodels.HOP) / ripmodels.RATE
    x = np.zeros(len(s))
    for k, f in enumerate(onsets):
        t = f * ripmodels.HOP / ripmodels.RATE
        x += ((s >= t) & (s < t + 0.4)) * 0.3 * np.sin(2 * np.pi * 220 * 2 ** (k / 3) * (s - t))
    return x


class TheDatacheck(unittest.TestCase):
    """datacheck.py's readings over a small corpus of loops of four tones, each loop's label naming its sound"""
    FRAMES, ONSETS = 600, (100, 200, 300, 400)

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        tone = ripmodels.band_frames(tones(cls.FRAMES, cls.ONSETS))
        nan = tone.copy()
        nan[50, ripmodels.BANDS + 3] = np.nan
        soft = tones(cls.FRAMES, cls.ONSETS[:1] + cls.ONSETS[2:]) + 10 ** (-30 / 20) * tones(cls.FRAMES, cls.ONSETS[1:2])
        cls.sounds = {"tones": tone, "quieter": ripmodels.band_frames(tones(cls.FRAMES, cls.ONSETS) / 32768), "silent": np.zeros_like(tone), "cut": tone[:-1], "nan": nan,
                      "soft": ripmodels.band_frames(soft)}

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def corpus(self, name, labels, **manifest) -> str:
        d = os.path.join(self.dir.name, name)
        os.mkdir(d)
        corpus(d, [{"frames": self.FRAMES, **lab} for lab in labels], self.FRAMES, lambda lab: self.sounds[lab.get("sound", "tones")],
               drums=["kick", "snare", "hihat"], families=list(ripmodels.FAMILIES), **manifest)
        return d

    def test_a_label_late_reads_the_rise_earlier(self):
        """Notes labelled at their onsets read the sound's rise within a frame of them, the sharp plane's window
        centred on the frame, and the same notes labelled four frames late read it four frames earlier."""
        lags = []
        for late in (0, 4):
            v = datacheck.check(self.corpus(f"late{late}", [{"events": [note(f + late, f + late + 30, 60, "piano") for f in self.ONSETS]}]))
            self.assertEqual((v["notes"], v["lags"]["hit"], v["lags"]["beat"]), ((0, 4), None, None))
            lags.append(v["lags"]["note"][0])
        self.assertIn(lags[0], (-1, 0, 1))
        self.assertEqual(lags[1], lags[0] - 4)

    def test_a_note_alone_far_under_the_loudest_is_not_in_its_loops_quiet(self):
        """A note 30 dB under the loop's loudest, nothing else sounding, is out of the loop's quiet: a frame's loudness is
        its energy over the sharp plane, which no band the note leaves empty lowers."""
        v = datacheck.check(self.corpus("soft", [{"sound": "soft", "events": [note(f, f + 30, 60, "piano") for f in self.ONSETS]}]))
        self.assertEqual(v["notes"], (0, 4))

    def test_a_corpus_reads_its_faults_at_any_scale(self):
        """A loop cut short, one not finite, labels out of their loop or the piano's keys, a doubled label, a loop of
        none, and a note and a hit over digital silence: each is counted, the silent loop's frames all quiet and its
        note and hit in silence; and a loop at 1/32768 of the scale reads as it does at its own."""
        hit = {"frame": 200, "end": 200, "drum": 0, "midi": 0, "vel": 0.8, "wave": -1, "family": F["drums"]}
        labels = [{"events": [note(100, 130, 60, "piano"), note(100, 130, 60, "piano"), note(200, 230, 110, "piano"), note(300, self.FRAMES + 9, 64, "piano")]},
                  {"sound": "silent", "events": [note(100, 130, 60, "piano"), hit]}, {"events": []}, {"events": [note(self.FRAMES + 5, self.FRAMES + 9, 60, "piano")]},
                  {"sound": "cut", "events": [note(100, 130, 60, "piano")]}, {"sound": "nan", "events": [note(100, 130, 60, "piano")]}]
        v = datacheck.check(self.corpus("faults", labels))
        self.assertEqual(v["bad"], {"size": 1, "nan": 1, "rng": 1})
        self.assertEqual((v["doubled"], v["empty"], v["outside"], v["past"]), ((1, 7), 1, 1, 2))  # past: a note ending late, one past its loop
        self.assertEqual((v["notes"][0], v["hits"]), (1, (1, 1)))
        self.assertEqual(v["quiet"][1], 4 * self.FRAMES)  # the loops whose sharp plane is finite and whole, one silent
        self.assertGreater(v["quiet"][0], self.FRAMES)
        self.assertEqual(str(datacheck.check(self.corpus("quieter", [{**labels[0], "sound": "quieter"}] + labels[1:]))), str(v))

    def test_a_manifest_of_another_layout_is_refused_by_name(self):
        for layout in (None, 0):
            with self.subTest(layout=layout):
                with self.assertRaises(SystemExit) as err:
                    datacheck.check(self.corpus(f"layout{layout}", [{"events": []}], layout=layout))
                self.assertIn(ripmodels.LAYOUT_NAME, str(err.exception))

    def test_the_script_reads_a_corpus(self):
        """datacheck.py run on a corpus prints its head and the corpus's line, every loop sampled."""
        d = self.corpus("script", [{"events": [note(f, f + 30, 60, "piano") for f in self.ONSETS]}] * 2)
        said = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "datacheck.py"), d],
                              capture_output=True, text=True, cwd=self.dir.name)
        self.assertEqual(said.returncode, 0, said.stderr)
        lines = said.stdout.splitlines()
        self.assertEqual(lines[0], datacheck.HEAD)
        self.assertRegex(lines[1], r"^script +2 +2 +0 +0 +0 +0\.0% +0\.0% \|")


class TheRoundsTable(unittest.TestCase):
    def test_a_set_under_which_a_kind_fell_is_said_before_the_rows(self):
        """A read-back's log gives a row a kind, what is struck beside its lines' hits, a minute where none plays, as the
        notes of a kind whose truth gives none, and the rips whose grid the joint path decoded; a fitted set's falls
        against the start are the first line, and with them such notes where they rose."""
        logs = {"start": "guitar: F1 0.817 before, 0.817 after\n"
                         "guitar: grids tracked 18 before, 17 after, joint 2, 3, cues 0, 0\n"
                         "guitar: ends 0.400 before, 0.400 after, of 2794 note(s) matched whose end is given\n"
                         "guitar: ends 1676 wrong, by stage: strikes early 1418 before and after\n"
                         "guitar: struck 0.2 strike(s) a minute before, 0.3 after, the lines' hits 0.1, 0.1, heard where none plays, against 231 instant(s) given elsewhere\n"
                         "voice: F1 0.424 before, 0.430 after\n"
                         "E-GMD: notes 35.9 a minute before, 35.9 after, heard where none plays\n"
                         "E-GMD: struck F1 0.912 before, 0.915 after, the lines' hits 0.882, 0.880, by time alone over 7767 instant(s) given; each of their 13622 strike(s) its own hit, 0.677, 0.678, the lines' 0.754, 0.753, and one hit an instant 0.726 at most\n",
                "fit-voice": "guitar: F1 0.790 before, 0.790 after\nvoice: F1 0.470 before, 0.470 after\n"
                             "E-GMD: notes 41.2 a minute before, 41.2 after, heard where none plays\n"
                             "E-GMD: struck F1 0.901 before, 0.901 after, the lines' hits 0.882, 0.882, by time alone\n"}
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for name, text in logs.items():
                paths.append(os.path.join(d, f"readback-{name}.log"))
                with open(paths[-1], "w") as f:
                    f.write(text)
            lines = table.table(paths).splitlines()
        self.assertEqual(lines[0], "fell under fit-voice, against start: notes on guitar 0.817 to 0.790, struck on E-GMD 0.912 to 0.901, "
                                   "notes where none plays on E-GMD 35.9 to 41.2 a minute")
        self.assertEqual(lines[2].split(), ["guitar", "start", "0.817", "0.400", "1118", "of", "2794", "-", "-", "0.2/min", "0.1/min", "-", "2", "of", "20"] + ["-"] * 3)
        self.assertEqual(lines[5].split()[:3], ["voice", "fit-voice", "0.470"])
        self.assertEqual(lines[6].split(), ["E-GMD", "start", "35.9/min"] + ["-"] * 4 + ["0.912", "0.882"] + ["-"] * 5)


if __name__ == "__main__":
    unittest.main()
