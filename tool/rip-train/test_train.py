"""The trainer's rules, each a test: python3 -m unittest discover tool/rip-train"""
import array
import contextlib
import hashlib
import io
import json
import math
import os
import random
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
import train

# a corpus of the frames the models hear: its manifest states their layout, and its labels hold every note
MANIFEST = {"layout": ripmodels.LAYOUT, "voice_genes": ripmodels.VOICE_GENES}
F = {k: i for i, k in enumerate(ripmodels.FAMILIES)}
LOW = ripmodels.LOW  # the piano's lowest key, a label's first cell
CFG = ripmodels.Config(inputs=ripmodels.STACK_BINS + 1, channels=ripmodels.STACK_CHANNELS, outputs=ripmodels.PITCHNET_HEADS, cell=4)


def voice(first: float) -> list:
    """an engine voice's genes, its first gene telling it from another"""
    return [first] + [0.5] * (ripmodels.VOICE_GENES - 1)


def note(frame, end, midi, family, genes=None, wave=-1, **more) -> dict:
    return {"frame": frame, "end": end, "drum": -1, "midi": midi, "vel": 0.8, "family": F[family], "wave": wave, **({"voice": genes} if genes else {}), **more}


def label(events, frames=40, **more) -> train.Label:
    return train.Label({"events": events, **more}, frames, MANIFEST)


def rendered(events, frames, bars) -> train.Label:
    """a rendered loop's label, bars of 4/4 held at 100 bpm from frame 0, its beats and bar lines as corpus lists them"""
    beats = [int(k * 51.68 + 0.5) for k in range(bars * 4)]
    return label(events, frames, bpm=100, bars=bars, beats=beats, downbeats=beats[::4])


class Frames:
    """a loop's frames in memory, as train.FrameFile gives them"""

    def __init__(self, n):
        self.path, self.fr = f"memory{id(self)}", np.zeros((n, ripmodels.VALUES), np.float32)

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

    def test_the_weight_file_carries_the_heads_and_the_measure(self):
        """The file names the heads of what plays each pitch after the cents, a value a pitch each, states the layout
        of the frames it hears as a constant after the heads, and carries the measure as a frame head whose weights
        are zero and whose bias is the measure."""
        model = ripmodels.PitchNet(CFG)
        model.sure.fill_(0.75)
        with tempfile.TemporaryDirectory() as d:
            ripmodels.export(model, os.path.join(d, "notes.bin"))
            with open(os.path.join(d, "notes.bin"), "rb") as f:
                body = f.read()
        at = 4 + 16 + len(ripmodels._cells_header(model))

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
        heads = [(name(), take("I")) for _ in range(take("I"))]
        self.assertEqual([(name(), take("f")) for _ in range(take("I"))], [("layout", float(ripmodels.LAYOUT))])
        p = ripmodels.PITCHES
        self.assertEqual(heads[4:8], [("cents", 2 * p), ("pitchfamily", ripmodels.NAMES * p), ("pitchwave", ripmodels.WAVES * p), ("pitchvoice", ripmodels.VOICE_GENES * p)])
        self.assertEqual(heads[-1], ("sure", 1))
        tensors = {}
        for _ in range(take("I")):
            k, dims = name(), [take("I") for _ in range(take("B"))]
            tensors[k] = np.frombuffer(body, "<f4", int(np.prod(dims)), at).reshape(dims)
            at += 4 * int(np.prod(dims))
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
        self.assertRegex(held, r", beats' F1 \d\.\d{3}, bar lines' \d\.\d{3} \(.* beat \d\.\d{3} downbeat \d\.\d{3}\)")
        self.assertIn(", no beat or bar line asked (", validation)
        self.assertNotIn("beat ", validation.split("asked (")[1])

    def test_init_grows_the_heads_and_carries_every_other_weight(self):
        """A frame model from before the beat and downbeat heads starts a run through --init: every weight carries
        over, the output layer's rows of the heads it had among them, the two heads' rows are new, and the weight
        file lists them after the family."""
        old = ripmodels.Config(dim=8, heads=2, layers=1, ffn=16, window=4, outputs=ripmodels.HEADS[:-2])
        before = ripmodels.FrameModel(old).state_dict()
        with tempfile.TemporaryDirectory() as d:
            at = lambda name: os.path.join(d, name)
            corpus(d, [{"bpm": 120, "bars": 1, "events": [note(0, 10, 60, "piano")]}])
            torch.save({"model": before, "layout": ripmodels.LAYOUT, "cfg": asdict(old)}, at("old.pt"))
            said = run("--corpus", d, "--out", at("new.bin"), "--init", at("old.pt"), "--checkpoint", at("new.pt"), "--lr", "0")
            after = torch.load(at("new.pt"))
            with open(at("new.bin"), "rb") as f:
                body = f.read()
        self.assertIn("its heads but beat, downbeat, which start fresh", said)
        self.assertEqual([tuple(h) for h in after["cfg"]["outputs"]], ripmodels.HEADS)
        had = old.out_size
        for k, v in before.items():
            with self.subTest(tensor=k):
                self.assertTrue(torch.equal(after["model"][k][:had] if k.startswith("out.") else after["model"][k], v))
        self.assertEqual([len(after["model"][k]) for k in ("out.weight", "out.bias")], [had + 2, had + 2])
        head = lambda name, size: struct.pack("<H", len(name)) + name.encode() + struct.pack("<I", size)
        layout = struct.pack("<IH", 1, 6) + b"layout" + struct.pack("<f", ripmodels.LAYOUT)
        self.assertIn(head("family", len(ripmodels.FAMILIES)) + head("beat", 1) + head("downbeat", 1) + layout + struct.pack("<I", len(after["model"])), body)

    def test_a_probe_trains_the_two_heads_alone_on_a_frame_model_held_as_it_was(self):
        """--probe on a frame model started from one without the beat and downbeat heads trains those two heads' rows
        of the output layer and holds the rest: after three steps every other weight is the start's, every other
        head says bit for bit what the start's did, the two heads say something else, and the last step is exported."""
        old = ripmodels.Config(dim=8, heads=2, layers=1, ffn=16, window=4, outputs=ripmodels.HEADS[:-2])
        before = ripmodels.FrameModel(old).state_dict()
        with tempfile.TemporaryDirectory() as d:
            at = lambda name: os.path.join(d, name)
            corpus(d, [{"bpm": 100, "bars": 1, "events": [note(0, 10, 60, "piano")]}])
            torch.save({"model": before, "layout": ripmodels.LAYOUT, "cfg": asdict(old)}, at("old.pt"))
            run("--corpus", d, "--out", at("start.bin"), "--init", at("old.pt"), "--checkpoint", at("start.pt"), "--lr", "0")  # the start: the two rows as grown
            said = run("--corpus", d, "--out", at("probed.bin"), "--init", at("old.pt"), "--checkpoint", at("probed.pt"), "--probe", "--lr", "0.01", steps=3)
            start, probed = torch.load(at("start.pt")), torch.load(at("probed.pt"))
        self.assertIn("a probe: exporting its last step, 3", said)
        had = old.out_size
        for k, v in before.items():
            with self.subTest(tensor=k):
                self.assertTrue(torch.equal(probed["model"][k][:had] if k.startswith("out.") else probed["model"][k], v))
        frames = torch.randn(1, 40, 2 * ripmodels.INPUTS, generator=torch.Generator().manual_seed(1))
        says = []
        for ck in (start, probed):
            model = ripmodels.FrameModel(ripmodels.shape(ck["cfg"]))
            model.load_state_dict(ck["model"])
            with torch.no_grad():
                says.append(model.eval()(frames))
        self.assertTrue(torch.equal(says[0][..., :had], says[1][..., :had]))
        for k in (had, had + 1):
            self.assertFalse(torch.equal(says[0][..., k], says[1][..., k]))

    def test_a_family_unasked_has_no_term_and_its_rows_stay_zero(self):
        """--family-unasked on a frame model grown from the nine heads before the family: the step's loss has no
        family term, and after three steps of a free body the family's rows of the output layer are zero."""
        old = ripmodels.Config(dim=8, heads=2, layers=1, ffn=16, window=4, outputs=ripmodels.HEADS[:-3])
        before = ripmodels.FrameModel(old).state_dict()
        with tempfile.TemporaryDirectory() as d:
            at = lambda name: os.path.join(d, name)
            corpus(d, [{"bpm": 100, "bars": 1, "events": [note(0, 10, 60, "piano")]}])
            torch.save({"model": before, "layout": ripmodels.LAYOUT, "cfg": asdict(old)}, at("old.pt"))
            said = run("--corpus", d, "--out", at("new.bin"), "--init", at("old.pt"), "--checkpoint", at("new.pt"), "--family-unasked", "--lr", "0.01", steps=3)
            after = torch.load(at("new.pt"))["model"]
        self.assertIn("its heads but family, beat, downbeat, which start fresh", said)
        last = next(line for line in said.splitlines() if line.startswith("step 3 loss "))
        self.assertIn(" downbeat ", last)
        self.assertNotIn("family", last)
        family = ripmodels.Config().slice("family")
        self.assertEqual([after[k][family].abs().sum().item() for k in ("out.weight", "out.bias")], [0, 0])
        self.assertFalse(all(torch.equal(after[k], v) for k, v in before.items() if not k.startswith("out.")))


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

    def test_a_windows_targets_are_pinned(self):
        """A digest of every target over windows cut anywhere and padded, of labels whose notes hold their pitch or move,
        placed or not, two to a cell or outside the pitches, beside drums: a change to how a window is built leaves it,
        and a change to what a window holds pins it again."""
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
                    for k in sorted(t):
                        digest.update(f"{k} {t[k].dtype} {t[k].shape}".encode() + np.ascontiguousarray(t[k]).tobytes())
        self.assertEqual(digest.hexdigest(), "fe0a3c89add1305d577616ac935b69ddb7754cf5")


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
        """A note whose label's vel is -1, as corpus writes for a list whose velocity column says none, is struck as
        any and asks the velocity head nothing: its cells' velocity target is unheard, and whatever the head says
        there moves no loss, while a note of a velocity is asked it."""
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


if __name__ == "__main__":
    unittest.main()
