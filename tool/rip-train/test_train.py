"""The trainer's rules, each a test: python3 -m unittest discover tool/rip-train"""
import contextlib
import io
import json
import os
import random
import struct
import sys
import tempfile
import unittest
from dataclasses import asdict
from unittest import mock

import numpy as np
import torch

import average
import ripmodels
import train

MANIFEST = {"mel_bands": ripmodels.MEL_BANDS, "min_midi": 28, "pitches": ripmodels.PITCHES, "voice_genes": ripmodels.VOICE_GENES}
F = {k: i for i, k in enumerate(ripmodels.FAMILIES)}
CFG = ripmodels.Config(inputs=ripmodels.STACK_INPUTS, channels=ripmodels.STACK_CHANNELS, outputs=ripmodels.PITCHNET_HEADS, cell=4)


def voice(first: float) -> list:
    """an engine voice's genes, its first gene telling it from another"""
    return [first] + [0.5] * (ripmodels.VOICE_GENES - 1)


def note(frame, end, midi, family, genes=None, wave=-1, **more) -> dict:
    return {"frame": frame, "end": end, "drum": -1, "midi": midi, "vel": 0.8, "family": F[family], "wave": wave, **({"voice": genes} if genes else {}), **more}


def label(events, frames=40, **more) -> train.Label:
    return train.Label({"events": events, **more}, frames, MANIFEST)


class Frames:
    """a loop's frames in memory, as train.MelFile gives them"""

    def __init__(self, n):
        self.path, self.mel = f"memory{id(self)}", np.zeros((n, ripmodels.MEL_BANDS), np.float32)

    def __len__(self):
        return len(self.mel)

    def __getitem__(self, sl):
        return self.mel[sl]


class Said(torch.nn.Module):
    """a notes model that says what a test wrote into its scores"""

    def __init__(self, frames):
        super().__init__()
        self.cfg, self.y = CFG, torch.full((1, frames, CFG.out_size), -9.0)
        self.y[..., CFG.slice("pitchvoice")] = 0.0  # every gene a half
        self.register_buffer("sure", torch.zeros(()))

    def say(self, a, b, midi, salience=9.0, family=None, wave=None, gene=None):
        """over frames a to b the pitch sounds, played by a family on a wave, its voice's first gene given"""
        p = midi - MANIFEST["min_midi"]
        self.y[0, a:b, self.cfg.slice("pitch")][:, p] = salience
        for head, value, score in (("pitchfamily", family and F[family], 9.0), ("pitchwave", wave, 9.0), ("pitchvoice", 0, gene)):
            if value is not None and score is not None:
                self.y[0, a:b, self.cfg.slice(head)][:, value * ripmodels.PITCHES + p] = score

    def forward(self, mel):
        return self.y[:, :mel.shape[1]]


class WhatPlaysEachPitch(unittest.TestCase):
    def tearDown(self):
        train.FAMILY_HOLDOUT = None

    def test_a_cell_says_what_plays_it_where_one_note_alone_sounds(self):
        """A cell one note sounds in is asked that note's family, wave and voice: an engine voice far from the
        bank's a synth, a recorded note its family with no wave or voice, one of no family nothing; a note that
        moves in each cell it sounds in; and a cell two notes sound in nothing, though both sound."""
        bend = [0.0] * 5 + [100.0] * 5
        lab = label([note(0, 10, 60, "piano", voice(0.1), 4), note(0, 10, 64, "strings"), note(0, 10, 67, "unknown", voice(0.2), 1),
                     note(0, 10, 70, "unknown"), note(0, 10, 72, "piano", voice(0.1), 4), note(5, 15, 72, "guitar", voice(0.3), 3),
                     note(20, 30, 50, "bass", voice(0.4), 4, cents=bend)])
        t = lab.window(0, 40)
        cell = lambda key, frame, midi: int(t[key][frame, midi - 28])
        rows = train.genes()
        for frame, midi, family, wave, first in ((3, 60, "piano", 4, 0.1), (3, 64, "strings", -1, None), (3, 67, "synth", 1, 0.2), (3, 70, None, -1, None),
                                                 (3, 72, "piano", 4, 0.1), (7, 72, None, -1, None), (12, 72, "guitar", 3, 0.3),
                                                 (22, 50, "bass", 4, 0.4), (27, 51, "bass", 4, 0.4), (27, 50, None, -1, None)):
            with self.subTest(frame=frame, midi=midi):
                self.assertEqual(cell("pitchfamily", frame, midi), F[family] if family else -1)
                self.assertEqual(cell("pitchwave", frame, midi), wave)
                got = cell("pitchvoice", frame, midi)
                self.assertEqual(None if got < 0 else round(float(rows[got][0]), 3), first)
        self.assertEqual([int(t["pitch"][f, m - 28]) for f, m in ((3, 70), (7, 72), (27, 51), (27, 50))], [1, 1, 1, 0])

    def test_a_family_held_out_is_named_to_no_head(self):
        """The notes of the family held out stay in the run, their sound, wave and voice taught, but neither the
        cells' head nor the frame's is told their family."""
        train.FAMILY_HOLDOUT = F["guitar"]
        t = label([note(0, 10, 60, "guitar", voice(0.3), 3), note(0, 20, 48, "bass", voice(0.4), 4)]).window(0, 20)
        self.assertEqual([int(t["pitchfamily"][3, m - 28]) for m in (60, 48)], [-1, F["bass"]])
        self.assertEqual([int(t["family"][f]) for f in (3, 15)], [-1, F["bass"]])
        self.assertEqual([int(t[k][3, 60 - 28]) for k in ("pitch", "pitchwave")], [1, 3])
        self.assertGreaterEqual(int(t["pitchvoice"][3, 60 - 28]), 0)

    def test_a_note_is_heard_as_its_cells_say_by_their_salience(self):
        """A note is named by its cells' say, each by its salience, over the cells among its frame's VOICE_CELLS most
        salient; the lines give the share named right, by family and under another note, the waves' and the genes'
        miss; and the surest name SURE of a held-out family's notes stay under is the measure."""
        train.FAMILY_HOLDOUT = F["guitar"]
        events = [note(0, 20, 72, "piano", voice(0.2), 4), note(0, 20, 40, "bass", voice(0.6), 3), note(20, 40, 60, "voice", voice(0.8), 1)]
        events += [note(40 + 10 * i, 50 + 10 * i, 55, "guitar") for i in range(21)]
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
            model.y[0, 40 + 10 * i:50 + 10 * i, CFG.slice("pitchfamily")][:, F["synth"] * ripmodels.PITCHES + 55 - 28] = 9.0 - 0.5 * i
        mels, tgs, notes = train.validation([(Frames(frames), label(events, frames))], frames, 2)
        v = train.evaluate(model, mels, tgs, "cpu", 24, notes)["voices"]
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
        """The file names the heads of what plays each pitch after the cents, a value a pitch each, and the measure as
        a frame head whose weights are zero and whose bias is the measure."""
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
                torch.save({"model": model.state_dict(), "cfg": asdict(CFG)}, file)

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


def corpus(d, labels, frames=64):
    """a corpus on disk of loops of silence, a label each"""
    with open(os.path.join(d, "corpus.json"), "w") as f:
        json.dump({**MANIFEST, "loops": len(labels), "waves": ["sine", "saw", "square", "string", "modal"]}, f)
    for i, lab in enumerate(labels):
        with open(os.path.join(d, f"loop_{i:03d}.mel"), "wb") as f:
            f.write(struct.pack("<II", frames, ripmodels.MEL_BANDS) + bytes(4 * frames * ripmodels.MEL_BANDS))
        with open(os.path.join(d, f"loop_{i:03d}.json"), "w") as f:
            json.dump(lab, f)


def run(*flags, steps=1) -> str:
    """steps of train.py at the smallest shape, and what it printed; on the CPU whatever the machine has, as every
    tensor a test builds is there"""
    argv = ["train.py", *flags, "--steps", str(steps), "--batch", "1", "--window", "32", "--dim", "8", "--heads", "2", "--layers", "1", "--ffn", "16", "--attention-window", "4", "--device", "cpu"]
    with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()) as said:
        train.main()
    return said.getvalue()


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

    def forward(self, mel):
        return self.y[:, :mel.shape[1]]


class TheBeatAndTheBarLine(unittest.TestCase):
    def test_a_rendered_loop_is_asked_the_beats_and_lines_of_its_grid(self):
        """A loop's beats are its meter's own note, a quarter in 4/4 and 3/4 and an eighth in 6/8, on the frames its
        tempo's sixteenth of whole samples rounds to (6,615 at 100 bpm, 12.92 frames), a bar line every bar's beats
        from frame 0, each marked a frame either side and asked to a frame short of the mark of the end's line,
        which no sound gives; a window takes its share of them."""
        quarters, eighths = [0, 52, 103, 155, 207, 258, 310, 362], [0, 26, 52, 78, 103, 129, 155, 181, 207, 233, 258, 284]
        for meter, frames, beats, lines, end in ((None, 414, quarters, [0, 207], 413), ("3/4", 311, quarters[:6], [0, 155], 310), ("6/8", 311, eighths, [0, 155], 310)):
            with self.subTest(meter=meter):
                lab = label([], frames, bpm=100, bars=2, **({"meter": meter} if meter else {}))
                t = lab.window(0, frames)
                self.assertEqual(t["beat"].tolist(), marks(frames, beats))
                self.assertEqual(t["downbeat"].tolist(), marks(frames, lines))
                self.assertEqual(t["metered"].tolist(), [1] * (end - 1) + [0] * (frames - end + 1))
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
        grid = label([], frames, bpm=100, bars=8)
        hits = [{"frame": f, "end": f, "drum": hat, "midi": 0, "vel": 0.8, "wave": -1, "family": F["drums"]} for f in grid.beats.tolist()]
        hits += [{**h, "drum": kick} for h in hits[::4]]
        _, t = train.bent(Frames(frames), label(hits, frames, bpm=100, bars=8), 512, 1, None, random.Random(2), None)
        asked = t["metered"] > 0
        self.assertGreater(int(asked.sum()), 400)
        self.assertEqual(t["beat"][asked].tolist(), t["drums"][asked, hat].tolist())
        self.assertEqual(t["downbeat"][asked].tolist(), t["drums"][asked, kick].tolist())
        apart = np.diff(np.flatnonzero(np.diff(t["beat"].astype(int)) == 1))  # from one mark's first frame to the next's
        self.assertFalse(set(apart.tolist()) <= {51, 52}, apart)

    def test_a_window_with_no_grid_of_its_own_is_asked_no_beat(self):
        """A window spliced of two clips and one real drums are laid under hold hits of another grid, and the padding
        past a loop's end holds none: each asks the beat and the bar line nothing, where a plain window asks its loop's."""
        frames, rng = 414, random.Random(1)  # two bars of 4/4 at 100 bpm, the end's line at frame 413
        mel, lab = Frames(frames), label([{"frame": 52, "end": 52, "drum": 0, "midi": 0, "vel": 0.8, "wave": -1, "family": F["drums"]}], frames, bpm=100, bars=2)
        _, spliced = train.shuffled(lambda: (mel, lab), 256, 1, rng)
        _, laid = train.sample(mel, lab, 0, 256, 1, (mel, lab, 0.0), rng)
        _, plain = train.sample(mel, lab, 200, 256, 1)
        self.assertEqual([int(t["metered"].sum()) for t in (spliced, laid)], [0, 0])
        self.assertEqual(plain["metered"].tolist(), [1] * (412 - 200) + [0] * (256 - 212))

    def test_the_two_heads_are_taught_as_posteriors(self):
        """Each head's loss is cross-entropy over the frames asked, a beat weighing as a frame without one: a head
        that says everywhere the share of those frames a beat marks scores that share's entropy, and no lower saying
        more or less, which is what audio reads a posterior for. Both count whole in the total."""
        frames = 414  # two bars of 4/4 at 100 bpm in a window of 512: 412 frames asked, 23 in a beat's mark and 5 in a line's
        mels, tgs, _ = train.validation([(Frames(frames), label([], frames, bpm=100, bars=2))], 512, 1)
        model = Heard(512)

        def loss(beat, downbeat):
            model.y[..., model.cfg.slice("beat")], model.y[..., model.cfg.slice("downbeat")] = beat, downbeat
            return {k: v.item() for k, v in train.losses(model, mels[0], tgs[0]).items()}
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
        loop, recording = label([], frames, bpm=100, bars=2), label([note(0, 10, 60, "guitar")], frames, source="a.wav")
        model = Heard(512)
        beat, downbeat = model.cfg.slice("beat").start, model.cfg.slice("downbeat").start
        model.y[0, [0, 54, 103, 155, 180, 258, 310, 413], beat] = 9.0  # six beats heard, one two frames late; one that is none; one past the frames asked
        model.y[0, 362, beat] = -0.1  # and one under a half
        model.y[0, 207, downbeat] = 9.0
        v = train.evaluate(model, *train.validation([(Frames(frames), loop)], 512, 1)[:2], "cpu")
        self.assertAlmostEqual(v["beats"][0], 2 * (6 / 7) * (6 / 8) / (6 / 7 + 6 / 8))
        self.assertAlmostEqual(v["beats"][1], 2 * 1 * (1 / 2) / (1 + 1 / 2))
        none = train.evaluate(model, *train.validation([(Frames(frames), recording)], 512, 1)[:2], "cpu")
        self.assertEqual((none["beats"], "beat" in none, "downbeat" in none, "onset" in none), (None, False, False, True))
        with tempfile.TemporaryDirectory() as d:
            os.mkdir(os.path.join(d, "recordings"))
            corpus(d, [{"bpm": 100, "bars": 1, "events": []}] * 2)
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
            torch.save({"model": before, "cfg": asdict(old)}, at("old.pt"))
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
        self.assertIn(head("family", len(ripmodels.FAMILIES)) + head("beat", 1) + head("downbeat", 1) + struct.pack("<I", len(after["model"])), body)

    def test_a_probe_trains_the_two_heads_alone_on_a_frame_model_held_as_it_was(self):
        """--probe on a frame model started from one without the beat and downbeat heads trains those two heads' rows
        of the output layer and holds the rest: after three steps every other weight is the start's, every other
        head says bit for bit what the start's did, the two heads say something else, and the last step is exported."""
        old = ripmodels.Config(dim=8, heads=2, layers=1, ffn=16, window=4, outputs=ripmodels.HEADS[:-2])
        before = ripmodels.FrameModel(old).state_dict()
        with tempfile.TemporaryDirectory() as d:
            at = lambda name: os.path.join(d, name)
            corpus(d, [{"bpm": 100, "bars": 1, "events": [note(0, 10, 60, "piano")]}])
            torch.save({"model": before, "cfg": asdict(old)}, at("old.pt"))
            run("--corpus", d, "--out", at("start.bin"), "--init", at("old.pt"), "--checkpoint", at("start.pt"), "--lr", "0")  # the start: the two rows as grown
            said = run("--corpus", d, "--out", at("probed.bin"), "--init", at("old.pt"), "--checkpoint", at("probed.pt"), "--probe", "--lr", "0.01", steps=3)
            start, probed = torch.load(at("start.pt")), torch.load(at("probed.pt"))
        self.assertIn("a probe: exporting its last step, 3", said)
        had = old.out_size
        for k, v in before.items():
            with self.subTest(tensor=k):
                self.assertTrue(torch.equal(probed["model"][k][:had] if k.startswith("out.") else probed["model"][k], v))
        frames = torch.randn(1, 40, ripmodels.INPUTS, generator=torch.Generator().manual_seed(1))
        says = []
        for ck in (start, probed):
            model = ripmodels.FrameModel(ripmodels.Config(**ck["cfg"]))
            model.load_state_dict(ck["model"])
            with torch.no_grad():
                says.append(model.eval()(frames))
        self.assertTrue(torch.equal(says[0][..., :had], says[1][..., :had]))
        for k in (had, had + 1):
            self.assertFalse(torch.equal(says[0][..., k], says[1][..., k]))


if __name__ == "__main__":
    unittest.main()
