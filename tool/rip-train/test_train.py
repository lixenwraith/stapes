"""The trainer's rules, each a test: python3 -m unittest discover tool/rip-train"""
import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from dataclasses import asdict
from unittest import mock

import numpy as np
import torch

import ripmodel
import train

MANIFEST = {"mel_bands": ripmodel.MEL_BANDS, "min_midi": 28, "pitches": ripmodel.PITCHES, "voice_genes": ripmodel.VOICE_GENES}
F = {k: i for i, k in enumerate(ripmodel.FAMILIES)}
CFG = ripmodel.Config(inputs=ripmodel.STACK_INPUTS, channels=ripmodel.STACK_CHANNELS, outputs=ripmodel.PITCHNET_HEADS, cell=4)


def voice(first: float) -> list:
    """an engine voice's genes, its first gene telling it from another"""
    return [first] + [0.5] * (ripmodel.VOICE_GENES - 1)


def note(frame, end, midi, family, genes=None, wave=-1, **more) -> dict:
    return {"frame": frame, "end": end, "drum": -1, "midi": midi, "vel": 0.8, "family": F[family], "wave": wave, **({"voice": genes} if genes else {}), **more}


def label(events, frames=40) -> train.Label:
    return train.Label({"events": events}, frames, MANIFEST)


class Frames:
    """a loop's frames in memory, as train.MelFile gives them"""

    def __init__(self, n):
        self.path, self.mel = f"memory{id(self)}", np.zeros((n, ripmodel.MEL_BANDS), np.float32)

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
                self.y[0, a:b, self.cfg.slice(head)][:, value * ripmodel.PITCHES + p] = score

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
            model.y[0, 40 + 10 * i:50 + 10 * i, CFG.slice("pitchfamily")][:, F["synth"] * ripmodel.PITCHES + 55 - 28] = 9.0 - 0.5 * i
        mels, tgs, notes = train.validation([(Frames(frames), label(events, frames))], frames, 2)
        v = train.evaluate(model, mels, tgs, "cpu", 24, notes)["voices"]
        self.assertEqual((round(v["families"][0], 6), v["families"][1]), (round(2 / 3, 6), 3))
        self.assertEqual(v["by"], [("piano", 1.0, 1), ("voice", 1.0, 1), ("bass", 0.0, 1)])
        self.assertEqual(v["under"], (0.0, 1))
        self.assertAlmostEqual(v["waves"], 1.0)
        # the piano's gene 0.2 heard as 0.79, the bass's 0.6 as 0.5, the voice's 0.8 as 1, each over the genes' RMS
        salience = torch.sigmoid(torch.tensor([9.0, -1.0]))
        piano = float((salience[0] * 1.0 + salience[1] * 0.0) / salience.sum())
        want = np.mean([abs(piano - 0.2), 0.1, 0.2]) / np.sqrt(ripmodel.VOICE_GENES)
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
        model = ripmodel.PitchNet(CFG)
        model.sure.fill_(0.75)
        with tempfile.TemporaryDirectory() as d:
            ripmodel.export(model, os.path.join(d, "notes.bin"))
            with open(os.path.join(d, "notes.bin"), "rb") as f:
                body = f.read()
        at = 4 + 16 + len(ripmodel._cells_header(model))

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
        p = ripmodel.PITCHES
        self.assertEqual(heads[4:8], [("cents", 2 * p), ("pitchfamily", ripmodel.NAMES * p), ("pitchwave", ripmodel.WAVES * p), ("pitchvoice", ripmodel.VOICE_GENES * p)])
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


def corpus(d, labels, frames=64):
    """a corpus on disk of loops of silence, a label each"""
    with open(os.path.join(d, "corpus.json"), "w") as f:
        json.dump({**MANIFEST, "loops": len(labels), "waves": ["sine", "saw", "square", "string", "modal"]}, f)
    for i, lab in enumerate(labels):
        with open(os.path.join(d, f"loop_{i:03d}.mel"), "wb") as f:
            f.write(struct.pack("<II", frames, ripmodel.MEL_BANDS) + bytes(4 * frames * ripmodel.MEL_BANDS))
        with open(os.path.join(d, f"loop_{i:03d}.json"), "w") as f:
            json.dump(lab, f)


class TheBeatAndTheBarLine(unittest.TestCase):
    def test_init_grows_the_heads_and_carries_every_other_weight(self):
        """A frame model from before the beat and downbeat heads starts a run through --init: every weight carries
        over, the output layer's rows of the heads it had among them, the two heads' rows are new, and the weight
        file lists them after the family."""
        old = ripmodel.Config(dim=8, heads=2, layers=1, ffn=16, window=4, outputs=ripmodel.HEADS[:-2])
        before = ripmodel.RipModel(old).state_dict()
        with tempfile.TemporaryDirectory() as d:
            at = lambda name: os.path.join(d, name)
            corpus(d, [{"bpm": 120, "bars": 1, "events": [note(0, 10, 60, "piano")]}])
            torch.save({"model": before, "cfg": asdict(old)}, at("old.pt"))
            argv = ["train.py", "--corpus", d, "--out", at("new.bin"), "--init", at("old.pt"), "--checkpoint", at("new.pt"), "--lr", "0", "--steps", "1",
                    "--batch", "1", "--window", "32", "--dim", "8", "--heads", "2", "--layers", "1", "--ffn", "16", "--attention-window", "4"]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()) as said:
                train.main()
            after = torch.load(at("new.pt"))
            with open(at("new.bin"), "rb") as f:
                body = f.read()
        self.assertIn("its heads but beat, downbeat, which start fresh", said.getvalue())
        self.assertEqual([tuple(h) for h in after["cfg"]["outputs"]], ripmodel.HEADS)
        had = old.out_size
        for k, v in before.items():
            with self.subTest(tensor=k):
                self.assertTrue(torch.equal(after["model"][k][:had] if k.startswith("out.") else after["model"][k], v))
        self.assertEqual([len(after["model"][k]) for k in ("out.weight", "out.bias")], [had + 2, had + 2])
        head = lambda name, size: struct.pack("<H", len(name)) + name.encode() + struct.pack("<I", size)
        self.assertIn(head("family", len(ripmodel.FAMILIES)) + head("beat", 1) + head("downbeat", 1) + struct.pack("<I", len(after["model"])), body)


if __name__ == "__main__":
    unittest.main()
