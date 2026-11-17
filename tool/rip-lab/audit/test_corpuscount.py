#!/usr/bin/env python3
"""test_corpuscount.py: corpuscount.py run on a corpus of recordings this test writes by hand, with no code of stapes:
two loops of frames and no WAV, their events at velocity -1, one stating its lag and one left out and named in
skip.txt. Run with `python3 -m unittest discover tool/rip-lab/audit`."""
import json, os, struct, subprocess, sys, tempfile, unittest

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'corpuscount.py')


class ACorpusOfRecordings(unittest.TestCase):
    def test_a_from_corpus_is_counted_from_its_labels_and_frames(self):
        """A velocity of -1 is a label that asks nothing and a loop with no WAV is counted from its label and its
        frames' header: the count exits 0 and says the hits, the lag and what is left out; a loop its label leaves
        out that skip.txt does not name may not be."""
        hit = lambda f, drum: {'frame': f, 'end': f, 'drum': drum, 'midi': 0, 'vel': -1, 'wave': -1, 'family': 7}
        note = {'frame': 4, 'end': 12, 'drum': -1, 'midi': 60, 'vel': -1, 'wave': -1, 'family': 1}
        head = {'bpm': 0, 'root': 0, 'scale': 0, 'bars': 0, 'tuning': 0}
        labels = [dict(head, frames=40, events=[hit(3, 0), hit(9, 1), hit(20, 2), hit(30, -1), note], source='a.wav', lag=0.0116, beats=[3, 20]),
                  dict(head, frames=30, events=[hit(5, 0)], source='b.wav', skip='hits that draw no onset')]
        with tempfile.TemporaryDirectory() as d:
            for i, l in enumerate(labels):
                with open(f'{d}/loop_{i:03d}.json', 'w') as f: json.dump(l, f)
                with open(f'{d}/loop_{i:03d}.bands', 'wb') as f: f.write(struct.pack('<II', l['frames'], 630) + bytes(4 * 630 * l['frames']))
            with open(d + '/skip.txt', 'w') as f: f.write('loop_001\n')
            run = subprocess.run([sys.executable, SCRIPT, d], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            for said in ('2 loops: 2 recorded; 1 notes', 'hits: 1 a strike of no line, 1 hihat, 2 kick, 1 snare; events of velocity -1, which ask nothing: 6',
                         'hits put on their sound in 1 loops, by a lag from +11.6 to +11.6 ms, median +11.6; left out: 1, 1 for hits that draw no onset; skip.txt names 1 loops',
                         "loops with no WAV, counted from their labels and their frames' headers: 2", 'nothing that may not be'):
                self.assertIn(said, run.stdout)
            os.remove(d + '/skip.txt')
            run = subprocess.run([sys.executable, SCRIPT, d], capture_output=True, text=True)
            self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
            self.assertIn('MAY NOT BE, a loop left out by its label that states a lag, or that skip.txt lacks: 1', run.stdout)


if __name__ == '__main__':
    unittest.main()
