#!/usr/bin/env python3
"""test_corpuscount.py: corpuscount.py run on corpora this test writes by hand, with no code of stapes: two loops of
recordings, of frames and no WAV, their events at velocity -1, one stating its lag and a hit it leaves out, held in
left.tsv, and one left out and named in skip.txt; and two arranged loops on kits and a recording by piece, their strikes
by piece and place. Run with
`python3 -m unittest discover tool/rip-lab/audit`."""
import json, os, struct, subprocess, sys, tempfile, unittest

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'corpuscount.py')


class ACorpusOfRecordings(unittest.TestCase):
    def test_a_from_corpus_is_counted_from_its_labels_and_frames(self):
        """A velocity of -1 is a label that asks nothing and a loop with no WAV is counted from its label and its
        frames' header: the count exits 0 and says the hits, the lag and what is left out; a loop its label leaves
        out that skip.txt does not name, and a hit left out that left.tsv does not hold, may not be."""
        hit = lambda f, drum: {'frame': f, 'end': f, 'drum': drum, 'midi': 0, 'vel': -1, 'wave': -1, 'family': 7}
        note = {'frame': 4, 'end': 12, 'drum': -1, 'midi': 60, 'vel': -1, 'wave': -1, 'family': 1}
        head = {'bpm': 0, 'root': 0, 'scale': 0, 'bars': 0, 'tuning': 0}
        labels = [dict(head, frames=40, events=[hit(3, 0), hit(9, 1), hit(20, 2), hit(30, -1), note], source='a.wav', lag=0.0116, beats=[3, 20],
                       left=[{'frame': 25, 'key': 44, 'drum': 2, 'vel': 45 / 127, 'why': 'no sound'}]),
                  dict(head, frames=30, events=[hit(5, 0)], source='b.wav', skip='hits that draw no onset')]
        with tempfile.TemporaryDirectory() as d:
            for i, l in enumerate(labels):
                with open(f'{d}/loop_{i:03d}.json', 'w') as f: json.dump(l, f)
                with open(f'{d}/loop_{i:03d}.bands', 'wb') as f: f.write(struct.pack('<II', l['frames'], 630) + bytes(4 * 630 * l['frames']))
            with open(d + '/skip.txt', 'w') as f: f.write('loop_001\n')
            with open(d + '/left.tsv', 'w') as f: f.write('loop\tsource\tsec\tframe\tkey\tdrum\tvel\twhy\trise\nloop_000\ta.wav\t0.27870\t25\t44\t2\t0.3543\tno sound\t5.78e-05\n')
            run = subprocess.run([sys.executable, SCRIPT, d], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            for said in ('2 loops: 2 recorded; 1 notes', 'hits: 1 a strike of no line, 1 hihat, 2 kick, 1 snare; events of velocity -1, which ask nothing: 6',
                         'hits put on their sound in 1 loops, by a lag from +11.6 to +11.6 ms, median +11.6; left out: 1, 1 for hits that draw no onset; skip.txt names 1 loops',
                         'hits left out of their labels: 1, 1 for no sound; by key: 44 1; left.tsv rows: 1',
                         "loops with no WAV, counted from their labels and their frames' headers: 2", 'nothing that may not be'):
                self.assertIn(said, run.stdout)
            os.remove(d + '/skip.txt')
            os.remove(d + '/left.tsv')
            run = subprocess.run([sys.executable, SCRIPT, d], capture_output=True, text=True)
            self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
            self.assertIn('MAY NOT BE, a loop left out by its label that states a lag, or that skip.txt lacks: 1', run.stdout)
            self.assertIn("MAY NOT BE, left.tsv's rows that are not the hits its labels leave out: 1", run.stdout)

    def test_b_kits_are_counted_by_piece_and_place(self):
        """A kit's strikes are counted by piece and by kit size, its genes by tenths of their range, and its strikes
        within a frame of another by whether they share its place: two pieces of places 0 and 1 struck on one frame
        share none, and a recorded kit's two pieces at one place on one frame do, its pieces counted by key and those at
        no place by why. A piece that strikes nothing, and a drawn piece's strike that names a drum, may not be."""
        head = {'bpm': 120, 'root': 0, 'scale': 0, 'bars': 1, 'tuning': 0, 'frames': 40, 'kind': 'arranged', 'beats': [0, 10, 20, 30], 'downbeats': [0]}
        strike = lambda f, n, drum=-1: {'frame': f, 'end': f, 'drum': drum, 'midi': 0, 'vel': 0.8, 'wave': -1, 'family': 7, 'part': 'drums', 'piece': n}
        piece = lambda g, b, p: {'genes': [g] * 13, 'band': b, 'place': p}
        labels = [dict(head, events=[strike(0, 0), strike(0, 1), strike(10, 1), strike(20, 1)], kit=[piece(0.05, 10, 0), piece(0.95, 150, 1)]),
                  dict(head, events=[strike(0, 0), strike(5, 1), strike(30, 2)], kit=[piece(0.5, 300, 2), piece(0.25, 301, 2), piece(0.75, 20, 0)]),
                  {'frames': 40, 'source': 'a.wav', 'lag': 0.0, 'kit': [{'key': 36, 'band': 10, 'place': 0, 'sure': 0.8}, {'key': 45, 'band': 30, 'place': 0, 'sure': 0.75},
                                                                         {'key': 49, 'band': -1, 'place': -1, 'sure': 0.5, 'why': 'a place unsure'}],
                   'events': [strike(4, 0, 0), strike(4, 1), strike(20, 2)]}]
        with tempfile.TemporaryDirectory() as d:
            with open(f'{d}/corpus.json', 'w') as f: json.dump({'loops': 3, 'struck_genes': [f'g{k}' for k in range(13)], 'places': 3, 'unplaced': {'a place unsure': 1}}, f)
            for i, l in enumerate(labels):
                with open(f'{d}/loop_{i:03d}.json', 'w') as f: json.dump(l, f)
            run = subprocess.run([sys.executable, SCRIPT, d, 'nowav'], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            for said in ('kits: 2 loops, by size: 2 1, 3 1; 5 pieces, 7 strikes; strikes a piece: fewest 1, median 1, most 3; pieces of one strike: 4',
                         'strikes a piece by kit size, the median and the fewest: 2: 2, 1, 3: 1, 1', 'g0 0 to 1 (0.05 to 0.95)',
                         "drawn kits' strikes: 7, 2 (28.6%) within a frame of another, 0 (0.0%) of them at its place too, of 3 places",
                         "recorded kits' strikes: 3, 2 (66.7%) within a frame of another, 2 (66.7%) of them at its place too",
                         'recorded kits by piece: 3 pieces, each key\'s by place (-1 none): 36 0:1; 45 0:1; 49 -1:1',
                         'recorded pieces at no place, by why: 1 for a place unsure', 'nothing that may not be'):
                self.assertIn(said, run.stdout)
            labels[1]['events'] = [dict(strike(0, 0), drum=2), strike(5, 1)]
            with open(f'{d}/loop_001.json', 'w') as f: json.dump(labels[1], f)
            run = subprocess.run([sys.executable, SCRIPT, d, 'nowav'], capture_output=True, text=True)
            self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
            self.assertIn('MAY NOT BE, a piece of a kit that strikes nothing: 1; first: loop 1 piece 2', run.stdout)
            self.assertIn("MAY NOT BE, a strike that names a key or an instrument, or a drawn piece's a drum: 1", run.stdout)


if __name__ == '__main__':
    unittest.main()
