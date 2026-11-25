#!/usr/bin/env python3
"""shapes.py: the notes VocalSet's sheet gives each exercise (SHAPE, the steps in semitones from one to the next), which
place.py places on a take and lines.py sings. ../README.md says where they come from."""
UP = [2, 2, 1, 2, 2, 2, 1, 2]
SHAPE = {'arpeggios': [4, 3, 5, -5, -3, -4], 'scales': UP + [-s for s in reversed(UP)], 'long_tones': [12, 5]}
