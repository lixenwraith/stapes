# Checks that share no code with what they check

Seven scripts that read what the generator and the label tools write, from the files alone: the labels, the WAVs, a set's own annotations. None imports stapes, the trainer or a label tool, so a figure here that agrees with theirs was reached twice. They print figures; `corpuscount.py`, `guitarcount.py`, `vocalcheck.py` and `linescheck.py` exit 1 on what may not be. They need Python with numpy; the listening page also needs `ffmpeg`. `corpuscount.py` has a smoke test, on a corpus of recordings it writes by hand (`python3 -m unittest discover tool/rip-lab/audit`, which CI does not run); the others have none yet.

## A corpus

- **`corpuscount.py <corpus dir> [nowav]`** counts a corpus, rendered or of recordings (`corpus <dir> from`).
  - From the labels: the loops by kind and by meter, those that start part-way into a bar, the notes each of the 88 keys starts and any outside them, the registers' shares, the parts an arranged loop holds, and of the overlaps the instruments a loop holds, the most notes sounding at once, the velocities' spread, the notes struck again while they ring and the intervals between notes that sound together.
  - From each WAV (left out with `nowav`): its loudest half second and its peak against the level a loop is laid at (-32 to -14 dBFS, a peak of -1 dBFS or under), a silent loop, and its length against its label's frames.
  - Of recordings: their hours, the hits by drum, the events of velocity -1, which ask the velocity head nothing and are no fault, the loops whose hits were put on their sound with the least, the median and the most lag, those left out by reason, the names in `skip.txt`, and the hits the labels leave out as drawing no sound, by reason and by key, against the rows of `left.tsv`.
  - A loop with no WAV, as a recording's is, is counted from its label and its frames' header, which must hold the label's frames whole.
  - What may not be: a note that ends before it starts or starts past its loop, a note outside the 88 keys, a velocity outside (0, 1] that is not -1, a beat in an overlap, an arranged loop with none, a loop outside the level rule, a loop its label leaves out that also states a lag or that `skip.txt` does not name, a hit left out of a label that states no lag, and a `left.tsv` that does not hold the hits the labels leave out.
- **`corpusheard.py <corpus dir> [NSynth's audio dir]`** reads whether a corpus's notes sound as its labels say. Run it before a sample of the corpus is cut for the ear.
  - By the genes each label carries, for the notes on the engine's voices: a sine an octave under the note, a filter whose cutoff lies under the note's own fundamental, a modulation on a ratio that is no whole number, an attack longer than the note, a release that rings on after the label ends the note.
  - By the sound, for every note: whether the power at its own first four partials rises 6 dB where the label starts it.
  - For the notes that sound alone and enter cleanly: the share of their power on the label's partials, on the partials only the octave under has (a half, one and a half and so on), and elsewhere.
  - For the notes whose key is not struck again: what is left at their partials half a second after the label ends them. Other notes' partials can land there, so read it as a tendency.
  - With NSynth's folder: the recorded notes let go before their recording comes within 6 dB of its loudest 50 ms.
  - How much of an overlap holds no note.
- **`loop_sheet.py <corpus dir> <out dir> [picks.tsv] [overlaps]`** builds a listening page, `sheet.html`, with `sheetkit.py`.
  - A clip is one timeline with two sounds on it, the loop as rendered and its label's notes as plain tones; each goes on from where the other was, and the playhead runs on a picture of the label. Both are set to one loudness by their loudest half second.
  - In the picture a note is a bar in its part's colour, as faint as its velocity is low in dB; a thin tail is the voice's release after the label ends the note; upright lines are the label's beats and bar lines.
  - Under each clip: every part's instrument, how many of its notes show at their start, and how far under the loop's strongest note it sits.
  - A picks file names loops, a line each: the loop's number, a tab, the one question. A time in a question is the clip's own.

## Labels made from a set

- **`guitarcount.py <guitar-list.csv>`** recounts GuitarSet's list and MIDI files from the set's JAMS files (`$GUITARSET`): each row's paths, its split by player (00 to 03 train, 04 val, 05 test), its duration against the WAV's header, and the MIDI file's notes against the JAMS file's `note_midi` observations, string by string. A note's end may come early only where the same string's next note of the same key starts before it; those are counted apart.
- **`vocalcheck.py <vocal-list.csv> [sample]`** checks VocalSet's list and MIDI files against the set (`$VOCALSET`).
  - Every row: its paths, its split by the set's own list of test singers, its family, its duration, and whether an earlier row holds the same sound byte for byte.
  - Every recording: the share of its sung time a note covers, sung time being the frames within 20 dB of its loud level.
  - A note under 0.25 s between two of 0.4 s or more: with no table beside the list it is a slide written as a note and a fault; with the table a take's notes are its sheet's, so it is a sheet note squeezed, counted, and the shortest note of any take is printed against its take's median for the label tool's bound to be read against.
  - A seeded sample: a second pitch reading (a normalised autocorrelation over 93 ms under 1 kHz) against the labels' keys.
  - Where the label tool wrote its table of takes beside the list: every recording of the set in the table or counted by its folder; the list against the table; each listed take's keys against the shape the set's sheet writes for its exercise; each listed note read from the sound (on its key, an octave off, half a semitone to one and a half off, another pitch, no clear pitch); and each held note's octave by where its power lies, which a period's measure cannot decide.
  - The second reading locks onto the third, fifth or seventh harmonic of some loud notes and spans the neighbours of a note under 0.2 s: read "another pitch" and the fast takes with that in mind. The doubtful notes are written to a file for a listening page.
- **`linescheck.py <draw dir>...`** checks the known lines that sing VocalSet's exercises, before a placing is fitted or scored on them: each truth's keys against the sheet's shape and against what its list says is wrong with the take, and each note's pitch twice. The comb of ten partials about the pitch the truth gives is sure within the octave and blind to it; it reads a vibrato at the middle of its swing and only where the held part holds a whole cycle. The difference function says the octave where no partial outweighs the first.

## Not here

A capped launcher for one machine, the readers of a round's rips (they follow once a round has a reading), and the page of VocalSet's labels beside their recordings (it follows once it stands on `sheetkit.py`).
