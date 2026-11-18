# rip-lab

What stands between a recorded set and the trainer: the tools that fetch a set
and look at it, write its labels and its `corpus ... from` list, and check both.
They are kept here so that every thread reads one copy. Python 3 with numpy
(the trainer's `.venv` has it), outside the Go module but for `labels/lines`,
the engine's line renderer. [Rip training](../../doc/rip-training.md) says what
a list and its labels are to `corpus`; [the sets'
issues](../../doc/set-issues.md) is what these tools found wrong in the sets'
own files.

- `datasets/`: a set fetched and looked at.
- `labels/`: a set's labels and list, and the known lines a label procedure is
  fitted and checked on.
- `audit/`: checks written apart from `labels/`, sharing no code with it, and
  the counts and the listening page of a rendered corpus
  ([its README](audit/README.md)).

## The order

1. A set lands under one folder (`datasets/fetch.sh` fetches NSynth and E-GMD)
   and is looked at before anything is read from it: `datasets/wavscan.py <set>
   datasets/logs/<set>-wavscan.tsv`.
2. GuitarSet: `labels/guitar.py <out dir>`, then `audit/guitarcount.py <out
   dir>/guitar-list.csv`.
3. VocalSet:
   - the known takes, three draws, each read by `audit/linescheck.py`: `go
     build -o <bin> ./tool/rip-lab/labels/lines`, then with `ENGINELINES=<bin>`,
     `labels/lines.py <truth root>/engine-shapes-fit engine 5 480 fit shapes`,
     `.../engine-shapes-check engine 6 240 fit shapes` and
     `.../engine-shapes-out engine 7 240 out shapes`;
   - `labels/placefit.py <truth root> write`, which writes `labels/place.json`;
   - `labels/label.py <out dir>`, then `labels/placed.py <out dir>` and
     `audit/vocalcheck.py <out dir>/vocal-list.csv`.
4. E-GMD: `datasets/egmd.py list <e-gmd-v1.0.0> train <hours>
   <e-gmd-v1.0.0>/egmd-train.csv`, and `... test all <e-gmd-v1.0.0>/egmd-test.csv`
   for the split held out, a corpus each.
5. A corpus, from a list or from the generator, is counted and read against its
   sound before it trains (`audit/corpuscount.py`, `audit/corpusheard.py`), and
   heard on `audit/loop_sheet.py`'s page.

A tool writes only where its arguments say. The sets' folders come from the
environment, each with the default of the machine the labels were made on:
`VOCALSET`, `GUITARSET`, and `NSYNTH` for the `audio` folder of NSynth's
training split. A rendered line's sound follows the engine the renderer is
built on, so a fit is run on the draws its own build rendered.

## What a label is refused for

A label that is doubtful is left out and counted by its reason, never mended by
hand: a first training wants few wrong labels more than it wants hours.

- GuitarSet: a note outside the piano's 88 keys.
- VocalSet: the excerpts and five techniques, which have no sheet to place; a
  recording that holds another's bytes; a take the placing's test refuses. Two
  of the test's bounds are ruled stricter than the fit took them, in
  `place.RULED`: a silence over 0.3 s inside a note, and a note under a third
  of the take's median note.
- E-GMD: nothing by the list; `corpus <dir> from` leaves out a recording whose
  hits it cannot put on their sound, by its reason, in the corpus's `skip.txt`
  ([Rip training §3](../../doc/rip-training.md#a-recordings-lag)).
- As ruled, of 2,978 takes tried 2,532 are listed, 26,142 notes in 5.73 hours:
  241 are refused for a note off its key, 111 for a squeezed note, 50 for a
  note under three frames and 44 for a silence inside a note.

A placed start is an aligned score's, not a label to the frame, and the list says so
(`starts` `placed`): on the check draws about 0.82 to 0.84 of starts lie within 50
ms of the truth and 0.97 within 100 ms. A semitone sung narrow and a wide vibrato
are where the starts are loosest.

## The tools

Each file's own header is five lines; what it does at length is here.

### labels/label.py

```
label.py <out dir>: VocalSet's singing as MIDI files, pitch tracks and a list for `corpus <dir> from <out
dir>/vocal-list.csv`, its test singers held out. A take's notes are its exercise's, as the set's sheet writes them
(an arpeggio's seven, a scale's seventeen, three long tones), placed on its sound by place.py: the key the singer
took, and where each note begins and ends. A take the placing refuses (it is not its exercise, or cannot be told to
be) is left out and counted by what it was refused for; so are the excerpts, which have no shape here yet, the
techniques in SKIP, and a recording that holds another's bytes (the set has a few twice; the copy of the longer name
goes). Each listed take's pitch, frame by frame, is written as a CSV of seconds and Hz (0 where nothing is voiced)
that the list names in `pitch`, so `corpus ... from` gives every note its cents and a slide or a vibrato is a note's
bend and no note. It is the pitch as the placing holds it: where place.py's `fold` is on, a frame the period's
measure read an octave from its note is written at the note's octave, which the sheet gives.
Writes only under <out dir>: the list's paths are relative to it, as `corpus ... from` reads them.
Every recording of the set is a voice, so the list gives each `voice` as its family: with none, a note whose MIDI
file sets no program is asked no name. The set states no velocity: every note is written at 80, which says nothing,
so the list's `velocity` cell is `measured` (stapes v0.37.0, doc/rip-training.md, "A recording's velocities"): a
corpus made from it gives each note the velocity heard in its take, -1 where it cannot be told, against a unit its
singer's takes share, the singer named in `unit`. A take's starts are placed, not heard struck, and its `starts` cell
says `placed`, so its notes ask the strike head nothing.
A recording's duration is its WAV header's. The placing's bounds are place.py's, printed first; vocal-placed.tsv has
a row a take: its exercise, technique, tonic, standard in cents, whether it is listed or why not, and the voiced
frames the fold moved an octave. vocadito is not here: it measures at most (stapes' AGENTS, Data), from its own
annotations.
```

### labels/place.py

```
place.py: the notes of a take of one of VocalSet's exercises, placed on its sound; imported by label.py and fitted
by placefit.py. The set's sheet says what an arpeggio, a scale and a long-tones take sing (shapes.py holds the steps),
so a take's notes are not read from its pitch: the sound gives the key the singer took, where each note begins and
ends, and whether the take is its exercise at all (Lixen's choice of 15 November 2026, after the labels a free tracker
cut were heard).
  The placing. Each frame's pitch (vocal.frames) and whether it is voiced; the pitch's centre, the median of the
  voiced frames within `smooth` seconds about each one in its own run of sound (half the take's mean note at most),
  which is a vibrato's middle and leaves a slide a slide; the take's own grid, the circular mean of its voiced
  frames' distance from A440's keys; a path over the voiced frames through the exercise's notes in order, each
  frame's centre paying its distance from its note's key, 3 semitones at most, for every tonic, the cheapest kept.
  Then the take's standard is read again as the median distance of its frames from the keys they were given and
  brought within half a semitone by moving the tonic, so a label's key is the take's own grid's, within 50 cents
  of A440's as a rendered loop's is. A note runs from its first to its last voiced frame within `reach` of its
  key, so a breath beside it is no part of it; nor, with `stub`, is sound at its edge of under `stub` seconds that
  a silence longer than itself parts from the rest (a frame or two of a breath read as a pitch, which would
  otherwise have the take refused for a silence inside a note; the silence is frames with no pitch, and sound off
  the note's key is none). A note that follows another starts where the centre arrives within
  `arrive` of where the note settles, or within `arrive_share` of the step into it where that is nearer, so a
  slide is the note before's; where it settles is its key, or with `settle` where its own frames do (`settled`:
  the median of the half of them that moves least), since a singer's note sits a little off its key and a start
  that waits for the key may wait too long. A note after silence starts with its first sound.
  With `split`, a step of `split` semitones or fewer is cut another way (the audit's idea, 15 November 2026): where
  the two notes' joint span parts into two levels with the least spread, the second level on the side the step
  says (`cut`), with a straight slide of up to `ramp` seconds between the levels, the second note starting where
  the slide ends (two bare levels part in the middle of a slide, which is early by half of it). A semitone sung
  narrow never arrives within a share of the step from its key, and the path, which
  measures from the keys, then gives one note of the pair twice its length and the other a sliver: on VocalSet's
  slow scales the longer note of a semitone pair is twice the shorter or more in 14.5% of pairs, of a tone pair
  in 5.5% (the audit's count on the labels of 07:48 UTC).
  With `fold`, a frame read one octave over or under its note's key is that note's frame all the same, paying
  `fold` of a semitone more than one read on it, and is moved onto its key in the pitch as placed: on a loud low
  note whose second harmonic stands 11 to 17 dB over its first a period's measure reads the octave over, a whole
  note long (289 such notes in the takes of VocalSet first refused for a note off its key, nearly all of them
  men's, measured 15 November 2026), and the sheet says which octave is sung. One octave and no more: a breath
  after a take's end is read two and three octaves up, and is no part of its last note.
  The test, what a take is refused for: a note under three frames; a note that settles more than `near` semitones
  from its key (`settled`, so that the slide a short note ends in does not count against it); a silence longer
  than `gap` seconds inside a note (a restart, a bar sung twice); more than
  `stray` of its voiced frames further than `far` from their note's key (sound that is not its notes); a note
  shorter than `even` of the take's median note, each taken against the length the sheet gives it (a note that was
  not sung, made of a slide's frames).
Every bound is in BOUNDS as it was first tried; place.json beside this file, when placefit.py has written it,
replaces them, and RULED, what Lixen has ruled on, is laid over that. A note sung most of a semitone off is still the note meant and has to pass; a take with a note
missing, a wrong note, a bar sung twice or a restart has not.
```

### labels/placefit.py

```
placefit.py <truth root> [write]: place.py's bounds fitted on known takes of VocalSet's exercises and checked on
takes the fit did not see. <truth root> holds lines.py's three `shapes` draws: engine-shapes-fit (the fit's),
engine-shapes-check (the same voices under another seed) and engine-shapes-out (a wave no fit take has). Six in ten
takes are right, some with one note sung most of a semitone off; the rest have a note missing, a wrong note, the bar
sung twice or a restart. Two fits, each from the bounds as first tried (place.TRIED), a bound at a time over its
grid, the best value taken only if its gain's 2.5th percentile over a thousand redraws of the takes is above
nothing (of values that score alike the middle one, which sits furthest from where the score falls off on either
side), passes until one takes nothing:
  1. the placing (the low-pass, the periodicity bound, the seconds the pitch's centre is taken over, what a frame
     read an octave from its note pays, how far from its key a note's own sound reaches, whether a note's start
     waits for its key or for where the note settles, how near the pitch has to have come, and the steps cut
     where two notes' own levels part instead), on the right
     takes: the mean of the sounding time given the right key (50 ms from a note's ends left out) and of the notes
     whose start is within 50 ms of the true one;
  2. the test (how far from its key a note may settle, the scrap of sound a silence may part from a note without
     being its own, then the silence inside a note, which is judged on notes with their scraps off; the stray
     frames, the shortest note against the others), on every take: the mean of the right takes that pass and the
     wrong takes refused.
A take tuned within 5 cents of half a semitone off is either key's, no standard being read from a take's sound more
nearly than that, so its notes may come out a key over or under as one.
Then both sets of bounds on the check seed and on the voices left out: the right takes that pass, those with a
note sung off among them, the wrong takes refused by kind, and on the takes that pass the keys and the starts a
label would hold; and of every two joined neighbours of those takes, by the step between them, how often one is
placed at twice the other's length or more, which the truth never is (a note of a slow take is a step in four
longer or shorter at most). With `write` the fitted bounds go to place.json beside place.py, which then reads them.
```

### labels/placed.py

```
placed.py <labels dir>: VocalSet's placed labels read back from what label.py wrote there (vocal-placed.tsv, the
list, each listed take's MIDI file and pitch track); no sound is read. Run after every run of label.py: it says what
the placing's own test cannot, since it holds every take at once.
  1. By exercise and technique: the takes tried, listed, and refused by what for.
  2. The tonic against the take's name. The set names the key of its fast and slow takes (`_c_` or `_f_`), and a
     singer may be a little off it, so: the takes whose tonic is the named key, a semitone from it, and elsewhere,
     listed and refused apart. `elsewhere` in a listed take is a placing on the wrong notes or a singer who moved the
     bar; either way it is named below the table.
  3. Every note of a listed take against the take's own pitch track: the share that settles within a quarter and
     within half a semitone of its key (the take's standard taken off; where a note settles is the median of the
     half of its frames, one after another, that moves least, as place.py's test reads it), the mean share of a
     note's voiced frames within half a semitone of its key, and the share of notes under half and under a quarter
     of the take's median note (a note the singer left out is placed in a slide, and is short).
  4. By the note's place in the bar, fast and slow takes apart: the median of the notes' median pitch less their
     key, in cents. A note holds the slide out of it, the next starting where the pitch arrives (the engine's
     notes do the same, and a label made here has to mean what a rendered loop's means), so a note's median leans
     to the next note: sharp on the way up, flat on the way down, most on a fast take, where a note is mostly
     slide. A lean well past a third of a semitone is starts placed late.
  5. By singer: the takes listed of those tried.
  6. The fold: the listed takes that hold frames moved an octave onto their note, women's and men's, and those in
     which a tenth of the voiced frames or more were (a whole note or more read an octave off).
  7. Of every two neighbours of a listed take with no silence between, by the step between them: how often one is
     twice the other's length or more (a scale's last note taken at half its length, the sheet writing it a
     quarter). A boundary lost between two notes a semitone apart shows here before it squeezes a note to a sliver.
```

### labels/shapes.py

```
shapes.py <vocal-list.csv>...: the shapes of VocalSet's exercises (SHAPE, which place.py places on a take's sound
since Lixen's choice of 15 November 2026), and a list's labels counted against them: a count that says something of
labels a free tracker cut, and nothing of placed ones, which hold their shape by construction.
The set's own sheet (DataSetVocalises.pdf, in its archive) gives what every take sings: an arpeggio
is a bar of seven notes (the tonic, third, fifth, octave and back down), a scale a bar of seventeen (a major scale up a
ninth and back), each on C or on F as the take's name says, and a long-tones take three whole notes (C4, C5, F5). A
singer may move the key, so what is checked is the count and the steps between the notes. By exercise and technique:
the takes, the share whose label holds exactly the shape's count, the share whose every step is the shape's too
(`steps`), the median count, the share with more notes than the shape (little notes written in) and with fewer (notes
lost). A take over 1.6 times its group's median length may hold a bar twice or both bars: those are counted apart and
judged by no shape. Excerpts have no one count: their median only. Reads the lists and their MIDI files alone.
```

### labels/vocal.py

```
Notes of a solo voice from its pitch, YIN then a vibrato-tolerant cut, each placed where its sound is: imported by
label.py, and scored by trackscore.py on lines whose notes are known. Every bound of it is in BOUNDS, each as it was
set by hand until trackfit.py fits it; bounds.json beside this file, when trackfit.py has written it, replaces them
(VOCAL_BOUNDS=hand in the environment keeps the hand-set ones, and a JSON object there sets some over them, to compare).
  lp        the corner in Hz of a low-pass before the periodicity measure, 0 for none; the level is read unfiltered
  dip       how low the difference function has to dip for a lag to be taken as the period
  ap_max    the periodicity bound: a frame is voiced under it
  top       the level gate: a frame is voiced within this many dB of the recording's loudest
  move      what a move of the path between semitones costs, against a frame's distance from its semitone
  gap       the frames of no voice a note may hold inside itself
  minlen    the shortest note, in frames
  join      the frames two notes of one key may stand apart and be one
  slip_len, slip_far   a note shorter than slip_len seconds and more than slip_far semitones from both its
            neighbours is the tracker's slip, an octave or so off, and is left out
  blip_len, blip_near  a note shorter than blip_len seconds that touches a note of one key on each side, within
            blip_near semitones of it, is a vibrato's swing written as a note: it goes and the two are one. 0: none
  slide_len, slide_share   a note shorter than slide_len seconds on a key between its two neighbours' that the pitch
            crosses by slide_share of their interval or more, or one before a note with none before it that rises
            or falls into that note by as much, is a slide or a scoop written as a note: the note before it holds
            through a slide, the note after starts with its scoop. 0: none
A little note between two long ones is likely an error and no note of its own (Lixen's ear, 15 November 2026): the
two blip rules are that, each with its bound fitted where the notes are known.
```

### labels/lines.py

```
lines.py <out dir> <engine|nsynth> <seed> <count> <fit|out> [tuned|shapes] | lines.py <out dir> verify: lines of one voice whose notes are known, laid so that
the pitch tracker (vocal.py) can be scored where the answer is known (trackscore.py, trackfit.py). A line is about
twelve seconds on a sixteenth-step grid at a drawn tempo: notes from 0.1 to 2.5 s, three in ten followed by a rest, the
next key a drawn interval from the last (repeats, steps, thirds, fourths, fifths, octaves), each note's velocity drawn
over 26 dB, so that a soft note beside a loud one is in every line.
  engine: a voice of the engine drawn by the seed (a saw or a square through a vowel's formants or none, with or
    without vibrato, its envelope and a low-pass drawn), rendered by labels/lines (ENGINELINES names its binary), with
    a slide into the next note at the end of seven in ten of the notes another follows (0.05 to 0.25 s long) and a
    scoop from below into half of the notes that follow silence, since a slide between two held notes is no note
    (Lixen's ear on VocalSet's labels, 15 November 2026). Known: every note's start, end and key, its bend, the
    voice's vibrato. Half the lines get a ring, the band from 1.5 to 4 kHz raised by 10 to 35 dB: the takes of
    VocalSet the tracker loses hold 0.75 to 0.83 of their energy there under a vibrato of 80 to 146 cents either way
    on low notes, and a line needs both for a period's measure to break on it (measured 15 November 2026; the
    engine's vibrato stops at 100 cents). `fit` voices are saws, `out` voices squares, a wave no fit hears. With
    `tuned` each line is rendered at a tuning of its own, up to 50 cents either way of A440 and one in eight at
    exactly 50, as a singer with no accompaniment keeps a pitch standard of their own; a tuned line's keys are its
    written ones, as a rendered loop's label keeps them (rip training, `tuning`). With `shapes` each line is a take
    of one of VocalSet's exercises as its sheet writes them (an arpeggio's seven notes, a scale's seventeen, three
    long tones; fast takes 0.12 to 0.24 s a note and half of them detached, slow ones 0.35 to 1.2 s, long tones 2 to
    5 s with breaths between), at a tuning of its own, for scoring a placing of the sheet's notes on a take's pitch.
    A voice sings one note at a time, so these voices let go in 0.01 to 0.04 s and slide into short notes too.
    Six in ten are right, and of those three in ten have one note sung 0.5 to 0.9 of a semitone off its key, which
    is still the note meant (Lixen's ear); one in ten each has a note missing, the bar sung twice, a restart after
    two to four notes, or a wrong note two to four semitones from the one written. A wrong take's truth is what is
    sung; lines.tsv says its exercise and what is wrong.
    A take is sung as a singer sings, which the engine's notes are not: each note sits its own distance off its
    key, a draw about nothing whose spread is the take's own, from 0.07 to 0.40 of a semitone (what the takes of
    VocalSet hold between their twentieth and their last twentieth, measured 15 November 2026 on the settled
    middle of the notes of its slow takes), and a slide ends on the pitch the next note is sung at; three in
    twenty takes carry a vibrato of 100 to 170 cents either way at 4.3 to 7.7 Hz in place of the engine's, which
    stops at 100 (a tenth of VocalSet's takes are wider than that, the widest 170 to 180); the notes of a slow take are
    not all one length (a boundary a step early or late); and a quarter of the takes have a chest's resonance, a
    peak of 14 to 28 dB a tenth to a quarter of an octave wide somewhere from 330 to 1050 Hz, so that a note whose
    second harmonic falls on it holds that harmonic far over its first, as the loud low notes of VocalSet do that a
    period's measure reads an octave over (their second harmonic 11 to 17 dB over the first and 13 to 23 over the
    third, twice their pitch from 330 to 1050 Hz). The sound is the engine's line played a little faster
    or slower sample by sample (`warp`), so one render holds all of it; a truth row gives each note's start and end
    in the sound as written and `off`, the semitones it is sung off its key. A take's standard is the tuning
    drawn: its notes' offsets are centred on nothing by their lengths.
  nsynth: notes of one NSynth vocal_acoustic instrument laid end to end, each its recording's first seconds at the
    level its velocity asks. Known: each note's key, by the set's label, and where it is laid. A note whose sound is
    not within 0.6 of a semitone of its label at seven points from 0.1 s to 2 s is left out and listed
    (notes-left-out.tsv, with its reading furthest from the label): a voice that scoops into its note, or a choir
    doubled at the octave, has no one key to be right about. Two thirds of the instruments on disk are `fit`, the rest
    `out`, by a seeded shuffle of their names.
Both get what a recording has beside its notes: a breath of noise in half the longer rests, a burst of high noise before
three in ten of the notes that start from silence, and a peak drawn from VocalSet's own peaks
(datasets/logs/vocalset-wavscan.tsv), written as 16 bits. A line is <name>.wav (44.1 kHz mono) and <name>.tsv, its
truth: a row a note (start, end, key, velocity, scoop, bend, where the bend starts) and a row a noise. lines.tsv lists
the lines. Neither kind is a singer's phrase: the engine has no consonants, breaths or words but the noise laid here,
and NSynth's notes are single held notes joined, with no glide between them and nothing above 8 kHz.
```

### labels/guitar.py

```
guitar.py <out dir>: GuitarSet's notes as MIDI files and a list for `corpus <dir> from <out dir>/guitar-list.csv`, read
from the set's own JAMS files alone (GUITARSET's annotation) beside its microphone recordings (audio).
A note is one `note_midi` observation of one of the six strings. It starts at the observation's time and ends at its time
plus its duration, in the recording's own seconds; where the same string's next note of the same key starts before that,
it ends there, since a MIDI note-off names no note. Its key is its pitch rounded to the nearest semitone, a half up.
Notes kept are the piano's 88 keys (MIDI 21 to 108); one outside is left out and counted. The set states no velocity:
every note is written at 80, which says nothing, so the list's `velocity` cell is `measured` (stapes v0.37.0,
doc/rip-training.md, "A recording's velocities"): a corpus made from it gives each note the velocity heard in its
recording, against the recording's own unit, -1 where the strings sounding with it hide it.
A MIDI file is one track at 120 bpm with 22,050 ticks a quarter, a tick a sample at 44,100 Hz, a string a channel, so
two strings on one pitch stay two notes. The list gives each recording its JAMS file as `pitch` and as `beats` (stapes
reads each string's pitch_contour and the beat_position from it), `guitar` as its family, its split by its player,
the first two characters of its name: 00 to 03 train, 04 val, 05 test, and as `strings` its debleeded hexaphonic file,
`audio/hex/<name>_hex_cln.wav` from the set's audio_hex-pickup_debleeded.zip, where it is unpacked and a pitch track is
named: stapes ends each note where its string stops sounding it (doc/rip-training.md, "A recording's strings").
Writes only under <out dir>: the list's paths are relative to it, as `corpus ... from` reads them.
```

### labels/trackscore.py

```
trackscore.py <truth dir>...: the pitch tracker (vocal.py, under its bounds as they stand) scored on lines whose
notes are known (lines.py lays them). trackfit.py imports it to fit the bounds. What is counted, a line at a time:
  by time, on a grid of hundredths of a second, leaving out 50 ms either side of a true start or end, a scoop's
    step, a glide's length (where the pitch is on its way and no key is the right one) and a voice's release after
    a note that a rest follows (where the sound is neither a note nor silence): of the time a note
    sounds, the share the tracker covers, the share on the right key, an octave off (any number of octaves) and
    elsewhere; of the time none sounds, the share it writes a note into, and that share inside the noises laid.
  by note: the key that holds the most of each true note's time: the right one, an octave off, elsewhere or none.
  by onset: a true note is found when a tracked note of its key starts within 50 ms of it (anywhere in a scoop);
    precision is the found over the tracked, recall the found over the true, F their harmonic mean; and how far a
    tracked start of the right key within 150 ms sits from the true one.
  two faults of their own: the share of the sounding time written a semitone over or under the true key (up1, down1:
    a whole line read off its own pitch standard shows here), and of the glides between two held notes, the share
    with a note tracked to start inside on neither of the two keys (slid: a slide is no note).
The score of a kind of line is the mean of two shares: the time said right (the right key where a note sounds,
nothing where none does) and the onsets' F. Over kinds it is their mean, a kind weighing as much as another.
```

### labels/trackfit.py

```
trackfit.py <truth root> [write]: the pitch tracker's bounds (vocal.py) fitted on lines whose notes are known, and
checked on lines the fit did not see. <truth root> holds lines.py's six draws: engine-fit and nsynth-fit (the fit's),
engine-check and nsynth-check (the same voices under another seed), engine-out and nsynth-out (voices left out: a
wave no fit line has, instruments no fit line has). The procedure:
  1. It starts from the bounds as they were set by hand (vocal.HAND) and scores them on the fit draw (trackscore's
     score: the mean of the time said right and the onsets' F, a kind of line weighing as much as the other).
  2. One bound at a time, in GRID's order, every value of its grid is scored with the others held. The best value
     is taken only if it is better than the one held on lines drawn again: over a thousand draws of the fit's lines
     with replacement, within each kind, the gain's 2.5th percentile is above nothing. A bound the score cannot tell
     apart stays as it is.
  3. Passes over the bounds repeat until one takes nothing (three at most).
  4. The bounds by hand and the bounds fitted are scored on the check draw and on the voices left out, and each bound
     the fit moved is put back alone on each, to show what it gives where the fit did not look.
A grid is the hand-set value and a few either side, about a third apart, longer on a side the fit ran to the end of;
a low-pass corner goes from none down to 600 Hz, and the two blip rules' lengths from none up to 0.45 s. With `write` the fitted bounds go to bounds.json beside vocal.py, which then reads them.
```

### datasets/egmd.py

```
egmd.py list <root> <split> <hours|all> <out.csv> [seed]: a list for `corpus <dir> from <out.csv>` of the recordings
of one of E-GMD's splits (train, validation, test) that <root> holds, the folder of e-gmd-v1.0.0.csv: a row of the set's
metadata whose WAV and MIDI file are both there. A split is named, so a recording of the test split is in no list
drawn for training; a performance the metadata puts in two splits stops it. Each row gives the recording, its MIDI
file as its hits' and again as its `beats`, its split and duration, `none` as its `velocity` (a kit's law is not the
engine's, so a hit is labelled at -1), and the set's own columns (drummer, session, id, style, bpm, beat_type,
time_signature, kit_name), which `corpus` does not read and a count can. Paths are relative to <out.csv>'s folder:
written into <root>, a label's `source` starts with its drummer and session.
  The draw, the same for a seed (1), measured in hours of audio since a recording runs from under a second to ten
  minutes. A performance is one take of a drummer, which the set holds on up to 43 kits, a recording each (the
  metadata's `id`). The performances are laid in an order that goes round their strata, a drummer, a style (its
  name before the slash) and a quarter of the performances' tempos, one performance of each stratum in turn, the
  strata and the performances of each shuffled; a pass over that order takes from each performance one recording
  more, the one on the kit the draw holds the fewest seconds of. So no performance is on two kits before every
  other is on one, no kit runs ahead of the others, and a draw that ends inside a pass is spread over drummers,
  styles and tempos as the performances are. A recording that would pass the hours asked is not taken, and the
  pass it is met in is the last: the draw ends within one recording of the hours, under them. `all` takes every
  recording. A drummer, a style or a tempo holds what its performances do: three drummers play three quarters of
  the set's hours.
  Its last lines are the spread: the recordings and hours drawn of those held, then the hours drawn of those held by
  drummer, by style and by quarter of tempo, the kits drawn with the least and the most hours a kit holds, and the
  performances drawn by how many kits each is on.
egmd.py extract and status are fetch.sh's: one zip member inflated and checked, and the files on disk a batch.
```

### datasets/vocalsing.py

```
vocalsing.py frames <labels dir> <cache.pkl> | vocalsing.py <standard|spread|octave|vibrato> <labels dir> [<cache.pkl>]:
how VocalSet's arpeggios, scales and long tones are sung, measured for doc/set-issues.md. It reads the takes
labels/label.py tried (<labels dir>/vocal-placed.tsv), the labels it wrote and the set's own recordings; it writes
nothing but the cache.
  frames    every tried take's frames (labels/vocal.py's pitch, periodicity and level under the placing's low-pass)
            kept in <cache.pkl>, so that `octave` and `vibrato` take seconds; about three minutes of a core.
  standard  each singer's pitch standard. The first note of every listed take, on the middle half of the note as the
            label places it, read by the pitch whose harmonics hold the most of the spectrum (labels/lines.py,
            sound_pitch: no period's measure, so no part of the placing), against the key the take's name gives (C
            where it gives none), folded to the nearest octave: by singer, the median, the middle half, and the takes
            more than a semitone and a half over the named key.
  spread    how far a singer's notes sit from their keys. On every listed take that is not a fast one, each note of
            twelve voiced frames or more: the median of the middle half of its frames, by the take's own pitch track,
            less its key and the take's standard. Over all notes, the share within 10, 20, 30 and 50 cents; and a
            take's own spread (the standard deviation of its notes), from the lowest twentieth to the highest.
  octave    the notes a period's measure reads an octave from the key their exercise gives them: every tried take
            placed without the placing's fold; a take counts when it is refused for a note off its key and its worst
            note's median sits 9 semitones or more out. Men's and women's takes, over and under; and on those notes
            (0.12 s or more) the level of the second harmonic against the first, the third and the fourth, beside
            the notes of the same takes read within half a semitone. A harmonic's level is the largest bin within 4%
            of its place in the spectrum of the note's middle half; the note's pitch is its key on the take's standard.
            And of those notes: the ones inside which the measure holds both octaves, a tenth of the frames or more
            each (a voice does not leap an octave ten times a second: the reader does), and what a reading by
            harmonics (lines.sound_pitch) says of them.
  vibrato   the vibrato of the held notes: on every note of 0.4 s or more of a listed take, the middle three fifths
            of its voiced frames less their median: the extent (half the spread between the tenth and the ninetieth
            percentile, over 0.951, a sinusoid's) and the rate (zero crossings). A take's extent is its notes' median;
            by technique and over all takes.
```

### datasets/wavscan.py

```
wavscan.py <folder> <out.tsv>: every WAV under a folder as its own header and samples say it. A line a file goes to
<out.tsv>: its path under the folder, rate, channels, bytes a sample, int or float, seconds, peak and RMS in dB of
full scale, its mean (a DC offset), how many samples sit at full scale and the longest run of them (a run of three or
more is a clipped peak). It prints the totals: the formats, the peaks' spread, the quiet files, the clipped ones and
the files that do not read. A set's level and format are looked at with it before anything is read from the set, and
what it finds wrong in a set goes into doc/set-issues.md. Reads only.
```
