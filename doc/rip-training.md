# Rip training: the data, byte by byte

How a recording or a rendered loop becomes the numbers the rip's models train on,
and how the trained models read back into the rip. The audio library holds the
frames, the input planes and the heads; [transformer's
format](https://github.com/lixenwraith/transformer/blob/main/doc/format.md) is the
weight file's contract; stapes holds the corpus verb and the trainer
([rip-train](../tool/rip-train/README.md)), and [Rip design](rip-design.md) keeps
the reasoning.

Where each part stands:

- **In released code:** `BandFrames` and its layout, the four input planes, the heads,
  weight file versions 3 and 4, `corpus`, `corpus <dir> from`, `corpus ... sampled`, `--mix`,
  `--channel`, `--over` with its heard-hit rule, draws by length, `skip.txt`,
  `make_noise.py`, `--pitchnet` and its `ripnotes.bin`, `--shuffle`, `--extra`,
  `--cell`, `--dip`, `--ring`, `--bend`, `--dilations`, `--sung`, each event's family (`families`,
  the list's `family` cell, each note's program, every part of a file), the family head and
  `--family-holdout`, `--validate`, `tune ... truth`, a moving note's `cents`, `corpus ... glide`,
  the notes model's cents head, `corpus ... labels` and a `from` list's `pitch` column,
  `rows`, `judge.py` and the root's `ripjudge.bin`, the frame model's beat and downbeat
  heads, a `from` list's `beats` and `velocity` columns, `corpus ... gen`'s arranged
  loops, `corpus ... meters`, `corpus ... overlaps`, a loop's level and air, and a label's
  `kind`, `chords`, an event's `part` and `instr`, and a rendered loop's `beats` and
  `downbeats`.

## 1. The path

1. A corpus is rendered or read: per loop, a `.bands` file and a `.json` file (§2, §3).
2. Each training step draws 8 windows of 512 frames, 5.9 s each (§7).
3. A window's frames may be changed in place, as §6 lists, then become the input
   planes (§4).
4. A window's events and its loop's beats become per-frame targets for the 12 heads,
   and a notes model's cents head (§5).
5. The frame model turns frames into 405 numbers a frame, and the losses train it (§7, §8).
6. The best step is written as a weight file (§9).
7. At a rip, Go computes the same planes, runs the file and hands the heads to the
   decoders (§10).
8. Where a judge is fitted, `rows` writes every strike's row beside its label,
   `judge.py` fits a network a stage on them, and the rip's accounting reads its
   file (§13).

## 2. The frames: `loop_NNN.bands`

The file is the header, then every value of every frame as a little-endian float32:

```
offset  size              field
0       u32               frames
4       u32               values, always 630: the fine plane's 315 bands, then the sharp plane's
8       4 x frames x 630  rows, frame 0's first, nothing padded
```

A frame costs 2,520 bytes. There are 86.13 frames a second, so an hour is 781 MB.

Each value is `log1p(100 e)`, made by audio's `BandFrames`, the layout `RipLayout`
(1) names:

1. The recording is resampled to 22,050 Hz and scaled so its largest sample is 1.
2. Band k is at MIDI 20 + k/3, 315 bands three a semitone from G#0, under A0, to
   10.96 kHz (`BandHz`).
3. Level 0 is the recording; level j, at 22,050 / 2^j Hz, is level j-1 through
   `Resample`'s low-pass, every other sample, down to level 5 at 689 Hz.
4. Frame t's window at level j is 2,048 samples of Hann centred on sample t x 256 /
   2^j, zero past either end of the recording, through an FFT of 4,096 points.
5. Band k's weights are a triangle over that level's bins from band k-1 to band k+1,
   a bin either side at least, scaled so a unit cosine at band k reads 1, and its
   energy e is the sum of weight x magnitude squared.
6. The fine plane hears band k at the shortest window whose bins fall three a
   semitone there: level 0 from band 158 (544 Hz) up, a level down for each 36 bands
   under it, and level 5, a window of 2.97 s, under band 14 (34 Hz). The sharp plane
   hears every band at level 0, the analysis's 93 ms window, so a strike reads where
   it comes.

Going back: `e = expm1(v) / 100`. The file holds energy only: with the phase gone, it
can't be played back exactly. Every key from A0 to C8 peaks at its own band, two keys a
semitone apart read two peaks, and under 270 Hz the sharp plane's bands share bins,
about twenty a bin at A0, which is what the fine plane is for.

## 3. The labels: `loop_NNN.json` and `corpus.json`

A label is UTF-8 JSON with no whitespace. The loop's fields:

- `bpm`, `bars`, and `meter` when it is not 4/4 (`"3/4"`, `"7/8"`): the tempo it was
  drawn at, its length in bars and its time signature. A drifting loop's beats keep
  tempos of their own about `bpm`, which `beats` places. A recording's and an
  overlap's are 0 and absent.
- `root`, `scale`: how it was drawn, an overlap's 0. The trainer doesn't read them.
- `kind`: a rendered loop's, `bank` for the bank's patterns, `arranged` for one the
  arranger wrote (below) and `overlap` for single notes in no meter (below); absent
  from a recording's.
- `frames`: the length of its `.bands`.
- `tuning`: cents off A440 it was rendered at, within 50 either way. Events keep the
  written pitch, so at nearly +50 cents a note labelled 55 sounds nearly as much like
  56. It is where every note of an engine voice sits from its pitch's cell
  centre, unless the note carries `cents`.
- `source`: in a corpus of recordings, the file it was heard from.
- `placed`: `true` in a corpus of recordings whose list says the notes' starts are
  placed on the recording, not heard struck (a recording's starts, below), and
  absent otherwise.
- `beats`, `downbeats`: its beats and bar lines, in frames. A rendered loop's are where
  the render puts them, each beat on the sample audio's `BeatStarts` gives it under
  its bars' tempos and rounded as an event's frame is, from the loop's first frame to
  the last beat before its end, the end's own mark left out: a loop that starts
  part-way lists its first beat where it falls and its first bar line at the bar's.
  A recording's are those its list names the file of (a recording's beats, below). An
  overlap and a recording of no such file list none.
- `chords`: an arranged loop's harmony, each chord from the frame its bar starts on
  (0 for the first, a pickup's), as `{"frame":f,"tones":[t,...]}`, its pitch classes
  from C with its root's first; a chord the bar before already holds is not listed
  again.
- `events`: in no time order; the trainer sorts them.

An event's fields:

- `frame`, `end`: its start and its release, in frames. A drum's are equal.
- `drum`: 0 kick, 1 snare, 2 hihat, -1 anything else.
- `midi`: the pitch, 0 for a drum.
- `vel`: how hard it was struck, 0 to 1, or -1 where nothing says, which asks the
  velocity head nothing. The engine's law makes a velocity an amplitude: audio's
  voice samples `raw * env * velocity * level` and a sampled voice `buffer *
  velocity`, so a tonal note's `vel` is its amplitude over a note's of velocity
  one on its instrument.
  - A rendered note's is the velocity the engine played it at.
  - A sampled note's is the velocity its recording is laid at: the recording's own
    level, the RMS of its loudest held window of 50 ms (`sampledWindow`), taken out
    and the velocity put in (`(*sampleSet).heard` and `play` in `sampler.go`).
  - A recorded note's is its MIDI file's, -1 where its list says those say
    nothing, or under `measured` its level heard in the recording over the
    recording's unit, -1 where it cannot be told from the notes sounding with it (a
    recording's velocities, below).
- `wave`: an index into the manifest's `waves`, -1 when unknown or for a drum.
- `family`: an index into the manifest's `families`, what plays it, -1 when unlabelled.
  An engine voice's is its name's (`voiceFamilies` in `tune.go`: the bank's piano and
  bass, the pluck a guitar, the pad and the lead synth, the focus voices) while it stays
  on its wave within 0.2 RMS of the registered voice's genes, else the nearest
  registered voice's within that reach, else unknown, as is every voice `aug` drives.
  NSynth's notes take their instrument's family (keyboard piano, brass, flute and reed
  winds, mallet and organ unknown), every hit and strike the drums', and a `from`
  recording's notes its list's `family` column. Where that cell is empty or absent,
  each note takes its MIDI program's (`programFamilies` in `tune.go`: 0-15 piano,
  16-23 and 80-103 synth, 24-31 and 104-108 guitar, 32-39 bass, 40-46, 48-51 and 110
  strings, 52-54 voice, 56-79, 109 and 111 winds, 47, 55 and 112-127 unknown), and a
  note whose file set no program none. A `from` loop holds every part's notes, the
  parts past the score's melody, under and bass too (`RipNote.Further`), so all five
  parts of a quintet sound in its labels.
- `voice`: the 32 genes of the voice that played it, absent when unknown.
- `part`: a rendered loop's, the line it plays in: an arranged loop's `drums`, `bass`,
  `left`, `chords`, `strum`, `pad`, `arpeggio`, `melody`, `double` or `counter`, a bank
  loop's slot (`rhythm`, `melody`, `fill`, `bass`), an overlap's `single`.
- `instr`: a rendered loop's, what it sounds on: the engine's instrument (`piano`,
  `kick`), or the recorded instrument's name in NSynth (`guitar_acoustic_002`).
- `cents`: on a note that moves, where it sits at each frame from `frame` to `end`, one
  frame at least, in cents from `midi`'s cell centre to a tenth, the tuning in it: its
  step's bend and its voice's vibrato as the engine played them (audio's
  `VoiceDef.NoteCents`). A note that holds its pitch carries none, so a corpus of
  the bank's voices and patterns, which neither bend nor sway, is byte for byte what it
  was before the field. A recorded note lists them where its list names the
  recording's pitch track, `null` at a frame the track does not place (a recording's
  pitch track, below); where any other recorded note sits is not known.

Six kinds of event are in use: a drum hit, `drum` 0 to 2 at `midi` 0; a note on a
voice, with its `wave` and `voice`; a recorded note, such as one of NSynth's, whose
`wave` is -1 and whose voice is unknown; a strike of no line, `drum` -1 at `midi` 0,
something struck that no drum head or pitch owns, such as a tom or a crash from E-GMD;
a note that moves, with its `cents`; and a tracked note, a recorded note whose `cents`
its pitch track writes (below). `glide` draws from two streams of its own, so a loop's patterns, key, voices and tuning are those
the same line draws without it:

- A fifth of the notes of an arranged loop's melody, double, counter-line and bass on
  the engine's voices bend into the note that follows them at once, when that is a
  fifth away or nearer, over their last 60 to 250 ms.
- A seventh scoop: written a semitone or two under and rising to their pitch over
  their length, which is all a step's bend holds of a scoop.
- Three in ten of a loop's oscillator voices take a vibrato of 4.5 to 7 Hz, 10 to 80
  cents either way, coming in over up to 0.4 s; a voice that has its own keeps it.

The same seeds render the same frames, byte for byte.

`corpus.json` names the layout for the whole folder:

```
{"version":1,"loops":40000,"rate":22050,"hop":256,"layout":1,"drums":["kick","snare","hihat"],
 "waves":["sine","saw","square","string","modal"],
 "families":["piano","guitar","voice","bass","strings","winds","synth","drums","unknown"],"voice_genes":32}
```

`loops` counts the files `loop_000` onward; the number grows past three digits as
needed (`loop_39999`). `layout` is the frames' (§2): the trainer refuses a corpus of
another, or of none, by name, and one is exported again for a new layout. A label holds
each note's own MIDI number, and a model's cells are the piano's 88 keys (§5). The kinds
of corpus:

- **Rendered:** `corpus <dir> 40000 4 aug gen meters overlaps nowav codec`, loops of 4
  bars: three in four arranged over the whole keyboard in seven meters, one in four
  single notes overlapping in none (below). [Rip design §3](rip-design.md#3-learning)
  lists what `aug` draws. `codec` sends a third through Opus, MP3 or AAC at 64 to 192
  kbps and back.
- **Sampled:** the same with `sampled <dir>`, so the same seeds draw the same notes.
  Each tonal part plays on NSynth's recorded notes with a chance of a half, an
  overlap's instruments are drawn among the recorded ones and the engine's alike, and
  a part on a recorded instrument keeps to the pitches within 3 semitones of a note
  it has, each shifted at most that far and cut at its note-off by a 60 ms fade, and
  to an instrument whose recordings speak, within 6 dB of their loudest 50 ms, before
  each note is let go.
  NSynth is recorded at 16 kHz, so a recorded note holds nothing over 8 kHz, where 17
  of the 315 bands' centres lie, the top one at 10,962 Hz. Where the engine plays
  beside it, its voices, drums and room fill those bands; a loop of recorded notes
  alone would be empty there, a tell no recording has, so it is laid with air (a
  loop's level, below).
- **E-GMD:** `corpus <dir> from <list.csv> split train codec`. Each loop is one
  recording, with its hits from the MIDI file. Its keys are mapped by
  `midiInstruments`: edges, ride, tambourine and shaker read as hihat, cross-stick as
  snare, and other keys as strikes of no line.

E-GMD needs a check `corpus` doesn't make yet. Lined up with its onset flux, a
recording's hits can sit a steady distance off its audio, the same across all of a
performance's kits and with no drift over time. Such labels are moved by that lag, and
those too far off go in the corpus's `skip.txt`, a list of loop names the trainer
leaves out.

### Arranged loops

`gen` arranges every loop that is not an overlap; without it a loop plays the bank's
rhythm, melody and bass patterns, in 4/4 from a bar line. The arranger (`arrange.go`)
writes a loop's parts in steps, plays them as a player would, and lays them out as
patterns the engine renders (audio's `Step.Late` holds an onset off its step). It
draws from a stream of its own, its meter and its start from another (`meters`), and
each rule below has its test in `stapes_test.go` on the labels of `corpus <dir> 2000 4
gen meters overlaps aug glide sampled <set> labels` over a set of four recorded
instruments, no render and no weights needed. The figures are those 2,000 loops',
1,478 of them arranged.

1. **The whole keyboard.** A loop's key has its root among C1 to B1 and its scale's
   steps run on under it and over it, so a part reaches every key of the 88 its scale
   holds. Each part keeps a register: the bass over a floor between A0 and G2, the
   left hand over one between C1 and A2, a chord's bottom between B♭2 and B♭4, two
   octaves at least over its instrument's lowest key where a left hand plays under it,
   a strum's between E2 and E3, a pad's between E3 and F♯4, an arpeggio two octaves
   from between C3 and E5, a counter-line between G3 and G5, the melody centred
   between D4 and D6 in three loops of four and between C6 and D♭7 in the rest. A note
   is folded by octaves into its instrument's range (`familyRanges`: the 88 keys for a
   keyboard and a synthesizer, A0 to G4 a bass, E2 to E6 a guitar, C2 to C6 a voice,
   E1 to E7 strings, B♭1 to D7 winds; a voice on the engine's `string` wave from E1,
   `stringLowest`, whatever its family, as its line holds no longer period), and a
   recorded instrument plays only within 3 semitones of a note it has. Every key of
   the 88 starts at least 50 notes and no other pitch one: of 118,769 notes the
   fewest on a key are 76, and A0 to G♯1 hold 4.2%, A1 to B3 30.6%, C4 to C6 53.9%
   and C♯6 to C8 11.3%. The middle holds most
   because most parts live there, as in arranged music: melody, chords, pads,
   arpeggios and counter-lines all cross it, the bass and the left hand reach under it
   and high melodies and arpeggios over it. The ends hold enough that no key's cell
   goes untaught.
2. **Harmony.** Each bar's chord comes from a progression of four degrees drawn for
   the key (`corpusKeys`: major, minor, dorian, harmonic minor, a minor pentatonic
   tune over its minor's chords, phrygian), a chord held two bars in a loop in five, a
   seventh on a quarter of them. The chordal parts sound the chord that sounds (54,345
   of 54,346 notes), the bass its root on every bar line (2,850 of 2,850) and the
   melody one of its tones on most of them (3,187 of 3,493). The label lists the
   chords (`chords`, above).
3. **Chords, hands and strings.** A chord is three to five notes struck at once, its
   inversion the one nearest the last, held, on the pulses, on their offbeats, as a
   charleston, syncopated, or rolled up half a sixteenth a note, held to the next or
   short. A keyboard's left hand plays under its right on one instrument: octaves or
   fifths held, the chord broken in eighths, or a stride, each note an octave down
   while it would reach the right hand's lowest. A strum is four or five strings, down
   on the pulses and up between them over the top three mostly, spread 15 to 60 ms
   down and 10 to 35 up. Struck within three frames: 2,337 chords of three notes,
   2,535 of four and 1,175 of five, and 1,047 single notes and 135 pairs, a rolled
   chord's struck apart; 3,933 of the 6,047 of three or more inverted; of 733 loops of
   chords 269 mostly short and 134 mostly held a bar; of 4,310 strokes 4,175 spread,
   2,997 down and 1,178 up.
4. **A melody from its motif.** A motif is a span's rhythm of cells and a contour of
   steps, leaps now and then, its strong pulses on the chord's tones; a fifth tie a
   note over a pulse and a quarter strike one early. Each span of the phrase plays it
   again, moved with its chord, varied, or answered, by a form drawn, and a pickup's
   last pulse leads into it. Of the bars after a melody's first, 1,090 play its rhythm
   again (104 as it was, 655 moved, 331 varied) and 880 answer it.
5. **Texture.** A loop is a line alone to six parts (`textures`: one in 12 loops of
   100, two in 18, three in 25, four in 25, five in 12, six in 8), so the arranged
   loops hold 190 of one part, 274 of two, 364 of three, 361 of four, 194 of five and
   95 of six. A double plays the melody's notes on another instrument, at the unison
   in half the loops, an octave up in three of ten and down in the rest, over some
   bars or all: 187 loops, 1,039 notes at the unison and 917 an octave apart, 156 over
   part of the melody. One part comes in part-way in a third of the loops and another
   drops out in a quarter: of 1,288 of more than one part, 430 and 309.
6. **Rhythm.** A quarter of the loops not in a compound meter swing, the first of a
   pair 56 to 68% of it, three in ten of them on the sixteenths: of 787 in quarters,
   129 swing and 629 are straight. Triplets sound in 397 of the 1,478, notes struck
   off the beat and held over the next in 759; 510 of 1,244 melodies rest a beat or
   more, and in a loop in seven every part stops for a pulse or two before a bar line,
   the melody playing on in half of them, a break: 105 of 765 with drums.
7. **Performance.** A part is played as a player would: the strong pulses accented,
   its velocities swelling, fading or both over the loop, its onsets off the grid by a
   feel of its own and a jitter of its kind's, the kick's feel none so it anchors the
   beat; the tempo steady in three loops of ten, else walking a beat at a time within
   6% or slowing or quickening to 3 to 8% off by the end. A bar line's notes sound 4.2
   dB over the off beat's, of 3,199 parts of 12 notes or more 584 swell and 625 fade,
   22,393 of 103,969 onsets fall more than 15 ms off the grid and none of 4,878
   on-beat kicks more than a frame and a half; 754 loops drift three frames or more
   off a steady beat and 459 hold within one. The label says where each note falls as
   played, and its beats and bar lines where the tempos put them: all 2,824 bar lines
   the drums play over carry a kick.
8. **Meters.** `meters` draws a loop's meter by share (`corpusMeters`): 4/4 40%, 3/4
   15%, 6/8 12%, 12/8 10%, 2/4 8%, 7/8 8% (2+2+3, 3+2+2 or 2+3+2 eighths), 5/4 7% (3+2
   or 2+3), and starts three loops in ten part-way into a bar: half an anacrusis, the
   melody leading in, the rest cut out of the music at a pulse, half of them half a
   pulse further in. Of the 1,478: 582 in 4/4, 230 3/4, 194 6/8, 166 12/8, 116 2/4,
   104 7/8 and 86 5/4; 413 start part-way. The label names the meter and lists a bar
   line every bar's beats, a part-way start less than a bar's before its first.

### Overlapping notes

`overlaps` gives a quarter of the loops single notes in no meter, or the share it is
given (`overlaps 0.4`), drawn from a stream of its own, so the loops it leaves are
those the line draws without it. Such a loop is 4 to 9 s of one to eight instruments
(by weights 30, 20, 15, 12, 8, 6, 5 and 4), the engine's tonal voices and, with
`sampled`, the recorded ones alike, each striking at a pace of its own, 0.15 to 1.5 s
apart on average and at no grid. Each note is held 0.12 to 2.5 s, its pitch drawn
evenly over its instrument's and its velocity 0 to 30 dB down, eight sounding at most.
A fifth strike again a pitch their instrument still rings, and three in five take the
nearest pitch within a tone of the one drawn that is the unison, an octave, a fifth or
a third of a note sounding. Its test counts 522 of the 2,000 loops: 177 of one
instrument, 94 of two, then 76, 60, 40, 31, 33 and 11 of three to eight, 232 sounding
eight notes at once at their fullest; 655 recorded instruments and 883 of the engine's
over them all; 7,404 of 14,800 notes 15 dB down or more, and 2,843 struck again
ringing. A note's interval to each note it sounds over is each related one 1,533 times
and each other one 358, 4.3 times as often.

An overlap is labelled as a recording with no beats file: `bpm`, `bars`, `root` and
`scale` 0, no `meter`, `beats`, `downbeats` or `chords`, its events' `part` `single`,
so the trainer asks it no beat (§5). No loop plays a beat without a beat track: every
arranged and bank loop lists its beats.

Two instruments on one pitch at once, an overlap's unison (12 in the test) or a
double's, are two events in the label, each with its own `part`, `instr`, `vel`,
family and voice, both struck; a label never merges them. What the trainer asks of
such a cell is §5's (two notes on one pitch).

### A loop's level

Every rendered loop, after its room, its mix and its codec, is laid at a level drawn
from a stream of its own whatever it plays: its loudest half second's RMS evenly
between -32 and -14 dBFS (`loopQuiet`, `loopLoud`), lowered where its peak would pass
-1 dBFS (`loopPeak`); a loop whose peak would hold it under -32 dBFS is laid at -32
with its peaks bent under -1 from 70% of it. Its test renders 24 loops, arranged,
recorded and overlapping, which lie from -31.3 to -15.2 dBFS.

A loop of recorded notes alone, whose engine render is silent, gets air before that:
white noise over 7.5 kHz, 45 to 65 dB under its loudest window, a recording's noise
floor, from a stream of its own. Its test reads three such loops at -63.6, -63.3 and
-49.1 dB under their loudest windows over 9 kHz.

### The words

- `gen` is one word for rules 1 to 7: they are one arrangement, its parts, chords,
  motif, texture, rhythm and performance drawn together, and none alone is music with
  a real arrangement's textures.
- `meters` is a word of its own: it moves every beat and bar line, so a run can keep
  4/4 and `bench ... meters` grade each meter alone. It changes only the loops it puts
  in another meter or starts part-way.
- `overlaps` is a word of its own: notes in no meter teach the beat heads silence and
  the pitch heads chords no harmony wrote, a share a run sets. It changes only the
  loops it takes.
- `glide`, `aug` and `sampled` keep their words: each changes only what it names, as
  its test checks, a moving note's cents, the voices, the instruments.
- `dense` and `midi <dir>` are gone with the MIDI files they played: the corpus plays
  generated music only, and the arranger writes the textures they were for.

### Labels without a render

A corpus's line with `labels` after it writes every `loop_NNN.json` of the line, and
`corpus.json`, over those on disk, and renders nothing: each is what the line's export
writes, byte for byte, and no `.bands` or WAV is touched. A field the labels gain
reaches a corpus that way without its hours of render.

- A loop's events are drawn and never heard: its arrangement or patterns, key,
  tempo, meter, voices and tuning come from the line's seeded streams, the same with
  audio and without.
- A label takes nothing from the render. Its length, `frames`, and with it the cut
  of the events that start past the end, is counted from the samples the loop is
  cut to as `BandFrames` makes frames of them (`loopFrames`), which a render checks
  against the frames it heard. A loop with no frames whole on disk (§2's header
  against the file's size, 630 values a frame) is labelled all the same and counted,
  and `corpus.json` is then not written.
- A `sampled` line's labels also say which recorded notes were loud enough to play
  and where each was let go. The pass reads those recordings again from the NSynth
  folder, their level and length only: none is shifted or mixed.
- `part k/n` labels every n-th loop from the k-th, as it renders them, and leaves
  `corpus.json` to a run without it.
- A `from` line writes each recording's label again from the list: its family, its
  velocity word, its starts, its MIDI file, its pitch track and its beats. The
  recording is known by the `source` of the label on disk, and its WAV is read only
  where its velocity word is `measured`, so it may otherwise be gone; a label that
  does not read stops the run, its recording being unknown then. A `measured`
  row whose WAV is gone keeps its label, each of its notes at `vel` -1, and the
  run's last line counts them (a recording's velocities, below).

### A recording's pitch track

A `from` list's `pitch` column names a recording's pitch track, a path from the
list's folder as its other files are. An empty cell names none, and that recording
is labelled as without the column, byte for byte. A track is a collection's
annotation, or, where the collection has none, a measure of the audio by a
procedure that is stated, fitted on sound whose answer is known and checked on
what it was not fitted on; never an unchecked estimate. Two forms, told apart by
the file, a JAMS file being the one that opens with a brace:

- A CSV of seconds and Hz for one voice, 0 Hz where nothing is voiced, under a
  header or none: vocadito's F0 files. Every note of the recording is that voice's.
- A JAMS file with a `pitch_contour` annotation for each string beside its
  `note_midi` annotation, the two paired by their `data_source`, or by their order
  where none is given: GuitarSet's. A note takes the contour of the string whose
  `note_midi` annotation holds a note within half a semitone of its pitch sounding
  at its onset, a frame's slack before it, the nearest onset when two strings do.
  A note no string holds lists nothing.

A tracked note lists `cents` as an engine's moving note does, from `frame` to `end`,
in cents from its cell centre to a tenth:

- Frame k is read at k x 256 / 22,050 s, on the line in semitones between the
  track's samples either side of it, or at the nearer one alone where the other is
  unvoiced or missing.
- A frame is `null` where the track does not voice it: its nearest sample is at 0
  Hz, marked unvoiced, or further off than three quarters of the track's usual
  step, which a missing sample is. It is `null` too where the track lies a semitone
  or more from the note. A `null` frame asks the cents head nothing (§5).
- A note the track places at no frame lists none, as a recorded note with no track.
- The run says how many it placed: `6 of 8 note(s) placed by a pitch track`.

A trainer from before the `null` stops on such a label or reads the frame as silent.

### A recording's beats

A `from` list's `beats` column names the file a recording's beats are read from, a
path from the list's folder. An empty cell names none, and that recording is labelled
as without the column, byte for byte. Beats are a collection's annotation or its
score's, never tracked from the audio. Two forms, told apart by the file's name:

- A MIDI file (`.mid`, `.midi`) whose bars are the recording's, as E-GMD's own is,
  so the cell may repeat `midi_filename`: its beats under its time signatures and
  tempo map, on the recording's clock, from the beat holding its first note or hit
  to its last's end, a bar line from the first full bar on (audio's
  `MIDIScore.Beats`).
- Any other file is a JAMS file with a `beat_position` annotation, as GuitarSet's:
  each datum a beat, a bar line where its position is 1 (audio's `ReadJAMSBeats`).

The label lists them as `beats` and `downbeats`, each second on the frame an event's
is, those at or past the recording's last frame left out:

```
...,"source":"a.wav","beats":[26,69,112,155,198],"downbeats":[26,198]}
```

`labels` after the line writes them into a corpus already exported, from the list
alone (above). What the trainer asks of them is in §5.

### A recording's velocities

A `from` list's `velocity` column holds a word for each recording, what its MIDI
file's velocities say of how its notes were played, a row a word (`velocityWords` in
`tune.go`):

- An empty cell, or no column: they say it, and each note and hit is labelled at its
  MIDI velocity.
- `none`: they say nothing, as a lab's one velocity for every note does (GuitarSet's,
  VocalSet's), and each note and hit is labelled at `vel` -1, which asks the velocity
  head nothing (§5).
- `measured`: they say nothing, and the recording says it instead: each note is
  labelled at the velocity it is heard at (below), -1 where that cannot be told,
  and each hit at -1.

Any other word is refused, naming those the table has. Without `measured` no
recording is opened for its velocities, and a list's labels are as they were
without the word, byte for byte.

A measured velocity is the one at which the engine plays a note as loud as the
recording played it. The engine's law is straight (`vel`, above), so it is the
note's level over its recording's unit, the level a note of velocity one has there,
and at most 1 (`velocity.go`):

- **The level** is a sampled note's: the RMS of its loudest window of 50 ms, hops of
  half a window (`windows`, the one function the sampler and the measure read a
  level by), over its held span, from its start to its end or 3 s on (`heldSpan`).
  It is read from the note's own partials (`hearLevels`), so that the notes
  sounding with it are kept out:
  - Each window is fitted under a Hann window, by least squares, to the note's
    partials at the pitch its track gives, or at its MIDI pitch where no track
    places it: k f0 √(1 + b k²) for k up to 80 and up to 20 kHz, b the note's
    stiffness, of a grid from none to a piano's top octave's the one whose partials
    hold most of its first window (`stiffnesses`), read to 60 dB under its
    strongest there. Fitted with them is every partial of the other notes sounding
    in the window, each taken to ring 0.25 s past its label's end, that lies within
    three bins of the window (60 Hz) of one of them or of a point midway between
    two.
  - A partial is the note's own where no other note's lies within those three bins,
    or where the others widen its estimate's variance twice at most
    (`measuredApart`), and its energy bounds the level from below and above by what
    is left unfitted either side of it. A partial two notes share past that is not
    told apart: it gives the low bound nothing and the high bound all the window
    holds at its frequency, so a shared partial widens the bounds and never moves
    the level by a guess at its split. A window another note is struck in holds an
    attack no partial fits, so it gives no low bound.
  - The loudest window's level lies between the greatest low bound of any window
    and the greatest high bound. The level is their middle, their mean squares'
    geometric mean, and it is known within half their width, 5 log10(hi / lo) dB.
- **The bound.** A note known only more widely than `measuredBound`, 1.9 dB, is
  given -1. The bound is fitted on sound whose levels are known
  (`TestAMeasuredLevelIsKnownWithinItsFittedBound`): overlaps (§3) of the bank's
  voices, one to eight notes at once at drawn velocities over every interval,
  three loops at seed 1 with no room and through `degrade`, each note's truth its
  level rendered alone and carried through the same room (`room`). The bound is
  the least, in tenths of a dB, within which nine in ten of the notes known within
  it are measured, of twenty at least. The test checks it on seed 2 and on the
  plucked and vocal focus's voices, which the fit leaves out, and logs the error by
  how many notes sound at once and by the most coinciding interval a note sounds
  at: unison, octave, fifth, third, another, or none. A note that shares most is
  known least: in those loops fewer than one note in ten is known, none at a
  unison, so a velocity is measured where a note stands apart, as in a solo line.
- **The unit** is one number for a recording and a MIDI program, by one rule: the
  recording's loudest notes are given the velocity the generator gives a line's
  loudest. `lineTop` reads that from the arranger's own draw, the median over the
  tonal parts of 64 loops it arranges at a fixed seed of a part's ninetieth
  percentile of velocity, 0.570. The unit is the ninetieth percentile, by nearest
  rank, of the levels known within the bound among the program's notes in the
  recordings measured together, over that. The `unit` column names which
  recordings are measured together, those whose cells are alike, as a singer's
  takes made at one gain, so a soft take's notes stand under a loud one's; an
  empty cell, or no column, measures a recording alone. A program none of whose
  notes is known has no unit, and its notes are given -1.

A voice whose tone moves with its velocity is not straight in its level: the bank's
piano and bass, whose hammer's filter opens and whose knock rises with it, and its
pluck, whose burst's filter does, rise under a dB a dB of velocity, and a filter
envelope moves a pad's and a lead's (`TestAVoicesLevelIsStraightInItsVelocityWhereItsToneIsNot`
logs each voice's slope). A measured velocity reads every note by the straight law,
the sampler's and every other voice's, so it is the velocity a sampled note is laid
at to sound as loud; that the bent voices' labels keep the velocity they were
played at is in the todo.

A `measured` run counts what it gave after what it heard: `wrote the labels of 2
recording(s) in out again from list.csv and heard 1, 4 note(s) given the velocity
measured and 4 none, 4 of those none in recordings gone from disk`. A row whose
recording is not on disk keeps its label, each of its notes at -1, as under `none`.

### A recording's starts

A `from` list's `starts` column says how its notes' starts were found, a row a word
(`startsWords` in `tune.go`):

- An empty cell, or no column: each is where its note is heard struck, and the
  label is as without the column, byte for byte.
- `placed`: they are placed on the recording by an alignment, as a take's notes
  that its exercise sheet writes and a procedure fitted on known sound lays on the
  take, and such a start can miss its strike by frames. A strike target a frame
  wide there would teach the alignment's miss, so the label says `placed` (§3) and
  its notes ask the strike head nothing (§5). Their frames, velocities and cents
  are asked as any note's, the onset head at each start and the release head at
  each end, which the same alignment places.

Any other word is refused.

## 4. The input: four planes

A frame becomes up to four planes of 316 values, laid end to end: the fine plane's
bands in dB, the sharp plane's, then each one's PCEN. A dB plane is its plane's band
shape and level, b over its 315 bands:

```
e[b]       = expm1(v[b]) / 100
db[b]      = 10 log10(e[b] + 1e-8)
level      = mean of db over the plane's 315 bands
plane[b]   = (db[b] - level) / 80
plane[315] = (level + 80) / 80
```

A gain moves only `plane[315]`, so a phrase 20 dB softer reaches the models with the
same shape. A PCEN plane (per-channel energy normalization) is each band of its plane
over its own recent average, worked out per band from the frame's energy:

```
s         = 1 - exp(-(256 / 22050) / 0.4)  = 0.028608
M[b](t)   = (1 - s) M[b](t-1) + s e[b](t),  M[b](-1) = e[b](0)
pcen[b]   = (e[b] / (1e-6 + M[b])^0.98 + 2)^0.5 - 2^0.5
pcen[315] = mean of pcen over the plane's 315 bands
```

- Dividing by M is a gain control: a band that has been loud is turned down.
- The 2 and the root compress the result: a band far under its average lands near 0,
  not at a logarithm's large negative number.
- M only looks back, so a live rip can compute it as the sound arrives.

What PCEN reads for bands driven with test envelopes:

```
steady at -6 dB                   0.31
steady at -46 dB                  0.25
fading 60 dB over 5 s             0.25 at 0.1 s, 0.08 at 1 s, 0.02 at 3 s
struck from -40 to -20 dB         0.27 before, 3.5 on it, 1.4 at 50 ms, 0.58 at 200 ms
struck 6 dB over a band sounding  0.29 before, 0.90 on it
```

So steady noise sits near 0.3 at any level and a note on top still rises over it, but
a note that's still ringing fades towards zero. That's why PCEN is two planes more and
not a replacement: they say when something happened, the dB planes what is sounding.

PCEN lies close enough to the dB planes' scale to need no rescaling. A frame model of
two channels reads the two dB planes, 632 values, and of four (`--pcen`) all four,
1,264; a notes model reads all four.

`--pitchnet` reads the four planes through a harmonic stack it computes itself. Bin k is
band k, so nothing is interpolated: each plane gives 8 channels, one per harmonic of 0.5,
1 and 2 to 7, its bands shifted by -36, 0, 36, 57, 72, 84, 93 or 101, so a pitch's
harmonics line up at its own bin, zero past the bands. The 270 bins run from MIDI 20, a
semitone under A0, to a semitone over C8, key 21+i's cell centred at bin 3+3i, in 32
channels, each ending in its plane's scalar.

## 5. The targets

`Label.window(a, b)` builds what each head should say for frames a to b. A note's pitch
index is its MIDI number less 21, A0's, and one outside the piano's 88 keys sets no pitch.
A note that carries `cents`
sounds, is struck and is let go in the cell nearest its frequency at each frame, so a
glide's pitch target walks the cells it crosses.

- `onset`: within one frame of any event, drum or note, that is `t-1 <= frame < t+2`.
- `drums[d]`: the same, for that drum's hits.
- `pitch[p]`: from a note's `frame` until its `end`, at least one frame.
- `voicing`: any pitch on.
- `strike[p]`: within one frame of a note's start, and 0 throughout a loop whose
  starts are `placed` (§3).
- `placed`: a loop's frames whose starts are placed, which the strike's loss and its
  F1 leave out (§8), so such a note asks the strike head nothing.
- `release[p]`: within one frame of a note's `end`, unless that's past the loop.
- `velocity[p]`: the note's `vel` on its strike frames, scored only there, and nowhere
  for a `vel` of -1 (§3).
- `wave`, `voice`: those of the highest note sounding with a known voice, scored only
  on frames that have one.
- `family`: that of the highest event sounding, a hit or a strike at midi 0 on its own
  frame, scored only on frames where it is labelled: a hit never stands in for an
  unlabelled note over it.
- `beat`, `downbeat`: within one frame of a beat of the meter and of a bar line, the
  frame model's alone, scored where `metered` says. A beat is the meter's own note, a
  quarter in 2/4, 3/4, 4/4 and 5/4 and an eighth in 6/8, 7/8 and 12/8, as audio's
  `Rip.Beats` lists them. Every loop's are its label's `beats` and `downbeats` (§3):
  a rendered loop's where its tempos put them, a recording's where its list says.
- `metered`: the frames those two are asked at.
  - A rendered loop: every frame but its last two, which a beat on the end's mark,
    past the list, would light. A loop that starts part-way is asked from frame 0:
    the beat it starts after is past, and its first listed beat is where the next
    falls.
  - A recording: from the first frame of its first listed beat's mark to the last
    frame of its last's, and no further. Past them the list says nothing: where the
    next beat of a played tempo would fall is a guess, and a zero there could teach
    silence on a beat nobody listed. A recording that lists none is asked nothing,
    and so is an overlap, whose notes keep no meter.
  - None of a shuffled window or of one `--over` lays drums under (§6), whose hits
    keep another grid, nor the padding past a loop's end.
- `cents[p]`: how far the sound nearest the cell sits from its centre, in semitones,
  asked where the label places that sound, an engine voice's note or one that lists
  its `cents`, and within a semitone: in the cell a pitch sounds in, and in the cell
  beside it on the side the sound leans to, so both cells at a boundary say where the
  sound is (cell k at +0.5 is cell k+1 at -0.5) and a cell lit by its neighbour's
  sound says so. A recorded note that lists none is unplaced, and its own cell is not
  asked; nor is a frame its pitch track left `null`, which sounds in the note's own
  cell. `moving[p]` marks a glide's frames for the scores of §8. An engine note's are
  those where the label's cents differ from the frame before or after. No pitch
  track holds a note still, so a recorded note's are every frame of a note whose
  track spans half a cell, 50 cents (`GLIDE_SPAN`), and none of a note under that.
- `pitchfamily[p]`, `pitchwave[p]`, `pitchvoice[p]`: what plays the cell, the notes
  model's alone: the family, the wave and the voice of the event that sounds in it,
  asked where one event alone sounds in the cell, in each cell a note that moves
  sounds in. A cell two events sound in asks nothing: its sound is both, and either's
  name would be taught on the other's sound. The family is one
  of the eight names, asked only where an instrument stands behind the sound: a
  recording's own instrument, a singer, a sampled note's family as its set gives it.
  A note the engine rendered asks no family, its voice registered or drawn far from the
  bank, by name or by a score's program: the family a corpus writes for it is the
  registered voice's it was drawn under, within a reach of genes (`voiceFamily`), so two
  redraws that sound alike carried different names. It asks its wave and its voice. A
  recorded note of no family or of none labelled asks no family, so `unknown` is never
  taught: it is the absence of a name. A recorded note asks no wave and no voice. The
  family held out (`--family-holdout`) asks no family here and in the frame's
  `family`, its notes' sound, wave and voice asked as any. A voice is a row of one
  table of the corpus's voices (`train.GENES`) and a cell's target that row's index.

**Two notes on one pitch.** Two parts often play one pitch at once. The yes-or-no planes (`pitch`, `strike`,
`release`) see one note there and `velocity` the later listed; the notes model is asked no
name there (`sole`, above). The frame model's `family` takes the highest note's, the first
listed where that note is doubled: one name of two, which no decoder reads, and asking none
there is a todo. A truth a rip is scored against holds one note a frame and pitch, as a rip
reports one; a name is right there when it is any family sounding in the cell, and doubled
notes' names are read apart. What a rip says of a doubled note is untaught. Its name is to
be decided over its line within a section, in the decoder. The corpus plays two
instruments on one note, a double at the unison and an overlap's unisons, each listed as
its own event (§3); teaching two names at once, a yes or no a family rather than one share
among them, is a todo.

A shuffled window (§6) asks a cell its cents where one of its two clips alone sounds
within a semitone of it, and what plays it where one of them alone sounds in it.

A rendered loop's beats are where its notes sound: audio's tests render late and
drifting steps and find each on the sample its expected event names, the render
ending on the last beat's mark, and stapes' find a kick within a frame of every bar
line an arranged loop's drums play over.

A window running past a loop's end is padded with zero frames, which a `valid` mask
leaves out of every loss.

## 6. What may change a window's frames

These run on the stored frames during training, so one corpus serves many runs. All
add in power: `e` is recovered, changed and stored back through `log1p`. Their order in
`sample` is the overlay, then the channel, then the planes. With PCEN's planes, 128
frames (1.5 s) before the window are read and dropped afterwards, so PCEN's average has
settled and matches what it would be mid-recording.

**`--channel file.npz:share`** hears a share of the windows through a small speaker, a
room and a phone. It's drawn per window:

- Bands under a low corner, 60 to 400 Hz, are cut by up to 24 dB, reached 1.5 octaves
  down.
- Bands over a high corner, 3 to 10 kHz, are cut by up to 24 dB, reached one octave up.
- A ripple of up to 6 dB runs over 9 knots spaced on the mel scale.
- A steady floor is added: a band mean drawn from -78 to -50 dB, shaped as the real
  room's in 2 draws of 5 and pink, brown or flat otherwise, with up to 4 dB of ripple.

The bank holds two arrays. `white`, 630 values a frame of float32, is two minutes of
white noise as band frames, both planes, with each band's mean set to 1, so a band keeps
its own frame-to-frame spread. `room`, 630 float32, is a real room's band shape in dB
about its mean, as a phone hears it. `make_noise.py` builds it from a WAV of the room
with no music; a bank of another width is refused.

A corpus's own noise sits 40 to 65 dB under the loop, a phone's far nearer the music.

**`--over dir:share`** lays real drums over rendered music. A share of the windows draw
a loop with no drums of its own, then a recording from a corpus of drums alone, placed
at random. Its level is -12 to +6 dB of its mean power
against the loop's, and uncorrelated sounds add in power, so `e = e_loop + gain x e_drums`.
A hit joins the onset and drum targets only where it can be heard: what it adds, over its
best ten bands of the sharp plane, must top everything else sounding (`stands_out`),
judged after the channel. The window is asked no beat and no bar line: the drums laid under it keep
their own.

**`--mix dir:share`** draws a share of the windows from another corpus as it stands.
`--extra dir:share,...` does the same for any number of corpora.

**`--shuffle share`** splices a share of the windows from chunks of a second, each two clips
from random places summed, the second at -6 to +6 dB of the first's power. Notes keep their
labels, a drum hit counts where it stands out of the other clip, and wave, voice,
family, the beat and the bar line go unasked.

**`--sung dir:share`** draws that share of the chunks from a corpus of solo singing.

**`--ring share`** adds to that share of the windows a body's or a room's ring at every attack:
one pitch whatever the note, 70 to 160 Hz, decaying in 0.05 to 0.3 s, 25 to 8 dB under the
attack's rise in energy, its harmonics absent and nothing labelled, as a guitar's air cavity
rings near 104 Hz on every pluck. The rise is read in the sharp plane, which hears the attack
when it comes, each band's energy over the times the plane's bands count a steady tone there,
about twenty at A0 and one at the top, so a low note's rise counts once.

**`--bend share`** reads that share of the plain windows along a tempo that sways: a rate up
to 18% either way and a slow sine of up to 12 frames, the features interpolated and the
targets taken nearest, so no steady grid or pulse is there to learn, as a pianist's isn't.
The beat, the bar line and their mask are taken the same way and sway with the hits.

**`--strike-weight w`** is a strike's weight against a frame without one, 20 by default.
**`--again-weight a`** weighs a strike of a pitch still sounding a times more.

Draws from both are by length, since E-GMD's recordings run from seconds to minutes:
drawn by recording, the short ones would take far more of the windows than of the
audio.

## 7. The models

`tool/rip-train/ripmodels.py` builds the frame model, and `transformer.Forward` computes
the same numbers in Go. The shapes through a window, with B the batch:

```
input          B x 512 x 1264
reshape        B x 4 x 512 x 316             channels x frames x features
conv 0         B x 16 x 512 x 316            3x3, GELU
conv 1         B x 16 x 512 x 158            3x3, GELU, max over 2 features
conv 2         B x 32 x 512 x 79             3x3, GELU, max over 2 features
flatten        B x 512 x 2528
input layer    B x 512 x 256
4 blocks       B x 512 x 256                 pre-norm attention and feed-forward 1024
norm, output   B x 512 x 405                 raw scores
```

Attention has 8 heads and looks 64 frames (0.74 s) either side, in every block. There
are no position embeddings: each head subtracts a slope of `2^(-8(h+1)/8)` per frame of
distance from its scores (ALiBi), so a window of any length works.

The 3,918,613 weights: 592 + 2,320 + 4,640 in the convolutions, 647,424 in the input
layer, 789,760 in each block, 512 in the final norm, 104,085 in the output layer, 2,313
of them the family head's and 514 the beat's and the downbeat's. A model of the two dB
planes has 288 fewer, all in `convs.0.weight`.

The 405 outputs, in the order the file lists the heads:

```
offset    size  head       read as
  0         1   onset      sigmoid
  1..3      3   drums      sigmoid: kick, snare, hihat
  4..91    88   pitch      sigmoid: MIDI 21 to 108
 92         1   voicing    sigmoid
 93..97     5   wave       softmax: sine, saw, square, string, modal
 98..129   32   voice      sigmoid: genes
130..217   88   strike     sigmoid
218..305   88   release    sigmoid
306..393   88   velocity   sigmoid
394..402    9   family     softmax: piano, guitar, voice, bass, strings, winds, synth, drums, unknown
403         1   beat       sigmoid: a beat of the meter within a frame
404         1   downbeat   sigmoid: a bar line within a frame
```

Every model's pitches are the piano's 88 keys, MIDI 21 to 108. A file has every head;
only a notes model's may lack the cents head, as a probe's does (§9).

`--pitchnet` builds another model on the same heads but `family`, the frame model's to hear:
convolutions only, the same weights at every pitch, so 172,181 weights in all, 66 of them the
cents head's and 1,485 those of what plays each pitch. Its file is `ripnotes.bin` (§9).
`--dilations 3,9,27` adds a residual 3x3 over the cells for each, its frames that far apart:
each note head then hears half a second either way instead of 80 ms, 199,925 weights, about
1.5 times the compute. Its per-pitch heads are a table, `CELL_HEADS` in `ripmodels.py`, a row
a head with the values it reads at a pitch; the cents head reads two, the offset in semitones
and the log of the error it states, and what plays the pitch is three heads: `pitchfamily`,
a score for each of the eight names, `pitchwave`, one for each wave, and `pitchvoice`, the 32
genes. They are heads of their own, read straight off the cell's 32 channels as every other
head is, and not a small embedding with linear readouts. A linear embedding of E values under
linear readouts is the same map from the channels held to rank E, so it can say no more than
the heads; what it could give is fewer bytes a pitch in audio's frames, which keeps the
heads at twelve pitches a frame, 552 bytes. The eight names share one softmax.

```
input          B x T x 1264             the four planes
stack          B x 32 x T x 271         the harmonic stack of section 4
reduce         B x 8 x T x 270          1x1, GELU, the scalars set aside
wide           B x 16 x T x 270         3x39, thirteen semitones, GELU
cells          B x 32 x T x 90          7x3, stride 3 over features, one cell a semitone, GELU
near           B x 32 x T x 90          7x3 with a residual, GELU
long           B x 32 x T x 90          3x3 dilated in time, one each of --dilations, with a residual, GELU
notes          B x 51 x T x 88          1x1: pitch, strike, release, velocity, the cents head's two,
                                        and the 8, 5 and 32 of what plays the pitch
frame          B x T x 42               wide pooled to 16 regions over every bin plus the four
                                        scalars, a 1D conv of 5
                                        and one of 9 dilated by 2, linear: the other heads
```

## 8. A training step

Each step: 8 windows, targets, forward, losses, backward, gradient clipped to norm 1,
AdamW at weight decay 0.01. The learning rate runs from 3e-4 to zero on a cosine over
the steps. The batches are built beside the device by `--builders` processes forked
from the run, one by default, each at most two batches ahead, handed over in index
order: a batch is Python holding the interpreter's lock, so a second thread builds
less, not more. Batch k draws every window from a stream of its own,
`random.Random(seed + (shard << 32) + (k << 64))` (`stream`), so it is the same
tensors whatever builds it, a run resumed at step k draws the batches the unbroken run
drew from k on, and two shards draw different batches.

```
onset, drums, pitch, voicing, strike, release: binary cross-entropy with a weight on
    the positive class of 8, 8, 12, 2, 20, 20, the strike's on frames not placed
velocity: squared error, on strike frames with a known velocity
wave:     cross-entropy, on frames with a known voice
voice:    squared error, on frames with a known voice
family:   cross-entropy, on frames with a known family, the frame model's alone
beat, downbeat: binary cross-entropy with no weight on the positive class, on the
          frames asked (metered), the frame model's alone
cents:    absolute error in semitones, where asked, the notes model's alone
pitchfamily, pitchwave: cross-entropy over the eight names and over the waves, at the
          cells asked, the notes model's alone
pitchvoice: squared error over the genes, at the cells asked
spread:   miss / e + log e where asked, e the error the head states and miss the
          offset's absolute error held fixed: a Laplace's scale fitted to it

total    = onset + drums + pitch + 0.5 voicing + strike + release + 2 velocity + 0.5 wave + 2 voice
           + 0.5 family + beat + downbeat + 2 cents + 0.2 spread
           + 0.25 (0.5 pitchfamily + 0.5 pitchwave + 2 pitchvoice)
decoders = onset + drums + pitch + 0.5 voicing + strike + release + beat + downbeat
```

The beat and the downbeat are the onset head's kind of loss with one difference: no
weight. Audio reads the two by their level, as posteriors: the tempo from how the beat
head repeats and the beats tracked on it, a bar line where the downbeat head summed by
place stands highest over its mean beat, beside the cues (audio §9), so a weight on
the marks, which lifts a head most where it is unsure, would move a decode. A frame
of a mark and a frame outside one weigh the same, and a head that knows nothing says
the share of frames a mark covers, 7% for the beat at 120 bpm and under 2% for a bar
line of 4/4.

The stated error learns from the offset's miss and never the other way: the miss
enters `spread` with its gradient stopped, so stating an error moves no offset. At
its best the stated error is the mean miss to expect of such a reading, which 63% of
a Laplace's fall within; it is what audio's decoder widens a reading by before it
calls two cells one frequency, with no factor on it. The decoders read the cents head
too, but it does not join `decoders`: its loss is in semitones and no likelihood.

What plays a cell is weighed as what plays the frame is, at a quarter of it (`PLAYS`, `--plays`).
Its heads teach the channels the decoders read, and they have to: read with the gradient
stopped at the cells they learn nothing past the commonest name, wave and voice, so the
cells carry no timbre the decoders did not ask of them. At the whole weight the strikes
pay. The names want recorded families and a round's length (todo, "The instrument
model").

The held-out and validation lines say what the head reads, after the strikes' F1: the
median miss, the median error stated and the share of readings whose miss is within
the error stated for them, over every placed reading and then by kind (`CENTS_KINDS`):
`gliding`, a cell its sound is on a glide in; `steady`, one it holds its pitch in;
`beside`, the cell next to a sound; and `unplaced`, a sounding cell the label does not
place, where only the error stated can be told: a recorded note with no pitch track,
and a frame its track left `null`. A recording whose list names its pitch track (§3)
prints the first three rows as an engine loop does. 63% within is a head that states
its misses' mean; a share well under it on the gliding readings is a head too sure
of itself where the decoder leans on it.

A frame model's lines give `beats' F1` and `bar lines'` after the decoders' loss:
each head's peaks at a half or over (`BEAT_BAR`) among the frames asked, a peak right
when a beat the labels give lies within four frames of it, counted as the strikes'
are (`struck`). Over a corpus that lists no beat, such as a `--validate` corpus whose
list has no `beats` column, the line says `no beat or bar line asked` and gives
neither head's loss: a mean over no frame prints 0.000, which reads as a head never
wrong, and adds nothing to `decoders` either way.

A notes model's lines then say what plays each note, read as audio reads it (§10): a
note's family, wave and genes are its cells' say over the frames it sounds, each frame
weighed by the pitch head's salience there, over the cells among the frame's
`VOICE_CELLS` (12) most salient, which are the cells audio keeps. The line gives the
share of the notes with a family asked whose highest share is theirs, by family, then
that share over the notes another note sounds over, the share of the engine's notes
whose wave is right, and the genes' miss, an RMS over the 32.

`--family-holdout <family>` tells no head that family's name: its notes stay in the run,
their sound, wave and voice taught, and neither the cells' head nor the frame's is asked
their family. The line then gives what those notes reach: the highest share any name
takes over each, at the `SURE` quantile (0.95) of them and at the median, and how many
of the other families' notes pass that reach and how many of those are named right. That
reach is the measure of where a name stops being sure: a note names a family in audio
only past it. The run writes it into its weight file (§9), and `--sure <share>` gives it
to a run of the same recipe that holds nothing out, as the notes model to ship has every
family's name; a file of neither names no note.

The held-out loops, the last `--holdout`, are scored every `--every` steps, and the step
with the best `decoders` score is what gets written, unless `--validate` names a corpus of recordings
(`corpus <dir> from`): cut into windows the same way and scored beside them, its best
`decoders` step is written, the held-out score printed beside it. A notes model is
picked there by its strikes instead: the strike head's peaks over the decoder's bars
(0.7, and 0.9 under E3) against the labelled strikes within four frames, as an F1
over the frames not placed (§5), since its loss holds an onset head no decoder
reads from it. The wave, voice, velocity and family heads don't choose it,
because they memorize the corpus. A positive
weight w shifts a head's scale: it scores a posterior p as wp / (wp + 1 - p), which is
part of why thresholds belong to the weights (§10); the beat and the downbeat carry
none and are read with no threshold.

The training lines name each head's loss, a model with the family, beat and downbeat
heads printing theirs last. The wave head is taught on the engine's voices alone, so on
a real instrument its guess means little, which is why a rip can play a piano on a saw.

The checkpoint, a PyTorch `.pt`, holds `model`, `opt`, `step`, `cfg` (the shape, which a
resumed run must match), `best` (the best held-out score, its step and its weights) and
`validated` (the best validation score, its step, its weights and the held-out score there);
no stream, since the step is all a batch's draws need.

## 9. The weight file

[Transformer's format](https://github.com/lixenwraith/transformer/blob/main/doc/format.md)
is the contract. All numbers are little-endian. An FNV-1a 64 of every byte before it
ends the file. The checksum catches damage, not tampering. A file only ever holds
numbers, and `Load` checks its shape against its tensors. The trainer writes version 6:
a `kind` word
after the version, 0 for the frame model, twelve heads, `beat` and `downbeat` of one value
each after `family`, an `out.weight` of 405 x 256, and after the heads the constants it
states, a count and for each a name and a float32: one, `layout`, the layout of the
frames it hears, 1 (audio's `ripmodels.Layout`, §2). Audio refuses a file of another
layout, or of none, by name, and so do the trainer's `--init` and `average.py`.

`--pitchnet` writes the same version, whose `kind` word, 1, makes the notes model
transformer's cell net. Its sizes follow, from `bins` to the frame path's dilation and
scalars, then the long convolutions' spacings and the bins its regions pool, the first and
how many (0 and 0, every bin), then the heads, the per-pitch ones first: pitch,
strike, release, velocity, cents, pitchfamily, pitchwave and pitchvoice, then onset, drums,
voicing, wave, voice and sure. `cents` is 176 numbers a frame, the 88 offsets then the 88 log
errors, `cellout`'s fifth and sixth channels; `pitchfamily` 704, `pitchwave` 440 and
`pitchvoice` 2,816, a value's 88 pitches together, `cellout`'s next 45. `sure` is the measure
of §8, written as a frame head of one value whose row of `frameout.weight` is zero and whose
bias is the measure, so every frame states it. A file without the cents head, a probe's,
loads and rips with no glides. Version 6 needs transformer v0.7.0, and its layout audio
v0.30.0. The tensors are
`reduce`, `wide`, `cells`, `near.0`, `long.{i}`, `cellout`, `frame1`, `frame2` and `frameout`, each as its `nn.Conv2d`,
`nn.Conv1d` or `nn.Linear` holds it. In the stapes root the file is `ripnotes.bin`.
Beside the frame model's `ripframe.bin` it gives pitch, voicing, strike, release, velocity
and cents, and the frame model keeps onset, drums, wave and voice. Alone it is unused. `average.py --out ripnotes.bin
a.pt` turns a `.pt` of either model into its file.

## 10. Reading the models back

`ripmodels.Hear` runs `Forward` over all of a recording's frames, and a notes model's over
their harmonic stack, built in Go as §4 builds it, takes the sigmoid of
each head and fills `RipFrames`, the same fields the analysis fills, and from a cents
head `Cents`, a hundred times each offset, and `CentsError`, a hundred times e to each
log error. What plays each pitch it keeps at each frame's twelve most salient pitches
alone, as bytes: the softmax of the family's and of the wave's scores and the sigmoid of
the genes (`RipFrames.VoiceAt`, `Voices`), and the measure once (`Sure`). A frame model's
beat and downbeat heads fill `RipFrames.Beat` and `Downbeat`, and where the frames carry
both, the tempo and the beats are read from the beat head and the bar lines from the
downbeat head beside the cues' rules ([audio
§9](https://github.com/lixenwraith/audio/blob/main/doc/audio.md#9-ripping-a-recording)).

The decoders then read the frames: hits by peak picking, the bass by one hidden path,
the melody register by strike peaks. `strike` and `strike_again` are the thresholds
that matter, and they belong to the weights, so a root pins them beside the file, in
the `[rip]` table of its `rip.toml`.

## 11. Reading a candidate

A round's file is read by what it can do, each on recordings no round trained on: notes
(the mean over the truth-backed conditions of F1 by pitch and onset), drums (hits' F1 on
real drum recordings, and false hits on recordings with none), bars (GuitarSet's unseen
players' clips whose bars are right, a scored recording's downbeats on a line) and voices
(notes named right on the solo sets). It is read with settings fitted for it, a judge on
its own rows and thresholds tuned on its own heads, never through the installed model's.

- Against what is installed each of the four holds, gains or loses, a difference
  counting only beyond what chance gives: until each has a measured spread, the notes'
  mean .01, any one condition .03, drums' F1 .02, and for clips the two-sided count of
  those only one side has right.
- Gains somewhere and holds elsewhere: installed. A part that gains is installed without
  the part that does not, as a body without its new heads.
- Gains somewhere and loses elsewhere: a trade, which goes to Lixen with the numbers and
  a pair of excerpts; no rule drops it.
- Gains nowhere: not installed and not thrown away. The model's next round starts from
  its weights wherever they hold, and the round's record says what it taught.
- A new head is read first as one more piece of evidence beside what the decoder already
  uses, and replaces something only after that. A round validates while it trains on
  real held-out recordings of what it teaches and picks its step there; held-out loops
  of the engine say how well it learned the engine.

A round's PR ends with a table by the four and one of: install, install this part, a
trade for Lixen, the next round's start.

## 12. The scores

- **`truth <name> <file>`** finds a time map from the longest common subsequence of
  pitches in time order, at the transposition within an octave that matches most, and
  counts a note right when one of its pitch starts within 0.1 s of its place on that map
  (audio's `MatchNotes`). For a rhythm given with no pitches it matches onsets on a
  fitted time map. Against a MIDI file it also
  counts hits per drum, within half a step, under a fitted stretch and offset.
- **`bench`** renders fresh loops and grades each against what was played: a note
  counts only at its exact step and pitch in the right bar. The total is `(tempo + bars
  + 2 drums + 2 melody + bass) / 7`, so its numbers can't be compared with F1 on
  recordings. `bench ... sampled <dir>` plays NSynth's held-out instruments.
- **Pooling** counts every note once: a mean over recordings would give a long
  recording's many notes the weight of a short one's few.

## 13. A judge's rows and file

Which strikes of the notes model are notes is the accounting's verdict, and a judge
rules in its place, a small network a stage over a strike's row ([Rip
design](rip-design.md) §7). The trainer's [README](../tool/rip-train/README.md#a-judge)
gives the procedure and [audio
§9](https://github.com/lixenwraith/audio/blob/main/doc/audio.md#9-ripping-a-recording)
the row's layout, 544 `float32` in version 1.

`rows <dir> from <list|corpus dir>` writes a directory. Its `rows.json` names the
kind, and a directory holds one: `version`, `row_version`, `row_size` and `notes`,
the checksum of the weights that give the notes, the last eight bytes of
`ripnotes.bin` (§9's trailer) read as one little-endian number and written in hex. A
recording is two files named by the first eight bytes of its audio's SHA-256. Its
`.rows` file is laid out as a loop's frames are (§2):

```
offset  size          field
0       u32           rows, a strike over a half each
4       u32           values a row, 544
8       4 x 544       row 0: the eleven terms the installed score weighs, then the
                      heads around the strike
2184    ...           row 1
```

Its `.json` file labels them, a strike a row in the same order:

- `source` is the recording as the list names it, and `condition` its list line's,
  or a corpus folder's name.
- `given` is the truth's notes and `unanswered` those no strike answers, which no
  verdict can return.
- `strikes` lists a row each: its `frame` and `midi`, then the four below.
- `reported` is the stage: a note the decoder reported, else a strike it dropped.
  `held`, where present, is a held note the accounting keeps unjudged, which a fit
  leaves out of the reported stage.
- `right` is the label: a truth note of the strike's pitch starts within the
  matcher's window of it, each truth note answering one strike, the nearest, a
  reported note before a dropped strike.
- `kept` is the installed verdict.

`judge.py --rows <dir>... --leave-out <condition>... --out ripjudge.bin` fits a
network a stage and writes the file audio's `LoadRipJudge` reads, 1,257,008 bytes at
hidden layers of 256 and 64:

```
offset   size        field
0        4 bytes     magic "VIFJ"
4        u32         version 1
8        u32         row version 1
12       u32         row size 544
16       u32         first hidden layer 256
20       u32         second hidden layer 64
24       u64         the notes weights' checksum, as rows.json names it
32       544         the reported stage: mean
2208     544         spread
4384     256 x 544   w1, an output's inputs together
561440   256         b1
562464   64 x 256    w2
628000   64          b2
628256   64          w3
628512   1           b3
628516               the dropped stage, the same eight again
1257000  u64         FNV-1a 64 of every byte before it
```

A stage's odds for a row are the sigmoid of `w3 · relu(w2 · relu(w1 · (row - mean) /
spread + b1) + b2) + b3`, in `float32`, and a strike stands at a half or more. The
file names the row and the notes weights it was fitted on, so audio refuses it beside
any other: new notes weights mean new rows and a new fit. In the stapes root the file
is `ripjudge.bin`, beside `ripnotes.bin`.
