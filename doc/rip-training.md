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
  `--channel`, `--over` with its heard-hit rule, draws by length, `skip.txt`, a recorded kit's lag
  and its hits that draw no sound (`left.tsv`),
  `make_noise.py`, `--pitchnet` and its `ripnotes.bin`, `--shuffle`, `--extra`,
  `--cell`, `--dip`, `--ring`, `--bend`, `--dilations`, `--sung`, each event's family (`families`,
  the list's `family` cell, each note's program, every part of a file), the family head and
  `--family-holdout`, `--validate`, `tune ... truth`, a moving note's `cents`, `corpus ... glide`,
  the notes model's cents head, `corpus ... labels` and a `from` list's `pitch` column,
  `rows`, `judge.py` and the root's `ripjudge.bin`, the frame model's beat and downbeat
  heads, a `from` list's `beats` and `velocity` columns, `corpus ... gen`'s arranged
  loops, `corpus ... meters`, `corpus ... overlaps`, a loop's level and air, and a label's
  `kind`, `chords`, an event's `part` and `instr`, a rendered loop's `beats` and
  `downbeats`, and the frame model's struck heads.

## 1. The path

1. A corpus is rendered or read: per loop, a `.bands` file and a `.json` file (§2, §3).
2. Each training step draws 8 windows of 512 frames, 5.9 s each (§7).
3. A window's frames may be changed in place, as §6 lists, then become the input
   planes (§4).
4. A window's events and its loop's beats become per-frame targets for the 14 heads,
   and a notes model's cents head (§5).
5. The frame model turns frames into 407 numbers a frame, and the losses train it (§7, §8).
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
- `root`, `scale`: how it was drawn, an overlap's 0. The trainer doesn't read them;
  `truth` reads them as the loop's key where `root` is over 0.
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
- `lag`: in a recording of hits alone, the seconds its sound sits after its file's
  hits, which its hits and beats are moved by; `skip`, in its place, why none could
  be told, the loop being in `skip.txt` (a recording's lag, below). Absent otherwise.
- `left`: in a recording put on its sound, the hits its events leave out as drawing
  none (a hit that draws no sound, below). Absent where none is.
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
  again. `truth` reads them, each to the next one's frame, the last to `frames`.
- `kit`: under `kits`, the kit an arranged loop's drums play, each piece in its
  number's order as `{"genes":[g,...],"band":b}`, its thirteen genes in the manifest's
  `struck_genes` order and the band its hit rises most in (a loop's kit, below).
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
  loop's slot (`rhythm`, `melody`, `fill`, `bass`), an overlap's `single`. `truth` reads
  a note's part as its track, `part` and `instr` together, since a slot's tracks and
  an overlap's instruments share their part's name.
- `instr`: a rendered loop's, what it sounds on: the engine's instrument (`piano`,
  `kick`), or the recorded instrument's name in NSynth (`guitar_acoustic_002`).
- `piece`: a strike of a loop's kit, its piece's number in the loop's `kit`; it names
  no `instr`, the drum it sounded on being no part of what was struck.
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
something struck that no drum head or pitch owns, such as a tom or a crash from E-GMD
or a piece of a loop's kit, which names its `piece`;
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

The same seeds render the same frames, byte for byte, whole or in parts: a corpus is
its line, the build that ran it and the notes its `sampled` names ([the round's
corpora](../tool/rip-train/README.md#the-rounds-corpora)).

`corpus.json` names the layout for the whole folder:

```
{"version":1,"loops":40000,"rate":22050,"hop":256,"layout":1,"drums":["kick","snare","hihat"],
 "waves":["sine","saw","square","string","modal"],
 "families":["piano","guitar","voice","bass","strings","winds","synth","drums","unknown"],"voice_genes":32}
```

Under `kits` it adds `struck_genes`, the names of a piece's thirteen genes in their
order (audio's `StruckGeneNames`). A corpus of placed starts adds `reach`, in frames,
how far a placed start's known start may lie from it (a recording's starts, below):
the trainer refuses one without it and says how to give it one. `loops` counts the files `loop_000` onward; the number grows past three digits as
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
  snare, and other keys as strikes of no line. The list is a draw of the recordings
  on disk, by hours, spread over drummers, kits, styles and tempos, the test split in
  a list of its own ([rip-lab](../tool/rip-lab/README.md#datasetsegmdpy)): a
  recording's frames take 781 MB an hour, so a corpus is a share of the set. The
  list's `velocity` is `none`: a kit's law is not the engine's (a recording's
  velocities, below).

A recording's hits can sit a steady distance off its audio, the same across all of a
performance's kits: `corpus` moves such labels by that lag, and a recording it cannot
place goes in the corpus's `skip.txt`, a list of loop names the trainer leaves out (a
recording's lag, below). A hit of a recording placed that draws no sound is left out
of its label and listed in `left.tsv` (a hit that draws no sound, below).

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

### A loop's kit

`kits` plays an arranged loop's drums on a kit drawn from audio's struck genome
(`kit.go`), from a stream of its own: the loop's notes, voices, tuning and the drums'
rhythm are those the line draws without it, which a test compares, and a corpus
without the word is what it was. A loop whose texture has no drums draws none.

1. **The draw.** Two to ten pieces, evenly; each piece's thirteen genes evenly over
   [0, 1] in `struck_genes`' order, then its level evenly in dB between the quietest
   and the loudest of the bank's drums (`kitLevels`). No piece is refused, whatever
   it sounds like.
2. **Its band.** A piece's band is where its hit rises most: its first take at full
   velocity laid on frame 4 after silence, as audio's `HitFrames` lays a hit, read
   through `BandFrames`, the band whose sharp plane rises most over the frame before
   on frames 3 to 5, the lowest of equals (`strikeBand`). Its test: a tone reads its
   pitch's band within one, and 24 drawn pieces read the same band laid on frame 8.
3. **Roles by band.** The pieces are ordered by band, of equals the first drawn
   first: the lowest is the low pulse and plays the kick's line, the highest the time
   and the hats' line, the one half the kit's size up the backbeat and the snare's
   line; those between the lowest and the backbeat are the fill, the highest of them
   first, and those between the backbeat and the highest the accents. Of two pieces
   the higher is backbeat and time.
4. **What a kit adds.** The accents strike the bar lines the kick strikes, at its
   velocity there: every accent on the first, then one in turn. The fill takes the
   last pulse into the bar line after the last whole bar the kick strikes: the
   snare's fill where it drew one, else one written as the snare's, dealt from the
   highest fill piece down in runs as even as they fall. These strikes are played
   from the kit's stream after every other part. A piece left with none strikes
   with the drums' first; a loop whose drums strike nothing in its span keeps no kit.
5. **Passes of four.** The engine plays four drums at once, so piece n sounds on drum
   n mod 4 (`SetKit`), and a kit of more than four renders in passes of four from one
   timeline and seed, the first with the rest of the loop and each other with the
   drums alone; the room, the mix and the level take their sum once. The arranger
   draws everything before the passes, so every pass plays the same loop: its test
   renders kits of four and five in one pass a piece and whole, the same to 1e-9 of
   their peak.

A strike is labelled as a strike of no line in the `drums` part, with its frame, its
velocity and its `piece`; the loop's `kit` lists each piece's genes and band once.
A band says where a piece sits, not which drum it is: any grouping of the 315 bands
into places can be read from the labels later. `tool/rip-lab/audit` reads a corpus
of kits ([its README](../tool/rip-lab/audit/README.md)). The trainer reads no piece
and no gene: a kit's strike is struck as any, and a loop that holds a kit asks the
drums head nothing (§5).

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
- `kits` is a word of its own and wants `gen`: a kit is what the drums sound on, and
  it changes only that, the strikes it adds and their labels.
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
  velocity word, its starts, its MIDI file, its pitch track, its beats and its
  strings. The recording is known by the `source` of the label on disk, and its WAV
  is read only where its velocity word is `measured` or its strings cell names a
  file, so it may otherwise be gone; a label that does not read stops the run, its
  recording being unknown then. A `measured` row whose WAV is gone or does not read
  keeps its label, each of its notes at `vel` -1, and the run's last line counts
  them by reason (a recording's velocities, below).

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
  MIDI velocity. On a kit the set states each hit's strength in the measure the
  engine's kits use: a hit read at n/127 sounds its take times that (audio's
  `DrumVoice`), its amplitude over the hit's at velocity one.
- `none`: they say nothing, as a lab's one velocity for every note does (GuitarSet's,
  VocalSet's), and each note and hit is labelled at `vel` -1, which asks the velocity
  head nothing (§5).
- `measured`: they say nothing, and the recording says it instead: each note is
  labelled at the velocity it is heard at (below), -1 where that cannot be told,
  and each hit at -1.

Any other word is refused, naming those the table has. Without `measured` no
recording is opened for its velocities, and a list's labels are as they were
without the word, byte for byte.

E-GMD's list says `none`: a module maps a velocity to a level and a layer by its own
curve, so n/127 is not known to be a hit's amplitude over its loudest. Whether it
states each hit's strength, an empty cell, is the lab's to set once the trainer has
read back whether a kit's stated strengths follow the levels its hits sound at
([todo](todo.md)). No head asks a hit's `vel` yet (§5).

A measured velocity is the one at which the engine plays a note as loud as the
recording played it. The engine's law is straight (`vel`, above), so it is the
note's level over its recording's unit, the level a note of velocity one has there,
and at most 1 (`velocity.go`):

- **The level** is a sampled note's: the RMS of its loudest window of 50 ms, hops of
  half a window (`windows`, the one function the sampler and the measure read a
  level by), over its held span, from its start to its end or 3 s on (`heldSpan`).
  It is read from the note's own partials (`hearLevels`), so that the notes still
  ringing are kept out; a note another note sounds with is given none (the bound,
  below):
  - Each window is fitted under a Hann window, by least squares, to the note's
    partials at the pitch its track gives, or at its MIDI pitch where no track
    places it: k f0 √(1 + b k²) for k up to 80 and up to 20 kHz, b the note's
    stiffness, of a grid from none to a piano's top octave's the one whose partials
    hold most of its first window (`stiffnesses`), read to 60 dB under its
    strongest there. Fitted with them is every partial of the notes still ringing
    in the window (below) that lies within three bins of the window (60 Hz) of one
    of them or of a point midway between two. Partials within those three bins of
    the next form a chain, solved whole, every pair in its Gram: a partial that
    stops inside the window spreads wider than three bins, and with its far pairs
    left out the matrix is no Gram and its factor fails.
  - A note rings past its label's end for as long as its sound does, at the pitch
    it was let go at (`heardRing`), with no time to cap it: a voice's is gone in a
    window, a string's lasts seconds. Each window from its end on, hops of half a
    window, fits its partials with those of the notes labelled there: they are
    found where those told apart hold more than is left unfitted around them. A
    partial not told apart is in doubt and could hold the most a told one under it
    kept of what it could hold in the note's last labelled window, a partial dying
    no faster than those over it, or, none under it telling, all the window holds
    there. A ring is let go from a measured note's windows at the first window
    where its partials are not found and what those within three bins of the
    note's could hold, added to them in phase, cannot move the note's level past
    the bound beside what the note's own bound alone already spans: (10^((B −
    u)/20) − 1)² of its level alone, u that bound's half-width. A window where a
    ring is found and its partials told apart could, a fit of two notes misleading
    by a dB within tight bounds, bounds the note's level from above only, by all
    it holds at its partials and what no partial explains; a partial in doubt
    already gives the low bound nothing where it inflates one of the note's. A
    ring called gone gives a wrong velocity and a gone note called ringing costs
    one, so a note in doubt rings.
  - A partial is the note's own where the partials fitted with it, its own note's
    among them, widen its estimate's variance twice at most (`measuredApart`), its
    chain's inverse read on the diagonal, and its energy bounds the level from below
    and above by what is left unfitted either side of it. A partial past that is not
    told apart: it gives the low bound nothing and the high bound all the window
    holds at its frequency, so a shared partial widens the bounds and never moves
    the level by a guess at its split.
  - What the window holds that no fitted partial does, a pluck, breath, bow or a
    sound off the note's pitch, is all in the level the sampler lays a note at, and
    may be the note's: the high bound holds it.
  - The loudest window's level lies between the greatest low bound of any window
    and the greatest high bound. The level is their middle, their mean squares'
    geometric mean, and it is known within half their width, 5 log10(hi / lo) dB.
- **The bound.** A note known only more widely than `measuredBound`, 0.8 dB, is
  given -1. The bound is fitted on recorded notes whose levels are known
  (`TestAMeasuredLevelIsKnownWithinItsFittedBound`, with `NSYNTH` naming the folder
  of NSynth's splits): the guitar, vocal, keyboard and string families, notes laid
  by the sampler one after another and in overlaps (§3), from two to eight sounding
  at once, at velocities 0 to 30 dB down, with no room and through `degrade`, each
  note's truth its level laid alone and carried through the same room (`room`).
  One after another is in lines of twenty, each note on an instrument drawn anew at
  a unison, step or leap from the note before, none or up to 2 s after it, short
  gaps likelier, a soft note 20 to 30 dB under a loud one among them; the note
  before is released into its sample's release, held on past its label to its
  file's end, as a string a label lets go too soon, or cut at its end by 60 ms, as
  a damped string or a voice moving on. A seeded half of the training split's
  instruments fits it, with three times a readback's lines, the least bound in
  tenths of a dB within which nine in ten of the notes known within it lie in every
  group of twenty known or more (a family, or the four, alone or with others, in a
  room), sure to one in twenty: the order statistic that bounds a group's
  ninetieth percentile from above at odds of 19 to 1, a binomial's, lies within
  it. Fitted without that, the bound held on the fit at 0.7 dB and missed for
  guitar on the instruments left out and the test split. The other half and the
  test split read it back, a group failing where it shows at 19 to 1 that more
  than a tenth of its known notes miss the bound. Notes sounding with others held
  a bound fitted on the fit's instruments and missed it on the others, nine in ten
  within 1.84 dB of 1.2 left out and 1.44 on the test split, so they are given
  none. A ring is let go by what the bound leaves, so the bound is fitted until the
  fit gives it back. With the fit's lines as many as a readback's, 0.9 missed by
  0.002 dB and 1.0 fitted 1.5, a group's two or three worst notes of some 70 known
  deciding it; tripled after that near miss, the fit gives 0.8 at 0.9, and 0.8
  back. At 0.8 dB, the share of notes known and nine in ten of their errors, the
  fit's groups of 450 notes or so alone, the readbacks' 150, about 350 with others:

  ```
                           fit            left out       test
  guitar    alone, no room   46%  0.58 dB   44%  0.66 dB   46%  0.62 dB
            alone, degrade   38%  0.61      35%  0.66      40%  0.63
  keyboard  alone, no room   55%  0.51      54%  0.57      52%  0.46
            alone, degrade   48%  0.49      50%  0.63      47%  0.76
  string    alone, no room   49%  0.49      49%  0.52      46%  0.42
            alone, degrade   44%  0.49      45%  0.54      36%  0.46
  vocal     alone, no room   56%  0.55      46%  0.49      32%  0.62
            alone, degrade   49%  0.51      36%  0.41      26%  0.40
  the four  alone, no room   52%  0.53      48%  0.55      44%  0.54
            alone, degrade   45%  0.54      42%  0.63      37%  0.60
  with others, every group    0%             0%             0%
  ```

  By what the note before did, the four on the fit, no room and through
  `degrade`: after none 64% and 63% known, nine in ten within 0.51 and 0.63 dB;
  after a release 59% and 51%, 0.52 and 0.50; after one held on past its label
  31% and 29%, 0.61 and 0.63; after one cut 61% and 50%, 0.53 and 0.51; 20 dB
  or more under it 44% and 31%, 0.56 and 0.59. A velocity is measured where a
  note sounds alone, as in a solo line. The test also checks the bound on the
  engine's voices' overlaps, the bank's and the plucked and vocal focus's.
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

A level, a bound or a unit that is not finite gives no velocity. A `measured` run
counts what it gave after what it heard, and the recordings it heard nothing in by
reason (`unmeasuredWhy`): `wrote the labels of 2 recording(s) in out again from
list.csv and heard 1, 4 note(s) given the velocity measured and 4 none, 4 of those
none in recording(s) not measured: 1 gone from disk`. A recording gone from disk,
whose sound does not read or with a level not finite keeps its label, each of its
notes at -1, as under `none`; but for one gone, each is named on a line of its own,
and the run goes on.

### A recording's strings

A `from` list's `strings` column names a recording's by-string file: a WAV of a
channel a string, each a pickup under its string with the other strings' bleed
taken out, as GuitarSet's debleeded hexaphonic recordings are, the channels in the
order the recording's pitch track names its strings (GuitarSet's `data_source`). An
empty cell names none, and that recording is labelled as without the column, byte
for byte. A named file is read with the recording and its pitch track, a JAMS file
whose `note_midi` annotations say which string plays each note (`hearPickup`):

- **A note's string** is the one whose annotation holds a note of its key starting
  within a frame of it, the nearest, two notes alike taking two strings
  (`stringsOf`). A note no string holds keeps its label.
- **A note ends where its string stops sounding it** (`heardEnds`): on its
  string's channel, from its start by hops of a quarter window, at the middle of
  the first window where its partials told apart no longer hold more than what is
  left unfitted around them, the test a ring is found by (above); or at its
  string's next note, which stops it, where it still sounds there. Past its label's
  end it is read at the pitch it was let go at. A note its first window does not
  find keeps its end.
- **Under `measured`** a note is measured on the microphone where it sounds alone
  by the ends its strings give, as a note alone is (above), those ends saying that
  nothing rings into it; a note sounding with another gets none. A note's level on
  its string's channel is not put on the microphone's scale: on GuitarSet's notes
  alone the microphone's level over the channel's strays note to note further than
  the bound (below).
- A recording whose by-string file is gone, does not read, holds fewer channels
  than its strings or is at another rate than the recording keeps its label, its
  notes at -1 under `measured`, and the run's last line counts it by reason
  (`unmeasuredWhy`) after the notes it ended: `the ends of 14 note(s) read on their
  strings`.

Which channel is which string is shown on GuitarSet
(`TestAByStringChannelIsItsString`, with `GUITARSET` naming the set's folder): of
the notes no other note's label comes within a window of, each string's hold the
most at their partials on its own channel, 273 of 289 on the lowest string, 466 of
500, 1,060 of 1,064, 1,409 of 1,409, 1,108 of 1,114 and 732 of 737 on the highest;
what they hold on each other channel is the bleed the truth below lays
(`stringBleed`).

The ends are read back on six lines of NSynth's guitars that hold what GuitarSet's
channels hold (`TestAStringsNoteIsReadFromItsChannel`, with `NSYNTH`): solo lines,
58 notes in a hundred on the string of the one before as in the set's solos, and
strummed chords of two to six spread over up to 60 ms, each note on an instrument
drawn anew 0 to 30 dB down, released, cut over `sampledFade`, held on past its
label, or struck again on a string still ringing. Each line is on its own channel
at a gain drawn 6 dB either way, the set's strings' channels against the
microphone spanning a median 8.7 dB a recording; each note strays from it by a
Laplace of 2.25 dB, as the set's notes alone stray from their string's; the other
lines bleed in at levels drawn from the set's (`stringBleed`), and a white floor
lies under each channel as far as the set's quietest windows lie under its
loudest (`stringFloor`). The microphone is the lines summed, with no room and
through `room`. A note's end laid is where its own sound at its partials, alone,
stops holding more than the rest of its channel holds there, or its string's next
note; one under that rest from its first window is buried, its label's end its
end. A seeded half of the training split's guitars fits each kind's bound
(`stringEndBound`, the least in frames within which nine in ten of the ends lie,
sure to one in twenty), with three times a readback's recordings; the other half
and the test split read it back, none past it at 19 to 1. The ends in seconds, the
median and nine in ten, and a note alone on the microphone, the share known and
nine in ten of its errors in dB:

```
                        fit             left out        test         bound
released             0.037  0.575    0.050  0.575    0.012  0.065    0.778
cut                  0.025  0.267    0.025  0.187    0.012  0.086    0.441
held on              0.037  0.987    0.037  0.825    0.012  0.475    1.370
struck again         0      0        0      0        0      0        0
buried               0.726  1.880    0.420  1.736    0.118  0.207    2.810
alone, no room       55%    0.48     55%    0.27     67%    0.50
alone, degrade       52%    0.42     55%    0.30     67%    0.70
```

A note sounding with another is known on the microphone none of the time. Its
level on its channel put on the microphone's by its line's gain, fitted on half
the notes alone both know, lies nine in ten within 5.7 to 13 dB of its level laid
on the others, as GuitarSet's own notes alone stray (the todo).

### A recording's lag

A recording whose file holds hits alone, as E-GMD's, is labelled where its drums
sound. `corpus` reads how far the sound sits after the file's hits from the frames
it writes, and under `labels` from those on disk, so no recording is opened again:

- The rise is each frame's sharp plane over the frame before, the positive part
  summed over its bands (`rise`), which is what `datacheck.py` reads a label's lag by.
- A strike is a hit or a strike of no line, one less than a frame after another
  being the same rise. A lag is looked for within half the usual gap between two
  strikes, their median, either way (`lagReach`): further, hits moved meet the rises
  of the hits beside them.
- The lag is where the strikes, moved by it, meet the most rise, read by half
  frames and then by eighths of a frame, 1.45 ms. The engine's own drums rise 5.8 ms
  before their labels (`lagRise`), a frame being centred on its time, and that is
  taken off: a recorded hit then sits against its sound as a rendered one does.
- The hits and the beats are moved by the lag, those moved out of the recording
  dropped, and the label states it: `"lag":0.01306`.

A recording is left out where its hits cannot be put on its sound, by one of three
reasons (`lagWhy`), read in this order:

- `hits that draw no onset`: fewer than 0.40 of its strikes (`lagDrawn`) have the
  largest rise within their reach inside a frame of where the lag puts them. A
  silent file, or one that holds another recording, is such.
- `no lag that stands clear`: another peak within the reach meets more than 0.97
  of what the best does (`lagClear`), or the score still rises at the reach.
- `a lag that drifts`: the median onset of the later half of its strikes lies more
  than 10.2 ms from the earlier half's (`lagBound`). A drift that leaves few
  strikes within a frame of one lag reads as hits that draw no onset.

Such a recording keeps its frames and its label, the hits as its file has them and
`"skip":"a lag that drifts"` in place of the lag; the run names it (`a.wav: left out
for a lag that drifts`), adds its loop to `skip.txt`, which holds a name until it is
taken out by hand, and goes on. Its last line counts both: `the hits of 2
recording(s) put on their sound and 1 left out in skip.txt: 1 for hits that draw no
onset`.

The five constants are fitted on drums whose answer is known
(`TestARecordedKitsHitsAreMovedOntoTheirSound`): four bars of the arranger's drums
on a kit drawn for each, the bank's drums under an eq of five bands 10 dB either
way, a pitch half an octave and a length an octave either way, each recording at a
level drawn as a loop's ([a loop's level](#a-loops-level)), a recorded kit's own and
not the bank's, with no room and through `degrade`, the hits moved by none, by a
steady lag up to two frames short of the reach either way, by a drift of three to
six frames over the recording, or replaced by another recording's, and the same
hits over silence.

- The reach is the widest share of the usual gap, doubling from a quarter, at
  which every recording of a draw of 24 whose hits were moved by none or a steady
  lag is read within a frame of their median: a half. At the whole gap one is read
  120.9 ms off.
- At a half, over those 96 readings: the median of the lag read less the lag laid
  is `lagRise`; the furthest from it, 9.1 ms, to the nearest step and one more is
  `lagBound`; `lagClear` is halfway from the most another lag meets, 0.943, to a
  tie; `lagDrawn` halfway from the least share of strikes that draw an onset,
  0.516, to the most another's hits or silence draw, 0.278. A draw holds only
  where the shares part and no recording's halves lie further apart than the
  bound: theirs lie 5.7 ms apart at most.
- A level moves only the takes through `degrade`, whose soft clip it drives: laid
  at one level throughout, the rise reads -3 steps at -32 and -26 dBFS and -4 at
  -20 and -14, the bound 6 or 7 and the other three as drawn; at -32 one reading
  lies 10.5 ms from the drawn rise. Drawn, the readings' median lies at -3.51
  steps, so a drum's norm moved alone by 2 dB moves the rise or the bound a step.
- On a second draw of 12, each of 48 lags laid up to 133 ms is found within 6.4 ms,
  none left out; of the 12 that drift 7 are left out for a lag that drifts and 5 for
  hits that draw no onset, and all 12 silent and 12 of another's hits for hits that
  draw no onset.

What the measure cannot tell: a lag past half the usual gap, where a steady pattern's
hits meet their neighbours' rises as well as their own did, is read as a nearer one,
or left out where the score still rises at the reach; and the constants are the
engine's kits', read back on no recorded kit yet ([todo](todo.md)).

### A hit that draws no sound

A recording whose hits were put on its sound leaves out of its label each hit that
draws none, hit by hit, the recording kept. A hit's onset reads the largest rise
within a frame of where the lag puts it (`onsetRise`), as the lag's measure reads
one; under 0.001 in the frames' units (`hitBound`) it draws none. A hit on the first
frame, whose rise is unread, stays. The rule names no key, line or strength: a hit
close after another on its line is judged as any is, so a buzz's bounces stay while
each sounds, and no gap merges hits.

The trainer asks what hears a strike nothing there (§5): a hit that draws no sound by this
measure an ear may still hear, so the label says neither a strike nor silence.
Each hit left out is listed in its label's `left`, the frame it would sit at, its
General MIDI key (audio's `RipHit.Key`), its line (-1 a strike of no line), its
file's velocity over 127 whatever the list says, and why (`hitWhy`):
`{"frame":140,"key":44,"drum":2,"vel":0.3543307086614173,"why":"no sound"}`. The
corpus's `left.tsv` holds a tab-separated row for each under the header `loop
source sec frame key drum vel why rise`: its file's seconds, and the rise its onset
read, for a measure to come; `labels` writes the rows of the loops it labels again,
and a corpus that never left a hit out has no table. The run's last line counts them
by why: `..., 4 of their hit(s) left out of the labels in left.tsv: 4 for no sound`.

The bound is fitted on drums whose strengths are known
(`TestAHitThatDrawsNoSoundIsToldByItsRise`): the lag's draw of the arranger's drums
on drawn kits, a fifth of each recording's hits laid silent and the rest at their
velocity under a gain drawn evenly from none to 40 dB down. Over 24 recordings with
no room, 23 of them put on their sound, it is halfway in decades, to the nearest
tenth, between the most nine in ten silent hits read, 8.5e-5, and the least nine in
ten hits at the engine's strengths (`softest` up) read, 0.0105, each sure to one in
twenty (`sure`); a silent hit within two frames of a sounding one is left out of the
fit. Read back on 12 more with no room and through `degrade`, every take put on its
sound, the share each keeps:

```
strength laid           no room             degrade
0 to -10 dB          42 kept 100.0%     42 kept 100.0%
-10 to -20          103 kept 100.0%    103 kept 100.0%
-20 to -30          122 kept 100.0%    122 kept 100.0%
-30 to -40          129 kept  81.4%    129 kept  86.8%
-40 to -50           95 kept  56.8%     95 kept  64.2%
-50 to -60            9 kept  11.1%      9 kept  33.3%
silent, alone        99 kept   1.0%     99 kept  15.2%
silent, under a hit  53 kept  92.5%     53 kept  92.5%
```

What it cannot tell: a silent hit within two frames of a sounding one reads that
one's rise and stays; a room's tails lift silence's rise; a hit laid 40 dB under
velocity one is about as often left as kept; and the bound is the engine's kits',
read back on no recorded kit yet ([todo](todo.md)).

### A recording's starts

A `from` list's `starts` column says how its notes' starts were found, a row a word
(`startsWords` in `tune.go`):

- An empty cell, or no column: each is where its note is heard struck, and the
  label is as without the column, byte for byte.
- `placed`: they are placed on the recording by an alignment, as a take's notes
  that its exercise sheet writes and a procedure fitted on known sound lays on the
  take, and such a start can miss its strike by frames. A target a frame wide
  there would teach the alignment's miss, so the label says `placed` (§3): its
  notes ask the onset head, and the strike head at their pitch, for a rise
  anywhere within the corpus's `reach` of each start (§5). Their frames,
  velocities and cents are asked as any note's, and the release head at each end,
  which the same alignment places.

Any other word is refused.

The reach is measured where the placing was fitted, on known sound:
`tool/rip-lab/labels/placefit.py <truth root> reach <corpus dir>...` places the
takes of its three engine-shapes draws under `place.py`'s bounds as a label is, and
writes into each `corpus.json` the 0.95 quantile of how far a start of the right
takes that pass sits from its known start, a high share and not the worst, in
frames rounded up. A `corpus ... from` run writes `corpus.json` afresh, so the line
runs again after it.

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

- `onset`: within one frame of any event, drum or note, that is `t-1 <= frame < t+2`,
  but a placed start (§3), which marks `stretch` instead; a strike marks it wherever it is.
- `stretch`: a placed start's reach either side of it and its mark's frame past
  that, `t-reach-1 <= frame < t+reach+2`, where the onset's loss asks a rise
  somewhere (§8); stretches that touch are one, asking one rise at least. 2 where
  the window holds it whole, 1 where the window's first or last frame cuts it,
  whose start may lie past the edge, so it asks nothing.
- `starts`: a placed start's mark, asked nothing: the onset head's peaks are read
  against it within the reach (§8).
- `drums[d]`: the same as the onset, for that drum's hits; asked nothing on a loop whose
  label holds a kit drawn from the genome (`kit`, its frames' mask), whose pieces name
  no line.
- `struck`: the same, for anything struck of no pitch: a hit of a line, a strike of no
  line and a kit's strike, every event at `midi` 0 (`Label.struck`). The frame model's
  alone, asked at every frame of every window.
- `struckvel`: the `vel` of what is struck, on its frames (`struckask`), of two strikes
  there the louder's; nothing where the label gives none, as E-GMD's gives none.
- `left`: the frames a hit its label leaves out covers (§3), the frame it would sit at and
  one either side, as its corpus's `left.tsv` lists it: an ear may hear it, so the onset,
  the drums, `struck` and `struckvel` are asked nothing there, neither a strike nor silence.
- `pitch[p]`: from a note's `frame` until its `end`, at least one frame.
- `voicing`: any pitch on.
- `strike[p]`: within one frame of a note's start, but a placed start (§3), which marks
  `pitchstretch[p]` instead.
- `pitchstretch[p]`, `pitchstarts[p]`: a placed start's `stretch` and its mark, in the
  row of the pitch it is struck at, where the strike's loss asks a rise somewhere and
  its peaks are read against the mark (§8); a pitch's stretches that touch are one,
  and two pitches' are two.
- `placed`: a loop's frames whose starts are placed, which the strikes' F1 leaves out
  (§8).
- `release[p]`: within one frame of a note's `end`, unless that's past the loop.
- `velocity[p]`: the note's `vel` on its strike frames, a placed note's about its
  placed start, scored only there, and nowhere for a `vel` of -1 (§3).
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
a loop where nothing is struck, then a recording from a corpus of drums alone, placed
at random. Its level is -12 to +6 dB of its mean power
against the loop's, and uncorrelated sounds add in power, so `e = e_loop + gain x e_drums`.
A strike, a line's hit or one of no line, joins the onset and struck targets, and a hit
its drum's, only where it can be heard: what it adds, over its best ten bands of the
sharp plane, must top everything else sounding (`stands_out`), judged after the channel,
a strike in a placed start's stretch as anywhere, the stretches kept. How hard goes
unasked, and a hit the drums' label leaves out asks nothing where it falls (`left`).
The window is asked no beat and no bar line: the drums laid under it keep
their own.

**`--mix dir:share`** draws a share of the windows from another corpus as it stands.
`--extra dir:share,...` does the same for any number of corpora.

**`--shuffle share`** splices a share of the windows from chunks of a second, each two clips
from random places summed, the second at -6 to +6 dB of the first's power. Notes keep their
labels, a placed start its stretches where they fall in its chunk, cut by the chunk's edges as
by a window's, a strike counts where it stands out of the other clip, as a laid one does, a
hit either clip's label leaves out asks nothing where it falls, a
chunk of a kit's loop asks no drums, and how hard, wave, voice,
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
The beat, the bar line and their mask are taken the same way and sway with the hits, and so
are a placed start's stretches, the onset's and the strike's, cut again by the window's edges.

**`--strike-weight w`** is a strike's weight against a frame without one, 20 by default.
**`--again-weight a`** weighs a strike of a pitch still sounding a times more, a placed start's
rise too (§8).

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
norm, output   B x 512 x 407                 raw scores
```

Attention has 8 heads and looks 64 frames (0.74 s) either side, in every block. There
are no position embeddings: each head subtracts a slope of `2^(-8(h+1)/8)` per frame of
distance from its scores (ALiBi), so a window of any length works.

The 3,919,127 weights: 592 + 2,320 + 4,640 in the convolutions, 647,424 in the input
layer, 789,760 in each block, 512 in the final norm, 104,599 in the output layer, 2,313
of them the family head's, 514 the beat's and the downbeat's and 514 the struck heads'.
A model of the two dB planes has 288 fewer, all in `convs.0.weight`.

The 407 outputs, in the order the file lists the heads:

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
405         1   struck     sigmoid: something of no pitch struck within a frame, of a line or of none
406         1   struckvel  sigmoid: how hard, a velocity
```

Every model's pitches are the piano's 88 keys, MIDI 21 to 108. A file has every head;
only a notes model's may lack the cents head, as a probe's does (§9).

`--pitchnet` builds another model on the same heads but `family`, `beat`, `downbeat`, `struck` and
`struckvel`, the frame model's to hear:
convolutions only, the same weights at every pitch, so 172,181 weights in all, 66 of them the
cents head's and 1,485 those of what plays each pitch. Its file is `ripnotes.bin` (§9).
`--dilations 3,9,27` adds a residual 3x3 over the cells for each, its frames that far apart:
each note head then hears half a second either way instead of 80 ms, 199,925 weights, about
1.5 times the compute. `--init` grows them onto a model of none, or of the first of them, every
weight it had carried whole and the optimizer fresh, as a head grows. Each adds its cells back
before its GELU, so one of zero weights would hand on gelu(c), not c: a grown one starts with
its kernel's centre reading each channel's own cell at a scale, a shift its bias, every other
tap zero, the scales and shifts of all the grown ones fitted channel by channel to the start's
cells over the first loops' windows, in the mean square (`start_long`), and the run's line
gives the miss left. The frame's heads hear the wide layer, not the cells, so they say what
the start's did. Its per-pitch heads are a table, `CELL_HEADS` in `ripmodels.py`, a row a
head with the values it reads at a pitch; the cents head reads two, the offset in semitones
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
AdamW at weight decay 0.01. The learning rate runs from `--lr` (3e-4) to zero on a
cosine over the steps, or follows the readings in a pass that carries on
([below](#a-pass-that-carries-on)). The batches are built beside the device by
`--builders` processes forked from the run, one by default, each at most two batches
ahead, handed over in index order: a batch is Python holding the interpreter's lock,
so a second thread builds less, not more. Batch k draws every window from a stream of
its own, `random.Random(seed + (shard << 32) + (k << 64))` (`stream`), so it is the
same tensors whatever builds it, a run resumed at step k draws the batches the
unbroken run drew from k on, and two shards draw different batches.

```
onset, drums, pitch, voicing, strike, release, struck: binary cross-entropy with a weight
    on the positive class of 8, 8, 12, 2, 20, 20, 8, the onset's, the drums' and struck's on
    frames no hit left out covers, the drums' of no kit, the rest on every frame; the onset's
    and the strike's ask a stretch the window holds whole one term, its highest logit against
    1, the strike's at its pitch, and no cell inside a stretch but another event's mark
velocity: squared error, on strike frames with a known velocity
struckvel: squared error, on what is struck's frames where its velocity is known
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
           + 0.25 (0.5 pitchfamily + 0.5 pitchwave + 2 pitchvoice) + struck + 2 struckvel
decoders = onset + drums + pitch + 0.5 voicing + strike + release + beat + downbeat + struck
           + 2 struckvel
```

A placed start's stretch asks the onset head to rise somewhere in it, and the strike
head at the pitch it is struck at, and is indifferent to where: a rise at any of its
frames costs the same, and none of its frames is asked silence, so neither head can
learn where in the reach the alignment put the start; stretches that touch ask one
rise between them, the strike's a pitch at a time. Outside every stretch a placed
loop's frames are asked what any loop's are. A head's loss is the mean over the
frames of their cells' (`rise_loss`): a stretch weighs one cell, at the weight its
start's mark has most, `--again-weight` times more where its pitch still sounds just
before it; a cell inside one is asked nothing but another event's mark, as a spliced
clip's strike; a batch without a stretch asks every cell as the other heads are. A
known start past its stretch, under one in twenty where the reach was measured, is
asked silence where it is, as any frame outside one.

A loop whose label holds a kit drawn from the genome asks the drums head nothing: its
pieces name no line, and the head would be taught silence under them.
A strike in a loop of placed starts marks the onset where it is, as a drum hit does:
it is heard, not placed.

The struck heads are the frame model's: that something of no pitch was struck, of a line
or of none, and how hard. The first is asked at every frame of every window, a recording's
of guitar or voice among them, a strike weighing 8 as a hit does in the drums head; at that
weight its score sits at `STRUCK_BAR`, 8/9, where a strike is as likely as not, which is
where audio's bar on it starts. Audio reads both (§10), so both join `decoders` and what a
pass that carries on is read by.

The beat and the downbeat are the onset head's kind of loss with one difference: no
weight. Audio reads the two by their level, as posteriors: the tempo from how the beat
head repeats, cut to the finest beat it marks between, and the beats tracked on it, a bar line where the downbeat head summed by
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
are (`peaks`). Over a corpus that lists no beat, such as a `--validate` corpus whose
list has no `beats` column, the line says `no beat or bar line asked` and gives
neither head's loss: a mean over no frame prints 0.000, which reads as a head never
wrong, and adds nothing to `decoders` either way.

Its lines then give `struck's F1`: the struck head's peaks at `STRUCK_BAR` or over against
what its corpus strikes within four frames, counted so, a peak where a hit was left out none;
or over a corpus where nothing is struck, as GuitarSet's and VocalSet's, `nothing struck,
struck's peaks N a minute`, which a head that hears no strike there keeps near 0. A corpus
that says nowhere how hard gives no `struckvel` loss.

Over loops of placed starts, held out or `--validate`'s, either model's lines then
give `placed starts' F1 ... within R frames, their strikes' ...`: the onset head's
peaks at a half or over (`ONSET_BAR`), a peak right when a placed start lies within
the corpus's reach R of it, counted as the strikes' are (`peaks`), then the strike
head's peaks over the strikes' bars against the same starts at their pitch, within R
too. No loss enters either, so they read any weights alike, whatever loss trained
them.

A notes model's lines then say what plays each note, read as audio reads it (§10): a
note's family, wave and genes are its cells' say over the frames it sounds, each frame
weighed by the pitch head's salience there, over the cells among the frame's
`VOICE_CELLS` (12) most salient, which are the cells audio keeps. The line gives the
share of the notes with a family asked whose highest share is theirs, by family, then
that share over the notes another note sounds over, the share of the engine's notes
whose wave is right, and the genes' miss, an RMS over the 32.

It ends on the names against their prior, each family's share of the cells the held-out
loops ask a name of (`shares`): the names' loss as a mean over the cells asked, where
`pitchfamily` among the line's parts is a mean over windows, one that asks no name
counted as zero; beside it the loss of those shares alone on the same cells, the held-out
line's their entropy, at or over which a head has learned no sound; and the share of the
notes named right with each name's say taken over its share, by family, which shows a
name the commonest outvotes.

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
over the frames not placed (§5), since its loss holds an onset head that no
decoder reads from it. The wave, voice, velocity and family heads don't choose it,
because they memorize the corpus. A positive
weight w shifts a head's scale: it scores a posterior p as wp / (wp + 1 - p), which is
part of why thresholds belong to the weights (§10); the beat and the downbeat carry
none and are read with no threshold.

The training lines name each head's loss, a model with the family, beat and downbeat
heads printing theirs last. The wave head is taught on the engine's voices alone, so on
a real instrument its guess means little, which is why a rip can play a piano on a saw.

The checkpoint, a PyTorch `.pt`, holds `model`, `opt`, `step`, `cfg` (the shape, which a
resumed run must match), `best` (the best held-out score, its step and its weights) and
`validated` (the best validation score, its step, its weights and the held-out score there),
and in a pass that carries on `rate` (below), with the weights and optimizer's state it
may go back to, up to twice the file; no stream, since the step is all a batch's draws
need.

### A pass that carries on

`--carry` makes a pass start where the last one ended: from `--init`'s checkpoint it
takes the weights, the optimizer's state and the rate's, and it reads its held-out loops
before a step, the start every later reading is judged against. A checkpoint of weights
alone, or one whose heads or long convolutions grow, starts one with the optimizer fresh; a
run resumed reads the rate back whole. The step written is picked as above, and the checkpoint
the next pass starts from is the last step's. Without the word a pass runs the cosine above,
its lines and its pick as they were, bit for bit.

Such a pass is read by the heads audio reads from its model in a pair (`READ`), each at
its weight in `decoders`: a notes model's pitch, strike and release, a frame model's
onset, drums, beat, downbeat, struck and struckvel; not yet a notes model's onset, which audio reads beside
its strikes (todo, "Tune the rip"). Their sum is its lines' `decoders`, picks its step
where the decoders' loss does, and is what the rate (`Rate`) follows over the held-out
windows; a line before its start names them.

- **Fine**: the rate between raises; `--lr` for a model no pass trained, half of it
  for a checkpoint that carries no rate, else the checkpoint's own, never over `--lr`.
  A reading under the reading kept past its noise is a gain, kept with its weights and
  the optimizer's state. One over it past its noise falls back: training goes on from
  the kept weights and state, and fine falls.
- **Raised**: readings without a gain, as many as a stall counts (below), raise the
  rate to `--lr`, down half a cosine to fine over `STALL` readings, and the raise is
  judged at its last. Better than the reading kept, it paid: its end is kept as a gain,
  and fine rises. Over where it began past the noise between the two, or a head past
  its own, it is undone: training goes on from the weights and state before it, and
  fine falls. Else it is kept as no gain, fine as it was, and its end is the reading
  kept from then on, so the readings after it are told against the weights they go on
  from. A stall at fine as high as `--lr` falls fine instead, and one with fewer than
  `STALL` readings left in the pass raises nothing.
- **Wide** (`--wide n`): after `n` passes in a row without a gain kept, `STALL` by
  default and counted in the checkpoint, or this pass with `--wide 0`, a pass runs from
  `--lr` down half a cosine to fine over all its steps and is judged as a raise is,
  against its start: it ends on its end, a gain where that is better, and on the
  start's weights and state only where its end is over them past the noise. It starts
  the count again.
- **A head** asked on the held-out windows, with rows of its own in the layer that
  outputs it (`rows`), is stalled when its own loss goes `STALL` readings without a gain
  past its own noise while the decoders' loss has not stalled, fine is under `--lr` and
  `STALL` readings are left. Its rows alone are raised: AdamW's step is the rate times
  what the rate leaves alone, so after each step their move is scaled by a raise's rate
  over fine (`scale`), and every other weight moves at fine. At the raise's end they and
  their optimizer's state go back to before it where the head's loss ended over where
  it began past its noise; kept without a gain, they take the head's reading kept with
  them.
- **Better** (`better`): the sum under the other reading's past its noise, and none of
  its heads over its own past that head's noise, so no gain in the sum is bought with a
  head falling. A raise's end and a wide pass's are judged so; a gain at fine by the
  sum alone, as a run picks its step, since a head at its floor drifts past a noise
  near zero, as the drums do over loops without them.
- **The lines**: each held-out line after the start says the rate since the last
  reading, `fine`, `raised` or `wide`, the difference on the reading kept and its
  noise, and what the rule did (`gained`, `3 without a gain of the 9 the gain they
  show takes`, `3 without a gain: raised to 1.0e-04`, `not paid ..., within its noise:
  kept, the reading kept now this one`, `not paid ...: fallen back to step 4000's
  weights, fine 2.5e-05`); a raised head's rows likewise. The validation line repeats
  the rate, and the pass's last line says what it carries on: the fine rate and the
  passes in a row without a gain kept.

Each constant comes from a procedure, and each procedure has its test (`TheRate`):

- **The noise** (`apart`, `NOISE` 2): two readings differ by the mean of their windows'
  differences, which counts past `NOISE` standard errors of it, a loop's windows one
  draw since they share its sound. By chance alone it passes one way `ALPHA` of the
  time, 2.3%; counted window by window it would pass over three times as often.
- **A stall** (`STALL` 3, `stall`): a figure gaining g noises a reading has gained two
  after 2/g readings, one to pass the noise and one more against chance, and ⌊2/g⌋ + 1
  is its count. `STALL` is that at one noise a reading, the least a reading can tell:
  from a reading kept, two readings go without a gain 3.0% of the time, three under
  0.01%. A figure gaining less shows its pace: its first `STALL` readings without a
  gain read d under the reading kept, in a noise n, so its count is ⌊2·`STALL`·n/d⌋ + 1,
  and `STALL` where they show none. Each reading off by chance, a still-gaining figure
  is stalled at `STALL` and at that count in 19% and 0.3% of its stretches at half a
  noise a reading, 51% and 5% at a third, 66% and 14% at a quarter; one gaining
  nothing, in 94% and 80%, the rest a gain by chance.
- **The raise**: to `--lr`, the rate a pass is given, which trains a fresh model;
  nothing in the rule is set to the lab's passes.
- **The fall and the rise** (`FALL` 0.5): fine halves at each fall and doubles where a
  raise or a wide pass pays, a search by octaves either way: where a rate holds as long
  as it is small enough, fine lands within a factor of two under the highest that
  holds, in as many falls as that lies octaves under `--lr`, and a rate found too low
  comes back the way it went. A raise kept without a gain moves neither: it says
  nothing of the fine rate.
- **The whole**, replayed with no model on drawn readings (`Replay`), a reading's noise
  growing with how far the weights moved: over chains of four passes of twelve readings
  from a checkpoint with no rate, a figure gaining a quarter of a reading's noise at
  every reading keeps its fine rate, at or over where it began in three chains of four
  and under a step in twenty rolled back; one gaining nothing ends within an octave
  under it at the median, under a step in ten rolled back.

## 9. The weight file

[Transformer's format](https://github.com/lixenwraith/transformer/blob/main/doc/format.md)
is the contract. All numbers are little-endian. An FNV-1a 64 of every byte before it
ends the file. The checksum catches damage, not tampering. A file only ever holds
numbers, and `Load` checks its shape against its tensors. The trainer writes version 6:
a `kind` word
after the version, 0 for the frame model, fourteen heads, `beat`, `downbeat`, `struck` and
`struckvel` of one value each after `family`, an `out.weight` of 407 x 256, and after the heads the
constants it states, a count and for each a name and a float32: one, `layout`, the layout
of the frames it hears, 1 (audio's `ripmodels.Layout`, §2). Audio refuses a file of another
layout, or of none, by name, and so do the trainer's `--init` and `average.py`; it loads a
frame model whose heads start with `FrameSpec`'s and reads nothing of a head after them.

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
loads and rips with no glides. Version 6 needs transformer v0.7.0, its layout audio
v0.30.0, and a head after its specs' audio v0.34.0. The tensors are
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

### A tempo's count

A rip's tempo is right within 3 percent of the true one, counted in quarters in every
meter: a GuitarSet recording's is in its name, a generated loop's in its label (`bpm`),
and a 6/8 or 12/8 read on its dotted quarter is two thirds off, its eighths falling
between the rip's sixteenths. GuitarSet's players the frame model trained on are counted
apart from those it did not hear: a head that learned a recording marks its beats under
nearly any decoder.

Audio v0.37.0's cut to the finest beat the head marks, read through the first round's
frozen frame model on players 00 to 03 (160 recordings in its training, 80 not) and on
264 arranged loops of the round's generator at a seed no pass drew, before and after:

```
group               n  before  after      group          n  before  after
guitar comp       120     108    111      loops 12/8    17       5     11
guitar solo       120     105    110      loops 2/4     10       8      7
guitar trained    160     141    152      loops 3/4     39      19     28
guitar unheard     80      72     69      loops 4/4    124      93     95
guitar all        240     213    221      loops 5/4     19      16     17
                                          loops 6/8     32       4     17
                                          loops 7/8     23      10     11
                                          loops all    264     155    186
```

The guitar's gain is all on what the head trained on, fast jazz whose every beat it
marks; of the 80 it did not hear, five recordings at 98, four of them funk, read at
196, their eighths marked as high as their beats. A player no round heard and the round's held-out loops
decide.

Audio v0.38.0 sets `beat_cut_2` at 0.7 and reads twelve heard eighths beside 6/8. A bar
of two of another reading's is read only where the downbeat head stands at its lines
over the other's along the same stretch, a bar of three or more never. The same sets,
before and after, a bar right where the bar read lasts the true bar within 3 percent:

```
                    tempo          bar                      tempo          bar
group           n  before after  before after   group   n  before after  before after
guitar trained 160    152   145     148   145   12/8   17     11    11       4    11
guitar unheard  80     69    72      66    71   2/4    10      7     8       1     0
guitar all     240    221   217     214   216   3/4    39     28    33      24    27
                                                4/4   124     95    96      95   100
                                                5/4    19     17    17       1     1
                                                6/8    32     17    17      17    14
                                                7/8    23     11    12       0     0
                                                loops 264    186   194     142   153
```

The guitar moves by the cut alone: nine fast jazz recordings the head trained on, at
187 and 200, and one it did not, at 150, read at half; six at 98 and 100, four of them
unheard, are no longer doubled. Four 6/8 loops read 12/8, their head marking every other
line over the rest, and three 4/4 loops read 12/8 at three halves of their tempo, a bar
as long in seconds and not theirs.

Audio v0.41.0's joint path (`joint` 1) decides the tempo, the bar line and the meter as
one path over the heads, in the corpus's seven meters. Read with no model, the cues the
analysis's, on the 574 loops of 4 s or more of `corpus <dir> 600 4 aug gen meters seed
5`: with the labels as the heads, the decoders' ceiling, and with heads spoiled to the
trained F1s, 0.778 and 0.839 (beats missed, thrice as often in the first second, weak
beats and a compound bar's inner eighths lower, a simple bar's eighths at half and some
over a half, false peaks, the half bar marked as a line, a 6/8's every other line low,
its first line missed). A tempo against the label's `bpm` and against the bars' median
(played), a bar within 3 percent:

```
               labels as heads                    spoiled heads
           present          joint            present          joint
meter   n  tempo played bar  tempo played bar  tempo played bar  tempo played bar
4/4   239    214    239 239    213    239 239    205    231 229    213    238 235
3/4    81     76     81  81     75     81  81     65     70  66     74     80  80
6/8    65     51     61  48     56     65  65     51     58  35     56     64  57
12/8   64     45     54  48     55     64  64     42     51  38     54     64  57
2/4    22     21     22   0     21     22  22     21     22   1     17     20  18
5/4    51     44     51   0     43     51  51     40     47   0     40     51  50
7/8    52     27     30   0     45     52  52     32     39   0     45     52  51
all   574    478    538 416    508    574 574    456    518 369    499    569 548
```

A label's `bpm` is its loop's first bar's, and seven loops in ten drift, so perfect heads
miss it where the tempo the loop plays is read. Perfect heads cost the readings every
2/4, 5/4 and 7/8 bar and a quarter of 6/8's and 12/8's (read 4/4 or 3/4); the joint
path's errors on spoiled heads are the heads' own, 6/8 and 12/8 read as each other (14)
and 2/4 and 4/4 (7).

### A pair's table

`tool/rip-train/pipeline.sh readback table` reads a pair on recordings no fit saw, a
kind of sound a condition of its list, and writes one table: the models alone, under
their judge, and under each set a search kept for a kind, each row `tune 2 1 truth`'s
figures before its search ([rip-train](../tool/rip-train/README.md)). A change to what
a rip writes is read so before it merges, each kind beside what it read before. The
pair in use, its judge fitted on the rows this build writes, on 20 clips of a guitarist
and 100 takes of five singers no model heard, 185 generated loops of a seed no round
drew and 40 of E-GMD's test takes, whose drummers its train split shares:

```
kind       set          notes   ends     ends right  loudness  drums   bars  parts    key  chords
guitar     judge        0.820  0.418   1325 of 3172     0.756      -  0.668      -      -       -
voice      judge        0.329  0.626     144 of 230     0.707      -      -      -      -       -
voice      fit-voice    0.459  0.571     213 of 373     0.580      -      -      -      -       -
generated  judge        0.625  0.552   3015 of 5465     0.619  0.893  0.747  0.638  0.643   0.800
E-GMD      judge            -      -              -         -  0.490  0.787      -      -       -
```

A guitar and the generated loops keep the built-in thresholds, as no search finds a set
they keep; singing takes a set of its own (`fit-voice`), which a root holds as
`rip.voice.toml` and `rip` uses on a recording it reads as singing (`KINDS`, the row
`kinds`). A count with a label's events in the heads' place says what the decoders
could do with perfect heads: it is given as that, and nothing is changed or merged on it.

### The key

Audio names a rip's key after its bars ([audio
§9](https://github.com/lixenwraith/audio/blob/main/doc/audio.md#9-ripping-a-recording)):
the scale whose tones hold the most of the chroma less what lies outside them, and of
the keys of those tones, as a minor's are its relative major's, the one whose tonic
chord, as the rip names a bar's chord in it, the most of the music's bars rest on
(`key_bars`); then the chroma's lean to the tonic. `truth` and `tune ... truth` say how
each key was missed: at the given tonic in a scale holding or held by the given's
(nested), at the given's tones under another tonic (relative), or else. The first and
the last bar weigh as any other: generated loops open on their tonic chord 78 times in
100 and close on it 48, and cut a bar or two in from each side 18 and 25, as a recording
cut from a longer piece is. What a generated label cannot tell: a loop that sounds a
relative's tonic chord as long as the label's has its tonic by the arranger's word.

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
