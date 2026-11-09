# Rip model training: the data, byte by byte

How a recording or a rendered loop becomes the numbers the rip's model trains on,
and how the trained model reads back into the rip. Every sample below is real:
corpus 11's first loop and the weights of round 12a. The audio library holds the
frames, the input planes and the heads; [transformer's
format](https://github.com/lixenwraith/transformer/blob/main/doc/format.md) is the
weight file's contract; stapes holds the corpus verb and the trainer
([rip-train](../tool/rip-train/README.md)), and [Rip design](rip-design.md) keeps
the reasoning and the history of the rounds.

Where each part stands:

- **In released code:** `MelFrames`, the first and second input planes, the heads,
  weight file versions 3 and 4, `corpus`, `corpus <dir> from`, `corpus ... sampled`, `--mix`,
  `--channel`, `--over` with its heard-hit rule, draws by length, `skip.txt`,
  `make_noise.py`, `corpus ... midi`, `--pitchnet` and its `ripnotes.bin`, `--shuffle`, `--extra`.
- **In no repository yet:** the phone take's scripts, which play, align and cut a
  recording of a known file.

## 1. The path

1. A corpus is rendered or read: per loop, a `.mel` file and a `.json` file (§2, §3).
2. Each training step draws 8 windows of 512 frames, 5.9 s each (§7).
3. A window's frames may be changed in place, as §6 lists, then become two input
   planes (§4).
4. A window's events become per-frame targets for the 10 heads (§5).
5. The model turns frames into 327 numbers a frame, and the losses train it (§7, §8).
6. The best step is written as a weight file (§9).
7. At a rip, Go computes the same planes, runs the file and hands the heads to the
   decoders (§10).

## 2. The frames: `loop_NNN.mel`

The file is the header, then every band of every frame as a little-endian float32:

```
offset  size            field
0       u32             frames
4       u32             bands, always 229
8       4 x frames x bands   rows, frame 0's 229 values first, nothing padded
```

Corpus 11's `loop_000.mel` is 789,600 bytes: 8 + 4 x 862 x 229. Its first 40 bytes:

```
00000000  5e 03 00 00 e5 00 00 00 8d 0e a5 3e 8f 4e 8d 3f  ^..........>.N.?
00000010  0a 25 e5 3f 6a 90 0d 40 71 23 0e 40 74 72 c8 3f  .%.?j..@q#.@tr.?
00000020  1c c1 96 3f 33 df 6b 3f                          ...?3.k?
```

- `5e 03 00 00` is 862 frames, 10.0 s of music, and `e5 00 00 00` is 229 bands.
- `8d 0e a5 3e` is 0x3ea50e8d, the float 0.32238: frame 0, band 0.
- `8f 4e 8d 3f` is 1.10396: frame 0, band 1, and so on up the bands.

A frame costs 916 bytes. There are 86.13 frames a second, so an hour is 284 MB.

Each value is `log1p(100 e)`, made by `MelFrames`:

1. The recording is resampled to 22,050 Hz and scaled so its largest sample is 1.
2. A 2,048-sample Hann window (93 ms) moves 256 samples (11.6 ms) at a time. Each
   FFT gives 1,025 magnitudes, 10.8 Hz apart, each scaled by 1 / (2048 / 4), so a
   full-scale sine's peak reads about 1.
3. 229 triangular filters, evenly spaced on the mel scale from 30 Hz to 11,025 Hz,
   each sum weight x magnitude squared. That sum is the band's energy e.

Going back: `e = expm1(v) / 100`. The first value above is e = 0.0038, or -24.2 dB
against a full-scale sine. The file holds energy only: with the phase gone, it can't
be played back exactly.

Bands are about 14 Hz apart at A4, where a semitone is 26 Hz, and about 10 Hz apart at
A2, where a semitone is 6.5 Hz. So the bottom octaves give the model less than one band
a semitone, and it must lean on upper partials there.

## 3. The labels: `loop_NNN.json` and `corpus.json`

A label is UTF-8 JSON with no whitespace. The start of corpus 11's `loop_000.json`,
as `corpus` writes it now:

```
{"bpm":96,"root":36,"scale":1,"bars":4,"frames":862,"tuning":49.64392715869181,"events":[{"frame":0,"end":0,"drum":0,"midi":0,"vel":0.9871591431922611,"wave":-1,"family":7},{"frame":108,"end":108,"drum":0,...
```

The loop's fields:

- `bpm`, `root`, `scale`, `bars`: how it was drawn. The trainer doesn't read them.
- `frames`: the length of its `.mel`.
- `tuning`: cents off A440 it was rendered at, within 50 either way. Events keep the
  written pitch, so at nearly +50 cents as here, a note labelled 55 sounds nearly as
  much like 56.
- `source`: in a corpus of recordings, the file it was heard from.
- `events`: in no time order; the trainer sorts them.

An event's fields:

- `frame`, `end`: its start and its release, in frames. A drum's are equal.
- `drum`: 0 kick, 1 snare, 2 hihat, -1 anything else.
- `midi`: the pitch, 0 for a drum.
- `vel`: how hard it was struck, 0 to 1.
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
  note whose file set no program none.
- `voice`: the 32 genes of the voice that played it, absent when unknown.

The four kinds of event in use, each from a real file:

```
a drum hit          {"frame":0,"end":0,"drum":0,"midi":0,"vel":0.987,"wave":-1,"family":7}
a note on a voice   {"frame":0,"end":40,"drum":-1,"midi":55,"vel":0.662,"wave":1,"family":8,"voice":[0.5,0.417,0,0.157,...]}
a recorded note     {"frame":0,"end":40,"drum":-1,"midi":55,"vel":0.662,"wave":-1}
a strike, no line   {"frame":23,"end":23,"drum":-1,"midi":0,"vel":0.787,"wave":-1}
```

A recorded note is one of NSynth's, so its voice is unknown. The last is a tom or a
crash from E-GMD: `drum` -1 with `midi` 0 means something struck that no drum head or
pitch owns. Loop 000 holds 110 events in 34 KB: 12 kicks, 12 snares, 28 hats, 58 notes.
Its notes are unknown's: `aug` drove its lead. The last two lines are corpus 12's and
E-GMD's, written before `family`, as every corpus below was: a loop of them labels no
family until it is exported again, its label removed first since `corpus` keeps a
labelled loop. The same seeds render the same frames, byte for byte over 24 loops of
`corpus <dir> 24 4 aug gen` compared, and of those loops' frames 40% are synth's, 29%
unknown's, 17% guitar's, 8% piano's, 4% bass's, 0.1% the drums', a hit sounding alone,
and 1% have no event sounding.

`corpus.json` names the layout for the whole folder:

```
{"version":1,"loops":40000,"rate":22050,"hop":256,"mel_bands":229,"min_midi":28,
 "pitches":69,"drums":["kick","snare","hihat"],"waves":["sine","saw","square","string","modal"],
 "families":["piano","guitar","voice","bass","strings","winds","synth","drums","unknown"],"voice_genes":32}
```

`loops` counts the files `loop_000` onward; the number grows past three digits as
needed (`loop_39999`). The corpora in use:

- **Corpus 11:** `corpus <dir> 40000 4 aug gen nowav codec`. Rendered loops of 4 bars,
  25.6 million frames, 24 GB. [Rip design §3](rip-design.md#3-learning) lists what
  `aug` and `gen` draw. `codec` sends a third through Opus, MP3 or AAC at 64 to 192
  kbps and back.
- **Corpus 12:** the same with `sampled <dir>`, so the same seeds draw the same notes.
  Each tonal line plays on NSynth's recorded notes with a chance of a half, 44% of all
  notes, each shifted at most 3 semitones and cut at its note-off by a 60 ms fade.
- **Corpus 14:** `corpus <dir> 20000 4 aug gen nowav codec midi <dir>`. Three quarters of
  the loops play a random 4-bar stretch of a random file from Lakh's clean MIDI set: its
  beat, melody, second line and bass, moved up to half an octave, the second line on a
  random tonal instrument.
- **Corpus 15:** `corpus <dir> 20000 4 aug gen nowav codec dense seed 151 midi <dir>`.
  Corpus 14's recipe playing every part of the file: the parts past its melody,
  second line and bass spread by octave over up to three tracks on instruments of
  their own, each up to 24 dB under the file's level, so a soft inner line sounds
  under a ring. Corpus 16 is the same at `seed 161`.
- **E-GMD:** `corpus <dir> from <list.csv> split train codec`. 5,691 real recordings
  of 43 kits, 55.6 hours, 17.2 million frames, 15 GB. Each loop is one recording, from
  2 seconds to 10 minutes, with its hits from the MIDI file. Its keys are mapped by
  `midiInstruments`: edges, ride, tambourine and shaker read as hihat, cross-stick as
  snare, and other keys as strikes of no line.

E-GMD needs a check `corpus` doesn't make yet. Lining each recording's hits up with
its onset flux showed 147 recordings of 30 hits or more sitting a steady distance off
their audio, 5.2% of the frames, the same across all of a performance's kits and with
no drift over time. Labels 46 to 116 ms off were moved by that lag, and the 15 further
off went in the corpus's `skip.txt`, a list of loop names the trainer leaves out.

## 4. The input: two planes

The model reads a frame as two planes of 230 values each, laid end to end, 460 in all.
Plane 1 is the band shape and the level:

```
e[b]      = expm1(v[b]) / 100
db[b]     = 10 log10(e[b] + 1e-8)
level     = mean of db over the 229 bands
plane1[b] = (db[b] - level) / 80
plane1[229] = (level + 80) / 80
```

A gain moves only `plane1[229]`, so a phrase 20 dB softer reaches the model with the
same shape. Plane 2 is PCEN (per-channel energy normalization), each band over its own
recent average, worked out per band from the frame's energy:

```
s         = 1 - exp(-(256 / 22050) / 0.4)  = 0.028608
M[b](t)   = (1 - s) M[b](t-1) + s e[b](t),  M[b](-1) = e[b](0)
plane2[b] = (e[b] / (1e-6 + M[b])^0.98 + 2)^0.5 - 2^0.5
plane2[229] = mean of plane2 over the 229 bands
```

- Dividing by M is a gain control: a band that has been loud is turned down.
- The 2 and the root compress the result: a band far under its average lands near 0,
  not at a logarithm's large negative number.
- M only looks back, so a live rip can compute it as the sound arrives.

Frame 20 of loop 000, where a G3 and a C3 sound. The frame's level is -53.14 dB, so
`plane1[229]` is 0.3358, and `plane2[229]` is 0.0650:

```
band   centre Hz  stored float (bytes)     e          dB      plane 1   plane 2
  6       94.4    0.002231 (bd 38 12 3b)   2.234e-05  -46.51  +0.0829   0.0005
 16      196.3    2.490533 (e4 64 1f 40)   1.107e-01   -9.56  +0.5447   0.5988
 32      387.2    0.061227 (34 c9 7a 3d)   6.314e-04  -32.00  +0.2643   0.0726
 46      587.4    0.033605 (66 a5 09 3d)   3.418e-04  -34.66  +0.2310   0.1718
 58      788.1    0.000521 (13 b1 08 3a)   5.216e-06  -52.82  +0.0040   0.0134
 78     1194.5    0.123396 (f2 b6 fc 3d)   1.313e-03  -28.82  +0.3040   0.6715
119     2407.6    0.000137 (3c cb 0f 39)   1.371e-06  -58.60  -0.0682   0.0024
166     4780.6    0.000062 (7b 9b 82 38)   6.228e-07  -61.99  -0.1106   0.0003
218     9567.0    0.000005 (cb 94 a4 36)   4.905e-08  -72.29  -0.2394   0.0004
```

Band 16, G3's fundamental, as the note starts and holds. PCEN jumps with the attack and
settles while the band's energy stays high:

```
frame   e          M          PCEN
  0     1.887e-02  1.887e-02  0.2956
  6     1.278e-01  3.242e-02  0.9690
 12     9.709e-02  4.631e-02  0.5787
 18     8.765e-02  4.743e-02  0.5194
 24     1.052e-01  5.717e-02  0.5189
 30     5.793e-02  6.002e-02  0.2924
 36     6.763e-02  6.047e-02  0.3343
 42     8.712e-02  6.337e-02  0.4027
```

What PCEN reads for bands driven with test envelopes:

```
steady at -6 dB                   0.31
steady at -46 dB                  0.25
fading 60 dB over 5 s             0.25 at 0.1 s, 0.08 at 1 s, 0.02 at 3 s
struck from -40 to -20 dB         0.27 before, 3.5 on it, 1.4 at 50 ms, 0.58 at 200 ms
struck 6 dB over a band sounding  0.29 before, 0.90 on it
```

So steady noise sits near 0.3 at any level and a note on top still rises over it, but
a note that's still ringing fades towards zero. That's why PCEN is a second plane and
not a replacement: plane 2 says when something happened, plane 1 what is sounding.

Over corpus frames, plane 2 averages 0.16, its 99.9th percentile is 2.4, and its largest
value is about 4, close enough to plane 1's scale to need no rescaling. A model with one
channel reads plane 1 alone, 230 values.

`--pitchnet` reads both planes through a harmonic stack it computes itself. A plane's bands
are read on a log-frequency axis of 36 bins an octave, by interpolation. Each plane then
gives 8 channels, one per harmonic of 0.5, 1 and 2 to 7, each shifted by -36, 0, 36, 57, 72,
84, 93 or 101 bins, so a pitch's harmonics line up at its own bin. That's 213 bins from MIDI
27 in 16 channels, each channel ending in its plane's scalar.

## 5. The targets

`Label.window(a, b)` builds what each head should say for frames a to b. A note's pitch
index is `midi - 28`, and one outside 0 to 68 sets no pitch.

- `onset`: within one frame of any event, drum or note, that is `t-1 <= frame < t+2`.
- `drums[d]`: the same, for that drum's hits.
- `pitch[p]`: from a note's `frame` until its `end`, at least one frame.
- `voicing`: any pitch on.
- `strike[p]`: within one frame of a note's start.
- `release[p]`: within one frame of a note's `end`, unless that's past the loop.
- `velocity[p]`: the note's `vel` on its strike frames, scored only there.
- `wave`, `voice`: those of the highest note sounding with a known voice, scored only
  on frames that have one.
- `family`: that of the highest event sounding, a hit or a strike at midi 0 on its own
  frame, scored only on frames where it is labelled: a hit never stands in for an
  unlabelled note over it.

Frames 0 to 7 of loop 000, from its kick and hat at 0, its G3 (55) and its C3 (48):

```
onset    1 1 0 0 0 0 0 0
kick     1 1 0 0 0 0 0 0
snare    0 0 0 0 0 0 0 0
hihat    1 1 0 0 0 0 0 0
pitch 20 1 1 1 1 1 1 1 1     C3, on until frame 27
pitch 27 1 1 1 1 1 1 1 1     G3, on until frame 40
voicing  1 1 1 1 1 1 1 1
strike   (frame 0, pitch 27, vel 0.662), (frame 0, pitch 20, vel 0.485), each on frames 0 and 1
release  (frame 40, pitch 27), (frame 27, pitch 20)
wave     1 on every frame: saw, from the highest note, G3
family   8 on every frame: unknown, from the highest note, G3
```

A window running past a loop's end is padded with zero frames, which a `valid` mask
leaves out of every loss.

## 6. What may change a window's frames

These run on the stored frames during training, so one corpus serves many runs. All
add in power: `e` is recovered, changed and stored back through `log1p`. Their order in
`sample` is the overlay, then the channel, then the planes. With two channels, 128
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

The bank holds two arrays. `white`, 10,328 x 229 float32, is two minutes of white noise
as mel frames with each band's mean set to 1, so a band keeps its own frame-to-frame
spread (0.91 of its mean at 57 Hz, 0.32 at 10 kHz). `room`, 229 float32, is a real
room's band shape in dB about its mean, from a phone 30 cm from a laptop with a fan
running: +12.1 at 114 Hz, +7.9 at 806 Hz, -13.3 at 10 kHz. `make_noise.py` builds it
from a WAV of the room.

A corpus's own noise sits 40 to 65 dB under the loop. On that phone take the floor sat
7 to 15 dB under the music, and 8b, trained without any of this, lost half its recall.
Loop 000's first 512 frames as stored and through three draws, the mean per band in dB:

```
Hz          60     110     200     400     800    1600    3200    6400   10000
stored   -19.5   -26.3   -13.4   -26.5   -27.9   -34.6   -43.2   -43.3   -48.6
draw 0   -27.1   -25.0   -15.7   -30.5   -28.6   -36.2   -42.8   -63.3   -52.5
draw 1   -26.5   -29.7   -13.3   -27.4   -26.7   -34.0   -40.8   -42.8   -52.0
draw 2   -21.7   -28.6   -12.4   -28.0   -27.8   -35.4   -45.3   -43.5   -66.9
```

**`--over dir:share`** lays real drums over rendered music. A share of the windows draw
a loop with no drums of its own (11,933 of corpus 11's 39,600), then a recording from a
corpus of drums alone, placed at random. Its level is -12 to +6 dB of its mean power
against the loop's, and uncorrelated sounds add in power, so `e = e_loop + gain x e_drums`.
A hit joins the onset and drum targets only where it can be heard: what it adds, over its
best ten bands, must top everything else sounding (`stands_out`), judged after the
channel. That keeps 86 to 96% of hits. Round 14a, without the rule and at -12 to +3 dB
of the loop's own level, had drums a median 13.4 dB under the loop against Billie Jean's
-0.3 dB, labelled whether heard or not, and it reported drums on bass notes.

**`--mix dir:share`** draws a share of the windows from another corpus as it stands.
`--extra dir:share,...` does the same for any number of corpora.

**`--shuffle share`** splices a share of the windows from chunks of a second, each two clips
from random places summed, the second at -6 to +6 dB of the first's power. Notes keep their
labels, a drum hit counts where it stands out of the other clip, and wave, voice and
family go unasked.

**`--sung dir:share`** draws that share of the chunks from a corpus of solo singing. 27a, at a
quarter with half the windows shuffled, took held-out sung notes from 0.54 to 0.63.

**`--ring share`** adds to that share of the windows a body's or a room's ring at every attack:
one pitch whatever the note, 70 to 160 Hz, decaying in 0.05 to 0.3 s, 25 to 8 dB under the
attack's rise, its harmonics absent and nothing labelled, as a guitar's air cavity rings near
104 Hz on every pluck. 32a, at a quarter, took GuitarSet's solo low extras from 424 to 362
against 31a and every real set up but the renders; a first try at 70 to 400 Hz and up to 0 dB
(30a) cut them further and cost every other set.

**`--bend share`** reads that share of the plain windows along a tempo that sways: a rate up
to 18% either way and a slow sine of up to 12 frames, the features interpolated and the
targets taken nearest, so no steady grid or pulse is there to learn, as a pianist's isn't.
It came in with corpus 15 (31a, at 0.3), so its share of that round's gain is unmeasured.

**`--strike-weight w`** is a strike's weight against a frame without one, 20 by default:
at 4 (28a) the head's outputs fell so far that no pair of bars matched 25a on the real
path. **`--again-weight a`** weighs a strike of a pitch still sounding a times more, the
miss under most of what the rip loses (strummed guitar's 2133 of 3764 misses); 31a and
since take 3, unmeasured alone for the same reason as the bend.

Draws from both are by length, since E-GMD's recordings run from 2 seconds to 10
minutes: 521 of its first batch's 819 are shorter than a window but hold only 4.8% of
its audio. Drawn by recording, they would be 64% of the windows.

## 7. The model

`tool/rip-train/ripmodel.py` builds it, and `transformer.Forward` computes the same
numbers in Go. The shapes through a window, with B the batch:

```
input          B x 512 x 460
reshape        B x 2 x 512 x 230             channels x frames x features
conv 0         B x 16 x 512 x 230            3x3, GELU
conv 1         B x 16 x 512 x 115            3x3, GELU, max over 2 features
conv 2         B x 32 x 512 x 57             3x3, GELU, max over 2 features
flatten        B x 512 x 1824                the last odd feature dropped by the pool
input layer    B x 512 x 256
4 blocks       B x 512 x 256                 pre-norm attention and feed-forward 1024
norm, output   B x 512 x 327                 raw scores
```

Attention has 8 heads and looks 64 frames (0.74 s) either side, in every block. There
are no position embeddings: each head subtracts a slope of `2^(-8(h+1)/8)` per frame of
distance from its scores (ALiBi), so a window of any length works.

The 3,718,055 weights: 304 + 2,320 + 4,640 in the convolutions, 467,200 in the input
layer, 789,760 in each block, 512 in the final norm, 84,039 in the output layer, 2,313
of them the family head's, which 12a's file lacks. A one-channel model has 144 fewer, all
in `convs.0.weight`.

The 327 outputs, in the order the file lists the heads:

```
offset    size  head       read as
  0         1   onset      sigmoid
  1..3      3   drums      sigmoid: kick, snare, hihat
  4..72    69   pitch      sigmoid: MIDI 28 to 96
 73         1   voicing    sigmoid
 74..78     5   wave       softmax: sine, saw, square, string, modal
 79..110   32   voice      sigmoid: genes
111..179   69   strike     sigmoid
180..248   69   release    sigmoid
249..317   69   velocity   sigmoid
318..326    9   family     softmax: piano, guitar, voice, bass, strings, winds, synth, drums, unknown
```

A file may lack the last 4 heads and still load: one without `family` hears no family.

`--pitchnet` builds another model on the same heads but `family`, the frame model's to hear:
convolutions only, the same weights at every pitch, 169,862 weights in all. Its file is
`ripnotes.bin` (§9). `--dilations 3,9,27` adds a residual 3x3 over the cells for each, its
frames that far apart: each note head then hears half a second either way instead of 80 ms,
197,606 weights, about 1.5 times the compute.

```
input          B x T x 460              the two planes
stack          B x 16 x T x 214         the harmonic stack of section 4
reduce         B x 8 x T x 213          1x1, GELU, the scalars set aside
wide           B x 16 x T x 213         3x39, thirteen semitones, GELU
cells          B x 32 x T x 71          7x3, stride 3 over features, one cell a semitone, GELU
near           B x 32 x T x 71          7x3 with a residual, GELU
long           B x 32 x T x 71          3x3 dilated in time, one each of --dilations, with a residual, GELU
notes          B x 4 x T x 69           1x1: pitch, strike, release, velocity
frame          B x T x 39               wide pooled to 16 regions plus the scalars, a 1D conv of 5
                                        and one of 9 dilated by 2, linear: the other heads
```

## 8. A training step

Each step: 8 windows, targets, forward, losses, backward, gradient clipped to norm 1,
AdamW at weight decay 0.01. The learning rate runs from 3e-4 to zero on a cosine over
the steps. The next batch is built on a second thread while the device works.

```
onset, drums, pitch, voicing, strike, release: binary cross-entropy with a weight on
    the positive class of 8, 8, 12, 2, 20, 20
velocity: squared error, on strike frames with a known velocity
wave:     cross-entropy, on frames with a known voice
voice:    squared error, on frames with a known voice
family:   cross-entropy, on frames with a known family, the frame model's alone

total    = onset + drums + pitch + 0.5 voicing + strike + release + 2 velocity + 0.5 wave + 2 voice
           + 0.5 family
decoders = onset + drums + pitch + 0.5 voicing + strike + release
```

The last 400 loops are held out and scored every 1,000 steps, and the step with the best
`decoders` score is what gets written, unless `--validate` names a corpus of recordings
(`corpus <dir> from`): cut into windows the same way and scored beside them, its best
`decoders` step is written, the held-out score printed beside it. The wave, voice,
velocity and family heads don't choose it, because they memorize the corpus. A positive
weight w shifts a head's scale: it scores a posterior p as wp / (wp + 1 - p), which is
part of why thresholds belong to the weights (§10).

Round 12a's log, first and last, on a GTX 1060:

```
3715742 weights, 39600 loops, 25330800 frames, 400 held out in 734 windows
step 20 loss 4.631 (onset 1.578 drums 0.748 pitch 0.752 voicing 0.041 strike 0.372 release 0.361 velocity 0.038 wave 1.213 voice 0.059) 5s
step 40000 loss 1.107 (onset 0.064 drums 0.017 pitch 0.327 voicing 0.029 strike 0.083 release 0.092 velocity 0.023 wave 0.778 voice 0.038) 16557s
step 40000 held-out 1.651, decoders 0.965 (onset 0.284 drums 0.035 pitch 0.393 voicing 0.059 strike 0.102 release 0.121 velocity 0.027 wave 1.079 voice 0.047) (best)
```

A model with the family head prints its loss last, as 20 steps on a CPU did over the 4
loops of `corpus <dir> 4 4 aug gen`, a model of the defaults without `--pcen`:

```
step 20 loss 4.817 (onset 1.521 drums 0.923 pitch 0.716 voicing 0.060 strike 0.530 release 0.513 velocity 0.026 wave 0.374 voice 0.077 family 0.380) 20s
```

The wave head's held-out 1.079 means about a third of its belief on the right wave,
against a fifth by chance, and it only ever learned synth voices. So on a real instrument
its guess means little, which is why a rip can play a piano on a saw.

The checkpoint, a PyTorch `.pt`, holds `model`, `opt`, `step`, `cfg` (the shape, which a
resumed run must match), `best` (the best held-out score, its step and its weights) and
`validated` (the best validation score, its step, its weights and the held-out score there).

## 9. The weight file

[Transformer's format](https://github.com/lixenwraith/transformer/blob/main/doc/format.md)
is the contract. All numbers are little-endian. Round 12a's file is 14,865,070 bytes:

```
00000000  56 49 46 54 03 00 00 00 e6 00 00 00 02 00 00 00  VIFT............
00000010  00 01 00 00 08 00 00 00 04 00 00 00 00 04 00 00  ................
00000020  40 00 00 00 03 00 00 00 10 00 00 00 03 00 00 00  @...............
00000030  03 00 00 00 01 00 00 00 10 00 00 00 03 00 00 00  ................
00000040  03 00 00 00 02 00 00 00 20 00 00 00 03 00 00 00  ........ .......
00000050  03 00 00 00 02 00 00 00 00 00 00 00 09 00 00 00  ................
00000060  05 00 6f 6e 73 65 74 01 00 00 00 05 00 64 72 75  ..onset......dru
```

```
offset  bytes         field
0       56 49 46 54   magic "VIFT"
4       03 00 00 00   version 3 (2 has no channels field, and reads as one channel)
8       e6 00 00 00   inputs 230
12      02 00 00 00   channels 2
16      00 01 00 00   dim 256
20      08 00 00 00   heads 8
24      04 00 00 00   layers 4
28      00 04 00 00   ffn 1024
32      40 00 00 00   window 64
36      03 00 00 00   3 convolutions, then 4 u32 each: out, time, freq, pool
40      16 3 3 1      conv 0
56      16 3 3 2      conv 1
72      32 3 3 2      conv 2
88      00 00 00 00   0 towers
92      09 00 00 00   9 heads, then each: u16 name length, the name, u32 size
96      05 00 "onset" 01 00 00 00, and so on
202     4c 00 00 00   76 tensors, then each: u16 name length, the name, u8 rank,
                      u32 dims, float32 data
206     convs.0.weight   dims 16 2 3 3, 288 floats, the first +0.40755
1391    convs.0.bias     dims 16
...
14538122 out.weight      dims 318 256
14863775 out.bias        dims 318
14865062 45 f5 b0 4b 8f 26 00 58   FNV-1a 64 of every byte before it
```

The checksum catches damage, not tampering. A file only ever holds numbers, and `Load`
checks its shape against its tensors.

`--pitchnet` writes version 5, whose `kind` word after the version, 1, makes the model
transformer's cell net. Its sizes follow, from `bins` to the frame path's dilation and
scalars, then the long convolutions' spacings, then the heads, the per-pitch ones first: pitch,
strike, release and velocity, then onset, drums, voicing, wave and voice. The tensors are
`reduce`, `wide`, `cells`, `near.0`, `long.{i}`, `cellout`, `frame1`, `frame2` and `frameout`, each as its `nn.Conv2d`,
`nn.Conv1d` or `nn.Linear` holds it. In the stapes root the file is `ripnotes.bin`.
Beside `ripmodel.bin` it gives pitch, voicing, strike, release and velocity, and the model
keeps onset, drums, wave and voice. Alone it is unused. `average.py --out ripnotes.bin
a.pt` turns a `.pt` of either model into its file.

## 10. Reading the model back

`ripmodel.Hear` runs `Forward` over all of a recording's frames, and a notes model's over
their harmonic stack, built in Go as §4 builds it, takes the sigmoid of
each head and fills `RipFrames`, the same fields the analysis fills. Loop 000 through
12a, with a sure strike on frame 0 where the labels say kick, hihat, G3 and C3:

```
frame  onset  kick   snare  hihat  voicing  pitch top three         strike top two
  0    1.000  1.000  0.001  0.998  1.000    37 .95  56 .92  49 .89  37 .96  36 .87
  1    1.000  0.999  0.000  0.995  1.000    37 .97  56 .93  49 .90  37 .97  49 .88
  2    0.007  0.001  0.000  0.000  1.000    37 .97  56 .93  49 .89  37 .09  49 .02
  3    0.000  0.000  0.000  0.000  1.000    37 .96  56 .92  49 .89  37 .00  49 .00
 20    0.001  0.000  0.000  0.000  0.999    37 .97  56 .92  49 .87  37 .00  56 .00
```

The drum heads follow the two-frame target closely. This loop also shows two weak spots:

- It's rendered 49.6 cents sharp, the edge of the corpus's range, so G3 and C3 read as
  their neighbours 56 and 49. The written pitch isn't what's heard here.
- 37 reads as a sure note, with sure strikes at frames 0 and 40, where no note is
  labelled; this loop has no bass. 11a and 12b do the same. There's little sound there
  (-41 dB at 66 Hz by frame 20), so the likely reading is a learned expectation: most
  loops have a bass on the root, and this root (36) sounds nearly as 37 at this tuning.
  False notes of this kind reach real recordings, such as the 18 in Billie Jean's 5 s
  drum intro.

The decoders then read the frames: hits by peak picking, the bass by one hidden path,
the melody register by strike peaks. `strike` and `strike_again` are the thresholds
that matter, and they belong to the weights, so a root pins them beside the file:

```
[rip]
strike = 0.7
strike_again = 0.9
release = 0.9
```

## 11. Rounds on record

From 11a on, all run 40,000 steps at batch 8, window 512, seed 1, with the shape in §7.
Round 8b, on an earlier corpus, has that shape but one input plane and no release or
velocity heads, 3,680,132 weights. `clean`, `30 cm`, `Pixel` and `Galaxy` are
pooled F1 of the melody register as one line, over Für Elise (869 notes), One More Bite's
origin (83) and its intro (83), every note once, at the column's best settings. The last
three are those recordings played on a laptop: recorded by a phone at 30 cm with a fan on,
and by a Pixel 8 Pro and a Galaxy S20 Ultra at 2 m. The 30 cm take also shaped the
channel's noise, so it flatters every round with a channel. `bench` is `bench 12 4` on the
same loops, clean and augmented. `kits` is F1 of kick, snare and hihat at bar 0.1 on 40
recordings of E-GMD's held-out split.

```
round     corpus  plane 2  channel  real drums  clean   30 cm   Pixel   Galaxy  bench         kits
analysis  -       -        -        -           0.632   0.597   0.544   0.515   0.383 0.380   0.775
8b        earlier -        no       no          0.676   0.518   0.495   0.504   0.485 0.434   0.701
11a       11      -        no       no          0.676   0.522   0.551   0.513   0.469 0.442   0.578
12b       11      -        yes      no          0.687   0.583   0.553   0.558   0.426 0.426
12a       11      PCEN     yes      no          0.711   0.623   0.629   0.639   0.427 0.413   0.659
14a       12      PCEN     yes      unheard     0.662   0.576   0.608   0.593   0.446 0.382   0.814
16a       12      PCEN     yes      heard       0.668   0.589   0.603   0.627   0.446 0.393   0.822
16b       11      PCEN     yes      heard       0.703   0.616   0.624   0.626   0.373 0.406   0.826
17a       13      PCEN     yes      + kits      0.666   0.590   0.628*  0.662*  0.408 0.424   0.810
```

\* Its training heard loops from those takes, so these two aren't a fair test. Corpus 13
adds long sung lines, long bass and soft notes to corpus 11; 17a also trained on 4000
drum loops on random kits, mixes darkened to 800 Hz, and 267 loops from the phones.

- **PCEN** is 12a against 12b: +0.024 clean, +0.040 from the phone.
- **The channel** is 12b against 11a: +0.011 clean, +0.061 from the phone.
- **The bench's drums** fell with both, from 0.66 (11a) to 0.56 (12b) to 0.47 (12a), yet
  real recordings improved. On Billie Jean from the phone, 12a hears 37 of 39 kicks and
  138 of 155 hats where 8b heard 2 and 38. The bench's engine drums aren't the target.
- **Real drums** are 16b against 12a: kits 0.659 to 0.826 (0.876 at bar 0.3) for 0.008
  clean. They cost phones: Billie Jean's beat from the Pixel at bar 0.5 has 2 of 36 kicks
  against 12a's 33 at bar 0.1, so 12a stays installed.
- **The labels' audibility** is 16a against 14a: clean +0.006, 30 cm +0.013, Pixel -0.005,
  Galaxy +0.034, and `bench sampled` augmented 0.421 to 0.487.
- **More steps** don't pay: 16b continued at half the rate moved its held-out decoders'
  loss from 0.994 to 0.985 by step 14000, then stayed flat.
- **Basic Pitch** on the same clips and scorer, at its default thresholds and its best for
  each take: clean 0.727 and 0.778, 30 cm 0.548 and 0.606, Pixel 0.608 and 0.614, Galaxy
  0.567 and 0.593. Its lead is Für Elise alone, 0.828 against 12a's 0.721, and it trained
  on real piano. Its own decoder on 12a's heads scores 0.619 clean against ours at 0.704,
  so that gap is the model's.

`--pitchnet` rounds run 20,000 steps at rate 1e-3 with the channel, `--over` and `--mix` of
16b, decoded at strike 0.7, strike_again 0.9, release 0.9 and voicing 0.6. Their frame heads
stay weak, 22a's kicks falling to 0 of 36 from the Pixel at bar 0.5, so these rounds are about
notes.

```
round  corpus                       clean   30 cm   Pixel   Galaxy
22a    11                           0.774   0.681   0.755   0.754
24a    14                           0.807   0.661   0.744   0.723
24b    11, MusicNet 20%, GuitarSet  0.764   0.653   0.692   0.697
```

Audio v0.5.3 reads low notes by strikes as well and drops shadows under real notes, which moves
the same weights with no training. The same scorer through stapes itself, on audio v0.6.0, two
later rounds added:

```
round  corpus                 windows       clean   30 cm   Pixel   Galaxy
22a    11                                   0.785   0.682   0.756   0.755
24a    14                                   0.819   0.660   0.749   0.722
25a    14, 11 at half                       0.819   0.681   0.767   0.752
26a    14, 11 at half         half shuffled 0.831   0.658   0.746   0.748
```

Through the accounting of audio v0.8.0 and the timed match, on sets chosen to stop fitting four songs; each trained on 25a's mix, with what the column names:

```
                        25a     29b, +-0.5 s    27a, singing, shuffled
Fur Elise renders       0.895   0.904           0.913
Fur Elise clean         0.826   0.824           0.820
Fur Elise Pixel         0.698   0.724           0.699
GuitarSet strummed      0.531   0.544           0.549
GuitarSet solo          0.779   0.812           0.787
MusicNet                0.583   0.595           0.593
vocadito, sung          0.539   0.616           0.634
```

Basic Pitch's best per take is 0.778, 0.606, 0.614, 0.593. Under E4 the decoder was most of the
gap: on rendered instruments E3 to D#4 went from 0.61-0.77 to 0.74-0.92 on 24a's weights.

- **Weights shared across pitches** are 22a against 19a, the same stack fed to the model of
  section 7 on 3.7M weights: 0.693, 0.604, 0.623, 0.638. The stack alone gains nothing.
- **Attention beside them** lost: both paths in one model of 3.4M weights scored 0.751, 0.678,
  0.721, 0.735, below 22a on every take and on seven rendered instruments, so notes get a
  model of their own and that path was not kept.
- **Real arrangements** are 24a against 22a: clean +0.033, phones -0.011 to -0.031.
- **Real recordings** at that mix are 24b against 22a: lower nearly everywhere.
- **Shuffled windows** are 26a against 25a: clean +0.012 and rendered instruments +0.02 to +0.04, phones -0.004 to -0.023.

To repeat 12a from a checkout with the trainer's changes:

```bash
CORPUS=<corpus 11> PCEN=1 CHANNEL=<noise.npz>:0.5 STEPS=40000 HOLDOUT=400 EVERY=1000 CONVS=16x3x3/1,16x3x3/2,32x3x3/2 tool/rip-train/pipeline.sh train
```

And 16b, which adds an E-GMD corpus:

```bash
CORPUS=<corpus 11> PCEN=1 CHANNEL=<noise.npz>:0.5 OVER=<e-gmd corpus>:0.25 MIX=<e-gmd corpus>:0.1 STEPS=40000 HOLDOUT=400 EVERY=1000 CONVS=16x3x3/1,16x3x3/2,32x3x3/2 tool/rip-train/pipeline.sh train
```

## 12. The scores

- **`truth <name> <file>`** finds a time map from the longest common subsequence of
  pitches in time order, at the transposition within an octave that matches most, and
  counts a note right when one of its pitch starts within 0.1 s of its place on that map
  (audio's `MatchNotes`). For a rhythm given with no pitches, as
  One More Bite's is, it matches onsets on a fitted time map. Against a MIDI file it also
  counts hits per drum, within half a step, under a fitted stretch and offset.
- **`bench`** renders fresh loops and grades each against what was played: a note
  counts only at its exact step and pitch in the right bar. The total is `(tempo + bars
  + 2 drums + 2 melody + bass) / 7`, so its numbers can't be compared with F1 on
  recordings. `bench ... sampled <dir>` plays NSynth's held-out instruments.
- **Pooling** counts every note once. A mean over recordings would give Für Elise's
  869 notes the weight of the others' 83.
