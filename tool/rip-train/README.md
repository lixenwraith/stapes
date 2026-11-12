# rip-train

The trainer of the rip's models, the frame model and the notes model: PyTorch,
outside the Go module, the twin of the two Go modules that run them. The audio
module's `ripmodel`
([`lixenwraith/audio/ripmodel`](https://github.com/lixenwraith/audio/tree/main/ripmodel))
is each model's shape, the features it hears and the heads the rip's decoders
read; [`lixenwraith/transformer`](https://github.com/lixenwraith/transformer) is
the forward pass it runs on and the weight file this writes
([format](https://github.com/lixenwraith/transformer/blob/main/doc/format.md)).
Where the frame model stands in the pipeline, and why its features are made in Go, is
in [Rip design](../../doc/rip-design.md) §2 and §3; the runs so far and their
numbers are in its §8. [Rip training](../../doc/rip-training.md) follows the data
byte by byte, from a loop's frames and labels through the inputs and targets to
the weight file, with the latest rounds' numbers. This file is the hands-on guide:
setup, the pipeline end to end, and how to turn a run toward the music you care
about.

## What a run is

1. **A corpus.** Stapes renders loops from the patterns and voices registered
   in its session and labels every event in them exactly, because the engine
   played them: the frames a model hears (`loop_NNN.mel`), the events
   (`loop_NNN.json`: frame, end, drum or MIDI note, the wave and genes of the
   voice, the family it sounds as), and `corpus.json` naming the layout.
   Nothing is hand-labelled; the synthesizer is the ground truth, and the
   augmentation is what keeps the models from learning the synthesizer.
2. **A frame model.** `train.py` trains `ripmodel.Spec` on random windows of the loops,
   holds some loops out, scores them every so often and exports the step whose
   onset, drum, pitch and voicing heads score best on them, the heads the rip's
   decoders read.
3. **A grade.** Stapes' `bench` rips fresh loops, clean and augmented, through
   the built-in analysis and through the frame model, and scores each rip against the
   loop's expected rip. A model that scores under the analysis is not ready for
   a recording; a recording is judged by ear, with `rip`, `notes` and `song score`.
4. **Data.** The weight file goes to the stapes root as `ripframe.bin`, beside
   `rip.toml`; `rip` hears with it when it is there and with the analysis when
   it is not. A notes model's, `--pitchnet`'s, goes beside it as `ripnotes.bin`
   and gives the notes.

## Setup

- Go, as `go.mod` names it; a CPU is enough.
- Python 3 with PyTorch:

```bash
cd tool/rip-train
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # a CPU PyTorch, first, or requirements.txt pulls PyPI's
pip install -r requirements.txt
```

`make venv` makes the same `.venv` from the checkout's root, and `PYTHON=` names
another interpreter to `pipeline.sh` and `make smoke`.

A GPU trains an order of magnitude faster (below): `whl/cu130` for a recent
NVIDIA card, and `torch==2.14.* --index-url https://download.pytorch.org/whl/cu126`
for a GTX 10-series (Pascal), whose last build that is; `train.py` picks the
GPU by itself. The CUDA build is 6 GB installed.

Everything below runs from the stapes checkout's root.

## The pipeline

`pipeline.sh` runs the four stages with every knob an environment variable,
and two more, `shard` and `merge`, that spread a run over several machines
([Training in parallel](#training-in-parallel)):

```bash
tool/rip-train/pipeline.sh all            # corpus, train, bench, install, with the defaults below
tool/rip-train/pipeline.sh corpus         # one stage at a time
tool/rip-train/pipeline.sh train
tool/rip-train/pipeline.sh bench
tool/rip-train/pipeline.sh install
```

| Knob | Default | What it does |
|---|---|---|
| `CORPUS` | `~/.local/state/stapes/corpus` | where the loops go, under `$XDG_STATE_HOME` when that is set |
| `LOOPS`, `BARS` | 20000, 4 | loops rendered, bars a loop; 200 loops is a smoke test |
| `PARTS` | 1 | exports at once, each rendering every `PARTS`-th loop on a core of its own; a last export renders any loop they left and writes `corpus.json` |
| `AUG` | `aug` | a room, a mix, voices of random genes near the registered ones, a pedal's ringing in a third of the loops, a soft clip in a third, a compressor and a noise floor in half, each loop rendered off A440 by up to a quarter tone; empty for clean renders |
| `GEN` | `gen` | beats, melodies, bass lines and accompaniments the bank never had, three loops in five per part: a broken chord in eighths or sixteenths in the free slot, and in half the loops a melody of longer notes over it; a third of the generated melodies and accompaniments half 6 to 18 dB softer; three generated loops in ten without their drums and one in seven without its bass; empty for the bank alone |
| `GLIDE` | empty | `glide`: a fifth of the generated notes bend into the note that follows, a seventh scoop into their pitch and three in ten of a loop's oscillator voices sway, each such note labelled with its cents at every frame, which is what teaches the notes model's cents head a glide |
| `WAV` | `nowav` | frames and labels only; empty keeps each loop's WAV too (300 MB per 200 loops) |
| `CODEC` | empty | `codec`: a third of the loops through Opus, MP3 or AAC at 64 to 192 kbps and back, by ffmpeg, which the export then needs |
| `SAMPLED` | empty | a directory of NSynth notes (`family_source_NNN-pitch-velocity.wav`, a partial set too): each tonal line plays on one of its instruments in half the loops, labelled with no voice and by its instrument's family; empty for the engine's voices alone |
| `TEMPOS` | the default spread | `80-120` for six tempos over a range, `96,128,160` for a list; the sequencer's range is 80 to 200 |
| `FOCUS` | none | a stapes script run before the corpus renders: the voices and patterns the loops draw from, see below |
| `STEPS`, `BATCH` | 20000, 8 | training steps and windows a step |
| `DIM`, `HEADS`, `LAYERS`, `FFN` | 256, 8, 4, 1024 | the frame model's width, attention heads, layers and feed-forward width, `train.py`'s `--dim`, `--heads`, `--layers` and `--ffn`; a `CHECKPOINT` or `INIT` of another shape is refused |
| `CONVS`, `TOWERS` | none | convolutions before the input layer, and towers of layers over it, each computing a run of the heads, `train.py`'s `--convs` and `--towers` |
| `MIX` | empty | another corpus and the share of windows drawn from it, `dir:share`, such as recordings with their MIDI files (`corpus <dir> from`) |
| `OVER` | empty | a corpus of drums alone and the share of windows that lay it over a loop with no drums of its own, `dir:share`, at -12 to +6 dB of the loop's power, a hit labelled only where it can be heard, `train.py`'s `--over` |
| `CHANNEL` | empty | a noise bank and the share of windows heard through a small speaker, a room and a phone, `file.npz:share`, `train.py`'s `--channel`; `make_noise.py` builds the bank ([Phones](#turning-a-run-toward-your-music)) |
| `PCEN` | empty | `pcen`: a second input channel, each band's PCEN and their mean, `train.py`'s `--pcen`; a `CHECKPOINT` or `INIT` of the other channel count is refused |
| `HOLDOUT`, `EVERY` | 200, 1000 | loops held out, and how often they are scored and a checkpoint written |
| `VALIDATE` | empty | a corpus of recordings, `corpus <dir> from`'s, cut into fixed windows and scored every `EVERY` steps beside the held-out loops: its best score picks the step `OUT` gets, the decoders' loss or, for a notes model, its strikes' F1 at the decoder's bars, the held-out loss printed beside it, `train.py`'s `--validate`; empty for the held-out loops' pick |
| `OUT`, `CHECKPOINT` | `$CORPUS/ripframe.bin`, `$CORPUS/ripframe.pt` | the weight file, and the checkpoint a stopped run resumes from |
| `ROOT` | `~/.config/stapes` | the stapes root, under `$XDG_CONFIG_HOME` when that is set: `install` copies the weights into it as `ripframe.bin`, and `bench` hears with its `rip.toml` |
| `NOTES` | empty | a notes model's weight file, a `--pitchnet` run's `OUT`, which `bench` and `install` put beside `OUT` as `ripnotes.bin` |
| `BENCH_LOOPS`, `BENCH_BARS` | 12, 4 | the fresh loops `bench` grades on |
| `INIT`, `SHARD` | none, 0 | `shard`: the `.pt` the round starts from, and the worker's stream of windows; it writes `$CORPUS/shard$SHARD.pt` and `.bin` |
| `MERGE_FILES`, `TRIM` | none, 0 | `merge`: the round's shard checkpoints, blank-separated or a glob, and the share of them dropped at each end of each weight before the mean |
| `MERGED` | `$CORPUS/merged.pt` | `merge`: the mean as a `.pt`, the next round's `INIT`; its weight file goes to `OUT` |

A smoke test of the whole path, under ten minutes on a laptop, with a root of
its own so that the weights stapes hears with stay as they are:

```bash
LOOPS=200 STEPS=300 HOLDOUT=20 EVERY=100 BENCH_LOOPS=4 CORPUS=/tmp/smoke ROOT=/tmp/smoke/root tool/rip-train/pipeline.sh all
```

`make smoke` runs the same in a fresh temporary folder, which it names and keeps.

What to expect at scale, on a 2017 laptop (an i7-7700HQ, four cores): an
export renders about 2.5 loops a second on one core and `PARTS=4` 7.0 on the
four, so 40000 loops of four bars take four and a half hours or 95 minutes
(`nowav`; the frames and labels are 9.5 GB); a step of batch 8 at the default
shape takes 1.45 s on the four cores and 0.11 s on its GTX 1060, 0.19 s in a
whole run, which reads the frames from disk and scores the held-out loops, so
40000 steps take two hours on the GPU (the first runs' width 128 took 0.65 s
on the cores); `bench 12 4` grades in a few minutes. Each stage shares the
cores with anything else the machine does.

## Reading a run

- The training lines name each head's loss; the line that matters is the
  held-out one every `EVERY` steps: `step N held-out X, decoders Y (onset ...
  drums ... pitch ... voicing ...)`, with `(best)` on the step `OUT` will get.
  With `VALIDATE` a `validation` line of the same form follows it and picks
  instead: its `(best)` is the step `OUT` gets, the held-out line's its own.
  A frame model's lines give `beats' F1` and `bar lines'` after the decoders'
  loss, each head's peaks over a half against the labelled beats within four
  frames, or `no beat or bar line asked` over a corpus that lists none.
  A training loss that keeps falling after the held-out loss turns is a
  model learning the corpus by heart; more loops, not more steps, is the cure.
- `bench` prints four lines: the analysis and the frame model, on clean and on
  augmented loops, each the mean score with its parts (tempo, bars, drums,
  melody, bass). The fourth run, on 20000 loops rendered at A440, stood at
  0.45 against the analysis's 0.49 clean and 0.51 against 0.48 augmented,
  ahead on drums and behind on pitch; the sixth, at 4× on 40000 loops, at
  0.51 against 0.47 and 0.460 against 0.458, the first to pass both lines,
  ahead on drums and behind on melodies on the recordings' truths: the
  design document's §8.
- A recording: `rip`, then `notes <name>` for the transcript with each note's
  support, `compress`, `refine`, and `song <name> score` for how the song
  sounds against the recording's own frames and how it rips back. The ear is
  the verifier; the recordings in `lixenwraith/scratch` are the held-out set.

## A round on a development machine

What a training round needs, as a hand-off to whoever or whatever runs it,
and what to bring back.

**The machine.** A stapes checkout; Go as `go.mod` names it; Python 3.11 or
later with PyTorch (the
[setup](#setup) above); 10 GB of free disk for a corpus of 40000 loops of four
bars (frames and labels; `WAV=` to keep each loop's audio adds about 60 GB)
and 50 MB for the checkpoint; memory for the labels' events, each window's
frames and targets being read and built as the training draws them; four
cores or more.

**The run**, from the checkout's root, in a corpus directory of its own (a
checkpoint of another shape there is refused) and every knob at its default
unless named:

```bash
. tool/rip-train/.venv/bin/activate   # the setup's PyTorch
export CORPUS=$HOME/ripcorpus6 LOOPS=40000 STEPS=40000 HOLDOUT=400 EVERY=1000 PARTS=4
tool/rip-train/round.sh               # corpus, train, bench, logging to $CORPUS/round.log, resumable
```

`round.sh` renders the corpus unless its `corpus.json`, which the export
writes last, is there, then trains and benches, every line appended to
`$CORPUS/round.log`. Run it again after a stop and it goes on: the corpus from
the loops on disk, whose randomness the export draws all the same, and the
training from `CHECKPOINT`, which carries the best held-out step; either way
the result is an unbroken run's. On the laptop above the corpus takes 95
minutes in four parts and the 40000 steps two hours on its GPU, about sixteen
on its cores.

`install` copies the weights into the stapes root (`ROOT`) as `ripframe.bin`;
do it when the bench says the frame model beats the analysis on both
lines, which the sixth run was the first to do, and judge it by ear on the
recordings, whose truths still favour the analysis (Rip design §8). For
another shape set `DIM`, `HEADS`, `LAYERS`
and `FFN`, with `LOOPS` and `STEPS` raised together, since a bigger model
overfits a small corpus sooner; the held-out line every `EVERY` steps is the
guide, and the run is done when it stops falling.

**What to bring back.** `$CORPUS/ripframe.bin` (the weights), `$CORPUS/ripframe.pt`
(the checkpoint, to resume or to start a round from), the training log with
its held-out lines, the bench's two summary lines, and, if the recordings are
to hand, `stapes -headless -config-dir <root> -s <recipe>` for each recording
in `lixenwraith/scratch` with the weights installed in `<root>/ripframe.bin`,
whose `truth` lines against the rhythm are the number that matters.

## Training in parallel

A run trains on one machine; a round spreads one over several. Every worker
starts from the same weights, trains on a stream of windows of its own, and the
round's checkpoints are averaged into the next round's start:

1. **The start.** The first round's `INIT` is an ordinary run's checkpoint, a
   few hundred steps on one machine, so the workers part from weights past the
   first steps, which move them most; each later round's is the last merge.
2. **The shards.** Each worker runs `shard` from `INIT` with a `SHARD` no worker
   has had in this round or an earlier one (worker k of W in round r takes
   r·W + k), so no two draw the same windows and no round resumes an earlier
   one's checkpoint. A shard writes `$CORPUS/shard$SHARD.pt`, the checkpoint the
   merge reads and a stopped shard resumes from, and `shard$SHARD.bin`, its best
   held-out step.
3. **The merge.** On one machine, with the shards' `.pt` files gathered, `merge`
   averages them weight by weight into `MERGED`, the next round's `INIT`, and
   writes the mean's weight file to `OUT`, which `bench` and `install` take as
   they take a run's.

A smoke round on one machine, two shards from one start, in a minute or two:

```bash
export CORPUS=/tmp/round HOLDOUT=4 EVERY=20
LOOPS=24 tool/rip-train/pipeline.sh corpus
STEPS=20 CHECKPOINT=/tmp/round/start.pt tool/rip-train/pipeline.sh train
INIT=/tmp/round/start.pt SHARD=0 STEPS=40 tool/rip-train/pipeline.sh shard
INIT=/tmp/round/start.pt SHARD=1 STEPS=40 tool/rip-train/pipeline.sh shard
MERGE_FILES='/tmp/round/shard*.pt' tool/rip-train/pipeline.sh merge   # /tmp/round/merged.pt, the next INIT
```

Every worker trains on the same corpus and holds out the same loops, so their
held-out lines compare: copy it, or render it on each from the same checkout
with the same knobs, since `corpus` draws from a fixed seed. A shard starts with
a fresh optimizer and a cosine schedule of its own, so each round is a warm
restart, and the merge takes each shard's last step, not its best held-out one.
`TRIM=0.1` drops each weight's highest and lowest tenth of the files before the
mean, so a worker that diverged does not drag the rest; it counts whole files,
rounded down, so under ten files it drops none. A notes model's files carry the
measure of where a family's name is sure ([below](#parity)), and the merge
carries their mean when every file has one and none when none has. A file that
never measured holds zero, which is no measure to average with one: a merge
whose files disagree on having a measure is refused, the files named, unless
`average.py --sure` gives it one.

**Why independent seeds cannot be averaged.** A network's hidden units can be
permuted, with the weights that read them, without changing what it computes,
and two runs from different random starts settle on different orders: what one
run's unit 7 does, the other's unit 31 does. Their mean, weight by weight,
blends units that mean different things and lands on the ridge of the loss
between the two, worse than either. Runs from one start, trained apart for a
while, keep its order and stay in one basin, where the mean of their weights is
about as good as each: hence one `INIT` a round, and rounds short enough that
the shards stay close. `average.py` refuses files of different shapes, but two
runs of one shape from different starts look alike to it; only the protocol
keeps them apart.

## Turning a run toward your music

The corpus is whatever stapes' session holds when `corpus` runs, so a focus
is a script that loads and applies voices and patterns first, named by `FOCUS`
and run by the pipeline before the export (and before `bench`, so the grade
draws from the same bank). Two are in `focus/`:

- **`focus/vocal.slab`: music with vocals.** A vocal is a line to the rip, pitch
  and onsets per frame, never words; nothing here transcribes lyrics, and the
  models have no notion of them. What the models need is lines that sound like
  voices among its loops, so the focus adds a sung-line voice and a choir pad
  (`focus/vocal.toml`) beside the shipped lead and bass; the generated melodies
  draw any tonal instrument, these among them, and the augmentation redraws
  each voice's genes near the registered one, keeping its vowel and vibrato,
  so the loops hold many voices like them. A voice has a `vowel` (a, e, i, o
  or u: three formants the oscillator passes through) and a vibrato (a rate,
  a depth in cents, a delay before it is full); neither is a gene, so the
  model's voice head and the fit do not search them, and a real vocal
  recording in the held-out set is the check that matters.
- **`focus/plucked.slab`: guitars and keys.** Nylon and steel strings and a
  plucked bass (`focus/plucked.toml`) beside the modal piano.

To write your own:

- **Instruments.** A `[[voice]]` is data ([audio §7](https://github.com/lixenwraith/audio/blob/main/doc/audio.md#7-music-sequencer)): wave, oscillator
  stack, filter, envelopes, or a string or modal bank. Write a sound document
  with the voices and `merge sound file.toml` then `apply name` for each, or
  make them in the REPL with `new voice`, `set` and `apply`; `fit` and `refine`
  find voices for a recording's lines, and `write` exports any voice as a
  document to start from. Every registered tonal voice is one the generated
  melodies and bass lines are played with.
- **Music.** The shipped pattern bank is `audio/bank`'s; `load pattern file.toml`
  or `merge pattern` brings in yours, in the pattern document's form
  (`[[pattern]]` with tracks of steps), and `gen` adds parts the bank never
  had. A bank that is all one genre makes a corpus of that genre; `gen` keeps
  it from being only that bank's few melodies. The rhythm patterns set the
  drums the frame model learns.
- **Tempos and keys.** `TEMPOS` sets the spread; the keys cycle the roots and
  scales the engine knows. A piece whose tempo the sequencer's range does not
  hold reads at a multiple: the origin recording near 74 bpm reads at 148.
- **Rooms, mixes, saturation, tuning.** `AUG` is on by default and is what
  stands between the synthesizer and a recording: leave it on for anything a
  recording will be ripped with; turn it off only to measure the clean case.
- **Real recordings.** One with the MIDI file it was played from is a loop of
  its own: `corpus <dir> from <list.csv> [split <name>] [hours <h>] [codec]`
  reads a CSV with `audio_filename` and `midi_filename` columns (and `split`
  and `duration`, as E-GMD's metadata has them, `family`, the family a
  recording's notes are played by: GuitarSet's guitar, a singer's voice,
  MusicNet's by its instrument labels, empty for a mix of families, and
  `pitch`, the recording's pitch track where its collection has one, a CSV of
  seconds and Hz as vocadito's or GuitarSet's JAMS file, which places its notes
  for the cents head, and `beats`, the file its beats and bar lines are read
  from, GuitarSet's JAMS file or a MIDI file whose bars are the recording's
  as E-GMD's, for the beat and downbeat heads: [Rip
  training](../../doc/rip-training.md) §3), and
  `train.py --mix <dir>:<share>` draws that share of the windows from it,
  labelled by the MIDI's notes, velocities and hits, a tom's or a cymbal's an
  onset of no drum, every hit the drums', the voices unknown, and `--over
  <dir>:<share>` lays a corpus of drums alone over loops with no drums in that
  share of the windows, a hit labelled only where it stands out of the rest.
  Both draw a recording by its length, so a minutes-long one
  counts for its minutes, and a corpus's `skip.txt`, a loop's name
  (`loop_123`) a line, lists those the trainer leaves out, such as recordings
  whose MIDI sits off their audio. A batch run into the same directory joins
  it, numbered after the loops there, so a batch's recordings can be deleted
  once it is in.
  Weights trained on a corpus carry its licence (E-GMD's CC BY 4.0 asks for
  credit; MAESTRO's CC BY-NC-SA 4.0 binds them). A recording without its MIDI
  is held out, judged by ear and by the two-witness check (two recordings of
  one piece, their transcripts compared, as `doc/rip-design.md` §8 does).
- **Phones.** A phone that hears a laptop's speaker cuts the lows and the highs
  and holds a room's noise 7 to 15 dB under the music, where a corpus's sits 40
  to 65 dB down. `CHANNEL=noise.npz:0.5` hears half the windows through a
  speaker, a room and a phone drawn per window ([Rip
  training §6](../../doc/rip-training.md#6-what-may-change-a-windows-frames)).
  The noise bank is two minutes of white noise as mel frames and the band shape
  of a real room, which `python3 tool/rip-train/make_noise.py <quiet.wav>
  <out.npz>` builds from a 16-bit mono WAV of a room with no music, as a phone
  hears it; keep it beside the corpus, out of the checkout.
- **The frame model's shape.** `--dim`, `--layers`, `--heads`, `--ffn` and
  `--attention-window` on `train.py` reshape it; the weight file carries the
  shape, so `ripmodel` loads any of them as long as the inputs and heads are its.
  `pipeline.sh` passes `DIM`, `HEADS`, `LAYERS` and `FFN`; keep a head 32 wide
  (`DIM=384 HEADS=12 FFN=1536`). A checkpoint records its shape, and `train.py`
  refuses to resume or start from one of another, naming both.

## By hand

The stages are three commands; the script only strings them together.

```bash
dir=$HOME/.local/state/stapes/byhand   # outside the checkout: the corpus, the weights, a root to bench in
mkdir -p "$dir/root"
printf 'corpus corpus 20000 4 aug gen nowav\n' > "$dir/corpus.slab"   # read from the script's folder
go run ./cmd/stapes -headless -s "$dir/corpus.slab"
python3 tool/rip-train/train.py --corpus "$dir/corpus" --out "$dir/ripframe.bin" --steps 20000 --batch 8 --every 1000 --holdout 200 --checkpoint "$dir/ripframe.pt"
printf 'bench 12 4\n' > "$dir/bench.slab"
cp "$dir/ripframe.bin" "$dir/root/ripframe.bin"
go run ./cmd/stapes -headless -config-dir "$dir/root" -s "$dir/bench.slab"
mkdir -p ~/.config/stapes && cp "$dir/ripframe.bin" ~/.config/stapes/ripframe.bin
```

`train.py` trains on `--batch` windows of `--window` frames (512, six seconds)
cropped at random from the loops, the next batch built while the device
trains on this one; `--checkpoint` saves every `--every` steps,
with the best held-out step's weights, and resumes from the file when it
exists, `--init` starts from a `.pt`'s weights alone (no optimizer state, step
0), and `--shard-seed` moves the stream the windows are drawn from while
`--seed` keeps the starting weights; `average.py --pt next.pt --out next.bin
[--trim 0.1] a.pt b.pt ...` is the merge. The defaults are `ripmodel.Spec`:
230 inputs, the corpus's 229 bands in dB over the frame's mean and the mean
itself (`ripmodels.features`, derived from the stored frames), width 256, eight
heads, four layers, feed-forward 1024, a
window of 64 frames either side, 3.3 million weights. `--convs
16x3x3/1,32x3x3/2` puts convolutions before the input layer, each its
channels, frames by bands kernel and the pooling over bands; `--towers 2:2,4:10`
splits the layers into towers over the shared input layer, each its layers and
the run of heads it computes in `ripmodels.HEADS` order: the onset and the drums
on two layers, the rest on four. `--pcen` gives the frame model a second channel of
230 inputs: each band's PCEN, its energy over its own average of the last 0.4
s, and their mean ([audio §9](https://github.com/lixenwraith/audio/blob/main/doc/audio.md#9-ripping-a-recording)),
the first convolution taking the two as its channels. Since the average looks
back, `train.py` computes a window's from 128 frames before it (`WARM`, three
of its time constants), so a window starts as settled as it would in a rip.
The losses: onsets, drums, pitches and each pitch's strikes and releases (within a frame of a
note's start and end) as weighted binary cross-entropy, voicing likewise, the
wave as cross-entropy and the voice genes as squared error over frames a note
sounds in, each strike's velocity as squared error where the corpus gives one,
and the family of the highest note or hit sounding as cross-entropy where the
corpus gives one, at half weight and outside the score that picks the step;
and whether a beat of the meter and a bar line fall within a frame, as binary
cross-entropy with no weight, in the score that picks the step
([the beat and the bar line](#the-beat-and-the-bar-line)).

## The beat and the bar line

A frame model has a beat and a downbeat head: whether a beat of the meter, its own
note, and a bar line fall within a frame. Where a weight file has both, audio reads
the tempo, the beats, the meter and the bar lines from them and none of the cues'
rules decides ([audio §9](https://github.com/lixenwraith/audio/blob/main/doc/audio.md#9-ripping-a-recording));
[Rip training](../../doc/rip-training.md) §3, §5 and §8 follow the labels, the
targets and the loss.

- **A rendered loop** needs nothing: its label's `bpm`, `bars` and `meter` are its
  grid, so a corpus on disk teaches the heads as it stands. `corpus ... meters`
  renders a third of the loops in 3/4 or 6/8.
- **A recording** is asked where its list's `beats` column names the file its
  beats are read from, a JAMS file's `beat_position` annotation or a MIDI file
  whose bars are the recording's; the line with `labels` after it writes them into
  a corpus already exported. It is asked from its first listed beat to its last. A
  recording that names none is asked nothing, nor is a shuffled window or one
  `--over` lays drums under.
- **The loss carries no positive weight**: audio reads the heads by rank, which
  neither a weight nor a head's level moves.
- **A model from before the heads** starts a run through `--init`, which grows
  them, every other weight carried over; its own file still loads, its bars read
  by the cues. With `--probe` such a run trains the two heads' rows of the output
  layer alone: every other weight stays the start's bit for bit, as a notes
  model's probe holds it ([below](#parity)), so the file's other heads say what
  the start's did and its rip with the two left out is the start's, and the last
  step is exported. A fine-tune without it moves the heads the decoders already
  read.
- **The lines** give `beats' F1` and `bar lines'`, or `no beat or bar line asked`.

The path at a size four cores train in 42 minutes, which shows that it holds
together and nothing of recordings:

```bash
printf 'corpus corpus 1600 12 gen meters\n' > "$dir/corpus.slab"   # with its audio: the held-out loops are ripped
go run ./cmd/stapes -headless -s "$dir/corpus.slab"
python3 tool/rip-train/train.py --corpus "$dir/corpus" --out "$dir/ripframe.bin" --checkpoint "$dir/ripframe.pt" \
    --steps 5000 --dim 128 --heads 4 --layers 4 --ffn 512 --pcen --holdout 160 --every 500
```

Its exported step reads `beats' F1 0.763, bar lines' 0.631` on the 160 loops held
out, 0.877 and 0.728 on those with drums and 0.535 and 0.359 on those without.
Ripped through the file on audio v0.24.0, 56 of the 160 have their bars right with
the heads, 124 with the same weights' heads left out and 109 through the analysis: a
head this weak cost the bars, since v0.24.0 took its word alone; since v0.24.1 audio
reads the heads by rank and leaves a clip they give no reading of to the cues ([Rip
training §8](../../doc/rip-training.md#the-two-heads-on-a-small-run) has the table
by meter and what the heads say under it). So read a round's bars with the heads
and without before its file is installed. The heads are left out of a rip by a
`RipFrontEnd` that sets the frames' `Beat` and `Downbeat` nil, in a scratch
program over audio's `Ripper` and `MatchBars`, or by a file of the same step
exported with ten heads, which stapes itself then benches; the two rip alike.

## A judge

Which strikes of the notes model are notes is the accounting's verdict, two
installed scores. A judge rules in their place: a small network a stage, the
notes the decoder reported and the strikes it dropped, over a strike's row
([audio §9](https://github.com/lixenwraith/audio/blob/main/doc/audio.md#9-ripping-a-recording)).
It is fitted on the rows one notes model gives, needs no round, and audio
refuses it beside any other notes model. None ships yet: one is installed on
its readings with a collection left out ([Rip design](../../doc/rip-design.md)
§7).

```bash
printf 'rows rows from truths.txt frames frames\nrows rows from corpus\n' > "$dir/rows.slab"   # read from the script's folder
go run ./cmd/stapes -headless -config-dir "$dir/root" -s "$dir/rows.slab"                      # the root's models, never its judge
python3 tool/rip-train/judge.py --rows "$dir/rows" --leave-out guitar --out "$dir/root/ripjudge.bin"
```

- **What a fit may take.** Licensed collections with truths, renders, and the
  engine's own mixes (a corpus written with its audio, `WAV=` empty). Never a
  full track such as a commercial song: no licence covers training on it, so
  it only measures, read under a judge with `rip` and `truth`, or listed under
  a condition every fit leaves out.
- **Rows.** `rows <dir> from <list>` rips each recording of a truth list, the
  lines `tune ... truth` reads, and `rows <dir> from <corpus dir>` each loop
  of a corpus that kept its audio, against its label, the corpus one condition
  under its folder's name. Each strike over a half is a row, as the decoder
  reads it:
  - `<checksum>.rows`, a recording's rows, laid out as a loop's frames are:
    the count, the row's size, then `float32` rows, little-endian;
  - `<checksum>.json`, its `source`, `condition`, `given` (the truth's notes),
    `unanswered` (those no strike answers) and `strikes`, a row each: `frame`,
    `midi`, `reported` (its stage), `held` (a held note the accounting keeps
    unjudged), `right` (a truth note of its pitch starts within the matcher's
    window of it; a truth note answers one strike, the nearest, a reported
    note before a dropped strike) and `kept` (the installed verdict);
  - `rows.json`, the row's version and size and the notes weights' checksum.
    A directory holds one kind, and `judge.py` refuses directories of two.

  A recording already written is kept, `part k/n` takes every `n`-th from the
  `k`-th so parts run side by side, and `frames <dir>` keeps what the models
  heard between runs, as `tune ... truth` does.
- **The fit**, one stated procedure, nothing picked by a held-out score. A
  stage's rows are standardized by their mean and spread (a value that never
  varies is left as it is); the network is two hidden layers of 256 and 64
  with ReLU, dropout of .2 after the first, and one output; the loss is binary
  cross-entropy against `right`; AdamW at 1e-3 with weight decay 1e-4 under
  PyTorch's `OneCycleLR` to a peak of 2e-3, which sets the rate at every step
  (from a twenty-fifth of the peak up, then down); batches of 2048, twelve
  passes, seed 0. The held notes are left out of the reported stage.
- **Left out.** `--leave-out <condition>...` keeps conditions out of the fit
  and counts each by pitch and onset, as `truth`'s `notes` row counts, under
  the installed verdicts, under the judge's, and under a right verdict on
  every row, which is what the rows could give at best. A run of the path
  alone, 360 rendered loops fitted on and 40 of another seed left out,
  through models trained an hour on CPU, so it says nothing of a judge's
  worth on recordings:

  ```
  108933 row(s) of 544 (version 1) read through the notes weights c2f75c44fd18b2e3; fitting on fit
  reported: 6403 row(s), 0.527 of them right; loss a pass 0.697 0.672 0.611 0.509 0.466 0.436 0.420 0.405 0.393 0.383 0.378 0.376
  dropped: 90773 row(s), 0.127 of them right; loss a pass 0.478 0.322 0.287 0.271 0.262 0.255 0.248 0.240 0.235 0.229 0.225 0.223
  held: 40 recording(s), 2838 note(s) given, 1262 of them no strike answers; 11664 strike(s), 801 reported
    installed  423 heard as given, given 2838, heard 761, F1 0.235
    judge      740 heard as given, given 2838, heard 1028, F1 0.383
    right      1576 heard as given, given 2838, heard 1580, F1 0.713
  ```

  A count from rows is near `truth`'s decode of the same recordings, and
  `truth` is the reading. Those 40 loops decoded read 424 heard as given of
  762 under the installed scores (F1 0.236) and 759 of 1059 under the judge
  (0.390). Under the installed scores a strike listed is the strongest of its
  pitch within `strike_gap`, and carries the verdict of the candidates near
  it: two the scores returned around one stronger strike are one row, and a
  returned note may start up to the gap from the strike its row is labelled
  at. Under a judge a dropped strike is measured against the notes the judge
  kept, where its row was written against those the installed scores kept.
- **The file.** `--out` writes `ripjudge.bin`, which stapes loads from the
  root beside `ripnotes.bin`: `rip`, `truth`, `tune` and `live` then run under
  it, and the session and `stat` name it. It names the row's version and the
  notes weights it was fitted on, so new weights or a new row refuse it by
  name and the installed scores judge until it is fitted again; remove the
  file to go back to them.

## Parity

The Go forward pass and the frame model agree to `float32` rounding: exact GELU,
layer norm with eps 1e-5, ALiBi slopes of 2^(-8h/H) per head on the frame
distance, attention banded to the window, and the output layer's slices as the
heads, in `ripmodels.HEADS` order. `export` writes the tensors by the names and in
the order `lixenwraith/transformer` reads them, with an FNV-1a 64 trailer the
loader checks ([format](https://github.com/lixenwraith/transformer/blob/main/doc/format.md)).
`--pitchnet` builds the convolutions-only model of [Rip training](../../doc/rip-training.md)
§7, and `--out` gets its weight file, `ripnotes.bin` in the stapes root, beside the
`ripframe.bin` it gives the notes for. `average.py --out ripnotes.bin a.pt` turns one `.pt`
of either model into its file. `--shuffle <share>` and
`--extra <dir>:<share>,...` are `train.py` flags with no `pipeline.sh` setting,
as are `--cell <n>`, the notes model's width (its channels at each cell, 32,
which its weight file carries), and `--dip <share>`, the share of the channel's
windows given a dip of 6 to 20 dB centred 100 to 500 Hz, as a phone across a room
hears. `--family-holdout <family>` tells no head a family's name: its notes stay
in the run as sound, and a notes model's lines say what their surest name reaches,
the measure a name must pass to be given (below). A checkpoint from before
a setting existed loads with that setting's default, and `--init` takes one of the
frame model from before the family head or the beat and downbeat heads, or of the
notes model from before the cents head or the heads of what plays each pitch: every
weight carries over and the head's rows start fresh, though resuming one is refused.

A notes model has a cents head: for each pitch, how far its sound sits from the
cell's centre and the error it states for that, taught where a sound the labels
place lies within a semitone of the cell: an engine voice's note, from the loop's
`tuning` and a moving note's `cents`, or a recorded note that lists its `cents`,
which one does where its list's `pitch` column names the recording's pitch track
([Rip training](../../doc/rip-training.md) §3, §5 and §8). `corpus ...
glide` is the word that makes notes move: bends into the next note, scoops and
vibratos. A corpus without it still teaches the head each loop's tuning; its labels
written before `cents` take a voice that sways, the vocal focus's, or a MIDI file's
bent note for a steady one, so write that corpus's labels again: its line, which
`pipeline.sh` keeps in `$CORPUS/corpus.slab`, with `labels` after it, which renders
nothing and takes seconds. The held-out line and
`--validate`'s say what the head reads, in cents, all the placed readings first,
then those of a cell a glide is in, of a cell a pitch holds in, and of the cell
beside a sound; of a sounding cell the labels do not place, only the error it
states (`unplaced 31.0 stated`). A recording with no pitch track is unplaced
throughout, and shows how sure the head is of real sound and not how right. One
whose list names its track prints the three rows as an engine loop does, a frame
its track does not voice left unplaced; since no track holds a note still, a
recorded note's readings are gliding where its track spans half a cell, 50 cents,
over the note, and steady under that. A small model's after 2,500 steps on 360
loops of a `glide` corpus:

```
cents 14.5 off, 21.4 stated, 63% within; gliding 22.5 off, 25.1 stated, 54% within; steady 11.9 off, 19.3 stated, 69% within; beside 16.0 off, 22.8 stated, 59% within
```

`off` is the median miss, `stated` the median error the head states, and `within`
the share of readings whose miss is within the error stated for them: 63% for a
head that states its mean miss of misses spread as a Laplace's. A share well under
that on the gliding readings is a head too sure of itself where the decoder leans on
it, since audio joins two cells only where their readings agree within the stated
errors, with no margin on top. A head that knows nothing of a sound reads a median
25 cents off, so the line above is a head that has a loop's tuning and little of a
glide's motion yet: it states as much there, and the decoder joins nothing on its word.
`--even-glides` gives a batch's gliding readings half the cents loss and half the stated
error's wherever it holds them beside steady ones, which outnumber them many times over.

A notes model says what plays each pitch: three more heads a cell, the family of
the eight names, the wave and the 32 genes, taught at the cells one note alone
sounds in ([Rip training](../../doc/rip-training.md) §5, §7 and §8). Audio reads
them into each note's voice, and names a note's family only where its share over
the note passes the measure the weight file carries. A run measures it by holding
a family out and ships it with none held out:

```bash
python3 train.py --pitchnet ... --family-holdout guitar --out held.bin   # its held-out line: guitar's notes reach R
python3 train.py --pitchnet ... --sure R --out ripnotes.bin              # every name taught, the measure carried
```

`--probe` trains those three heads alone on the notes model `--init` gives: every
other weight stays the start's bit for bit (the gradient is masked to the heads' rows
of the cells' output layer, which holds every cell head, and the rest are put back
after each step against AdamW's decay), the probed notes model carries no cents head,
and the last step is exported, since no score that picks a step moves. Its file rips as its start's
does, so only the new heads need reading. A frame model's probe trains its beat and
downbeat heads the same way, the rows of its one output layer ([the beat and the bar
line](#the-beat-and-the-bar-line)); a frame model in towers, whose heads lie over
several output layers, is refused.

The held-out run's last lines give the reach; the shipped run prints what its file
carries (`a family naming a note past a share of R`), and a file with no measure
says so and names no note. `average.py --sure` gives a merge its measure; without
it the merge carries the mean of its files' when each carries one, and refuses
files of which only some do, since a file with none holds zero and would pull the
mean under anything a run measured. On a held-out or validation line,
after the cents:

```
families F of N notes (piano F of N, ...), F of the N under another, waves F, genes G off; guitar's N notes, held out, reach R, M at the median: P% of the others' notes pass it, Q% of them right
```

The share named their family is over the notes with a family to ask, read as
audio reads a note: its cells' say over its frames, weighed by salience, at the
cells among the frame's twelve most salient. `under another` is that share over
the notes a higher note sounds over, where the frame's family head never looked;
`waves` the share of the engine's notes given their wave; `genes` the RMS miss
over a note's 32 genes. The held-out family's part gives the share its notes'
surest name reaches at the 95th percentile of them, which is the measure, the
median, how many of the other families' notes pass the measure and how many of
those are named right: a measure most named notes fall under is a model that
tells too little apart yet, and its file names few notes rather than wrong ones.

The trainer's rules for what plays each pitch and for the beat and the bar line
are tests of their own, with no corpus and in two seconds, on the CPU whatever
the machine has: `python3 -m unittest discover tool/rip-train`.

It writes the file's version 3, the channels after the inputs, which needs
transformer v0.2.0 or later (audio v0.2.0): a stapes built on an older audio
refuses it, of one channel or two. A frame model's file lists the beat and
downbeat heads, which needs audio v0.24.0: an older one refuses it by its count
of heads, and one from before them still loads in any. `--pitchnet`'s file is version 5, a cell net
with its long convolutions listed, which needs transformer v0.5.0 (audio v0.8.2);
the cents head changes neither, being one more per-pitch head of two values, and
an audio before v0.21.0 reads such a file's other heads and hears no glide; nor
do the heads of what plays each pitch and the measure, a frame head of one value
whose weights are zero and whose bias is the measure, and an audio before v0.23.0
reads such a file and tells no note's voice.
A change to either side is a change to both: check a random model's outputs on
a corpus loop from the two before trusting a trained file, as the module's
[agent notes](https://github.com/lixenwraith/transformer/blob/main/doc/agent-notes.md)
describe.
