# rip-train

The trainer of the rip's models, the frame model and the notes model: PyTorch,
outside the Go module, the twin of the two Go modules that run them. The audio
module's `ripmodels`
([`lixenwraith/audio/ripmodels`](https://github.com/lixenwraith/audio/tree/main/ripmodels))
is each model's shape, the features it hears and the heads the rip's decoders
read; [`lixenwraith/transformer`](https://github.com/lixenwraith/transformer) is
the forward pass it runs on and the weight file this writes
([format](https://github.com/lixenwraith/transformer/blob/main/doc/format.md)).
Where the frame model stands in the pipeline, and why its features are made in Go, is
in [Rip design](../../doc/rip-design.md) §2 and §3. [Rip
training](../../doc/rip-training.md) follows the data byte by byte, from a loop's
frames and labels through the inputs and targets to the weight file. This file is
the hands-on guide:
setup, the pipeline end to end, and how to turn a run toward the music you care
about.

## What a run is

1. **A corpus.** Stapes renders loops from the patterns and voices registered
   in its session and labels every event in them exactly, because the engine
   played them: the frames a model hears (`loop_NNN.bands`), the events
   (`loop_NNN.json`: frame, end, drum or MIDI note, the wave and genes of the
   voice, the family it sounds as), and `corpus.json` naming the layout.
   Nothing is hand-labelled; the synthesizer is the ground truth, and the
   augmentation is what keeps the models from learning the synthesizer.
2. **A frame model.** `train.py` trains `ripmodels.FrameSpec` on random windows of the loops,
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

A GPU trains much faster: `whl/cu130` for a recent
NVIDIA card, and `torch==2.14.* --index-url https://download.pytorch.org/whl/cu126`
for a GTX 10-series (Pascal), whose last build that is; `train.py` picks the
GPU by itself. The CUDA build is 6 GB installed.

Everything below runs from the stapes checkout's root.

## The pipeline

`pipeline.sh` runs a stage a word, several in a row, with every knob an
environment variable: the four that make a frame model, `notes` for the notes
model, the six that fit a pair and say where the rip stands ([the fit of a
pair](#the-fit-of-a-pair)), and `shard` and `merge`, which spread a run over
several machines ([Training in parallel](#training-in-parallel)):

```bash
tool/rip-train/pipeline.sh all            # corpus, train, bench, install, with the defaults below
tool/rip-train/pipeline.sh corpus         # one stage at a time
tool/rip-train/pipeline.sh train
tool/rip-train/pipeline.sh notes          # the notes model, into NOTES
tool/rip-train/pipeline.sh bench
tool/rip-train/pipeline.sh install
tool/rip-train/pipeline.sh line           # the corpus's script, printed: nothing renders
tool/rip-train/pipeline.sh rows judge fit readback table keep   # the fit of a pair
```

| Knob | Default | What it does |
|---|---|---|
| `CORPUS` | `~/.local/state/stapes/corpus` | where the loops go, under `$XDG_STATE_HOME` when that is set |
| `LOOPS`, `BARS` | 20000, 4 | loops rendered, bars a loop; 200 loops is a smoke test |
| `PARTS` | 1 | exports at once, each rendering every `PARTS`-th loop on a core of its own; a last export renders any loop they left and writes `corpus.json` |
| `AUG` | `aug` | a room, a mix, voices redrawn near the registered ones in sound and no pedal, a soft clip in a third, a compressor and a noise floor in half, each loop rendered off A440 by up to a quarter tone; empty for clean renders |
| `GEN` | `gen` | every loop but an overlap arranged over the whole keyboard in place of the bank's patterns: a key and its chords, a line alone to six parts (drums, bass, a keyboard's two hands, a guitar's strokes, a pad, an arpeggio, a melody from its motif, its double, a counter-line), parts entering and dropping out, swing, triplets, rests and breaks, played with accents, swells, a feel off the grid and a drifting tempo ([Rip training §3](../../doc/rip-training.md#arranged-loops)); empty for the bank alone |
| `METERS` | `meters` | the arranged loops in 4/4, 3/4, 6/8, 12/8, 2/4, 7/8 and 5/4 by share, three in ten starting part-way into a bar; empty for 4/4 from a bar line |
| `OVERLAPS` | `overlaps` | a quarter of the loops single notes of one to eight instruments overlapping in no meter and labelled with no beat; `overlaps <share>` for another share, empty for none |
| `GLIDE` | empty | `glide`: a fifth of an arranged loop's melody, double, counter-line and bass notes bend into the note that follows, a seventh scoop into their pitch and three in ten of a loop's oscillator voices sway, each such note labelled with its cents at every frame, which is what teaches the notes model's cents head a glide |
| `WAV` | `nowav` | frames and labels only; empty keeps each loop's WAV too (300 MB per 200 loops) |
| `CODEC` | empty | `codec`: a third of the loops through Opus, MP3 or AAC at 64 to 192 kbps and back, by ffmpeg, which the export then needs |
| `SAMPLED` | empty | a directory of NSynth notes (`family_source_NNN-pitch-velocity.wav`, a partial set too): each tonal part plays on one of its instruments in half the loops, and an overlap draws among them and the engine's voices alike, labelled with no voice and by its instrument's family; empty for the engine's voices alone |
| `TEMPOS` | the default spread | `80-120` for six tempos over a range, `96,128,160` for a list; the sequencer's range is 80 to 200 |
| `SEED` | empty | `seed n`: every loop drawn from seed `n`; empty for the verb's own, 1 |
| `FOCUS` | none | a stapes script run before the corpus renders: the voices and patterns the loops draw from, see below |
| `STAPES` | empty | a stapes binary every stage runs, such as the build a corpus was rendered by ([the round's corpora](#the-rounds-corpora)); empty for the checkout's |
| `STEPS`, `BATCH` | 20000, 8 | training steps and windows a step |
| `DIM`, `HEADS`, `LAYERS`, `FFN` | 256, 8, 4, 1024 | the frame model's width, attention heads, layers and feed-forward width, `train.py`'s `--dim`, `--heads`, `--layers` and `--ffn`; a `CHECKPOINT` or `INIT` of another shape is refused |
| `CONVS`, `TOWERS` | none | convolutions before the input layer, and towers of layers over it, each computing a run of the heads, `train.py`'s `--convs` and `--towers` |
| `MIX` | empty | another corpus and the share of windows drawn from it, `dir:share`, such as recordings with their MIDI files (`corpus <dir> from`) |
| `OVER` | empty | a corpus of drums alone and the share of windows that lay it over a loop where nothing is struck, `dir:share`, at -12 to +6 dB of the loop's power, a strike labelled only where it can be heard, `train.py`'s `--over` |
| `CHANNEL` | empty | a noise bank and the share of windows heard through a small speaker, a room and a phone, `file.npz:share`, `train.py`'s `--channel`; `make_noise.py` builds the bank ([Phones](#turning-a-run-toward-your-music)) |
| `PCEN` | empty | `pcen`: a second input channel, each band's PCEN and their mean, `train.py`'s `--pcen`; a `CHECKPOINT` or `INIT` of the other channel count is refused |
| `HOLDOUT`, `EVERY` | 200, 1000 | loops held out, and how often they are scored and a checkpoint written |
| `VALIDATE` | empty | a corpus of recordings, `corpus <dir> from`'s, cut into fixed windows and scored every `EVERY` steps beside the held-out loops: its best score picks the step `OUT` gets, the decoders' loss or, for a notes model, its strikes' F1 at the decoder's bars, the held-out loss printed beside it, `train.py`'s `--validate`; empty for the held-out loops' pick |
| `OUT`, `CHECKPOINT` | `$CORPUS/ripframe.bin`, `$CORPUS/ripframe.pt` | the weight file, and the checkpoint a stopped run resumes from |
| `ROOT` | `~/.config/stapes` | the stapes root, under `$XDG_CONFIG_HOME` when that is set: `install` copies the weights into it as `ripframe.bin`, and `bench` hears with its `rip.toml` |
| `NOTES` | empty | a notes model's weight file: `notes` trains it by `train`'s knobs with `--pitchnet`, its checkpoint beside it as `.pt`, and `bench`, `install` and the fit put it beside `OUT` as `ripnotes.bin` |
| `DILATIONS` | empty | `notes`: the notes model's long convolutions, their frames apart, `train.py`'s `--dilations` |
| `LR`, `EXTRA`, `CARRY` | empty | `train.py`'s `--lr`; `--extra`, more corpora and the share of windows each draws, `dir:share` by commas, such as the recorded sets; and `carry` for `--carry`, a pass that carries on from `INIT` |
| `JUDGE`, `SET` | empty | `install`: a judge's file and a `rip.toml`, one of the fit's sets, put in the root as `ripjudge.bin` and `rip.toml` |
| `KINDS` | empty | `install`, `readback`: the sets a kind of sound is ripped under, `kind=file` blank-separated (`voice=$FIT/sets/voice.toml`), put in the root as `rip.<kind>.toml` |
| `BENCH_LOOPS`, `BENCH_BARS` | 12, 4 | the fresh loops `bench` grades on |
| `INIT`, `SHARD` | none, 0 | the `.pt` a training starts from, `train.py`'s `--init`: a shard's round, or a pass's, the pass before it; and `shard`'s stream of windows, which writes `$CORPUS/shard$SHARD.pt` and `.bin` |
| `MERGE_FILES`, `TRIM` | none, 0 | `merge`: the round's shard checkpoints, blank-separated or a glob, and the share of them dropped at each end of each weight before the mean |
| `MERGED` | `$CORPUS/merged.pt` | `merge`: the mean as a `.pt`, the next round's `INIT`; its weight file goes to `OUT` |

`corpus` keeps its script in `$CORPUS/corpus.slab`, its line the words the knobs
set a blank apart, and each part's beside it, the line ending `part k/n`; `line`
prints the script. The exports run on an empty root and config folder, so no
model loads and no override of the bank's documents is read: a corpus is its
line's and its build's alone.

A smoke test of the whole path, under ten minutes on a laptop, with a root of
its own so that the weights stapes hears with stay as they are:

```bash
LOOPS=200 STEPS=300 HOLDOUT=20 EVERY=100 BENCH_LOOPS=4 CORPUS=/tmp/smoke ROOT=/tmp/smoke/root tool/rip-train/pipeline.sh all
```

`make smoke` runs the round whole in a fresh temporary folder, which it names
and keeps: the two models, then 24 loops rendered with their audio and heard,
their rows and judge, a search, the read-back, its table, the files kept and
the pair installed with its judge.

Each stage shares the cores with anything else the machine does, a run most
of all: its batches are
built on the cores beside the device, a core a builder (`--builders`, [by
hand](#by-hand)), and its steps a second follow what they get.

## The fit of a pair

A pair of weights is fitted and read by six stages, each going on from what it
left, in `FIT`: a judge of its strikes, its thresholds a kind of sound at a
time, and one table of every set on recordings no fit saw, which is where the
rip stands. Nothing here trains a model.

```bash
export OUT=ripframe.bin NOTES=ripnotes.bin FIT=fit                    # the pair, and where its fit goes
export ROWS="judge-guitar.tsv judge-voice.tsv corpus-kept-with-audio"  # what the judge is fitted on
export TRUTH=search.tsv READBACK=readback.tsv START=rip.toml           # the search's list, the read-back's, the set in use
tool/rip-train/pipeline.sh rows judge fit readback table keep
```

- `rows` writes each of `ROWS` through the pair as `rows <dir> from` does,
  into `$FIT/rows`, and `judge` fits [a judge](#a-judge) on them, into
  `$FIT/root`; one that stands there is kept.
- `fit` searches each kind apart: a kind is a condition of `TRUTH`'s lines
  (`wav<TAB>truth<TAB>condition<TAB>words`, `tune ... truth`'s), and its search
  is `tune POOL GENERATIONS truth` over its lines from `START`, weighing `WEIGH`
  (the ends, so a search of the notes cannot move them blind). The set it
  leaves is `$FIT/sets/<kind>.toml`, empty where nothing beat the built-in
  thresholds. A set stands while what it was searched from does (the root,
  the start, the kind's lines, the search's words), and is searched again
  when one of them changes.
- `readback` reads `READBACK` under the models alone, under their judge,
  under `START`, under each kind's set that is not the start's own, and under
  `KINDS`, each recording under the set of the kind stapes reads it as, which
  is what `rip` does with that root:
  `tune 2 1 truth`, the smallest search, whose figures before are the set's.
  A reading stands while its root and the list do, and one whose set is gone
  leaves the logs.
- `table` writes `$FIT/table.txt` from those logs (`table.py`): a kind a
  block and a set a row, notes, ends (their share and the notes whose end is
  right), loudness, drums, bars, parts, key and chords, and before the rows
  each fitted set's falls of more than 0.005 against the start, or the judge
  where no start is named.
- `keep` copies the pair, its checkpoints, the judge, the sets and the table
  under `$WEIGHTS/$NAME`, a file not there yet or changed, each with its line
  in `$WEIGHTS/MANIFEST.txt`. `install` takes `JUDGE`, `SET` and `KINDS` by
  name: which sets a root hears with is a decision, not a stage's.

A list's paths are absolute, since `fit` cuts it a kind into `FIT`. `TRUTH`
and `READBACK` may name a corpus directory that kept its audio: its loops of
four seconds or more are listed against their labels under the folder's name.
A new pair takes a new `FIT`: a judge and the kept frames are their pair's.
Kept frames are their build's too: a build that reads a head an older one
did not keep decodes them as if the head were not there, so `FRAMES` kept
before such a change are heard again.

| Knob | Default | What it does |
|---|---|---|
| `FIT` | `$CORPUS/fit` | where a pair's fit goes: its root, rows, lists, sets, logs and table |
| `ROWS` | empty | `rows`: truth lists and corpus directories that kept their audio, blank-separated |
| `LEAVE_OUT` | empty | `judge`: conditions its fit leaves out and counts, `judge.py`'s `--leave-out` |
| `TRUTH`, `READBACK` | empty | the list `fit` searches on, a kind of sound a condition, and the list `readback` reads every set on |
| `FRAMES` | `$FIT/frames` | what the models heard of each recording, kept between runs |
| `START` | empty | the `rip.toml` every search starts from and the read-back reads beside the sets; empty for the built-in thresholds |
| `POOL`, `GENERATIONS` | 12, 5 | a search's sets a generation and its generations |
| `GUARD`, `HOLD` | the verb's | the score a set may take from a condition, and the GB of recordings held between rips |
| `WEIGH` | `ends` | the readings a search weighs beside the notes, by commas; empty for the notes alone |
| `GENES` | empty | the genes a search moves, by commas; empty for all the list can score |
| `WEIGHTS`, `NAME` | empty, `CORPUS`'s name | `keep`: the folder the round's files go under, and the round's folder in it |

## Reading a run

- A corpus, before a round trains on it: `python3 tool/rip-train/datacheck.py
  <dir>...` reads a sample of its loops and prints a row a corpus (a row a
  source with `GROUP=1`): files whole and finite, labels inside their loop,
  notes outside the piano's keys, doubled notes, notes and hits in their loop's
  own quiet (a frame's energy in dB near the loop's quietest), and the lag in
  frames from labelled starts, hits and beats to where the sharp plane rises.
  It refuses a corpus whose manifest states another layout than the band
  frames'. Hand labels read 0 or 1; a tool's labels read its offset
  ([agent notes](../../doc/agent-notes.md#corpus-and-bench)).
- The training lines name each head's loss; the line that matters is the
  held-out one every `EVERY` steps: `step N held-out X, decoders Y (onset ...
  drums ... pitch ... voicing ...)`, with `(best)` on the step `OUT` will get.
  With `VALIDATE` a `validation` line of the same form follows it and picks
  instead: its `(best)` is the step `OUT` gets, the held-out line's its own.
  A frame model's lines give `beats' F1` and `bar lines'` after the decoders'
  loss, each head's peaks over a half against the labelled beats within four
  frames, or `no beat or bar line asked` over a corpus that lists none.
  Over loops of placed starts either model's lines then give `placed starts' F1
  ... within R frames`, the onset head's peaks over a half against those starts
  within the corpus's reach, and `their strikes'`, the strike head's at the
  strikes' bars against them at their pitch.
  A pass that carries on (`--carry`) reads its start before a step; each held-out
  line then says, before its parts, the rate since the last reading and what the
  rule did (`; rate 1.0e-04 fine: gained, -0.012 on the reading kept at step 3000,
  noise 0.008`), the validation line the rate, and the pass's last line what it
  carries on. Its `decoders` is the sum of the heads audio reads from its model in
  a pair, which a line before its start names: a notes model's pitch, strike and
  release, a frame model's onset, drums, beat and downbeat; not yet a notes
  model's onset, which audio reads beside its strikes.
  A training loss that keeps falling after the held-out loss turns is a
  model learning the corpus by heart; more loops, not more steps, is the cure.
- `bench` prints four lines: the analysis and the frame model, on clean and on
  augmented loops, each the mean score with its parts (tempo, bars, drums,
  melody, bass).
- A recording: `rip`, then `notes <name>` for the transcript with each note's
  support, `compress`, `refine`, and `song <name> score` for how the song
  sounds against the recording's own frames and how it rips back. The ear is
  the verifier; the recordings in `lixenwraith/scratch` are the held-out set.

## A round on a development machine

What a training round needs, as a hand-off to whoever or whatever runs it,
and what to bring back.

**The machine.** A stapes checkout; Go as `go.mod` names it; Python 3.11 or
later with PyTorch (the
[setup](#setup) above); 63 GB of free disk for a corpus of 40000 loops of four
bars at the default tempos (1.5 MB of frames and 31 KB of labels a loop; `WAV=`
to keep each loop's audio adds about 50 GB)
and 50 MB for the checkpoint; memory for the labels' events, each window's
frames and targets being read and built as the training draws them; four
cores or more.

**The run**, from the checkout's root, in a corpus directory of its own (a
checkpoint of another shape there is refused) and every knob at its default
unless named:

```bash
. tool/rip-train/.venv/bin/activate   # the setup's PyTorch
export CORPUS=$HOME/corpus LOOPS=40000 STEPS=40000 HOLDOUT=400 EVERY=1000 PARTS=4
tool/rip-train/round.sh               # corpus, the models, the bench and the fit, logging to $CORPUS/round.log, resumable
```

`round.sh` renders the corpus unless its `corpus.json`, which the export
writes last, is there, then trains and benches, the notes model too where
`NOTES` names one, and runs [the fit](#the-fit-of-a-pair)'s stages whose lists
are named (`ROWS`, `TRUTH`, `READBACK`, `WEIGHTS`), every line appended to
`$CORPUS/round.log`. Run it again after a stop and it goes on: the corpus from
the loops on disk, whose randomness the export draws all the same, and the
training from `CHECKPOINT`, which carries the best held-out step, drawing from
its step the batches the unbroken run drew; either way the result is an
unbroken run's.

`install` copies the weights into the stapes root (`ROOT`) as `ripframe.bin`;
do it when the bench says the frame model beats the analysis on both
lines, and judge it by ear on the recordings. For another shape set `DIM`,
`HEADS`, `LAYERS`
and `FFN`, with `LOOPS` and `STEPS` raised together, since a bigger model
overfits a small corpus sooner; the held-out line every `EVERY` steps is the
guide, and the run is done when it stops falling.

**What to bring back.** `$CORPUS/ripframe.bin` (the weights), `$CORPUS/ripframe.pt`
(the checkpoint, to resume or to start a round from), the training log with
its held-out lines, the bench's two summary lines, the fit's `table.txt`
where its lists were named, and, if the recordings are
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

## The round's corpora

A corpus is its line, the build that ran it and, under `sampled`, the notes
folder it names: rendered again from the three, whole or in parts, it is the same
bytes, so a corpus no pass reads may go once its line is here. The refresh's
round, `<c>` the folder of its corpora and `<n>` the notes folder:

- `gen-dad4f44`, passes 2 and 3: deleted, its loops 19800 to 19999 kept apart as
  the round's 200 held-out loops; by v0.36.0 (dad4f44), whose bytes v0.37.0 renders.
  - `corpus <c>/gen-dad4f44 20000 4 aug gen meters overlaps sampled <n> seed 1`
- `gen-feaef3e-s<k>`, k from 14 to 31, pass k - 10's own (pass 21 the notes
  model's alone): deleted but s30, the frozen pair's; by v0.37.0 (feaef3e).
  - `corpus <c>/gen-feaef3e-s<k> 10000 4 aug gen meters overlaps sampled <n> seed <k>`
- `clean-feaef3e`, pass 1 alone, then a fifth of every pass's windows: kept; by v0.37.0.
  - `corpus <c>/clean-feaef3e 8000 4 gen meters overlaps 0.75 sampled <n> seed 3`
- `engine-dad4f44`, no pass of the chain: kept; by v0.36.0.
  - `corpus <c>/engine-dad4f44 6000 4 aug gen meters overlaps seed 2`
- `fit-gen-s1001`, the judge's rows, and `fit-gen-s1002`, the thresholds' search,
  training nothing: kept; by v0.37.0.
  - `corpus <c>/fit-gen-s1001 300 4 aug gen meters overlaps sampled <n> seed 1001`
  - `corpus <c>/fit-gen-s1002 100 4 aug gen meters overlaps sampled <n> seed 1002`

One is rendered again from this checkout:

1. **The notes folder**, for a line with `sampled`: every note of NSynth's training
   split, 953 instruments in 11 families, 289,205 notes, by
   [`datasets/nsynth_families.py`](../rip-lab/README.md#datasetsnsynth_familiespy),
   `<audio>` the split's `audio` folder:
   `tool/rip-lab/datasets/nsynth_families.py <audio> <n> 1 bass=all brass=all flute=all guitar=all keyboard=all mallet=all organ=all reed=all string=all synth_lead=all vocal=all`.
2. **The build of its commit**: `git worktree add <tree> <commit>`, then
   `go -C <tree> build -o <build> ./cmd/stapes`.
3. **The line in parts**: `STAPES=<build>`, `PARTS=3` and the knobs that write the
   line, which `line` prints for a look first: `CORPUS`, `LOOPS` and `SEED` as it
   says, `WAV=`, `SAMPLED=<n>` but for `engine-dad4f44`, and `AUG=` and
   `OVERLAPS='overlaps 0.75'` for `clean-feaef3e`, the other corpus knobs unset.

```bash
STAPES=<build> PARTS=3 CORPUS=<c>/gen-feaef3e-s30 LOOPS=10000 WAV= SAMPLED=<n> SEED=30 tool/rip-train/pipeline.sh corpus
```

A corpus on disk is shown to be its build's bytes in minutes, without its render
whole: its line naming an empty folder, with `labels` after it, run by the build on
an empty root and config folder, writes every label, each of which must `cmp` with
the corpus's; then the line with `part k/40` after it renders one loop in forty
whole into another, each of its files the corpus's.

## Turning a run toward your music

The corpus is whatever stapes' session holds when `corpus` runs, so a focus
is a script that loads and applies voices and patterns first, named by `FOCUS`
and run by the pipeline before the export (and before `bench`, so the grade
draws from the same bank). Two are in `focus/`:

- **`focus/vocal.slab`: music with vocals.** A vocal is a line to the rip, pitch
  and onsets per frame, never words; nothing here transcribes lyrics, and the
  models have no notion of them. What the models need is lines that sound like
  voices among its loops, so the focus adds a sung-line voice and a choir pad
  (`focus/vocal.toml`) beside the shipped lead and bass; the arranged parts
  draw their instruments by family, these among them, and the augmentation redraws
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
  document to start from. Every registered tonal voice plays the arranged
  parts: by its family where `voiceFamilies` in `tune.go` names one, else as
  any part's.
- **Music.** `gen` arranges every loop itself; with `GEN` empty the loops play
  the shipped pattern bank, `audio/bank`'s, or yours: `load pattern file.toml`
  or `merge pattern` brings it in, in the pattern document's form
  (`[[pattern]]` with tracks of steps). A bank that is all one genre makes a
  corpus of that genre, and its rhythm patterns set the drums the frame model
  learns.
- **Tempos and keys.** `TEMPOS` sets the spread; the keys cycle the roots and
  scales the engine knows. A piece whose tempo the sequencer's range does not
  hold reads at a multiple.
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
  as E-GMD's, for the beat and downbeat heads, and `velocity`, a word for what
  the MIDI's velocities say, `none` where they say nothing, as GuitarSet's
  and the singing set's one constant does, which asks the velocity heads
  nothing there, or `measured`, each note's velocity heard in the recording
  instead, none where the notes with it hide its level, with `unit`, the
  recordings measured together, and `starts`, `placed` where an alignment
  placed the notes' starts, which asks the onset head, and the strike head at
  their pitch, a rise within the corpus's `reach`, which `placefit.py ... reach`
  writes: [Rip training](../../doc/rip-training.md) §3), and
  `train.py --mix <dir>:<share>` draws that share of the windows from it,
  labelled by the MIDI's notes, velocities and hits, a tom's or a cymbal's an
  onset of no drum, every hit the drums', the voices unknown, and `--over
  <dir>:<share>` lays a corpus of drums alone over loops where nothing is struck
  in that share of the windows, a strike labelled only where it stands out of
  the rest.
  Both draw a recording by its length, so a minutes-long one
  counts for its minutes, and a corpus's `skip.txt`, a loop's name
  (`loop_123`) a line, lists those the trainer leaves out, such as recordings
  whose MIDI sits off their audio. A batch run into the same directory joins
  it, numbered after the loops there, so a batch's recordings can be deleted
  once it is in.
  Weights trained on a corpus carry its licence: a CC BY set asks for credit
  ([CREDITS](../../CREDITS.md)), and a non-commercial or research-only one
  would bind them, so it stays out, as MAESTRO (CC BY-NC-SA 4.0) does. A
  licence is not enough ([the data rule](../../AGENTS.md#data)): a set trains
  only where its recordings were made for it and what they play is original
  to it, improvised or in the public domain, and a song's recording, anything
  taken from one or a commercial song's transcription only measures. A
  recording without its MIDI is held out, judged by ear and by the two-witness
  check (two recordings of one piece, their transcripts compared).
- **Phones.** A phone that hears a laptop's speaker cuts the lows and the highs
  and holds a room's noise far nearer the music than a corpus's, which sits 40
  to 65 dB down. `CHANNEL=noise.npz:0.5` hears half the windows through a
  speaker, a room and a phone drawn per window ([Rip
  training §6](../../doc/rip-training.md#6-what-may-change-a-windows-frames)).
  The noise bank is two minutes of white noise as band frames, both planes, and
  the band shape of a real room, which `python3 tool/rip-train/make_noise.py <quiet.wav>
  <out.npz>` builds from a 16-bit mono WAV of a room with no music, as a phone
  hears it; keep it beside the corpus, out of the checkout.
- **The frame model's shape.** `--dim`, `--layers`, `--heads`, `--ffn` and
  `--attention-window` on `train.py` reshape it; the weight file carries the
  shape, so `ripmodels` loads any of them as long as the inputs and heads are its.
  `pipeline.sh` passes `DIM`, `HEADS`, `LAYERS` and `FFN`; keep a head 32 wide
  (`DIM=384 HEADS=12 FFN=1536`). A checkpoint records its shape, and `train.py`
  refuses to resume or start from one of another, naming both.

## By hand

The stages are three commands; the script only strings them together.

```bash
dir=$HOME/.local/state/stapes/byhand   # outside the checkout: the corpus, the weights, a root to bench in
mkdir -p "$dir/root" "$dir/bare"
printf 'corpus corpus 20000 4 aug gen meters overlaps nowav\n' > "$dir/corpus.slab"   # read from the script's folder
go build -o "$dir/stapes" ./cmd/stapes
XDG_CONFIG_HOME="$dir/bare" "$dir/stapes" -headless -config-dir "$dir/bare" -s "$dir/corpus.slab"   # no model, the bank alone
python3 tool/rip-train/train.py --corpus "$dir/corpus" --out "$dir/ripframe.bin" --steps 20000 --batch 8 --every 1000 --holdout 200 --checkpoint "$dir/ripframe.pt"
printf 'bench 12 4\n' > "$dir/bench.slab"
cp "$dir/ripframe.bin" "$dir/root/ripframe.bin"
go run ./cmd/stapes -headless -config-dir "$dir/root" -s "$dir/bench.slab"
mkdir -p ~/.config/stapes && cp "$dir/ripframe.bin" ~/.config/stapes/ripframe.bin
```

`train.py` trains on `--batch` windows of `--window` frames (512, six seconds)
cropped at random from the loops, built beside the device by `--builders`
processes forked from the run, one by default, each up to two batches ahead.
Batch k draws from a stream of `--seed`, `--shard-seed` and k alone, so it is
the same whatever builds it ([Rip training §8](../../doc/rip-training.md#8-a-training-step)).
`--checkpoint` saves every `--every` steps, with the best held-out step's
weights, and resumes from the file when it exists, drawing from its step the
batches the unbroken run drew; `--init` starts from a `.pt`'s weights alone (no
optimizer state, step 0), and with `--carry` carries on from it, the optimizer's
state and the rate's too, the rate following the held-out readings in place of
the cosine and `--wide` making a pass swing wide ([Rip training
§8](../../doc/rip-training.md#a-pass-that-carries-on)); `--shard-seed` moves the
streams while `--seed` keeps the starting weights; `average.py --pt next.pt --out
next.bin [--trim 0.1] a.pt b.pt ...` is the merge. A builder holds memory of its own
and, once it has drawn from every loop, a copy of the run's objects its reads
write to, so each builder adds to a run's memory with the corpus's loops. The
defaults are `ripmodels.FrameSpec`:
two planes of 316 inputs, the fine plane's 315 bands in dB over the plane's
mean and the mean itself, then the sharp plane's (`ripmodels.features`, derived
from the stored frames), width 256, eight heads, four layers, feed-forward
1024, a window of 64 frames either side, 3.4 million weights. `--convs
16x3x3/1,32x3x3/2` puts convolutions before the input layer, each its
channels, frames by bands kernel and the pooling over bands; `--towers 2:2,4:11`
splits the layers into towers over the shared input layer, each its layers and
the run of heads it computes in `ripmodels.HEADS` order: the onset and the drums
on two layers, the rest on four. `--pcen` gives the frame model two more planes of
316 inputs: each band's PCEN, its energy over its own average of the last 0.4
s, and their mean, the fine plane's and then the sharp one's ([audio §9](https://github.com/lixenwraith/audio/blob/main/doc/audio.md#9-ripping-a-recording)),
the first convolution taking the four as its channels. Since the average looks
back, `train.py` computes a window's from 128 frames before it (`WARM`, three
of its time constants), so a window starts as settled as it would in a rip.
The losses: onsets, drums, pitches and each pitch's strikes and releases (within a frame of a
note's start and end, a placed start's onset and strike, at its pitch, anywhere within its
reach) as weighted binary cross-entropy, voicing likewise, the
wave as cross-entropy and the voice genes as squared error over frames a note
sounds in, each strike's velocity as squared error where the corpus gives one,
and the family of the highest note or hit sounding as cross-entropy where the
corpus gives one, at half weight and outside the score that picks the step;
whether a beat of the meter and a bar line fall within a frame, as binary
cross-entropy with no weight, in the score that picks the step
([the beat and the bar line](#the-beat-and-the-bar-line)).
A loop whose label holds a kit drawn from the genome asks the drums head nothing,
since such a kit's pieces name no line.

## The beat and the bar line

A frame model has a beat and a downbeat head: whether a beat of the meter, its own
note, and a bar line fall within a frame. Where a weight file has both, audio reads
them by their level as a plain tracker does, the tempo and the beats from the beat
head and the bar lines from the downbeat head beside the cues' rules ([audio §9](https://github.com/lixenwraith/audio/blob/main/doc/audio.md#9-ripping-a-recording));
[Rip training](../../doc/rip-training.md) §3, §5 and §8 follow the labels, the
targets and the loss.

- **A rendered loop** is asked the beats and bar lines its label lists, where its
  tempos put them, over all its frames but the last two; `corpus ... labels`
  writes them into a corpus already on disk. `corpus ... meters` draws the loops
  in seven meters, and an overlap lists none and is asked none.
- **A recording** is asked where its list's `beats` column names the file its
  beats are read from, a JAMS file's `beat_position` annotation or a MIDI file
  whose bars are the recording's; the line with `labels` after it writes them into
  a corpus already exported. It is asked from its first listed beat to its last. A
  recording that names none is asked nothing, nor is a shuffled window or one
  `--over` lays drums under.
- **The loss carries no positive weight**: audio reads each head's level as a
  posterior, which a weight on the marks would lift most where the head is unsure.
- **A head added later** grows through `--init` onto weights trained since the
  refresh but before it, whose heads are the first of the run's: every other
  weight carries over and the new head's rows of the output layer start fresh.
  That is what growing is for, since no file from before the refresh starts a
  run ([below](#parity)). Audio loads a frame model whose heads start with
  `FrameSpec`'s, any after them read by nothing ("frame model's heads are [...],
  want [...] first"), so the start's file loads in no audio that reads the new
  head. With `--probe` a run trains the
  beat and downbeat heads' rows of the output layer alone: every other weight
  stays the start's bit for bit, as a notes model's probe holds it
  ([below](#parity)), so the file's other heads say what the start's did and its
  rip with the two left out is the start's, and the last step is exported. A fine-tune without it moves the heads the decoders already
  read. `--family-unasked` asks the family head nothing: no term in the loss and
  its rows of the output layer zero from the first step to the last, so every
  frame's shares are even and the file says nothing there.
- **The lines** give `beats' F1` and `bar lines'`, or `no beat or bar line asked`.

The path at a small size, which shows that it holds together and nothing of
recordings:

```bash
printf 'corpus corpus 1600 12 gen meters\n' > "$dir/corpus.slab"   # with its audio: the held-out loops are ripped
go run ./cmd/stapes -headless -s "$dir/corpus.slab"
python3 tool/rip-train/train.py --corpus "$dir/corpus" --out "$dir/ripframe.bin" --checkpoint "$dir/ripframe.pt" \
    --steps 5000 --dim 128 --heads 4 --layers 4 --ffn 512 --pcen --holdout 160 --every 500
```

Read a round's bars with the heads and without before its file is installed,
since a weak head can cost the bars. The heads are left out of a rip by a
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

`pipeline.sh rows judge` runs the two for a pair ([the fit of a
pair](#the-fit-of-a-pair)); by hand:

```bash
printf 'rows rows from truths.txt frames frames\nrows rows from corpus\n' > "$dir/rows.slab"   # read from the script's folder
go run ./cmd/stapes -headless -config-dir "$dir/root" -s "$dir/rows.slab"                      # the root's models, never its judge
python3 tool/rip-train/judge.py --rows "$dir/rows" --leave-out guitar --out "$dir/root/ripjudge.bin"
```

- **What a fit may take.** The CC BY collections with truths that train
  ([CREDITS](../../CREDITS.md): E-GMD, VocalSet, GuitarSet), the engine's
  renders of public-domain scores, and its own mixes (a corpus written with
  its audio, `WAV=` empty). Never a set that only measures (MusicNet,
  vocadito), a song's recording or anything taken from one, nor a commercial
  song's transcription: no licence covers training on a song, so a song only
  measures, read under a judge with `rip` and `truth`, or listed under a
  condition every fit leaves out.
- **Rows.** `rows <dir> from <list>` rips each recording of a truth list, the
  lines `tune ... truth` reads, and `rows <dir> from <corpus dir>` each loop
  of a corpus that kept its audio, against its label, the corpus one condition
  under its folder's name. Each strike over a half is a row, as the decoder
  reads it:
  - `<checksum>.rows`, a recording's rows, laid out as a loop's frames are:
    the count, the row's size, then `float32` rows, little-endian;
  - `<checksum>.json`, its `source`, `condition`, `given` (the truth's notes),
    `unanswered` (those no strike answers) and `strikes`, a row each: `frame`,
    `midi`, `reported` (its stage), `held` (a note the accounting keeps
    unjudged, one the fit cannot see), `right` (a truth note of its pitch starts within the
    matcher's window of it; a truth note answers one strike, the nearest, a
    reported note before a dropped strike) and `kept` (the installed verdict);
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
  every row, which is what the rows could give at best. A count from rows is
  near `truth`'s decode of the same recordings, and `truth` is the reading.
  Under the installed scores a strike listed is the strongest of its
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
as are `--builders <n>` (by hand, above), `--cell <n>`, the notes model's width
(its channels at each cell, 32, which its weight file carries),
and `--dip <share>`, the share of the channel's
windows given a dip of 6 to 20 dB centred 100 to 500 Hz, as a phone across a room
hears. `--family-holdout <family>` tells no head a family's name: its notes stay
in the run as sound, and a notes model's lines say what their surest name reaches,
the measure a name must pass to be given (below). A checkpoint states
the band frames' layout, and `--init`, `--checkpoint` and `average.py` refuse one
of another layout or of none before they read its shape, so no file from before
the refresh starts a run. One from before a setting existed loads with that
setting's default, and `--init` takes one whose heads are the first of the
run's, weights trained before a head added later: every weight carries over and
a new head's rows start fresh ([Rip training
§7](../../doc/rip-training.md#7-the-models)), though resuming one is refused. A
notes model's long convolutions grow so under `--dilations` from a model of none
or of the first of them, each new one started to hand its cells on as near as the
GELU after it lets, and the run's line says how near (`handing its cells on, 0.031
off them`, the root-mean-square miss).

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
states (`unplaced ... stated`). A recording with no pitch track is unplaced
throughout, and shows how sure the head is of real sound and not how right. One
whose list names its track prints the three rows as an engine loop does, a frame
its track does not voice left unplaced; since no track holds a note still, a
recorded note's readings are gliding where its track spans half a cell, 50 cents,
over the note, and steady under that. `off` is the median miss, `stated` the median error the head states, and `within`
the share of readings whose miss is within the error stated for them: 63% for a
head that states its mean miss of misses spread as a Laplace's. A share well under
that on the gliding readings is a head too sure of itself where the decoder leans on
it, since audio joins two cells only where their readings agree within the stated
errors, with no margin on top. A head that knows nothing of a sound reads a median
25 cents off.
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
families F of N notes (piano F of N, ...), F of the N under another, waves F, genes G off; guitar's N notes, held out, reach R, M at the median: P% of the others' notes pass it, Q% of them right; names L a cell, S by their shares alone, F of the notes with the shares divided out (piano F, ...)
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
`names` is the family head's loss as a mean over the cells asked a name, beside
the loss of the names' shares alone there, each family's share of the cells the
held-out loops ask: a head at or over it has learned no sound, and the notes
named right with each name's say taken over its share show one the commonest
name outvotes.

The trainer's rules for the band frames and their layout, for what plays each
pitch, for the beat and the bar line, for a list's velocities, for the batches,
for growing the long convolutions and for `datacheck.py` are tests of their own, with no corpus, on
the CPU whatever the machine has: `python3 -m unittest discover tool/rip-train`,
a minute and a half on four cores, half of it the passes that carry on. One hears
a grown notes model's file through the transformer module stapes' `go.mod` names,
which `go` must have (`go mod download`), and holds it to the forward in float64.

It writes the file's version 6, which states the band frames' layout as a
constant after the heads (`layout`, `ripmodels.LAYOUT`) and, for `--pitchnet`'s
cell net, no bins for its regions to pool, and needs audio v0.30.0 (transformer
v0.7.0); audio from v0.34.0 loads a file with a head after its specs' and reads
nothing of it. Audio refuses a file of another layout or of none, naming its own,
and a file without a head its specs name but a notes model's cents head, which a
probe's file lacks; the measure is a frame head of one value whose weights are
zero and whose bias is the measure.
A change to either side is a change to both: check a random model's outputs on
a corpus loop from the two before trusting a trained file, as the module's
[agent notes](https://github.com/lixenwraith/transformer/blob/main/doc/agent-notes.md)
describe.
