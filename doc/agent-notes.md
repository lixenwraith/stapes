# Agent Notes

What agents working on stapes learned that no rule in [AGENTS.md](../AGENTS.md)
and nothing in the [README](../README.md), [stapes.md](stapes.md) or
[rip-design.md](rip-design.md) states: facts that cost a session an hour to
rediscover, and how to check them. Read it before starting. When a task teaches
you one, add it in the same PR, one bullet under its topic; delete a note that
proves wrong or that a document now states, and link the document instead.

## Toolchain

- The `gofmt` on a container's `PATH` can be older than the module's Go and reject
  generic methods ("method must have no type parameters"); run
  `"$(go env GOROOT)/bin/gofmt" -l`.
- The engine is `github.com/lixenwraith/audio` and the searches' engine
  `github.com/lixenwraith/genetic`: a change to either lands there, under its own
  `AGENTS.md` and `doc/agent-notes.md`, and stapes takes its tag.
- `lixenwraith/transformer` and `tool/rip-train` agree to float32 rounding: redo
  the check its [agent notes](https://github.com/lixenwraith/transformer/blob/main/doc/agent-notes.md#the-float32-check)
  give before trusting a trained file after any change to either side. A corpus
  loop's frames come from `stapes -s` over a script of `corpus <dir> 2 4`,
  about a minute; `corpus 8 4` plus sixty training steps of a width-32 model
  also take under a minute, enough to exercise the whole training path.
- Build the binary from `./cmd/stapes`: `go build -o bin/stapes .` writes the
  library's archive under that name, which then refuses to run.

## Tests that flake

- A test waiting on mixer state gives it seconds, since the mixer drops ticks
  under load ([audio notes](https://github.com/lixenwraith/audio/blob/main/doc/agent-notes.md#tests)): load
  the root package with ~24 race processes pinned to one CPU (`taskset -c 0`)
  to see one time out.

## Verbs, words and documents

- `slot` is asynchronous: the mixer applies the pattern a tick later and
  `SlotPattern` reads silence until then, so a verb that acts on what was just
  slotted reads the session's own `slots`, as `render` does; a test that checks
  the engine's readout waits a tick.
- The searches stapes runs are sized by fields a test shrinks
  (`Session.fit`, `Session.compressOpt`), not by constants: the rip test fits two
  notes over one generation of four and still walks every path. Seeds past the
  pool size are dropped by the engine without a word, so a pool smaller than the
  registered voices fits from the first few only.
- A command line splits on blanks (`splitArgs`); a path with a space travels in
  double quotes, which the files overlay adds. Quotes at an argument's start are
  grammar now, so a `set` value in quotes loses them.
- A recipe's relative paths were the process's: `run /elsewhere/play.slab`
  failed to find the documents beside it, and an installed stapes has no
  working directory worth the name. A running script's relative paths are
  read from its own directory (`Session.base`, `RunScriptFile`), nested runs
  from theirs, and the scratch recipes name their files from where they lie.
- `-config-dir` names the root alone: the search path still holds vif's audio
  folder under `os.UserConfigDir()`, so a run that must play the bank alone, a
  corpus or a bench to compare, points `XDG_CONFIG_HOME` at an empty directory
  too.
- **A word is a path by its dot, never by its digits.** `badName` refuses
  dots, so a dotted word is always a path; a bare number is a valid entry name
  (`cp p.track.0 5` makes the pattern `5`). An index argument goes only where no
  name can stand: `mv`'s second word after a dotted first, `add`'s and `dup`'s
  optional second, never a last word `cp` reads as a new name.
- **A document changes only through its methods.** `put`, `del`, `rename`,
  `markDirty` and `replaceAll` record an entry for undo from the document's
  shadow before they change it, since a command mutates an entry in place and
  marks it after. A write to `byName` or `order` that skips them is one undo
  cannot see, and it then restores a stale order.

## Songs and the TUI

- A song plays through the live sequencer by scheduling each bar's patterns
  quantized while the bar before plays (`conduct`), never by setting them on the
  tick that sees the bar line, which lands them a tick late and loses the
  downbeat's hits. The melody slot's boundary is its pattern's cycle, so a
  four-bar pattern there holds the change for up to four bars; the dictionary's
  one-bar patterns change every bar. A front end that ticks sets `Session.ticked`
  and calls `conduct`; without one `song play` conducts itself until the song
  ends, with a deadline past its length in case the transport never moves.
- A running transport holds a new tempo for the next beat and a new meter for
  the next bar line, and `ResetMusic` drops what it holds: `playSong` stops and
  resets before it sets the song's, or a song played over a running transport
  kept the old tempo and meter.
- A verb that takes seconds freezes the TUI if it runs in the event loop: the
  first rip from the files overlay read as a crash. `longVerbs` run on a
  goroutine behind a working box; while one runs the screen draws the last frame
  and the log's tail only, so nothing reads the session the goroutine owns, and
  `LogBuffer` takes a lock because both sides write and read it. A test of such
  a verb ticks (`KeyNone`) until `busy` clears.
- A song's `kit` names sounds `SetKit` finds in the registry: `load sound`
  leaves a document's sounds unregistered until `apply`, so a song loaded that
  way and played before refuses its kit; the root's sounds file registers at
  start. The engine keeps one drum's takes for offline renders, so a render on
  a kit other than the transport's renders both kits' takes again.

## The rip and its truths

- Write a truth from a score or a MIDI file, never from memory: a remembered
  Billie Jean riff had its first note an octave low, the synthesizer's doubling
  the analysis hears, and turned a 96-to-94 tie into a 110-to-84 loss.
- Score the three recordings pooled (every note once) beside their mean: the
  origin and the intro have 83 notes each, where five notes move F1 by 0.06.
- The test loop scores 0.98 under `DefaultRipParams` and loops of the shipped bank
  0.51 (`CompareRips.Total`): busy drum lanes whose chance hits count as misses,
  several tracks a slot and melodies under chord changes are where the model falls
  short, not the thresholds. `tune 4` over 12 four-bar loops took 36 s on four
  cores and reached 0.59; a set tuned on renders alone pushes `tightness` and
  `kick_share` to their range's ends, which a recording with tempo drift would
  pay for, so the defaults stay hand-set until the corpus has real material.
- Two transcripts of one piece are a check no single rip gives: the sample and
  its origin (three semitones apart, one stretched, on different metrical
  levels) agree on 41 of 49 melody notes, 27 of 29 under notes and all the
  bass after transposition and a fitted time map (a scratch comparison over
  two `notes` listings), so the pitch content is mostly consistent and what
  the ear misses is elsewhere: the two lines swapping a held dyad between them,
  and durations. Ask for the melody as notes and for a solo take of the lead;
  they are the ground truth the numbers lack.
- The sweep's moves were all relative to the note the rip heard, so a note it
  missed had no way in and one wrong by a third no way out. The recording's
  own strongest pitch over a step (its salience rows, which the fidelity's
  sound already holds) is the proposal: tried as a move on a step that has a
  note and as an insertion on one that has none, kept when the bar gains.
- A search over every step's genes at once (312 genes, six generations of
  twelve) kept no move on the sample: one note's move changes one frame in a
  thousand of the fidelity, under the noise of a dozen genes moving with it.
  A sweep that tries each step's five moves alone and keeps a gain sees it;
  five renders a step is the price, minutes a round on a four-bar song.
- **A score of the mix drifts the parts.** Fidelity grades the whole render,
  so a voice that fills the spectrum scores above the voice that sounds like
  its line, and `refine` turned every voice electronic while the number rose.
  Where a part has a measure of its own, the fit's distance from the
  recording's notes, hold the part to it and let the whole's score choose
  among what passes.
- The dictionary search (`dictionary.go`) rarely beats the greedy grouping it
  is seeded with under a Jaccard distance, which is expected, not a bug: the
  size term and the greedy pass apply the same pressure.
- **A truth's lines are its file's, not the rip's.** `MatchNotes` pairs a note
  only within its line, and a MIDI truth files notes by `ReadMIDI`'s rule: a
  guitar's has no bass line, so the low strings the rip puts on its own count
  as an extra and a miss each. GuitarSet's strummed clips read .58 that way and
  .79 by pitch and onset within 100 ms, 982 of 1,395 "low extras" right notes;
  MusicNet .61 and .71, its solo cello .47 and .77. `truth`, `tune ... truth`
  and the wide table all count within lines: read a MIDI-backed set by pitch
  and onset (`vif/tmp/audit2/lowline.py`, `score2`) before calling a change
  on it real.
- A truth on the recording's own clock (GuitarSet's) needs no time map, and
  the matcher's, fitted from its in-line pairs, costs it: strummed guitar by
  pitch and onset reads .72 through the map and .79 without.
- With a model loaded `rip` fits no voice: every line arrives with one, modal at
  the heads' mean genes, and `fitVoices` fits only a line that lacks one. By the
  fit's own distance that voice was farther than the nearest bank voice in 32
  of 37 lines (ten of NSynth's held-out instruments and five recordings) and
  2.3 times as far as `fit` came; type `fit <name>` before judging how a rip
  sounds. The distance favours a sparse, dark voice (it fitted modal to 32 of
  the 37, a singer and a reed among them), so its choice of wave wants ears.
- A miss is the decoder's more often than the model's: on strummed guitar 78%
  of the notes the rip misses have a strike over `strike` at their pitch within
  four frames, the restrikes under a ring mostly between it and `strike_again`.
  Print the strike head at the missed notes (`vif/tmp/audit2/why`) before
  training for a miss.

## Corpus and bench

- The user judges the corpus by ear, as music: measure what they hear before
  changing the generator. The bank's drums and bass put 82 to 89% of a loop's
  energy under 250 Hz (Für Elise and the sample 7 to 14%, Billie Jean 76%), a
  white floor a recording's mix would cover sits bare over a synthesizer's empty
  treble (a recording's hiss is about 39 dB under its median frame), and a
  reverb after the clip passes full scale.
- `bench` and `tune` loops shorter than four seconds give no rip (a two-bar loop
  at 176 bpm is 2.7 s): size a test's loops by bars, not loops, and read the
  count `bench` prints; `bench ... verbose` names each loop that gave none.
- An augmented corpus draws its room, mix and voices from a stream seeded apart
  from the loop's patterns and key, so `corpus` with and without `aug` renders
  the same music; a test comparing the two relies on it. Generated parts register
  under `corpus_*` names replaced in place each loop, and `corpus` leaves those
  names out of the bank's pools, or the next draw would include the last loop's
  generated patterns and no two runs would draw alike.
- `corpus ... meters` draws a loop's meter from a stream of its own, and a loop
  in 4/4 renders as it does without `meters`, which a test compares byte for
  byte; a loop in 3/4 or 6/8 generates every part, the bank having only 4/4.
  The rip reads a meter other than 4/4 only over more than eight bars of beats,
  so `bench 24 16 meters`: it read the 19 loops in 4/4 as 4/4, and the bar of
  7 of 10 rips of the others but their label in 4 (a 6/8 tracked in quarters
  reads 3/4, a 3/4 in dotted quarters 6/8), which is why the bench counts both.
- `bench` grades the analysis as it is today, so a run's numbers compare only
  with the analysis benched beside them: the seventh pass moved the analysis
  from 0.47 to 0.49 on clean loops, and the fourth model run's 0.45 and 0.51
  stand against that, not against the third run's table. The model's corpus
  is rendered at A440, so nothing it saw was detuned; a recording off the
  standard is a recording it was never shown, whatever its held-out loss.
- A `corpus <dir> from` loop is known by its `source`, the CSV's
  `audio_filename`: the set may move, but an entry renamed joins as new. A clap
  in a test file is a hit for a song, not for `truth`.
- NSynth's notes are each normalised: on the test set a pitch's velocity layer
  peaks a median 0.1 to 0.6 dB from its 127's, nine in ten within -1 to +4 dB,
  so a layer is timbre, not loudness. `corpus ... sampled` takes each note's level
  out and plays it at its step's velocity, as the engine's voices do. A release
  (from 3 s) falls 20 dB in a median 108 ms, an organ's 40 and a plucked or
  struck one's 65 to 82, which the cut's crossfade follows.
- The bench's loops play no piano: a piano change moves no bench number, a bass
  change does.
- `corpus <dir> from` writes a MIDI file's velocities as labels. GuitarSet's,
  MusicNet's and the singing set's are one constant, which the velocity head
  would learn as a fact about recordings: strip `vel` from such a corpus's
  labels (`vif/tmp/round26/strip.py`) before a round reads it.
- A score aligned to a recording is not a label to the frame. The notes model's
  strike head peaks within a frame of the label on 98% of a rendered loop's
  strikes, 80% of GuitarSet's (a frame early) and 36% of MusicNet's, spread
  over 80 ms either way (`vif/tmp/audit2/offsets.py`), so a strike target a
  frame wide teaches MusicNet's jitter and a loss there measures it.
- A drum change is benched piece by piece without editing the tree:
  `stapes -config-sounds piece.toml -s bench.slab` overrides one sound of the
  built-in kit. The analysis hears a kick still ringing at 50 to 60 Hz as the bass
  line, and a snare's fixed modes move the melody's key: the bench's clean bass
  rests on three loops of twelve, which a kick of the old 42 Hz glide restored.

## Training rounds

- A training round runs from its own copy of the tree: `pipeline.sh` runs
  stapes through `go run` at every stage, so an edit to the checkout while a
  round trains changes the corpus or the bench under it. The bench draws its
  loops from the tree's code too, so rounds whose `aug` or `gen` differ bench on
  different loops: bench one round's weights through the other's stapes.
- A model's decoder thresholds belong to its weights: a new head's scale moves
  them, so tune them (`tune ... truth <list>`) before judging a round (10a read
  0.530 pooled at 8b's, 0.650 at its own), and pin them in the root's
  `rip.toml` beside the weights. Compare rounds on one tree's bench loops.
- With strikes the melody register never reaches the pair decoder, so
  `second_voice` and `cross` move no note on the model path, while the bass
  line's own path reads `jump`, `switch` and `retrigger`: `modelGenes` is what
  audio v0.9.0's decoders read; check it against a new decoder. A candidate of
  `tune ... truth` pays the decoders alone, and the accounting stage
  (`RipFrames.account`) was 90% of them in a profile.
- A round that gains on the bench may still lose on the recordings (10a over 8b):
  the bench's loops are the corpus's own kind. Judge by the recordings.
- Towers (`transformer.Tower`, `TOWERS=` in `tool/rip-train`) are in no shipped
  weight file, yet are not dead code: round 7d found no gain for 43% more
  weights, and they are to be tried again once new heads (offset, velocity,
  pedal) compete with the strike head for the shared layers. Keep them, their
  loader and their round-trip test.
- A training loss that keeps falling says nothing about fresh loops: 12000
  steps on 300 loops reached 0.31 and benched worse than 3000 steps on 120,
  each loop seen three hundred times. `train.py` holds a tenth of the loops
  out, scores them every `--every` steps and exports the best step; judge a run
  by its held-out line and by `bench` on the file, never by the last training
  loss, and prefer more loops to more steps.
- A `.pt`'s held-out loss, a merge's or any checkpoint's, is read without
  training: `train.load_corpus`, `train.validation` over the last `--holdout`
  loops and `train.evaluate` in a scratch script, the model built from the
  file's `cfg`. `--steps 0` scores nothing, since the score runs in the loop. A
  `pip install --user` torch is not seen under `python3 -I`, which drops the
  user's site directory.
- `train.py` reads each window's frames from its `.mel` as it draws it
  (`MelFile`) and builds the window's targets from the loop's events then
  (`Label.window`): rows per frame held up front came to 207 bytes a frame,
  5.2 GB for 40000 loops, and MAESTRO's 159 training hours would add 10 GB. A
  `np.memmap` a loop would keep a file descriptor open each, 40000 of them,
  past a 1024 limit or macOS's 256.
- A training run shares the cores with everything else the session does: the
  fourth run slowed from 0.4 to 4 seconds a step under a rip and a test run,
  and the two-hour waiter on it timed out twice. Size a waiter by the run
  alone and expect to re-arm it, or follow the run by a reminder instead.
- A run's noise differs by set: five rounds of one recipe (32a, 32c, 33a, 33c,
  34c) spread .003 on the renders, the clean take and MusicNet, .005 to .008 on
  the phone takes and guitar, .010 on vocadito and .003 on the mean of nine.
  Two runs that differ in anything else are not a repeat: 34a against 34b,
  once read as noise, is width, six spreads apart on the clean take.
- A notes model's loss on recordings is mostly its onset head (57% on
  GuitarSet's player 04, 68% on MusicNet), which Go never reads from it, and
  over ten rounds it ranked them unlike their test sets; the strike head's
  peaks at the decoder's bars ranked them as solo guitar's test player did
  (Spearman .93), which is what `--validate` picks a notes model by. One
  instrument's score trades against another's (the same peaks against
  MusicNet, -.58), so validate on every kind of recording at once.
- A checkpoint scores a third lower on recordings it trained on: 24b and 33b
  mixed in GuitarSet's player 04 and the 19 MusicNet pieces before those became
  `~/ripcorpus-val`, and 33b's validation loss reads 2.16 against 2.9 to 3.3.
- `ripmodel.features` over a whole long recording overflows: its PCEN average
  is one running sum over growing weights, past float64 near 24,000 frames
  (284 s), and every frame after is NaN under a RuntimeWarning alone. Hear a
  long recording in chunks with the trainer's 128 frames before each; a
  training window never reaches it.
- A model run over a whole recording in Python takes memory to match: 12a's
  attention over ten minutes took 13 GB. Chunk it (3,584 frames with 256 of
  context) and run any long job as a capped unit (`systemd-run --user -p
  MemoryMax=...`), where running out stops the unit and not the session.
- `tune ... truth` pays three CPU-minutes a candidate over the validation
  list's 60 recordings after 25 minutes of hearing them, six hours for the
  default 16 by 8, and keeps no frames between runs: tune on a third of the
  list. Its score counts within lines and its list is guitar alone, so its
  first run silenced the bass line (`strike_low` to .99): its own score rose
  .608 to .670 and solo guitar .06 by pitch and onset, while the renders, the
  clean take, the phones and MusicNet fell .02 to .04 and Billie Jean's bass
  from 111 right to 93. Decode a tuned `rip.toml` over every condition
  (`vif/tmp/audit2/cond-v0140`, frames kept on disk) before it ships; no change
  of one bar from the hand-swept five gained the mean of twenty conditions.
- The live rip's notes are the offline rip's (95 in a hundred over twelve
  recordings through 12a and 32a) and its bars are not: tempo, meter and phase
  hold from the first six bars, so the two agreed on three recordings. It costs
  three times the offline rip, each model run three times over where the ear's
  half-second steps overlap (`vif/tmp/audit2/live-v0140`); the exact GELU is
  29% of either, and offline the notes model, not the frame model, is the
  larger share (47% against 37%).
- A tool that names its corpus in its source dies with the corpus: the edge
  probe read corpus 16, deleted a day later. Take the corpus, the checkpoint
  and the first loop as arguments.
