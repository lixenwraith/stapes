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
  also take under a minute, enough to exercise the whole training path. A grown
  notes model's file is held so by a test (`TheLongConvolutions`), Go's float32
  against PyTorch's forward in float64 at the cell net test's bound, 1e-5 and
  as much again of each value: against PyTorch's float32 forward the two sides'
  roundings add, and on frames of noise passed that bound by 4.5e-6.
- Build the binary from `./cmd/stapes`: `go build -o bin/stapes .` writes the
  library's archive under that name, which then refuses to run.
- A script fed as `-s /dev/stdin` reads its relative paths from `/dev`: write
  the script to a file in the folder its paths are meant from.
- An untyped integer constant divides as an integer: with `sampledSpeak = 6`,
  `math.Pow(10, -sampledSpeak/20)` is `math.Pow(10, 0)`. Write `20.0`, as
  `sampledFloor/20.0` does; nothing warns.
- A phrase in the docs wraps, so count one over the whole file, `\s+` between
  its words: `perl -0777 -ne 'print scalar(() = /\b[Tt]he\s+model\b/g)'` over
  v0.24.0's tree counts 26 where `git grep` by line found 20.
- A change that needs an audio not yet tagged names the tag in `go.mod` and
  builds through a workspace kept outside both trees: a `go.work` of `use
  <stapes>` and `replace github.com/lixenwraith/audio vX => <audio>`, named by
  `GOWORK=` on each stapes command. A `replace` in `go.mod` cannot be
  committed, `use` of both trees still reads the tag's `go.mod`, and a
  `go.work` in a parent directory breaks the builds inside the audio tree.
  `go.sum`'s two lines for the version are a hash of the module's files, so
  they can be computed from the audio branch's tree before the tag and are
  the tag's while a squash merge leaves that tree as it is; compute them
  again when the audio branch changes.
- A test that needs a property of a loop writes the loop or walks a bounded draw
  for one, failing as no fixture: the bank's music and mix move every render, and
  a bank loop that held one held it by chance (`truthFixture`'s `holds`, the
  crescendo of `TestTuneTruthScoresDrumsAndBarsWhereATruthGivesThem`). To tell
  the bank from the engine when a test moves with an audio version, lay the last
  version's `bank/*.toml` over a copy of the new tree (`git archive`, `git show
  <tag>:bank/<file>`), a file or a field at a time, and run it through a
  workspace on that copy.
- A cloud container holds no PyTorch: `make venv` installs the CPU build in
  two minutes, and `tool/rip-train/.venv/bin/python3` runs the trainer and
  `judge.py` from then on. Where the container cannot resolve
  `download-r2.pytorch.org`, which the CPU index's links name, `make venv`
  stops at the wheel: the index page lists each file with its sha256, the same
  file is at `https://download.pytorch.org/whl/cpu/<file>`, and `pip install`
  takes it from disk once `sha256sum` agrees, then `requirements.txt`.
- A task names main's commit as it was written, and main can be rewritten before
  it runs, a docs commit replaced by its amended self, the one named then on no
  branch: `git fetch origin main` and branch from that. A cloud clone's `main`
  and `origin/main` can be days older still.

## Tests that flake

- A test waiting on mixer state gives it seconds, since the mixer drops ticks
  under load ([audio notes](https://github.com/lixenwraith/audio/blob/main/doc/agent-notes.md#tests)): load
  the root package with ~24 race processes pinned to one CPU (`taskset -c 0`)
  to see one time out.

## The race suite and CI

- The suite takes 69 s plain and 16 minutes under the race detector on four
  cores; a CI runner takes from 0.7 to 1.6 times that. A test that renders or
  hears audio runs 9 to 20 times slower raced, and 85% of that CPU is the
  detector watching one goroutine's arithmetic; a light test costs 0.65 s
  raced, a session's start and stop.
- CI races the suite as seven jobs (`.github/workflows/test.yml`): each lists
  the tests, deals them heaviest first to the lightest of seven shares, and runs
  its own. `.github/race-seconds.txt` weighs the tests of five seconds and over and
  any other weighs a second, so the list only balances: a test it does not name
  runs all the same, and a name no test has is ignored. A new test that renders
  or hears audio for a third of a second plain goes in at fifteen times its
  plain seconds, until the list is measured again.
- To measure it: `go test -race -count=1 -v . > raced.log`, sixteen minutes,
  then under the file's comment lines
  `awk '/^--- PASS/ { gsub(/[()s]/, "", $4); if ($4 + 0 >= 5) printf "%.0f %s\n", $4, $3 }' raced.log | sort -k1,1nr -k2,2`.
  More jobs than the suite's time over its heaviest test's gain nothing: 941 s
  over 129 is seven, a share of 134 s here.
- What a test costs raced is the seconds of audio it renders and hears, so a
  fixture is sized by its seconds: a rip refuses under four, which two bars at
  110 bpm clear, and `live` hears twelve, since of ten no bar freezes ahead of
  the recording's last second. A test that needs a rip, a bench or a save
  another test makes of the same recording goes into that test, not beside it.
- Tests cannot run side by side in one process, so `t.Parallel` is out and a
  job uses one of its runner's four cores: a session registers its sounds,
  voices and patterns in the audio module's process-wide registries, which
  `teardown` resets.

## Verbs, words and documents

- `slot` is asynchronous: the mixer applies the pattern a tick later and
  `SlotPattern` reads silence until then, so a verb that acts on what was just
  slotted reads the session's own `slots`, as `render` does; a test that checks
  the engine's readout waits a tick.
- A test that hears what the session plays hands `NewSession` its speakers
  (`Options.Sink`, `earSink`): the mixer writes them a period at a time in real
  time, so the test waits on what it reads (`waitFor`), never on a count of
  ticks. The binaural bed is the one thing the stream's two channels differ by.
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
  too, as `pipeline.sh corpus` does. Build stapes first: `go run` under that
  folder reads its `go env` from it (`GOENV`) and writes its telemetry there.
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
- **A save's source is the file at the document's provenance, read at the
  save** (`authored`): never the target, so `save <kind> <other>` carries the
  macros and comments there, and never kept from the load, so undo restores a
  path and no bytes. With none to read, or over a file `Marshal` wrote, the
  library's save gives `Marshal`'s bytes, so a document stapes made is written
  out in full through any load and save, as before. The sound save passes the
  root's `include` too: left out, the author's line is deleted.
- **Patterns seeded from the built-ins carry `doc.bank`**, provenance with no
  file: `seedBuiltinPatterns` sets it after `replaceAll`, which clears it for
  anything else, and an edit records and restores it beside `src`, so undoing
  a load brings it back. `src` cannot say it: the TUI reads `src` as a path or
  none (`saveModified`, `openFiles`, `promptSaveHere`), and a seeded
  document's first `^S` must still ask where. While `src` is empty the save
  writes over `bank.MusicTOML()`; once a save names a file, the file wins. The
  seed parses the bank's document for its order, 9.6 ms once: the registry
  holds the code-owned `melody_gen` first, by its fixed ID, where the bank
  writes it 29th, and only in the bank's order is a seeded document saved
  with no edit the bank's bytes.
- Nothing checks a doc's links or its verbs. A short script slugs each heading
  as GitHub does (lowercase, punctuation and `§` dropped, spaces to hyphens)
  over every `*.md`, the other modules' from a `git clone --depth 1` of their
  main, which a cloud container reads though they are not its repositories;
  another looks each usage of `command.go`'s table up in the README, the
  alternatives split at ` | `.
- A test of a document stapes made passes with no source at all, so each
  save's source is held by an authored document alone: the patterns' by
  `TestASavedDocumentIsStillItsAuthors`, the sounds' and their `include` by
  `TestADocumentSavedElsewhereCarriesItsMacrosAndComments`. To see that what
  stapes made saves as it did before a change, run one `.slab` (render, rip,
  compress, save, load, edit, save) through a binary of each tree, headless
  with an empty `XDG_CONFIG_HOME` and `-config-dir`: the rip and the default
  fit repeat, so the files `cmp` equal, ten seconds a run.

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

- **A test of what the heads decide writes them over the analysis's frames.**
  The analysis carries no head and a test's small random model marks no beat,
  so a rule the heads settle (a 12/8 read over its eighths or over its dotted
  quarters) is tested through a front end that hears through the analysis and
  writes `Beat` and `Downbeat` by hand (`builtHeads`).
- **`live` keeps the offline rip, and its test compares that, not the
  stream.** `RipStream.Final` rips the samples the stream kept as float32,
  which a 16-bit WAV's are exactly (a 32-bit integer or 64-bit float one's
  are rounded), so the rip `live` registers is `rip`'s to the last field,
  fitted voices included (`TestLiveKeepsTheRipRipMakes`). The stream's own
  rip differs by design:
  through the test's random model it reads another tempo than the offline
  rip, so nothing in a test should pin the measure's numbers.
- **A bar's delay is from its end, and paced it is the clock's.** Unpaced,
  where the sound is means the samples fed, as if the stream took no time;
  with `pace` it is the time since the first block, which counts what the
  stream took. The first bars freeze together once six are tracked, seconds
  behind, so the worst bar is taken after them.
- **`live`'s CPU is the process's** (`syscall.Getrusage`): the transformer
  runs a model across the cores, so the time a call takes is not what it
  cost. The mixer's and a screen's are in it; measure with the music stopped.
- **The ear's short wait shows in the drums, not the notes.** Audio's
  `FrontEnd.Ahead` at half a second leaves every pitched head as the whole
  recording's (they are the notes model's, whose reach it still waits) and moves
  the frame model's, the drums among them: count the stream's kicks and snares a
  bar, not its notes.
- **A thread's evidence is random weights.** A cloud container holds no
  trained weights and no recordings. After threads merge, `live` over a few
  recordings through the installed pair, the offline rip's note counts against
  the last reading and the two `corpus` digests take minutes and say whether
  the merge holds on what the rip is for.
- **An installed root's frame model is `ripframe.bin`.** A root from before
  v0.23.0 holds `ripmodel.bin`, which stapes names at the start ("frame model:
  ripmodel.bin is now ripframe.bin, so rename ...") and never loads, so the rip
  hears with the analysis, the notes model and the judge unread.

- A truth holds one note a frame and pitch, since a rip reports one, and none
  outside the pitches the rip's front end tells apart, or notes no rip can give
  cap its recall. `tune ... truth` and `rows` leave the second kind out
  themselves (`matchHeard`, by `Rip.Range`, under the transposition the match
  finds). `truth` still lists such a note as only given
  ([todo](todo.md#train-the-pianos-pitches)). A doubled note's name is right
  when it is any family sounding there, and its names are printed apart ([Rip
  training §5](rip-training.md#5-the-targets)).
- A scorer is code nobody trained: a new scorer gets a test or a count made
  apart before a figure from it is used. A table is one decode's: when audio
  moves, rebuild the reading tools between rounds and read the baseline again.
- A bank loop and every loop `bench` draws start on a bar line, which rewards a
  decoder that leans on a recording's start; only `corpus ... meters` starts
  loops part-way. Read bars on loops cut a quarter to three quarters of a bar
  in, the truth's lines moved by the cut, beside whole ones: a search fitted on
  whole loops alone can learn the lean.
- Write a truth from a score or a MIDI file, never from memory.
- Score recordings pooled (every note once) beside their mean: a few notes move
  a short recording's F1 far.
- Under `DefaultRipParams` the shipped bank's loops fall short of the test loop
  (`CompareRips.Total`) in busy drum lanes whose chance hits count as misses,
  several tracks a slot and melodies under chord changes, not in the thresholds.
  A set tuned on renders alone pushes `tightness` and `kick_share` to their
  range's ends, which a recording with tempo drift would pay for, so the
  defaults stay hand-set until the corpus has real material.
- Two transcripts of one piece are a check no single rip gives: two recordings
  of it, compared after transposition and a fitted time map (a scratch
  comparison over two `notes` listings), show how far the pitch content agrees
  and what the ear misses. Ask for the melody as notes and for a solo take of
  the lead; they are the ground truth the numbers lack.
- The sweep's moves were all relative to the note the rip heard, so a note it
  missed had no way in and one wrong by a third no way out. The recording's
  own strongest pitch over a step (its salience rows, which the fidelity's
  sound already holds) is the proposal: tried as a move on a step that has a
  note and as an insertion on one that has none, kept when the bar gains.
- A search over every step's genes at once keeps no move: one note's move
  changes a sliver of the fidelity, under the noise of a dozen genes moving with
  it. A sweep that tries each step's five moves alone and keeps a gain sees it;
  five renders a step is the price.
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
  guitar's has no bass line, so the low strings the rip puts on its own count as
  an extra and a miss each. `truth`'s `notes` row and `tune ... truth` count by
  pitch and onset (`Agreement.All`); `truth`'s rows by line, its `every part`
  row and the wide table still count within lines.
- A truth on the recording's own clock (GuitarSet's) needs no time map, and the
  matcher's costs it. Say `timed`, to `truth` or as a list line's fourth field;
  its hits too: a groove matches as well a whole number of beats off, and a
  search put a drum take's hits two beats off.
- The voice a model hears for a line is the fit's first seed, no more, so `rip`
  fits every line. A change to the fit's measure is read apart from the rip
  first, `fits ... given` on loops whose lines are each rendered alone and on
  their whole loops, then on ripped notes.
- **`fits` holds a line to its label's voice by the genes, at the rip's notes.**
  A bank voice rebuilt from its genes (`VoiceFromGenes`) renders 0.0000 in mel
  distance from itself, so `own` is the voice that played; but it plays the
  notes the rip heard, and a line whose `matched` is far under its `notes`
  reads the label's voice at notes it never played, which `given` does not.
  The analysis hears no voice, so a container's rows carry no `heard`: a test
  fills them with a random frame model, whose rips hold a note or two a line.
- **A window is heard as the mix of what plays in it.** Set against one note
  rendered alone at its peak, the recording around it holds the line's other
  notes, the other lines and the drums, and a voice whose attack or ring fills
  them wins: on 40 clean loops at the labels' own notes the label's voice was
  nearest on 0 of 86 lines, a string or modal wave on 57. The line's notes in
  the window, the others' and the hits beside them, each at a gain of its own,
  put it nearest on 40, most of the rest losing to a voice near its genes.
- Read so (`fits ... given`, 50 loops of seed 23, 113 lines a reading, the
  default sizes), the fitted wave is the label's on 109 lines alone, 98 in
  their loops and 82 with lengths and velocities moved as the analysis's rip
  moves them, and 104, 90 and 83 under `aug`, where one note at its peak got
  82, 48, 45 and 78, 56, 47. The best seed is now the label's voice, or the
  bank voice it was redrawn from, and a fit that beats it does so by a
  twentieth of its distance, wandering 0.1 to 0.25 from its genes where the
  measure hears little: the fits nearer the genes than their seed fell.
- **A unison stack or a string is never its own note again.** Audio seeds a
  stack's start phases and drift and a string's burst by the note's pitch and
  its voice's last trigger (`noteSeed`), and `RenderNoteAt` triggers at 0: at
  the render's own onsets, alone, a modal voice reads 0.0000 from its own
  voice, the lead 0.005 to 0.025, the pad up to 0.020, the pluck 0.003 to
  0.019, and a voice within 0.02 of its genes is as near as often as not.
- A label's and a rip's onsets sit on the frame grid, up to 5.8 ms off the
  sound: alone at the labels' frames a modal voice reads up to 0.003 from its
  own where at its render's own onsets it reads 0.0000.
- A loop's line alone, as the corpus plays it, is `RenderBars` with that
  instrument alone, carried through the loop's room by `room`'s second return
  and laid at the loop's gain: with the drums' the lines sum to the loop.
- **A fit is its seed's on any number of cores, but for its memo's key.** The
  engine draws every candidate from one stream and scores them side by side;
  `genomeKey` names a genome to three decimals, so two new candidates of one
  generation that differ past the third take whichever was scored first. A
  part read on one core matched the same loops read on four, byte for byte.
- A miss can be a note the notes model heard, with a strike over `strike` at its
  pitch within four frames: print the strike head at the missed notes before
  training for a miss, and try a rule on every condition before moving it.
- **`Fidelity` puts no price on a wrong note**, so it cannot set a decode's
  thresholds: a wrong note on a partial, a ring or a glide the recording has
  earns it part of what a right one does, so a decode laid again without its
  wrong notes gains F1 and loses `Fidelity`.
- Two renders compared by `Fidelity` are rendered under the same names, each in
  a fresh session: a sound's noise is seeded by its name.
- A judge's fit is read with a collection left out, never a recording.
- A truth that holds one line alone counts every chord the rip hears as wrong:
  read it by that line's notes right.
- Hold-outs are by source, never by file: GuitarSet's players (00 to 03 train,
  04 validates, 05 tests), VocalSet's singers (f2, f8, m3, m5 and m10 held out),
  E-GMD's split, NSynth's instruments, a generated set's seed. Check every
  label's `source` against every validation, fit and test list before calling a
  set unmet, and take a set a round trains on out of every held-out reading, a
  judge's fit included. Music is shared too: an evaluation's loop can share a
  stretch of a score with a training loop, so after a round read whether such
  loops pull ahead, naming every corpus trained on.
- **A song only measures.** No song's recording, bought or not, nor anything
  taken from one (a take of it through a room or a phone), nor a commercial
  song's transcription is in a corpus or a fit's list, and no song shapes an
  input, as a noise bank shaped by a take's quiet would: no licence covers
  training on them (AGENTS, Data). What trains or fits is the engine's generated
  parts and mixes, public-domain scores it renders, and the CC BY sets whose
  recordings were made for the set, of music original to it, improvised or in
  the public domain: by that rule E-GMD, VocalSet, NSynth and GuitarSet train,
  and MusicNet and vocadito only measure. At a fresh start, list every input
  with where it came from.
- **A collection's licence is read on its own page, and its parts may carry
  their own.** MusicNet's record is CC BY 4.0, yet its recordings are
  Creative Commons and public-domain performances. The Isabella Stewart
  Gardner Museum's are among them, and the museum released its concerts
  under CC BY-NC-ND 2.0; the set's metadata names a recording's performer,
  not its licence. URMP's pages state no licence, and its Dryad deposit is
  CC0 by Dryad's terms. A set a round adds joins [CREDITS](../CREDITS.md)
  with its title, creators, link, licence and what stapes changed in it.
- **`rows` knows a recording by its audio alone**: its files are named by
  the first eight bytes of the WAV's SHA-256, the name `frames <dir>` starts
  with. Two list lines over one WAV write one recording, under the first
  line's truth and condition, and a truth corrected later is not read again
  until the recording's `.json` is removed. A corpus directory is one
  condition, so a share of loops left out is a second corpus under another
  `seed`, or a list of the loops with their labels written as transcripts.
- **A test's judge is `judgeBytes`**, a unit a layer whose hidden layers give
  0 and whose output's bias is the logit of the odds named, so a stage rules
  every row alike. It must name the notes weights: a test model's checksum is
  its file's last eight bytes, which `ripmodels.Load` keeps as `NotesSum`; a
  front end built by `ripmodels.New` has 0 there.
- **The trainer's judge and audio's are held together by their odds.**
  `judge.py` repeats byte for byte on CPU, a directory's fits writing one file,
  and on audio's own rows of left-out loops the two sides' odds differ by
  rounding alone, with no verdict apart. A scratch program rips a WAV under the
  root's judge with `Ripper.Strikes` and `Rows` and dumps each strike's stage,
  odds and row; the file's layers applied to those rows in NumPy give the other
  side. Redo it after any change to the file, the row or either forward pass.
- **A count from rows is near `truth`'s decode, not it.** The installed scores
  weigh the candidates that stand in time order, and the read lists the
  strongest strike of a pitch within `strike_gap`, which carries the verdict of
  the candidates near it and is labelled at its own frame: two candidates
  returned around one stronger strike are one row, and a candidate returned a
  few frames after a stronger strike can be a right note whose row is labelled
  wrong. Under a judge a dropped strike is measured against the notes the judge
  kept, where its row was written against those the installed scores kept, so
  some verdicts differ. Read a fit by `truth` over the conditions left out, and
  use the rows' count to see that a fit is sane.
- Singing is the last to come: little truth, many kinds of voice. Report it
  beside every reading and hold no stage back on it.
- Lixen's order of work: fix what is already known to help, train a little,
  then measure with focus. Do not measure further ahead of a round to pin
  every open piece first. A reading where little is trained is not a verdict:
  a line only moves forward, and what it cannot do yet is read again after
  the next round.

## Corpus and bench

- The user judges the corpus by ear, as music: measure what they hear before
  changing the generator. The bank's drums and bass put 82 to 89% of a loop's
  energy under 250 Hz, a white floor a recording's mix would cover sits bare
  over a synthesizer's empty treble, and a reverb after the clip passes full
  scale.
- `bench` and `tune` loops shorter than four seconds give no rip (a two-bar loop
  at 176 bpm is 2.7 s): size a test's loops by bars, not loops, and read the
  count `bench` prints; `bench ... verbose` names each loop that gave none.
- An augmented corpus draws its room, mix and voices from a stream seeded apart
  from the loop's patterns and key, so `corpus` with and without `aug` renders
  the same music; a test comparing the two relies on it. Arranged parts register
  under `corpus_*` names replaced in place each loop, and `corpus` leaves those
  names out of the bank's pools, or the next draw would include the last loop's
  arranged patterns and no two runs would draw alike.
- `corpus ... meters` draws a loop's meter and its start from a stream of its
  own, and a loop it leaves in 4/4 from a bar line is drawn as without the word,
  which a test compares; without `gen` a loop it moves is arranged, the bank
  having only 4/4 from a bar line. The rip reads a meter other than 4/4 only
  over more than eight bars of beats, so
  `bench 24 16 meters`: a rip can find a loop's bar and miss its label (a 6/8
  tracked in quarters reads 3/4, a 3/4 in dotted quarters 6/8), which is why the
  bench counts both.
- `bench` grades the analysis as it is today, so a run's numbers compare only
  with the analysis benched beside them.
- `corpus ... glide` draws its bends and scoops from one stream and its
  vibratos from another, each seeded apart from the loop's own, so a loop's
  patterns, key, voices and tuning are those the line draws without the word,
  which a test compares. A label's `cents` is written only on a note that
  moves, so a corpus without the word is byte for byte what it was: a new label
  field stays out of the file where it is unset. A voice with a vibrato of its
  own (the vocal focus's) moves without the word, and its labels gain the list.
- `corpus ... labels` plays nothing, so nothing a label holds may wait on the
  mixer: `DefineVoice` queues a command and fails once 256 wait unplayed. The
  pass checks a voice with `ValidateVoice` and registers none, so the registered
  voices stay as they are and need no restoring. A label takes nothing from the
  render: its length is counted from the loop's samples (`loopFrames`), which a
  render checks against the frames it heard, and a `.bands` header of another
  width (`bandsLength`) counts as no frames on disk. A field a label gains must
  be drawn before the render, or the pass cannot write it.
- A `from` corpus's labels pass knows a recording by its label's `source`, so
  a label that does not read stops it. It reads the list again, its `velocity`
  column with it, so a list changed to `none` is relabelled in seconds; under
  `measured` it opens every recording and hears each note, VocalSet's two
  splits in an hour and a half on half a core but GuitarSet's chords some 220 s
  of a core a recording, their old rings in doubt and walked window by window,
  and it takes as many recordings at once as the process runs threads: four at
  once on one thread held 3.2 GB of rings and, under a memory cap, crawled.
  Read by their strings, GuitarSet's 360 took 2.3 s of a core each and 446 MB
  at the most, the set in under half an hour on half a core.
  A recording gone from disk, whose sound does not read or with a level not
  finite keeps its label with every note at -1, which the run's last line
  counts by reason: a recording's labels are never lost to its measure.
- A pitch track at 44.1 kHz and a hop of 256 has two samples a frame, one on
  the frame's own time, so a missing sample lies a whole step from its
  neighbours. A reach of a step read a neighbour's pitch there by a rounding's
  luck; the reach is three quarters of a step (`pitchTrackReach`).
- A tracker's jitter makes every frame unlike its neighbour, which is how the
  trainer knows an engine note's glide, so by that rule a jittered track glides
  throughout. A recorded note glides when its track spans half a cell
  (`GLIDE_SPAN`). A change to how `train.py` reads a label is checked by
  `Label.window` before and after over a `glide` corpus: its targets from engine
  labels must not move. A rewrite of `window` for speed is checked so over every
  target, byte for byte, of fixed windows of generated, `glide` and drum loops
  and of recordings' labels made up with nulls in their cents, a 50-cent tuning
  and notes outside the pitches. NumPy 2 compares a bare float with a float32
  array in float32, where an array of float64 offsets compared in float64: keep
  `np.float64`, which the pinned digest's labels cannot tell apart.
- A `corpus <dir> from` loop is known by its `source`, the CSV's
  `audio_filename`: the set may move, but an entry renamed joins as new. A clap
  in a test file is a hit for a song, not for `truth`.
- NSynth's notes are each normalised: on the test set a pitch's velocity layer
  peaks a median 0.1 to 0.6 dB from its 127's, nine in ten within -1 to +4 dB,
  so a layer is timbre, not loudness. `corpus ... sampled` takes each note's level
  out and plays it at its step's velocity, as the engine's voices do. A release
  (from 3 s) falls 20 dB in a median 108 ms, an organ's 40 and a plucked or
  struck one's 65 to 82, which the cut's crossfade follows.
- NSynth's family is an instrument's name up to its source: `synth_lead` holds an
  underscore, and a name's first word, `synth`, is no family of the set's. A
  container holds no NSynth, but the training split's names stream out of its
  archive in minutes, nothing stored (`curl ... | tar -tvz`): 289,205 notes of
  128,044 bytes, enough to check a folder `nsynth_families.py` makes by name.
- A string's greatest rise near a note's annotated start does not say the note
  starts late: 14% of GuitarSet's notes with a clear rise have it over 100 ms
  after the annotation, and for all but 0.8% the string was already sounding (a
  swell, a second strike). Ask whether the string was silent at the annotated
  start, as `tool/rip-lab/audit/stringrise.py` does.
- A listening page plays in a hidden tab, where `requestAnimationFrame` never
  fires: what must run while a clip plays, such as the check that it moves, is
  on a timer.
- The bench's loops play no piano: a piano change moves no bench number, a bass
  change does.
- `corpus <dir> from` writes a MIDI file's velocities as labels. GuitarSet's and
  the singing set's are one constant, which the velocity head would learn as a
  fact about recordings: their list's `velocity` cell says `none`, and each note
  and hit is labelled at -1, which the trainer asks nothing of, or `measured`,
  and each note is labelled at the velocity heard in its recording, -1 where the
  measure cannot tell it from the notes with it. A cell is a word, a row of
  `velocityWords` in `tune.go`; another kind of velocity is a row more, and a
  word the table lacks is refused, naming those it has. A `starts` cell is the
  same kind of table, `startsWords`.
- A score aligned to a recording is not a label to the frame: a strike target a
  frame wide on an aligned score teaches its jitter, and a loss there measures
  it.
- A note's level is read from its partials, each at the note's own stiffness:
  a piano's and a plucked string's run sharp of k f0, the more the higher the
  note, and fitted at k f0 they miss its energy by decibels, so `stiffnesses`
  searches each note's in its first window. A note sounds on past its label's
  end, and unfitted its partials pass for the next note's. No one time holds for
  every ring: at a quarter second soft notes after louder ones read over within
  their bound, and at NSynth's second of release a soft note after one held past
  its label read a median 20 dB over, so a ring is read from the sound
  (`heardRing`). Past a label's end a voice's or a string's pitch track is the
  next note's: a ring keeps the pitch it was let go at, or a semitone's sits on
  the next note's partials.
- A ring's partials that the next note's hide are bounded by those told apart
  under them, never over: a piano's high partials die first, and bounded by them
  a unison's low partials passed for gone and the next note read 1.1 dB over. A
  ring as loud as the note it rings into misleads the fit as a note sounding with
  it does, by a dB within tight bounds, so a window where a ring is found and its
  partials told apart could move the level past the bound bounds it from above
  only. Held for its partials in doubt too, every window after a man's tone or
  semitone was, his fundamental under a bin from the note before's, and held for
  a gone ring's background a soft note's were: VocalSet lost a tenth of its
  velocities so, where the fit's own doubt costs the one partial.
- Partials closer than a window's bins are nearly one column to least squares.
  Dropping one leaks its sinusoid into its neighbours' estimates, and several
  can blow the solve up together, through partials beyond a pair's reach: a
  partial is its own by its variance inflation over its whole chain, the
  inverse's diagonal, which caught a lone partial fitted at twenty times a
  mix's whole power that the inflation by its neighbours alone passed. Each
  chain of partials within `measuredNear` bins of the next is factored with
  every pair in its Gram: a partial that stops inside the window is wider
  than three bins, and banded the matrix was no Gram, its factor's pivots
  negative and its levels NaN.
- A level read from a note's partials alone misses what it holds between them,
  which the sampler's level counts: NSynth's high keys and some instruments
  hold most of their sound off their partials and read 3 to 47 dB under within
  a bound under 1.5 dB, so the window's power no fitted partial explains goes
  into the high bound. A dump of one note's windows (`lo`, `hi`, its own
  partials' energies) finds a tail note's cause in minutes.
- A note's truth in a room is the note alone carried through that room: `room`
  returns, beside the mix, what carries one sound through the gains, the clip's
  share, the compressor's turn and the reverb the mix met, the noise left out.
  Carry a stem cut to its span: one the loop's length runs the filters on
  subnormal numbers after its sound has died, many times slower.
- A lone low note is measured up to 0.2 dB from its RMS: the RMS is a plain
  window's, in which a low note's partials, a few bins apart, beat, and the
  fit's Hann does not see that beat. The sampler lays the note at its velocity;
  the gap is the window's.
- `measuredBound` is fitted on recorded notes: after a change to the measure,
  run `TestAMeasuredLevelIsKnownWithinItsFittedBound` with `-v` and `NSYNTH`
  naming the folder of NSynth's `nsynth-train` and `nsynth-test`, set the bound
  it fits and write the documents' figures from its log; on a container, which
  holds no NSynth, it reads the engine's voices alone. A fit that checks only
  the groups knowing twenty notes passes by refusing a hard group below twenty,
  so the four families are a group too, which always knows enough. A bound
  that holds on the fit's instruments can miss on the others: NSynth's notes
  sounding with others read 1.13 dB at nine in ten on the fit, 1.84 left out
  and 1.44 on the test split at a bound of 1.2, and a least bound at the fit's
  edge, 0.7 for notes alone, missed for guitar, so the fit is sure to one in
  twenty and the readbacks gate it. A ring is let go by what the bound leaves,
  so a bound the fit moves is fitted again until the fit gives it back: a ring
  given the whole bound fed it (0.8 fitted 1.1, and 1.1 fitted 1.2), so it gets
  what the note's own bound leaves. With the fit's lines a readback's 150, 0.9
  missed by 0.002 dB and 1.0 fitted 1.5, a group's two or three worst notes of
  some 70 known deciding it: tripled after that near miss, 0.9 fitted 0.8 and 0.8
  gave itself back. A line kept on one instrument piled one instrument's bias into
  a family's tail; the lines draw an instrument a note. The test runs about two
  and a half hours on one core at half a core's quota beside a training. The
  measure's time is its projections, which `project` runs as Goertzel's
  recurrence four partials side by side.
- **A by-string file's channel k is its pitch track's k-th string.** GuitarSet's
  JAMS files name `data_source` 0 to 5 in that order in all 360, and its notes
  alone hold the most on their own string's channel nine in ten times or more a
  string (`TestAByStringChannelIsItsString`). A vote by notes a recording named
  the wrong channel for a string in 41 recordings: in a comp few notes sound
  alone, and another string ringing on holds more at a note's partials than the
  note's own rise. audio's `ReadWAV` mixes channels to one, so `readChannels`
  patches the header's count to one and deals the samples out by the file's.
- **A pickup is not the microphone**, by several dB a note
  ([todo](todo.md#tune-the-rip), item 11): a map from a string's
  level to the microphone's, fitted any way, missed the bound on notes left out.
  Look at the same note played again with every other string silent before
  fitting a finer one.
- An end read over exact silence comes late: a window that holds the last few
  samples of a cut note at its edge is found. GuitarSet's channels have a floor
  37 to 59 dB under their loudest windows, and the strings' truth lays one
  (`stringFloor`); a test's fixture with none allows a window.
- A label a tool made carries the tool's offset. VocalSet states no notes:
  its labels are the notes its exercise sheet writes for each take, placed on
  the take by an alignment the lab fitted on sound whose notes are known, so
  its list says `placed` in its `starts` column and its notes ask the onset
  head, and the strike head at their pitch, a rise within the corpus's `reach`. A
  `corpus ... from` run writes `corpus.json` afresh, without it: run
  `placefit.py ... reach` again, as the trainer's refusal says. The engine-shapes
  draws the reach is measured on are rendered at VocalSet's levels
  (`datasets/logs/vocalset-wavscan.tsv`), so a container renders none. Before a corpus trains,
  `tool/rip-train/datacheck.py` with `GROUP=1` reads each source's lag from a
  labelled start to the sound's rise: 0 or 1 frame is a label on its sound.
- A recording's scale says nothing of its labels: `BandFrames` reads it with
  its peak at one, so a float WAV divided by 32,768 gives the frames it gives
  whole (`datacheck.py`'s test pins its report at that scale). What tells a
  label from its sound is the share of its notes and hits in their loop's own
  quiet and the lag at which the sharp plane rises after them, which
  `datacheck.py` prints a source apart with `GROUP=1`.
- `datacheck.py` reads the sharp plane alone: the fine plane hears its low bands
  through windows up to 2.97 s, so a low strike rises there over seconds and
  its rise says nothing of the frame a label puts it on. A frame's loudness, by
  which a note or hit is in its loop's quiet, is its energy over that plane in
  dB: the mean of the plane's 315 values would count each band a sound leaves
  empty, two thirds of them under 1 kHz, and read a note alone as quiet.
- A real set repeated is not more sound: a small set at a large share of a run's
  windows is heard many times over. `--extra`'s shares are drawn in order, each
  of the windows the ones before it left, not of all of them. Before a round,
  count the hours of windows it draws from each corpus against the hours each
  holds.
- `part k/n` counts k from 0: `part 4/4` is refused, and the pass with no part
  then renders that quarter's loops itself, on one core.
- A drum change is benched piece by piece without editing the tree: `stapes
  -config-sounds piece.toml -s bench.slab` overrides one sound of the built-in
  kit. The analysis hears a kick still ringing at 50 to 60 Hz as the bass line,
  and a snare's fixed modes move the melody's key.

- A frame model's heads start with audio's `FrameSpec`'s, and any after them
  are read by nothing, as a notes model's heads are read by name:
  `ripmodels.New` refuses a frame model whose heads do not start with them
  ("frame model's heads are [...], want [...] first"), so after a head audio
  reads is added the weights trained before it load in no audio that reads it.
  A head no decoder reads goes after them, as `struck` does, and loads in audio
  from v0.34.0, which keeps the prefix rule; an older audio wants the heads
  exactly. Install frame weights beside the audio tag whose `FrameSpec` they
  start with; `--init` grows the head onto the weights from before it.
- **A rendered loop's beats are in its label**, where audio's `BeatStarts`
  puts them under the bars' tempos (`music.label`), and the trainer reads them
  as it reads a recording's, asking every frame of a rendered loop but its last
  two (`train.Label`). A swing, a drift or a pickup moves the label with the
  notes; what the label cannot see is a render that sounds a note off its
  expected event, which audio's `TestALateStepSoundsWhereItsEventSays` holds for
  late steps under drifting tempos and stapes' performance test for the kick
  on every bar line under drums.
- **The arrangement's tests share one label corpus** (`theLabels`, 2,000 loops
  drawn once): drawing it closes a session of its own and resets audio's
  registries, so a test calls `labelCorpus` before it opens its own session. The
  figures [Rip training §3](rip-training.md#arranged-loops) quotes are those
  tests' logs: after a change to the arranger, run them with `-v` and write the
  figures again from what they print.
- A part on a recorded instrument keeps to that instrument's pitches, so a
  sampled guitar's chords can sit over a left hand drawn from a keyboard's
  floor: the left hand folds under the chords' lowest, and chords with a left
  hand keep two octaves over their instrument's lowest key. A note is cut to the
  loop's span before the performance moves it, and one played early then starts
  the loop, or a pickup's first note vanishes at frame 0.
- An overlap's notes last whole steps of its 120 bpm: a label's `end` is the
  step's, so a ring of notes the generator counts must use the step's length, or
  the label shows more sounding at once than the generator allowed.
- A corpus cycles six tempos (`tuneTempos`) and its last loops are the ones
  held out, so a held-out loop's tempo was always trained on: a beat head's
  held-out F1 says nothing of a tempo it never met. `tempos a,b,c` takes a
  list of any length, and `--bend` sways a window's tempo.
- A change that must not move the corpus or the bench is read before and after
  with an empty root and `XDG_CONFIG_HOME`: the hash of `corpus <dir> 24 4 aug
  gen meters overlaps nowav`'s files, `bench 12 4` and `bench 24 16 meters`.
- A gene at its range's floor is not near the voice: an unset parameter reads
  as gene 0, so a draw about it held to 0..1 can only raise it, and turned a
  sub, a filter or FM on in half the redraws. `VoiceFromGenes(w, v.Genes())`
  builds an unset field back at its range's floor (`cutoff` 50 Hz, `ring` 0.05 s,
  a bank of a few partials), so a voice moved to another wave by its genes is a
  click. `redraw` moves a set field by its row of `redraws` and takes back from
  the round trip only the fields it moved; a new wave's fields are a registered
  voice's of it.
- A redraw sets its level by two notes of 0.06 s it renders (`loudness`), and a
  labels pass draws every loop's voices: the 2,000 loops of `corpus <dir> 2000 4
  gen meters overlaps aug glide labels` take 3.6 s, half of it those notes.
- `onPitch` reads A0 played 70 ms as two periods through the damper, its share
  moving from 0.88 to 0.98 as its window's start moves 20 ms. The lowest of
  thousands of redraws scattered around five voices falls under the lowest of
  the five (33 of 36,400 did, all A0 at 70 ms, down to 0.940), so the rule's
  bound sits halfway in dB between their lowest and an even split.
- A recorded kit's hits are put on its sound by the frames' own rise
  (`placeHits`, [Rip training §3](rip-training.md#a-recordings-lag)). A hit
  train laid against a steady pattern meets nearly as much rise a pulse off as
  on it: over every lag of a four-bar loop a bar's shift scored within 1% of
  the truth and beat it where the lag laid dropped hits off the start, so a
  lag is looked for within half the usual gap between strikes, and a rise at
  the reach's edge is no peak. A kit's own rise sits up to a frame from its
  hit by its length and its tone, which is most of `lagBound`. After a change
  to the measure run `TestARecordedKitsHitsAreMovedOntoTheirSound` with `-v`,
  set the constants it fits and write the documents' figures from its log;
  two thirds of its seconds are the kits' renders, three a kit. Its takes are
  laid at a drawn level before their room: the frames read a take's peak at
  one, but `degrade`'s soft clip is driven by the level it comes at, and at the
  bank's own level a cut of every drum's norm to 0.62 moved the rise a step.
- **No offline render passes the limiter**: `RenderBars` and `RenderMusic`, and
  so every corpus, bench and test loop, sum the sequencer's slots alone; the
  mixer's limiter is the live stream's. On the parent bank's levels 85 of 500
  generated loops of four bars peaked at its 0.9 or over, 34 at full scale; on
  audio v0.35's none, the loudest at -1.07 dBFS.
- **The analysis hears a recording's scale.** Its percussive flux is
  `log1p(100·magnitude)` of the samples as they come, so a gain moves its
  marginal hits (a steady beat's 60 read 58 at half its level), and so does the
  16-bit truncation of a WAV written from a float render: a test whose truth is
  a rip rips the file the count reads, its bytes read back.
- A hit that draws no sound is told by the rise its own onset reads
  (`onsetRise`, [Rip training §3](rip-training.md#a-hit-that-draws-no-sound)).
  The sharp plane's window is 93 ms, so a hit's rise spreads over about two
  frames either way: a silent hit nearer a sounding one reads that one's, and
  a loud snare's tail reads up to 6e-4 seven frames on. A measure over the
  recording (its median strike, its loudest) parted silence from soft hits no
  better than the rise in the frames' own units. The fit's silent side has a
  long tail: 12 recordings read 2.5e-3, 16 and 24 read 1e-3. After a change to
  the rise or the lag run `TestAHitThatDrawsNoSoundIsToldByItsRise` with `-v`.
- **A kit's pieces register under `corpus_piece_<n>`**, each loop's piece n
  over the last's: a name new to the process through `DefineSound`, which
  renders it and sends it to the mixer, a known one through
  `audio.RegisterSound`, which puts the new definition's pointer in the
  registry. The engine keeps a drum's offline takes by that pointer
  (`offlineTakes`), so a piece never plays on the last one's takes. Binding a
  piece (`SetKit`) renders its 18 takes, 0.3 s, most of a kit loop's render;
  its band renders one hit more, 20 ms, which a labels pass pays too.
- Passes of a kit sum to the loop whole because no arranged step draws: the
  engine's `Prob` and `Humanize` draw from one stream for every track, so a
  pass that empties a track would move the others'. The arranger sets neither.
- A set's labels and list are made and checked by `tool/rip-lab`
  ([its README](../tool/rip-lab/README.md)); what its tools found wrong in the
  sets' own files is in [set-issues.md](set-issues.md).
- A voice's notes cut from its pitch alone are no labels: a slide and a
  vibrato's swing come out as notes. Where a set's sheet says what is sung,
  the sheet's notes are placed on the take's sound, and a take that does not
  fit is left out and counted.
- A period's measure reads a loud low sung note the octave over, a whole note
  long, where its second harmonic stands 11 dB or more over its first: the
  sheet says which octave is sung.

## Training rounds

- A training round runs from its own copy of the tree: `pipeline.sh` builds
  stapes from the checkout at every stage, so an edit to the checkout while a
  round trains changes the corpus or the bench under it. The bench draws its
  loops from the tree's code too, so rounds whose `aug` or `gen` differ bench on
  different loops: bench one round's weights through the other's stapes
  (`STAPES`).
- A model's decoder thresholds belong to its weights: a new head's scale moves
  them, so tune them (`tune ... truth <list>`) before judging a round, and pin
  them in the root's `rip.toml` beside the weights. Compare rounds on one tree's
  bench loops. The drums' `onset_min` is one: a share of the recording's loudest
  frame, so a surer drums head wants a higher bar. The search sets it, and the
  bars' weights, only where its list's truths give hits and bars: give it
  rendered loops' labels beside the recordings, at a seed no training corpus or
  evaluation set uses. Never read a new model through the old one's `rip.toml`.
- With strikes the melody register never reaches the pair decoder, so
  `second_voice` and `cross` move no note on the models' path, while the bass
  line's own path reads `jump`, `switch` and `retrigger`: `modelGenes` is what
  audio's strike decoder reads, its heard starts' five since v0.39.0; check it
  against a new decoder. A candidate of `tune ... truth` pays the decoders
  alone. Frames kept by `frames <dir>` before v0.39.0 hold no starts and decode
  as if the notes model had no onset head, so the search sees none of those five
  move.
- A round that gains on the bench may still lose on the recordings: the bench's
  loops are the corpus's own kind. Judge by the recordings.
- Towers (`transformer.Tower`, `TOWERS=` in `tool/rip-train`) are not dead code:
  they are to be tried once new heads (offset, velocity, pedal) compete with the
  strike head for the shared layers. Keep them, their loader and their
  round-trip test.
- A notes model's per-pitch heads are rows of two tables, `CELL_HEADS` in
  `ripmodels.py` and audio's `notesHeads`: a new head adds a row to each, with
  the values it reads at a pitch, and goes after the others in
  `PITCHNET_HEADS`, since `--init` grows a model only from weights whose heads
  are the first of the new one's. The file's version holds while its header
  does; an older audio reads the heads it knows by name.
- What plays each pitch is three more of those rows, and the trainer reads a
  note from them as audio does (`said`, `noted` against `RipFrames.tell`):
  the cells' say weighed by salience, over the cells among a frame's twelve
  most salient (`VOICE_CELLS`, audio's `RipVoiceCells`). Change one side and
  the held-out line no longer says what a rip will. The rules are tests with
  no corpus: `python3 -m unittest discover tool/rip-train`, which CI, without
  PyTorch, does not run.
- Two runs of `train.main()` in one process number their voices' genes on
  from each other, as `GENES` grows with every `Label`, so a cell's
  `pitchvoice`, a row of it, differs between them though the genes are the
  same: a test that compares two runs' batches gives each a fresh `GENES`
  (`handed`), as a process of its own starts.
- A head no window asks still moves, as the shared layers under it move: a term
  that takes a head's output through an exponent takes it only where the head is
  asked (the cents head's stated error, `torch.where(asked > 0, spread, 0.0)`),
  or a head drifting where nothing asks it overflows, and inf times a zero mask
  is NaN in the whole loss. A head that moves the shared layers' path shows it
  first in the test that teaches a note at either end of the keys.
- `evaluate`'s keys are printed as the line's parts unless `scored` leaves them
  out by name: a reading that is no head's loss takes a key no head has
  (`strikes`, `hits`, `beats`) and joins that list, as a head named alike would
  overwrite it.
- A family's name is learned long after its sound, the wave and the genes moving
  from the start. Each engine voice far from the bank is a synth, so read a
  family's own share, never the total.
- The heads of what plays a pitch teach the cells the decoders read. With the
  gradient stopped at the cells they learn only the commonest name, and at the
  whole weight of the frame's like heads they cost the strikes, so they weigh a
  quarter (`PLAYS`). Before changing that weight or those heads, run the same
  steps without them and read the strikes, beside a second run taught alike: the
  starting draw alone parts two runs.
- Generated loops cannot judge a family's name: a piano or a guitar there is an
  engine voice within 0.2 of a bank voice's genes among random ones on the same
  waves. Read names on recorded families; a container run shows that the heads,
  the measure and the file work, not how well they hear.
- A notes model of a few thousand steps rips few notes at the shipped strike
  thresholds. To see what a rip says of such a model's notes, set `strike`,
  `strike_low` and `strike_again` in the root's `rip.toml` and say so beside the
  numbers.
- A weight file's constant can ride as a frame head of no weights (`sure`:
  `export` appends a zero row to `frameout.weight` and the constant to its
  bias), which no version of the file needs to know; `--init` and `average.py`
  carry the notes model's buffer, and a run again on a finished checkpoint exports
  its pick with a new `--sure`.
- The cents head states its own error, and audio joins two cells where their
  readings agree within the errors stated (`oneFrequency`), with no factor on
  them: a head too sure of itself merges steps, and nothing downstream corrects
  it. The stated error is fitted to the offset's miss with the miss held fixed
  (`losses`), so read the held-out line's `within` before a round's notes: near
  63% is a head that states its mean miss, and the gliding readings are the
  ones the decoder leans on. Weighting the miss by the stated error, or a
  margin in the decoder, would put a number set by hand back in the rule.
- A fine-tune reads as a loss under the installed scores, which were set with
  the start's heads in hand: read each model under a judge fitted on its own
  rows.
- A probe of a model's heads reads what its frozen cells already hold, and what
  the heads are asked bounds it as it bounds a free run.
- A name the corpus gives by a voice's genes is not one its sound says: named by
  the voice they were drawn under, or `synth` far from the bank, modal and
  string sounds were asked two names, so a note the engine rendered asks no
  family. Count a name's cells by family and source before training for it.
- A training loss that keeps falling says nothing about fresh loops. `train.py`
  holds a tenth of the loops out, scores them every `--every` steps and exports
  the best step; judge a run by its held-out line and by `bench` on the file,
  never by the last training loss, and prefer more loops to more steps.
- A `.pt`'s held-out loss, a merge's or any checkpoint's, is read without
  training: `train.load_corpus`, `train.validation` over the last `--holdout`
  loops and `train.evaluate` in a scratch script, the network built from the
  file's `cfg`. `--steps 0` scores nothing, since the score runs in the loop. A
  `pip install --user` torch is not seen under `python3 -I`, which drops the
  user's site directory.
- `train.py` reads each window's frames from its `.bands` as it draws it
  (`FrameFile`) and builds the window's targets from the loop's events then
  (`Label.window`): rows per frame held up front came to 207 bytes a frame, 5.2
  GB for 40000 loops. A `np.memmap` a loop would keep a file descriptor open
  each, 40000 of them, past a 1024 limit or macOS's 256.
- A training run shares the cores with everything else the session does, and its
  steps a second follow what its builders get. A batch is Python holding the
  interpreter's lock, so two threads build less than one, and `--builders` forks
  processes, a core each, each paying to hand its batch over. Size a waiter by
  the run alone and expect to re-arm it, or follow the run by a reminder
  instead.
- **A builder shares the run's pages until a write copies one, and a read writes
  an object's reference count.** Once it has drawn from every loop a builder
  holds its own copy of each loop's `Label`, arrays' and `FrameFile`'s headers,
  the arrays' data still shared. A full collection would copy every object's
  page, so the run freezes the collector while they fork. Read a builder by
  `Private_Clean` and `Private_Dirty` in `/proc/<pid>/smaps_rollup`, its worst
  case by touching every loop in it and collecting.
- A run's noise differs by set, so a difference on one set is read against that
  set's own spread. Two runs that differ in anything else are not a repeat.
- `--validate` picks a notes model by its strike head's peaks at the decoder's
  bars, not by its loss on recordings, which holds its onset head's, which the
  pick leaves out though audio reads it beside the strikes. One instrument's
  score can trade against another's, so
  validate on every kind of recording at once. A frame model is still picked by
  the decoders' loss there, in a pass that carries on the sum of the heads audio
  reads from it, which can rise as its onset head grows surer while
  its beats' F1 rises. Read a line at its last step, each head by its own score
  on the recordings, not by one sum. A round's `.bin` is its pick's export: a
  line's file is its `.last.pt` through `average.py --out`.
- `ripmodels.features` over a whole long recording overflows: its PCEN average
  is one running sum over growing weights, past float64 near 24,000 frames
  (284 s), and every frame after is NaN under a RuntimeWarning alone. Hear a
  long recording in chunks with the trainer's 128 frames before each; a
  training window never reaches it.
- A model run over a whole recording in Python takes memory to match. Chunk it
  (3,584 frames with 256 of context) and run any long job as a capped unit
  (`systemd-run --user -p MemoryMax=...`), where running out stops the unit and
  not the session.
- `tune ... truth` hears every recording of its list before it weighs a
  candidate: give it `frames <dir>`, and a later run reads them back. A file's
  name holds the recording's and the weight files' checksums, not the models'
  code: clear the folder after a change to `ripmodels` or the transformer.
- A fit over a list of one kind can gain on it at every other's cost: scored
  within lines over a list of guitar alone, a search can silence the bass line
  (`strike_low` near 1). The guard refuses that only for conditions its list
  names: list every kind a fit may hold, and decode a tuned `rip.toml` over
  every condition before it ships.
- `tune ... truth` weighs a condition, not a recording: a set scores the plain
  mean of every score its conditions carry (`scores`, `guarded`), a condition's
  notes pooled over its recordings. Give a condition recordings enough to stand
  for its kind, and read a searched file on recordings its list left out.
- **A search writes back what it moved by chance unless each gene is put back.**
  A near pool moves one gene in four, and a score sees only some of the genes it
  is given: a model without the two heads reads no `beat_comb`, heard beats no
  `bar_beat` or `bar_move`, a loop under six bars no margin of how a reading
  repeats. `tune` through the analysis draws the heads' weights across their
  ranges. `search` puts each moved gene of its best back alone and keeps it back
  unless the score falls (`settled`); a new gene needs no rule of its own.
- A MIDI truth gives the search notes and drum hits, never bars: a
  transcription's file holds its writer's default signature and tempo
  (GuitarSet's, MusicNet's), and the search would fit the bars to it. Bars
  come from a corpus label: a rendered loop's listed beats, or a recording's from
  its list's `beats` column (`corpus <dir> from`).
- A test's small random model hears no drum hit and a bar at most of a
  two-bar loop: a test of the drums' or the bars' score hears through the
  analysis (`analysisEar`), which reads a rendered loop's.
- The analysis hears drums where none play and drops played ones as fast: a bank
  loop without its rhythm slot reads its bass as kicks, so a search rightly
  holds the bar there: a hit heard where none plays weighs what a false hit does
  in as long a recording with drums. A test that wants the bar to rise takes a
  beat of one velocity alone as its recording with drums.
- A pool drawn near its start suits `tune ... truth` and not `tune`, so `search`
  takes `near` from the truth search alone.
- A test of `tune ... truth` cannot count on the pool: a near draw moves a
  gene one time in four, a rip of the tiny model's 4.3 s loop costs 0.1 s, and
  a recording under four seconds is refused. Force the outcome in the fixture
  and read it over seeds: a truth the start hears exactly leaves nothing to
  beat; from `strike_hold` 0.3 the tiny model's loop is bettered by `strike`,
  `strike_low`, `strike_again` or `voicing` lowered, its `strike_hold` moving
  nothing, and a pool of four wrote nothing for four seeds of sixteen, one of
  eight for none. A gene added to the list, however idle, moves every draw.
- A drum's loudest onset is often its first stroke, rising from the render's
  silence: in a bank loop each drum's first alone stood over 0.75 of its
  loudest, so from `onset_min` at its range's top no near step heard another.
  A beat whose strokes grow evenly louder holds hits under every step from it.
- The live rip's notes are the offline rip's and its bars are not: tempo, meter
  and phase hold from the first six bars, whatever the ear waits, and each frame
  runs through each model once, however the ear's half-second steps overlap.
- A tool that names its corpus in its source dies with the corpus: take the
  corpus, the checkpoint and the first loop as arguments.
- **A judge's whole path runs in ninety minutes on four cores**, nearly all of
  it the notes model: `corpus fit 360 4 aug gen` and `corpus held 40 4 aug gen
  seed 2`, both with their audio; a frame model of width 64, two layers, 2000
  steps; a notes model (`--pitchnet`), 2500 steps; `rows` of the 400 loops; the
  fit; `rip` and `truth ... timed` over the 40 left out, against their labels
  written as transcripts. It shows that the path holds and nothing of a judge's
  worth: the loops left out are the fit's own kind under another seed, and so
  short a notes model reports few of the truth's notes.
- **A weak beat or downbeat head costs the bars where audio takes its word.**
  Since v0.25.2 audio reads the heads by level as a plain tracker does, the
  downbeat head beside the cues. Read a round's bars with its heads and without
  them before its file is installed.
- The lines' F1 does not say which beats a head misses. Print what each head
  says within a frame of the loops' own beats by their place in the bar
  (`train.Label.beats`, the frame model over a whole loop).
- A model's heads are left out of a rip through the same weights by a
  `RipFrontEnd` that wraps `ripmodels`' and sets the frames' `Beat` and
  `Downbeat` nil, in a scratch module on audio's tag over `Ripper` and
  `MatchBars`, the truth the bar lines the loop's label lists; or by
  the same step exported with the output layer's last two rows cut, which stapes
  benches. The two rip row for row alike. `BarScore.Phase` is a median folded
  into a bar, so half a bar off reads as anything between: count `Downbeats`.
- **A checkpoint states its layout, and one of another or of none is no
  start.** `train.py` and `average.py` refuse it before reading its shape,
  naming the band frames' layout (`ripmodels.refused`), so no run grows or
  resumes a model that heard other frames.
- A notes model of width 8 taught one loop, an A0 and a C8 twice each, for 150
  steps of one 64-frame window: at a rate of 0.01 both keys read 1.00 over
  their notes, struck at 0.97, and no other pitch over 0.001; at 0.03 the A0
  read 0.43. The trainer's test of the piano's two ends runs at 0.01.
- A frame model's probe (`--probe` from a model without the beat and downbeat
  heads) is checked on real weights by its held-out line and by a rip: the other
  ten heads' losses print to the digit as the start's, and its file ripped with
  the two heads left out is the start's row for row.
- **A level set against the sharp plane's summed energy is set against a low
  note many times over**: its bands share bins under about 270 Hz and count a
  tone about twenty times at A0 ([audio notes](https://github.com/lixenwraith/audio/blob/main/doc/agent-notes.md#front-ends-and-fidelity)).
  Against the plain sum a ring stood 13.6 dB over its stated level under a low
  attack and on it under a treble one; each band over `ripmodels.sharp_counts()`
  reads a steady tone from A0 to C8 within 0.6 dB of its energy, noise within 3.
  Energy read back from stored frames (`np.expm1`) is already a hundred times
  the energy, so what is laid from it takes no other hundred.
- **A pass that carries on is judged by readings of a few loops, which a toy
  corpus starves.** `TheRate` renders 40 loops of notes over a kick and a hat, 8
  held out, at batch 8: with 12 loops or batch 2 the readings' own noise hid
  carrying on from a restart, and over loops without drums the drums head sits at
  its floor, where a loss near zero drifts past a noise near zero, which is why a
  gain at the fine rate is the decoders' sum alone.
- Two runs alike up to a step read alike there, so their paired noise is near
  zero: a test that a head moved alone leaves the others where they were reads
  each head's difference between the runs against that head's own noise since the
  raise began.
- A reading line's parts end it, `(onset ...)` and then `(best)`, and the lab's
  readers anchor on that: what a line adds goes before its parts, as the rate's
  say does.
- `torch.load` reads with `weights_only`, which refuses a NumPy array: the rate's
  readings ride in the checkpoint as tensors. `Optimizer.load_state_dict` keeps the
  tensors it is handed and AdamW steps them in place, so a state kept to go back to
  is a deep copy, and so is the one loaded from it.
- A pass that carries on, resumed, reads its rate back whole, and its weights and
  lines match the unbroken run's: checked by raising `KeyboardInterrupt` from a
  patched `train.losses` between readings and running the same words again. No
  test holds it.
- From a checkpoint of today's schedule, its rate near zero at the end, a pass that
  carries on at half `--lr` can read worse at its first reading on a toy corpus:
  the fall back halves the fine rate until readings hold, and the next pass carries
  the rate found.
- **A reading's paired noise is how far the weights moved.** Two readings a raise
  apart differ by the raise's whole way, so a reading kept left behind a kept
  raise registers no later gain: the reading kept moves with the weights.
- **The rate's rule is read with no model first** (`test_train.Replay`): each
  loop's loss a head, moved by the rate, down by a gain and about by a jitter,
  or laid by a test to a level and a noise. Chains of passes on it say what the
  rule does with such readings in a second, and nothing of what a chain's
  readings are: replay the lab's checkpoints before trusting a change.
- **A line's masked losses are means over windows, an empty window a zero.**
  `pitchfamily`, `pitchwave`, `wave`, `velocity`, `family`, `cents`, `beat`: a
  window that asks nothing of a head reads 0 and counts, so the line's figure
  sits under the loss a cell by the share of such windows. The names' is printed
  a cell apart (`names`); pool any other the same way before holding it to a
  bound.
- **No decoder reads the voicing head**, from either model: audio's `read` fills
  no plane from it, and `RipParams.Voicing` is a bar on the salience. It weighs
  a half in `DECODERS` all the same, and is in no pass's sum that carries on.
- **A grown long convolution is read by how far each head moves before a step**,
  through the start and the grown model over a batch of engine windows, each head
  as audio reads it; the frame's heads stay to the bit, hearing the wide layer
  and never the cells. On a checkpoint of 2,000 steps zero weights moved the
  cells 0.198 in the mean square and the strike head 0.169 at the most, the fit
  0.031 and 0.054; every weight of the three fitted to the same cells by LBFGS
  reached 0.029 in 150 evaluations, so the taps across channels and frames buy
  little. The held-out reading before a step barely parts them: decoders 0.661
  the checkpoint's, 0.662 fitted, 0.666 at zero.
- **`start_long` fits a channel at a time.** One LBFGS over every channel's
  scales and shifts stalled at 0.039 after 200 iterations, one line search for
  curvatures that differ by channel; each channel's own reached 0.032 within 50
  and 0.031 within 200, 13 s for 65,536 cells a channel on four CPU cores.
- A full-shape notes model trains on four CPU cores at 3.5 s a step of 8 windows
  of 512 frames, 4.2 s with `--dilations 3,9,27`, and at 0.75 s on windows of
  128: a container's checkpoint to measure against trains on the short windows,
  2,000 steps in 25 minutes. A run beside it on the same cores slows both many
  times over: the trainer's tests took 646 s so where they take 80 alone.
