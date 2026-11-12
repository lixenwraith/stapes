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
- A cloud container holds no PyTorch: `make venv` installs the CPU build in
  two minutes, and `tool/rip-train/.venv/bin/python3` runs the trainer and
  `judge.py` from then on. Where the container cannot resolve
  `download-r2.pytorch.org`, which the CPU index's links name, `make venv`
  stops at the wheel: the index page lists each file with its sha256, the same
  file is at `https://download.pytorch.org/whl/cpu/<file>`, and `pip install`
  takes it from disk once `sha256sum` agrees, then `requirements.txt`.

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
- **The ear's short wait shows in the drums, not the notes.** Through 12a and
  32a, audio's `FrontEnd.Ahead` at half a second leaves every pitched head as
  the whole recording's (they are the notes model's, whose reach it still
  waits) and moves the frame model's: the onset by 15 to 20% of its rms, the
  drums by 30%, a reading by nearly 1 at worst. The stream's notes count the
  same under both waits, so count its kicks and snares a bar.
- **A thread's evidence is random weights.** A cloud container holds no
  trained weights and no recordings. After threads merge, `live` over a few
  recordings through the installed pair, the offline rip's note counts against
  the last reading and the two `corpus` digests take minutes and say whether
  the merge holds on what the rip is for.
- **An installed root must rename its frame model's file to `ripframe.bin`.**
  A root from before v0.23.0 holds `ripmodel.bin`, which stapes names at the
  start ("frame model: ripmodel.bin is now ripframe.bin, so rename ...") and
  never loads, so the rip hears with the analysis, the notes model and the
  judge unread, until it is renamed: Lixen's root and any `-config-dir` a
  recipe names. The archived names under `weights/` keep theirs.
  `pipeline.sh`'s checkpoint is `ripframe.pt`, so a round stopped under the
  old name resumes with `CHECKPOINT=` naming its file.

- Write a truth from a score or a MIDI file, never from memory: a remembered
  Billie Jean riff had its first note an octave low, the synthesizer's doubling
  the analysis hears, and turned a 96-to-94 tie into a 110-to-84 loss.
- Score the three recordings pooled (every note once) beside their mean: the
  origin and the intro have 83 notes each, where five notes move F1 by 0.06.
- The test loop scores 0.98 under `DefaultRipParams` and loops of the shipped bank
  0.51 (`CompareRips.Total`): busy drum lanes whose chance hits count as misses,
  several tracks a slot and melodies under chord changes are where the rip falls
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
  MusicNet .61 and .71, its solo cello .47 and .77. `truth`'s `notes` row and
  `tune ... truth` count by pitch and onset (`Agreement.All`); `truth`'s rows
  by line, its `every part` row and the wide table still count within lines.
- A truth on the recording's own clock (GuitarSet's) needs no time map, and
  the matcher's costs it: strummed guitar by pitch and onset read .72 through
  audio v0.14.0's map and .79 without. The map has held a steady tempo since
  v0.24.2, unread on guitar. Say `timed`, to
  `truth` or as a list line's fourth field; a timed truth's drum hits still
  pass through `MatchHits`' own map.
- The voice a model hears for a line is the fit's first seed, no more: modal at
  the heads' mean genes, it stood farther by the fit's own distance than the
  nearest bank voice in 32 of 37 lines (ten of NSynth's held-out instruments
  and five recordings) and 2.3 times as far as the fit, and on the six
  recordings Lixen heard the fit was the nearer every time, so `rip` fits every
  line. The distance favours a sparse, dark voice (modal for 32 of the 37, a
  singer and a reed among them): a change to it wants ears again.
- A miss is rarely a note the notes model never heard: on strummed guitar 78% of the
  notes the rip misses have a strike over `strike` at their pitch within four
  frames. No bar or rule returns them at a fair price, though: `strike_again`
  at .7 brought back 44 for 130 more reported, the partial and within-a-tone
  rules off about 30, the accounting off 43 for 323 at .03 off the mean of
  twenty conditions. The head hears as many false strikes at that level, and
  nothing after it tells the two apart. Print the strike head at the missed
  notes (`vif/tmp/audit2/why-v0140`) before training for a miss, and try a
  rule on every condition (`cond-v0140`) before moving it.
- **`Fidelity` puts no price on a wrong note**, so it cannot set a decode's
  thresholds. Over 91 recordings under eleven threshold sets it ordered 54 in
  a hundred of the pairs .02 apart in F1 as their truth does (51 through the
  models), and a pick by any of its terms scored under the installed set. The
  installed decode laid again without its wrong notes gains .09 of F1 and
  loses `Fidelity` on 71 of 87 recordings: a wrong note is worth a third of a
  right one to it (a tenth on a solo guitar's line, two fifths on a sung
  one, a half on chords), wherever it lies on a partial, a ring or a glide
  the recording has. It sees a gross fault two times in three
  (`vif/tmp/audit2/fidtable.py`, `fidxtable.py`).
- Two renders compared by `Fidelity` are rendered under the same names, each
  in a fresh session: a sound's noise is seeded by its name, and one decode
  rendered under two names scored up to .004 apart.
- **A judge's fit is read with a collection left out, never a recording.** A
  small network on the heads around a strike took guitar from .73 to .84
  with the player left out and to .79 with GuitarSet left out; singing lost
  up to .04 unmet. The accounting's logistics refitted on 427 recordings
  gained nothing, and no second pass over a recording did. The frame model's
  drum onsets and the note's length added to the network's input moved no
  collection by .015 (`vif/tmp/audit2/README.md`, the judge's tools).
- Billie Jean's truth is its bass line alone, so its F1 counts every chord
  the rip hears as wrong: read it by bass notes right (111 of 138 clean). A
  judge fitted on recordings, which hold no drums, kept 49 to 76 of them;
  fitted on the engine's mixes alone, 102.
- 32a trained on vocadito's clips 1 to 30: its singing corpus
  (`~/ripcorpus-vocal`) holds them beside VocalSet, 30 loops of 2,346, which
  a look at every fiftieth label missed. Search every label's `source` before
  calling a set unmet. GuitarSet, MusicNet, vocadito's clips 31 to 40 and the
  takes are unmet by it (its log names corpora 15, 14 and 11, the singing
  corpus and E-GMD). A round that trains on one of them takes it out of
  every held-out reading, a judge's fit included.
- **A full track only measures.** Billie Jean, the pop set and the piano
  takes are never in a corpus or a fit's list: no licence covers training on
  them (AGENTS, Data). The audit's scratch fits of a judge held Billie Jean's
  rows whenever another collection was left out; nothing from them is kept.
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
  `judge.py` repeats byte for byte on CPU (three fits of one directory wrote
  one file), and on audio's own rows of six left-out loops, 1,869 strikes,
  the two sides' odds differ by at most 3.2e-07 with no verdict apart. A
  scratch program rips a WAV under the root's judge with `Ripper.Strikes` and
  `Rows` and dumps each strike's stage, odds and row; the file's layers
  applied to those rows in NumPy give the other side. Redo it after any
  change to the file, the row or either forward pass.
- **A count from rows is near `truth`'s decode, not it.** Over 40 left-out
  loops, 2,838 notes given, the installed scores read 423 right of 761 heard
  from rows and 424 of 762 decoded by `truth` (F1 .235 and .236), and a
  judge 740 of 1,028 and 759 of 1,059 (.383 and .390). Installed, the
  difference is one loop's and one cause: the installed scores weigh the
  candidates that stand in time order, and the read lists the strongest
  strike of a pitch within `strike_gap`, which carries the verdict of the
  candidates near it and is labelled at its own frame. Two candidates
  returned 17 frames apart around one stronger strike are one row, and a
  candidate returned eight frames after a stronger strike is a right note
  whose row is labelled wrong, 104 ms from the truth's. Under a judge a
  dropped strike is measured against the notes the judge kept, where its row
  was written against those the installed scores kept, so some verdicts
  differ. Read a fit by `truth` over the conditions left out, and use the
  rows' count to see that a fit is sane.
- Singing is the last to come: little truth, many kinds of voice. Report it
  beside every reading and hold no stage back on it.
- Lixen's order of work: fix what is already known to help, train a little,
  then measure with focus. Do not measure further ahead of a round to pin
  every open piece first.

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
  stand against that, not against the third run's table. The frame model's corpus
  is rendered at A440, so nothing it saw was detuned; a recording off the
  standard is a recording it was never shown, whatever its held-out loss.
- `corpus ... glide` draws its bends and scoops from one stream and its
  vibratos from another, each seeded apart from the loop's own, so a loop's
  patterns, key, voices and tuning are those the line draws without the word,
  which a test compares. A label's `cents` is written only on a note that
  moves, so a corpus without the word is byte for byte what it was: a new label
  field stays out of the file where it is unset. A voice with a vibrato of its
  own (the vocal focus's) moves without the word, and its labels gain the list.
- A `midi` corpus holds glides without the word: a MIDI file's pitch bend is a
  step's `bend`, which the engine plays, so such a note lists its `cents`.
  Labels written before the cents head call it steady: in corpus 17's first
  24 loops one event of 3,486, the frames byte for byte the same. The line
  with `labels` after it writes them again and renders nothing.
- `corpus ... labels` plays nothing, so nothing a label holds may wait on the
  mixer: `DefineVoice` queues a command and fails once 256 wait unplayed,
  which dropped voices from a pass's labels from its loop 34 on, 166 of 200
  differing. The pass checks a voice with `ValidateVoice` and registers none,
  so the registered voices stay as they are and need no restoring. A label takes
  only its length from the render, read from the `.mel` header; a field a
  label gains must be drawn before the render, or the pass cannot write it.
  2,000 loops are labelled in under a second, against 11.6 minutes to export.
- A `from` corpus's labels pass knows a recording by its label's `source`, so
  a label that does not read stops it. It writes the MIDI's `vel` again:
  strip a constant-velocity corpus's once more after it (below).
- A pitch track at 44.1 kHz and a hop of 256 has two samples a frame, one on
  the frame's own time, so a missing sample lies a whole step from its
  neighbours. A reach of a step read a neighbour's pitch there by a rounding's
  luck; the reach is three quarters of a step (`pitchTrackReach`).
- A tracker's jitter makes every frame unlike its neighbour, which is how the
  trainer knows an engine note's glide: by that rule all 258 readings of a
  jittered track were gliding. A recorded note glides when its track spans
  half a cell (`GLIDE_SPAN`). A change to how `train.py` reads a label is
  checked by `Label.window` before and after over a `glide` corpus: its
  targets from engine labels must not move.
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
  frame wide teaches MusicNet's jitter and a loss there measures it. Snapped to
  32a's peaks, models outside its line find 80 to 88% within a frame; a rendered
  loop's exact labels would only take on 32a's frame of play, so only MusicNet
  is snapped (`vif/tmp/round37/tools/snap.py`).
- `part k/n` counts k from 0: `part 4/4` is refused, and the pass with no part
  then renders that quarter's loops itself, on one core.
- A drum change is benched piece by piece without editing the tree:
  `stapes -config-sounds piece.toml -s bench.slab` overrides one sound of the
  built-in kit. The analysis hears a kick still ringing at 50 to 60 Hz as the bass
  line, and a snare's fixed modes move the melody's key: the bench's clean bass
  rests on three loops of twelve, which a kick of the old 42 Hz glide restored.

- A frame model's heads are a prefix of audio's `FrameSpec`, unlike a notes model's,
  which audio reads by name: an audio from before a head refuses a frame
  model's file that has it ("frame model has 12 heads, want 10"). Take the audio tag
  that knows a new head before installing weights trained with it; weights
  without it load in either.
- **A rendered loop's beats are in no label: the trainer derives them** from
  `bpm`, `bars` and `meter` (`train.Label`), by the arithmetic `exportCorpus`
  lays an event with, a sixteenth of whole samples and each frame rounded
  half up. A change to how a step is placed (a swing, a tempo that moves
  inside a loop, a pickup before bar one) breaks the rule without a word and
  no test sees it: redo its check, a scratch script over `aug`, `codec`,
  `midi`, `dense`, `meters`, `glide` and `tempos` corpora that asks that every
  event's frame is on its loop's sixteenths and the end's line within a frame
  of the `.mel`'s length (0 of 24,810 events off, the line the last frame or
  the one past it).
- A corpus cycles six tempos (`tuneTempos`) and its last loops are the ones
  held out, so a held-out loop's tempo was always trained on: a beat head's
  held-out F1 says nothing of a tempo it never met. `tempos a,b,c` takes a
  list of any length, and `--bend` sways a window's tempo.
- Baselines for a change that must not move the corpus or the bench, taken at
  v0.18.0 on audio v0.21.0 with an empty root and `XDG_CONFIG_HOME`, and
  unmoved at v0.22.0 on audio v0.24.0 and at v0.23.0 on v0.24.2: `corpus
  <dir> 24 4 aug gen nowav` hashes `1c86cc77...e2b4` over 49 files; `bench 12
  4` reads the analysis at 0.453 clean (tempo 0.83 bars 0.90 drums 0.43
  melody 0.17 bass 0.25) and 0.425 augmented (0.83 0.90 0.43 0.13 0.11), and
  `bench 24 16 meters` at 0.491 and 0.428, the meter read in 19, 1 and 2 of
  the 19, 2 and 3 loops in 4/4, 3/4 and 6/8. Audio's tempo bound at 240 bpm
  moved none of those lines, and no pick of 48 loops of `corpus <dir> 48 8 aug
  gen meters` lay under it.

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
  `second_voice` and `cross` move no note on the models' path, while the bass
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
- A family's name is learned long after its sound: on 720 loops every held-out
  note was a synth at step 500 and no piano or sung note was named at step
  1,500, the bass at .43, while the wave and the genes moved from the start.
  Six notes in ten of that corpus are synth's, each engine voice far from the
  bank being one, so read a family's own share, never the total.
- The heads of what plays a pitch teach the cells the decoders read. With the
  gradient stopped at the cells they learn only the commonest name; at the
  whole weight of the frame's like heads a 3,000-step run's strikes' F1 fell
  from 0.328 to 0.278, at a quarter (`PLAYS`) to 0.301. Before changing that
  weight or those heads, run the same steps without them and read the
  strikes: two such runs taught alike part by 0.012 on the starting draw alone.
- Generated loops cannot judge a family's name. A piano or a guitar there is
  an engine voice within 0.2 of a bank voice's genes among random ones on the
  same waves, and each note given its octave's most common family, wave and
  mean voice scores what a 3,000-step model does. Read names on recorded
  families; a container run shows that the heads, the measure and the file
  work, not how well they hear.
- A notes model of a few thousand steps rips few notes at the shipped strike
  thresholds. To see what a rip says of such a model's notes, set `strike`,
  `strike_low` and `strike_again` in the root's `rip.toml` (0.5, 0.5, 0.8 on
  three training loops here) and say so beside the numbers.
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
- A fine-tune reads as a loss under the installed scores, which were set with the
  start's heads in hand: 37d, 32a's recipe and a cents head, scored its strikes
  above 32a's on the validation corpus (.523 against .515) and lost the clean
  take by .03 and the phones by up to .045. Read each model under a judge fitted
  on its own rows. Two runs that `--validate` on `~/ripcorpus-val37` do not fit
  beside a desktop on 30 GB: its 2,506 windows hold 5 GiB in each.
- A probe of a model's heads reads what its frozen cells already hold, and
  what the heads are asked bounds it as it bounds a free run: 32a's cells gave
  bass and the synths alone (38a), as did the body free (38c, 38e).
- A name the corpus gives by a voice's genes is not one its sound says: named
  by the voice they were drawn under, or `synth` far from the bank, modal and
  string sounds were asked two names, and the heads of what plays each pitch,
  given plenty of each family, named bass and the synths alone (38c, 38e) while
  naming solo singing, which every corpus names one way, on 96% of its notes;
  so a note the engine rendered asks no family. Count a name's cells by family
  and source (`vif/tmp/round37/tools/askcount2.py`) before training for it.
- A training loss that keeps falling says nothing about fresh loops: 12000
  steps on 300 loops reached 0.31 and benched worse than 3000 steps on 120,
  each loop seen three hundred times. `train.py` holds a tenth of the loops
  out, scores them every `--every` steps and exports the best step; judge a run
  by its held-out line and by `bench` on the file, never by the last training
  loss, and prefer more loops to more steps.
- A `.pt`'s held-out loss, a merge's or any checkpoint's, is read without
  training: `train.load_corpus`, `train.validation` over the last `--holdout`
  loops and `train.evaluate` in a scratch script, the network built from the
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
- `ripmodels.features` over a whole long recording overflows: its PCEN average
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
  default 16 by 8: give it `frames <dir>`, and a later run reads them back
  (six rendered recordings, 356 s, at full-size random weights and 2 by 1: 16.4
  s and 36 CPU-seconds the first run, 6.8 and 6.5 the second; 180 kB of frames
  a second of recording). A file's name holds the recording's and the weight
  files' checksums, not the models' code: clear the folder after a change to
  `ripmodels` or the transformer.
- Scored within lines over a list of guitar alone, its first run silenced the
  bass line (`strike_low` to .99): its own score rose .608 to .670 and solo
  guitar .06 by pitch and onset, while the renders, the clean take, the phones
  and MusicNet fell .02 to .04 and Billie Jean's bass from 111 right to 93. The
  guard refuses that only for conditions its list names: list every kind a
  fit may hold, and decode a tuned `rip.toml` over every condition, the full
  tracks among them (`vif/tmp/audit2/cond-v0140`), before it ships; no change
  of one bar from the hand-swept five gained the mean of twenty conditions.
- A pool drawn near its start suits `tune ... truth` and not `tune`: over 12
  rendered loops of 4 bars at 16 by 8, from .616, the pool drawn across the
  ranges reached .678, .677 and .686 on three seeds and the near one .650,
  .676 and .642, so `search` takes `near` from the truth search alone.
- A test of `tune ... truth` cannot count on the pool: a near draw moves a
  gene one time in four, a rip of the tiny model's 4.3 s loop costs 0.1 s, and
  a recording under four seconds is refused. Force the outcome in the fixture:
  a truth the start hears exactly leaves nothing to beat, and a start at
  `strike_hold` 0.3 hears so much that every seed's pool betters it.
- The live rip's notes are the offline rip's (95 in a hundred over twelve
  recordings through 12a and 32a) and its bars are not: tempo, meter and phase
  hold from the first six bars, so the two agree on three to five recordings,
  whatever the ear waits. It cost three times the offline rip while each
  model ran three times over the ear's overlapping half-second steps, and
  costs 1.4 times since each frame runs once (`live`, one core: 0.24 s a
  second of audio against 0.17).
- A tool that names its corpus in its source dies with the corpus: the edge
  probe read corpus 16, deleted a day later. Take the corpus, the checkpoint
  and the first loop as arguments.
- **A judge's whole path runs in ninety minutes on four cores**, nearly all
  of it the notes model: `corpus fit 360 4 aug gen` and `corpus held 40 4 aug
  gen seed 2`, both with their audio; a frame model of width 64, two layers,
  2000 steps (four minutes); a notes model (`--pitchnet`), 2500 steps (71
  minutes, its strikes' held-out F1 .26); `rows` of the 400 loops (a minute
  in four parts, 235 MB for 108,933 strikes); the fit (eight seconds); `rip`
  and `truth ... timed` over the 40 left out, against their labels written as
  transcripts (nine seconds a loop, most of it the voices' fit). It shows
  that the path holds and nothing of a judge's worth: the loops left out are
  the fit's own kind under another seed, and the notes model reports a seventh of
  the truth's notes.
- **A weak beat or downbeat head costs the bars where audio takes its word.**
  A width-128 frame model after 5,000 steps on `corpus <dir> 1600 12 gen
  meters` read `beats' F1 0.763, bar lines' 0.631` held out, and through its
  file on audio v0.24.0, which took the heads' word alone, the bars of 56 of
  the 160 held-out loops were right with the heads, 124 with the same weights'
  heads left out and 109 through the analysis. Since v0.24.1 the heads are
  read by rank and a clip they give no reading of is the cues' (85 right on
  audio's branch before the tag). Read a round's bars both ways before its
  file is installed.
- The lines' F1 does not say which beats a head misses. Print what each head
  says within a frame of the loops' own beats by their place in the bar
  (`train.Label.beats`, the frame model over a whole loop): the downbeat head said
  0.73 on one and 0.44 on three of 4/4 with drums, the generated beats kicking
  on both, and 0.28 on a line without drums, where v0.24.0 picked the
  longest bar offered; the beat head 0.30 on 6/8's eighths between its dotted
  quarters, so v0.24.0 kept the felt beat.
- A model's heads are left out of a rip through the same weights by a
  `RipFrontEnd` that wraps `ripmodels`' and sets the frames' `Beat` and
  `Downbeat` nil, in a scratch module on audio's tag over `Ripper` and
  `MatchBars`, the truth the loop's own grid with its end's line left out; or
  by the same step exported with the output layer's last two rows cut, which
  stapes benches. The two ripped row for row alike over 40 loops.
  `BarScore.Phase` is a median folded into a bar, so half a bar off reads as
  anything between: count `Downbeats`.
- That run's path takes an hour on four cores: the corpus with its audio 264 s
  in four parts (7.9 GB), 5,000 steps at 0.45 s and 30 s a held-out score of
  160 loops, 480 rips in three minutes over four processes.
- A frame model's probe (`--probe` from a model without the beat and downbeat
  heads) is checked on real weights by its held-out line and by a rip: the
  other ten heads' losses print to the digit as the start's, and its file
  ripped with the two heads left out is the start's row for row. It was run
  only from a model cut from one trained with the heads, whose features hold
  the beat already; what a probe reaches on the installed model is unmeasured.
