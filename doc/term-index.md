# Term index

A reverse glossary of stapes: look up a word met in its documents or its code to
learn what it means here, where it is explained, and where it lives. A word with
several meanings lists each sense on its own line, and **bold** names in a
definition are entries of their own. Links point to the section that explains
the term; code locations name the package (`stapes`, `cmd/stapes`) or the
trainer's file (`tool/rip-train`) and the main identifier. The audio engine's own
terms, the sound, the voice, the sequencer, the rip and the rest, are in the
audio library's [term index](https://github.com/lixenwraith/audio/blob/main/doc/term-index.md);
an entry here that names one links there.

[A](#a) · [B](#b) · [C](#c) · [F](#f) · [L](#l) · [M](#m) · [O](#o) · [P](#p) · [R](#r) · [S](#s) · [T](#t) · [V](#v)

## A

**accidental** (also: `acc`) — A pattern step's semitones off its scale degree, the audio library's ([audio terms](https://github.com/lixenwraith/audio/blob/main/doc/term-index.md#a)): `take` writes one for a key played outside it, and the grid marks it `#` or `b` after the degree. [Stapes §1](stapes.md#1-what-it-offers) · `stapes` `cmdTake`

**agree** (also: two witnesses) — The comparison of two **rip**s of one piece where no score is to hand, through the audio library's `MatchNotes` ([audio terms](https://github.com/lixenwraith/audio/blob/main/doc/term-index.md#a)): what both hear is likely; what one hears is to doubt. See also **truth**. [Stapes §4](stapes.md#4-the-rips-verbs) · `stapes` truth.go

## B

**bright** — A **rip**'s brightness of a note, the audio library's ([audio terms](https://github.com/lixenwraith/audio/blob/main/doc/term-index.md#b)), which `notes` prints, counting the unmeasured. [Stapes §4](stapes.md#4-the-rips-verbs) · `stapes` `cmdNotes`

## C

**channel** (also: plane, `--channel`, `CHANNEL`, noise bank, `noise.npz`, `make_noise.py`) — two senses.
- *Input*: one plane of 230 inputs a frame the rip's model reads, the first the bands in dB over the frame's level and the level, a second, when the file has one, each band's **PCEN** and their mean. [Rip training §4](rip-training.md#4-the-input-two-planes) · `tool/rip-train` `ripmodel.features`
- *Phone*: what a small speaker, a room and a phone do to a training window, drawn per window for `CHANNEL`'s share of them (`train.py --channel <file.npz>:<share>`): the lows and highs cut, a ripple across the bands and a steady noise floor added in power, shaped from a noise bank, two minutes of white noise as mel frames and a real room's band shape, which `make_noise.py <quiet.wav> <out.npz>` builds. [Rip training §6](rip-training.md#6-what-may-change-a-windows-frames) · [rip-train README](../tool/rip-train/README.md#turning-a-run-toward-your-music) · `tool/rip-train` `Channel`, make_noise.py

## F

**fidelity** (also: `Fidelity`, `songJudge`) — How much a render sounds like the recording it stands for, the audio library's grade ([audio terms](https://github.com/lixenwraith/audio/blob/main/doc/term-index.md#f)): `refine` judges the voices and the steps by it, the rip-back reported beside it, and `song <name> score` prints both. [Stapes §4](stapes.md#4-the-rips-verbs) · [Rip design §2](rip-design.md#2-stages) · `stapes` `songJudge`

## L

**lead rule** (also: `rip ... lead`) — Which of the melody register's two lines a **rip** calls the melody, the audio library's rule ([audio terms](https://github.com/lixenwraith/audio/blob/main/doc/term-index.md#l)), which `rip ... lead` names. [Stapes §4](stapes.md#4-the-rips-verbs) · [Rip design §9](rip-design.md#9-decisions) · `stapes` `cmdRip`

## M

**meter** (also: `meter`, time signature, 3/4, 6/8) — A bar's time signature, the audio library's ([audio terms](https://github.com/lixenwraith/audio/blob/main/doc/term-index.md#m)): a step stays a sixteenth, so 3/4 and 6/8 are twelve steps. The `meter` verb shows or sets the transport's, which a **rip**, a **song** played and a **MIDI file** loaded set too, and `render`, `where`, the **mixer** and the playhead count in; a pattern's own (`meter = "3/4"`) lays out its bars in the **pattern editor** and in `cp <pattern>.bar.<k>`. `rip ... meter <n/d>` pins a rip's, and `corpus ... meters` and `bench ... meters` render a third of their loops in 3/4 or 6/8, the bench grading each meter apart. [Stapes §4](stapes.md#4-the-rips-verbs) · `stapes` `cmdMeter`, `beatsByMeter`, `benchTally`

**MIDI file** (also: `.mid`, score) — Patterns, songs and a **rip**'s transcript as Standard MIDI Files, the audio library's ([audio terms](https://github.com/lixenwraith/audio/blob/main/doc/term-index.md#m)): `write <pattern|song> <file.mid>` and `song <name> <file.mid>` write, `load` and `merge <file.mid> [name]` read, `notes <name> <file.mid>` writes a rip's transcript, `truth <name> <file.mid>` scores a rip against one, and `rip <file.mid> <name>` reads one as a rip with no recording behind it. See also **truth**. [Stapes §3](stapes.md#3-midi) · `stapes` `cmdNotes`, `cmdTruth`, `readScore`, `loadMIDI`, `writeMIDI`

**mixer** — The pane that draws every slot on one timeline, a row a track, and mutes a slot or a track or solos a track through `mute slot`, `mute track` and `solo track`, a silent track dimmed; the audio engine's mixer goroutine is in the [audio terms](https://github.com/lixenwraith/audio/blob/main/doc/term-index.md#m). See also **solo**. [Stapes §1](stapes.md#1-what-it-offers) · `stapes` `mixerKey`, `drawMixRow`

## O

**over** (also: `--over`, `OVER`, real drums, heard hit, `stands_out`) — The trainer's overlay of a corpus of drums alone, such as E-GMD's, on a loop with no drums of its own, for `OVER`'s share of the windows (`train.py --over <dir>:<share>`), at -12 to +6 dB of the loop's power and added in power; a hit joins the onset and drum targets only where what it adds tops everything else sounding over its best ten bands (`stands_out`), judged after the **channel**. [Rip training §6](rip-training.md#6-what-may-change-a-windows-frames) · `tool/rip-train` `sample`, `stands_out`

## P

**pattern editor** (also: editor, keys, grid, piano roll, take, yank, `n`, `e`) — The editor of one pattern, new (`n`, empty and playing along in the first silent slot) or edited (`e`), which its frame and header say, on two tabs: keys, two octaves and a third of a tracker layout drawn as a piano, each note of the key `key` set dotted, whose take is recorded over the pattern's length and written into it as a track a voice through `take`, each note on the degree of the key that sounds it; and grid, the tracks on a step grid under a bar ruler and the playhead, or with `r` one tonal track as a piano roll; `y` yanks the cursor's track, `Y` adds it to the yank, and `p` pastes the yank into the pattern open (`mix <pattern>.track.<n>...`), the tracks meeting at a length each loops in, or in the browser makes a new pattern of it (`cp`), refusing a yanked track that has changed since; `J`/`K` move the cursor's track down or up (`mv <pattern>.track.<n> <j>`), the order a build-up reveals the tracks in (`armReveal`). Every edit runs a command, so a script can make it, and `u` undoes what the last key did; `^D` twice deletes the pattern. See also **stapes**. [Stapes §1](stapes.md#1-what-it-offers) · `stapes` tui_editor.go, tui_keys.go

**PitchNet** (also: harmonic stack, `--pitchnet`, notes model, `ripnotes.bin`) — The rip's convolutions-only model, the same weights at every pitch, over both input planes stacked by harmonic, [Rip training §4 and §7](rip-training.md#7-the-model); `ripmodel.py`'s `PitchNet`. Its weight file, the **root**'s `ripnotes.bin`, gives the notes beside `ripmodel.bin`, [Rip training §9](rip-training.md#9-the-weight-file); audio's `ripmodel.NotesSpec`.

**PCEN** (also: `--pcen`, `PCEN`, `WARM`) — The rip model's optional second input channel, each band's per-channel energy normalization and their mean, the audio library's ([audio terms](https://github.com/lixenwraith/audio/blob/main/doc/term-index.md#p)): `train.py --pcen`, `PCEN=pcen` to the pipeline, trains a model of two channels, reading 128 frames before each window (`WARM`) so the average has settled where the window starts. See also **ripmodel**, **channel**. [rip-train README](../tool/rip-train/README.md) · [Rip training §4](rip-training.md#4-the-input-two-planes) · [Rip design §8](rip-design.md#8-measured-on-the-sample-recording) · `tool/rip-train` `ripmodel.features`

## R

**rip** (also: `rip`) — The audio engine's estimate of a recording, the audio library's ([audio terms](https://github.com/lixenwraith/audio/blob/main/doc/term-index.md#r)), and stapes' verbs over it: `rip` adds its patterns to the document and sets the transport's tempo, **meter** and key; `tune` searches its parameters with `lixenwraith/genetic`, a tuned set read from the **root**'s `rip.toml`; `notes` lists its transcript or writes it as a **MIDI file**; `rip` and `fit` fit a voice to it, and `compress` writes its bars as a dictionary and a **song**, whose render ripped back is the score of the compression; `refine` closes the loop judged by **fidelity**, the steps' pitches moved and the voices fitted until a round gains nothing, each voice held within a tenth of its tonal floor, the fit's distance from the recording's notes. `render` writes what the slots hold to a WAV, `corpus` the loops a model trains on, clean or augmented, or recordings with the MIDI files they were played from, and `bench` grades the analysis and a loaded model on fresh renders. See also **stapes**, **ripmodel**, **song**. [Stapes §4](stapes.md#4-the-rips-verbs) · [Rip design](rip-design.md) · `stapes` `cmdRip`, fit.go, dictionary.go, tune.go

**ripmodel** (also: RipModel, `ripmodel.bin`, rip model, front-end model) — The rip's learned front end, the audio library's ([audio terms](https://github.com/lixenwraith/audio/blob/main/doc/term-index.md#r)): `tool/rip-train` is its PyTorch twin and trainer, and the **root**'s `ripmodel.bin` the weights stapes hears with. See also **rip**, **training round**, **transformer**. [Rip design §3](rip-design.md#3-learning) · [Rip training](rip-training.md) · `lixenwraith/audio/ripmodel`; `tool/rip-train`

**root** (also: `-config-dir`, `stapes.toml`) — The folder that holds stapes' own files, flat: `-config-dir`, or else `os.UserConfigDir()` plus `stapes`. `stapes.toml` is its settings, today `[audio] buffer_ms`, the mixer period, a key it has no field for refused by name; `rip.toml` the parameters `rip` listens with and `tune` writes; `ripmodel.bin` the weights `rip` hears with, and `ripnotes.bin` beside it the notes model's. The documents are not the root's but the **search path**'s. [Stapes §2](stapes.md#2-its-files) · `stapes` `ResolveFiles`, `Files`

## S

**search path** (also: `-config-music`, `-config-sounds`) — The ordered folders `music.toml` and `sounds.toml` are found in, today vif's audio folder alone (`os.UserConfigDir()` plus `vif/audio`), so stapes opens the overrides vif reads from its user root; a document saves back to its file, and an untitled one lands in the first folder, where vif reads it next run. `-config-music` and `-config-sounds` name a document's file outright. See also **root**. [Stapes §2](stapes.md#2-its-files) · [todo](todo.md#search-more-than-one-folder) · `stapes` `ResolveFiles`, `findDocument`

**shard** (also: `SHARD`, `--shard-seed`) — One worker's run in a **training round**, from the round's start on a stream of windows its `SHARD` picks (`train.py --shard-seed`), unique to the worker and the round. [rip-train README: Training in parallel](../tool/rip-train/README.md#training-in-parallel) · `tool/rip-train` `pipeline.sh shard`

**shift** (also: `shift`, `cmdShift`) — The move of every note of the tracks named, by octaves, by degrees of the transport's key carried into the octave, by semitones spelt again on the key (`PitchStep`), or by steps rotated around the pattern's length, a drum track only by step; a note past `ValidatePattern`'s step bounds refuses it whole. For a line an octave or a step off. See also **split**, **accidental**. [Stapes §1](stapes.md#1-what-it-offers) · `stapes` `cmdShift`, `shifts`

**shuffled window** (also: `--shuffle`) — A training window spliced from chunks of a second, each two clips summed, [Rip training §6](rip-training.md#6-what-may-change-a-windows-frames); `train.py`'s `shuffled`.

**skip list** (also: `skip.txt`) — A corpus's list of loop names the trainer leaves out, one a line, such as recordings whose MIDI sits off their audio further than a steady lag explains. [Rip training §3](rip-training.md#3-the-labels-loop_nnnjson-and-corpusjson) · `tool/rip-train` `load_corpus`

**solo** (also: `solo track`) — A soloed track of the pattern a slot plays sounds with the slot's other soloed tracks alone, a muted one never; set by `solo track` or the mixer's `s`, and dropped when the slot changes pattern. See also **mixer**. [Stapes §1](stapes.md#1-what-it-offers) · `stapes` `trackMutes`

**song** (also: `compress`, `song`, dictionary, matched kit) — A timeline as data, the audio library's ([audio terms](https://github.com/lixenwraith/audio/blob/main/doc/term-index.md#s)): `load <file.mid>` reads one from a **MIDI file** with its patterns, and `compress` writes one from a **rip** with its dictionary and the drums matched to the recording's, which its `kit` names, and `song` shows, its bars numbered as the paths `add`, `dup`, `mv` and `del` edit them by (`<song>.bars.<i>`), renders (one line alone when instruments are named after the file) or conducts it through the live sequencer in its **meter**, each bar's patterns scheduled to land on its bar line. vif's conductor places patterns by tier, not by bar; a song is stapes' form until a level wants a fixed piece. See also **rip**, **stapes**. [Stapes §4](stapes.md#4-the-rips-verbs) · [Rip design §2](rip-design.md#2-stages) · `stapes` dictionary.go

**split** (also: `split`, `cmdSplit`) — The parting of a tonal track at a MIDI pitch under the transport's key: the notes from it up stay, the rest go to a new track of the same voice after it, for two lines a rip heard as one; a drum track or an empty side is refused. See also **shift**. [Stapes §1](stapes.md#1-what-it-offers) · `stapes` `cmdSplit`

**stapes** (also: `cmd/stapes`, `.slab`, undo) — The authoring tool over the audio engine and the banks vif plays: a REPL, a TUI over any terminal and headless `.slab` scripts for validation, apply and revert, audition, sequencing, TOML save and WAV export; a document saves back to its file, an untitled one to the **search path**'s first folder, where vif reads it next run. `undo` steps back over its edits, a command that changed a document each (a TUI key's commands one), each recorded from a shadow of the document as the last edit left it, the registry and transport untouched. The session, documents, verbs and TUI are the package `stapes`, which a host with its own screen and speakers runs in-process; `cmd/stapes` is its terminal command. See also **root**. [Stapes §1](stapes.md#1-what-it-offers) · `stapes`, `history`, `cmdUndo` · `cmd/stapes`

## T

**training round** (also: round, `round.sh`, merge, `average.py`, `--init`, `INIT`, `MERGED`, trimmed mean, `TRIM`) — two senses. See also **ripmodel**.
- *On one machine*: one run of the rip's model from its corpus to its bench, `round.sh`'s three stages logged to `$CORPUS/round.log`; run again after a stop, it goes on from the checkpoint, which carries the best held-out step, so it exports what an unbroken run would. [rip-train README: A round on a development machine](../tool/rip-train/README.md#a-round-on-a-development-machine) · `tool/rip-train` `round.sh`
- *Across machines*: one step of training the rip's model over several machines: every **shard** starts from one checkpoint's weights (`train.py --init`, no optimizer state, step 0), trains apart, and `average.py` averages the shards' checkpoints weight by weight, after dropping each weight's `--trim` share of highest and lowest values, into the next round's start and a weight file; runs from different random starts are never averaged, since their units sit in different orders. [rip-train README: Training in parallel](../tool/rip-train/README.md#training-in-parallel) · [Rip design §3](rip-design.md#3-learning) · `tool/rip-train` `average.py`, `pipeline.sh merge`

**transformer** (also: `lixenwraith/transformer`, weight file, ALiBi, `Conv`, `Tower`) — Inference over sequences of frames in pure Go, float32, no CGO: convolutions over frames and features when a file has them, each pooled over the features (`Conv`), a linear input, pre-norm attention blocks whose heads lean on nearer frames by a slope per head (ALiBi) within a window, exact-GELU feed-forwards, a final norm and named output heads, the blocks one stack or towers that each compute a run of the heads (`Tower`); `Load` reads the versioned weight file `tool/rip-train` writes, tensors by name with an FNV trailer, and `Forward` runs it. Standard library only, a module of its own; **ripmodel** is its one user. [Rip design §3](rip-design.md#3-learning) · [Weight file](https://github.com/lixenwraith/transformer/blob/main/doc/format.md) · `lixenwraith/transformer`

**truth** (also: `truth`, ground truth) — The score of a **rip**'s transcript against a ground truth, notes given as text, a tapped rhythm or a **MIDI file**, the form the audio library reads ([audio terms](https://github.com/lixenwraith/audio/blob/main/doc/term-index.md#g)): how many of the notes given the rip heard and how many it heard that were not given, per line, aligned as **agree** aligns two rips, and for a rhythm the given notes cut in two and the heard notes spanning two; notes with pitches are also counted by the level the recording plays them at, in 6 dB bands under the loudest tenth, and alone or under another; a MIDI file's drum channel scores the drums. The ground truth the fidelity and rip-back numbers lack: they grade the mix, not the melody. [Stapes §4](stapes.md#4-the-rips-verbs) · `stapes` truth.go

**tuning** (also: `tuning`, cents) — The cents a recording's pitches sit off A440, the audio library's ([audio terms](https://github.com/lixenwraith/audio/blob/main/doc/term-index.md#t)): `rip` names it when it is off, an augmented corpus loop is rendered off A440 by up to a quarter tone, and the `tuning` verb shows or moves the transport's. [Stapes §4](stapes.md#4-the-rips-verbs) · `stapes`

## V

**vibrato** (also: vowel) — Two voice-like traits of an oscillator voice, the audio library's ([audio terms](https://github.com/lixenwraith/audio/blob/main/doc/term-index.md#v)), which `tool/rip-train/focus/vocal.toml` draws into a corpus. · `tool/rip-train` focus

**voice** (also: tonal voice, `fit`) — A tonal instrument as data, the audio library's ([audio terms](https://github.com/lixenwraith/audio/blob/main/doc/term-index.md#v)), which `rip` and `fit` fit to a recording's notes. See also **rip**. [Stapes §4](stapes.md#4-the-rips-verbs) · `stapes` fit.go
