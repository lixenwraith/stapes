# Stapes

Stapes is the authoring environment over
[`lixenwraith/audio`](https://github.com/lixenwraith/audio), the engine and
banks vif plays. The engine's own design, from its stream contract to the rip,
is the [audio library's document](https://github.com/lixenwraith/audio/blob/main/doc/audio.md)
(cited below as "library §n"); the [README](../README.md) is the command and key
reference, and [Rip design](rip-design.md) is where the rip is going.

## 1. What it offers

The package `stapes` holds the session, its documents, the verb table and the
TUI, which runs over any `terminal.Terminal`; `cmd/stapes` is its terminal
command, the flags and the REPL, and a host with its own screen and speakers
runs the package in-process, handing `NewSession` the speakers. It offers:

- a command REPL;
- a full-screen TUI: a browser of patterns, sounds and voices beside a mixer that
  draws every slot on one timeline, mutes a slot or a track and solos a track, a
  structural inspector, and the pattern editor, new or on an existing pattern: a
  drawn keyboard that records takes onto the key's degrees, and a step grid with a
  piano roll;
- headless `.slab` script execution;
- validation, live registry apply/revert, audition, sequencer slot assignment,
  TOML save, WAV export, and MIDI both ways: a pattern or a song written as a
  Standard MIDI File, and a MIDI file loaded as patterns and a song.

It registers the sound and pattern banks vif ships, from the audio module's
`bank`, and the overrides its search path holds (§2), as vif registers them, so
its registry and that of a game run on the same user root hold identical specs. It does not apply vif's sfx
levels and sound shapes, so its auditions and exports play a spec as authored.

The working document and live registry are separate: edits do not affect audio
until `apply`; `revert` restores the document from the canonical registry. Its
dotted path grammar follows TOML tags and Go types, while domain validation
runs at `validate`/`apply`.

## 2. Its files

The root is `-config-dir`, or else `os.UserConfigDir()` plus `stapes`
(`~/.config/stapes` on Linux and FreeBSD), and holds stapes' own files, flat:

- `stapes.toml`, the settings: `[audio] buffer_ms`, the mixer period, the
  engine's 50 ms when unset; a key it has no field for is refused by name;
- `rip.toml`, the rip's parameters: what `rip` listens with when it is there,
  and where `tune` writes;
- `ripframe.bin`, the frame model, the rip's learned front end: what `rip`
  hears with when it is there, the built-in analysis when it is not
  ([rip-train](../tool/rip-train/README.md)); under its old name,
  `ripmodel.bin`, it is named at the start to be renamed and never loaded;
- `ripnotes.bin`, the notes model: beside `ripframe.bin` it gives the notes, the
  pitches and their strikes, releases and velocities, and alone it is unused;
- `ripjudge.bin`, a judge of the strikes: beside the weights it was fitted on
  it rules which strikes are notes, in the installed scores' place, and beside
  any others it is refused by name and they judge
  ([a judge](../tool/rip-train/README.md#a-judge)).

The documents, `music.toml` and `sounds.toml`, are found along a search path,
whose one folder is vif's audio folder, `os.UserConfigDir()` plus `vif/audio`:
the overrides vif reads from its user root, opened as the working documents. A
document saves back to the file it came from, and an untitled save lands in the
search path's first folder, where vif reads it next run. `-config-music` and
`-config-sounds` name a document's file outright, read and saved there.
`ResolveFiles` reads nothing of vif's code and never probes the working
directory; `-config-dir` names the root alone, so the search path holds vif's
folder whatever it names.

A save writes the document over the text of the file it came from, read when
the save runs (`SavePatternFile`, `SaveSoundsFile`, the library's
[Documents](https://github.com/lixenwraith/audio#documents)): the file's
`[macros]`, its own comments and the comment above each entry stay, every entry
is written anew, an invocation of those macros where one expands to it, and a
save to another file carries them there. A comment inside an entry is lost.
Patterns seeded by `builtin`, as the TUI seeds an empty document at its start,
stand in the bank's own order, have no file behind them and are saved over the
bank's own document (`bank.MusicTOML`) until a save gives them a file: the
first save is the bank's document byte for byte but for the session's edits.
Any other document with no file behind it, made in the session, read from a
MIDI file or sounds seeded by `builtin`, is written out in full, as `write`'s
one entry is ([todo](todo.md#keep-what-a-save-still-loses)).

A user `sounds.toml` may `include` files inside its own directory, which vif,
its `-check` and stapes resolve alike, so a file stapes saves with its includes
is what play loads.

### Paths on every host

Stapes names its files by path on every host, each verb alike: `load`, `merge`
and `save` of a document, `write` of an entry (TOML, or a `.mid` for a pattern or
a song), `rip` and `truth` of a `.wav` or a `.mid`, and `render`, `export` and
`song` to a WAV or a `.mid`. A relative path reads from the working directory, or
from a running script's own, and a bare `save` writes back to the document's file
or the search path's first folder, so a terminal, over ssh too, needs no dialog;
the files overlay (`^O`) browses the same paths. Stapes has no browser host,
though the audio library writes and reads MIDI there too, from any reader or
writer.

On Android, where stapes mobile, an app of stapes' own, is to run it in-process
([todo](todo.md#build-stapes-mobile)), the root and the search path are in the
app's own storage, which these paths reach as they are and vif's app does not.
A MIDI file or a recording another app holds, a download or a shared file, is
a content URI behind the Storage Access Framework, with no path. There the app
opens the system's picker (`ACTION_OPEN_DOCUMENT`), copies what
was picked into the search path's first folder and runs the verb on the copy; a
file to hand back is written there and offered through `ACTION_CREATE_DOCUMENT`
or a share. The verbs stay as they are, and so does every other host.

## 3. MIDI

MIDI goes through the library's one table, `midiInstruments` (library
[§8](https://github.com/lixenwraith/audio/blob/main/doc/audio.md#8-midi-files)): `write <pattern|song> <file.mid>`
writes a pattern or a song (`WriteSongMIDI`, `WritePatternMIDI`), and `load` and
`merge <file.mid> [name]` read a file as patterns and a song (`MIDIScore.Song`)
in its first meter, which the transport takes as a song played does (`meter`,
`SetMusicMeter`), a bar of another meter or tempo naming it at its head
(`"2/8"`, `"72bpm"`), which `song` lists and `song play` sets at the bar line. `truth` and `corpus <dir> from` read `ReadMIDI`'s transcript. Stapes turns the
phrase fill off (`SetAutoFill(false)`), and its mixer mutes slots and tracks and
solos tracks (library
[§6](https://github.com/lixenwraith/audio/blob/main/doc/audio.md#6-mixer-buses-mute-and-pause)).
`binaural` lays the library's binaural bed under what the transport plays
(`SetBinaural`, the same section), by a carrier, a beat and a level, and takes
it away: the one thing the stream's two channels differ by, and in no offline
render, so a corpus and a bench are the bytes they are without it.

## 4. The rip's verbs

Over the library's rip (library
[§9](https://github.com/lixenwraith/audio/blob/main/doc/audio.md#9-ripping-a-recording)), stapes' verbs:

- `rip` adds a rip's patterns to the document, dirty, and sets the transport's
  tempo, meter and key, so a slot plays the estimate at once, and says the
  tempo a player reads where the meter's beat is not the quarter (a 3/8 at 73
  is eighth 146, `Meter.FeltTempo`); `rip ... bpm <n>`
  pins the tempo alone, `rip ... meter <n/d>` the meter (`Ripper.Meter`),
  `rip ... lead` names the rule that picks the melody (`Ripper.Lead`), and
  `rip <file.mid> <name>` reads a MIDI file as a rip (`MIDIScore.Rip`). For
  each line `rip` searches the voice's genes and wave for the nearest, seeded
  with the voice the front end heard and every registered one: around each
  note it listens to, the recording against the line's notes there on the
  candidate beside the other lines' on their voices and the drums' hits on
  the kit, each part at a gain of its own (`fitDistance`, `mixDistance`,
  [Rip design §2](rip-design.md#2-stages)); `fit` runs it again at the sizes
  its words name (`fitWords`: `notes`, `pool`, `generations`, `seed`, and
  `cells`, off unless named, which hears each note on its own cells,
  `fitScene.own`), and `genes` says a voice's wave and genes as a label states them
  (`VoiceDef.Genes`).
- `fits` reads that search against voices that are known: each loop of a
  corpus directory that kept its audio is ripped and fitted as `rip` does it
  (`Session.ripper`, `fitVoice`), with `heads` through the label's own heads
  as `truth`'s (`newLabelHeads`), no model heard under `all`, or with `given` each voice the label states
  is a line fitted on its own notes at the label's tuning, nothing ripped, so
  the measure is read apart from the rip. A row a line (`fitRow`) says whose
  line it is, the label's voice that plays most of the line's notes that
  match a labelled note (`lineOwner`, `MatchNotes` timed, a voice a wave and
  its genes), its share of those notes and of the notes listened to
  (`windows`), and four distances by the fit's measure (`fitDistance`) on the
  notes the fit listened to and on the line's other notes: the voice the
  front end heard, the best seed, the fit, and the label's own voice, rebuilt
  from its genes (`VoiceFromGenes`), played at the same notes. The label's
  own voice nearer than the fit is a search that stopped short; the fit
  nearer than it is a measure that prefers another voice. A loop is a file
  under its audio's checksum, so a run keeps what is there and `part k/n`
  needs no numbering; `fits.json` names the sizes and the weights, the
  heads and `cells`, or `given`. The
  search draws its candidates from one seeded stream and scores them side by
  side, so a seed's rows are the same on any number of cores
  ([todo](todo.md#close-the-rips-loop), item 9, for the one way they may not be).
- `agree` compares two rips of one piece (`MatchNotes`); `truth` scores a rip
  against notes given as text or a tapped rhythm (`ParseTranscript`), a MIDI
  file (`truth <name> <file.mid>`) or a corpus loop's label (`truth <name>
  <loop.json>`, read as `tune ... truth` reads one), first every note as one
  line by pitch and onset (`Agreement.All`, the `notes` row), then by line,
  counting the notes heard by level as well (`NoteLevels`); `timed` matches a
  truth on the recording's own clock, its notes and its hits (`MatchNotes`' and
  `MatchHits`' `timed`: no map, no transposition); a
  MIDI file's parts past its melody, under and bass count in the melody
  register as one line, or the bass under E3, in an `every part` row. It prints a
  `bars` row for a MIDI file's bars, placed as its notes matched
  (`MIDIScore.Beats`), or a JAMS file's beats, a truth of beats
  alone (`truth <name> <file.jams>`, `ReadJAMSBeats`): the rip's bar lines
  against the downbeats and beats given (`MatchBars`), with the beats before the
  first downbeat and after the last it lacks (`truthCount.edges`). A label is read past
  them too, a row a reading it gives (`readings`, `reportReadings`), audio's
  over the notes and hits matched, an end past the recording read at its end
  (`Agreement.Until`, `Ends`, `Levels`, `Parts`,
  `HitsAgreement.Levels`) and against its key and chords (`MatchKey`,
  `MatchChords`), a key's miss by its kind on a line beside (`keyMissed`): its
  notes filed under their track, `part` and `instr` joined,
  as a bank slot's tracks share their part's name and an overlap's instruments
  theirs (`corpusLoop.notes`), its key where its root is over MIDI 0, as no
  recording's or overlap's is, and its chords each to the next one's frame, the
  last to the loop's end (`corpusLoop.said`); a velocity of -1 and a note in no
  part give none, and a last line names what the label gives none of. `heads
  <head,...>` reads the decoders apart from the models that feed them: the
  recording is ripped again as it was (`ripped.by`), the heads named written off
  the label as the trainer's targets are (`labelHeads`, `corpusLoop.planes`, a
  row a head in `labelHeadRows`), the rest the front end's, laid on the piano's
  keys where it hears fewer, and that rip is scored; `all` names every head
  the decoders read and hears nothing where the label states each, the label's
  frames standing alone. A head the label does not state, as the trainer asks
  it nothing there, stays the front end's (a row's `stated`,
  `labelHeads.kept`); its pitch states no tuning, as a model's. Every truth ends with
  the rip's stages read against it (audio's `ReadStages` over `Rip.Stages`,
  which the session's rips keep, `stageLines`): the notes lost by stage, why the
  strikes refused those they lost, the notes added, the ends wrong and the hits
  lost and added. `notes`
  lists a transcript, each note's brightness with the unmeasured counted, and
  `notes <name> <file.mid>` writes one (`WriteMIDI`) to correct in a sequencer.
  `corpus <dir> from` labels a file's other drum strikes (`MIDIScore.Struck`)
  as onsets alone.
- `live` drives the library's stream over a file (`RipStream`): `Write` a
  block at a time, as fast as they go or paced by the clock, the bars `Bars`
  gains said as they freeze against the samples fed, or paced against the
  clock; `Close` ends the stream's rip and `Final` is the offline rip, which
  `rip`'s registration takes (`takeRip`). The measure is the stream's rip
  against it: `MatchNotes` counted every note once, `MatchBars` with the
  offline rip's bar lines and beats as the truth, and the process's CPU over
  each (`getrusage`, so the mixer's and a screen's are in it).
- `rip` listens with the root's `rip.toml` when one is there, and hears with
  the root's `ripframe.bin`, and the notes from its `ripnotes.bin` when that is
  there, which `tool/rip-train` trains on loops stapes exports (`corpus`); `render` writes what the slots hold to a WAV through
  `RenderMusic`, so an estimate is heard without a device. The root's
  `ripjudge.bin` is the rip's judge (`Ripper.Judge`, loaded with
  `LoadRipJudge` under the front end's `NotesSum`): `Session.ripper` carries
  it, so `rip`, `live` and what `truth` scores run under it, and `tune ...
  truth` rips its recordings under it; `bench` and `rows` do not.
- `rows` writes what a judge is fitted on: each recording of a truth list, or
  each loop of a corpus directory that kept its audio against its label
  (`truthSources`), ripped with no judge and the strikes read with their rows
  (`Ripper.Strikes`, `Rows`), each strike labelled by `rightStrikes`: the
  library's matcher run over the reported strikes and then over the dropped
  against the truth notes left, the truth placed by the map the rip's notes
  fit (`Agreement.At`). A recording is a `.rows` file, laid out as a loop's
  frames, and a `.json` of its strikes, named by the recording's checksum, so
  a run keeps what is there and parts need no numbering; `rows.json` names
  the row and the notes weights, and a directory of another kind is refused.
- `tune` searches the rip's parameters over loops rendered from the registered
  patterns across tempos and keys with `lixenwraith/genetic`'s generational
  engine, the current set seeding the pool, and writes a set that beats it
  ([todo](todo.md#tune-the-rip) for what the corpus still lacks). `tune ...
  truth <list>` searches instead the thresholds the loaded models' decoders
  read (`modelGenes`) and `twins` and `ring_on` (`modelCounts`, integers it moves
  as floats and reads rounded), the rest held, against recordings, a line
  `wav<TAB>truth[<TAB>condition[<TAB>[timed] [drumless]]]` each, from a pool
  drawn near the current thresholds (`search`'s `near`). A truth gives notes,
  and may give
  drum hits (a `.mid`'s drum channel, a corpus label's) and bars (a label's
  alone, its `beats` and `downbeats`, `corpusLoop.grid`, a recording's from
  its list's beats, GuitarSet's its JAMS files' `beat_position`; a
  transcription's `.mid` holds its writer's default signature). Each
  condition then carries up to three scores (`truthCount`, `scores`): its
  notes' F1 by pitch and onset, its drums' F1 (`MatchHits`, as `truth` counts
  them) and its bars, the mean of two shares (`MatchBars`): the downbeats
  given on a bar line of a rip whose bar is the given's length, and the beats
  given with a beat of the rip's on them, so a bar of the given's length over
  beats not its own, a 12/8 read over its dotted quarters, scores two thirds.
  A truth knows the drums
  (`truthTake.drums`) when it is a label, gives hits or its line says
  `drumless`: a hit heard that it does not give is heard and not given, so a
  recording without drums takes from its condition's F1. A condition that
  gives no hit scores by `drumScore`, the F1 of a rip that heard those hits
  beside every hit as many minutes hold at the list's own rate, the hits a
  minute its recordings with drums give (`hitRate`): 2g/(2g+h) for h heard
  against g, one with none heard, and what the same hits cost pooled with a
  recording as long heard whole. A list that gives no hit has no rate and no
  drums' score. A set scores their mean:
  each is a share of one capability, none converts into another, and the
  guard holds each, so a set that takes more than `guard` from any one at the
  start scores under zero (`guarded`) and is never written. `weigh` adds the
  readings it names (a word a row of `readings`, `loudness` the notes' and the
  hits') to a condition's scores where its labels give them, none unless asked,
  and each condition's readings are said before and after, weighed or not, and
  those its labels give none of. Of the genes the search moves, `release` is
  read by nothing but where a note ends, in the strikes' `until` and the glide
  follower: no F1, drums' or bars' score sees it, save where the installed
  scores' rows read a kept note's ends, and the ends reading does, as the parts
  reading does where a note's length moves the parting, which holds no two notes
  at once on a line. The key moves with its own genes (audio's `key_bars`,
  `key_open`, `key_close`, `key_bar`), which no search here moves, and with the
  bars it is read over, the chords with the bars' genes and the key, which the
  bars' score sees, and the hits'
  loudness with `onset_min` and the bars, which the drums' F1 sees. Through
  audio's `struck` over two loops, with and without a start plane, F1's counts
  also held while `min_note`, and `strike_hold` without starts, moved the ends
  and the loudness, and `strike_low`, `voicing` and `start` moved onsets within
  the window and the loudness: those F1 sees elsewhere, as notes come and go.
  With hits `onset_min` is searched too (`drumGenes`), with bars the tempo's
  prior, combs, cuts, `tightness`, the bars' weights, the meter's margins and
  the joint path's costs and levels (`barGenes`). `genes` frees of these those
  it names alone (`parseGenes`), a gene by its key in `rip.toml` or a part,
  `notes`, `drums` or `bars`, by its genes (`geneParts`), every other held at
  the current set, so a part is searched apart; a name it does not know is
  refused with those it knows, and a word that leaves none free. A search that
  names every part draws what one without the word draws. It says the genes
  free before it starts, a whole part by its name, and last, once it writes a
  set, each gene that moved, from and to as `rip.toml` writes them. `heads`
  rips each recording whose truth is a label with the heads it names the
  label's, as `truth`'s does (`truthTake.read`), the rest through the analysis
  where no model is loaded, and counts by head the takes whose label states one
  not; each condition's stages follow its readings (`truthCount.stages`);
  and either search's best has each gene whose move the score does not see,
  put back alone, back at the start's (`settled`), since a pool moves genes by
  chance and a search through the analysis, a model without the heads or loops
  under six bars sees only some of them. The front end's `Hear` ignores the
  parameters searched, so each recording is heard once (`heardOnce`) and a
  candidate pays the decoders alone; `frames <dir>` keeps what was heard
  between runs, a gob of the `RipFrames` named by the checksums of the
  recording and of the weight files (`Session.earSum`). What a rip reads of a
  take (`takeSound`), its samples at the rip's rate and its frames or its
  label's planes, some 450 kB a second of sound, is held after its first rip
  in the list's order while `hold` has room (`truthTake.held`); every other
  take is read again as each candidate rips it (`truthTake.sound`), its
  recording resampled, its frames read back from `frames <dir>` or heard again
  without it and its label's planes built again, the bits held, so every
  candidate scores the same. That costs some 5 ms of a core a second of sound
  where a rip takes some 20, and a model's hearing again many times that; a
  take that cannot be read again ends the search with its error, writing
  nothing. The collector lets the heap grow to twice what is live, so a run
  under a memory cap sets `GOMEMLIMIT` under the cap.
- `compress` gives a first bar that starts partway the beat of the bar most
  like it (`Rip.MendPickup`), runs the grouping (`Rip.Cluster`) at each part's
  threshold (`compressPercent`), refines the assignment with
  `lixenwraith/genetic` under a term for the patterns in use, registers the
  drums matched to the recording's (`Rip.Kit`, `<name>_kick` ...) that the song's
  `kit` names, rips the song's render back on the song's own grid
  (`Ripper.Grid`) and scores it against the rip; `song` shows, renders or
  conducts a song through the live sequencer on its kit (`SetKit`), the
  transport keeping a played song's, or renders one line of it alone, the
  instruments named after the file.
- `refine` closes the loop: the voices fitted again and each tonal step
  swept, its pitch, place and length moved, silenced, and an empty step given
  the recording's strongest pitch, with `Fidelity` as the fitness, a move kept
  only when it rises, each voice held within a tenth of its tonal floor, the
  fit's own distance from the recording's notes, so the score of the mix does not
  drift a voice into one that fills the mix.
- `bench` grades the analysis and the loaded models on fresh loops arranged as
  `corpus ... gen` arranges them, each at a steady tempo from a bar line, clean
  and augmented (`corpus ... aug`), and with `meters` each meter apart, the loops
  in the corpus's seven meters by its shares, so the bars of a loop match its
  meter (`ExpectedRip`'s). With `sampled <dir>`, both play half of a loop's
  tonal lines on NSynth's recorded notes in place of the voices: an instrument
  whose pitches reach the line's notes within three semitones, each note shifted
  from the nearest, cut into its recording's release at the note-off, mixed
  against the rest of the loop, and labelled with no voice.

[Rip design](rip-design.md) is where the rip is going from here: the loop that
refines voices and patterns against that score.
