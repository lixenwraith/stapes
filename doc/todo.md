# TODO

Where stapes may grow. Each item states what remains and the smallest thing that
would close it. Delete an item when it lands; this file is not a changelog.
Source files do not carry TODO comments.

Priorities: P1 is wanted next, P2 is convenient follow-up, and P3 is an idea.

The audio library's own items are in [its todo](https://github.com/lixenwraith/audio/blob/main/doc/todo.md);
the items here change stapes, and the library with it where they name its
files. Work that also changes vif is planned in [vif's
todo](https://github.com/lixenwraith/vif/blob/main/doc/todo.md#audio): stapes
on Android, which vif's app is to host.

## The rip

### Tune the rip

- Priority: P1
- Affected files: `lixenwraith/audio` (`rip.go`, `rip_test.go`), the root
  package

`RipSamples` ([Audio library
§9](https://github.com/lixenwraith/audio/blob/main/doc/audio.md#9-ripping-a-recording)) reads tempo, bars, key,
drums, melody and bass off a recording under `RipParams`, through the built-in
analysis or `ripmodel`; [Rip design](rip-design.md) is the whole plan.
The search over the parameters is in place (`tune`), as are the corpus the
engine exports (`corpus`), the model, its trainer (`tool/rip-train`) and the
front-end seam. What remains, in the design's order:

1. A corpus beyond the shipped bank: `corpus ... aug gen` puts a room, a mix,
   saturation and a tuning off A440 behind each loop and draws beats,
   melodies, bass lines and, in the free slot, a broken-chord accompaniment
   under a melody of longer notes, and since round seven an equaliser, a
   compressor, halls and a pedal's ringing; what it still lacks is a codec,
   chords as chords, phrases far softer than the rest of their loop (the
   soft notes below), chance steps graded as optional rather than as
   misses, the generative lead (left out since an offline render plays its
   first bar as the last render left it and labels every bar with its last:
   regenerate it at bar 0 and label it as played), and real recordings with
   their MIDI in the corpus and held out: a
   loader for a WAV beside a MIDI file (`ReadMIDI` reads the notes; the
   frames and labels a loop carries are what `corpus` writes) and a few
   recordings the user plays or owns with their scores, so neither a model
   nor a tuned set learns the synthesizer. Public sets with aligned MIDI
   (MAESTRO's piano, Slakh's rendered multitracks, GuitarSet, URMP) fit the
   same loader.
2. The model's pitch: on 20000 loops over 20000 steps the held-out decoders'
   loss was still falling (1.04) and the fourth run passed the analysis on
   augmented loops (0.51 against 0.48) on its drums while losing clean loops
   on tempo and both on pitch (melody 0.11 against 0.17); 300 loops
   overfitted in 12000 steps; the fifth run, on the detuned corpus, died at
   step 6820 of 20000 under the cloud container's restarts and was not
   finished. The sixth, at the new default four times the size (width 256,
   eight heads, 3.2 million weights), on 40000 loops of the accompaniment
   corpus over 40000 steps on a development machine's GPU
   (`tool/rip-train/README.md`, "A round on a development machine",
   `round.sh`), passed the analysis on both bench lines (0.51 against 0.47
   clean, 0.460 against 0.458 augmented, on its drums; melody 0.08 against
   0.13), its held-out loss flat from step 33000, and on recordings beat
   the analysis on drums (Billie Jean's false kicks a third) and lost on
   melodies (the origin's register 34 of 83 tapped notes against 50, Für
   Elise's 266 of 869 against 444). Round seven (229 bands, a corpus nearer
   a real mix, convolutions) and round eight (a strike head per pitch, an
   input in dB against the frame's own level) brought the model level with or
   past the analysis on most of the recordings' lines and past it on the
   bench; the eighth's strike decoder hears nearly every note at a low
   threshold and five times too many, and at its best threshold (0.9) keeps a
   fifth of the softest notes. Filtering the strikes (a pitch that never
   sounds, a stronger strike's partial in the register, a repeat within
   0.12 s, a ringing pitch restarted only over 0.95 and by the surest strike
   within a tone) brought the three recordings' pooled F1 to 0.678 at 0.8
   (the analysis 0.633). The release and velocity heads gained on the bench and
   not on the recordings (0.650). Next for it: real recorded notes as voices of
   the corpus (Iowa's samples carry no condition, NSynth's and E-GMD's only
   credit; MAESTRO's non-commercial licence would bind the weights), since
   better synthesis alone has not closed the gap; the corpus's pedal varied by
   the bar; the constants tuned by `tune` against the truths rather than swept
   by hand. Towers (7d: no gain for 43% more weights) are tried again once the
   heads compete with the strike head. For a rip of live music: a bench of the
   loops through a phone mic, a room and crowd noise at 20 to 0 dB, the same
   chain in the corpus, and the live rip in the hands (`RipStream` is the library's; its host is
   vif's Android app, and a verb over a file in stapes the step before it). Also: the fit gave Für Elise's bass a non-piano voice, with
   [vif#989](https://github.com/lixenwraith/vif/pull/989)'s drive trap under it. Then: real recordings with their notes in the
   corpus (item 1: `corpus <dir> from` writes E-GMD's, `train.py --mix` mixes
   them in; still lacking a held-out score for a `from` corpus, E-GMD's ET
   split being the clean one, drum velocity in the labels, a check that drops
   a render whose hits draw no onset; NSynth's notes play in `corpus ...
   sampled`, lacking the tempo-synced and multiphonic filters `examples.json`
   allows and a pipeline bench on the 53 held-out instruments), then more
   loops, or width 384 (`DIM=384 HEADS=12 FFN=1536`, 7 million weights),
   each measured by `bench` on clean and augmented loops and by the truths on
   the recordings, and shipped only when it beats the analysis on both; a
   hundredfold or more needs a GPU to train and would not run in the engine's
   budget. A vocal line is a line like any other to the rip; a
   voice has a vibrato and a vowel's formants now
   (`tool/rip-train/focus/vocal.toml` draws them), neither a gene, so the fit
   and the model's voice head do not search them: genes for both once a
   corpus with them has trained, and a real vocal recording held out. Tonal
   round two's voices (a pizzicato bass whose thump the analysis takes for a
   kick at times, drifting saws) cost the analysis a little on clean loops
   (`bench 48 4` 0.455 to 0.432, drums 0.43 to 0.39) and gain it a little on
   augmented ones (0.399 to 0.406); a corpus rendered with them trains the
   model on them. The analysis also leans on a lone oscillator starting every
   note at phase 0: at random phases two lines on one instrument stop parting
   (`TestRipTellsTheLeadByItsCharacter`), as a recording's would. Then
   the weights and a tuned parameter set shipped as data, `DefaultRipParams`
   kept for a recording nothing tuned for. [Rip design §8](rip-design.md#8-measured-on-the-sample-recording)
   has the runs' numbers.
3. A bass attack reads as a kick and a chord's attack as a snare in the analysis:
   a harmonic check on the hit (a percussive rise with a pitched tail is a note)
   until the model takes the front end.
4. Key changes within a recording are outside the model: a key per phrase (a
   chromatic note is an accidental on its step now). Through the analysis two
   voices in one register are decoded together; from a model's strikes any
   number of notes sound at once, parted into two lines, so a chord's inner
   notes land in the under line rather than a line of their own.
5. A sustained note with a wide vibrato can still read as two: the sample's
   wobble was its tuning, forty cents off A440, which the front end reads first
   now, and its notes hold for four to eight steps; a vibrato wider than a
   quarter tone would still cross rows. A pitch prior that holds a semitone
   wobble (a higher cost for a one-semitone move than per semitone of a leap,
   or the salience smoothed across the neighbours before the path), measured
   on a rendered vibrato voice, which the voices have now; or a cents head on
   the notes model and a step's `bend`, so a glide is one note
   ([Rip design §7](rip-design.md#7-plan)).
6. The decoder's and the parting's constants tuned against truths: `tune`
   searches `RipParams` over rendered loops whose truth is the render's; with
   `truth` files (rhythms, corrected transcripts, MIDI) on real recordings,
   the same search can take their agreement as its score, and the stream
   costs (`streamJump`, `streamLength`, `streamLevel`, `streamSwitch`), the
   lead rule's default and `cross` would be tuned on music rather than set by
   hand. On the origin the rhythm truth has the lead at 17 of 25 notes, 8
   missed, 10 extra and 7 cut in two, the half notes among them: the cut half
   notes are the first thing to chase, note by note, with the `truth` listing
   of which were cut. The model path's thresholds (`strike`, `strike_low`,
   `strike_again`, `release`, `voicing`) were swept by hand per weight file;
   `tune ... truth <list>` searches them through the model, each recording
   heard once, by the pooled note F1: run after every round on the real
   validation list (`~/data/rip/val-list.tsv`, GuitarSet's player 04; the
   MusicNet pieces join it once their audio is on disk again), it is the
   smallest close ([Rip design §7](rip-design.md#7-plan)). A bar-phase score belongs
   beside it: the note scorer's time map never sees a bar line, and the rip's
   lines fall within an eighth of a bar of GuitarSet's downbeats on 72% of
   them after audio v0.8.9, 63% before. When `tune ... truth` scores the bars
   too (`MatchBars`), the tempo and beat genes (`tempo_prior`, `tempo_width`,
   `tightness`) join `modelGenes`, which stays a list, not a rule. Its first
   run through 32a gained solo guitar .06 and lost .02 to .04 on the renders,
   the clean take, the phones and MusicNet, as its list is guitar alone and
   its score counts within lines, where a guitar's truth has no bass line:
   before it writes a `rip.toml` it wants a count by pitch and onset, a list
   with a condition a kind, a guard on every set's loss, and the frames it
   heard kept between runs (six CPU-hours for sixty recordings without).
7. The pedal: a piano arpeggio under pedal rings three or four notes at once,
   and the analysis's two lines hold two; the second line's eighths read as
   held notes (30 of 58 heard, 11 spanning two). A model's strikes start every
   note struck however many ring (the intro's under line 55 of 58 at a
   strike threshold of 0.5); what the arrangement still lacks is a chord read
   per beat as the set of ringing pitches, rendered as a pad under the lines.
8. The fit as it fits a real lead: the user hears the lead as a plucked string
   (a harp or a santur) and the bass as a flute at times; the fit picks modal
   voices for every line of both recordings. A plucked voice (`string`) with
   a doubled, beating course, and the fit's samples cleared of the other
   lines (item 4 of the loop below), measured by the floor the fit reaches.
9. The judge as a trained stage ([Rip design](rip-design.md) §7): a verb that
   writes a list of truth-backed recordings as rows (each reported note and
   each strike over a half with the accounting's evidence, the heads around
   it and whether a truth note answers it), the fit in `tool/rip-train`, and
   the weights read by audio beside the models'. Its fit takes licensed
   collections, renders and the engine's mixes, never a full track, and is
   read with a collection left out, singing reported and never a gate. Open
   once one is trained: a bass line under drums when recordings are in the
   fit.

### The instrument model

- Priority: P1
- Affected files: `tune.go` (`corpus`), `tool/rip-train`, `lixenwraith/audio`
  (`ripmodel`, `rip.go`)

What plays is heard per pitch, on the notes model, from the first round ([Rip
design §7](rip-design.md#7-plan)): a family, a wave and the genes of each pitch
cell's note, so that every note has a voice the engine plays and a name only
where one is sure. `corpus` labels each event's family and, for the engine's
notes, its wave and genes; the frame model has a family head a frame, which
names the highest note alone (the bass on 2.6% of corpus 17's frames, so never
a lower line) and is taught `unknown` on a third of them, every engine voice
far from the bank. What remains: the heads a cell and their reading into a
note; an engine voice far from the bank taught as `synth`, and no name where no
family's posterior reaches what a family held out reaches (`--family-holdout`);
real notes labelled by the fit, an instrument of NSynth's at a time; the frame
model's planes as context to those heads, measured as a pair; and the round,
judged on NSynth's held-out instruments by each family's recall a note and the
rip's voice over the fit's distance (2.3 now), with E-GMD's kits and Billie
Jean's kicks, clean and from the Pixel, unmoved. Then lines by voice: notes
gathered by their voices and their continuity, first inside the three lines a
rip holds, then as many as were heard, through the song and the game.

### A live rip

- Priority: P2
- Affected files: `cmd_doc.go`, vif's Android app

`live` drives the library's stream (`RipStream`) over a file in blocks and
measures it against the offline rip of the same recording, the measurement the
Android app's microphone mode (vif's todo) is judged by ([Rip design
§7](rip-design.md#7-plan)). What is left:
- The measure through the trained models on real recordings: no weights or
  recordings were where the verb was written. A scratch tool through 12a and
  32a on twelve recordings read 95 in a hundred of the offline rip's notes and
  its bar grid on three, tempo, meter and phase being held from the first six
  bars, at three times the offline cost; `live` is to say the same.
- The lines. On a loop of the bank through the analysis the stream has 79 of
  the offline rip's 80 notes and every bar line, but 18 of the melody
  register's 48 on the line the offline rip files them on (47 as one
  register): the stream parts the register's two lines over the window it
  holds when a bar freezes (`streams`), the offline rip over the whole
  recording. A view that colours a line shows it.
- The stream's own reads (`Frames`, `Open`) are for the view; `live` does not
  ask for them, so a recording does not yet check them through a trained
  model.
- Decided since the audit: a forward that runs each frame once brings the
  stream's cost to the offline rip's; then a recording is read in sections,
  tempo, meter, phase and key decided a part at a time and again at every bar
  until two decisions agree, nothing carried across a boundary between two
  different parts of a song.

### Close the rip's loop

- Priority: P1
- Affected files: `lixenwraith/audio` (`rip.go`, `song.go`), the root package

[Rip design](rip-design.md) §2 stage 7. The voice fit (`rip`, `fit`), the
dictionary and the song (`compress`, `song`) and the song's score against the
rip are in place; what remains:

1. The refinement loop (`refine`) searches the voices and sweeps the tonal
   steps, a semitone or an octave either way or silence per step, judged by
   `Fidelity` against the recording's own frames; what it does not yet move is
   a step's position or length or the drum part, and a note the front end
   never heard it cannot add. Measure its convergence on rendered loops whose
   truth is known, and whether the notes it inserts are the truth's:
   `Fidelity` puts no price on a wrong note ([rip design](rip-design.md) §7).
2. Matching by reconstruction: a bar's distance to a member as the rip of the
   member's render against the bar, in place of `BarDistance`, with `Memoize`
   paying for each render once; measure it against the mechanical grouping.
3. Several voices in a register: the two lines are decoded together now and
   parted again by pitch, length and strength, the lead the busier line or
   the one the listener names (`rip ... lead`); a colour per note as one more
   thing a line keeps in `streams`, from `RipNote.Bright`, an unmeasured note
   (`RipUnmeasured`, its fundamental under a concurrent note's partial
   throughout) costing a neutral amount; if the unmeasured share matters
   there, subtract the covering partial, estimated from its own note's
   spectrum, and measure it against the rhythm truth; a third line, and a
   chord's inner notes, are not heard. The bass register's rows end at E3 and
   the bass decoder reaches A3 over the melody's rows; a line that lives
   around the join is still heard on two scales.
4. The fit's sample notes are the mix around them: subtract the other register's
   estimate from the recording before the mel frames, or fit on the frames a
   trained front end attributes to the note.
5. `refine` takes minutes a round, every judge a render and a hearing of the
   song: candidates judged in parallel through private engines (the registry
   is the one shared thing), and the sweep's bars rendered once a pattern.
6. Listening as data: an A/B of two renders where the listener picks the
   nearer, and a line named as the lead, kept as files beside the truths; a
   few dozen such pairs would set the weights of `Fidelity`'s three terms and
   the lead rule's default by what ears prefer rather than by hand.

7. A rip's voice from what was heard: the voice heads give nearly the mean of
   their training genes on real recordings and the wave head a saw for
   anything bowed, blown or sung, so `rip` fits every line and the heard voice
   is its first seed. What remains is the voice of every note (the
   instrument model, above) and a distance that does not favour a dark voice.
8. Put off: a second piece of music under the first, which the rip
   transcribes with it (a third to a half more notes); it wants the judge to
   follow instruments as it follows notes.

## Files and banks

### Search more than one folder

- Priority: P2
- Affected files: `session.go`, `tui_files.go`, `tui.go`, `cmd/stapes/main.go`

The search path is one folder, vif's audio folder, and `stapes.toml` holds the
mixer period alone ([Stapes §2](stapes.md#2-its-files)). What remains: flags and
`stapes.toml` keys that set the search path's folders in order and remember the
last folder the files overlay showed; every document listed across the path's
folders, not only the first that holds a name; and the UI to show them, the
files overlay naming each file and a status bar at the foot of the screen giving
the full path of the one selected, as a file manager or a browser shows a link.

### Give stapes a bank of its own

- Priority: P3
- Affected files: `session.go`, a new `bank/`, `AGENTS.md` (Data)

Stapes registers `audio/bank`'s sounds, voices and patterns, the set vif ships.
A generic set of its own, rarely updated and filled mostly by rips, would share
only the high-quality tonals from `audio/bank`, so what stapes plays and
exports stops moving with what vif ships; the rule that stapes embeds no files
then names that folder.

### Keep what a save still loses

- Priority: P3
- Affected files: `session.go`, the library's `bank` and `sound_load.go`

`save` writes a document over the file it came from, and patterns seeded by
`builtin` over the bank's own document, so the macros and comments there stay
([Stapes §2](stapes.md#2-its-files)). What it does not keep:

- A sound document seeded by `builtin`, as the TUI seeds an empty one at its
  start, has no file behind it, so its save writes every sound out and no
  comment: 17 KB in 1321 lines for the 19 KB in 964 of the bank's three files,
  which one source does not span.
- A sound document's root gains every entry its includes hold, written out
  among its own, since the root shadows them by name (`saveSounds`). The
  smallest close: the save leaves out an entry its includes load the same.
- A comment inside an entry is lost, which is the library's to keep ([its
  todo](https://github.com/lixenwraith/audio/blob/main/doc/todo.md#keep-a-comment-inside-an-entry-through-a-save)).

## The session

### Keep the slots through `music reset`

- Priority: P2
- Affected files: `cmd_audio.go`, `tui.go`

`music reset`, which the browser's `m` runs to bring the music back to its first
bar, resets the sequencer, and that empties every slot, while the session's
`slots` keep what `slot` last asked: the music then plays nothing, `render` still
renders the old slots, and a slot's track mutes wait for their pattern to be put
back. The fix is reset slotting the session's patterns again, or a sequencer
reset that keeps them.
