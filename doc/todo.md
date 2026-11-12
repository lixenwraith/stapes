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
analysis or `ripmodels`; [Rip design](rip-design.md) is the whole plan.
The search over the parameters is in place (`tune`), as are the corpus the
engine exports (`corpus`), the frame model and the notes model, their trainer
(`tool/rip-train`) and the front-end seam. What remains, in the design's order:

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
   same loader. MusicNet's labels are snapped to the strike peaks; a note left
   unsnapped (`"snap": false`) is still to be masked out of the strike loss and
   the strikes' F1, its pitch's target within eight frames of the label.
2. The frame model's pitch: on 20000 loops over 20000 steps the held-out decoders'
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
   input in dB against the frame's own level) brought the frame model level with or
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
   and the models' voice heads do not search them: genes for both once a
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
   until the frame model takes the front end.
4. Key changes within a recording are outside the rip: a key per phrase (a
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
   on a rendered vibrato voice, which the voices have now. Through a notes
   model with the cents head the decoder follows the sound across the cells
   and writes one note ([Rip design §7](rip-design.md#7-plan)); left there:
   - A round trained on a `glide` corpus, judged on vocadito by its fragments
     (156), wrong cells (76), repeats (69) and how many of the 241 legato
     steps still read as two notes. A small run (360 loops, 3,000 steps on a
     CPU) learned a loop's tuning and little of a glide inside a mix: steady
     readings 11.9 cents off, gliding ones 22.5 where a head that knows
     nothing reads 25, pulled toward the cell's centre at every slope with no
     lag. It states 25 there, so the decoder joins nothing; on a solo render
     it states 3 to 11 and joins. If a full round stays there, the glides'
     share of a corpus or a loss that weighs the moving frames is next.
   - The stated error is fitted to the offset's miss with the miss's gradient
     stopped, and its own gradient still reaches the features the heads
     share. The offset read the same with the channel and without, 14.5
     against 14.6 cents on the same held-out loops. If a full round shows a
     cost, the channel reads the features with their gradient stopped too:
     no cost by construction, and an error that can only say what the other
     heads' features hold.
   - One error from validation carried in the weight file, the same margin at
     every reading, was the other rule with no number set by hand. Not taken:
     the miss differs twofold by place (steady 11.9, gliding 22.5), so one
     number joins legato steps on a steady voice or nothing on a glide, and
     the file gains a field (a transformer change and a format version). It
     is the fallback if `within` reads well under 63% on real singing.
   - A corpus scoop rises over its note's whole length, since a step's bend
     ends with its note: a scoop that settles early, as a singer's does, needs
     audio's step to glide to a share of its length (audio's todo).
   - A recorded note is placed where its list's `pitch` column names the
     recording's pitch track
     ([Rip training §3](rip-training.md#a-recordings-pitch-track)), and
     `--validate` then prints its gliding, steady and beside rows; left:
     - The runs: GuitarSet's and vocadito's tracks into their corpora, the
       cents head read on that playing, then trained on it (`--mix`), and the
       labels pass over corpus 17.
     - A collection with no pitch annotation (MusicNet, a phone take) stays
       unplaced. A track estimated from the audio (pYIN on a solo line) was
       not taken: its errors would be taught as the label.
     - `sampled` lines still ask the head nothing: where an NSynth note sits
       in its cell after its shift is not in its label.
     - A recorded note's readings are gliding when its track spans half a
       cell over the note. Not taken: the engine's rule, a frame unlike its
       neighbour, which a tracker's jitter meets at every frame (258 of 258
       readings of a jittered track, against 172 gliding and 86 steady by the
       span), so the steady row would be empty; and a frame moving faster
       than a rate, which would leave a scoop's settled frames steady, where
       the span calls the whole note gliding, but needs a rate and a
       smoothing set by hand. The rate is next if settled frames fill the
       gliding row on real singing.
     - Two strings holding one pitch at one onset, a unison, give both of the
       score's notes the first string's contour.
6. The decoder's and the parting's constants tuned against truths: `tune`
   searches `RipParams` over rendered loops whose truth is the render's; with
   `truth` files (rhythms, corrected transcripts, MIDI) on real recordings,
   the same search can take their agreement as its score, and the stream
   costs (`streamJump`, `streamLength`, `streamLevel`, `streamSwitch`), the
   lead rule's default and `cross` would be tuned on music rather than set by
   hand. On the origin the rhythm truth has the lead at 17 of 25 notes, 8
   missed, 10 extra and 7 cut in two, the half notes among them: the cut half
   notes are the first thing to chase, note by note, with the `truth` listing
   of which were cut. The models' thresholds (`strike`, `strike_low`,
   `strike_again`, `release`, `voicing`) were swept by hand per weight file;
   `tune ... truth <list>` searches them through the loaded models by pitch and
   onset, as the mean over its list's conditions under a guard, the frames
   kept between runs ([Rip design §7](rip-design.md#7-plan)). Left of it: the
   real validation list (`~/data/rip/val-list.tsv`, GuitarSet's player 04;
   the MusicNet pieces join it once their audio is on disk again) names no
   condition and is guitar alone, so it wants a condition a kind, of the
   collections and renders a fit may hold and never a full track, and
   `timed` on GuitarSet's lines, before a run after every round writes a
   `rip.toml`; a timed truth's drum hits still pass through `MatchHits`' own
   map; and kept frames outlive a change to the models' code, which their
   names do not hold. A bar-phase score belongs beside it: the note scorer's
   time map never sees a bar line, and the rip's lines fall within an eighth
   of a bar of GuitarSet's downbeats on 72% of them after audio v0.8.9, 63%
   before. When `tune ... truth` scores the bars too (`MatchBars`), the tempo
   and beat genes (`tempo_prior`, `tempo_width`, `tightness`) join
   `modelGenes`, which stays a list, not a rule.
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
9. The judge as a trained stage ([Rip design](rip-design.md) §7) is built and
   none is fitted on recordings or installed: `rows`, `tool/rip-train`'s
   `judge.py` and the root's `ripjudge.bin`
   ([a judge](../tool/rip-train/README.md#a-judge)). Left:
   - The fit: licensed collections, renders and the engine's mixes, never a
     full track, read with a collection left out, singing reported and never
     a gate; and a bass line under drums when recordings are in the fit. The
     first, on 32a's rows, passed with each collection left out and is not
     installed: on Billie Jean it reports as many low notes with eight fewer
     right, and on vocadito the fit holding everything reports 400 notes for
     742, which the row's next version, with the note's voice, is for.
   - `bench` grades the loaded models under the installed scores whatever the root
     holds, as a round is benched before a judge is fitted on its weights.
     Once a judge is accepted it should grade what `rip` hears.
   - `rows` knows a recording by its audio's checksum, so a truth corrected
     after its rows were written is not read again: remove the recording's
     `.json`. The smallest close: the truth's checksum beside the rows.
   - A count from rows reads apart from `truth`'s decode, by a note under
     the installed scores and by .007 of F1 under a small run's judge (the
     trainer's README says why); audio's own open items are in
     [its todo](https://github.com/lixenwraith/audio/blob/main/doc/todo.md#close-what-a-judge-leaves-open).
10. A performance's character, generated: how loudness and timing move in the
    licensed performances (GuitarSet's and MusicNet's annotations against a
    steady grid), measured there and then drawn in `corpus`, judged as any
    corpus change, a pair. No round may learn it from a track.

### Teach the frame model the beat and the bar line

- Priority: P1
- Affected files: `tool/rip-train` (`train.py`, README), `tune.go` (`corpus`),
  `doc/rip-training.md`, the weights in the root

The trainer's half is built on audio v0.24.0, which reads the two heads: the
heads, a rendered loop's targets from its label and a recording's from its
list's `beats` column, the mask, the loss with no weight, each head's F1 on the
lines ([Rip training §3, §5 and §8](rip-training.md#the-two-heads-on-a-small-run),
[rip-train](../tool/rip-train/README.md#the-beat-and-the-bar-line)). One small
run on rendered loops holds it together and no more: its heads read the bars of
56 of 160 held-out loops right through v0.24.0, and of 85 through the decoders
audio's branch `claude/audio-fixes` mended to read them by rank, as audio has
since v0.24.1 (not read again on a tag), where the same weights' cues read 124.
What is left is the reading on trained weights:

- **A round with the heads**, from the installed frame model through `--init`,
  on `meters` corpora and on recordings whose lists name their beats
  (GuitarSet's JAMS files, E-GMD's own MIDI). `--probe` first: it trains the
  two heads alone and cannot cost the heads the decoders read, and says
  whether the installed model's features hold the beat at all; a fine-tune
  only if they do not. Then the 120 GuitarSet clips no
  round met (37 right through the cues, 58 under perfect beats), the Für Elise
  takes and the stream's twelve recordings, each with the heads and with the
  same weights' heads left out, and what [audio's
  todo](https://github.com/lixenwraith/audio/blob/main/doc/todo.md#hear-the-beat-and-the-bar-line-on-trained-weights)
  leaves to them, read by [the rule](rip-training.md#reading-a-candidate). Run
  as R2 (39a, 40a; [Rip training §8](rip-training.md#the-two-heads-on-12a-the-probe-and-the-round)):
  the heads lose bars beyond chance; the body gains on real kits alone and
  loses under a mix, which the round never heard. Neither is installed. A
  plain tracker reads 39 of the 120 clips' bars right from 40a's heads where
  audio's decode reads 24, so mending the decode is audio's. R3's probes from
  40a (43a, 44b, 44c) moved drums under a mix and not the beats. Left: the
  frame model trained again from no weights on what these rounds taught
  (stapes#49), its bars read through audio's mended decode.
- **What the small run says a round must change.** It was read twice: through
  audio v0.24.0 ([Rip training §8](rip-training.md#the-two-heads-on-a-small-run)),
  and through the decoders mended on audio's branch `claude/audio-fixes`,
  which read the heads by rank, as audio has since v0.24.1, and leave a clip
  they give no reading of to the cues. Mended, the heads' own grid stands on
  139 of the 160 loops and is right on 74, where the cues are right on 113 of
  those; the other 21 are the cues' rips, 11 right. In the order of what is
  left:
  - Which note is the beat. The beat head marks an eighth that holds an
    onset about alike whatever the meter. With drums it says 0.16 halfway
    between 4/4's beats and 0.31 between 3/4's, where no beat is, and 0.30
    on 6/8's eighths between its dotted quarters, where one is; without
    drums 0.40, 0.48 and 0.47. Of the 65 loops still wrong on the heads' own
    grid, 44 count another note as the beat: 9 of 6/8's 14 with drums, 16 of
    4/4's 23 without. No decoder reads a level the head does not state:
    mended, 14 loops of 4/4 are read as bars of four eighths, where v0.24.0
    read 2 so. The meter is to be heard before the beat is, and a model's
    reach is 2.97 s either side, a bar or two.
  - 4/4's line against its third beat, less than it first read. The
    generated beats kick on both; on the loops' own beats the line tops its
    bar's beats in 76 bars of a hundred with drums and three in 21, and the
    line elsewhere is 7 loops of 71 mended, where it was 21 of 78. By rank
    on the right beats the downbeat head has the line in most bars of 123 of
    the 160 loops, 33 of the 53 without drums; the cues read 124 and 28
    right. So the downbeat head is near the cues once it has the beats.
  - A quiet downbeat head is no longer a round's to mend: it said 0.28 on a
    drumless line, v0.24.0 took the longest bar offered for it, and by rank
    it is read where it is right. What the drumless loops still want is the
    first point, and recordings with their beats.
- Left open by what landed, each with the option not taken:
  - The two losses count once each in the total, beside ten heads' losses
    that sum to ten times theirs or more, and a bar line's mark is under 2
    frames in a hundred.
    Not taken: a weight on the two losses whole, which leaves a posterior a
    posterior and hands them more of the shared layers; it is the first
    thing to try, as a pair of runs. Never a weight on the positive class.
  - A recording is asked from its first listed beat's mark to its last's.
    Not taken: the mask run on to where the next beat would fall, as a
    rendered loop's runs to its end's line: a played tempo's next beat is a
    guess, and a zero there could teach silence on a beat nobody listed. If
    a round's heads fire in a recording's tail, the decay after its last
    beat is the place to ask.
  - A window `--over` lays drums under is asked nothing, though the drums'
    own beats are known where their list names them. Not taken: asking the
    drums' grid there, since the loop under them keeps another and which a
    listener counts is not the label's to say.
  - A corpus cycles six tempos, which its held-out loops share: a round wants
    `tempos` over a list, or `--bend`, before its held-out F1 says anything
    of a tempo it never met. `pipeline.sh` has no knob for `meters`.
  - The F1 counts a peak right within four frames, the strikes' reach, 46 ms;
    beat trackers are scored at 70 ms. Not taken: a reach of its own, a
    number set by hand for a line audio does not read.

### The instrument model

- Priority: P1
- Affected files: `tune.go` (`corpus`), `tool/rip-train`, `lixenwraith/audio`
  (`ripmodels`, `rip.go`)

What plays is heard per pitch, on the notes model, from the first round ([Rip
design §7](rip-design.md#7-plan)): a family, a wave and the genes of each pitch
cell's note, so that every note has a voice the engine plays and a name only
where one is sure. `corpus` labels each event's family and, for the engine's
notes, its wave and genes; the frame model has a family head a frame, which
names the highest note alone (the bass on 2.6% of corpus 17's frames, so never
a lower line) and is taught `unknown` on a third of them, every engine voice
far from the bank. Landed: the heads a cell, their targets and their reading
into a note's voice ([Rip training §5, §7 and
§8](rip-training.md#5-the-targets)), an engine voice far from the bank taught as
`synth`, and no name where no family's share passes what a family held out
reaches (`--family-holdout`, the measure the weight file carries). What
remains: real notes labelled by the fit, an instrument of NSynth's at a time;
the frame model's planes as context to those heads, measured as a pair; and the
round, judged on NSynth's held-out instruments by each family's recall a note
and the rip's voice over the fit's distance (2.3 now), with E-GMD's kits and
Billie Jean's kicks, clean and from the Pixel, unmoved. Then lines by voice:
notes gathered by their voices and their continuity, first inside the three
lines a rip holds, then as many as were heard, through the song and the game.

Left open by what landed, each with the option not taken:

- The measure is one family's reach at one quantile (`SURE`, 0.95), and the
  file that ships takes it from another run of its recipe (`--sure`), since a
  run that teaches every name has no family left to measure with. It bounds
  how often a family never taught is named, a note in twenty, and not how
  often a name given is right: on 720 loops for 3,000 steps 2% of the taught
  families' notes pass it and half of those are named right. Not taken:
  each family held out in turn and the highest reach carried, a run a family;
  and the reach of notes outside the eight families on a validation corpus,
  NSynth's mallets and organs, which would measure the shipped run itself and
  wants those notes in `--validate`'s list first.
- An engine voice far from the bank is asked `synth` whatever it sounds like,
  41% of the asked cells, beside piano, bass and guitar for the same modal and
  string sounds: with the body free and at full weight the heads named bass
  and the synths alone, and solo singing, named one way everywhere, on 96% of
  its notes. Not taken yet: an engine voice named by what it sounds nearest,
  its wave and genes against the families', or asked nothing far from the bank.
- A cell two events sound in is asked nothing, 4% of the sounding cells. Not
  taken: asking it where the two are one voice, a twentieth of those, a second
  rule for a fifth of a percent; and asking it both names, which a softmax
  over the names cannot hold and a head of one name against the rest did not
  earn on the held-out family.
- An embedding with linear readouts in place of the heads: no better at any
  width, short of the genes under 16 values, and its readouts need a place in
  the weight file ([Rip training §7](rip-training.md#7-the-models)). It is the
  way if audio must tell every pitch of a frame its voice, not twelve.
- The measure rides in the weight file as a frame head of no weights, and a
  measure a family would fit the same head at eight values. Not taken: named
  constants in a version of the file after 5, which changes transformer's
  format and loader; what brings them is a constant not the notes model's own
  or one a caller needs before any frame is heard ([audio's
  todo](https://github.com/lixenwraith/audio/blob/main/doc/todo.md#carry-a-notes-voice-further)).
- What plays a cell is taught at a quarter of the frame's like weights
  (`PLAYS`), since its heads teach the channels the decoders read: on 720
  loops for 3,000 steps the whole weight took the strikes' F1 from 0.328 to
  0.278 and a quarter leaves 0.301, the heads all but as good ([Rip training
  §8](rip-training.md#8-a-training-step)). Not taken: the heads read with the
  gradient stopped at the cells, which costs the decoders nothing and teaches
  the heads nothing; channels of the cells kept for what plays, or a layer
  between the cells and those heads, which the weight file has no tensor for;
  and a smaller weight still. A real round reads the strikes against a run
  without the heads at equal steps before any of them.
- A short run on generated loops teaches the names little past register:
  families 0.551 against 0.563 by the octave's most common name, waves 0.456
  against 0.427, the genes 0.218 off against 0.209 by the octave's mean voice,
  and a piano, a guitar or a sung voice, engine voices near the bank's among
  random ones, under 0.07 of their notes. The names want recorded families
  (NSynth's) and a round's length. Not taken: weighing a name by its rarity,
  which moves the shares the measure reads.
- The lines count a note told on any frame: one never among a frame's twelve
  most salient pitches is in no share, so the count of notes moves a little
  with the notes model's salience from step to step.
- `test_train.py` runs by hand: CI has no PyTorch.

### A live rip

- Priority: P2
- Affected files: `cmd_doc.go`, vif's Android app

`live` drives the library's stream (`RipStream`) over a file in blocks and
measures it against the offline rip of the same recording, the measurement the
Android app's microphone mode (vif's todo) is judged by ([Rip design
§7](rip-design.md#7-plan)). What is left:
- Measured through 12a and 32a on twelve recordings: the stream has 95 in a
  hundred of the offline rip's notes (2,721 of 2,850) and its bar grid on
  three or four, tempo, meter and phase being held from the first six bars,
  as a scratch tool read before each frame ran once; a bar freezes 6 to 8 s
  behind its end, the first ones 11 to 17 s in; a second of audio costs it
  0.24 s of a core, 1.4 times the offline rip, where it cost three times. Not
  measured: `pace`, and a phone.
- The wait. The ear waits the frame model's reach, 2.97 s, so the stream's
  frames are the offline rip's; audio's `FrontEnd.Ahead` waits less and no
  verb sets it. Half a second leaves the notes where they are and lands the
  bar grid on five of the twelve, and moves the drums ([Rip design
  §7](rip-design.md#7-plan)). A pitched note needs the notes model's 0.53 s
  alone: each model's frames at its own reach is audio's todo.
- The lines. On a loop of the bank through the analysis the stream has 79 of
  the offline rip's 80 notes and every bar line, but 18 of the melody
  register's 48 on the line the offline rip files them on (47 as one
  register): the stream parts the register's two lines over the window it
  holds when a bar freezes (`streams`), the offline rip over the whole
  recording. A view that colours a line shows it.
- The clock. The stream and the offline rip share the recording's, and `live`
  still matches them untimed, through a fitted map and the transposition that
  matches most (`MatchNotes`): timed, as `truth ... timed` matches, a stream
  that hears late or a semitone off counts as wrong.
- The stream's own reads (`Frames`, `Open`) are for the view; `live` does not
  ask for them, so a recording does not yet check them through a trained
  model.
- Next, since the bar grid waits on it and not on the ear: a recording is
  read in sections, tempo, meter, phase and key decided a part at a time and
  again at every bar until two decisions agree, nothing carried across a
  boundary between two different parts of a song.

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

## The tests

### Take the race suite's time down

- Priority: P2
- Affected files: `stapes_test.go`, `.github/workflows/test.yml`,
  `.github/race-seconds.txt`

The suite takes sixteen minutes under the race detector on four cores, up to
twenty-five on a CI runner, which the workflow splits over seven jobs, each
about as long as the heaviest test
([agent notes](agent-notes.md#the-race-suite-and-ci)). What is left:

- The three heaviest tests are 37% of it and set the shortest a job can be:
  the fitted and refined song 129 s, the judge 120 s, `live` 97 s. Each runs
  what its rule takes at the least. The refinement's every judge renders and
  hears the whole song; the judge test's time is the rip's accounting
  (`RipFrames.account`, 72% of it plain), which only audio shortens; `live`
  hears twelve seconds through two ears, each a stream and two rips.
- Two splits were weighed and not taken, since each takes tests from the
  detector. Racing the light tests alone and running every test plain: the 27
  tests that render or hear audio for half a second skip under `-short`, and
  the raced run is 113 s on four cores with the lightest test of each
  goroutine stapes starts kept in it (the fit's and `tune`'s searches,
  `corpus from`'s workers, the screen's long verbs), but a race only a heavy
  test reaches goes unseen. Or that on a pull request and the whole raced
  suite on main, which shows such a race after its merge.
- A push to a branch with a pull request open runs the workflow twice, for the
  push and for the pull request; `claude/*` in the push trigger is what tests
  a branch before its pull request opens.
- A job uses one core of four: tests run one at a time while a session
  registers in the audio module's process-wide registries.
