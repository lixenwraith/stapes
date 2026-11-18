# TODO

Where stapes may grow. Each item states what remains and the smallest thing that
would close it. Delete an item when it lands; this file is not a changelog.
Source files do not carry TODO comments.

Priorities: P1 is wanted next, P2 is convenient follow-up, and P3 is an idea.

The audio library's own items are in [its todo](https://github.com/lixenwraith/audio/blob/main/doc/todo.md);
the items here change stapes, and the library with it where they name its
files. What vif's app does to host stapes on Android, and to give it the
microphone, is in [vif's
todo](https://github.com/lixenwraith/vif/blob/main/doc/todo.md#audio); an item
here that meets it links there.

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

1. A corpus beyond the shipped bank: `corpus ... aug gen meters overlaps`
   arranges every loop over the 88 keys in seven meters or lays single notes
   in none ([Rip training §3](rip-training.md#arranged-loops)), and puts a
   room, a mix, saturation, a codec and a tuning off A440 behind it; what it
   still lacks is chance steps graded as optional rather than as misses, the
   generative lead (left out
   since an offline render plays its first bar as the last render left it and
   labels every bar with its last: regenerate it at bar 0 and label it as
   played), and real recordings with their MIDI in the corpus and held out: a
   loader for a WAV beside a MIDI file (`ReadMIDI` reads the notes; the
   frames and labels a loop carries are what `corpus` writes) and a few
   recordings the user plays from public-domain scores, with the scores, so
   neither a model nor a tuned set learns the synthesizer. Public sets with
   aligned MIDI (GuitarSet, URMP) fit the same loader; MAESTRO, non-commercial,
   and a commercial song's transcription or a render of one stay out (AGENTS,
   Data).
2. The frame model's pitch: next for it, real recorded notes as voices of the
   corpus, from the sets the [data rule](../AGENTS.md#data) lets train; the
   constants tuned by `tune` against the truths rather than swept by hand.
   Towers are tried once the heads compete
   with the strike head. For a rip of live music: a bench of the loops through
   a phone mic, a room and crowd noise at 20 to 0 dB, the same chain in the
   corpus, and the live rip in the hands (`RipStream` is the library's; its
   host is vif's Android app, and a verb over a file in stapes the step before
   it). Then: real recordings with their notes in the corpus (item 1: `corpus
   <dir> from` writes E-GMD's, its hits moved onto their sound or the
   recording left out, `train.py --mix` mixes them in; still lacking a
   held-out score for a `from` corpus, E-GMD's ET split being the clean one,
   drum velocity in the labels, its list saying `none` till a kit's law is
   decided, the lag's constants read back on E-GMD itself (by drummer and
   session, and whether a performance's kits agree), a lag past half the usual
   gap between strikes, which reads as a nearer one, and a check that drops a
   render whose hits draw no onset; NSynth's notes play in `corpus ... sampled`, lacking the
   tempo-synced and multiphonic filters `examples.json` allows and a pipeline
   bench on the 53 held-out instruments), then more loops, or width 384
   (`DIM=384 HEADS=12 FFN=1536`, 7 million weights), each measured by `bench`
   on clean and augmented loops and by the truths on the recordings, and
   shipped only when it beats the analysis on both; a hundredfold or more
   needs a GPU to train and would not run in the engine's budget. A vocal line
   is a line like any other to the rip; a voice has a vibrato and a vowel's
   formants now (`tool/rip-train/focus/vocal.toml` draws them), neither a
   gene, so the fit and the models' voice heads do not search them: genes for
   both once a corpus with them has trained, and a real vocal recording held
   out. The analysis also leans on a lone oscillator starting every note at
   phase 0: at random phases two lines on one instrument stop parting
   (`TestRipTellsTheLeadByItsCharacter`), as a recording's would. Then the
   weights and a tuned parameter set shipped as data, `DefaultRipParams` kept
   for a recording nothing tuned for.
3. A bass attack reads as a kick and a chord's attack as a snare in the analysis:
   a harmonic check on the hit (a percussive rise with a pitched tail is a note)
   until the frame model takes the front end.
4. Key changes within a recording are outside the rip: a key per phrase (a
   chromatic note is an accidental on its step now). Through the analysis two
   voices in one register are decoded together; from a model's strikes any
   number of notes sound at once, parted into two lines, so a chord's inner
   notes land in the under line rather than a line of their own.
5. A sustained note with a wide vibrato can still read as two: the front end
   reads a recording's tuning first now, and a vibrato wider than a quarter
   tone would still cross rows. A pitch prior that holds a semitone wobble (a
   higher cost for a one-semitone move than per semitone of a leap, or the
   salience smoothed across the neighbours before the path), measured on a
   rendered vibrato voice, which the voices have now. Through a notes model
   with the cents head the decoder follows the sound across the cells and
   writes one note ([Rip design §7](rip-design.md#7-plan)); left there:
   - A round trained on a `glide` corpus, judged on vocadito by its fragments,
     wrong cells, repeats and how many of its legato steps still read as two
     notes. If a round's head learns little of a glide inside a mix, the
     glides' share of a corpus or a loss that weighs the moving frames is
     next.
   - The stated error is fitted to the offset's miss with the miss's gradient
     stopped, and its own gradient still reaches the features the heads
     share. If a full round shows a cost, the channel reads the features with
     their gradient stopped too: no cost by construction, and an error that
     can only say what the other heads' features hold.
   - One error from validation carried in the weight file, the same margin at
     every reading, was the other rule with no number set by hand. Not taken:
     the miss differs by place, steady or gliding, so one number joins legato
     steps on a steady voice or nothing on a glide, and the file gains a field
     (a transformer change and a format version). It is the fallback if
     `within` reads well under 63% on real singing.
   - A corpus scoop rises over its note's whole length, since a step's bend
     ends with its note: a scoop that settles early, as a singer's does, needs
     audio's step to glide to a share of its length (audio's todo).
   - A recorded note is placed where its list's `pitch` column names the
     recording's pitch track
     ([Rip training §3](rip-training.md#a-recordings-pitch-track)), and
     `--validate` then prints its gliding, steady and beside rows; left:
     - The runs: GuitarSet's and vocadito's tracks into their corpora and the
       cents head read on that playing, then trained on GuitarSet's (`--mix`),
       since vocadito only measures.
     - A collection with no pitch annotation (MusicNet) stays unplaced. A
       track measured from the audio is taken only by a procedure stated,
       fitted on sound whose answer is known and checked on what it was not
       fitted on, as VocalSet's is; an unchecked estimate (pYIN on a solo
       line) is not: its errors would be taught as the label.
     - `sampled` lines still ask the head nothing: where an NSynth note sits
       in its cell after its shift is not in its label.
     - A recorded note's readings are gliding when its track spans half a
       cell over the note. Not taken: the engine's rule, a frame unlike its
       neighbour, which a tracker's jitter meets at every frame, so the steady
       row would be empty; and a frame moving faster than a rate, which would
       leave a scoop's settled frames steady, where the span calls the whole
       note gliding, but needs a rate and a smoothing set by hand. The rate is
       next if settled frames fill the gliding row on real singing.
     - Two strings holding one pitch at one onset, a unison, give both of the
       score's notes the first string's contour.
6. The decoder's and the parting's constants tuned against truths: `tune`
   searches `RipParams` over rendered loops whose truth is the render's; with
   `truth` files (rhythms, corrected transcripts, MIDI) on real recordings,
   the same search can take their agreement as its score, and the stream
   costs (`streamJump`, `streamLength`, `streamLevel`, `streamSwitch`), the
   lead rule's default and `cross` would be tuned on music rather than set by
   hand. The models' thresholds (`strike`, `strike_low`, `strike_again`,
   `release`, `voicing`) are searched by `tune ... truth <list>` through the
   loaded models by pitch and onset, as the mean over its list's conditions
   under a guard, the frames kept between runs ([Rip design
   §7](rip-design.md#7-plan)). Left of it: the real validation list,
   GuitarSet's player 04, is guitar alone, so it wants a condition a kind, of
   the collections and renders a fit may hold and never a full track,
   `timed` on GuitarSet's lines and `drumless` on every line no drum plays
   in (a `.mid` says nothing of drums without it, and the phantom hits on a
   guitar or a singer alone are then unseen), before a run after every round
   writes a `rip.toml`; a timed truth's drum hits still pass through
   `MatchHits`' own map; and kept frames outlive a change to the models'
   code, which their names do not hold. It scores the drums and the bars
   beside the notes where a truth gives them, a hit heard where a truth
   knows no drum plays against the drums, and then searches `onset_min` and
   the tempo's, the beats' and the bars' parameters (`drumGenes`, `barGenes`); bars come from
   a corpus label alone, so a recording's want its list's beats in a label
   (`corpus <dir> from`), and a `.jams` truth, beats and no notes, is not read
   yet. It has run on no generated truth of every meter: that run sets the
   bars' weights audio still holds at the values they were written with. Nor
   on loops cut off the bar line: some fit loops are to start a quarter to
   three quarters of a bar in, their beats and lines given as a recording's
   are, so the bars' weights do not learn a rendered loop's start ([agent
   notes](agent-notes.md#the-rip-and-its-truths)); the table reads bars on
   such loops beside the whole ones.
7. The pedal: a piano arpeggio under pedal rings three or four notes at once,
   and the analysis's two lines hold two, so the second line's eighths read as
   held notes. A model's strikes start every note struck however many ring;
   what the arrangement still lacks is a chord read per beat as the set of
   ringing pitches, rendered as a pad under the lines. The corpus plays no
   pedal: one that holds the notes, with labels that say so, is still to build.
8. The fit as it fits a real lead: it picks modal voices where the user hears
   a plucked string (a harp or a santur) or a flute. A plucked voice
   (`string`) with a doubled, beating course, and the fit's samples cleared of
   the other lines (item 4 of the loop below), measured by the floor the fit
   reaches.
9. The judge as a trained stage ([Rip design](rip-design.md) §7) is built and
   none is fitted on recordings or installed: `rows`, `tool/rip-train`'s
   `judge.py` and the root's `ripjudge.bin`
   ([a judge](../tool/rip-train/README.md#a-judge)). Left:
   - The fit: the CC BY collections that train, renders and the engine's
     mixes, never a set that only measures nor a full track, read with a
     collection left out, singing reported and never a gate; a bass line
     under drums when recordings are in the fit; and the row's next version,
     with the note's voice.
   - `bench` grades the loaded models under the installed scores whatever the root
     holds, as a round is benched before a judge is fitted on its weights.
     Once a judge is accepted it should grade what `rip` hears.
   - `rows` knows a recording by its audio's checksum, so a truth corrected
     after its rows were written is not read again: remove the recording's
     `.json`. The smallest close: the truth's checksum beside the rows.
   - A count from rows reads apart from `truth`'s decode (the trainer's README
     says why); audio's own open items are in
     [its todo](https://github.com/lixenwraith/audio/blob/main/doc/todo.md#close-what-a-judge-leaves-open).
10. A performance's character, generated: `corpus ... gen` draws one by hand,
    accents, swells and fades, a feel and a jitter a part and a tempo's drift
    ([Rip training §3](rip-training.md#arranged-loops)). Left: how loudness
    and timing move in the performances of a set that trains (GuitarSet's
    annotations against a steady grid), measured there and the draws set from
    it, judged as any corpus change, a pair. No round may learn it from a track.
11. A recorded note's velocity is measured from its recording under the
    `measured` word ([Rip training §3](rip-training.md#a-recordings-velocities)).
    Left:
    - The bent voices: the bank's piano and bass, whose hammer's filter and
      knock move with the velocity, and its pluck, whose burst's filter does,
      rise under a dB a dB of velocity, and a filter envelope lifts the pad
      and the lead a little over (`TestAVoicesLevelIsStraightInItsVelocityWhereItsToneIsNot`
      logs each). Their labels keep the velocity they were played at, while a
      measured one is the straight law's, the sampler's: one law across the
      corpus needs those voices' curves, or their tone's move taken out of
      their level.
    - The measure's limits: a detuned unison of saws drifts apart within a
      window and reads under; a ringing note's partial in opposite phase to
      the note's cancels where the high bound cannot see it; a vibrato wider
      than a bin moves a partial within its window. A note another note
      sounds with gets none: on NSynth's overlaps such notes held the bound
      fitted on the fit's instruments and missed it on the others and the
      test split, so a chord's notes and a let-ring line's go without until
      a measure holds there. A ring in doubt rings on (`heardRing`): a note
      sounded again with no gap, or under a ring an octave over it, gets
      none, as does a plucked note struck into a ring loud enough to move
      it; a ring's hidden partial is bounded by the told ones under it,
      which a partial dying slower than those under it would defeat.
    - The measure's cost on a guitar's chords and lines: for each note it
      measures, every earlier ring is walked window by window, and where
      pitches recur an old ring stays in doubt up to that note. GuitarSet's
      00_BN1-129-Eb comp took 39 s of CPU and 326 MB at its peak, giving none
      of its 133 notes a velocity; with its solo, 14m51s on half a core, 442 s
      of CPU, 470 MB and 3 of 204 given: some 44 hours for the set's 360. A
      recording with a by-string file is measured by its strings' ends instead
      ([Rip training §3](rip-training.md#a-recordings-strings)), which walk no
      ring.
    - A chord's notes on the microphone's scale. A string's channel holds each
      of its notes apart, but on GuitarSet's 7,713 notes alone the microphone's
      level over the channel's strays a median 0.5 dB and nine in ten within
      2.1 dB for the same note played again in a recording, every other string
      silent, and 1.6 and 5.1 dB over a string's keys; a ratio by player,
      string and frequency, partial by partial, read on recordings it was not
      fitted on, strays 1.5 and 4.9 dB, past `measuredBound`, so a note with
      others gets none. A measure on the microphone that fits a chord's notes
      with the strings' partials, ends and levels as their shape is the way left;
      the set's original, not debleeded, by-string archive would say whether the
      debleeding adds to the stray.
    - A placed start still lights the onset head's target and a placed end the
      release head's, where the alignment puts them (`placed` asks the strike
      head alone nothing); the same mask on those two heads is a line each.
    - E-GMD's list says `none`, so its hits are labelled at -1: a kit's law
      is not the engine's, and whether a hit's velocity is its file's under
      that law or measured from its sound is a decision still to make.

### Teach the frame model the beat and the bar line

- Priority: P1
- Affected files: `tool/rip-train` (`train.py`, README), `tune.go` (`corpus`),
  `doc/rip-training.md`, the weights in the root

The trainer's half is built on audio v0.24.0, which reads the two heads: the
heads, a rendered loop's targets from its label and a recording's from its
list's `beats` column, the mask, the loss with no weight, each head's F1 on the
lines ([Rip training §3, §5 and §8](rip-training.md#a-recordings-beats),
[rip-train](../tool/rip-train/README.md#the-beat-and-the-bar-line)). Since
v0.25.2 audio reads the heads by level, as a plain tracker does, the downbeat
head beside the cues. What is left is the reading on trained weights:

- **A round with the heads**, from the frame model trained again (below)
  through `--init`, on `meters` corpora and on recordings whose lists name
  their beats (GuitarSet's JAMS files, E-GMD's own MIDI). `--probe` first: it
  trains the two heads alone and cannot cost the heads the decoders read, and
  says whether that model's features hold the beat at all; a fine-tune only if
  they do not. Then GuitarSet's clips no round met, each with the heads and
  with the same weights' heads left out, and what [audio's
  todo](https://github.com/lixenwraith/audio/blob/main/doc/todo.md#hear-the-beat-and-the-bar-line-on-trained-weights)
  leaves to them, read by [the rule](rip-training.md#11-reading-a-candidate).
- **What a round must change**, in the order of what is left:
  - Which note is the beat: no decoder reads a level the head does not state,
    so the meter is to be heard before the beat is, and a model's reach is
    2.97 s either side, a bar or two.
  - A quiet downbeat head is no longer a round's to mend: v0.25.2 weighs its
    sum at a place against its own mean beat, which no scale moves. What the
    drumless loops still want is the first point, and recordings with their
    beats.
- Left open by what landed, each with the option not taken:
  - The two losses count once each in the total, beside ten heads' losses,
    and a bar line's mark is under 2 frames in a hundred.
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
    of a tempo it never met.
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
names the highest note alone, so never a lower line, and is taught `unknown`
on every engine voice far from the bank. Landed: the heads a cell, their
targets and their reading into a note's voice ([Rip training §5, §7 and
§8](rip-training.md#5-the-targets)), a family asked only of a note an instrument
played (one the engine rendered asks its wave and genes alone), and no name
where no family's share passes what a family held out reaches
(`--family-holdout`, the measure the weight file carries). What
remains: real notes labelled by the fit, an instrument of NSynth's at a time;
the frame model's planes as context to those heads, measured as a pair; and the
round, judged on NSynth's held-out instruments by each family's recall a note
and the rip's voice over the fit's distance, with E-GMD's kits unmoved. Then
lines by voice: notes gathered by their voices and their continuity, first
inside the three lines a rip holds, then as many as were heard, through the
song and the game.

Left open by what landed, each with the option not taken:

- The measure is one family's reach at one quantile (`SURE`, 0.95), and the
  file that ships takes it from another run of its recipe (`--sure`), since a
  run that teaches every name has no family left to measure with. It bounds
  how often a family never taught is named, a note in twenty, and not how
  often a name given is right. Not taken: each family held out in turn and the
  highest reach carried, a run a family; and the reach of notes outside the
  eight families on a validation corpus, NSynth's mallets and organs, which
  would measure the shipped run itself and wants those notes in
  `--validate`'s list first.
- A note the engine rendered is asked no family: named by the registered voice
  it was drawn under (`voiceFamily`), or `synth` far from the bank, one modal
  or string sound was asked two names. Not taken: an engine voice named by
  what it sounds nearest, its wave and genes against the families'.
- `Label.name` reads a note's genes as "the engine rendered it", which holds
  while nothing else gives a note genes. A fit's wave and genes written into a
  sampled note's label would cost that note its family: before that lands, a
  label says whether its voice was rendered or fitted.
- A cell two events sound in is asked nothing. Not taken: asking it where the
  two are one voice, a second rule; and asking it both names, which a softmax
  over the names cannot hold. The frame model's `family` is still asked the
  first listed name of a doubled top note ([Rip training
  §5](rip-training.md#5-the-targets)): it is to ask none there, as a cell
  does, a line in `Label.window` riding with a frame round.
- An embedding with linear readouts in place of the heads, whose readouts need
  a place in the weight file ([Rip training §7](rip-training.md#7-the-models)),
  is the way if audio must tell every pitch of a frame its voice, not twelve.
- The measure rides in the weight file as a frame head of no weights, and a
  measure a family would fit the same head at eight values. Not taken: named
  constants in a version of the file after 5, which changes transformer's
  format and loader; what brings them is a constant not the notes model's own
  or one a caller needs before any frame is heard ([audio's
  todo](https://github.com/lixenwraith/audio/blob/main/doc/todo.md#carry-a-notes-voice-further)).
- What plays a cell is taught at a quarter of the frame's like weights
  (`PLAYS`), since its heads teach the channels the decoders read ([Rip
  training §8](rip-training.md#8-a-training-step)). Not taken: the heads read
  with the gradient stopped at the cells, which costs the decoders nothing and
  teaches the heads nothing; channels of the cells kept for what plays, or a
  layer between the cells and those heads, which the weight file has no tensor
  for; and a smaller weight still. A real round reads the strikes against a
  run without the heads at equal steps before any of them.
- Generated loops teach the names little past register: they want recorded
  families (NSynth's) and a round's length. Not taken: weighing a name by its
  rarity, which moves the shares the measure reads.
- The lines count a note told on any frame: one never among a frame's twelve
  most salient pitches is in no share, so the count of notes moves a little
  with the notes model's salience from step to step.
- `test_train.py` runs by hand: CI has no PyTorch.

### Train both models again

- Priority: P1
- Affected files: `tool/rip-train` (the rounds' recipes), `CREDITS.md`

The frame and notes models train again from nothing, on what the [data
rule](../AGENTS.md#data) lets train: the engine's generated parts and mixes,
and E-GMD, VocalSet, NSynth and GuitarSet, while MusicNet and vocadito only
measure ([credits](../CREDITS.md)). The generator writes every key A0 to C8
([Rip training §3](rip-training.md#arranged-loops)). What closes it: both models trained from nothing and installed, their
rounds naming every collection they took, and CREDITS.md crediting exactly
those.

### Bring the rest of the lab in

- Priority: P2
- Affected files: `tool/rip-lab`

The tools that fetch a set, write its labels and its list and check them are in
[`tool/rip-lab`](../tool/rip-lab/README.md). What remains:

- The scripts that render a corpus in parts, run a round and read its rips
  predate the band frames: they go in once a round has run them.
- Only `audit/corpuscount.py` has a test: a smoke test each for the others.
- `datasets/fetch.sh egmd` reads `datasets/egmd/batches.tsv`, each file's byte
  range in the set's archive, which the tree does not hold: the batches fetch
  only beside the lab's copy of it.
- The free tracker (`labels/vocal.py`, `trackscore.py`, `trackfit.py`) labels
  nothing. It stays for a set with no sheet to place, VocalSet's excerpts
  first, and goes if none is labelled with it.

### Train the piano's pitches

- Priority: P1
- Affected files: `tool/rip-train` (a round's recipe)

The models are built for MIDI 21 to 108 on band frames whose low bands part
each key from the next (v0.34.0, audio v0.30.0), and every line trains from
nothing on them; a corpus's arranged loops play every key of the 88 ([Rip
training §3](rip-training.md#arranged-loops)). What is left:

- `truth` counts a note given outside the rip's pitches as only given, where
  `tune ... truth` and `rows` leave it out (`matchHeard`): a score read
  against its own file shows the note it dropped that way
  (`TestRipReadsAMIDIFileAsAScore`). Decide whether the verb follows, with a
  line saying how many it left out.

### A live rip

- Priority: P2
- Affected files: `cmd_doc.go`, vif's Android app

`live` drives the library's stream (`RipStream`) over a file in blocks and
measures it against the offline rip of the same recording, the measurement the
Android app's microphone mode (vif's todo) is judged by ([Rip design
§7](rip-design.md#7-plan)). What is left:
- Tempo, meter and phase are held from the first six bars, so the stream's
  bar grid is not the offline rip's: on twelve recordings through the models
  of 9 November, since deleted, it found 95 in a hundred of the offline rip's
  notes but read its grid on three (five at another tempo, a rubato piano take
  keeping 5 to 38 in a hundred of the offline downbeats, a song's intro half a
  bar off). Not measured: `pace`, and a phone.
- The wait. The ear waits the frame model's reach, 2.97 s, so the stream's
  frames are the offline rip's; audio's `FrontEnd.Ahead` waits less and no
  verb sets it. Half a second leaves the notes where they are and moves the
  drums ([Rip design §7](rip-design.md#7-plan)). A pitched note needs the
  notes model's 0.53 s alone: each model's frames at its own reach is audio's
  todo.
- The lines. The stream parts the melody register's two lines over the window
  it holds when a bar freezes (`streams`), the offline rip over the whole
  recording, so a note can land on the other line. A view that colours a line
  shows it.
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

### Show a live rip as it is made

- Priority: P2
- Affected files: `session.go` (`Options`), `tui.go`, `tui_panes.go`,
  `cmd_doc.go` (`live`)
- Guide: [Rip design §7](rip-design.md#7-plan)

vif's Android app is to rip a song as its microphone hears it, through stapes
run in-process ([vif's
todo](https://github.com/lixenwraith/vif/blob/main/doc/todo.md#rip-live-on-android)).
The library's stream and the reads a view takes from it are in place (`Bars`,
`Frames`, `Open`, `Strikes`, `Final`, [audio
§9](https://github.com/lixenwraith/audio/blob/main/doc/audio.md#9-ripping-a-recording)).
What remains here:

- The source: the session takes the host's microphone as it takes its speakers
  (`Options.Sink`), its blocks written into the stream off the audio thread.
- The view: a scrollable stack with the recording's spectrum on top, each stage
  under it (the planes the models are given, the pitches, strikes and drums
  they hear, the instrument, what the decoder keeps and drops, the open bars)
  and the note lines at the bottom; the frozen bars a few seconds behind the
  music, the open ones revised until they freeze. A file through `live` feeds
  the same view on a terminal, where it is checked without a phone.
- The end: a `Final` the player asks for, and the song saved into the search
  path's first folder as a rip of a file is.

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

7. A rip's voice from what was heard: `rip` fits every line, and the heard
   voice is its first seed. What remains is the voice of every note (the
   instrument model, above) and a distance that does not favour a dark voice.
8. Put off: a second piece of music under the first, which the rip
   transcribes with it; it wants the judge to follow instruments as it
   follows notes.

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

### Audition the binaural bed

- Priority: P3
- Affected files: `command.go` (`commands`), `cmd_audio.go`
- Prerequisite: the library's bed ([its
  todo](https://github.com/lixenwraith/audio/blob/main/doc/todo.md#give-the-mix-a-binaural-bed))

The library is to mix a binaural bed under the music (`SetBinaural`: a carrier,
a beat and a level), which vif offers as a setting ([its
todo](https://github.com/lixenwraith/vif/blob/main/doc/todo.md#add-binaural-beats-to-the-music)).
A verb that sets it over what the session plays lets a carrier, a beat and a
level be heard before vif's presets are tuned.

## Android

### Play the screen by touch

- Priority: P1
- Affected files: `tui.go`, `tui_panes.go`, `tui_editor.go`, `tui_keys.go`,
  `tui_files.go`, `doc/stapes.md`

vif's app is to run stapes in-process on Android, over a `terminal.Terminal` of
its view with its speaker as the sink, the system keyboard serving `:` and the
prompts ([vif's
todo](https://github.com/lixenwraith/vif/blob/main/doc/todo.md#bring-stapes-to-android-stapes-mobile)).
The host turns taps and drags into `terminal.Event`s, but the TUI reads keys
alone (`handle`). What remains here:

- Touch: a tap hit-tests what was drawn: a key of the keyboard plays its note, a
  step toggles, a tab or a pane takes the keys, a mixer row mutes its slot, a
  browser row selects. A strip of buttons stands for the keys a phone lacks
  (record, play, delete, save, new, edit, tempo, Tab, Esc).
- Size: the views need 60×16 cells. A phone held landscape fits at the default
  cell; held portrait it needs the host's pinch zoom out, or a keyboard of fewer
  keys.

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
- The lag's fit and readback, `TestARecordedKitsHitsAreMovedOntoTheirSound`,
  is weighed at 207 s, fifteen times its plain 13.8, the heaviest until the
  list is measured again: two thirds of it are eighteen kits' renders, each
  drum rendered three times as its eq, its shape and its binding are set.
- A push to a branch with a pull request open runs the workflow twice, for the
  push and for the pull request; `claude/*` in the push trigger is what tests
  a branch before its pull request opens.
- A job uses one core of four: tests run one at a time while a session
  registers in the audio module's process-wide registries.

## The repository

### Rewrite the history without the songs' names

- Priority: P1
- Affected files: every commit

v0.31.1 gave the test songs neutral labels in the docs and the code. Older
trees and three commit messages still name the songs, as do PR bodies on
GitHub. One `git filter-repo` pass with `--replace-text` and
`--replace-message`, from the mapping kept in the project's files outside the
repository since it spells the names, takes them out of every branch. What it
costs, and what it does not reach:

- Every hash changes: each clone re-clones, open branches are rebased by hand,
  and signed commits, each squash merge GitHub signed among them, lose their
  signatures.
- stapes has no tags, so no published version's checksum is at stake, but
  proxy.golang.org keeps any pseudo-version someone fetched as it was, names
  and all.
- GitHub keeps every PR's `refs/pull/<n>/head`, which no push deletes, so the
  old commits stay reachable there and by hash in cached views. GitHub Support
  removes only sensitive data that rotating a credential cannot make safe, so it
  will not purge them. PR titles and bodies are edited on GitHub by hand.

### Carry Go's notice with every binary

- Priority: P1
- Affected files: `Makefile` (`install`), a notices file; vif's app

A stapes binary links Go's standard library and golang.org/x/sys and x/term,
whose BSD 3-Clause licence (`$(go env GOROOT)/LICENSE`, the same in each
module's folder under `go env GOMODCACHE`) asks that a binary reproduce the Go
Authors' notice, conditions and disclaimer. The other modules in `go.mod`,
lixenwraith's audio, color, genetic, terminal, toml and transformer, carry
stapes' own `LICENSE` word for word, which `make install` installs; nothing
carries Go's. What closes it: a notices file with Go's licence, installed
beside `LICENSE` and put in every desktop archive, and the About screen of
vif's app, which hosts the Android build, listing both; each is checked
against `go list -deps ./...` when `go.mod` changes.
