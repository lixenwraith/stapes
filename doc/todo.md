# TODO

Where stapes may grow. Each item states what remains and the smallest thing that
would close it. Delete an item when it lands; this file is not a changelog.
Source files do not carry TODO comments.

Priorities: P1 is wanted next, P2 is convenient follow-up, and P3 is an idea.

The audio library's own items are in [its todo](https://github.com/lixenwraith/audio/blob/main/doc/todo.md);
the items here change stapes, and the library with it where they name its
files. Stapes imports nothing of vif, on [Android](#android) too: an item here
that meets one of [vif's](https://github.com/lixenwraith/vif/blob/main/doc/todo.md#audio)
links there.

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
   generative lead (left out since an offline render plays its first bar as
   the last render left it and labels every bar with its last: regenerate it
   at bar 0 and label it as played), and a few recordings the user plays from
   public-domain scores, with the scores, in the corpus through `corpus <dir>
   from` and held out, so neither a model nor a tuned set learns the
   synthesizer. URMP's aligned MIDI fits the same loader; MAESTRO,
   non-commercial, and a commercial song's transcription or a render of one
   stay out (AGENTS, Data).
2. The frame model's pitch: next for it, real recorded notes as voices of the
   corpus, from the sets the [data rule](../AGENTS.md#data) lets train; the
   constants tuned by `tune` against the truths rather than swept by hand.
   Towers are tried once the heads compete with the strike head. For a rip of
   live music: a bench of the loops through a phone mic, a room and crowd noise
   at 20 to 0 dB, the same chain in the corpus, and the live rip in the hands
   ([Rip live on Android](#rip-live-on-android)). Then: real recordings with
   their notes in the corpus (item 1: `corpus <dir> from` writes E-GMD's, its
   hits moved onto their sound or the recording left out and a hit that draws no
   sound left out of its label, `train.py --mix` mixes them in; still lacking a
   held-out score for a `from` corpus, E-GMD's ET split being the clean one,
   drum velocity in the labels, its list saying `none` till a kit's law is
   decided, the lag's and the silent hit's constants read back on E-GMD itself
   (by drummer and session, whether a performance's kits agree, `left.tsv`'s
   hits by key and kit; the fit's rise lies at its rounding edge, -3.51 steps, so
   a drum's norm moved alone by 2 dB moves it or the bound a step, and a kit's
   balance drawn as its level is moves the reach to the whole gap and `lagClear`
   to 1), a lag past half the usual gap between strikes, which reads as a
   nearer one, and a check that drops a render whose hits draw no onset; NSynth's notes play in `corpus ... sampled`, lacking the tempo-synced
   and multiphonic filters `examples.json` allows and a pipeline bench on the 53
   held-out instruments), then more loops, or width 384 (`DIM=384 HEADS=12
   FFN=1536`, 7 million weights), each measured by `bench` on clean and
   augmented loops and by the truths on the recordings, and shipped only when it
   beats the analysis on both; a hundredfold or more needs a GPU to train and
   would not run in the engine's budget. A vocal line is a line like any other
   to the rip; a voice has a vibrato and a vowel's formants now
   (`tool/rip-train/focus/vocal.toml` draws them), neither a gene, so the fit
   and the models' voice heads do not search them: genes for both once a corpus
   with them has trained, and a real vocal recording held out. The analysis also
   leans on a lone oscillator starting every note at phase 0: at random phases
   two lines on one instrument stop parting
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
   - A notes model's note heads hear 80 ms either way, in which a sung pitch
     that leaves its semitone and comes back and a new note that stays are one
     picture, the likely cause of singing's wrong strikes. `--init` grows the
     long convolutions, half a second, onto a pass's checkpoint ([Rip training
     §7](rip-training.md#7-the-models)). Left: that pass, read on VocalSet's
     held-out singers against the pass before it, the strikes' F1 at placed
     starts, the strikes over a long tone and the notes with none, and the
     guitar's reading held.
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
   `strike_alone`, `release`, `voicing`, `start` and its window) are searched
   by `tune ... truth <list>` through the loaded models by pitch and onset, as
   the mean over its list's conditions under a guard, the frames kept between
   runs ([Rip design §7](rip-design.md#7-plan)). The trainer's `READ` and
   `--validate`'s pick leave out the notes model's onset head, which audio
   reads beside the strikes since v0.39.0: the smallest close, `onset` in
   `READ[True]` and the placed starts' F1 beside the strikes' in the pick,
   once a round's starts are read on recordings. Left of it: the real
   validation list, GuitarSet's player 04, is guitar alone, so it wants a
   condition a kind, of the collections and renders a fit may hold and never
   a full track,
   `timed` on GuitarSet's lines and `drumless` on every line no drum plays
   in (a `.mid` says nothing of drums without it, and the phantom hits on a
   guitar or a singer alone are then unseen), before a run after every round
   writes a `rip.toml`; and kept frames outlive a change to the models'
   code, which their names do not hold. It scores the drums and the bars
   beside the notes where a truth gives them, a hit heard where a truth
   knows no drum plays against the drums, and then searches `onset_min` and
   the tempo's, the beats' and the bars' parameters (`drumGenes`, `barGenes`); bars come from
   a corpus label alone, so a recording's want its list's beats in a label
   (`corpus <dir> from`), and a `.jams` truth, beats and no notes, is not read
   yet. It has run on no generated truth of every meter: that run sets the
   bars' weights audio still holds at the values they were written with. Its
   bars score is the mean of the downbeats on a line, given only where the bar
   is the truth's length, and the beats on a beat (`MatchBars`), so the tempo's
   prior, combs and cuts move it on truths of every meter, the guitar's where
   they are the sheets' labels and not their MIDI files. A beat finer than the
   truth's falls on all its beats: a 3/4 read over its eighths as 6/8 scores in
   full, its meter's name alone wrong. Nor
   on loops cut off the bar line: some fit loops are to start a quarter to
   three quarters of a bar in, their beats and lines given as a recording's
   are, so the bars' weights do not learn a rendered loop's start ([agent
   notes](agent-notes.md#the-rip-and-its-truths)); the table reads bars on
   such loops beside the whole ones. What a label says past pitch and onset,
   its notes' ends and loudness, its hits' loudness, its parts, key and chords,
   `truth` reads and `tune ... truth` says by condition, weighing it only under
   `weigh`; `release`, which nothing else sees, is searched so alone. Left:
   GuitarSet's JAMS files hold each take's key (`key_mode`) and its chords,
   instructed and performed, which its labels do not carry; a `.mid` or a
   transcript gives no reading, its velocities and lengths its writer's;
   E-GMD's hits state no velocity (its list's `none`); and a `release` a
   weighed search writes is read on recordings before it ships.
7. The pedal: a piano arpeggio under pedal rings three or four notes at once,
   and the analysis's two lines hold two, so the second line's eighths read as
   held notes. A model's strikes start every note struck however many ring;
   what the arrangement still lacks is a chord read per beat as the set of
   ringing pitches, rendered as a pad under the lines. The corpus plays no
   pedal: one that holds the notes, with labels that say so, is still to build.
8. The fit as it fits a real lead: it picks modal voices where the user hears
   a plucked string (a harp or a santur) or a flute. A plucked voice
   (`string`) with a doubled, beating course. The measure now hears each
   window as the mix of the line's notes, the other lines' and the drums' hits,
   each at a gain of its own (`mixDistance`), read on the labels' own notes
   with `fits ... given`. Left: the reading through the trained weights on
   ripped loops they did not train on, which alone fills `heard`; a take for
   `RenderNoteAt`, so a unison stack's or a string's measure averages the
   phases and bursts its notes are seeded with ([agent
   notes](agent-notes.md#the-rip-and-its-truths)); and onsets off the frame
   grid.
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
   - `rows`, `truth` and `tune ... truth` count a note within audio's
     `matchWindow`, 0.1 s, which knows no reach: on the frame grid it holds 8
     frames, the reach measured today, so a placed truth's right strike counts
     wrong only past it, where the lab's check draws hold 3% of starts; a reach
     measured over 8 counts more of singing's right strikes wrong.
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
    - A placed end still lights the release head's target where the alignment
      puts it, and a placed note's velocity is asked about its placed start;
      its onset and its strike ask a rise within its reach. On the
      engine-shapes takes placed as a label is, the release head learns the
      alignment's miss if its peaks agree better with the placed ends than
      with the known ones, the velocity head if it reads a note's velocity
      nearer at the placed start than at the known one, where the strike
      rises and the decoder reads it, each more so pass on pass; then an end's
      own reach, measured as a start's is, asks the release one rise within
      it, and a velocity is asked where the strike rises.
    - E-GMD's list says `none`, so its hits are labelled at -1: a kit's law
      is not the engine's. An empty cell would label each hit at its file's
      n/127, the engine's kits' measure; whether the lab's list says so waits
      on the trainer's reading of whether a kit's stated strengths follow the
      levels its hits sound at, and where they do not a hit's strength
      measured from its sound is the way left. No head asks a hit's velocity.
    - A hit that draws no sound is told by its own rise alone: a silent hit
      within two frames of a sounding one reads that one's and stays, and a
      room's tails lift silence's rise, so through `degrade` 15% of lone silent
      hits stay. `left.tsv` keeps each hit left out, with the rise it read,
      for a measure that tells them apart.

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
  - Every beat alike. The decoder reads the head's heights (audio's `finest`),
    and the first round's head drops them where the decoder needs them: on a
    solo line with no chord under it, its marks fall on a fifth to a third of
    the beats, at two thirds or four thirds of their spacing; on the guitar
    it marks strong beats far above weak ones; in 6/8 and 12/8 the eighths
    between the dotted quarters stand at a fifth to a third of them. Its F1
    counts a mark right whatever its height, so it hides all three. A round
    asks the same marks with more of what lacks them, loops of a line alone
    in every meter and GuitarSet's solos at the comps' share, and is read by
    the head's mean at each beat of the bar on the held-out loops beside its F1.
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
- The names' prior a line reads against is the held-out loops' shares. A corpus
  mixed in (`--extra`, `--mix`) moves the shares a head is taught and not that
  prior. Not taken: the shares of the batches drawn, a count the checkpoint
  would carry.
- The lines count a note told on any frame: one never among a frame's twelve
  most salient pitches is in no share, so the count of notes moves a little
  with the notes model's salience from step to step.
- `test_train.py` runs by hand: CI has no PyTorch.

### Hear what is struck

- Priority: P1
- Affected files: `kit.go`, `tool/rip-train`, `tool/rip-lab/audit`, `lixenwraith/audio`
  (`ripmodels`, the decoder)

A hit is to be described by what it sounds like, not named. Landed: `corpus ... gen
kits` plays an arranged loop's drums on a kit of two to ten struck sounds drawn over
audio's struck genome, each strike labelled with its piece and the loop with each
piece's genes, band and place ([Rip training §3](rip-training.md#a-loops-kit)); the
places, three even runs of the bands by a count ([§3](rip-training.md#a-kits-places));
a recording's hits labelled by piece, a piece each key, placed by its own strikes
([§3](rip-training.md#a-recordings-kit)); a struck head on both models, asked at every
frame of every corpus; and on the frame model, at each place, whether something is
struck there, its genes with the error the head states for each and how hard, grown on
the weights that stand, a recording's genes taught by its pieces' strikes told alike
and apart, and read by audio into the frames ([Rip training §5](rip-training.md#5-the-targets),
[§8](rip-training.md#8-a-training-step)). What remains, each read before the next:

- The count of 200 loops with the word, by `corpuscount.py` and `corpusheard.py`,
  and a listening sample of kits of two, five and ten pieces; E-GMD's labels made
  again by piece (`corpus ... from ... labels`) and counted by key.
- A pass that trains the places' heads on the kits' corpus and E-GMD, read by its
  lines: the places' F1, the genes' miss beside the error stated, the loudness' miss,
  and one piece's strikes nearer than two pieces'.
- The decoder making hits from those heads, and `tune` reading strikes so, the struck
  head's peaks among them; then the grouping of a track's hits into its kit. Until a
  decoder reads them, the heads join neither `DECODERS` nor `READ`: in `DECODERS` their
  loss would pick the step a run exports, in `READ` a pass's rate would follow it.
- Scores that count hits without names.

Left open by what landed:

- A piece's level is drawn and not labelled: a strike says its velocity, not where
  its piece sits in the mix.
- A strike its loop hides is labelled all the same. `corpusheard.py` names the
  pieces never heard; whether such a strike leaves its label, as a recording's
  silent hit does (`left.tsv`), waits on the count.
- A piece costs 0.4 s of a loop's render, most of it the engine rendering all 18
  of its takes, whichever the loop plays.
- A sound's takes are seeded by its name, so the pieces numbered alike share their
  noise and their takes' spread from loop to loop: a seed of the loop's wants audio
  to take one.
- A place is a piece's, not a strike's: a third of the generated strikes alone sound
  most at another place than their piece's, a soft layer lower. A place by each
  strike's own take wants the engine to say which take a strike plays.
- A recorded piece is placed only where its strikes agree (`pieceSure`): on the
  genome's kits that keeps a quarter of the strikes read, 0.94 of them right, and on
  E-GMD's draw 0.553 of its strikes, 662 of its 2,383 pieces; the rest ask no place.
  More would be kept by reading the takes of one kit together, a list's column naming
  a take's kit as `unit` names a velocity's recordings, each piece's lone strikes then
  counted over every take; or by a head that keeps a frame's strikes apart by what
  they sound like, a set of slots matched to the labelled strikes by their
  description, for which no place is measured at all.
- `datacheck.py` reads no lag of a drawn kit's strikes, and the `--over` drums are
  laid only where nothing is struck.
- The loudness of a recorded strike is its MIDI velocity only where its list's
  `velocity` cell is empty: E-GMD's says `none`, so its lines say no loudness until the
  lab sets it ([Rip training §3](rip-training.md#a-recordings-velocities)).

### Read each head off the label on trained weights

- Priority: P1
- Affected files: none: readings

`heads <head>` reads one head off the label and the rest through the loaded pair
([Rip training §11](rip-training.md#the-decoder-alone)), where `heads all` reads
the decoders alone. Left: on the installed pair, one head at a time, on
generated loops at a seed no round drew and on the recorded sets' held-out
lists, each beside the pair alone and `heads all`, by its readings and the
stages `tune ... truth` prints: `strike` first, which holds the notes' F1 the
decoders give, then `pitch` for the ends, the key and the chords, `start` and
`cents` on singing, `release` for the ends; `velocity` waits on `dynamics`
([audio's todo](https://github.com/lixenwraith/audio/blob/main/doc/todo.md#keep-what-perfect-heads-say)).

### Train both models again

- Priority: P1
- Affected files: `tool/rip-train` (the rounds' recipes), `CREDITS.md`

The frame and notes models train again from nothing, on what the [data
rule](../AGENTS.md#data) lets train: the engine's generated parts and mixes,
and E-GMD, VocalSet, NSynth and GuitarSet, while MusicNet and vocadito only
measure ([credits](../CREDITS.md)). The generator writes every key A0 to C8
([Rip training §3](rip-training.md#arranged-loops)). What closes it: both models trained from nothing and installed, their
rounds naming every collection they took, and CREDITS.md crediting exactly
those. Before the round: under `aug` a loop meets `degrade` at the level the
bank's mix renders it at and is laid at its drawn level only after, and the room
soft-clips every take, at drive one `tanh(y)/tanh(1)` taking 2.4 dB off a
full-scale peak where its comment says a third clip, so the bank's levels move
the corpus's bytes; decide whether a loop meets its room at a drawn level, as the
lag's fit does.

### Carry a pass's rate further

- Priority: P2
- Affected files: `tool/rip-train/train.py` (`Rate`)

A pass that carries on follows its readings by a rule whose gaps wait on what the
lab's chain shows ([Rip training §8](rip-training.md#a-pass-that-carries-on)):

- The rule is read on drawn readings alone (`Replay`), no chain's checkpoints
  through it yet. Read a chain's fine rate and what each raise did, pass by pass,
  first.
- A raise's height is `--lr` whatever its raises did: a model that rate hurts is
  raised, undone and raised again at each stall, its fine rate an octave down
  each time. Not taken: a height carried beside the fine rate, halved where a
  raise is undone.
- A gain kept at a fine rate too small to move a weight is still one: it holds
  off the stall and the wide pass, and the fine rate rises only where a raise
  pays.
- A raise kept without a gain leaves the fine rate where it was, so a model at
  its rate's floor is not taken down by the raises that do not pay. Not taken:
  the fall kept where a stall's readings showed no gain at all.
- A stall's count is read once, at its third reading; the readings after it move
  no count.
- A frame model that gives the notes itself, no notes model beside it, carries
  on under the pair's sum, its pitched heads unread: the smallest close is a
  word that says it stands alone.
- A head at its floor, nothing left to learn, still stalls and is raised, which
  costs its rows a raise and a fall back, as the velocity head's over the tests'
  notes. Its floor could be read from its noise against its loss.
- A raised head moves its rows of the layer that outputs it alone; what it shares
  with the other heads moves at fine.
- `better` reads each head of the sum at its own noise: past it by chance alone,
  one of three or four refuses up to one good raise in eleven, and a head at its
  floor drifts past a noise near zero.
- A pass that carries on, resumed, has no test.

### Bring the rest of the lab in

- Priority: P2
- Affected files: `tool/rip-lab`

The tools that fetch a set, write its labels and its list and check them are in
[`tool/rip-lab`](../tool/rip-lab/README.md). What remains:

- The scripts that run a round and read its rips predate the band frames: they
  go in once a round has run them. A corpus renders in parts by `pipeline.sh`.
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
- Affected files: `cmd_doc.go`

`live` drives the library's stream (`RipStream`) over a file in blocks and
measures it against the offline rip of the same recording, the measurement a
[live rip on Android](#rip-live-on-android) is judged by ([Rip design
§7](rip-design.md#7-plan)). What is left:
- Tempo, meter and phase are held from the first six bars, so the stream's
  bar grid is not the offline rip's. Not measured since the refresh: how many
  of the offline rip's notes and bar lines the stream finds, `pace`, and a
  phone.
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

Stapes is to rip a song as a microphone hears it, a phone's first ([Rip live
on Android](#rip-live-on-android)). The library's stream and the reads a view
takes from it are in place (`Bars`, `Frames`, `Open`, `Strikes`, `Final`, [audio
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
   steps, judged by `Fidelity` against the recording's own frames; what it
   does not yet move is the drum part. Measure its convergence on rendered
   loops whose truth is known, and whether the notes it inserts are the
   truth's: `Fidelity` puts no price on a wrong note ([rip
   design](rip-design.md) §7).
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
   instrument model, above) and a distance that does not part a unison voice
   from its own note by where its oscillators start ([tune the
   rip](#tune-the-rip), item 8).
8. Put off: a second piece of music under the first, which the rip
   transcribes with it; it wants the judge to follow instruments as it
   follows notes.
9. The fit's memo names a genome to three decimals (`genomeKey`): two new
   candidates of one generation within it take whichever was scored first,
   which the cores decide. Key it by the genome's bits and read `fits` before
   and after.
10. The yardstick of a line two voices share: `fits` counts it right when the
    wave fitted is that of the label's voice that plays most of its notes
    (`lineOwner`), but where the other voice's notes are the long and loud
    ones the sound names a voice that plays them, and no one voice plays
    both ([agent notes](agent-notes.md#the-rip-and-its-truths)). Whether the
    line's voice is the one that plays most of its notes or the one its sound
    is nearest is open: listening settles it, the line rendered on each and
    the nearer picked (item 6), as does the song's score against the
    recording (`song ... score`) on each. Before either, `cells` is read on
    trained models over whole corpora, where the count decides; on 49 loops of
    each corpus here (`fits ... heads all cells 50`) it named 6 of 16 shared
    strings where the measure names none, and 7 of 26 shared saws where it
    names 15 (aug; without `aug` 6 of 14 and 15 of 26 against 4 and 14), at
    half its CPU a loop.

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

## Android

### Build stapes mobile

- Priority: P1
- Affected files: a new `android/` module (`go.mod`, the bound package, the
  Java app, its Gradle build), `go.mod`, `Makefile`, `doc/stapes.md`

Stapes mobile is an Android app of stapes' own: the package run in-process
over the terminal module's Android view, as vif's app runs the game, and
nothing of vif imported. `NewSession` takes the speaker as `Options.Sink` and
`RunTUI` any `terminal.Terminal`, the host posting `EventClosed` to end it.
What remains:

1. The view: `go.mod` takes `lixenwraith/terminal` v0.7.4, the release vif's
   app builds on, for `android.View` and `terminal.NewWithHost`; the module
   builds and its suite passes against it unchanged.
2. The bound entry, a module of its own as vif's `android/` is, so `go build
   ./...` never sees gomobile: a `Start` that points `HOME` and
   `XDG_CONFIG_HOME` into the app's files directory, where `ResolveFiles` finds
   the root and the search path, opens the session on the speaker and runs
   `RunTUI` over `NewWithHost` of the view; and the view's calls from the UI
   thread and the drawer's (size, taps, drags, keys, text, zoom, the keyboard's
   cover, the frames and the strike's texture).
3. The Java side, which the terminal module leaves to the application: copied
   from vif's app, the activity, the view with its taps, keys and pinch, the
   cell drawer, the speaker on an `AudioTrack` and the Gradle build with its
   toolchain check.
4. The system keyboard serves `:` and the prompts, raised from the touch
   strip ([below](#play-the-screen-by-touch)).
5. Files: the root and the search path in the app's own storage, which the
   verbs' paths reach; a `.mid`, a `.wav` or a weight file another app holds
   comes through the system's picker, copied into the search path's first
   folder or, a `.bin`, the root, and a written file goes back out through it
   ([Stapes §2](stapes.md#paths-on-every-host)). vif's app keeps its own
   storage, so a document reaches the phone's game only as a file handed over
   until vif's app bundles stapes, long after stapes mobile works ([vif's
   todo](https://github.com/lixenwraith/vif/blob/main/doc/todo.md#audio)).

What closes it: the APK opens the TUI with sound, a pattern edited is saved and
played again, and a picked `.wav` is ripped. Not known until it is built: how
the TUI's ticker and long verbs fare when Android pauses or ends the process,
the APK's size with the engine, the banks and the transformer, and whether a
phone's cores carry a rip through the models.

### Play the screen by touch

- Priority: P1
- Affected files: `tui.go`, `tui_panes.go`, `tui_editor.go`, `tui_keys.go`,
  `tui_files.go`, `doc/stapes.md`

On [stapes mobile](#build-stapes-mobile) the view turns taps and drags into
`terminal.Event`s, but the TUI reads keys alone (`handle`). What remains here:

- Touch: a tap hit-tests what was drawn: a key of the keyboard plays its note, a
  step toggles, a tab or a pane takes the keys, a mixer row mutes its slot, a
  browser row selects. A strip of buttons stands for the keys a phone lacks
  (record, play, delete, save, new, edit, tempo, Tab, Esc, the keyboard).
- Size: the views need 60×16 cells. A phone held landscape fits at the default
  cell; held portrait it needs the pinch zoom out, or a keyboard of fewer keys.

### Rip live on Android

- Priority: P2
- Affected files: the bound package and the Java app under `android/`
- Prerequisite: [Build stapes mobile](#build-stapes-mobile), and the session's
  source and view ([Show a live rip as it is
  made](#show-a-live-rip-as-it-is-made))

The app reads the microphone (`AudioRecord`, behind the `RECORD_AUDIO`
permission asked at first use) and hands its blocks to the session as its
source, as the speaker is its sink. What closes it: a song ripped as the phone
hears it, shown as it is made and saved, measured as `live` measures a file
([A live rip](#a-live-rip)). Not known until it is built: how far behind the
sound a phone's rip runs, beside the ear's 2.97 s wait.

## The tests

### Take the race suite's time down

- Priority: P2
- Affected files: `stapes_test.go`, `.github/workflows/test.yml`,
  `.github/race-seconds.txt`

The suite takes sixteen minutes under the race detector on four cores, up to
twenty-five on a CI runner, which the workflow splits over seven jobs, each
about as long as the heaviest test
([agent notes](agent-notes.md#the-race-suite-and-ci)). What is left:

- Beside the lag's and the silent hit's fits below, the heaviest set the
  shortest a job can be: the judge 269 s, the fitted and refined song 129 s,
  `live` 97 s. Each runs what its rule takes at the least. The refinement's
  every judge renders and hears the whole song; the judge test's time is the
  rip's accounting (`RipFrames.account`), which only audio shortens, a third of
  it plain the walk for a loop the judge and the installed scores rule apart
  (`truthFixture`'s `holds`); `live` hears twelve seconds through two ears, each
  a stream and two rips.
- Two splits were weighed and not taken, since each takes tests from the
  detector. Racing the light tests alone and running every test plain: the 27
  tests that render or hear audio for half a second skip under `-short`, and
  the raced run is 113 s on four cores with the lightest test of each
  goroutine stapes starts kept in it (the fit's and `tune`'s searches,
  `corpus from`'s workers, the screen's long verbs), but a race only a heavy
  test reaches goes unseen. Or that on a pull request and the whole raced
  suite on main, which shows such a race after its merge.
- The lag's fit and readback, `TestARecordedKitsHitsAreMovedOntoTheirSound`,
  takes 212 s raced, fourteen times its plain 15.1: two thirds of it are
  eighteen kits' renders, each drum rendered three times as its eq, its shape
  and its binding are set. The silent hit's,
  `TestAHitThatDrawsNoSoundIsToldByItsRise`, at 384 s, fifteen times its plain
  25.6, is the heaviest until the list is measured again: 36 kits' renders, and
  the band frames of their 48 takes, a quarter second each. A fit on 12
  recordings in place of 24 reads 2.5e-3, a silent hit's rise having a long tail.
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
- Affected files: `Makefile` (`install`), a notices file, stapes mobile

A stapes binary links Go's standard library and golang.org/x/sys and x/term,
whose BSD 3-Clause licence (`$(go env GOROOT)/LICENSE`, the same in each
module's folder under `go env GOMODCACHE`) asks that a binary reproduce the Go
Authors' notice, conditions and disclaimer. The other modules in `go.mod`,
lixenwraith's audio, color, genetic, terminal, toml and transformer, carry
stapes' own `LICENSE` word for word, which `make install` installs; nothing
carries Go's. What closes it: a notices file with Go's licence, installed
beside `LICENSE`, put in every desktop archive and packaged in [stapes
mobile](#build-stapes-mobile) with it; each is checked against `go list -deps
./...` when `go.mod` changes.
