# Rip model: corpora and rounds

Every corpus and training round so far, what each was for and what came of it. How
the data and the model work byte by byte is in `rip-training.md`, the reasoning in
`rip-design.md`. A score here is pooled F1 of notes over the test recordings: clean, then
from phones. Weights of deleted corpora are in `~/ripcorpus-archive`.

## Corpora

**Corpus 6** (deleted)
- 80 mel bands, loops rendered from the engine's bank.
- Goal: the first trained front end.

**Corpus 7a** (deleted)
- 40,000 loops at 229 bands, with better generated parts.
- Goal: finer input than 80 bands.

**Corpus 7b** (deleted)
- 7a plus a pedal, swell, equaliser, compressor, halls and noise, some loops without drums or bass.
- Goal: sound nearer a real mix. Served rounds 7b to 8b.

**Corpus 10** (deleted)
- 40,000 loops with note ends and velocities labelled.
- Goal: train the release and velocity heads.

**Corpus 11** (`~/ripcorpus11`, in use)
- 40,000 loops: new voices and kit, soft phrases, driven voices, a third through a codec.
- Goal: cover soft notes and lossy audio. The base of every round since.

**Corpus 12** (deleted)
- Corpus 11's loops with half the tonal lines played on NSynth's recorded notes.
- Goal: real instrument sound. Hasn't helped on instruments it never heard; its one more test, 32b, moved nothing beyond noise. No note records its NSynth family, so an instrument-labelled corpus needs writing anew.

**Corpus 13** (deleted)
- Hard links: corpus 11 plus the long-lines set. Used only by 17a.

**Corpus 14** (`~/ripcorpus14`, in use)
- 20,000 loops; three in four play bars of Lakh's clean MIDI on the engine's voices.
- Goal: real arrangements in place of generated ones. First test is round 24a.

**Corpus 15** (`~/ripcorpus15`, in use)
- 20,000 loops of whole Lakh arrangements: every part of the file, the parts past melody, second line and bass spread by octave over up to three tracks on instruments of their own, up to 24 dB softer (`corpus ... dense seed 151`).
- Goal: soft inner lines under a ring, as real arrangements have. The base of every round since 31a.

**Corpus 16** (deleted 9 November)
- Corpus 15's recipe at seed 161, notes as played. A first try read the sustain pedal into the labels, mixing them with the augmentation's pedal labelled as played; deleted unused.
- Goal: more of what lifted 31a. Seven rounds on it (33a to 34d) added nothing over corpus 15: a new seed of the same recipe is not new data.

**E-GMD** (`~/ripcorpus-egmd`, in use)
- 5,691 recordings of a drummer on real kits, its train split.
- Labels moved on 132 recordings that sat a steady 46 to 116 ms off; 15 worse ones skipped.
- Goal: real drums, laid over music or trained on alone.

**Random kits** (deleted)
- 4,000 drum loops, each on a kit of its own.
- Goal: many kits, since E-GMD's are all one drum module.
- Flaw: hats went too dark, which likely caused 17a's false hats.

**Random kits 2** (`~/ripcorpus-kits2`, waiting)
- The same with hats kept bright. Not trained on yet.

**Long lines** (`~/ripcorpus-long`, waiting)
- 6,000 loops of long sung lines, long bass, soft notes, sparse beats.
- Goal: long notes, which generated parts lacked. It lengthened 17a's notes.

**Over** (deleted)
- Hard links: E-GMD plus random kits. Used only by 17a.

**Phone loops** (`~/ripcorpus-phone`, waiting)
- 267 labelled loops played on a laptop speaker and recorded by two phones.
- Goal: real speaker and room. Its takes are also test takes, so it taints those scores.

**GuitarSet** (`~/ripcorpus-guitarset`, in use)
- 300 recordings of a real acoustic guitar; a sixth player held out as a test.
- Goal: real instrument recordings. No velocities.

**MusicNet** (`~/ripcorpus-musicnet`, in use)
- 320 real classical recordings, 33.7 hours, scores aligned with about 4% label errors; 10 more held out as a test.
- Goal: real instruments in full recordings, piano most of all. No velocities.

**MusicNet, snapped** (`~/ripcorpus-musicnet-snap`, `~/ripcorpus-val-mn-snap`, waiting)
- Each note's onset moved to 32a's strongest strike peak at its pitch within eight frames, where that is over a half (81% of the training notes, a median three frames), else marked `"snap": false` (19%: Solo Flute 6% to String Sextet 50%). Two models outside 32a's line then find a note within a frame of its label on 80 to 88%, 36% before.
- Goal: strikes MusicNet can teach and judge. The trainer's mask for an unsnapped note is still to come.

**The glide pair** (`~/ripcorpus37-plain`, `~/ripcorpus37-glide`, waiting)
- Corpus 17's line at seed 371, 2,000 loops each, without and with `glide`: the same loops but for the motion (765 of the glide corpus's loops move, 24% of their placed note frames), the last 200 of each held out of every run.
- Goal: rounds 37a to 37d.

**Validation** (`~/ripcorpus-val37`, in use)
- 969 loops of about ten thousand notes each of guitar (GuitarSet's player 04), piano and an ensemble (snapped MusicNet validation pieces), voice (VocalSet's five test singers) and phone (the engine's loops recorded by two phones at 2 m); no round trained on any of them.
- Goal: `--validate` with one condition a kind.

## Rounds

**6**
- 80 bands. Heard drums better and melodies worse than the analysis path.

**7a**
- 229 bands. Better on every head.

**7b**
- 7a's model on the harder corpus. The best model then on real recordings.

**7c**
- Three convolutions before attention. A large gain; kept since.

**7d**
- Separate attention towers for drums and notes. No gain, dropped.

**8b**
- Added the strike head and a true-log input. First to beat the analysis path on notes (.676, .518).
- Installed for a while. The first real thresholds came from it.

**10a**
- Added release and velocity heads. Fewer notes right than 8b (.650), not installed.

**11a**
- Corpus 11, for soft notes. Level with 8b (.676, .522).

**12a** (installed)
- PCEN as a second channel, and a simulated phone channel in training.
- Goal: hear through a phone. Large gain (.704, .620 to .645). Best drums from phones.

**12b**
- 12a without PCEN (.687, .583), which measured what PCEN is worth.

**14a**
- Corpus 12 plus E-GMD's real drums. Real kits much better, notes lower, many false drums.
- Cause: drums were laid too quietly and unheard hits were labelled.

**15a**
- 14a's idea on corpus 11. Stopped when that flaw was found.

**16a**
- 14a with drums at a mix's level, and only audible hits labelled. Fixed most of it (.668, .589 to .627).

**16b**
- The same on corpus 11. 12a's notes with real-kit drums (.702, .616 to .624), but drums weak from phones.

**17a**
- Long lines, random kits, darkened mixes, phone loops.
- Goal: dark mixes and long notes. First to hear a dark beat; longer notes; lower on clean (.665) and more false hats.

**18a**
- 16b trained further at half the rate. Loss went flat and scores held, so more steps don't pay. Stopped early.

**19a**
- 16b with harmonic stacking fed into the same model. No gain (.693, .604 to .638).

**20a**
- Shuffled windows, chunks of two clips summed, on 16b's recipe. Coded, not run, waiting for the winning model.

**22a**
- PitchNet: convolutions only, the same weights at every pitch, on the harmonic stack, 170,000 weights.
- Goal: test the design Basic Pitch and others use, on our data.
- Best notes by far (.774, .681 to .755), better on unseen instruments and effects. Weak under E4; drums weak from phones.

**23a**
- 22a's pitch path with attention over it in one model, 3.4M weights.
- Goal: keep those notes, and get drums and context from attention.
- Lower than 22a on every take and render (.751, .678 to .735), though its held-out loss was lower. Drums weak from phones too. Dropped: notes get their own model.

**24a**
- PitchNet on corpus 14. Goal: do real arrangements help?
- Yes on clean: best so far (.807, .661 to .744). Slightly lower than 22a on phones.

**24b**
- PitchNet on corpus 11 with MusicNet and GuitarSet mixed in. Goal: do real recordings help?
- No at this mix: lower than 22a nearly everywhere (.764, .653 to .697).

**25a** (installed as `ripnotes.bin`)
- PitchNet on corpus 14 with corpus 11 at half the windows. Goal: real arrangements without losing phones.
- Best balance: .813 clean, .677 to .763 on phones, on the strike decoder below. 24a stays ahead on rendered instruments.

**26a**
- 25a's mix with half the windows shuffled, chunks of a second, two clips summed.
- Goal: unknown sounds. Best yet on clean and rendered instruments (.829; ahead of Basic Pitch on 5 of 7 renders), phones a little under 25a (.658 to .747).

**26b**
- 25a's mix with NSynth's real notes at a fifth of the windows. No gain (.804, .672 to .756). NSynth stays out.

**26c**
- 25a's mix with the engine's long sung lines at 15%, a first test for singing. Stopped at step 9,700 for the real singing corpus.

**28a**
- 25a's recipe with a strike weighted 4 instead of 20 against a frame without one.
- Goal: fewer sure extra notes. At 20 the head over-reports: a .9 output is about .31 calibrated.
- Lost. At matching bars it finds fewer right notes for the same extras on every take (.806, .661 to .760 against .819, .681 to .767). Being over-sure wasn't the cause; the weight stays 20.

**29a**
- 25a's recipe with four more convolutions over the cells, dilated 3, 9, 27 and 81 frames: each note head sees ±1.5 s instead of ±80 ms. 207,000 weights, a new weight file version.
- Goal: a strike knows what is already ringing. Held-out loss much lower (.863 against .947), real takes mixed: through the accounting stage better on Pixel (.808 against .792 on Für Elise), the hardest guitar (.833 against .817) and singing (.551 against .500, octave errors 80 to 23), worse on clean (.838 against .857) and clean renders by about .01.

**27a**
- 25a's mix with real singing from `~/ripcorpus-vocal` at 5% of the windows, half shuffled, sung windows unmasked for voicing.
- Goal: hear a sung line. vocadito .539 to .634; best on effects (distorted guitar .825 to .888); phones level.

**29b** (installed as `ripnotes.bin`, 7 to 8 November)
- 29a with its longest layer gone: +-0.5 s, 197,606 weights.
- Goal: keep the reach's gains and clean's notes. Best overall on the wide test: Pixel .724, solo guitar .812, vocadito .616 with octave errors 73 to 12, phones' low extras halved. Billie Jean's bass the one loss, F#1 for F#2 on kick hits.

**30a**
- 27a's recipe with a body's ring at every attack, unlabelled, 70 to 400 Hz at -15 to 0 dB of the attack's rise.
- Goal: stop reporting a guitar's body ring as a note. Low extras on solo guitar 661 to 463, everything else lower than 27a. Too loud and too wide; narrowed in the trainer.

**31a**
- Corpus 15 with 14, 11 and singing, from 29b's weights, shuffle, singing, +-0.5 s, swaying tempo, restrike weight 3.
- Goal: every new thing soaked together. The day's biggest jump: renders .931, comp .581, MusicNet .608, vocadito .665 with 4 octave errors.

**32a** (Sheet 15's model, installed as `ripnotes.bin` 8 November)
- 31a with the narrower ring (70 to 160 Hz, -25 to -8 dB, a quarter of the windows).
- Better than 31a on every real set but the renders: clean .832, Pixel .700, Galaxy .701, comp .582, solo .821 (low extras 424 to 362), MusicNet .610, vocadito .682.

**32b**
- 32a with NSynth's recorded notes (corpus 12) at a quarter of the windows.
- Corpus 12's one more test: within .008 of 32a everywhere, up on Galaxy and MusicNet, down on solo guitar.

**32c, 32d**
- From 32a on corpus 15 in 15,000 steps, 32d training only the heads Go reads from a notes model, 32c every head as the control. On the steady check (audio v0.8.8):
  32c renders .930, clean .828, Pixel .697, Galaxy .714, comp .593, solo .816, MusicNet .615, vocadito .698;
  32d .931, .839, .694, .697, .573, .796, .613, .685.
- The heads Go never reads still teach the shared layers: every head stays, and 33b (the same test on corpus 16) was dropped.

**Drums from the notes model** (8 November)
- Drum heads on 100 held-out E-GMD recordings, F1 at each model's best bar: 12a kick .34, snare .54, hihat .65; 32c .88, .82, .81, as are 32a, 33a and 33c within .01; on 12a's own synth loops both .93 to .96.
- Through the whole rip at Go's drum bar (0.1 over twice the local median), on the 65 recordings over 4 s: 12a .36, .55, .68; the notes model's scores as they come .72, .62, .75, flooding (precision .53 to .59); turned into posteriors (the trainer's positive weight of 8 undone) .85, .78, .80.
- But on Billie Jean the notes model hears the bass line's notes as kicks: 125 kicks on the clean recording, on the and of two and of three, on four and on the bar's last sixteenth as well as on one and three, against 12a's 38 on one and three, and kicks as snares. Its frame path reads the harmonic stack pooled, where a bass attack and a kick look alike. 12a stays.

**33a, 33b, 33c**
- From 32b on corpus 16 with 15, 14, 11 and singing, every head: 33a as such; 33b with GuitarSet's and MusicNet's training recordings; 33c, 33a fine-tuned 7,000 steps at rate 3e-4 with E-GMD under half the drumless windows.
- None beat 32a or 32c. Mean of nine sets .717 each against 32a's .722 and 32c's .724; 33b's solo .829 the best yet but the Pixel .681; 33c's drums moved under .005.

**34a, 34b** (the width test)
- From scratch, 20,000 steps on corpus 16's mix: 34a with 48 channels at each cell (264,566 weights), 34b with 32.
- 34a's held-out loss 1.043 against 1.077; the mean .718 against .717: clean .845 (the best yet) and the Pixel +.012, solo -.024. The rip takes 25% longer on one core (74.0 s against 59.4 s on the clean take).
- Width doesn't pay under the real-time budget. Both from scratch land near 32a's .722: the data sets the level, not the size or the rounds behind a model.

**34c, 34d**
- 32c fine-tuned 15,000 steps at rate 3e-4 on corpus 16's mix; 34d adds `--dip 0.5`, a 6 to 20 dB dip centred 100 to 500 Hz in a quarter of the windows, as the phones at 2 m measured.
- Billie Jean's bass from the Pixel: 4 right of 138 for 34c, 7 for 34d, 3 for 32a. The four takes' low notes right, clean, 30 cm, Pixel, Galaxy: 94, 5, 23, 32 for 34c and 91, 6, 24, 34 for 34d. GuitarSet's low extras: comp 1370 and 1417, solo 440 and 415.
- F1, 34c then 34d: renders .933, .929; clean .829, .832; 30 cm .622, .616; Pixel .690, .696; Galaxy .702, .711; comp .575, .596; solo .807, .808; MusicNet .616, .616; vocadito .693, .688.
- The dip moves the phones' bass the right way by a few notes and lifts strummed guitar, but 34d still trails 32a on the 30 cm and Pixel takes and solo guitar. Neither installed; 32a stays.

**Soups** (8 November)
- Weights averaged: 32a with 32c, and 32a, 32c, 33a, 33b and 33c. Means .724 and .723: each lands between its members. No more soups.

**Bass on phones** (8 November)
- 32a's bass recall: clean .50 (91 of 182), 30 cm .03, Pixel .13, Galaxy .18. At the missed notes the pitch head hears the bass (median .65 to .82) but the strike head is weak (median .38 to .55): 24 to 47 sit between .7 and .9, which the bass's bar of .9 drops before the accounting judges them.
- The phones at 2 m sit 15 dB down at 160 to 320 Hz against the clean take, 7 to 9 dB around it. That is 34d's dip.

**37a, 37b, 37c, 37d** (11 November, the glides)
- 32a fine-tuned 3,000 steps at rate 3e-4 on its recipe with the cents head, validated on `~/ripcorpus-val37`: 37a with the plain small corpus first among the extras at 0.6, 37b the glide one, 37c 37b with `--even-glides`, 37d with no small corpus.
- The glide corpus taught the head a cent: gliding readings 18.2 off against 37a's 19.2 on the glide loops neither trained on (25 for a head that knows nothing), both too sure there (51% and 47% within). `--even-glides` made the head state its error (62%) and read 16.9 off, and split sung notes: vocadito .615 against .687, fragments 165 against 143.
- Every arm lost to 32a on the clean take (.78 to .80 against .835), the phones and MusicNet (.675 to .696 against .711), 37d too, whose strikes scored above 32a's on the validation corpus (.523 against .515): under scores set with 32a's heads, a change to the heads reads as a loss. None installed.

## Decoder changes that moved scores without training

- Audio v0.5.1 reads low notes by strikes, the old bass line only where no strike sounds. 22a to .784, .680 to .758; 24a to .821, .661 to .750. Much of the gap to Basic Pitch under E4 was the decoder.
- Audio v0.5.1 also keeps a drum silent when the model is never sure of it, and gives each register one lasting voice.
- Audio v0.7.0 holds the notes to the recording (the accounting stage). 25a to .825, .650 to .769 at its bars; Für Elise on the renders up .03 to .06, on phones up .01 to .02.

## What has held up

- Convolutions before anything else (7c).
- PCEN and a phone channel in training (12a).
- Labelling only what can be heard when mixing sources (16a).
- Harmonic stacking only with weights shared across pitches (19a against 22a).
- More training steps and more weights have not been what helps (18a, 22a).
- Keeping notes apart from attention in training (22a against 23a).
- Training every head, the ones Go never reads included (32c against 32d).
- New kinds of data, not more of one kind: corpus 16 at another seed added nothing (33a to 34d), and width neither (34a against 34b).
