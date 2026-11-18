# Faults found in the sets' own files

The rip's models train on public sets of recordings (the [credits](../CREDITS.md) name each, its version and its
page). This is every fault found in a set's own files while it was prepared, written down for the set's publishers
and for anyone else who uses it: the files, the count, how it was measured and the day, so that each can be checked
against the set's archive and mended at its source.

- Only a set's own files are described, as its archive unpacks. Nothing here is a fault of this repository's tools.
- **Found wrong** is something a publisher could mend. **Worth knowing** is no fault, but a user of the set would
  want it said.
- The tools that measured them are in [`tool/rip-lab`](../tool/rip-lab/README.md): `datasets/wavscan.py` for formats,
  levels and clipping, `datasets/vocalsing.py` for how VocalSet is sung, `labels/guitar.py` for GuitarSet's notes,
  `audit/stringrise.py` for where a string rises against its note's start.
- Each entry says how it was measured, in a line. A hash is a SHA-256 of a whole file. A level is in dB of full scale,
  the largest sample of a file against the largest a 16-bit file can hold.
- Every entry was measured on the day it gives, in November 2026, on the files alone; none is taken from an earlier
  reading. An entry is added when a fault is found, and a count taken on a part of a set says so.

## VocalSet v1.2

20 singers, 3,613 recordings under `data_by_singer/`, 8.79 hours, every one 44.1 kHz mono 16-bit.

Found wrong, 15 November 2026:

- **Three recordings are held twice, byte for byte**, the second under a name that ends `(1)`; all three are
  female2's. By one hash a file: 3,613 files, 3,610 distinct.
  - `female2/scales/straight/f2_scales_straight_u(1).wav` is `f2_scales_straight_u.wav`
  - `female2/scales/slow_piano/f2_scales_f_slow_piano_u(1).wav` is `f2_scales_f_slow_piano_u.wav`
  - `female2/scales/vibrato/f2_scales_vibrato_a(1).wav` is `f2_scales_vibrato_a.wav`
- **File names that break the set's own pattern** (`<f or m><number>_<exercise>_..._<vowel>.wav`), by reading each
  name against the folders it lies in:
  - 33 names do not start with their singer: 13 of male10, 10 of female4, 4 of female2, 2 of male3, 2 of male8, 1 of
    female3, 1 of female9. So `female2/arpeggios/lip_trill/lip_trill_arps.wav`,
    `female4/scales/straight/scales_straight_a.wav`, `male10/scales/fast_piano/scales_c_fast_piano_a.wav`,
    `male3/scales/lip_trill/scales_lip_trill.wav`.
  - One of those starts with another singer: a vibrato excerpt under `male8/excerpts/vibrato/` is named `m9_...`,
    male9's prefix. Whether the name or the folder is wrong, the files do not say.
  - 20 of female2's are spelled `arepggios` (`female2/arpeggios/fast_forte/f2_arepggios_c_fast_forte_a.wav` and its
    like), and 10 of male8's `sow_forte` (`male8/scales/slow_forte/m8_scales_c_sow_forte_a.wav` and its like).
  - One name says another technique than its folder: `female9/arpeggios/fast_forte/arps_fast_piano_c.wav`. female9
    has 11 recordings under fast forte arpeggios and 5 under fast piano arpeggios, where most singers have 10 and 10.
- **Singers differ in how many recordings a technique holds**, most often by a missing half. By a count of the files
  in each folder: the count most singers have, then the singers who differ.
  - arpeggios, belt: 5; female6 has 10. Arpeggios, fast forte: 10; female9 11, male6 8. Arpeggios, fast piano: 10;
    female2 11, female9 5, male3 6, male6 8. Arpeggios, lip trill and vibrato: 5; female2 6 each.
  - scales, fast forte and slow forte: 10; female4 5 each. Scales, fast piano: 10; female3 11, female4 5. Scales,
    slow piano: 10; female2 11, female4 5, female7 11. Scales, lip trill: 5; male3 6. Scales, straight and vibrato: 5;
    female2 6 each. Scales, vocal fry: 5; female2 6, male1 2.
  - long tones, messa: 5; female4 4. Long tones, trill: 5; female4 has none. Excerpts, vibrato: 3; male2 2.
  - female4 has 156 recordings in all, where the others have 178 to 189.
- **Some recordings are very quiet for a 16-bit file, and three are clipped.** By each file's largest sample, and its
  longest run of samples at full scale:
  - 174 recordings peak under -30 dB and 18 under -40 dB, the lowest at -46.6 dB
    (`female1/arpeggios/vocal_fry/f1_arpeggios_vocal_fry_i.wav`); the median peak is -14.7 dB. A file that peaks at
    -40 dB uses about nine of its sixteen bits. The quietest are vocal fry and breathy takes.
  - 50 recordings hold a sample at full scale, and three a run of three or four of them, a clipped peak:
    `female9/arpeggios/slow_forte/f9_arpeggios_f_slow_forte_a.wav`,
    `female1/arpeggios/fast_forte/f1_arpeggios_f_fast_forte_a.wav`,
    `female7/arpeggios/fast_forte/f7_arpeggios_f_fast_forte_a.wav`.
- `test_singers_technique.txt` ends without a newline, so a count of its lines reads 4 for its 5 singers.
- **The set's page says what its files do not** (the page as read on 15 November 2026): "9 male, 11 female", where
  the folders and the set's own `readme-anon.txt` have 11 male and 9 female singers; and "10.1 hours", where
  `data_by_singer/` holds 8.79 hours by its WAV headers.

Worth knowing:

- The set states no notes, no pitch and no loudness beyond a technique's name. Notes for it have to be read from the
  sound by a pitch tracker, and are that tracker's and not the set's. Its sheet (`DataSetVocalises.pdf`, in the
  archive) does give what each take sings: an arpeggio is a bar of seven notes, a scale a bar of seventeen, a
  long-tones take three whole notes, so a label can be checked by its count of notes.
- The sheet lists an "Excerpt of Singer's Choice (ca. 1 minute)" after its three excerpts, each a piece in the
  public domain. The files hold the three alone: 139 recordings under `excerpts/`, 60, 40 and 39 by piece. So no
  recording of the set is a piece a singer chose.
- A trained voice's loud takes are hard for a pitch tracker that measures a period on the whole band. The takes of
  male10 and male1 sung forte or with vibrato hold 0.75 to 0.83 of their energy between 1.5 and 4 kHz, under a
  vibrato of 82 to 146 cents either way at 4 to 5.6 Hz on low notes (by the spectrum of a whole take, and by a pitch
  read under 700 Hz with a second's drift taken out; ten takes, 15 November 2026). A YIN reading of the whole band
  finds a clear period in 0.09 to 0.24 of their loud frames, and in 0.93 to 0.99 once the signal is low-passed at
  700 Hz.
- **Each singer keeps a pitch standard of their own, and two rise as they go on.** The takes are sung with no
  accompaniment. Against the key a take's name gives (C where it gives none), the first note of 2,554 takes of the
  arpeggios, scales and long tones, read by the pitch whose harmonics hold the most of the spectrum: a singer's
  median runs from 63 cents flat (male1) to 56 sharp (male6); female2, female3, female7, female9 and male6 sit 37 to
  56 cents sharp, male1 and male3 38 and 63 flat. female2 and male6 rise through a technique's takes: 12 and 15 of
  theirs begin more than a semitone and a half over the named key
  (`female2/scales/slow_forte/f2_scales_f_slow_forte_o.wav`, named for F, begins 263 cents over it;
  `female2/scales/slow_piano/f2_scales_c_slow_piano_e.wav`, named for C, is sung on D). Within a take the notes
  scatter about that standard: on 20,006 notes of 1,999 takes that are not fast ones, the settled middle of a note
  is within 10 cents of its key in 0.40, within 20 in 0.67, within 30 in 0.82 and within 50 in 0.95. So a take's
  name does not give its key, and a note's key by A440 is not settled for a take sung half a semitone off.
  15 November 2026.
- **A period's measure reads a loud low note an octave over.** Of the 2,978 takes of the arpeggios, scales and long
  tones, 240 hold a note that a YIN reading of the signal under 1.3 kHz puts 9 semitones or more from the key its
  exercise gives it, a whole note long: 233 an octave over and 7 under, 214 of them men's takes, most of them sung
  forte, belted or with vibrato (`male1/arpeggios/belt/m1_arpeggios_belt_i.wav`: the G at 196 Hz, both times it is
  sung). On 287 such notes the second harmonic stands 14 dB over the first (median; 11 to 17 for the middle half)
  and 19 over the third, twice the note's pitch lying from 333 to 1047 Hz; on 2,272 notes of the same takes read
  right it stands 5 dB over the first. Inside 211 of the 287 the measure holds both octaves, a tenth of the frames or
  more each, which no voice sings. A reading by harmonics (the pitch whose first ten harmonics hold the most of the
  spectrum) does little better: it says the octave over for 128 of them and the written octave for 139. The
  exercise says which octave is sung. 15 November 2026.
- **Vibrato reaches 180 cents either way.** By the held notes (0.4 s or more) of 2,518 takes, the middle three
  fifths of each: 56 cents either way at the median take, 81 at three quarters of them, over 100 in a tenth, the
  widest 180, at 5.7 Hz (4.3 to 7.7 for nine notes in ten). Takes named `straight` hold 12 to 16. A quarter of the
  takes hold 80 cents or more, under which a scale's semitone step is hard to find from the pitch alone.
  15 November 2026.

## GuitarSet v1.1.0

360 microphone recordings (`audio/*_mic.wav`), 3.05 hours, every one 44.1 kHz mono 16-bit; 360 JAMS files.

Found wrong, 15 November 2026:

- **Seven notes run past the start of the next note on their own string**, by nothing a float can hold in four and
  by one sample (22.676 microseconds) in three. By each `note_midi` annotation's times and durations, a string at a
  time, counted by two programs written apart.
  - by a sample: `00_BN1-147-Gb_comp.jams` string 4 at 1.4526 s; `00_SS2-88-F_solo.jams` string 0 at 2.7570 s;
    `02_BN2-131-B_comp.jams` string 3 at 2.9636 s
  - by a float's last digit: `00_Funk3-98-A_comp.jams` string 0 at 17.1188 s; `00_Jazz1-130-D_comp.jams` string 3
    at 16.4670 s; `00_Jazz2-110-Bb_solo.jams` string 1 at 26.9983 s; `01_SS1-68-E_comp.jams` string 1 at 24.7513 s

Nothing else: the 360 recordings read, none is held twice (by hash), none is clipped (36 touch full scale, none for
three samples running), and their peaks lie from -24.7 dB to full scale.

Found wrong, 18 November 2026, on the debleeded pickup recordings (`audio/hex/*_hex_cln.wav`, a channel a string)
beside the JAMS files. For each annotated note: where its own string's channel rises most (the 60 ms after over the
60 ms before) within 0.3 s either side of its start and between its neighbours on the string. A rise of 15 dB is
clear; 36,214 of the 62,476 notes have one, and the other 26,262 are not judged. A note starts on a silent string
when the 60 ms after its annotated start lie 20 dB or more under the 60 ms after the rise. By `audit/stringrise.py`,
a row a recording.

- **Two recordings are annotated ahead of their sound.** In `04_BN3-154-E_comp`, 41% of the 119 notes with a clear
  rise start on a silent string and sound over 100 ms later, and its median note rises 153 ms later than the set's
  median note does. Its chords annotated at 13.28, 16.37, 17.75, 19.53 and 20.85 s are heard, and rise on their
  strings by 27 to 40 dB, at 13.41, 16.53, 17.89, 19.67 and 21.00 s. In `04_Jazz1-200-B_comp` it is 18% of 131 and
  93 ms. No other recording is over 5.2%.
- **Over the set, 282 notes with a clear rise (0.8%) start on a silent string and sound over 100 ms later:** 154 in
  players 00 to 03 (of 23,797), 105 in player 04 (of 6,645), 23 in player 05 (of 5,772).
- **A note the annotation lacks:** `03_BN3-154-E_comp`, string 5 rises 25 dB at 18.23 s (to -31 dB) and no note
  starts on it within 0.2 s. It is heard as a short high note.
- **A start with no strike:** `01_SS1-68-E_comp`, string 0 at 4.13 s (a note of 1.13 s): the channel's level does
  not rise within 0.12 s either side (+0.3 dB). It is heard as the note before it sounding on.

Worth knowing:

- The set states no velocity for a note.
- A `note_midi` value is a note's pitch as it was played, not a key: 816 of the 62,476 notes sit more than 0.4 of a
  semitone from the nearest key, the furthest at 0.5.

## NSynth

Counted on 15 November 2026 on the part of the training split then unpacked, 48,635 notes of all 953 instruments
(about a sixth of the split), and on the test split's 4,096. To be taken again on the whole training split.

Found wrong:

- **Notes outside the pitches the set's page gives** ("every pitch of a standard MIDI piano (21-108)"). By the pitch
  in each file's name: 337 of the 48,635 training notes lie outside 21 to 108 (292 under, 45 over; the lowest 9, the
  highest 120), and 36 of the 4,096 test notes (30 under, 6 over; 9 to 119).
- **Notes of two pitches, or of two instruments, that are one file**, byte for byte. By one hash a note: 41 contents
  are held by notes of more than one instrument or pitch, 84 files in all.
  - 9 are one instrument at two pitches, eight a semitone apart and one a tone, so one label of the two is not the
    sound's: `bass_electronic_012-059-025.wav` is `bass_electronic_012-060-025.wav`,
    `guitar_acoustic_013-061-075.wav` is `guitar_acoustic_013-062-075.wav`, `bass_electronic_019-025-025.wav` is
    `bass_electronic_019-027-025.wav`. Read for its pitch (the pitch whose first ten harmonics hold the most of the
    spectrum of a tenth of a second, at five points of the note, octaves aside), the shared sound is the lower
    label's in six of the nine, all `guitar_acoustic` (013 at 61, 85 and 92 twice, 023 at 61, 025 at 58), so the
    note a semitone over is the mislabelled one; the reading settles nothing for the other three.
  - 31 are two instruments at one pitch: `string_acoustic_026-088-100.wav` is `string_acoustic_038-088-100.wav`.
- **Velocity layers that are one recording.** Of 11,301 pairs of an instrument and a pitch with two layers or more
  unpacked, 65 hold two layers that are one file byte for byte (77 files), most among `organ_electronic` (32 pairs),
  `string_acoustic` (15) and `keyboard_electronic` (6). Many more hold the same sound in other bytes: by the
  correlation of the softest layer's samples with the hardest's, up to 150 pairs a family, the two correlate above
  0.999 at a level gap of 0.00 dB in 97 of 149 pairs of `vocal_acoustic`, 72 of 129 of `keyboard_synthetic`, 31 of
  150 of `flute_acoustic`, 27 of 150 of `brass_acoustic` and of `bass_synthetic`. In those a velocity names nothing
  in the sound.
- **Voices whose sound is not their label's pitch throughout.** Of the 653 `vocal_acoustic` notes, 70 read more than
  0.6 of a semitone from their label at one of seven points between 0.1 s and 2 s (the same pitch reading): some
  scoop up from as much as a semitone under in their first fifth of a second (`vocal_acoustic_004-052-127.wav` reads
  51.0 at 0.1 s and 52.0 from 0.3 s), and some read an octave over their label (`vocal_acoustic_001-046-100.wav`,
  `vocal_acoustic_004-074-025.wav`), as a choir doubled at the octave would. This is what a voice does and may be
  no fault; it is here because a user who takes the label as the pitch of every instant is wrong about them.

Worth knowing:

- A note's velocity is in its timbre and not in its level: over 400 pairs of an instrument and a pitch with three
  velocity layers or more, the loudest 50 ms of the hardest layer sat a median 0.1 dB from the softest's.

## E-GMD v1.0.0

A part of the set is unpacked; what follows is from a draw of 194 recordings (128,601 notes), 18 November 2026.

Worth knowing:

- **A hi-hat pedal note (key 44) may not sound.** In `drummer8/session1/12_latin_118_beat_4-4_11`, the pedal notes
  at 135.23 and 137.29 s (struck at 45 and 40 of 127) are heard as nothing. Pedal notes are 14,448 of the draw's
  notes (11.2%). Found by ear at those two places; how many sound over the set is not measured yet.
- **A buzz stroke is held as a note a bounce.** In `drummer1/session2/6_rock_102_beat_3-4_51` the snare notes at 60.73,
  60.77, 60.81 and 60.83 s (struck at 35, 34, 26 and 20) are one stick left to bounce, heard as one hit that dies
  away. In the draw 5,737 of 44,430 snare notes (12.9%) come under 60 ms after the snare note before. By each
  MIDI file's note-ons.
