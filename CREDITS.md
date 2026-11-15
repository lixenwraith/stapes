# Credits

stapes' code is under the BSD 3-Clause licence in [LICENSE](LICENSE). The rip's
models learn from the collections below or are measured on them, each credited
here as its licence asks. No file of any collection is in this repository, and
neither are the weight files (`ripframe.bin`, `ripnotes.bin`): a weight file
passed on carries this file.

A collection under the Creative Commons Attribution 4.0 International licence
(CC BY 4.0, <https://creativecommons.org/licenses/by/4.0/>, its text in
[CC-BY-4.0.txt](CC-BY-4.0.txt)) is offered as is, under the licence's
disclaimer of warranties and limitation of liability (its section 5). What
stapes changes in each:

- A recording is read into log-mel frames (22,050 Hz, a hop of 256 samples,
  229 bands) and cut into training windows, some heard through a simulated
  speaker, room and phone, some through a codec, some mixed with other
  windows; its annotations are rewritten as stapes' labels.
- A MIDI file's stretches are played on the engine's voices, and the render is
  read as a recording.

## The collections

The corpora take these as [Rip training](doc/rip-training.md) and the
[rounds](doc/rip-rounds.md) say, and their held-out parts measure the models:

- **The Expanded Groove MIDI Dataset (E-GMD)** v1.0.0, by Lee Callender,
  Curtis Hawthorne and Jesse Engel, made available by Google LLC,
  <https://magenta.withgoogle.com/datasets/e-gmd>, CC BY 4.0. Its train
  split's recordings were laid over loops with no drums or drawn as they
  stand, their MIDI hits the labels (132 recordings' moved by a steady lag, 15
  left out); its test split measures the drums.
- **VocalSet: A Singing Voice Dataset** v1.2, by Julia Wilkins, Prem
  Seetharaman, Alison Wahl and Bryan Pardo,
  <https://doi.org/10.5281/zenodo.1442513>, CC BY 4.0. Its recordings are in
  the singing corpus, their notes read from a pitch track of each; five
  singers are held out to measure.
- **vocadito** v3, by Rachel Bittner, Katherine Pasalo, Juan José Bosch,
  Gabriel Meseguer Brocal and David Rubinstein,
  <https://doi.org/10.5281/zenodo.5578807>, CC BY 4.0. Clips 1 to 30 and their
  notes are in the singing corpus; clips 31 to 40 measure.
- **The NSynth Dataset**, by Jesse Engel, Cinjon Resnick, Adam Roberts, Sander
  Dieleman, Douglas Eck, Karen Simonyan and Mohammad Norouzi, made available
  by Google Inc., <https://magenta.withgoogle.com/datasets/nsynth>, CC BY 4.0.
  Its notes played a rendered loop's tonal lines, a line with a chance of a
  half (`corpus ... sampled`), each note shifted up to three semitones, its
  level taken out and cut at its note-off; its held-out instruments measure
  what the models name.
- **GuitarSet** v1.1.0, by Qingyang Xi, Rachel M. Bittner, Johan Pauwels,
  Xuzhou Ye and Juan P. Bello, <https://doi.org/10.5281/zenodo.3371780>, CC BY
  4.0. Its recordings are labelled with their notes, pitch contours and beats;
  players 00 to 03 train, 04 validates and 05 tests.
- **MusicNet** v1.0, by John Thickstun, Zaid Harchaoui and Sham M. Kakade,
  <https://doi.org/10.5281/zenodo.5120004>, CC BY 4.0, whose recordings are
  Creative Commons and public-domain performances from the Isabella Stewart
  Gardner Museum, the European Archive Foundation and Musopen, each one's
  source in its metadata. They are labelled with their aligned scores, each
  onset moved to a model's strike peak where one stands near it; 19 pieces
  validate and 10 test.

## The engine's sounds

The audio module fitted its piano and bass to the University of Iowa Musical
Instrument Samples, by Lawrence Fritts
(<https://theremin.music.uiowa.edu/MIS.html>, free for any use without
restriction), and measured its kit against them and against drum one-shots on
Freesound dedicated to the public domain (CC0 1.0). Neither asks for credit;
stapes renders its corpora on those voices.

## Not used

- The Lakh MIDI Dataset v0.1, by Colin Raffel,
  <https://colinraffel.com/projects/lmd/>, CC BY 4.0, is out: its files are
  transcriptions of commercial songs, and its licence covers the collection,
  not the songs. Earlier rounds played its clean MIDI subset (corpora 14 to 17
  and the glide pair); those corpora and every weight trained before they went
  were deleted on 2026-11-14, and the models train again from nothing.
  Slakh2100 (CC BY 4.0) is rendered from it, so it is out with it.
- MAESTRO (Google LLC, CC BY-NC-SA 4.0) and MAPS (V. Emiya and B. David, CC
  BY-NC-SA 2.0 France) are non-commercial, so they stay out of training.
- URMP, by Bochen Li, Xinzhao Liu, Karthik Dinesh, Zhiyao Duan and Gaurav
  Sharma, states no licence on its own pages; its Dryad deposit is CC0 1.0
  under Dryad's terms. Planned, not used.
