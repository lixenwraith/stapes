package stapes

import (
	"cmp"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"

	"github.com/lixenwraith/audio"
)

// music is what a corpus loop plays: a timeline RenderBars plays under heads,
// the key and each bar's chord, each slot's tracks as parts, and the samples of
// the render that are the loop
type music struct {
	kind     string // bank, arranged or overlap
	timeline [][audio.MusicSlots]audio.PatternID
	heads    []audio.BarHead
	root     int
	scale    audio.ScaleID
	prog     []int
	chords   [][]int // each bar's chord, its pitch classes from its root's; nil for none
	parts    [audio.MusicSlots][]*part
	from, to int
	kit      *loopKit // the kit its drums play, the bank's drums where nil
}

// part is one line of a loop, a track of its slot's patterns: its kind, the
// instrument it plays on, or the recorded one in place of it, the pitches that
// instrument has, and its notes in steps from the timeline's start; or a kit's
// piece, or the strikes a kit writes beside the drums' lines
type part struct {
	kind   *partKind
	instr  audio.InstrumentType
	sample *sampleInstr
	lo, hi int
	notes  []note
	piece  *kitPiece
	kit    bool // played from the kit's stream
}

// note is a part's note: its onset and length in steps, its pitch, 0 for a
// drum's, its velocity, and its bend as a Step holds one; a drum's, whether it
// is a fill's, and the kit's piece it is dealt to where not its line's
type note struct {
	at, dur      float64
	midi         int
	vel          float64
	bend, bendAt float64
	fill         bool
	piece        *kitPiece
}

// partKind is a kind of line: the slot it plays in, the families it is drawn
// on by weight, its velocities' range, its onsets' jitter in ms, and its writer
type partKind struct {
	name     string
	slot     int
	families map[string]float64
	level    [2]float64
	jitter   float64
	write    func(a *arranger, ps []*part)
}

// partKinds are the lines the arranger writes, in the order it writes them:
// the left hand on the chords' instrument, a double on the melody's notes
var partKinds = []partKind{
	{"drums", 0, nil, [2]float64{0.75, 1}, 2, writeDrums},
	{"bass", 3, map[string]float64{"bass": 6, "piano": 1, "synth": 2}, [2]float64{0.6, 0.9}, 4, writeBass},
	{"chords", 2, map[string]float64{"piano": 5, "guitar": 2, "synth": 2, "strings": 1}, [2]float64{0.35, 0.7}, 6, writeChords},
	{"left", 2, nil, [2]float64{0.4, 0.75}, 6, writeLeft},
	{"strum", 2, map[string]float64{"guitar": 6, "piano": 1}, [2]float64{0.35, 0.7}, 5, writeStrum},
	{"pad", 2, map[string]float64{"synth": 4, "strings": 3, "voice": 2, "winds": 1}, [2]float64{0.25, 0.5}, 6, writePad},
	{"arpeggio", 1, map[string]float64{"piano": 3, "guitar": 3, "synth": 2}, [2]float64{0.35, 0.65}, 5, writeArpeggio},
	{"melody", 1, map[string]float64{"synth": 3, "piano": 3, "guitar": 2, "voice": 2, "winds": 2, "strings": 2}, [2]float64{0.6, 0.95}, 8, writeMelody},
	{"double", 1, map[string]float64{"synth": 3, "piano": 2, "guitar": 2, "voice": 2, "winds": 2, "strings": 2}, [2]float64{0.6, 0.9}, 4, writeDouble},
	{"counter", 1, map[string]float64{"strings": 3, "winds": 3, "synth": 2, "piano": 1}, [2]float64{0.4, 0.65}, 8, writeCounter},
}

// textures are the parts an arranged loop plays, by weight a hundred loops: a
// line alone in 12, two parts in 18, three in 25, four in 25, five in 12 and
// six in 8, left and chords a keyboard's two hands
var textures = []struct {
	parts  []string
	weight float64
}{
	{[]string{"melody"}, 6}, {[]string{"bass"}, 2}, {[]string{"arpeggio"}, 2}, {[]string{"chords"}, 1}, {[]string{"strum"}, 1},
	{[]string{"left", "chords"}, 6}, {[]string{"bass", "melody"}, 4}, {[]string{"chords", "melody"}, 3},
	{[]string{"arpeggio", "melody"}, 2}, {[]string{"strum", "melody"}, 2}, {[]string{"pad", "melody"}, 1},
	{[]string{"drums", "bass", "melody"}, 6}, {[]string{"left", "chords", "melody"}, 5}, {[]string{"bass", "chords", "melody"}, 4},
	{[]string{"drums", "bass", "chords"}, 3}, {[]string{"bass", "strum", "melody"}, 3}, {[]string{"bass", "pad", "melody"}, 2},
	{[]string{"chords", "melody", "double"}, 2},
	{[]string{"drums", "bass", "chords", "melody"}, 8}, {[]string{"drums", "bass", "strum", "melody"}, 4},
	{[]string{"drums", "left", "chords", "melody"}, 4}, {[]string{"drums", "bass", "pad", "melody"}, 4},
	{[]string{"drums", "bass", "arpeggio", "melody"}, 2}, {[]string{"bass", "chords", "melody", "double"}, 3},
	{[]string{"drums", "bass", "chords", "melody", "double"}, 3}, {[]string{"drums", "bass", "pad", "arpeggio", "melody"}, 3},
	{[]string{"drums", "bass", "strum", "melody", "counter"}, 2}, {[]string{"drums", "bass", "chords", "pad", "melody"}, 2},
	{[]string{"drums", "left", "chords", "melody", "double"}, 2},
	{[]string{"drums", "bass", "chords", "pad", "melody", "double"}, 3}, {[]string{"drums", "bass", "strum", "arpeggio", "melody", "counter"}, 3},
	{[]string{"drums", "bass", "chords", "arpeggio", "melody", "double"}, 2},
}

// familyRanges is the pitches an instrument of each family plays, the 88 keys
// for a keyboard and a synthesizer
var familyRanges = map[string][2]int{
	"piano": {21, 108}, "synth": {21, 108}, "unknown": {21, 108}, "bass": {21, 67},
	"guitar": {40, 88}, "voice": {36, 84}, "strings": {28, 100}, "winds": {34, 98},
}

// feel is how a bar is counted: the steps of each pulse, the pulses its weight
// falls on, and its backbeats
type feel struct {
	pulses       []int
	strong, back []int
}

// corpusMeters are the meters a corpus of meters draws its loops in, by share,
// each felt one of its ways a loop: 5/4 as 3+2 or 2+3 beats, 7/8 as 2+2+3,
// 3+2+2 or 2+3+2 eighths. Without meters every loop is in the first.
var corpusMeters = []struct {
	m     audio.Meter
	share float64
	feels []feel
}{
	{audio.Meter{}, 0.40, []feel{{[]int{4, 4, 4, 4}, []int{0, 2}, []int{1, 3}}}},
	{audio.Meter{Num: 3, Den: 4}, 0.15, []feel{{[]int{4, 4, 4}, []int{0}, []int{1, 2}}}},
	{audio.Meter{Num: 6, Den: 8}, 0.12, []feel{{[]int{6, 6}, []int{0}, []int{1}}}},
	{audio.Meter{Num: 12, Den: 8}, 0.10, []feel{{[]int{6, 6, 6, 6}, []int{0, 2}, []int{1, 3}}}},
	{audio.Meter{Num: 2, Den: 4}, 0.08, []feel{{[]int{4, 4}, []int{0}, []int{1}}}},
	{audio.Meter{Num: 7, Den: 8}, 0.08, []feel{{[]int{4, 4, 6}, []int{0, 2}, []int{1}}, {[]int{6, 4, 4}, []int{0, 2}, []int{1}}, {[]int{4, 6, 4}, []int{0, 2}, []int{1}}}},
	{audio.Meter{Num: 5, Den: 4}, 0.07, []feel{{[]int{4, 4, 4, 4, 4}, []int{0, 3}, []int{2, 4}}, {[]int{4, 4, 4, 4, 4}, []int{0, 2}, []int{1, 4}}}},
}

// corpusKeys are the scales an arranged loop is in, by weight, each with the
// scale its chords stack in, a pentatonic tune over its minor's, and the
// progressions drawn, four chords' degrees
var corpusKeys = []struct {
	scale, chords audio.ScaleID
	weight        float64
	progs         [][]int
}{
	{audio.ScaleMajor, audio.ScaleMajor, 30, [][]int{{0, 4, 5, 3}, {0, 3, 4, 0}, {5, 3, 0, 4}, {0, 5, 3, 4}, {1, 4, 0, 0}, {0, 3, 5, 4}, {0, 4, 3, 4}, {3, 0, 4, 4}}},
	{audio.ScaleMinor, audio.ScaleMinor, 30, [][]int{{0, 5, 2, 6}, {0, 3, 4, 0}, {0, 6, 5, 6}, {0, 3, 6, 2}, {0, 5, 6, 0}, {0, 2, 6, 3}, {5, 6, 0, 0}, {0, 3, 0, 4}}},
	{audio.ScaleDorian, audio.ScaleDorian, 12, [][]int{{0, 3, 0, 3}, {0, 6, 3, 0}, {0, 1, 2, 3}, {0, 3, 6, 0}}},
	{audio.ScaleHarmonicMinor, audio.ScaleHarmonicMinor, 10, [][]int{{0, 3, 4, 0}, {0, 5, 4, 0}, {0, 1, 4, 0}, {0, 3, 5, 4}}},
	{audio.ScaleMinorPent, audio.ScaleMinor, 10, [][]int{{0, 5, 2, 6}, {0, 3, 4, 0}, {0, 6, 5, 6}, {0, 3, 6, 2}}},
	{audio.ScalePhrygian, audio.ScalePhrygian, 8, [][]int{{0, 1, 0, 1}, {0, 1, 2, 1}, {0, 6, 5, 1}, {0, 5, 1, 0}}},
}

// pick is an index drawn by weight
func pick(rng *rand.Rand, weights ...float64) int {
	var sum float64
	for _, w := range weights {
		sum += w
	}
	u := rng.Float64() * sum
	for i, w := range weights {
		if u -= w; u < 0 {
			return i
		}
	}
	return len(weights) - 1
}

// arranger writes one loop's parts in steps from its timeline's start, the
// key's root low enough that every key of the 88 is a step of the scale over it
type arranger struct {
	rng      *rand.Rand
	m        audio.Meter
	feel     feel
	steps    int // a bar's
	bars     int // the timeline's, a pickup bar among them
	first    int // the bar every part plays from: 1 after a pickup bar
	bpm      int
	root     int
	scale    audio.ScaleID
	iv       []int
	degrees  []int   // each bar's chord's degree
	chords   [][]int // each bar's chord, its pitch classes from its root's
	swing    float64 // the share of a beat pair its first note takes: 0.5 straight
	swing16  bool    // the sixteenths swing in place of the eighths
	from, to float64 // the steps the loop keeps
	parts    []*part
	kit      *loopKit   // the kit its drums play, none where nil
	krng     *rand.Rand // the kit's stream
}

// newArranger draws an arranged loop's meter and feel and, with starts, where
// it starts, from mrng under meters, and its key and progression from rng
func newArranger(rng, mrng *rand.Rand, bars, bpm int, meters, starts bool) *arranger {
	a := &arranger{rng: rng, bars: bars, bpm: bpm, swing: 0.5}
	row := corpusMeters[0]
	pickup := false
	if meters {
		shares := make([]float64, len(corpusMeters))
		for i, r := range corpusMeters {
			shares[i] = r.share
		}
		row = corpusMeters[pick(mrng, shares...)]
		pickup = starts && mrng.Float64() < 0.3
	}
	a.m, a.feel, a.steps = row.m, row.feels[mrng.IntN(len(row.feels))], row.m.Steps()
	a.from, a.to = 0, float64(bars*a.steps)
	if pickup {
		a.bars++
		starts := a.pulseStarts()
		j := 1 + mrng.IntN(len(starts)-1)
		a.from = float64(starts[j])
		if mrng.Float64() < 0.5 {
			a.first = 1 // an anacrusis: the melody leads in, the rest from the bar line
		} else if mrng.Float64() < 0.5 {
			a.from += float64(a.feel.pulses[j]) / 2 // cut out of the music half a pulse in
		}
		a.to = a.from + float64(bars*a.steps)
	}
	weights := make([]float64, len(corpusKeys))
	for i, k := range corpusKeys {
		weights[i] = k.weight
	}
	key := corpusKeys[pick(rng, weights...)]
	a.root, a.scale, a.iv = 24+rng.IntN(12), key.scale, audio.ScaleIntervals(key.scale)
	prog := key.progs[rng.IntN(len(key.progs))]
	rhythm := 1
	if rng.Float64() < 0.2 {
		rhythm = 2 // a chord holds two bars
	}
	civ := audio.ScaleIntervals(key.chords)
	for b := range a.bars {
		d := prog[len(prog)-1] // a pickup bar sounds the chord the progression comes back from
		if b >= a.first {
			d = prog[(b-a.first)/rhythm%len(prog)]
		}
		size := 3
		if rng.Float64() < 0.25 {
			size = 4 // a seventh on a quarter of the chords
		}
		tones := []int{}
		for k := range size {
			j := d + 2*k
			tones = append(tones, (a.root+civ[j%len(civ)])%12)
		}
		if b > a.first && d == a.degrees[b-1] {
			tones = a.chords[b-1]
		}
		a.degrees, a.chords = append(a.degrees, d), append(a.chords, tones)
	}
	if !a.compound() && rng.Float64() < 0.25 {
		a.swing, a.swing16 = 0.56+0.12*rng.Float64(), rng.Float64() < 0.3
	}
	return a
}

// compound is whether the bar is felt in pulses of three eighths
func (a *arranger) compound() bool { return slices.Contains(a.feel.pulses, 6) }

func (a *arranger) pulseStarts() []int {
	out, at := []int{}, 0
	for _, n := range a.feel.pulses {
		out, at = append(out, at), at+n
	}
	return out
}

func (a *arranger) barAt(b int) float64 { return float64(b * a.steps) }

// ms is a span in ms as steps of the loop's tempo
func (a *arranger) ms(ms float64) float64 {
	return ms / 1000 * audio.AudioSampleRate / float64(audio.SamplesPerStep(a.bpm))
}

// above is the lowest pitch of pitch class pc at or over from
func above(pc, from int) int { return from + ((pc-from)%12+12)%12 }

// fold moves a pitch by octaves into lo to hi
func fold(midi, lo, hi int) int {
	for midi < lo {
		midi += 12
	}
	for midi > hi && midi-12 >= lo {
		midi -= 12
	}
	return min(midi, hi)
}

// pitch is the step idx of the scale over the root, a step an octave of the
// scale's apart from idx plus its length
func (a *arranger) pitch(idx int) int {
	n := len(a.iv)
	o := idx / n
	if idx%n < 0 {
		o--
	}
	return a.root + 12*o + a.iv[idx-o*n]
}

// index is the step of the scale nearest midi
func (a *arranger) index(midi int) int {
	deg, oct, _ := audio.PitchStep(a.root, a.scale, midi)
	return oct*len(a.iv) + deg
}

// chordIndex is the step of the scale nearest the root of bar b's chord, in the
// octave over the key's root
func (a *arranger) chordIndex(b int) int {
	b = min(max(b, 0), a.bars-1)
	return a.index(a.root+(a.chords[b][0]-a.root%12+12)%12) % len(a.iv)
}

// voicing is size notes of a chord from about bottom up: an inversion's tones,
// its root and fifth again over them for the notes past its tones, the one
// nearest the last voicing three times in four; spread leaves a fourth between
func (a *arranger) voicing(tones []int, size, bottom, lo, hi int, last []int, spread bool) []int {
	var best []int
	cost := math.Inf(1)
	gap := 1
	if spread {
		gap = 3
	}
	near := last != nil && a.rng.Float64() < 0.75
	inv := a.rng.IntN(len(tones))
	for i := range tones {
		for _, from := range []int{bottom - 5, bottom} {
			if !near && (i != inv || from != bottom) {
				continue
			}
			order := append(slices.Clone(tones[i:]), tones[:i]...)
			for len(order) < size {
				order = append(order, tones[(len(order)-len(tones))*2%len(tones)])
			}
			v := []int{above(order[0], from)}
			for _, pc := range order[1:size] {
				v = append(v, above(pc, v[len(v)-1]+gap))
			}
			for len(v) > 0 && v[len(v)-1] > hi {
				for k := range v {
					v[k] -= 12
				}
			}
			for v[0] < lo {
				for k := range v {
					v[k] += 12
				}
			}
			c := 0.0
			for k := range min(len(v), len(last)) {
				c += math.Abs(float64(v[k] - last[k]))
			}
			if c < cost {
				best, cost = v, c
			}
		}
	}
	for k := range best {
		best[k] = fold(best[k], lo, hi)
	}
	slices.Sort(best)
	return slices.Compact(best)
}

func (p *part) add(at, dur float64, midi int, vel float64) {
	p.notes = append(p.notes, note{at: at, dur: max(dur, 0.25), midi: midi, vel: vel})
}

func (a *arranger) partOf(name string) *part {
	for _, p := range a.parts {
		if p.kind.name == name {
			return p
		}
	}
	return nil
}

// writeDrums is a kit's three parts: the kick on the strong pulses with an
// eighth between now and then, the snare on the backbeats and ghosts in a third
// of the loops, the hats on eighths, sixteenths, quarters or triplets, and a
// fill into the end in three loops of ten
func writeDrums(a *arranger, ps []*part) {
	kick, snare, hat := ps[0], ps[1], ps[2]
	rng := a.rng
	every := []float64{2, 1, 4, 4.0 / 3}[pick(rng, 50, 20, 15, 15)]
	if a.compound() {
		every = []float64{2, 6, 1}[pick(rng, 70, 15, 15)]
	} else if a.swing > 0.5 {
		every = 2
	}
	ghosts, fill := rng.Float64() < 0.3, rng.Float64() < 0.3
	starts := a.pulseStarts()
	for b := a.first; b < a.bars; b++ {
		x := a.barAt(b)
		for k, at := range starts {
			at := x + float64(at)
			if k == 0 || slices.Contains(a.feel.strong, k) && rng.Float64() < 0.9 {
				kick.add(at, 1, 0, 0.9+0.1*rng.Float64())
			}
			if slices.Contains(a.feel.back, k) && rng.Float64() < 0.92 {
				snare.add(at, 1, 0, 0.8+0.15*rng.Float64())
			}
			if off := 2.0 + float64(a.feel.pulses[k]/6*2); rng.Float64() < 0.18 {
				kick.add(at+off, 1, 0, 0.6+0.25*rng.Float64())
			}
		}
		end := float64(a.steps)
		if fill && b == a.bars-1 {
			end = float64(starts[len(starts)-1])
			writeFill(snare, x+end, x+float64(a.steps))
		}
		for t := 0.0; t < end-1e-9; t += every {
			v := 0.35 + 0.15*rng.Float64()
			if slices.Contains(starts, int(t)) && t == math.Trunc(t) {
				v += 0.2
			}
			if rng.Float64() < 0.9 {
				hat.add(x+t, 1, 0, v)
			}
		}
		if ghosts {
			for t := 1; t < int(end); t += 2 {
				if rng.Float64() < 0.12 {
					snare.add(x+float64(t), 1, 0, 0.15+0.15*rng.Float64())
				}
			}
		}
	}
	if a.first == 1 && rng.Float64() < 0.3 { // a pickup on the snare into the bar line
		for t := float64(a.steps - 2); t < float64(a.steps); t++ {
			snare.add(t, 1, 0, 0.5+0.2*rng.Float64())
		}
	}
}

// writeFill is a fill into the bar line at to, from from on: a sixteenth a
// step, rising from 0.45 by half of full over the run
func writeFill(p *part, from, to float64) {
	for t := from; t < to; t++ {
		p.add(t, 1, 0, 0.45+0.5*(t-from)/(to-from))
		p.notes[len(p.notes)-1].fill = true
	}
}

// writeBass is a bass line over a floor between A0 and G2: held roots, roots
// and fifths on the pulses, driven eighths, a syncopated figure, or a walk of
// a note a pulse into the next chord's root
func writeBass(a *arranger, ps []*part) {
	p, rng := ps[0], a.rng
	floor := max(21+rng.IntN(23), p.lo)
	style := pick(rng, 15, 25, 15, 20, 15, 10) // held, pulses, eighths, syncopated, walking, octaves
	starts := a.pulseStarts()
	for b := a.first; b < a.bars; b++ {
		x, tones := a.barAt(b), a.chords[b]
		r := fold(above(tones[0], floor), p.lo, p.hi)
		fifth, oct := fold(above(tones[2], r+1), p.lo, p.hi), fold(r+12, p.lo, p.hi)
		next := fold(above(a.chords[min(b+1, a.bars-1)][0], floor), p.lo, p.hi)
		for k, at := range starts {
			n, at := float64(a.feel.pulses[k]), x+float64(at)
			strong := k == 0 || slices.Contains(a.feel.strong, k)
			switch style {
			case 0:
				if k == 0 {
					p.add(at, float64(a.steps), r, 0.9)
				}
			case 1:
				m := r
				if !strong {
					m = []int{fifth, oct, r}[rng.IntN(3)]
				}
				p.add(at, n, m, 0.85)
			case 2:
				for t := 0.0; t < n; t += 2 {
					m := r
					if rng.Float64() < 0.2 {
						m = oct
					}
					p.add(at+t, 1.5, m, 0.8)
				}
			case 3:
				cell := [][]float64{{0}, {0, 3}, {0, 2}, {2}, {0, 2, 3}}[rng.IntN(5)]
				if n == 6 {
					cell = [][]float64{{0}, {0, 4}, {0, 3}, {2, 4}}[rng.IntN(4)]
				}
				if k == 0 && cell[0] > 0 {
					cell = append([]float64{0}, cell...)
				}
				for i, t := range cell {
					d := n - t
					if i+1 < len(cell) {
						d = cell[i+1] - t
					}
					m := r
					if t > 0 {
						m = []int{r, fifth, oct}[rng.IntN(3)]
					}
					p.add(at+t, d, m, 0.85)
				}
			case 4:
				m := r
				switch {
				case k == len(starts)-1:
					m = next + []int{-1, 1, -2, 2}[rng.IntN(4)]
				case k > 0:
					m = []int{fifth, above(tones[1], r+1), oct, r + 2}[rng.IntN(4)]
				}
				p.add(at, n, fold(m, p.lo, p.hi), 0.8)
			case 5:
				p.add(at, n/2, r, 0.85)
				p.add(at+n/2, n/2, oct, 0.7)
			}
		}
	}
}

// writeLeft is a keyboard's left hand under its chords, over a floor between
// C1 and A2: octaves or fifths held, the chord broken in eighths, or a stride
// of the root on the strong pulses and the chord on the others; each note an
// octave down while it would reach the right hand's lowest
func writeLeft(a *arranger, ps []*part) {
	p, rng := ps[0], a.rng
	floor := max(24+rng.IntN(22), p.lo)
	style := pick(rng, 25, 20, 30, 25)
	starts := a.pulseStarts()
	defer func() {
		top := p.hi
		if c := a.partOf("chords"); c != nil && len(c.notes) > 0 {
			top = slices.MinFunc(c.notes, func(x, y note) int { return x.midi - y.midi }).midi - 1
		}
		if top-p.lo < 12 {
			top = p.hi // no octave under the right hand: the hands cross
		}
		for i := range p.notes {
			p.notes[i].midi = fold(p.notes[i].midi, p.lo, top)
		}
	}()
	for b := a.first; b < a.bars; b++ {
		x, tones := a.barAt(b), a.chords[b]
		r := fold(above(tones[0], floor), p.lo, p.hi)
		fifth := above(tones[2], r+1)
		switch style {
		case 0, 1:
			top := r + 12
			if style == 1 {
				top = fifth
			}
			for _, m := range []int{r, top} {
				p.add(x, float64(a.steps), fold(m, p.lo, p.hi), 0.8)
			}
		case 2:
			line := []int{r, fifth, r + 12, fifth}
			for i, t := 0, 0.0; t < float64(a.steps); i, t = i+1, t+2 {
				p.add(x+t, 2, fold(line[i%len(line)], p.lo, p.hi), 0.7)
			}
		case 3:
			chord := a.voicing(tones, 3, max(r+12, 48), p.lo, p.hi, nil, false)
			for k, at := range starts {
				n := float64(a.feel.pulses[k])
				if k == 0 || slices.Contains(a.feel.strong, k) {
					p.add(x+float64(at), n, r, 0.85)
					continue
				}
				for _, m := range chord {
					p.add(x+float64(at), n*0.75, m, 0.6)
				}
			}
		}
	}
}

// writeChords is a chord of three to five notes struck at once, with a seventh
// now and then, its bottom between B♭2 and B♭4 and each led to the nearest
// inversion: held, on the pulses, on their offbeats, a charleston, syncopated,
// or rolled up half a sixteenth a note, held to the next or short
func writeChords(a *arranger, ps []*part) {
	p, rng := ps[0], a.rng
	size := []int{3, 4, 5}[pick(rng, 40, 40, 20)]
	bottom := min(46+rng.IntN(25), p.hi-12)
	if a.partOf("left") != nil {
		bottom = min(max(bottom, p.lo+24), p.hi-12) // two octaves under it for the left hand
	}
	rhythm := pick(rng, 25, 20, 15, 10, 20, 10)
	legato := rng.Float64() < 0.6
	starts := a.pulseStarts()
	var last []int
	for b := a.first; b < a.bars; b++ {
		x := a.barAt(b)
		var at []float64
		switch rhythm {
		case 0, 5:
			at = []float64{0}
			if rhythm == 5 && rng.Float64() < 0.5 {
				at = append(at, float64(starts[len(starts)/2]))
			}
		case 1:
			for _, s := range starts {
				at = append(at, float64(s))
			}
		case 2:
			for k, s := range starts {
				at = append(at, float64(s+a.feel.pulses[k]/2))
			}
		case 3:
			at = []float64{0, float64(starts[min(1, len(starts)-1)] + 2)}
		case 4:
			at = []float64{0}
			for t := 2; t < a.steps; t += 2 {
				if rng.Float64() < 0.35 {
					at = append(at, float64(t))
				}
			}
		}
		v := a.voicing(a.chords[b], size, bottom, p.lo, p.hi, last, false)
		last = v
		for i, t := range at {
			d := float64(a.steps) - t
			if i+1 < len(at) {
				d = at[i+1] - t
			}
			if !legato {
				d = min(d, 1+float64(rng.IntN(2)))
			}
			for j, m := range v {
				on := t
				if rhythm == 5 {
					on += 0.5 * float64(j)
				}
				p.add(x+on, d-(on-t), m, 0.7)
			}
		}
	}
}

// writeStrum is a guitar's strokes of four or five strings, its bottom between
// E2 and E3: down on the pulses, up between them over the top three strings
// mostly, each spread 15 to 60 ms down and 10 to 35 up, ringing or muted
func writeStrum(a *arranger, ps []*part) {
	p, rng := ps[0], a.rng
	size := 4 + rng.IntN(2)
	bottom := max(40+rng.IntN(13), p.lo)
	muted := rng.Float64() < 0.3
	starts := a.pulseStarts()
	var last []int
	for b := a.first; b < a.bars; b++ {
		x := a.barAt(b)
		v := a.voicing(a.chords[b], size, bottom, p.lo, p.hi, last, true)
		last = v
		var strokes []float64
		for t := 0; t < a.steps; t += 2 {
			if on := slices.Contains(starts, t); on && rng.Float64() < 0.95 || !on && rng.Float64() < 0.55 {
				strokes = append(strokes, float64(t))
			}
		}
		for i, t := range strokes {
			d := float64(a.steps) - t
			if i+1 < len(strokes) {
				d = strokes[i+1] - t
			}
			if muted {
				d = 1
			}
			down := slices.Contains(starts, int(t)) || int(t)%4 == 0
			strings, spread := v, a.ms(15+45*rng.Float64())
			if !down {
				spread = a.ms(10 + 25*rng.Float64())
				if rng.Float64() < 0.7 {
					strings = v[len(v)-3:]
				}
				strings = slices.Clone(strings)
				slices.Reverse(strings)
			}
			for j, m := range strings {
				off := spread * float64(j) / float64(max(len(strings)-1, 1))
				p.add(x+t+off, d-off, m, 0.65)
			}
		}
	}
}

// writePad is a chord of three or four notes held while it sounds, its bottom
// between E3 and F♯4
func writePad(a *arranger, ps []*part) {
	p, rng := ps[0], a.rng
	size, bottom := 3+rng.IntN(2), min(52+rng.IntN(15), p.hi-12)
	var last []int
	for b := a.first; b < a.bars; {
		e := b + 1
		for e < a.bars && a.degrees[e] == a.degrees[b] {
			e++
		}
		v := a.voicing(a.chords[b], size, bottom, p.lo, p.hi, last, false)
		last = v
		for _, m := range v {
			p.add(a.barAt(b), float64((e-b)*a.steps), m, 0.7)
		}
		b = e
	}
}

// writeArpeggio is a chord's tones over two octaves from between C3 and E5, up,
// down, up and down, an Alberti bass or in no order, in eighths, sixteenths or
// triplets, each note let ring now and then
func writeArpeggio(a *arranger, ps []*part) {
	p, rng := ps[0], a.rng
	bottom := min(48+rng.IntN(29), p.hi-24)
	order := rng.IntN(5)
	rate := []float64{2, 1, 4.0 / 3}[pick(rng, 45, 30, 25)]
	if a.compound() {
		rate = []float64{2, 1}[pick(rng, 80, 20)]
	}
	ring := 1.0
	if rng.Float64() < 0.4 {
		ring = 2.5
	}
	var last []int
	for b := a.first; b < a.bars; b++ {
		v := a.voicing(a.chords[b], 3+rng.IntN(2), bottom, p.lo, p.hi, last, false)
		last = v
		tones := slices.Clone(v)
		for _, m := range v {
			if m+12 <= p.hi {
				tones = append(tones, m+12)
			}
		}
		var seq []int
		switch order {
		case 0:
			seq = tones
		case 1:
			seq = slices.Clone(tones)
			slices.Reverse(seq)
		case 2:
			seq = append(slices.Clone(tones), tones[1:len(tones)-1]...)
			slices.Reverse(seq[len(tones):])
		case 3:
			seq = []int{v[0], v[len(v)-1], v[1], v[len(v)-1]}
		case 4:
			seq = slices.Clone(tones)
			rng.Shuffle(len(seq), func(i, j int) { seq[i], seq[j] = seq[j], seq[i] })
		}
		for k, t := 0, 0.0; t < float64(a.steps)-1e-9; k, t = k+1, t+rate {
			p.add(a.barAt(b)+t, rate*ring, seq[k%len(seq)], 0.6)
		}
	}
}

// motifNote is a note of a motif: its onset and length in steps from the
// motif's start, and its pitch in steps of the scale over the chord's root
type motifNote struct {
	pos, dur float64
	deg      int
}

// cells are the rhythms a melody fills a pulse with, by the pulse's steps: each
// note's onset and length, an eighth's rest before a note and a triplet among them
var cells = map[int][][][2]float64{
	4: {{{0, 4}}, {{0, 2}, {2, 2}}, {{0, 3}, {3, 1}}, {{0, 1}, {1, 1}, {2, 2}}, {{0, 2}, {2, 1}, {3, 1}},
		{{0, 1}, {1, 1}, {2, 1}, {3, 1}}, {{0, 1}, {1, 3}}, {{0, 4.0 / 3}, {4.0 / 3, 4.0 / 3}, {8.0 / 3, 4.0 / 3}}, {{2, 2}}},
	6: {{{0, 6}}, {{0, 4}, {4, 2}}, {{0, 2}, {2, 2}, {4, 2}}, {{0, 2}, {2, 4}}, {{0, 3}, {3, 3}},
		{{0, 1}, {1, 1}, {2, 2}, {4, 2}}, {{0, 3}, {3, 1}, {4, 2}}, {{2, 2}, {4, 2}}},
}

var cellWeights = map[int][]float64{4: {3, 5, 2, 3, 3, 1, 1, 2, 1}, 6: {2, 4, 4, 2, 1, 2, 2, 1}}

// melodyForms are a melody's phrases, an op a motif's span: A the motif again
// over its chord, S moved up or down the scale, I upside down, R backwards, V
// varied, B the answer to it, H a held close
var melodyForms = []string{"AAVB", "ASAH", "ABAB", "AAIH", "AVSB", "ARAH", "ABAH", "AASB", "AVAB"}

// motif is a span's rhythm of cells and a contour of steps, leaps now and then,
// from one of the chord's tones, its strong pulses on the chord's tones; a fifth
// of the motifs tie a note over a pulse and a quarter strike one early
func (a *arranger) motif(span int) []motifNote {
	rng := a.rng
	var out []motifNote
	for pos, k := 0, 0; pos < span; k++ {
		n := a.feel.pulses[k%len(a.feel.pulses)]
		if k == 0 || rng.Float64() >= 0.12 {
			for _, c := range cells[n][pick(rng, cellWeights[n]...)] {
				out = append(out, motifNote{pos: float64(pos) + c[0], dur: c[1]})
			}
		}
		pos += n
	}
	if len(out) < 2 {
		out = append(out, motifNote{pos: float64(span) / 2, dur: float64(span) / 2})
	}
	if rng.Float64() < 0.2 { // tied over the pulse it ends on
		i := rng.IntN(len(out) - 1)
		out[i].dur = out[i+1].pos + out[i+1].dur - out[i].pos
		out = slices.Delete(out, i+1, i+2)
	}
	if rng.Float64() < 0.25 && len(out) > 2 { // a strong pulse struck early
		for i := 1; i < len(out); i++ {
			if out[i].pos == math.Trunc(out[i].pos) && int(out[i].pos)%a.steps != 0 && slices.Contains(a.pulseStarts(), int(out[i].pos)%a.steps) && out[i-1].dur > 1 {
				out[i].pos, out[i].dur, out[i-1].dur = out[i].pos-1, out[i].dur+1, out[i-1].dur-1
				break
			}
		}
	}
	tones := a.tones()
	deg := tones[rng.IntN(len(tones))]
	for i := range out {
		if i > 0 {
			deg += []int{1, -1, 2, -2, 3, -3, 4, 0}[pick(rng, 28, 27, 10, 10, 7, 7, 3, 8)]
			deg = min(max(deg, -5), 9)
		}
		if p := int(out[i].pos); out[i].pos == float64(p) && slices.Contains(a.pulseStarts(), p%a.steps) && !slices.Contains(tones, ((deg%len(a.iv))+len(a.iv))%len(a.iv)) {
			deg++ // a strong pulse sounds the chord
		}
		out[i].deg = deg
	}
	return out
}

// tones are the steps of the scale over a chord's root its tones lie on
func (a *arranger) tones() []int {
	if len(a.iv) == 5 {
		return []int{0, 1, 3}
	}
	return []int{0, 2, 4}
}

// writeMelody is a motif over the melody's phrase, its centre between D4 and
// D6 in three loops of four and between C6 and D♭7 in the rest: each span the
// motif again, moved with its chord, varied or answered, a form drawn; a pickup
// bar's last pulse leads into it
func writeMelody(a *arranger, ps []*part) {
	p, rng := ps[0], a.rng
	centre := 62 + rng.IntN(25)
	if rng.Float64() < 0.25 {
		centre = 84 + rng.IntN(16)
	}
	centre = min(max(centre, p.lo+7), p.hi-7)
	span := a.steps
	if span <= 8 {
		span *= 2
	}
	motif := a.motif(span)
	var answer []motifNote
	form := melodyForms[rng.IntN(len(melodyForms))]
	detached := rng.Float64() < 0.4
	units := ((a.bars-a.first)*a.steps + span - 1) / span
	for u := range units {
		op := form[u%len(form)]
		if u == units-1 && u >= len(form) {
			op = form[len(form)-1]
		}
		notes := slices.Clone(motif)
		switch op {
		case 'S':
			shift := []int{-2, -1, 1, 2}[rng.IntN(4)]
			for i := range notes {
				notes[i].deg += shift
			}
		case 'I':
			for i := range notes {
				notes[i].deg = 2*motif[0].deg - notes[i].deg
			}
		case 'R':
			for i := range notes {
				notes[i].deg = motif[len(motif)-1-i].deg
			}
		case 'V':
			for range 1 + rng.IntN(2) {
				i := rng.IntN(len(notes))
				notes[i].deg += []int{-2, -1, 1, 2}[rng.IntN(4)]
			}
		case 'B':
			if answer == nil {
				answer = a.motif(span)
				answer[len(answer)-1].deg = 0
			}
			notes = slices.Clone(answer)
		case 'H':
			notes = notes[:min(2, len(notes))]
			at := notes[len(notes)-1].pos + notes[len(notes)-1].dur
			notes = append(notes, motifNote{pos: at, dur: max(float64(span)-at-1, 2), deg: a.tones()[rng.IntN(2)]})
		}
		start := float64(a.first*a.steps + u*span)
		idx := make([]int, len(notes))
		var mean float64
		for i, n := range notes {
			idx[i] = a.chordIndex(int(start+n.pos)/a.steps) + n.deg
			mean += float64(a.pitch(idx[i])) / float64(len(notes))
		}
		shift := int(math.Round((float64(centre) - mean) / 12))
		for i, n := range notes {
			d := n.dur
			if detached {
				d *= 0.5 + 0.3*rng.Float64()
			}
			p.add(start+n.pos, d, fold(a.pitch(idx[i]+shift*len(a.iv)), p.lo, p.hi), 0.8)
		}
	}
	if a.first == 1 && len(p.notes) > 0 { // a pickup leading up or down into the first note
		into, n := a.index(p.notes[0].midi), 1+rng.IntN(3)
		dir := []int{-1, 1}[rng.IntN(2)]
		last := float64(a.steps - a.feel.pulses[len(a.feel.pulses)-1])
		for k := range n {
			t := last + float64(a.feel.pulses[len(a.feel.pulses)-1])*float64(k)/float64(n)
			m := fold(a.pitch(into+dir*(n-k)), p.lo, p.hi)
			p.notes = append(p.notes, note{at: t, dur: float64(a.feel.pulses[len(a.feel.pulses)-1]) / float64(n), midi: m, vel: 0.7})
		}
	}
}

// writeDouble plays the melody's notes for a span of the loop's bars on another
// instrument, at the unison in half the loops, an octave up in three of ten and
// down in the rest, softer
func writeDouble(a *arranger, ps []*part) {
	p, rng, mel := ps[0], a.rng, a.partOf("melody")
	if mel == nil {
		return
	}
	shift := []int{0, 12, -12}[pick(rng, 50, 30, 20)]
	from := a.first + rng.IntN(max(a.bars-a.first-1, 1))
	to := from + 1 + rng.IntN(a.bars-from)
	gain := 0.6 + 0.3*rng.Float64()
	for _, n := range mel.notes {
		if b := int(n.at) / a.steps; b >= from && b < to {
			m := n.midi + shift
			if m < p.lo || m > p.hi {
				m = n.midi
			}
			if m >= p.lo && m <= p.hi {
				p.add(n.at, n.dur, m, n.vel*gain)
			}
		}
	}
}

// writeCounter is a slow line under the melody on each chord's guide tones,
// its third and its seventh or fifth, the nearest the last, between G3 and G5
func writeCounter(a *arranger, ps []*part) {
	p, rng := ps[0], a.rng
	lo, hi := max(55, p.lo), min(79, p.hi)
	cur := (lo + hi) / 2
	every := a.steps
	if a.steps >= 12 && rng.Float64() < 0.5 {
		every = a.steps / 2
	}
	for b := a.first; b < a.bars; b++ {
		tones := a.chords[b]
		for t := 0; t < a.steps; t += every {
			best := 0
			for m := lo; m <= hi; m++ {
				if (m%12 == tones[1] || m%12 == tones[len(tones)-1]) && (best == 0 || abs(m-cur) < abs(best-cur)) {
					best = m
				}
			}
			if best > 0 {
				p.add(a.barAt(b)+float64(t), float64(every), best, 0.6)
				cur = best
			}
		}
	}
}

func abs(x int) int { return max(x, -x) }

// enter brings one part in part-way through a third of the loops and lets
// another drop out in a quarter, and stops every part but the melody's in half
// of them for a pulse or two at a bar's end in a loop in seven: a break
func (a *arranger) enter() {
	rng := a.rng
	var kinds []string
	for _, p := range a.parts {
		if !slices.Contains(kinds, p.kind.name) {
			kinds = append(kinds, p.kind.name)
		}
	}
	if a.bars-a.first < 2 {
		return
	}
	cut := func(name string, keep func(n note) bool) {
		for _, p := range a.parts {
			if p.kind.name == name || name == "melody" && p.kind.name == "double" {
				p.notes = slices.DeleteFunc(p.notes, func(n note) bool { return !keep(n) })
			}
		}
	}
	entering := -1
	if len(kinds) > 1 && rng.Float64() < 0.35 {
		at := a.barAt(a.first + 1 + rng.IntN(a.bars-a.first-1))
		entering = rng.IntN(len(kinds))
		cut(kinds[entering], func(n note) bool { return n.at >= at })
	}
	if len(kinds) > 1 && rng.Float64() < 0.25 {
		at := a.barAt(a.first + 1 + rng.IntN(a.bars-a.first-1))
		k := rng.IntN(len(kinds))
		if k == entering { // a part that comes in does not drop out
			k = (k + 1) % len(kinds)
		}
		cut(kinds[k], func(n note) bool { return n.at < at })
	}
	if rng.Float64() < 0.15 {
		b := a.first + rng.IntN(a.bars-a.first-1)
		pulses := 1 + rng.IntN(min(2, len(a.feel.pulses)-1))
		to := a.barAt(b + 1)
		from := to - float64(slices.Max(a.feel.pulses)*pulses)
		keepMelody := rng.Float64() < 0.5
		for _, p := range a.parts {
			if keepMelody && (p.kind.name == "melody" || p.kind.name == "double") {
				continue
			}
			p.notes = slices.DeleteFunc(p.notes, func(n note) bool { return n.at >= from && n.at < to })
			for i := range p.notes {
				if n := &p.notes[i]; n.at < from && n.at+n.dur > from {
					n.dur = from - n.at
				}
			}
		}
	}
}

// swung is a position in steps as a swung loop plays it: each beat's first
// eighth, or sixteenth, takes swing of the pair, a position between its grid's
// moved in proportion; one off the grid, a triplet's, as it is
func (a *arranger) swung(x float64) float64 {
	if a.swing == 0.5 || x != math.Trunc(x) {
		return x
	}
	pair, r := 4.0, a.swing
	if a.swing16 {
		pair = 2
	}
	base := math.Floor(x/pair) * pair
	y := (x - base) / pair
	if y < 0.5 {
		return base + pair*r*y*2
	}
	return base + pair*(r+(1-r)*(y-0.5)*2)
}

// perform plays each part as a player would: swung where the loop swings, its
// onsets off the grid by a feel of its own and a jitter of its kind's, its
// velocities shaped over the loop, swelling, fading or both, the strong pulses
// accented, each note a little apart; a kit's own strikes from the kit's
// stream after the rest, so the rest play as they would without it
func (a *arranger) perform() {
	starts := a.pulseStarts()
	for _, p := range a.parts {
		if !p.kit {
			a.play(p, a.rng, starts)
		}
	}
	for _, p := range a.parts {
		if p.kit {
			a.play(p, a.krng, starts)
		}
	}
}

// play is perform's playing of one part, its draws from rng
func (a *arranger) play(p *part, rng *rand.Rand, starts []int) {
	k := p.kind
	feel := min(max(rng.NormFloat64()*6, -15), 15)
	if p.instr == audio.InstrKick {
		feel = 0 // the kick anchors the grid
	}
	gain := k.level[0] + (k.level[1]-k.level[0])*rng.Float64()
	shape, depth := pick(rng, 40, 20, 20, 20), 4+8*rng.Float64()
	for i := range p.notes {
		n := &p.notes[i]
		on, off := a.swung(n.at), a.swung(n.at+n.dur)
		t := (n.at - a.from) / max(a.to-a.from, 1)
		db := []float64{0, -depth * (1 - t), -depth * t, -depth * math.Abs(2*t-1)}[shape]
		if y := n.at - math.Floor(n.at/float64(a.steps))*float64(a.steps); !p.instr.IsDrum() {
			switch {
			case y == 0:
				db += 3
			case y == math.Trunc(y) && slices.Contains(starts, int(y)):
				db += 1
			default:
				db -= 1.5
			}
		}
		db += rng.NormFloat64() * 1.2
		n.vel = min(max(n.vel*gain*math.Pow(10, db/20), softest), 1)
		n.at, n.dur = on+a.ms(feel+rng.NormFloat64()*k.jitter), off-on
	}
}

// glide bends a fifth of a line's notes into the note that follows them at
// once, over their last 60 to 250 ms, and scoops a seventh, written a semitone
// or two under and rising to their pitch: on the engine's voices alone
func (a *arranger) glide(rng *rand.Rand) {
	step := float64(audio.SamplesPerStep(a.bpm)) / audio.AudioSampleRate
	for _, p := range a.parts {
		if p.sample != nil || p.instr.IsDrum() || !slices.Contains([]string{"melody", "double", "counter", "bass"}, p.kind.name) {
			continue
		}
		into := -1
		for i := range p.notes {
			e := &p.notes[i]
			dur := float64(max(int(math.Round(e.dur)), 1))
			switch u := rng.Float64(); {
			case u < 0.2:
				if i+1 == len(p.notes) || math.Abs(p.notes[i+1].at-(e.at+e.dur)) > 0.25 {
					continue
				}
				if d := p.notes[i+1].midi - e.midi; d != 0 && abs(d) <= glideReach {
					e.bend, e.bendAt, into = float64(d), max(1-(0.06+0.19*rng.Float64())/(dur*step), 0), i+1
				}
			case u < 0.2+1.0/7 && i != into:
				under := 1 + rng.IntN(2)
				if e.midi-under >= p.lo {
					e.midi, e.bend, e.bendAt = e.midi-under, float64(under), 0
				}
			}
		}
	}
}

// glideReach is the widest step in semitones a note bends into: a fifth
const glideReach = 7

// softest is the softest velocity the arranger plays a note at, an overlap's
// single notes 30 dB under full at the softest
const softest = 0.03

// arrange writes a loop: its texture, each part's instrument and notes, played;
// steady keeps its tempo, else it drifts beat by beat in seven loops of ten
func (s *Session) arrange(a *arranger, set *sampleSet, srng, glrng, krng *rand.Rand, steady bool) (*music, error) {
	if err := a.compose(set, srng, glrng, krng); err != nil {
		return nil, err
	}
	return a.music(steady)
}

// compose draws a loop's texture and writes each part's instrument and notes,
// played: all of a loop but its patterns. With krng a loop with drums plays
// them on a kit drawn from it (kit.go), the rest as they would without it.
func (a *arranger) compose(set *sampleSet, srng, glrng, krng *rand.Rand) error {
	rng := a.rng
	weights := make([]float64, len(textures))
	for i, t := range textures {
		weights[i] = t.weight
	}
	names := textures[pick(rng, weights...)].parts
	if krng != nil && slices.Contains(names, "drums") {
		k, err := newKit(krng)
		if err != nil {
			return err
		}
		a.kit, a.krng = k, krng
	}
	for i := range partKinds {
		k := &partKinds[i]
		if !slices.Contains(names, k.name) {
			continue
		}
		if k.name == "drums" {
			for _, d := range []audio.InstrumentType{audio.InstrKick, audio.InstrSnare, audio.InstrHihat} {
				a.parts = append(a.parts, &part{kind: k, instr: d})
			}
			continue
		}
		p := &part{kind: k}
		switch k.name {
		case "left":
			c := a.partOf("chords")
			p.instr, p.sample, p.lo, p.hi = c.instr, c.sample, c.lo, c.hi
		default:
			a.instrument(p, set, srng)
		}
		a.parts = append(a.parts, p)
	}
	for i := range partKinds {
		var ps []*part
		for _, p := range a.parts {
			if p.kind == &partKinds[i] {
				ps = append(ps, p)
			}
		}
		if len(ps) > 0 {
			partKinds[i].write(a, ps)
		}
	}
	a.enter()
	for _, p := range a.parts {
		p.notes = slices.DeleteFunc(p.notes, func(n note) bool { return n.at < a.from || n.at >= a.to })
	}
	if a.kit != nil {
		a.strike()
	}
	a.perform()
	for _, p := range a.parts { // a note written in the loop and played early starts it
		for i := range p.notes {
			if n := &p.notes[i]; n.at < a.from {
				n.at, n.dur = a.from, n.dur-(a.from-n.at)
			}
		}
		p.notes = slices.DeleteFunc(p.notes, func(n note) bool { return n.at >= a.to })
		slices.SortStableFunc(p.notes, func(x, y note) int { return cmp.Compare(x.at, y.at) })
	}
	if glrng != nil {
		a.glide(glrng)
	}
	if a.kit != nil {
		a.deal()
	}
	if a.kit != nil { // dealt: a loop whose drums strike nothing keeps none
		return a.kit.placed(a.parts)
	}
	return nil
}

// instrument draws a part's instrument by its kind's families, a voice no
// family names at a weight of 1: among the engine's voices, or at the set's
// share the recorded ones, a bass's among the set's basses; a double plays on
// another than the melody's
func (a *arranger) instrument(p *part, set *sampleSet, srng *rand.Rand) {
	k, rng := p.kind, a.rng
	mel := a.partOf("melody")
	other := func(i audio.InstrumentType, in *sampleInstr) bool {
		return k.name != "double" || mel == nil || mel.instr != i || mel.sample != in
	}
	tonal := audio.TonalInstruments()
	w := make([]float64, len(tonal))
	for i, t := range tonal {
		if other(t, nil) {
			if w[i] = k.families[familyOf(t)]; familyOf(t) == "unknown" {
				w[i] = 1 // a voice a session added plays any part
			}
		}
	}
	if slices.Max(w) == 0 {
		for i, t := range tonal {
			if t != audio.InstrBass && other(t, nil) {
				w[i] = 1
			}
		}
	}
	p.instr = tonal[pick(rng, w...)]
	r := familyKeys(audio.InstrumentName(p.instr))
	p.lo, p.hi = r[0], r[1]
	if set != nil && srng != nil && srng.Float64() < set.share {
		pool := set.other
		if k.name == "bass" {
			pool = set.bass
		}
		ws := make([]float64, len(pool))
		for i, in := range pool {
			if mel == nil || k.name != "double" || mel.sample != in {
				ws[i] = k.families[audio.RipFamilies[max(in.family, 0)]] + 0.5
			}
		}
		if len(pool) > 0 && slices.Max(ws) > 0 {
			in := pool[pick(srng, ws...)]
			p.sample = in // the engine's instrument drawn plays it if the recordings cannot
			p.lo, p.hi = max(in.pitches[0]-sampledReach, 21), min(in.pitches[len(in.pitches)-1]+sampledReach, 108)
		}
	}
}

// familyOf is the family an engine's instrument plays as by its name, unknown
// for one voiceFamilies does not name
func familyOf(i audio.InstrumentType) string {
	if f, ok := voiceFamilies[audio.InstrumentName(i)]; ok {
		return f
	}
	return "unknown"
}

// heads is each bar's meter and its beats' tempos: steady, a walk of a beat at
// a time within 6% of bpm, or slowing or quickening to 3 to 8% off it by the end
func (a *arranger) heads(steady bool) []audio.BarHead {
	rng := a.rng
	heads := make([]audio.BarHead, a.bars)
	mode := 0
	if !steady {
		mode = pick(rng, 30, 40, 15, 15)
	}
	depth := 0.03 + 0.05*rng.Float64()
	total, d := a.bars*a.m.Beats(), 0.0
	for b := range heads {
		heads[b] = audio.BarHead{Meter: a.m, BPM: a.bpm}
		if mode == 0 {
			continue
		}
		heads[b].Beats = make([]int, a.m.Beats())
		for k := range heads[b].Beats {
			x := float64(b*a.m.Beats()+k) / float64(max(total-1, 1))
			switch mode {
			case 1:
				d = min(max(0.85*d+rng.NormFloat64()*0.006, -0.06), 0.06)
			case 2:
				d = -depth * x
			case 3:
				d = depth * x
			}
			heads[b].Beats[k] = min(max(int(math.Round(float64(a.bpm)*(1+d))), audio.MinBPM), audio.MaxBPM)
		}
	}
	return heads
}

// music lays the parts out as a pattern a slot a bar, named corpus_<slot>_<bar>
// and so written over by the next loop's, each note at its step and the share
// of a step after it; past the ring a step's late onsets wait in, one sounds on
// the step
func (a *arranger) music(steady bool) (*music, error) {
	m := &music{kind: "arranged", root: a.root, scale: a.scale, prog: a.degrees, chords: a.chords, kit: a.kit}
	m.heads = a.heads(steady)
	for _, p := range a.parts {
		m.parts[p.kind.slot] = append(m.parts[p.kind.slot], p)
	}
	m.timeline = make([][audio.MusicSlots]audio.PatternID, a.bars)
	for slot, ps := range m.parts {
		if len(ps) == 0 {
			continue
		}
		pats := make([]*audio.Pattern, a.bars)
		for b := range pats {
			pats[b] = &audio.Pattern{Name: fmt.Sprintf("corpus_%d_%d", slot, b), Steps: a.steps, Meter: a.m, Role: audio.Role(slot + 1), Tracks: make([]audio.Track, len(ps))}
			for t, p := range ps {
				pats[b].Tracks[t].Instr = p.instr
			}
		}
		late := map[[2]int]int{}
		for t, p := range ps {
			for _, n := range p.notes {
				g := math.Floor(n.at)
				st := audio.Step{Vel: n.vel, Late: n.at - g, Dur: min(max(int(math.Round(n.dur)), 1), audio.MaxPatternLen), Bend: n.bend, BendAt: n.bendAt}
				if st.Late >= 1 {
					g, st.Late = g+1, 0
				}
				b, pos := int(g)/a.steps, int(g)%a.steps
				if b >= a.bars {
					continue
				}
				if st.Late > 0 {
					if late[[2]int{b, pos}] >= audio.MaxPendingTrigs {
						st.Late = 0
					} else {
						late[[2]int{b, pos}]++
					}
				}
				st.Pos = pos
				if !p.instr.IsDrum() {
					st.Deg, st.Oct, st.Acc = audio.PitchStep(a.root, a.scale, n.midi)
				}
				pats[b].Tracks[t].Events = append(pats[b].Tracks[t].Events, st)
			}
		}
		for b, pat := range pats {
			if err := audio.ValidatePattern(pat); err != nil {
				return nil, err
			}
			m.timeline[b][slot] = audio.RegisterPattern(pat)
		}
	}
	starts := audio.BeatStarts(m.heads)
	sample := func(x float64) int {
		beat := a.m.Beat()
		k := min(int(x)/beat, len(starts)-1)
		if k == len(starts)-1 {
			return starts[k]
		}
		tempo := m.heads[k/a.m.Beats()].Tempo(k % a.m.Beats())
		return starts[k] + int((x-float64(k*beat))*float64(audio.SamplesPerStep(tempo)))
	}
	m.from, m.to = sample(a.from), sample(a.to)
	return m, nil
}

// overlapShare is the share of a corpus's loops overlaps gives single notes
// when it names none
const overlapShare = 0.25

// overlap is a loop of no arrangement: single notes of one to eight of the
// tonal instruments, the engine's voices and the recorded ones alike, each
// pitch even over its instrument's own, 4 to 9 s of onsets off any grid, as
// doc/rip-training.md's overlaps say
func (s *Session) overlap(rng *rand.Rand, set *sampleSet) (*music, error) {
	type inst struct {
		instr  audio.InstrumentType
		sample *sampleInstr
		lo, hi int
	}
	var pool []inst
	for _, t := range audio.TonalInstruments() {
		r := familyKeys(audio.InstrumentName(t))
		pool = append(pool, inst{t, nil, r[0], r[1]})
	}
	if set != nil {
		for _, in := range append(slices.Clip(set.bass), set.other...) {
			pool = append(pool, inst{audio.InstrPiano, in, max(in.pitches[0]-sampledReach, 21), min(in.pitches[len(in.pitches)-1]+sampledReach, 108)})
		}
	}
	rng.Shuffle(len(pool), func(i, j int) { pool[i], pool[j] = pool[j], pool[i] })
	k := min(1+pick(rng, 30, 20, 15, 12, 8, 6, 5, 4), len(pool))
	const bpm = 120
	sps := audio.SamplesPerStep(bpm)
	n := int((4 + 5*rng.Float64()) * audio.AudioSampleRate)
	bars := (n + 16*sps - 1) / (16 * sps)
	m := &music{kind: "overlap", root: 0, scale: audio.ScaleMajor, to: n}
	type sounding struct {
		at, end, midi int
		p             *part
	}
	var ring []sounding
	parts := make([]*part, k)
	for i := range parts {
		parts[i] = &part{kind: &overlapKind, instr: pool[i].instr, sample: pool[i].sample, lo: pool[i].lo, hi: pool[i].hi}
		m.parts[1+i%3] = append(m.parts[1+i%3], parts[i])
	}
	type onset struct {
		at int
		p  *part
	}
	var onsets []onset
	for _, p := range parts {
		gap := (0.15 + 1.35*rng.Float64()) * audio.AudioSampleRate
		for at := int(rng.Float64() * gap); at < n; at += int(rng.ExpFloat64()*gap) + 1 {
			onsets = append(onsets, onset{at, p})
		}
	}
	slices.SortStableFunc(onsets, func(x, y onset) int { return cmp.Compare(x.at, y.at) })
	related := []int{0, 12, -12, 7, -7, 3, -3, 4, -4}
	for _, o := range onsets {
		p, at := o.p, o.at
		ring = slices.DeleteFunc(ring, func(s sounding) bool { return s.end <= at })
		if len(ring) >= 8 {
			continue
		}
		steps := max(int(math.Round(math.Exp(math.Log(0.12)+rng.Float64()*math.Log(2.5/0.12))*audio.AudioSampleRate/float64(sps))), 1)
		end := at + steps*sps // as long as the step it is held for sounds
		midi := p.lo + rng.IntN(p.hi-p.lo+1)
		own := slices.IndexFunc(ring, func(s sounding) bool { return s.p == p })
		switch u := rng.Float64(); {
		case u < 0.2 && own >= 0:
			midi = ring[own].midi // struck again while it rings
		case u < 0.8: // the related pitch nearest, within a tone either way, of a note that sounds
			best, near := midi, 3
			for _, s := range ring {
				for _, r := range related {
					if m := s.midi + r; m >= p.lo && m <= p.hi && abs(m-midi) < near {
						best, near = m, abs(m-midi)
					}
				}
			}
			midi = best
		}
		ring = append(ring, sounding{at, end, midi, p})
		p.notes = append(p.notes, note{at: float64(at), dur: float64(end - at), midi: midi, vel: math.Pow(10, -30*rng.Float64()/20)})
	}
	m.heads = make([]audio.BarHead, bars)
	m.timeline = make([][audio.MusicSlots]audio.PatternID, bars)
	for b := range m.heads {
		m.heads[b] = audio.BarHead{BPM: bpm}
	}
	for slot, ps := range m.parts {
		if len(ps) == 0 {
			continue
		}
		pats := make([]*audio.Pattern, bars)
		for b := range pats {
			pats[b] = &audio.Pattern{Name: fmt.Sprintf("corpus_%d_%d", slot, b), Steps: 16, Role: audio.Role(slot + 1), Tracks: make([]audio.Track, len(ps))}
			for t, p := range ps {
				pats[b].Tracks[t].Instr = p.instr
			}
		}
		for t, p := range ps {
			for _, x := range p.notes {
				at := int(x.at)
				g := at / sps
				st := audio.Step{Pos: g % 16, Vel: x.vel, Late: (float64(at-g*sps) + 0.5) / float64(sps), Dur: min(max(int(math.Round(x.dur/float64(sps))), 1), audio.MaxPatternLen)}
				st.Deg, st.Oct, st.Acc = audio.PitchStep(audio.DefaultRootNote, m.scale, x.midi)
				pats[g/16].Tracks[t].Events = append(pats[g/16].Tracks[t].Events, st)
			}
		}
		for b, pat := range pats {
			if err := audio.ValidatePattern(pat); err != nil {
				return nil, err
			}
			m.timeline[b][slot] = audio.RegisterPattern(pat)
		}
	}
	m.root = audio.DefaultRootNote
	return m, nil
}

// overlapKind is the kind of an overlap's parts: a note on its own
var overlapKind = partKind{name: "single", level: [2]float64{1, 1}}
