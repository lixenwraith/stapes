package stapes

import (
	"cmp"
	"fmt"
	"math"
	"slices"
	"strings"
	"sync"

	"github.com/lixenwraith/audio"
)

// labelHead is a head the decoders read, by the word that names it, the
// fields of the frames a label's planes fill in its place, and whether a label
// states it, as the trainer asks it something there; nil where every label does
type labelHead struct {
	word   string
	fill   func(f *audio.RipFrames, l *labelPlanes)
	stated func(l *corpusLoop) bool
}

// labelHeadRows are the heads the decoders read: the frame model's onset,
// drums, voice genes, beat and downbeat, and the notes model's pitch, strike,
// release, velocity, cents, onset (start) and what plays each pitch (plays,
// its three heads and its measure), the frame model's four where no notes
// model is loaded. The frame model's wave and family and either's voicing no
// decoder reads.
var labelHeadRows = []labelHead{
	{"onset", func(f *audio.RipFrames, l *labelPlanes) { f.Flux = l.onset }, nil},
	{"drums", func(f *audio.RipFrames, l *labelPlanes) { f.Onset = l.drums }, func(l *corpusLoop) bool { return len(l.Kit) == 0 }},
	{"voice", func(f *audio.RipFrames, l *labelPlanes) { f.Voice = l.voice }, func(l *corpusLoop) bool {
		return l.anyNote(func(e corpusEvent) bool { return len(e.Voice) > 0 && e.Wave >= 0 })
	}},
	{"beat", func(f *audio.RipFrames, l *labelPlanes) { f.Beat = l.beat }, func(l *corpusLoop) bool { return len(l.Beats) > 0 }},
	{"downbeat", func(f *audio.RipFrames, l *labelPlanes) { f.Downbeat = l.downbeat }, func(l *corpusLoop) bool { return len(l.Downbeats) > 0 }},
	{"pitch", func(f *audio.RipFrames, l *labelPlanes) { // its cells A440's, its tuning its cents', as a model's
		f.Salience, f.Chroma, f.BassChroma, f.Tuning = l.pitch, l.chroma, l.bassChroma, 0
	}, nil},
	{"strike", func(f *audio.RipFrames, l *labelPlanes) { f.Strike = l.strike }, nil},
	{"release", func(f *audio.RipFrames, l *labelPlanes) { f.Release = l.release }, nil},
	{"velocity", func(f *audio.RipFrames, l *labelPlanes) { f.Velocity = l.velocity }, func(l *corpusLoop) bool {
		return l.anyNote(func(e corpusEvent) bool { return e.Vel >= 0 })
	}},
	{"cents", func(f *audio.RipFrames, l *labelPlanes) { f.Cents, f.CentsError = l.cents, l.centsError }, func(l *corpusLoop) bool {
		return l.anyNote(func(e corpusEvent) bool {
			return len(e.Voice) > 0 || slices.ContainsFunc(e.Cents, func(c float64) bool { return !math.IsNaN(c) })
		})
	}},
	{"start", func(f *audio.RipFrames, l *labelPlanes) { f.Start = l.start }, nil},
	{"plays", func(f *audio.RipFrames, l *labelPlanes) { f.VoiceAt, f.Voices, f.Sure = l.voiceAt, l.voices, labelSure }, func(l *corpusLoop) bool {
		unknown := slices.Index(audio.RipFamilies[:], "unknown")
		return l.anyNote(func(e corpusEvent) bool { return len(e.Voice) > 0 || e.Family >= 0 && e.Family != unknown })
	}},
}

// anyNote is whether any of a label's notes, no hit or kit's strike, is so
func (l *corpusLoop) anyNote(so func(corpusEvent) bool) bool {
	return slices.ContainsFunc(l.Events, func(e corpusEvent) bool { return e.Drum < 0 && e.MIDI > 0 && so(e) })
}

// labelSure is the measure a label's say names a family by: a cell's say is a
// name whole or every family's even share, so anything between tells them apart
const labelSure = 0.5

// labelUnsure is the error in cents a label states where it places no sound: a
// cell's, so the reading names no frequency; labelCentsGrain where it places
// one, what the trainer's cents head states at a miss of none (CENTS_GRAIN)
const labelUnsure, labelCentsGrain = 100, 0.1

// parseHeads reads heads' word, the heads a label gives by name, comma
// separated, or all for every one
func parseHeads(w string) ([]string, error) {
	var out []string
	for _, h := range strings.Split(w, ",") {
		switch {
		case h == "all":
			for _, r := range labelHeadRows {
				out = append(out, r.word)
			}
		case slices.ContainsFunc(labelHeadRows, func(r labelHead) bool { return r.word == h }):
			out = append(out, h)
		default:
			words := make([]string, len(labelHeadRows))
			for i, r := range labelHeadRows {
				words[i] = r.word
			}
			return nil, fmt.Errorf("heads %q: want all or heads of %s, comma separated", w, strings.Join(words, ", "))
		}
	}
	slices.Sort(out)
	return slices.Compact(out), nil
}

// labelHeads hears through fe, the analysis where nil, with the heads named
// that its label states written as the trainer's targets (planes), those it
// does not state (kept) the front end's; given every head it hears nothing,
// its frames the label's alone, as a front end of the models' range that
// states no tuning
type labelHeads struct {
	fe          audio.RipFrontEnd
	lab         *corpusLoop
	heads, kept []string
	once        sync.Once
	l           *labelPlanes // built once, for the frames the first hearing gives
}

// newLabelHeads is fe with the heads named its label's where it states them
func newLabelHeads(fe audio.RipFrontEnd, lab *corpusLoop, named []string) *labelHeads {
	h := &labelHeads{fe: fe, lab: lab}
	for _, r := range labelHeadRows {
		switch {
		case !slices.Contains(named, r.word):
		case r.stated == nil || r.stated(lab):
			h.heads = append(h.heads, r.word)
		default:
			h.kept = append(h.kept, r.word)
		}
	}
	return h
}

// every is whether the label gives every head the decoders read
func (h *labelHeads) every() bool { return len(h.heads) == len(labelHeadRows) }

// says is what a rip through h was read with: the label's heads, and those it
// does not state, kept the front end's
func (h *labelHeads) says() string {
	if h.every() {
		return "every head its label's, no model heard"
	}
	with := "the label's " + strings.Join(h.heads, ", ") + " as its heads, the rest its front end's"
	if len(h.kept) > 0 {
		with += ", its " + strings.Join(h.kept, ", ") + " too, which the label does not state"
	}
	return with
}

func (h *labelHeads) Hear(x []float64, rate int, p audio.RipParams) (*audio.RipFrames, error) {
	g := audio.RipFrames{Hop: audio.RipFrameSec, Range: audio.RipPiano}
	n := h.lab.Frames
	if n == 0 && rate > 0 { // a label written by hand: the recording's frames
		n = (len(x)*audio.RipRate/rate + audio.RipHop - 1) / audio.RipHop
	}
	if !h.every() {
		f, err := audio.Ripper{Params: p, FrontEnd: h.fe}.Hear(x, rate)
		if err != nil {
			return nil, err
		}
		g, n = *f, len(f.Flux) // the frames a heardOnce keeps stay its front end's
		if g.Range != audio.RipPiano {
			g.Range = audio.RipPiano
			pianoWide(&g, f.Pitches())
		}
	}
	h.once.Do(func() { h.l = h.lab.planes(n) })
	for _, r := range labelHeadRows {
		if slices.Contains(h.heads, r.word) {
			r.fill(&g, h.l)
		}
	}
	return &g, nil
}

// pianoWide lays a front end's planes of a value a pitch, heard over rg, on the
// piano's keys, silent where it hears none, beside a label's
func pianoWide(f *audio.RipFrames, rg audio.RipRange) {
	n, k := len(f.Flux), audio.RipPiano.Len()
	for _, plane := range []*[]float32{&f.Salience, &f.Strike, &f.Release, &f.Velocity, &f.Cents, &f.CentsError} {
		if *plane == nil {
			continue
		}
		wide := make([]float32, n*k)
		for t := range n {
			copy(wide[t*k+rg.Min-audio.RipPiano.Min:], (*plane)[t*rg.Len():(t+1)*rg.Len()])
		}
		*plane = wide
	}
	f.VoiceAt, f.Voices = nil, nil // cells of the narrower range; no front end of one tells them
}

// labelPlanes are a label's events as the heads the decoders read would say
// them, each shaped as tool/rip-train's target of it (Label.window, targets):
// a mark a frame either side, a note's cells over its frames, a value where the
// trainer asks one and its middle where it asks none
type labelPlanes struct {
	onset, start, beat, downbeat     []float64
	drums                            [audio.InstrClap][]float64
	pitch, strike, release, velocity []float32
	cents, centsError, voice         []float32
	chroma, bassChroma               [][12]float64
	voiceAt, voices                  []uint8
}

// planes writes a label's events over n frames, in the label's order, as the
// trainer does: a placed start is marked where it is placed, where the trainer
// asks a rise within its reach
func (l *corpusLoop) planes(n int) *labelPlanes {
	rg, genes := audio.RipPiano, audio.VoiceGeneCount
	k := rg.Len()
	out := &labelPlanes{onset: make([]float64, n), start: make([]float64, n), beat: make([]float64, n), downbeat: make([]float64, n),
		pitch: make([]float32, n*k), strike: make([]float32, n*k), release: make([]float32, n*k), velocity: make([]float32, n*k),
		cents: make([]float32, n*k), centsError: make([]float32, n*k), voice: make([]float32, n*genes),
		chroma: make([][12]float64, n), bassChroma: make([][12]float64, n)}
	for d := range out.drums {
		out.drums[d] = make([]float64, n)
	}
	mark := func(plane []float64, t int) {
		for u := max(t-1, 0); u < min(t+2, n); u++ {
			plane[u] = 1
		}
	}
	for i := range out.velocity {
		out.velocity[i], out.centsError[i] = 0.5, labelUnsure
	}
	for i := range out.voice {
		out.voice[i] = 0.5
	}
	near := make([]float32, n*k) // how far from a cell's centre the sound it reports sits
	for i := range near {
		near[i] = float32(math.Inf(1))
	}
	sole := slices.Repeat([]int{-1}, n*k) // the event sounding in a cell, -2 where a second does
	top := slices.Repeat([]int{-1}, n)    // the highest note sounding with genes, whose the frame's voice is
	unknown := slices.Index(audio.RipFamilies[:], "unknown")
	for i, e := range l.Events {
		t := e.Frame
		if t < 0 || t >= n {
			continue
		}
		mark(out.onset, t)
		mark(out.start, t)
		if e.Drum >= 0 || e.MIDI == 0 {
			if e.Drum >= 0 && e.Drum < len(out.drums) && len(l.Kit) == 0 {
				mark(out.drums[e.Drum], t)
			}
			continue
		}
		p, end := e.MIDI-rg.Min, min(max(e.End, t+1), n)
		engine := len(e.Voice) > 0 // an engine voice plays where the label says; a recording only where it lists its cents
		// off is where the sound sits at frame u, in cells from the range's first, and whether the label places it there
		off := func(u int) (float64, bool) {
			if len(e.Cents) == 0 {
				return float64(p) + l.Tuning/100, engine
			}
			if c := u - t; c < len(e.Cents) && !math.IsNaN(e.Cents[c]) {
				return float64(p) + e.Cents[c]/100, true
			}
			return float64(p), false
		}
		cell := func(u int) int { o, _ := off(u); return int(math.RoundToEven(o)) }
		if len(e.Cents) == 0 {
			cell = func(int) int { return p }
		}
		for u := t; u < end; u++ {
			if c := cell(u); c >= 0 && c < k {
				at := u*k + c
				out.pitch[at] = 1
				if sole[at] == -1 {
					sole[at] = i
				} else {
					sole[at] = -2
				}
			}
			o, placed := off(u)
			if !placed {
				if q := u*k + p; p >= 0 && p < k {
					near[q], out.cents[q], out.centsError[q] = 0, 0, labelUnsure
				}
				continue
			}
			for b := int(math.RoundToEven(o)) - 1; b <= int(math.RoundToEven(o))+1; b++ {
				if d := o - float64(b); math.Abs(d) < 1 && b >= 0 && b < k && float32(math.Abs(d)) < near[u*k+b] {
					near[u*k+b], out.cents[u*k+b], out.centsError[u*k+b] = float32(math.Abs(d)), float32(100*d), labelCentsGrain
				}
			}
		}
		if first := cell(t); first >= 0 && first < k {
			for u := max(t-1, 0); u < min(t+2, n); u++ {
				out.strike[u*k+first] = 1
				if e.Vel >= 0 {
					out.velocity[u*k+first] = float32(e.Vel)
				}
			}
		}
		if last := cell(end - 1); last >= 0 && last < k && e.End < n {
			for u := max(e.End-1, 0); u < min(e.End+2, n); u++ {
				out.release[u*k+last] = 1
			}
		}
		if engine && e.Wave >= 0 {
			for u := t; u < end; u++ {
				if top[u] < 0 || e.MIDI > l.Events[top[u]].MIDI {
					top[u] = i
				}
			}
		}
	}
	for u, i := range top {
		if i >= 0 {
			for g, v := range l.Events[i].Voice[:min(len(l.Events[i].Voice), genes)] {
				out.voice[u*genes+g] = float32(v)
			}
		}
	}
	for _, marks := range []struct {
		at    []int
		plane []float64
	}{{l.Beats, out.beat}, {l.Downbeats, out.downbeat}} {
		for _, t := range marks.at {
			if t >= -1 && t <= n {
				mark(marks.plane, t)
			}
		}
	}
	names, waves := len(audio.RipFamilies)-1, len(audio.VoiceWaves)
	cells := make([]int, k)
	for u := range n {
		row := out.pitch[u*k : (u+1)*k]
		for c, v := range row {
			out.chroma[u][(rg.Min+c)%12] += float64(v)
			if rg.Min+c < audio.RipBassTop {
				out.bassChroma[u][(rg.Min+c)%12] += float64(v)
			}
			cells[c] = c
		}
		slices.SortStableFunc(cells, func(a, b int) int { return cmp.Compare(row[b], row[a]) })
		for _, c := range cells[:audio.RipVoiceCells] {
			out.voiceAt = append(out.voiceAt, uint8(c))
			e := corpusEvent{Wave: -1, Family: -1}
			if i := sole[u*k+c]; i >= 0 {
				e = l.Events[i]
			}
			name, wave := e.Family, -1
			if len(e.Voice) > 0 || name == unknown {
				name = -1 // an engine's note asks no family, and unknown is no name
			}
			if len(e.Voice) > 0 {
				wave = e.Wave
			}
			out.voices = appendSay(out.voices, name, names)
			out.voices = appendSay(out.voices, wave, waves)
			for g := range genes {
				v := 0.5
				if len(e.Voice) > 0 {
					v = e.Voice[g]
				}
				out.voices = append(out.voices, uint8(math.Round(255*v)))
			}
		}
	}
	return out
}

// appendSay appends a head's shares of n values as bytes of 255 a whole: one
// whole where the label names k, an even share each where it names none
func appendSay(to []uint8, k, n int) []uint8 {
	for j := range n {
		switch {
		case k < 0 || k >= n:
			to = append(to, uint8(math.Round(255/float64(n))))
		case j == k:
			to = append(to, 255)
		default:
			to = append(to, 0)
		}
	}
	return to
}
