package stapes

import (
	"cmp"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"

	"github.com/lixenwraith/audio"
)

// The sampler plays a corpus loop's tonal parts on NSynth's recorded notes: 4 s
// of 16 kHz mono each, held for 3 s and released over the last.
const (
	nsynthRate    = 16000
	nsynthHold    = 3.0   // seconds a recorded note is held before its release
	sampledReach  = 3     // semitones a recorded pitch is shifted at most
	sampledDraws  = 8     // instruments drawn for a bank part before its engine voice plays it
	sampledFade   = 0.060 // seconds a cut note crossfades into its sample's release, near a damper's fall
	sampledFloor  = -50   // dBFS a note's loudest held window must pass, or its noise would play as the note
	sampledTop    = 7500  // Hz, the shift's low-pass, under the recordings' 8 kHz
	sampledHalf   = 29    // the low-pass's half width in source samples: 60 dB by 8 kHz
	sampledPhases = 512   // the low-pass's table steps a source sample
	sampledWindow = 0.050 // seconds a note's level is measured over, its loudest held window
	sampledSpeak  = 6     // dB under its loudest window a note has spoken by
)

// sampleNote is one recorded note on disk, its velocity layer
type sampleNote struct {
	file string
	vel  int
}

// sampleInstr is one recorded instrument, its notes by pitch, the layers in order
type sampleInstr struct {
	name    string
	family  int // its index in audio.RipFamilies, by nsynthFamilies
	pitches []int
	notes   map[int][]sampleNote
}

// nsynthFamilies is the family each of NSynth's instrument families plays as: a
// mallet and an organ are of none of the others
var nsynthFamilies = map[string]string{
	"bass": "bass", "brass": "winds", "flute": "winds", "guitar": "guitar", "keyboard": "piano", "mallet": "unknown",
	"organ": "unknown", "reed": "winds", "string": "strings", "synth_lead": "synth", "vocal": "voice",
}

// sampleSet indexes a directory of NSynth notes, named
// family_source_NNN-pitch-velocity.wav, by instrument, the bass family's apart;
// share is the chance a loop's tonal part plays on them. What it has read of a
// note is kept for the corpus, a label's draws reading each file once.
type sampleSet struct {
	dir         string
	bass, other []*sampleInstr
	share       float64
	mu          sync.Mutex
	read        map[string]heard
}

// heard is what a set has read of a recorded note: its samples, its loudest
// held window's RMS, 0 for one that does not sound or does not decode as
// NSynth's, and the samples it takes to speak
type heard struct {
	n, speak int
	level    float64
}

func loadSampleSet(dir string) (*sampleSet, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	set := &sampleSet{dir: dir, share: 0.5, read: map[string]heard{}}
	byName := map[string]*sampleInstr{}
	for _, e := range entries {
		base, ok := strings.CutSuffix(e.Name(), ".wav")
		if !ok || e.IsDir() {
			continue
		}
		// from the right: a family's name can hold an underscore (synth_lead)
		rest, v, ok1 := cutLast(base, "-")
		name, p, ok2 := cutLast(rest, "-")
		family, _, ok3 := cutLast(name, "_")
		family, _, ok4 := cutLast(family, "_")
		pitch, err1 := strconv.Atoi(p)
		vel, err2 := strconv.Atoi(v)
		if !ok1 || !ok2 || !ok3 || !ok4 || err1 != nil || err2 != nil {
			continue
		}
		in := byName[name]
		if in == nil {
			in = &sampleInstr{name: name, family: familyIndex(nsynthFamilies[family]), notes: map[int][]sampleNote{}}
			byName[name] = in
			if family == "bass" {
				set.bass = append(set.bass, in)
			} else {
				set.other = append(set.other, in)
			}
		}
		if in.notes[pitch] == nil {
			in.pitches = append(in.pitches, pitch)
		}
		in.notes[pitch] = append(in.notes[pitch], sampleNote{filepath.Join(dir, e.Name()), vel})
	}
	if len(byName) == 0 {
		return nil, fmt.Errorf("sampled %s: no family_source_NNN-pitch-velocity.wav notes", dir)
	}
	for _, in := range byName {
		slices.Sort(in.pitches)
		for _, layers := range in.notes {
			slices.SortFunc(layers, func(a, b sampleNote) int { return cmp.Compare(a.vel, b.vel) })
		}
	}
	return set, nil
}

func cutLast(s, sep string) (before, after string, ok bool) {
	i := strings.LastIndex(s, sep)
	if i <= 0 || i == len(s)-len(sep) {
		return s, "", false
	}
	return s[:i], s[i+len(sep):], true
}

// nearest is the recorded pitch closest to midi, the higher of two so a note is
// shifted down, which keeps it under the band's top
func (in *sampleInstr) nearest(midi int) int {
	i, _ := slices.BinarySearch(in.pitches, midi)
	if i == len(in.pitches) || i > 0 && midi-in.pitches[i-1] < in.pitches[i]-midi {
		i--
	}
	return in.pitches[i]
}

// layer is the recorded note at pitch whose velocity is nearest vel's
func (in *sampleInstr) layer(pitch int, vel float64) sampleNote {
	layers := in.notes[pitch]
	return slices.MinFunc(layers, func(a, b sampleNote) int {
		return cmp.Compare(math.Abs(float64(a.vel)-vel*127), math.Abs(float64(b.vel)-vel*127))
	})
}

// fits says whether a recorded instrument plays every note: each in reach of
// a recorded pitch, its layer one that sounds and that speaks before the note
// is let go, shifted as play shifts it, the loop's tuning (under 3%) aside
func (set *sampleSet) fits(in *sampleInstr, notes []audio.RipEvent) (bool, error) {
	for _, e := range notes {
		pitch := in.nearest(e.MIDI)
		if d := pitch - e.MIDI; d < -sampledReach || d > sampledReach {
			return false, nil
		}
		h, err := set.heard(in.layer(pitch, e.Vel).file)
		if err != nil || h.level == 0 || e.For < shiftedLen(h.speak, math.Exp2(float64(e.MIDI-pitch)/12)) {
			return false, err
		}
	}
	return true, nil
}

// heard reads a recorded note once for the set: its length, what listen hears
// of its held part, its level 0 under the floor or for a file that is not
// NSynth's
func (set *sampleSet) heard(file string) (heard, error) {
	set.mu.Lock()
	defer set.mu.Unlock()
	if h, ok := set.read[file]; ok {
		return h, nil
	}
	x, err := readNote(file)
	if err != nil {
		return heard{}, err
	}
	h := listen(x[:min(len(x), int(nsynthHold*nsynthRate))], nsynthRate)
	h.n = len(x)
	if h.level <= math.Pow(10, sampledFloor/20.0) {
		h.level = 0
	}
	set.read[file] = h
	return h, nil
}

// listen is a note's level at rate, and the samples it takes to speak: to the
// middle of its first window within sampledSpeak of the loudest
func listen(x []float64, rate int) heard {
	rms := windows(len(x), rate, sampledWindow, meanSquare(x))
	h := heard{level: slices.Max(rms)}
	first := slices.IndexFunc(rms, func(r float64) bool { return r >= h.level*math.Pow(10, -sampledSpeak/20.0) })
	h.speak = (first + 1) * (int(sampledWindow*float64(rate)) / 2)
	return h
}

// windows is a sound's level window by window: the RMS of each window of sec at
// rate over its n samples, the hop half a window, from power, a window's mean
// square by its first sample and length; a sound shorter than a window is one.
// A note's level is its loudest held window's, a sampled one's and a measured one's
func windows(n, rate int, sec float64, power func(at, w int) float64) []float64 {
	w := min(int(sec*float64(rate)), n)
	var rms []float64
	for at := 0; at+w <= n; at += max(w/2, 1) {
		rms = append(rms, math.Sqrt(power(at, w)))
	}
	return rms
}

// meanSquare is the power windows reads x's samples by
func meanSquare(x []float64) func(at, w int) float64 {
	return func(at, w int) float64 {
		var e float64
		for _, v := range x[at : at+w] {
			e += v * v
		}
		return e / float64(max(w, 1))
	}
}

// readNote is a recorded note's samples, none for one that does not decode
// as NSynth's
func readNote(file string) ([]float64, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	x, rate, err := audio.ReadWAV(f)
	f.Close()
	if err != nil || rate != nsynthRate {
		return nil, nil
	}
	return x, nil
}

// sampleKey is a recorded note shifted by semitones, as one loop plays it
type sampleKey struct {
	file  string
	shift int
}

// play renders a part's notes on an instrument's recordings, each at the layer
// nearest its velocity, shifted from the nearest recorded pitch, its own level
// taken out and its velocity put in: a note held past its sample's hold ends
// there, a shorter one is cut at its note-off into the release. It returns each
// note's end sample.
func (set *sampleSet) play(in *sampleInstr, notes []audio.RipEvent, n int, cents float64, cache map[sampleKey][]float64, render bool) ([]int, []float64, error) {
	sr := float64(audio.AudioSampleRate)
	var out []float64
	if render {
		out = make([]float64, n)
	}
	fade := int(sampledFade * sr)
	ends := make([]int, len(notes))
	for k, e := range notes {
		pitch := in.nearest(e.MIDI)
		key := sampleKey{in.layer(pitch, e.Vel).file, e.MIDI - pitch}
		ratio := math.Exp2(float64(key.shift)/12 + cents/1200)
		h, err := set.heard(key.file)
		if err != nil {
			return nil, nil, err
		}
		if h.level == 0 {
			return nil, nil, fmt.Errorf("%s: a note that does not sound", key.file)
		}
		hold := min(int(nsynthHold/ratio*sr), shiftedLen(h.n, ratio))
		cut := e.For
		ends[k] = e.At + min(cut, hold)
		if !render {
			continue
		}
		x, ok := cache[key]
		if !ok {
			raw, err := readNote(key.file)
			if err != nil {
				return nil, nil, err
			}
			x = shiftNote(raw, ratio)
			for i := range x {
				x[i] /= h.level
			}
			cache[key] = x
		}
		put := func(i int, v float64) {
			if i += e.At; i < n {
				out[i] += v * e.Vel
			}
		}
		if cut+fade >= hold {
			for i, v := range x {
				put(i, v)
			}
			continue
		}
		for i := range cut {
			put(i, x[i])
		}
		for i := range len(x) - hold {
			v := x[hold+i]
			if i < fade {
				g := float64(i) / float64(fade)
				v = x[cut+i]*(1-g) + v*g
			}
			put(cut+i, v)
		}
	}
	return ends, out, nil
}

// shiftedLen is how many samples shiftNote makes of n read at ratio times their rate
func shiftedLen(n int, ratio float64) int {
	return int(float64(n) / (ratio * nsynthRate / audio.AudioSampleRate))
}

// shiftNote reads x at ratio times its rate into the engine's rate through a
// Kaiser-windowed sinc, its cutoff under sampledTop in the output and under
// the source's Nyquist, where audio.Resample's linear upsampling would image
func shiftNote(x []float64, ratio float64) []float64 {
	step := ratio * nsynthRate / audio.AudioSampleRate     // source samples an output sample
	fc := float64(sampledTop) / nsynthRate / max(ratio, 1) // cycles a source sample
	const beta = 5.65                                      // a Kaiser window of 60 dB
	table := make([]float64, 2*sampledHalf*sampledPhases+2)
	i0 := func(v float64) float64 { // the modified Bessel function of order zero
		sum, term := 1.0, 1.0
		for k := 1; k < 32; k++ {
			term *= v * v / (4 * float64(k*k))
			sum += term
		}
		return sum
	}
	for j := range table {
		u := float64(j)/sampledPhases - sampledHalf
		w := 0.0
		if r := u / sampledHalf; r*r <= 1 {
			w = i0(beta*math.Sqrt(1-r*r)) / i0(beta)
		}
		s := 2 * fc
		if u != 0 {
			s = math.Sin(2*math.Pi*fc*u) / (math.Pi * u)
		}
		table[j] = s * w
	}
	y := make([]float64, shiftedLen(len(x), ratio))
	for i := range y {
		t := float64(i) * step
		c := int(t)
		var acc float64
		for j := max(c-sampledHalf+1, 0); j <= min(c+sampledHalf, len(x)-1); j++ {
			pos := (float64(j) - t + sampledHalf) * sampledPhases
			k := int(pos)
			acc += x[j] * (table[k] + (pos-float64(k))*(table[k+1]-table[k]))
		}
		y[i] = acc
	}
	return y
}
