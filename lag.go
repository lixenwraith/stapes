package stapes

import (
	"bufio"
	"encoding/binary"
	"iter"
	"math"
	"os"
	"slices"

	"github.com/lixenwraith/audio"
)

// What puts a recording's hits on its sound: the grid a lag is read on, and five
// constants fitted on the engine's kits by
// TestARecordedKitsHitsAreMovedOntoTheirSound, which names their procedure
const (
	lagStep  = audio.RipFrameSec / 8 // the grid a lag is read on
	lagReach = 0.5                   // the share of the usual gap between strikes a lag is looked for within
	lagRise  = -4 * lagStep          // where the rise a hit of the engine's draws sits from the hit
	lagBound = 7 * lagStep           // how far from the lag laid one found lies at most
	lagClear = 0.97                  // the most another lag may meet of what the lag found meets
	lagDrawn = 0.4                   // the least share of a recording's hits that draw an onset
)

// lagWhy is why a recording's hits are not put on its sound, each a reason a
// run counts its recordings left out by
var lagWhy = []string{"hits that draw no onset", "no lag that stands clear", "a lag that drifts"}

// hitBound is the least rise a hit's onset reads where it draws a sound, in the
// frames' own units, fitted on the engine's kits by
// TestAHitThatDrawsNoSoundIsToldByItsRise, which names its procedure
const hitBound = 1e-3

// hitWhy is why a hit of a recording put on its sound is left out of its label,
// each a reason its loop's left lists and a run counts hits by
var hitWhy = []string{"no sound"}

// onsetRise is the rise a hit's onset reads at seconds at, where the lag puts
// its sound's onset: the largest within a frame, as lagOf reads one. A frame's
// rise is read against the frame before, so the first frame's is unread: a hit
// on it, or past the last, reads NaN.
func onsetRise(flux []float64, at float64) float64 {
	p := at / audio.RipFrameSec
	a, b := max(int(math.Ceil(p-1)), 1), min(int(math.Floor(p+1)), len(flux)-1)
	if a > b {
		return math.NaN()
	}
	return slices.Max(flux[a : b+1])
}

// rise is how far each frame's sharp plane rises over the frame before, summed
// over its bands, none at the first: the onset flux a hit's lag is read by,
// as the trainer's datacheck.py reads it
func rise(rows iter.Seq[[]float32]) []float64 {
	var out []float64
	var last []float32
	for row := range rows {
		sharp, up := row[audio.RipBands:], 0.0
		for b := range last {
			up += max(float64(sharp[b]-last[b]), 0)
		}
		out, last = append(out, up), append(last[:0], sharp...)
	}
	return out
}

// bandsRows is the frames of a loop's .bands, read one at a time and held for
// the step alone; a file that stops short ends them
func bandsRows(file string) iter.Seq[[]float32] {
	return func(yield func([]float32) bool) {
		f, err := os.Open(file)
		if err != nil {
			return
		}
		defer f.Close()
		r := bufio.NewReaderSize(f, 1<<20)
		var head [2]uint32
		if binary.Read(r, binary.LittleEndian, &head) != nil {
			return
		}
		row := make([]float32, head[1])
		for range head[0] {
			if binary.Read(r, binary.LittleEndian, row) != nil || !yield(row) {
				return
			}
		}
	}
}

// strikeTimes is where a recording's drums are struck, in order, of the times its
// hits and its strikes of no line start at: one less than a frame after
// another is the same rise, and is left out
func strikeTimes(at []float64) []float64 {
	at = slices.Sorted(slices.Values(at))
	out := at[:0]
	for _, t := range at {
		if len(out) == 0 || t-out[len(out)-1] >= audio.RipFrameSec {
			out = append(out, t)
		}
	}
	return out
}

// met is the rise hits meet when moved by lag: flux at each, on the line
// between its two frames
func met(flux, hits []float64, lag float64) float64 {
	var sum float64
	for _, h := range hits {
		p := (h + lag) / audio.RipFrameSec
		i := int(math.Floor(p))
		if i < 0 || i+1 >= len(flux) {
			continue
		}
		w := p - float64(i)
		sum += flux[i]*(1-w) + flux[i+1]*w
	}
	return sum
}

// hitsLag is how a recording's hits lie against its rise: the lag at which they
// meet the most of it, what the best lag of another peak meets over that, the
// share of them that draw an onset, and how far its later half's onsets lie
// from its earlier half's
type hitsLag struct{ lag, other, drawn, drift float64 }

// reachOf is how far either way a lag is looked for: share of the usual gap
// between two strikes, their median, past which hits moved meet the rises of
// those beside them; a lone strike's is the recording's length
func reachOf(hits []float64, frames int, share float64) float64 {
	if len(hits) < 2 {
		return float64(frames) * audio.RipFrameSec
	}
	gaps := make([]float64, len(hits)-1)
	for i := range gaps {
		gaps[i] = hits[i+1] - hits[i]
	}
	return share * median(gaps)
}

// lagOf reads hits against flux over the lags within reach of lagRise, by half
// frames and then by lagStep. Another peak is one within the reach past where
// the score stops falling, a tie where it still rises at the reach. A hit's onset
// is the largest rise about it, drawn where that lies within a frame of where
// the lag puts the hit; the drift is between the halves' median onsets.
func lagOf(flux, hits []float64, reach float64) hitsLag {
	half := audio.RipFrameSec / 2
	n := max(int(reach/half), 1)
	curve := make([]float64, 2*n+1)
	top := 0
	for k := range curve {
		if curve[k] = met(flux, hits, lagRise+float64(k-n)*half); curve[k] > curve[top] {
			top = k
		}
	}
	if curve[top] <= 0 {
		return hitsLag{other: 1}
	}
	a, b := top, top
	for a > 0 && curve[a-1] < curve[a] {
		a--
	}
	for b < 2*n && curve[b+1] < curve[b] {
		b++
	}
	r := hitsLag{lag: lagRise + float64(top-n)*half}
	for k := 1; k < 2*n; k++ {
		if v := curve[k]; (k < a || k > b) && v >= curve[k-1] && v >= curve[k+1] {
			r.other = max(r.other, v)
		}
	}
	best := curve[top]
	for k := -3; k <= 3; k++ {
		at := lagRise + float64(top-n)*half + float64(k)*lagStep
		if v := met(flux, hits, at); v > best {
			best, r.lag = v, at
		}
	}
	if r.other /= best; top == 0 || top == 2*n {
		r.other = 1
	}
	// each hit's onset in frames from where the lag puts it: over its reach to
	// say whether it draws one, over the best lag's own peak for its half's drift
	var halves [2][]float64
	most := func(from, to float64) (int, bool) {
		a, b := max(int(math.Ceil(from)), 0), min(int(to), len(flux)-1)
		at := a
		for f := a; f <= b; f++ {
			if flux[f] > flux[at] {
				at = f
			}
		}
		return at, a <= b && flux[at] > 0
	}
	for i, h := range hits {
		p, by := (h+r.lag)/audio.RipFrameSec, reach/audio.RipFrameSec
		if at, ok := most(p-by, p+by); ok && math.Abs(float64(at)-p) <= 1 {
			r.drawn++
		}
		if at, ok := most(p+float64(a-top)/2, p+float64(b-top)/2+1); ok {
			halves[2*i/len(hits)] = append(halves[2*i/len(hits)], float64(at)-p)
		}
	}
	r.drawn /= float64(len(hits))
	if len(halves[0]) > 0 && len(halves[1]) > 0 {
		r.drift = math.Abs(median(halves[1])-median(halves[0])) * audio.RipFrameSec
	}
	return r
}

// placeHits is how far a recording's sound sits after its hits, by its rise, as
// the engine's own drums sit after theirs, in seconds on lagStep's grid; or why
// its hits are not put on it, a word of lagWhy
func placeHits(flux, hits []float64) (float64, string) {
	r := lagOf(flux, hits, reachOf(hits, len(flux), lagReach))
	switch {
	case r.drawn < lagDrawn:
		return 0, lagWhy[0]
	case r.other > lagClear:
		return 0, lagWhy[1]
	case r.drift > lagBound:
		return 0, lagWhy[2]
	}
	return r.lag - lagRise, ""
}
