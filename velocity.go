package stapes

import (
	"cmp"
	"errors"
	"fmt"
	"io/fs"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"os"
	"slices"
	"sort"

	"github.com/lixenwraith/audio"
)

// A recorded note's velocity is its level over its recording's unit: the
// velocity at which the engine's straight law (sampler.go's) plays a note as
// loud. Its level is read from its partials, each held window's fitted with
// those of the notes still ringing near them under a Hann window of sampledWindow.
const (
	measuredPartials = 80    // partials a note is read from at most
	measuredTop      = 20000 // Hz, the highest partial read, short of nine tenths of a recording's Nyquist
	measuredNear     = 3     // a window's inverse lengths within which partials are fitted together
	measuredApart    = 2     // the most the partials fitted with it may widen a partial's estimate's variance for it to be its own
	measuredBound    = 0.8   // dB, the half-width a level is known within to give a velocity (TestAMeasuredLevelIsKnownWithinItsFittedBound)
	measuredDepth    = 60    // dB under its strongest a note's partials are read to, in its first window
)

// measuredStiffness is the stiffnesses a note's partials are tried stretched by:
// none, and from a nylon string's to a piano's top octave a factor of √2 apart,
// the best then refined by factors of 2^¼ and 2^⅛ either side
var measuredStiffness = func() []float64 {
	b := []float64{0}
	for s := 1e-5; s < 2e-2; s *= math.Sqrt2 {
		b = append(b, s)
	}
	return b
}()

// heardNote is a note as its recording is measured: its label's first and end
// samples, and its pitch at a sample as a MIDI number
type heardNote struct {
	start, end int
	pitch      func(at int) float64
}

// heardLevel is a note's level bounded, mean squares of its loudest held
// window: from below by the partials no other note shares, from above with all
// a window holds at the frequency of each one shared
type heardLevel struct{ lo, hi float64 }

// level is the root of the middle of the bounds
func (h heardLevel) level() float64 { return math.Sqrt(math.Sqrt(h.lo * h.hi)) }

// unsure is how far in dB the level may lie from its middle, unbounded for a
// note none of whose partials is its own
func (h heardLevel) unsure() float64 {
	if h.lo <= 0 {
		return math.Inf(1)
	}
	return 5 * math.Log10(h.hi/h.lo)
}

// heardPartial is a partial of the note a window measures: whether the partials
// fitted with it leave it its own (measuredApart), its fitted energy, the RMS no
// partial explains either side of it, and all the window holds at its
// frequency, mean squares
type heardPartial struct {
	own       bool
	e, u, all float64
}

// heardWindow is a note's partials fitted in a window, and the mean square of
// what the window holds that no fitted partial explains
type heardWindow struct {
	partials []heardPartial
	rest     float64
}

// bounds is what a window holds of its note: a partial of its own within what
// is left unfitted either side of it, one not told apart at most all the window
// holds there, and what no partial explains, a pluck, breath or bow that may be
// the note's, at most
func (h heardWindow) bounds() (lo, hi float64) {
	for _, p := range h.partials {
		if p.own {
			lo += sq(max(math.Sqrt(p.e)-p.u, 0))
			hi += sq(math.Sqrt(p.e) + p.u)
		} else {
			hi += p.all
		}
	}
	return lo, hi + h.rest
}

// above is the most a window holds of its note where another note could move
// what it measures past the bound: all the window holds at each of its
// partials and what no partial explains
func (h heardWindow) above() (hi float64) {
	for _, p := range h.partials {
		hi += p.all
	}
	return hi + h.rest
}

// heardSounding is another note a window fits, over its samples from and to
type heardSounding struct{ j, from, to int }

// heardRing is a note in a window past its label's end: whether its partials
// told apart are found there, holding more than is left unfitted around them,
// the most each of its partials could hold there, and which are told apart
type heardRing struct {
	found bool
	could []float64
	told  []bool
}

// hearLevels is each note's level in x at rate, none for a note another note
// sounds with, which the bound does not hold for. Each window fits a note's
// partials with those of the notes still ringing near them (rings); one whose
// estimate the partials fitted with it widen past measuredApart cannot be told
// apart, so the low bound counts none of it and the high bound all the window holds there
func hearLevels(x []float64, rate int, notes []heardNote) []heardLevel {
	w := int(sampledWindow * float64(rate))
	hop := w / 2
	hann, xw := make([]float64, w), make([]float64, w)
	for i := range hann {
		hann[i] = 0.5 - 0.5*math.Cos(2*math.Pi*(float64(i)+0.5)/float64(w))
	}
	near := 2 * math.Pi * measuredNear / float64(w)
	window := func(a int) {
		for t := range w {
			xw[t] = hann[t] * x[a+t]
		}
	}
	partials := func(n heardNote, at int, b float64, f []float64) []float64 { return partialsOf(n, at, rate, b, f) }
	stiff, reach := stiffnesses(x, rate, notes)
	var f, mids, probes []float64
	var cols []heardColumn
	var proj []complex128
	// add appends a column for each of fs sounding from lo to hi, with the
	// window's projection on it
	add := func(fs []float64, lo, hi int, own bool) {
		proj = slices.Grow(proj[:0], len(fs))[:len(fs)]
		project(xw, fs, lo, hi, proj)
		for k, fk := range fs {
			cols = append(cols, heardColumn{f: fk, lo: lo, hi: hi, own: own, b: 2 * proj[k]})
		}
	}
	gram := func(p, q heardColumn) complex128 {
		if from, to := max(p.lo, q.lo), min(p.hi, q.hi); to > from {
			return hannSum(q.f-p.f, from, to, w)
		}
		return 0
	}
	// fit is note ni's partials in the window from a, fitted with those of the
	// others near them over the samples of the window each sounds in
	fit := func(ni, a int, others []heardSounding) heardWindow {
		window(a)
		cols = cols[:0]
		own := slices.Clone(partials(notes[ni], a+w/2, stiff[ni], f))
		own = own[:min(len(own), reach[ni])]
		out := heardWindow{partials: make([]heardPartial, len(own))}
		if len(own) == 0 {
			return out
		}
		add(own, 0, w, true)
		// midway below, between and above its partials, where the sound no
		// partial holds is read
		mids = append(mids[:0], own[0]/2)
		for k, fk := range own {
			if k+1 < len(own) {
				mids = append(mids, (fk+own[k+1])/2)
			} else if k > 0 {
				mids = append(mids, fk+(fk-own[k-1])/2)
			} else {
				mids = append(mids, fk*1.5)
			}
		}
		probes = append(append(probes[:0], own...), mids...)
		slices.Sort(probes)
		for _, o := range others {
			f = partials(notes[o.j], a+(o.from+o.to)/2, stiff[o.j], f)
			f = slices.DeleteFunc(f, func(g float64) bool {
				i := sort.Search(len(probes), func(i int) bool { return probes[i] > g-near })
				return i == len(probes) || probes[i] >= g+near
			})
			add(f, o.from, o.to, false)
		}
		slices.SortFunc(cols, func(p, q heardColumn) int { return cmp.Compare(p.f, q.f) })
		explained := 0.0
		for s := 0; s < len(cols); {
			e := s + 1
			for e < len(cols) && cols[e].f-cols[e-1].f < near {
				e++
			}
			g, at := cols[s:e], s
			s = e
			m := len(g)
			// every pair, not only those within near: a partial that stops in the
			// window is wider than a window's bins, and left out its pairs leave a
			// matrix no Gram is, whose factor fails
			G, b := make([]complex128, m*m), make([]complex128, m)
			for p := range g {
				b[p] = g[p].b
				for q := p; q < m; q++ {
					G[p*m+q] = gram(g[p], g[q])
					G[q*m+p] = cmplx.Conj(G[p*m+q])
				}
			}
			L := cholesky(G, m)
			c := cholSolve(L, b, m)
			for p := range g {
				cols[at+p].c = c[p]
				explained += real(cmplx.Conj(c[p])*b[p]) / 2
			}
			unit := make([]complex128, m)
			for p := range g {
				if !g[p].own {
					continue
				}
				// how far the partials fitted with it widen its estimate's variance:
				// its sum of squares times the inverse's entry on the diagonal
				k, d := sort.SearchFloat64s(own, g[p].f), b[p]/G[p*m+p]
				out.partials[k] = heardPartial{all: real(d*cmplx.Conj(d)) / 2}
				clear(unit)
				unit[p] = 1
				if real(G[p*m+p])*real(cholSolve(L, unit, m)[p]) <= measuredApart {
					out.partials[k].own, out.partials[k].e = true, real(c[p]*cmplx.Conj(c[p]))/2
				}
			}
		}
		// what no fitted partial holds, either side of each of its own
		proj = slices.Grow(proj[:0], len(mids))[:len(mids)]
		project(xw, mids, 0, w, proj)
		rest := make([]float64, len(mids))
		for k, mid := range mids {
			r := proj[k]
			for j := sort.Search(len(cols), func(j int) bool { return cols[j].f > mid-near }); j < len(cols) && cols[j].f < mid+near; j++ {
				r -= cols[j].c / 2 * hannSum(cols[j].f-mid, cols[j].lo, cols[j].hi, w)
			}
			rest[k] = 8 * real(r*cmplx.Conj(r)) / float64(w*w)
		}
		for k := range out.partials {
			out.partials[k].u = math.Sqrt((rest[k] + rest[k+1]) / 2)
		}
		var all float64
		for t := range w {
			all += xw[t] * x[a+t]
		}
		out.rest = max(all-explained, 0) / (float64(w) / 2)
		return out
	}
	// labelled is the notes but j whose labels hold some of the window from a,
	// each over its labelled samples there
	labelled := func(j, a int) []heardSounding {
		var on []heardSounding
		for k, m := range notes {
			if k != j && m.start < a+w && m.end > a {
				on = append(on, heardSounding{k, max(m.start-a, 0), min(m.end-a, w)})
			}
		}
		return on
	}
	// could is the most each of a note's partials could hold in a window: one
	// told apart its fitted energy and what is unfitted around it, one not all
	// the window holds there
	could := func(h heardWindow) []float64 {
		c := make([]float64, len(h.partials))
		for i, p := range h.partials {
			c[i] = p.all
			if p.own {
				c[i] = sq(math.Sqrt(p.e) + p.u)
			}
		}
		return c
	}
	// rings[j][k] is note j in the window k hops past its label's end, fitted
	// with the notes labelled there and read as needed. A partial not told apart
	// is in doubt: it could hold the most a told one under it kept of what it
	// could hold in the note's last labelled window, a partial dying no faster
	// than those over it, or, none under it telling, all the window holds there
	rings, last, freqs := make([][]heardRing, len(notes)), make([][]float64, len(notes)), make([][]float64, len(notes))
	ring := func(j, k int) heardRing {
		for len(rings[j]) <= k {
			a := notes[j].end + len(rings[j])*hop
			if a < 0 || a+w > len(x) {
				rings[j] = append(rings[j], heardRing{})
				continue
			}
			if last[j] == nil {
				last[j] = []float64{}
				if at := max(notes[j].end-w, notes[j].start, 0); at+w <= len(x) {
					last[j] = could(fit(j, at, labelled(j, at)))
				}
			}
			h := fit(j, a, labelled(j, a))
			r := heardRing{could: could(h), told: make([]bool, len(h.partials))}
			var e, u2, kept float64
			for i, p := range h.partials {
				switch {
				case p.own:
					e, u2, r.told[i] = e+p.e, u2+p.u*p.u, true
					if i < len(last[j]) && last[j][i] > 0 {
						kept = max(kept, r.could[i]/last[j][i])
					}
				case kept > 0 && i < len(last[j]):
					r.could[i] = last[j][i] * kept
				}
			}
			r.found = e > u2
			rings[j] = append(rings[j], r)
		}
		return rings[j][k]
	}
	out := make([]heardLevel, len(notes))
	for ni, n := range notes {
		span, with := heldSpan(n, rate, len(x)), false
		for j, m := range notes {
			with = with || j != ni && m.start < n.start+span && m.end > n.start
		}
		if n.start < 0 || span < w || with {
			continue
		}
		// what a ring may hold near ni's partials: what moves ni's level, its
		// loudest window's alone, past the bound with what ni's bound alone
		// already spans, the ring's partials added to ni's in phase
		var top, over, bar float64
		windows(span, rate, sampledWindow, func(at, _ int) float64 {
			lo, hi := fit(ni, n.start+at, nil).bounds()
			top, over = max(top, lo), max(over, hi)
			return 0
		})
		if top > 0 {
			bar = top * sq(math.Pow(10, max(measuredBound-5*math.Log10(over/top), 0)/20)-1)
		}
		// each note before rings into ni's windows from its label's end until a
		// window where its partials are not found and what those near ni's could
		// hold there cannot move ni's level past the bound. A window where a ring
		// is found and those told apart could, ni's not told from them, bounds ni
		// from above only; a partial in doubt gives ni's low bound nothing already
		mine := partials(n, n.start+w/2, stiff[ni], nil)
		mine = mine[:min(len(mine), reach[ni])]
		nearby := func(j, k int, told bool) (c float64) {
			if freqs[j] == nil {
				freqs[j] = partials(notes[j], notes[j].end, stiff[j], nil)
			}
			r := ring(j, k)
			for i, g := range freqs[j][:min(len(freqs[j]), len(r.could))] {
				if p := sort.SearchFloat64s(mine, g-near); p < len(mine) && mine[p] < g+near && (r.told[i] || !told) {
					c += r.could[i]
				}
			}
			return c
		}
		var sounding []heardSounding // a ring and the sample it is let go at, in to
		for j, m := range notes {
			if j == ni || m.end > n.start {
				continue
			}
			at := n.start + span
			for k := 0; m.end+k*hop < n.start+span; k++ {
				if !ring(j, k).found && nearby(j, k, false) <= bar {
					at = m.end + k*hop
					break
				}
			}
			if at > n.start {
				sounding = append(sounding, heardSounding{j, 0, at})
			}
		}
		var his []float64
		los := windows(span, rate, sampledWindow, func(at, _ int) float64 {
			a, held := n.start+at, false
			var others []heardSounding
			for _, s := range sounding {
				if s.to > a {
					others = append(others, heardSounding{s.j, 0, min(s.to-a, w)})
					k := (a - notes[s.j].end) / hop
					held = held || ring(s.j, k).found && nearby(s.j, k, true) > bar
				}
			}
			h := fit(ni, a, others)
			if held {
				his = append(his, h.above())
				return 0
			}
			lo, hi := h.bounds()
			his = append(his, hi)
			return lo
		})
		lo := slices.Max(los)
		out[ni] = heardLevel{lo * lo, slices.Max(his)}
	}
	return out
}

// heldSpan is the samples a note's level is read over: from its start to its
// end or nsynthHold on, at least a window, within a recording of total samples
func heldSpan(n heardNote, rate, total int) int {
	w := int(sampledWindow * float64(rate))
	return min(max(min(n.end, n.start+int(nsynthHold*float64(rate)))-n.start, w), total-n.start)
}

// stiffnesses is each note's stiffness, of measuredStiffness the one whose
// partials hold the most of its first window, and how many of its partials
// there reach within measuredDepth of its strongest
func stiffnesses(x []float64, rate int, notes []heardNote) ([]float64, []int) {
	w := int(sampledWindow * float64(rate))
	xw, stiff, reach := make([]float64, w), make([]float64, len(notes)), make([]int, len(notes))
	var f []float64
	var proj []complex128
	energy := func(n heardNote, b float64, k int) []complex128 {
		f = partialsOf(n, n.start+w/2, rate, b, f)
		f = f[:min(len(f), k)]
		proj = slices.Grow(proj[:0], len(f))[:len(f)]
		project(xw, f, 0, w, proj)
		return proj
	}
	for i, n := range notes {
		reach[i] = measuredPartials
		if n.start < 0 || n.start+w > len(x) {
			continue
		}
		for t := range w {
			xw[t] = (0.5 - 0.5*math.Cos(2*math.Pi*(float64(t)+0.5)/float64(w))) * x[n.start+t]
		}
		best := -1.0
		try := func(b float64) {
			var e float64
			for _, d := range energy(n, b, measuredPartials/2) {
				e += real(d * cmplx.Conj(d))
			}
			if e > best {
				best, stiff[i] = e, b
			}
		}
		for _, b := range measuredStiffness {
			try(b)
		}
		for _, r := range []float64{1.0 / 4, 1.0 / 8} {
			if b := stiff[i]; b > 0 {
				try(b * math.Exp2(r))
				try(b * math.Exp2(-r))
			}
		}
		var e []float64
		for _, d := range energy(n, stiff[i], measuredPartials) {
			e = append(e, real(d*cmplx.Conj(d)))
		}
		if len(e) > 0 {
			floor := slices.Max(e) * math.Pow(10, -measuredDepth/10.0)
			for reach[i] = len(e); reach[i] > 1 && e[reach[i]-1] < floor; reach[i]-- {
			}
		}
	}
	return stiff, reach
}

// partialsOf is a note's partials at a sample, in radians a sample into f: k
// times its pitch's, stretched by a string's stiffness b as k√(1+bk²), up to
// measuredPartials and measuredTop
func partialsOf(n heardNote, at, rate int, b float64, f []float64) []float64 {
	f0 := 2 * math.Pi * 440 * math.Exp2((n.pitch(at)-69)/12) / float64(rate)
	top := 2 * math.Pi * min(measuredTop, 0.45*float64(rate)) / float64(rate)
	f = f[:0]
	for k := 1.0; k <= measuredPartials; k++ {
		if fk := k * f0 * math.Sqrt(1+b*k*k); fk < top {
			f = append(f, fk)
		}
	}
	return f
}

// heardColumn is a partial a window fits: its frequency in radians a sample, the
// samples of the window it sounds in, and whether it is the measured note's
type heardColumn struct {
	f      float64
	lo, hi int
	own    bool
	b, c   complex128 // twice the window's projection on it, and its fitted amplitude
}

// project is the Hann-weighted window xw's sum against a partial at each of f,
// radians a sample, over its samples from lo to hi, into out: Goertzel's
// recurrence four partials at a time, which the processor runs side by side
func project(xw, f []float64, lo, hi int, out []complex128) {
	for j := 0; j < len(f); j += 4 {
		var c [4]float64
		for k := range min(4, len(f)-j) {
			c[k] = 2 * math.Cos(f[j+k])
		}
		var a0, a1, a2, a3, b0, b1, b2, b3 float64
		for _, v := range xw[lo:hi] {
			a0, b0 = v+c[0]*a0-b0, a0
			a1, b1 = v+c[1]*a1-b1, a1
			a2, b2 = v+c[2]*a2-b2, a2
			a3, b3 = v+c[3]*a3-b3, a3
		}
		a, b := [4]float64{a0, a1, a2, a3}, [4]float64{b0, b1, b2, b3}
		for k := range min(4, len(f)-j) {
			s, co := math.Sincos(f[j+k])
			ps, pc := math.Sincos(f[j+k] * float64(hi-1))
			out[j+k] = complex(pc, -ps) * complex(a[k]-co*b[k], s*b[k])
		}
	}
}

// hannSum is the sum of a Hann window of w samples times e^(iθt) over t from lo
// to hi, in closed form: each of its three terms a geometric series
func hannSum(theta float64, lo, hi, w int) complex128 {
	n, a := float64(hi-lo), 2*math.Pi/float64(w)
	mid := float64(lo) + (n-1)/2
	series := func(phi float64) complex128 {
		d := n
		if s := math.Sin(phi / 2); math.Abs(s) > 1e-12 {
			d = math.Sin(n*phi/2) / s
		}
		ps, pc := math.Sincos(phi * mid)
		return complex(d*pc, d*ps)
	}
	hs, hc := math.Sincos(a / 2)
	return 0.5*series(theta) - 0.25*complex(hc, hs)*series(theta+a) - 0.25*complex(hc, -hs)*series(theta-a)
}

// cholesky is a Hermitian matrix's lower factor under a ridge a billionth of
// its mean diagonal, which partials at one frequency would leave singular; a
// row's entries before its first not zero stay zero
func cholesky(G []complex128, m int) []complex128 {
	var ridge float64
	for i := range m {
		ridge += real(G[i*m+i]) * 1e-9 / float64(m)
	}
	L, first := make([]complex128, m*m), make([]int, m)
	for i := range m {
		for first[i] = 0; first[i] < i && G[i*m+first[i]] == 0; first[i]++ {
		}
		for j := first[i]; j <= i; j++ {
			s := G[i*m+j]
			for k := max(first[i], first[j]); k < j; k++ {
				s -= L[i*m+k] * cmplx.Conj(L[j*m+k])
			}
			if i == j {
				L[i*m+i] = complex(math.Sqrt(max(real(s)+ridge, 1e-300)), 0)
			} else {
				L[i*m+j] = s / L[j*m+j]
			}
		}
	}
	return L
}

// cholSolve solves L Lᴴ c = b
func cholSolve(L, b []complex128, m int) []complex128 {
	c := slices.Clone(b)
	for i := range m {
		for k := range i {
			c[i] -= L[i*m+k] * c[k]
		}
		c[i] /= L[i*m+i]
	}
	for i := m - 1; i >= 0; i-- {
		for k := i + 1; k < m; k++ {
			c[i] -= cmplx.Conj(L[k*m+i]) * c[k]
		}
		c[i] /= L[i*m+i]
	}
	return c
}

// heardUnit is the level of a note of velocity one among notes measured
// together: the ninetieth percentile of the levels known within measuredBound
// over top, the velocity the generator gives a line's (lineTop); 0 where none is known
func heardUnit(levels []heardLevel, top float64) float64 {
	var known []float64
	for _, h := range levels {
		if h.unsure() <= measuredBound {
			known = append(known, h.level())
		}
	}
	if len(known) == 0 {
		return 0
	}
	return ninetieth(known) / top
}

// heardVelocity is a note's level over unit, at most 1, and -1 for one whose
// level is not known within measuredBound or that has no unit; a level, a
// bound or a unit not finite is none of them
func heardVelocity(h heardLevel, unit float64) float64 {
	if !finite(unit) || unit <= 0 || !finite(h.level()) || !(h.unsure() <= measuredBound) {
		return -1
	}
	return min(h.level()/unit, 1)
}

func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

// lineTop is the velocity the generator gives a line's loudest notes: the
// median over the tonal parts of 64 loops it arranges at a fixed seed of a
// part's ninetieth percentile
func lineTop() float64 {
	var tops []float64
	for i := range 64 {
		stream := func(salt uint64) *rand.Rand { return rand.New(rand.NewPCG(salt, uint64(i)+1)) }
		a := newArranger(stream(0x5a5a5a5a), stream(0x3e7e3e7e), 4, 120, false, false)
		a.compose(nil, nil, nil)
		for _, p := range a.parts {
			if p.instr.IsDrum() || len(p.notes) == 0 {
				continue
			}
			vel := make([]float64, len(p.notes))
			for j, n := range p.notes {
				vel[j] = n.vel
			}
			tops = append(tops, ninetieth(vel))
		}
	}
	slices.Sort(tops)
	return tops[len(tops)/2]
}

// median is the middle of v, 0 for none
func median(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	return slices.Sorted(slices.Values(v))[len(v)/2]
}

func sq(v float64) float64 { return v * v }

// ninetieth is the ninetieth percentile of v by nearest rank: the least that
// nine tenths of v are at most
func ninetieth(v []float64) float64 {
	v = slices.Sorted(slices.Values(v))
	return v[int(math.Ceil(0.9*float64(len(v))))-1]
}

// unmeasuredWhy is why a recording's notes may be given no velocity measured,
// in the order a run's last line counts them (unmeasured)
var unmeasuredWhy = []string{"gone from disk", "whose sound does not read", "with a level not finite"}

// unmeasured is a recording the measure gives no levels, and why, a row of unmeasuredWhy
type unmeasured struct{ why, what string }

func (u *unmeasured) Error() string { return u.what + ": " + u.why }

// hearRecording is the level of each of a score's notes in its recording, along
// its pitch track where one places it, or why it gives none (unmeasured). Past
// its label's end a note rings at the pitch it was let go at: the track there
// is the next note's.
func hearRecording(wav string, score *audio.MIDIScore, tracks []pitchTrack) ([]heardLevel, *unmeasured) {
	f, err := os.Open(wav)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, &unmeasured{unmeasuredWhy[0], wav}
	} else if err != nil {
		return nil, &unmeasured{unmeasuredWhy[1], err.Error()}
	}
	x, rate, err := audio.ReadWAV(f)
	f.Close()
	if err != nil {
		return nil, &unmeasured{unmeasuredWhy[1], fmt.Sprintf("%s: %v", wav, err)}
	}
	notes := make([]heardNote, len(score.Notes))
	for i, n := range score.Notes {
		key, tr := float64(n.MIDI), trackOf(tracks, n)
		start, end := int(n.Start*float64(rate)), int(n.End*float64(rate))
		notes[i] = heardNote{start, end, func(at int) float64 {
			if tr != nil {
				if p := tr.at(float64(min(at, end-1)) / float64(rate)); math.Abs(p-key) < 1 {
					return p
				}
			}
			return key
		}}
	}
	levels := hearLevels(x, rate, notes)
	for i, h := range levels {
		if !finite(h.lo) || !finite(h.hi) {
			return nil, &unmeasured{unmeasuredWhy[2], fmt.Sprintf("%s: the note at %.3f s", wav, score.Notes[i].Start)}
		}
	}
	return levels, nil
}
