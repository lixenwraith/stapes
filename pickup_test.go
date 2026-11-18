package stapes

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/lixenwraith/audio"
)

// stringOpen is the laid strings' open keys, a guitar's in standard tuning,
// each playing stringFrets keys over it
var stringOpen = []int{40, 45, 50, 55, 59, 64}

const stringFrets = 19

// stringsLaid is a recording a test lays on six lines, a line a string: the
// microphone's mix, a channel a line, the notes as the measure is told them and
// the line of each; and of each note its level alone on the microphone, the
// sample its sound stops standing over the rest of its channel at, what ended it
// and the most notes sounding at once with it
type stringsLaid struct {
	mic   []float64
	ch    [][]float64
	notes []heardNote
	of    []int
	level []float64
	end   []int
	kind  []string
	poly  []int
}

// stringBleed is how loud a string's notes alone are on each other channel of
// GuitarSet's debleeded by-string recordings against their own, dB at their
// partials in their first window: the tenth, the median and the ninetieth
// percentile over the set's 5,113 notes alone (TestAByStringChannelIsItsString)
var stringBleed = [6][6][3]float64{
	{{0, 0, 0}, {-59, -36, -12}, {-62, -50, -18}, {-65, -54, -26}, {-71, -59, -24}, {-73, -64, -38}},
	{{-59, -45, -21}, {0, 0, 0}, {-49, -29, -4}, {-63, -52, -27}, {-70, -59, -34}, {-73, -62, -42}},
	{{-77, -65, -46}, {-73, -61, -42}, {0, 0, 0}, {-64, -49, -27}, {-80, -69, -46}, {-83, -73, -55}},
	{{-84, -73, -58}, {-82, -74, -59}, {-71, -59, -44}, {0, 0, 0}, {-76, -61, -41}, {-85, -74, -58}},
	{{-86, -78, -64}, {-88, -78, -65}, {-84, -73, -59}, {-68, -54, -30}, {0, 0, 0}, {-72, -60, -42}},
	{{-90, -76, -52}, {-90, -80, -65}, {-86, -74, -59}, {-85, -73, -56}, {-67, -53, -21}, {0, 0, 0}},
}

// stringFloor is how far each channel's quietest twentieth of GuitarSet's
// by-string windows of sampledWindow lies under its loudest twentieth, dB: the
// tenth, median and ninetieth percentile over its 360 recordings
var stringFloor = [6][3]float64{{24.4, 36.6, 53.2}, {22.2, 37.1, 52.2}, {30.9, 46.9, 57.4}, {41.1, 54.6, 60.9}, {35.9, 59.2, 69.3}, {21.4, 52.3, 66.6}}

// decile draws from a measure's tenth, median and ninetieth percentile, along
// the line through them and held at the two outer ones beyond them
func decile(rng *rand.Rand, q [3]float64) float64 {
	u := min(max(rng.Float64(), 0.1), 0.9)
	if u < 0.5 {
		return q[0] + (q[1]-q[0])*(u-0.1)/0.4
	}
	return q[1] + (q[2]-q[1])*(u-0.5)/0.4
}

// stringScatter is the dB by which a note's level on its string's channel strays
// from what its string's other notes say of the microphone's, a Laplace's scale:
// GuitarSet's notes alone stray a median 1.56 dB within a recording and string,
// nine in ten within 5.05, b ln 2 and b ln 10 of a Laplace's b of 2.25
const stringScatter = 2.25

// stringSame is how many in a hundred of a solo's notes are played on the
// string of the note before, GuitarSet's solos' 58 (9,710 of 16,681); and
// stringGains the dB a laid channel's gain is drawn within either way, its
// strings' channels over the microphone spanning a median 8.7 dB a recording
// and 17.9 at nine in ten (136 recordings with three strings' notes alone)
const stringSame, stringGains = 58, 6.0

// stringKinds is how a laid note ends: let go into its sample's release, cut
// over sampledFade as a damped string, rung on past its label, stopped by its
// string's next note while it still stands over the rest of its channel, or
// under that rest from its first window, where its label's end is its end
var stringKinds = []string{"released", "cut", "held on", "struck again", "buried"}

// layStrings lays about secs of solo lines and strummed chords of two to six in
// turn, each note on an instrument of ins drawn anew, 0 to 30 dB down; a channel
// holds its line at a gain drawn stringGains either way, each note strayed by
// stringScatter, each other line's bleed at a level drawn from stringBleed's and
// a white floor drawn from stringFloor's; the microphone, the lines summed,
// through a room where rm is set
func layStrings(t *testing.T, set *sampleSet, ins []*sampleInstr, rng, rm *rand.Rand, secs float64) stringsLaid {
	t.Helper()
	rate := audio.AudioSampleRate
	total, fade := int(secs*float64(rate)), int(sampledFade*float64(rate))
	type drawn struct{ line, at, n, how int }
	var ds []drawn
	line := rng.IntN(6)
	for at := rate / 10; at < total-rate; {
		if rng.IntN(2) == 0 {
			for k := 2 + rng.IntN(7); k > 0 && at < total-rate; k-- {
				if rng.IntN(100) >= stringSame {
					line = rng.IntN(6)
				}
				d := drawn{line, at, int((0.12 + 1.4*rng.Float64()) * float64(rate)), rng.IntN(3)}
				ds = append(ds, d)
				gap := int(0.6 * sq(rng.Float64()) * float64(rate))
				if rng.IntN(4) == 0 {
					gap = 0
				}
				at += d.n + gap
			}
			continue
		}
		for k := 2 + rng.IntN(4); k > 0 && at < total-rate; k-- {
			lines := rng.Perm(6)[:2+rng.IntN(5)]
			slices.Sort(lines)
			if rng.IntN(2) == 0 {
				slices.Reverse(lines)
			}
			spread, n := int(0.06*rng.Float64()*float64(rate)), int((0.3+1.7*rng.Float64())*float64(rate))
			for q, l := range lines {
				lag := q * spread / max(len(lines)-1, 1)
				ds = append(ds, drawn{l, at + lag, n - lag, rng.IntN(3)})
			}
			at += n + int(0.3*sq(rng.Float64())*float64(rate))
		}
	}
	var out stringsLaid
	var xs [][]float64
	cache := map[sampleKey][]float64{}
	for _, d := range ds {
		if d.at >= total-rate/4 {
			continue
		}
		vel := math.Pow(10, -30*rng.Float64()/20)
		for range 20 {
			in, key := ins[rng.IntN(len(ins))], stringOpen[d.line]+rng.IntN(stringFrets+1)
			held := d.n
			if d.how == 2 {
				if ok, err := set.fits(in, []audio.RipEvent{{For: d.n, MIDI: key, Vel: vel}}); err != nil || !ok {
					continue
				}
				held = 5 * rate
			}
			l, ok := laySampled(t, set, in, d.at, key, vel, held, total-d.at, cache)
			if !ok {
				continue
			}
			l.n.end = min(l.n.end, d.at+d.n)
			if d.how == 1 {
				for i := l.n.end - d.at; i < len(l.x); i++ {
					l.x[i] *= max(0, 1-float64(i-(l.n.end-d.at))/float64(fade))
				}
			}
			out.notes, out.of, out.kind, xs = append(out.notes, l.n), append(out.of, d.line), append(out.kind, stringKinds[d.how]), append(xs, l.x)
			break
		}
	}
	// a string sounds one note: the next one struck stops the one before as a damper does
	next := make([]int, len(out.notes))
	for i, n := range out.notes {
		next[i] = total
		for j, m := range out.notes {
			if j != i && out.of[j] == out.of[i] && m.start > n.start && m.start < next[i] {
				next[i] = m.start
			}
		}
		for k := next[i] - n.start; k < len(xs[i]); k++ {
			xs[i][k] *= max(0, 1-float64(k-(next[i]-n.start))/float64(fade))
		}
	}
	gain, bleed := make([]float64, 6), [6][6]float64{}
	for c := range gain {
		gain[c] = math.Pow(10, (2*stringGains*rng.Float64()-stringGains)/20)
		for d := range bleed[c] {
			if d != c {
				bleed[c][d] = math.Pow(10, decile(rng, stringBleed[c][d])/20)
			}
		}
	}
	out.mic, out.ch = make([]float64, total), make([][]float64, 6)
	for c := range out.ch {
		out.ch[c] = make([]float64, total)
	}
	own := make([]float64, len(out.notes)) // each note's gain on its own channel
	for i, n := range out.notes {
		u := rng.Float64() - 0.5
		own[i] = gain[out.of[i]] * math.Pow(10, -stringScatter*math.Copysign(math.Log(1-2*math.Abs(u)), u)/20)
		for k, v := range xs[i] {
			out.mic[n.start+k] += v
			for c := range out.ch {
				g := own[i]
				if c != out.of[i] {
					g *= bleed[out.of[i]][c]
				}
				out.ch[c][n.start+k] += g * v
			}
		}
	}
	for c, x := range out.ch {
		rms := windows(total, rate, sampledWindow, meanSquare(x))
		slices.Sort(rms)
		floor := rms[len(rms)*19/20] * math.Pow(10, -decile(rng, stringFloor[c])/20)
		for k := range x {
			x[k] += floor * rng.NormFloat64()
		}
	}
	carry := func(int, []float64) {}
	if rm != nil {
		out.mic, carry = room(out.mic, rm)
	}
	// each note's end where its own sound at its partials no longer holds more
	// than the rest of its channel holds there, read at hops of a quarter window,
	// or its string's next note, where it still does
	w := int(sampledWindow * float64(rate))
	xw, yw := make([]float64, w), make([]float64, w)
	var f []float64
	out.end, out.level, out.poly = make([]int, len(out.notes)), make([]float64, len(out.notes)), make([]int, len(out.notes))
	for i, n := range out.notes {
		alone := make([]float64, total)
		copy(alone[n.start:], xs[i])
		stiff, reach := stiffnesses(alone, rate, []heardNote{n})
		out.end[i] = next[i]
		for a := n.start; a+w <= next[i]; a += w / 4 {
			for k := range w {
				h := 0.5 - 0.5*math.Cos(2*math.Pi*(float64(k)+0.5)/float64(w))
				xw[k], yw[k] = h*own[i]*alone[a+k], h*(out.ch[out.of[i]][a+k]-own[i]*alone[a+k])
			}
			f = partialsOf(n, a+w/2, rate, stiff[0], f)
			f = f[:min(len(f), reach[0])]
			if len(f) == 0 || partialEnergy(xw, f) <= partialEnergy(yw, f) {
				out.end[i] = a + w/2
				break
			}
		}
		switch {
		case out.end[i] == n.start+w/2:
			out.end[i], out.kind[i] = n.end, stringKinds[4]
		case out.end[i] == next[i] && next[i] < total:
			out.kind[i] = stringKinds[3]
		}
		span := heldSpan(heardNote{n.start, out.end[i], n.pitch}, rate, total)
		heard := slices.Clone(xs[i][:span])
		carry(n.start, heard)
		out.level[i] = slices.Max(windows(span, rate, sampledWindow, meanSquare(heard)))
	}
	for i, n := range out.notes {
		out.poly[i] = 1
		span := heldSpan(heardNote{n.start, out.end[i], n.pitch}, rate, total)
		for j, m := range out.notes {
			if j == i || m.start >= n.start+span || out.end[j] <= n.start {
				continue
			}
			at, k := max(m.start, n.start), 0
			for q, p := range out.notes {
				if p.start <= at && out.end[q] > at {
					k++
				}
			}
			out.poly[i] = max(out.poly[i], k)
		}
	}
	return out
}

// partialEnergy is what a windowed sound holds at partials f
func partialEnergy(xw, f []float64) float64 {
	proj := make([]complex128, len(f))
	project(xw, f, 0, len(xw), proj)
	var e float64
	for _, d := range proj {
		e += real(d * cmplx.Conj(d))
	}
	return e
}

// stringCheck is a laid note read on its string: how it ended, the room it was
// heard through, the most notes sounding with it, the seconds its end was read
// off, and the dB its level was read off and its bound's half-width: on the
// microphone where it sounds alone by its strings' ends (mic), and on its own
// channel put on the microphone's scale by its line's gain (pickup), fitted on a
// seeded half of the notes alone both know, its fit marking those
type stringCheck struct {
	kind, room          string
	poly                int
	endErr              float64
	micErr, micUnsure   float64
	pickErr, pickUnsure float64
	fit                 bool
}

// readStrings measures a laid recording through hearStrings, and each note on
// its line's channel with the ends its strings give
func readStrings(t *testing.T, l stringsLaid, seed uint64) []stringCheck {
	t.Helper()
	rate := audio.AudioSampleRate
	heard, why := hearStrings(l.mic, l.ch, rate, l.notes, l.of)
	if why != "" {
		t.Fatalf("a laid recording's strings say nothing: %s", why)
	}
	channel := make([]heardLevel, len(heard))
	for s := range 6 {
		var on []int
		var line []heardNote
		for i, n := range l.notes {
			if l.of[i] == s {
				n.end = heard[i].end
				on, line = append(on, i), append(line, n)
			}
		}
		for k, h := range hearLevels(l.ch[s], rate, line) {
			channel[on[k]] = h
		}
	}
	pick, fit := rand.New(rand.NewPCG(seed, 0xf17)), make([]bool, len(heard))
	gains := make([]float64, 6)
	for s := range gains {
		var db []float64
		for i, h := range heard {
			if fit[i] = l.of[i] == s && channel[i].unsure() <= measuredBound && h.mic.unsure() <= measuredBound && pick.IntN(2) == 0; fit[i] {
				db = append(db, 20*math.Log10(h.mic.level()/channel[i].level()))
			}
		}
		gains[s], _ = stringGain(db)
	}
	var out []stringCheck
	for i, h := range heard {
		out = append(out, stringCheck{kind: l.kind[i], poly: l.poly[i], endErr: float64(h.end-l.end[i]) / float64(rate),
			micErr: 20 * math.Log10(h.mic.level()/l.level[i]), micUnsure: h.mic.unsure(),
			pickErr: 20*math.Log10(channel[i].level()/l.level[i]) + gains[l.of[i]], pickUnsure: channel[i].unsure(), fit: fit[i]})
	}
	return out
}

// stringReads lays recordings of six lines of NSynth's guitars, with no room and
// through degrade, for three reads: a seeded half of the training split's
// guitars (fit), with three times a readback's recordings, the other half (left
// out) and the test split's (test); each note read on its string against its
// level alone and the end its sound was laid to
func stringReads(t *testing.T, dir string, recs int) map[string][]stringCheck {
	t.Helper()
	train, err := loadSampleSet(filepath.Join(dir, "nsynth-train", "audio"))
	if err != nil {
		t.Fatal(err)
	}
	test, err := loadSampleSet(filepath.Join(dir, "nsynth-test", "audio"))
	if err != nil {
		t.Fatal(err)
	}
	guitars := func(set *sampleSet) []*sampleInstr {
		var ins []*sampleInstr
		for _, in := range set.other {
			if strings.HasPrefix(in.name, "guitar_") {
				ins = append(ins, in)
			}
		}
		slices.SortFunc(ins, func(a, b *sampleInstr) int { return strings.Compare(a.name, b.name) })
		return ins
	}
	all := guitars(train)
	rand.New(rand.NewPCG(11, 2)).Shuffle(len(all), func(i, j int) { all[i], all[j] = all[j], all[i] })
	reads := map[string][]stringCheck{}
	for name, from := range map[string]struct {
		set  *sampleSet
		ins  []*sampleInstr
		recs int
	}{"fit": {train, all[:len(all)/2], 3 * recs}, "left out": {train, all[len(all)/2:], recs}, "test": {test, guitars(test), recs}} {
		for d, how := range []string{"no room", "degrade"} {
			for r := range from.recs {
				l := layStrings(t, from.set, from.ins, rand.New(rand.NewPCG(1, uint64(r))), roomOf(d == 1, 1, uint64(r)), 24)
				for _, c := range readStrings(t, l, uint64(r)) {
					c.room = how
					reads[name] = append(reads[name], c)
				}
			}
		}
	}
	return reads
}

// stringEndBound is the seconds within which nine in ten of the ends the strings
// give lie of where each note's sound was laid to stop standing over the rest of
// its channel, by kind of end, stringKinds' order (TestAStringsNoteIsReadFromItsChannel)
var stringEndBound = []float64{0.7779, 0.4412, 1.37, 0, 2.8096}

// stringEndFit is the least bound in frames within which nine in ten of the ends
// of e lie, sure to one in twenty
func stringEndFit(e []float64) float64 {
	e = slices.Sorted(slices.Values(e))
	for f := 0; f <= 400; f++ {
		if b := float64(f) * audio.RipFrameSec; sure(e, true) <= b {
			return math.Round(b*1e4) / 1e4
		}
	}
	return math.Inf(1)
}

// TestAStringsNoteIsReadFromItsChannel: with NSYNTH, recordings laid on six
// lines of NSynth's guitars (stringReads): each kind's stringEndBound is the
// fit's, with no room and through degrade together, which no readback misses
// at 19 to 1; a note alone known within measuredBound on the microphone lies
// within it, which no readback misses at 19 to 1; logs both, and a note's level
// on its channel put on the microphone's by its line's gain, fitted on half the
// notes alone, against the others
func TestAStringsNoteIsReadFromItsChannel(t *testing.T) {
	dir := os.Getenv("NSYNTH")
	if dir == "" {
		t.Skip("NSYNTH names no folder")
	}
	recs := 8
	if n, err := strconv.Atoi(os.Getenv("STRINGRECS")); err == nil {
		recs = n
	}
	reads := stringReads(t, dir, recs)
	ends := func(checks []stringCheck, kind, room string) []float64 {
		var e []float64
		for _, c := range checks {
			if c.kind == kind && (room == "" || c.room == room) {
				e = append(e, math.Abs(c.endErr))
			}
		}
		return slices.Sorted(slices.Values(e))
	}
	bounds := make([]float64, len(stringKinds))
	for k, kind := range stringKinds {
		bounds[k] = stringEndFit(ends(reads["fit"], kind, ""))
	}
	if !slices.Equal(bounds, stringEndBound) {
		t.Errorf("the fit gives the ends' bounds %v s, stringEndBound is %v", bounds, stringEndBound)
	}
	row := func(name string, e []float64, n int) string {
		if len(e) == 0 {
			return fmt.Sprintf("\n  %-34s %5d notes, none known", name, n)
		}
		return fmt.Sprintf("\n  %-34s %5d notes, %5.1f%% known, median %5.3f, nine in ten within %5.3f", name, n, 100*float64(len(e))/float64(n), median(e), ninetieth(e))
	}
	for _, name := range []string{"fit", "left out", "test"} {
		checks := reads[name]
		var b strings.Builder
		b.WriteString("\n  ends, seconds:")
		for k, kind := range stringKinds {
			for _, room := range []string{"no room", "degrade"} {
				e := ends(checks, kind, room)
				b.WriteString(row(kind+", "+room, e, len(e)))
			}
			if e := ends(checks, kind, ""); len(e) >= 20 && name != "fit" && sure(e, false) > bounds[k] {
				t.Errorf("%s, ends %s: nine in ten lie within %.3f s at least, at odds of 19 to 1, over the bound of %.3f", name, kind, sure(e, false), bounds[k])
			}
		}
		b.WriteString("\n  levels, dB, on the microphone where a note sounds alone, and from its string:")
		for _, room := range []string{"no room", "degrade"} {
			for p := 1; p <= 6; p++ {
				var mic, pick []float64
				n, m := 0, 0
				for _, c := range checks {
					if c.room != room || c.poly != p && !(p == 6 && c.poly > 6) {
						continue
					}
					n++
					if c.micUnsure <= measuredBound {
						mic = append(mic, math.Abs(c.micErr))
					}
					if !c.fit {
						m++
						if c.pickUnsure <= measuredBound {
							pick = append(pick, math.Abs(c.pickErr))
						}
					}
				}
				if n == 0 {
					continue
				}
				mic, pick = slices.Sorted(slices.Values(mic)), slices.Sorted(slices.Values(pick))
				b.WriteString(row(fmt.Sprintf("%d at once, %s, microphone", p, room), mic, n) + row(fmt.Sprintf("%d at once, %s, string", p, room), pick, m))
				if p == 1 && len(mic) >= 20 && name != "fit" && sure(mic, false) > measuredBound {
					t.Errorf("%s, %s: of the %d notes alone known within %.1f dB on the microphone, nine in ten lie within %.2f at least, at odds of 19 to 1", name, room, len(mic), measuredBound, sure(mic, false))
				}
			}
		}
		t.Logf("%s, %d notes:%s", name, len(checks), b.String())
	}
}

// stringGain is a line's level on the microphone over its level on its own
// channel, in dB, of the notes both know: the median of their ratios, and how
// far it may lie from it at odds of 19 to 1 each way, by the binomial's order
// statistics; unbounded where too few notes say it
func stringGain(db []float64) (gain, unsure float64) {
	if len(db) == 0 {
		return 0, math.Inf(1)
	}
	db = slices.Sorted(slices.Values(db))
	n, mid := len(db), median(db)
	lo, hi, cdf := -1, n, 0.0
	for k := range n {
		nk, _ := math.Lgamma(float64(n) + 1)
		kk, _ := math.Lgamma(float64(k) + 1)
		rk, _ := math.Lgamma(float64(n-k) + 1)
		if cdf += math.Exp(nk - kk - rk - float64(n)*math.Ln2); cdf <= 0.05 {
			lo, hi = k, n-1-k
		}
	}
	if lo < 0 {
		return mid, math.Inf(1)
	}
	return mid, max(mid-db[lo], db[hi]-mid)
}

// TestAByStringChannelIsItsString: with GUITARSET, each string's notes alone,
// no other note's label within a window either side of their first window,
// hold the most at their partials there on their own string's channel, the
// channel of the place their pitch track names them in, nine in ten of each
// string's at least; and what they hold on each other channel over their own,
// its tenth, median and ninetieth percentile, is stringBleed's to the dB
func TestAByStringChannelIsItsString(t *testing.T) {
	g := os.Getenv("GUITARSET")
	if g == "" {
		t.Skip("GUITARSET names no folder")
	}
	names, err := filepath.Glob(filepath.Join(g, "annotation", "*.jams"))
	if err != nil || len(names) == 0 {
		t.Fatalf("%s: no JAMS files: %v", g, err)
	}
	rate := audio.AudioSampleRate
	w := int(sampledWindow * float64(rate))
	xw := make([]float64, w)
	var f []float64
	held := func(x []float64, a int) float64 {
		for k := range w {
			xw[k] = (0.5 - 0.5*math.Cos(2*math.Pi*(float64(k)+0.5)/float64(w))) * x[a+k]
		}
		return partialEnergy(xw, f)
	}
	own, alone := make([]int, 6), make([]int, 6)
	var bleed [6][6][]float64
	for _, jams := range names {
		clip := strings.TrimSuffix(filepath.Base(jams), ".jams")
		tracks, err := readPitch(jams)
		if err != nil {
			t.Fatal(err)
		}
		ch, at, err := readChannels(filepath.Join(g, "audio", "hex", clip+"_hex_cln.wav"))
		if err != nil || at != rate || len(ch) < len(tracks) {
			t.Fatalf("%s: %d channels at %d Hz: %v", clip, len(ch), at, err)
		}
		var score audio.MIDIScore
		var of []int
		for s, tr := range tracks {
			for _, a := range tr.notes {
				score.Notes, of = append(score.Notes, audio.RipNote{Start: a[0], End: a[1], MIDI: int(math.Floor(a[2] + 0.5))}), append(of, s)
			}
		}
		notes := heardNotes(&score, tracks, rate)
		for i, n := range notes {
			near := 0 // the notes whose labels come within a window either side, itself among them
			for _, m := range notes {
				if m.start < n.start+w && m.end > n.start-w {
					near++
				}
			}
			if near > 1 || n.start < w || n.start+w > len(ch[0]) {
				continue
			}
			f = partialsOf(n, n.start+w/2, rate, 0, f)
			e := make([]float64, len(ch))
			for c, x := range ch {
				e[c] = held(x, n.start)
			}
			s := of[i]
			alone[s]++
			if slices.Index(e, slices.Max(e)) == s {
				own[s]++
			}
			for c := range 6 {
				if c != s && e[s] > 0 && e[c] > 0 {
					bleed[s][c] = append(bleed[s][c], 10*math.Log10(e[c]/e[s]))
				}
			}
		}
	}
	for s := range 6 {
		t.Logf("string %d: %d of %d notes alone hold the most on its channel", s, own[s], alone[s])
		if alone[s] == 0 || 10*own[s] < 9*alone[s] {
			t.Errorf("string %d: %d of %d notes alone hold the most on its channel, want nine in ten", s, own[s], alone[s])
		}
		for c := range 6 {
			if c == s {
				continue
			}
			e := slices.Sorted(slices.Values(bleed[s][c]))
			at := func(p float64) float64 { return math.Round(e[max(int(math.Ceil(p*float64(len(e))))-1, 0)]) }
			if got := [3]float64{at(0.1), at(0.5), at(0.9)}; got != stringBleed[s][c] {
				t.Errorf("string %d on channel %d: %v dB, stringBleed says %v", s, c, got, stringBleed[s][c])
			}
		}
	}
}

// writeChannels writes channels as one WAV of 16-bit PCM, a channel each
func writeChannels(t *testing.T, file string, ch [][]float64) {
	t.Helper()
	n, c := len(ch[0]), len(ch)
	data, le := make([]byte, 44+2*n*c), binary.LittleEndian
	copy(data, "RIFF")
	le.PutUint32(data[4:], uint32(36+2*n*c))
	copy(data[8:], "WAVEfmt ")
	le.PutUint32(data[16:], 16)
	le.PutUint16(data[20:], 1)
	le.PutUint16(data[22:], uint16(c))
	le.PutUint32(data[24:], uint32(audio.AudioSampleRate))
	le.PutUint32(data[28:], uint32(2*c*audio.AudioSampleRate))
	le.PutUint16(data[32:], uint16(2*c))
	le.PutUint16(data[34:], 16)
	copy(data[36:], "data")
	le.PutUint32(data[40:], uint32(2*n*c))
	for i := range n {
		for k := range c {
			le.PutUint16(data[44+2*(i*c+k):], uint16(int16(math.Round(max(-1, min(1, ch[k][i]))*32767))))
		}
	}
	if err := os.WriteFile(file, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestACorpusFromListReadsANoteOnItsString: a list's strings column names a
// recording's by-string file, a channel a string, each at a gain of its own:
// a note ends where its string stops sounding it, and under measured a note
// alone by those ends is given its level on the microphone, one in a chord
// none; an empty cell labels the recording as without the column, byte for
// byte, and a file of fewer channels than strings is counted by reason
func TestACorpusFromListReadsANoteOnItsString(t *testing.T) {
	dir := t.TempDir()
	at := func(name string) string { return filepath.Join(dir, name) }
	sr := float64(audio.AudioSampleRate)
	type laid struct {
		str, key     int
		start, label float64
		sounds, amp  float64
	}
	var laids []laid
	for k := range 6 {
		laids = append(laids, laid{0, 52 + k%3, 0.3 + 2*float64(k), 0.3, 0.6, 0.1 + 0.05*float64(k)}, laid{1, 59 + k%2, 1.3 + 2*float64(k), 0.3, 0.6, 0.3 - 0.04*float64(k)})
	}
	laids = append(laids, laid{0, 52, 12.3, 0.5, 1, 0.2}, laid{1, 59, 12.31, 0.5, 1, 0.08})
	total := int(14 * sr)
	mic, ch := make([]float64, total), [][]float64{make([]float64, total), make([]float64, total)}
	gain := []float64{2, 0.5}
	var notes []audio.RipNote
	type obs = map[string]any
	held, contour := [2][]obs{}, [2][]obs{}
	for _, l := range laids {
		f, a := audio.NoteFreq(l.key), int(l.start*sr)
		for i := range int(l.sounds * sr) {
			sec := float64(i) / sr
			v := 0.0
			for p := 1.0; p*f < 7000; p++ {
				v += math.Sin(2*math.Pi*p*f*sec) / p
			}
			v *= l.amp * math.Exp(-sec/0.5) * min(1, (l.sounds-sec)/0.005)
			mic[a+i] += v
			for c := range ch {
				g := 0.01
				if c == l.str {
					g = 1
				}
				ch[c][a+i] += g * gain[l.str] * v
			}
		}
		notes = append(notes, audio.RipNote{Start: l.start, End: l.start + l.label, MIDI: l.key, Vel: 0.8})
		held[l.str] = append(held[l.str], obs{"time": l.start, "duration": l.label, "value": float64(l.key)})
		for fr := int(l.start / audio.RipFrameSec); float64(fr)*audio.RipFrameSec < l.start+l.label; fr++ {
			contour[l.str] = append(contour[l.str], obs{"time": float64(fr) * audio.RipFrameSec, "duration": 0.0, "value": obs{"index": 0, "frequency": f, "voiced": true}})
		}
	}
	var anns []obs
	for s := range 2 {
		src := obs{"data_source": strconv.Itoa(s)}
		anns = append(anns, obs{"namespace": "note_midi", "annotation_metadata": src, "data": held[s]}, obs{"namespace": "pitch_contour", "annotation_metadata": src, "data": contour[s]})
	}
	jam, err := json.Marshal(obs{"annotations": anns})
	if err != nil {
		t.Fatal(err)
	}
	writeFile(t, at("a.jams"), string(jam))
	var mid bytes.Buffer
	if err := audio.WriteMIDI(&mid, notes, nil, 120, audio.Meter{}); err != nil {
		t.Fatal(err)
	}
	writeFile(t, at("a.mid"), mid.String())
	f, err := os.Create(at("a.wav"))
	if err != nil {
		t.Fatal(err)
	}
	audio.WriteWAV(f, mic)
	f.Close()
	writeChannels(t, at("a-hex.wav"), ch)
	writeChannels(t, at("one-hex.wav"), [][]float64{mic})
	var out bytes.Buffer
	s := newTestSession(t, &out)
	defer teardown(t, s)
	label := func(corpus string) corpusLoop {
		var l corpusLoop
		data, _ := os.ReadFile(filepath.Join(corpus, "loop_000.json"))
		if err := json.Unmarshal(data, &l); err != nil {
			t.Fatal(err)
		}
		return l
	}
	writeFile(t, at("list.csv"), "audio_filename,midi_filename,pitch,velocity,strings\na.wav,a.mid,a.jams,measured,a-hex.wav\n")
	runOrFatal(t, s, &out, fmt.Sprintf("corpus %s from %s\n", at("strung"), at("list.csv")))
	if !strings.Contains(out.String(), "12 note(s) given the velocity measured and 2 none, the ends of 14 note(s) read on their strings") {
		t.Errorf("the run says %q", out.String())
	}
	l := label(at("strung"))
	var vel []float64
	for _, e := range l.Events {
		i := slices.IndexFunc(laids, func(m laid) bool { return m.key == e.MIDI && math.Abs(m.start/audio.RipFrameSec-float64(e.Frame)) < 1 })
		if i < 0 {
			t.Fatalf("a label's note at frame %d, key %d, laid nowhere", e.Frame, e.MIDI)
		}
		if want := (laids[i].start + laids[i].sounds) / audio.RipFrameSec; math.Abs(float64(e.End)-want) > sampledWindow/audio.RipFrameSec {
			t.Errorf("the note at %.2f s ends at frame %d, its sound at %.1f", laids[i].start, e.End, want)
		}
		if alone := laids[i].start < 12; alone != (e.Vel >= 0) {
			t.Errorf("the note at %.2f s given %.3f", laids[i].start, e.Vel)
		} else if alone {
			vel = append(vel, e.Vel/laids[i].amp)
		}
	}
	if lo, hi := slices.Min(vel), slices.Max(vel); lo <= 0 || 20*math.Log10(hi/lo) > 0.5 {
		t.Errorf("velocities over the amplitudes laid from %.4f to %.4f, want one ratio within 0.5 dB", lo, hi)
	}
	// an empty cell, and a file of fewer channels than strings
	writeFile(t, at("plain.csv"), "audio_filename,midi_filename,pitch,velocity\na.wav,a.mid,a.jams,none\n")
	writeFile(t, at("list.csv"), "audio_filename,midi_filename,pitch,velocity,strings\na.wav,a.mid,a.jams,none,\n")
	runOrFatal(t, s, &out, fmt.Sprintf("corpus %s from %s\ncorpus %s from %s\n", at("plain"), at("plain.csv"), at("empty"), at("list.csv")))
	for _, name := range []string{"loop_000.json", "loop_000.bands", "corpus.json"} {
		a, _ := os.ReadFile(filepath.Join(at("plain"), name))
		b, _ := os.ReadFile(filepath.Join(at("empty"), name))
		if len(a) == 0 || !bytes.Equal(a, b) {
			t.Errorf("%s with an empty strings cell differs from the list without the column", name)
		}
	}
	writeFile(t, at("list.csv"), "audio_filename,midi_filename,pitch,velocity,strings\na.wav,a.mid,a.jams,measured,one-hex.wav\n")
	out.Reset()
	runOrFatal(t, s, &out, fmt.Sprintf("corpus %s from %s labels\n", at("strung"), at("list.csv")))
	if !strings.Contains(out.String(), "0 note(s) given the velocity measured and 14 none, 14 of those none in recording(s) not measured: 1 whose channels are fewer than its strings") {
		t.Errorf("one channel: the run says %q", out.String())
	}
}
