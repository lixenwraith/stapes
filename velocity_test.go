package stapes

import (
	"bytes"
	"cmp"
	"fmt"
	"maps"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/lixenwraith/audio"
)

// heardCheck is a drawn note's measure against its level alone: what plays it,
// the error in dB, its bound's half-width, the most notes sounding with it at
// once and the most coinciding interval it sounds at with another
type heardCheck struct {
	voice       string
	err, unsure float64
	poly        int
	interval    string
}

// heardLaid is a note laid alone for a draw: as it is measured, its key, what
// plays it and its samples from its start
type heardLaid struct {
	n    heardNote
	midi int
	name string
	x    []float64
}

// measureLaid mixes notes laid alone over to samples, through a room rng draws
// where it is set, and measures each against its level alone carried through
// the same room
func measureLaid(notes []heardLaid, to int, rng *rand.Rand) []heardCheck {
	rate := audio.AudioSampleRate
	mix := make([]float64, to)
	for _, l := range notes {
		for j, v := range l.x[:min(len(l.x), to-l.n.start)] {
			mix[l.n.start+j] += v
		}
	}
	carry := func(int, []float64) {}
	if rng != nil {
		mix, carry = room(mix, rng)
	}
	heard := make([]heardNote, len(notes))
	for j, l := range notes {
		heard[j] = l.n
		heard[j].rings = min(l.n.rings, to)
	}
	got := hearLevels(mix, rate, heard)
	var out []heardCheck
	for j, l := range notes {
		span := heldSpan(l.n, rate, to)
		alone := make([]float64, span)
		copy(alone, l.x)
		carry(l.n.start, alone)
		truth := slices.Max(windows(span, rate, sampledWindow, meanSquare(alone)))
		if truth <= 0 {
			continue
		}
		c := heardCheck{voice: l.name, err: 20 * math.Log10(got[j].level()/truth), unsure: got[j].unsure(), poly: 1, interval: "alone"}
		for q, o := range notes {
			if q == j || o.n.start >= l.n.start+span || o.n.end <= l.n.start {
				continue
			}
			c.interval = nearer(c.interval, abs(o.midi-l.midi))
			at, n := max(o.n.start, l.n.start), 0
			for _, p := range notes {
				if p.n.start <= at && p.n.end > at {
					n++
				}
			}
			c.poly = max(c.poly, n)
		}
		out = append(out, c)
	}
	return out
}

// roomOf is the room drawn for a draw's loop i, none where it is not degraded
func roomOf(degraded bool, seed, i uint64) *rand.Rand {
	if !degraded {
		return nil
	}
	return rand.New(rand.NewPCG(seed^0xd, i))
}

// drawHeard measures each note of loops of V's overlaps drawn at seed against
// its level alone, carried through the same room when degraded. With voices, a
// loop's parts play them in turn, each note moved by octaves into the keys the
// arranger gives its voice.
func drawHeard(t *testing.T, s *Session, seed uint64, loops int, degraded bool, voices []*audio.VoiceDef) []heardCheck {
	t.Helper()
	rate := audio.AudioSampleRate
	var out []heardCheck
	for i := range loops {
		m, err := s.overlap(rand.New(rand.NewPCG(seed, uint64(i))), nil)
		if err != nil {
			t.Fatal(err)
		}
		var notes []heardLaid
		k := 0
		for _, ps := range m.parts {
			for _, p := range ps {
				v := audio.VoiceDefByName(audio.InstrumentName(p.instr))
				if voices != nil {
					v = voices[k%len(voices)]
				}
				k++
				keys := familyKeys(v.Name)
				for _, nt := range p.notes {
					midi := nt.midi
					for midi < keys[0] {
						midi += 12
					}
					for midi > keys[1] {
						midi -= 12
					}
					at, end, key := int(nt.at), min(int(nt.at+nt.dur), m.to), float64(midi)
					n := heardNote{at, end, end + int(measuredRing*float64(rate)), func(int) float64 { return key }}
					notes = append(notes, heardLaid{n, midi, v.Name, audio.RenderNote(v, midi, nt.vel, nt.dur/float64(rate), 1)})
				}
			}
		}
		out = append(out, measureLaid(notes, m.to, roomOf(degraded, seed, uint64(i)))...)
	}
	return out
}

// laySampled is a recorded note of in laid alone at a sample, let go after held
// samples, its sound kept for n; false where the sampler would not play it
func laySampled(t *testing.T, set *sampleSet, in *sampleInstr, at, midi int, vel float64, held, n int, cache map[sampleKey][]float64) (heardLaid, bool) {
	t.Helper()
	e := audio.RipEvent{For: held, MIDI: midi, Vel: vel}
	if ok, err := set.fits(in, []audio.RipEvent{e}); err != nil || !ok {
		return heardLaid{}, false
	}
	ends, x, err := set.play(in, []audio.RipEvent{e}, n, 0, cache, true)
	if err != nil {
		t.Fatal(err)
	}
	end, key := at+ends[0], float64(midi)
	return heardLaid{heardNote{at, end, end + int(measuredRing*float64(audio.AudioSampleRate)), func(int) float64 { return key }}, midi, in.name, x}, true
}

// drawSampled is drawHeard on recorded notes: each part of a loop is played by
// an instrument of ins the seed draws, each note moved by octaves into its
// recorded pitches and left out where the sampler would not play it
func drawSampled(t *testing.T, s *Session, set *sampleSet, ins []*sampleInstr, seed uint64, loops int, degraded bool) []heardCheck {
	t.Helper()
	var out []heardCheck
	for i := range loops {
		m, err := s.overlap(rand.New(rand.NewPCG(seed, uint64(i))), nil)
		if err != nil {
			t.Fatal(err)
		}
		pick, cache := rand.New(rand.NewPCG(seed^0x5a, uint64(i))), map[sampleKey][]float64{}
		var notes []heardLaid
		for _, ps := range m.parts {
			for _, p := range ps {
				in := ins[pick.IntN(len(ins))]
				for _, nt := range p.notes {
					midi := nt.midi
					for midi < in.pitches[0] {
						midi += 12
					}
					for midi > in.pitches[len(in.pitches)-1] {
						midi -= 12
					}
					at := int(nt.at)
					if l, ok := laySampled(t, set, in, at, midi, nt.vel, min(int(nt.dur), m.to-at), m.to-at, cache); ok {
						notes = append(notes, l)
					}
				}
			}
		}
		out = append(out, measureLaid(notes, m.to, roomOf(degraded, seed, uint64(i)))...)
	}
	return out
}

// drawAlone lays count notes of ins one at a time, two seconds of nothing after
// each, twenty a stretch, at an instrument, a recorded pitch, a velocity 0 to
// 30 dB down and a length of 0.12 to 2.5 s the seed draws, as V's overlaps do,
// and measures each against its level alone
func drawAlone(t *testing.T, set *sampleSet, ins []*sampleInstr, seed uint64, count int, degraded bool) []heardCheck {
	t.Helper()
	rate := audio.AudioSampleRate
	rng := rand.New(rand.NewPCG(seed, 0xa1))
	var out []heardCheck
	for stretch := uint64(0); count > 0; stretch++ {
		var notes []heardLaid
		at, cache := rate/10, map[sampleKey][]float64{}
		for len(notes) < 20 && count > 0 {
			in := ins[rng.IntN(len(ins))]
			midi, vel, n := in.pitches[rng.IntN(len(in.pitches))], math.Pow(10, -1.5*rng.Float64()), int((0.12+2.38*rng.Float64())*float64(rate))
			if l, ok := laySampled(t, set, in, at, midi, vel, n, n+2*rate, cache); ok {
				notes, at, count = append(notes, l), at+n+2*rate, count-1
			}
		}
		out = append(out, measureLaid(notes, at, roomOf(degraded, seed, stretch))...)
	}
	return out
}

// intervals are the kinds a note's partials coincide with another's by, the most first
var intervals = []string{"unison", "octave", "fifth", "third", "other", "alone"}

// nearer is the more coinciding of a kind and the interval of d semitones
func nearer(kind string, d int) string {
	k := "other"
	switch {
	case d == 0:
		k = "unison"
	case d%12 == 0:
		k = "octave"
	case d%12 == 7 || d%12 == 5:
		k = "fifth"
	case d%12 == 3 || d%12 == 4 || d%12 == 8 || d%12 == 9:
		k = "third"
	}
	if slices.Index(intervals, k) < slices.Index(intervals, kind) {
		return k
	}
	return kind
}

// within is how many of checks are known within bound, and the ninetieth
// percentile of their errors' size
func within(checks []heardCheck, bound float64) (n int, p90 float64) {
	var e []float64
	for _, c := range checks {
		if c.unsure <= bound {
			e = append(e, math.Abs(c.err))
		}
	}
	if len(e) == 0 {
		return 0, math.Inf(1)
	}
	return len(e), ninetieth(e)
}

// sure is the order statistic of e, sorted, that bounds its ninetieth
// percentile from above, or from below, at odds of 19 to 1, by a binomial
func sure(e []float64, above bool) float64 {
	n, cdf, lower := float64(len(e)), 0.0, 0
	for k := range e {
		nk, _ := math.Lgamma(n + 1)
		kk, _ := math.Lgamma(float64(k) + 1)
		rk, _ := math.Lgamma(n - float64(k) + 1)
		if cdf += math.Exp(nk - kk - rk + float64(k)*math.Log(0.9) + (n-float64(k))*math.Log(0.1)); above && cdf >= 0.95 {
			return e[k]
		} else if cdf <= 0.05 {
			lower = k
		}
	}
	if above {
		return e[len(e)-1]
	}
	return e[lower]
}

// knownErrs is the size of the errors of the notes of checks known within bound, sorted
func knownErrs(checks []heardCheck, bound float64) []float64 {
	var e []float64
	for _, c := range checks {
		if c.unsure <= bound {
			e = append(e, math.Abs(c.err))
		}
	}
	slices.Sort(e)
	return e
}

// fitBound is the least bound, in tenths of a dB, within which nine in ten of
// the notes known within it are measured in every group that knows twenty,
// sure to one in twenty: its ninetieth percentile's bound from above is within it
func fitBound(groups map[string][]heardCheck) float64 {
	for d := 1; d <= 60; d++ {
		b, holds := float64(d)/10, true
		for _, checks := range groups {
			if e := knownErrs(checks, b); len(e) >= 20 {
				holds = holds && sure(e, true) <= b
			}
		}
		if holds {
			return b
		}
	}
	return math.Inf(1)
}

// heardRow is a group's notes, the share known within bound and those's median
// and ninetieth percentile of error in dB
func heardRow(name string, checks []heardCheck, bound float64) string {
	n, p90 := within(checks, bound)
	var e []float64
	for _, c := range checks {
		if c.unsure <= bound {
			e = append(e, math.Abs(c.err))
		}
	}
	if n == 0 {
		return fmt.Sprintf("\n  %-26s %4d notes %5.1f%%", name, len(checks), 0.0)
	}
	return fmt.Sprintf("\n  %-26s %4d notes %5.1f%% median %4.2f p90 %5.2f", name, len(checks), 100*float64(n)/float64(len(checks)), median(e), p90)
}

// heardTable is checks by how many notes sound at once and by interval
func heardTable(checks []heardCheck, bound float64) string {
	var b strings.Builder
	row := func(name string, keep func(heardCheck) bool) {
		var of []heardCheck
		for _, c := range checks {
			if keep(c) {
				of = append(of, c)
			}
		}
		b.WriteString(heardRow(name, of, bound))
	}
	for p := 1; p <= 8; p++ {
		row(fmt.Sprintf("%d", p), func(c heardCheck) bool { return c.poly == p || p == 8 && c.poly > 8 })
	}
	for _, k := range intervals {
		row(k, func(c heardCheck) bool { return c.interval == k })
	}
	row("all", func(heardCheck) bool { return true })
	return b.String()
}

// nsynthReads draws NSynth's notes of four families at seed 1, alone and in V's
// overlaps, with no room and through degrade, for three reads: a seeded half of
// the training split's instruments (fit), the other half (left out) and the
// test split, each a group by family or the four, alone or with others, and room
func nsynthReads(t *testing.T, s *Session, dir string) map[string]map[string][]heardCheck {
	t.Helper()
	train, err := loadSampleSet(filepath.Join(dir, "nsynth-train", "audio"))
	if err != nil {
		t.Fatal(err)
	}
	test, err := loadSampleSet(filepath.Join(dir, "nsynth-test", "audio"))
	if err != nil {
		t.Fatal(err)
	}
	reads := map[string]map[string][]heardCheck{"fit": {}, "left out": {}, "test": {}}
	for _, fam := range []string{"guitar", "vocal", "keyboard", "string"} {
		of := func(set *sampleSet) []*sampleInstr {
			var ins []*sampleInstr
			for _, in := range set.other {
				if strings.HasPrefix(in.name, fam+"_") {
					ins = append(ins, in)
				}
			}
			slices.SortFunc(ins, func(a, b *sampleInstr) int { return strings.Compare(a.name, b.name) })
			return ins
		}
		all := of(train)
		rand.New(rand.NewPCG(11, 1)).Shuffle(len(all), func(i, j int) { all[i], all[j] = all[j], all[i] })
		for name, from := range map[string]struct {
			set *sampleSet
			ins []*sampleInstr
		}{"fit": {train, all[:len(all)/2]}, "left out": {train, all[len(all)/2:]}, "test": {test, of(test)}} {
			for d, how := range []string{"no room", "degrade"} {
				for _, c := range append(drawAlone(t, from.set, from.ins, 1, 150, d == 1), drawSampled(t, s, from.set, from.ins, 1, 12, d == 1)...) {
					kind := " alone, "
					if c.poly > 1 {
						kind = " with others, "
					}
					for _, g := range []string{fam + kind + how, "the four" + kind + how} {
						reads[name][g] = append(reads[name][g], c)
					}
				}
			}
		}
	}
	return reads
}

// TestAMeasuredLevelIsKnownWithinItsFittedBound: notes measured against each
// alone. With NSYNTH, the folder of NSynth's splits, measuredBound is fitBound's
// on nsynthReads' fit, and no group of the two left out shows at 19 to 1 that
// more than a tenth of its known notes miss it; it holds on the engine's voices'
// overlaps, the bank's and the focus's, where no note sounding with another is known
func TestAMeasuredLevelIsKnownWithinItsFittedBound(t *testing.T) {
	var out bytes.Buffer
	s := newTestSession(t, &out)
	defer teardown(t, s)
	if dir := os.Getenv("NSYNTH"); dir != "" {
		reads := nsynthReads(t, s, dir)
		b := fitBound(reads["fit"])
		if b != measuredBound {
			t.Errorf("the fit gives %.1f dB, measuredBound is %.1f", b, measuredBound)
		}
		for _, name := range []string{"fit", "left out", "test"} {
			var rows strings.Builder
			for _, g := range slices.Sorted(maps.Keys(reads[name])) {
				rows.WriteString(heardRow(g, reads[name][g], b))
				if e := knownErrs(reads[name][g], b); len(e) >= 20 && sure(e, false) > b {
					t.Errorf("%s, %s: of the %d notes known within %.1f dB, nine in ten lie within %.2f at least, at odds of 19 to 1", name, g, len(e), b, sure(e, false))
				}
			}
			t.Logf("%s, at the bound fitted, %.1f dB:%s", name, b, rows.String())
		}
	}
	for _, c := range []struct {
		name   string
		voices []*audio.VoiceDef
	}{{"the bank's voices", nil}, {"the focus's voices", focusVoices(t, s)}} {
		checks := append(drawHeard(t, s, 1, 3, false, c.voices), drawHeard(t, s, 1, 3, true, c.voices)...)
		for _, k := range checks {
			if k.poly > 1 && k.unsure <= measuredBound {
				t.Errorf("%s: a note sounding with %d others known within %.2f dB, want none", c.name, k.poly-1, k.unsure)
			}
		}
		n, p90 := within(checks, measuredBound)
		t.Logf("%s: %d of %d notes known within %.1f dB, nine in ten of them within %.2f%s", c.name, n, len(checks), measuredBound, p90, heardTable(checks, measuredBound))
		if n > 0 && p90 > measuredBound {
			t.Errorf("%s: nine in ten of the notes known within %.1f dB are within %.2f", c.name, measuredBound, p90)
		}
	}
}

// TestEveryMeasuredLevelIsFinite: a fast sung scale's cut drawn as harmonic
// tones, each note ringing a quarter second into the next one's windows, where
// a ring that ends in a window couples its partials wider than a window's
// bins: each level and bound is finite, and a velocity of one that is not, or
// of a unit that is not, is none
func TestEveryMeasuredLevelIsFinite(t *testing.T) {
	rate := audio.AudioSampleRate
	keys := []float64{56, 54, 53, 50.845, 49}
	starts := []float64{0.1010, 0.3219, 0.3687, 0.4729, 0.6240}
	ends := []float64{0.3219, 0.3687, 0.4729, 0.6240, 1.2042}
	for _, shift := range []float64{0.92, 0.95, 0.98} {
		x := make([]float64, int(1.6*float64(rate)))
		var notes []heardNote
		for i, k := range keys {
			key, f := k+shift, audio.NoteFreq(0)*math.Exp2((k+shift)/12)
			at, end := int(starts[i]*float64(rate)), int(ends[i]*float64(rate))
			rings := end + rate/4
			for j := at; j < rings; j++ {
				sec := float64(j-at) / float64(rate)
				for p := 1.0; p*f < 7000; p++ {
					x[j] += 0.1 * math.Sin(2*math.Pi*p*f*sec) / p * math.Exp(-sec/0.5)
				}
			}
			notes = append(notes, heardNote{at, end, rings, func(int) float64 { return key }})
		}
		for i, h := range hearLevels(x, rate, notes) {
			if !finite(h.lo) || !finite(h.hi) {
				t.Errorf("shifted %+.2f: the note at %.2f s measured between %g and %g", shift, starts[i], h.lo, h.hi)
			}
		}
	}
	for _, c := range []struct {
		h    heardLevel
		unit float64
	}{{heardLevel{math.NaN(), 1}, 1}, {heardLevel{1, math.Inf(1)}, 1}, {heardLevel{1, 1}, math.NaN()}, {heardLevel{1, 1}, math.Inf(1)}} {
		if v := heardVelocity(c.h, c.unit); v != -1 {
			t.Errorf("between %g and %g over a unit of %g: velocity %g, want none", c.h.lo, c.h.hi, c.unit, v)
		}
	}
}

// focusVoices registers the plucked and vocal focus's voices, those of the
// trainer's corpora no test session holds
func focusVoices(t *testing.T, s *Session) []*audio.VoiceDef {
	t.Helper()
	var voices []*audio.VoiceDef
	for _, f := range []string{"tool/rip-train/focus/plucked.toml", "tool/rip-train/focus/vocal.toml"} {
		data, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		defs, err := audio.LoadSoundsTOML(data)
		if err != nil {
			t.Fatal(err)
		}
		for _, v := range defs.Voice {
			if _, err := s.eng.DefineVoice(v.Clone()); err != nil {
				t.Fatal(err)
			}
			voices = append(voices, v)
		}
	}
	return voices
}

// harmonicSet is a recorded set the sampler plays: at each key from 40 to 80,
// a tone of partials falling as their number to 7 kHz, decaying, at a level
// each key its own, held 1.5 s and released over half a second
func harmonicSet(t *testing.T) *sampleSet {
	t.Helper()
	dir := t.TempDir()
	if err := writeNotes(dir, "keyboard_acoustic_000", 40, 80, 1, func(p int) []float64 {
		x, f := make([]float64, 2*nsynthRate), audio.NoteFreq(p)
		amp := 0.05 + 0.1*float64(p*7%10)/10
		for i := range x {
			sec := float64(i) / nsynthRate
			s1, c := math.Sin(2*math.Pi*f*sec), 2*math.Cos(2*math.Pi*f*sec)
			var v, s0 float64 // sin kθ by the recurrence sin(k+1)θ = 2cosθ sin kθ - sin(k-1)θ
			for k := 1.0; k*f < 7000; k++ {
				v, s0, s1 = v+s1/k, s1, c*s1-s0
			}
			x[i] = amp * v * math.Exp(-sec/1.5) * min(1, 4-2*sec)
		}
		return x
	}); err != nil {
		t.Fatal(err)
	}
	set, err := loadSampleSet(dir)
	if err != nil {
		t.Fatal(err)
	}
	return set
}

// lay is a line of notes on the sampler's recorded instrument at the keys,
// velocities and lengths given, each alone, and the notes as measured
func lay(t *testing.T, set *sampleSet, keys []int, vels, secs []float64) ([]float64, []heardNote) {
	t.Helper()
	rate := float64(audio.AudioSampleRate)
	var events []audio.RipEvent
	var notes []heardNote
	at := int(0.1 * rate)
	for i, k := range keys {
		n := int(secs[i] * rate)
		events = append(events, audio.RipEvent{At: at, For: n, MIDI: k, Vel: vels[i]})
		notes = append(notes, heardNote{at, at + n, at + n + int(measuredRing*rate), func(int) float64 { return float64(k) }})
		at += n + int(rate)
	}
	_, x, err := set.play(set.other[0], events, at, 0, map[sampleKey][]float64{}, true)
	if err != nil {
		t.Fatal(err)
	}
	return x, notes
}

// TestALineLaidAloneIsMeasuredAtItsVelocities: recorded notes the sampler lays
// alone at drawn keys, velocities and lengths are measured back at their
// velocities within 0.2 dB, the sampler's level and the measure's one
// definition; what is left is what the window's partials cross, a low note's
func TestALineLaidAloneIsMeasuredAtItsVelocities(t *testing.T) {
	set := harmonicSet(t)
	rng := rand.New(rand.NewPCG(7, 1))
	var keys []int
	var vels, secs []float64
	for range 24 {
		keys, vels, secs = append(keys, 40+rng.IntN(41)), append(vels, 0.05+0.95*rng.Float64()), append(secs, 0.1+1.4*rng.Float64())
	}
	x, notes := lay(t, set, keys, vels, secs)
	worst := 0.0
	for i, h := range hearLevels(x, audio.AudioSampleRate, notes) {
		db := 20 * math.Log10(h.level()/vels[i])
		worst = max(worst, math.Abs(db))
		if math.Abs(db) > 0.2 || h.unsure() > 0.1 {
			t.Errorf("key %d at %.2f for %.2f s measured %+.2f dB off, within %.2f", keys[i], vels[i], secs[i], db, h.unsure())
		}
	}
	t.Logf("the worst of %d notes %.3f dB off", len(notes), worst)
}

// TestAVoicesLevelIsStraightInItsVelocityWhereItsToneIsNot: each wave, the
// bank's voices and the focus's rendered over the velocities: the level of the
// loudest window rises a dB a dB of velocity within 0.05 dB on a voice no
// hammer, knock or filter envelope moves with the velocity; the others' slopes
// are logged, a measured velocity reading their notes by the straight law
func TestAVoicesLevelIsStraightInItsVelocityWhereItsToneIsNot(t *testing.T) {
	var out bytes.Buffer
	s := newTestSession(t, &out)
	defer teardown(t, s)
	voices := slices.Clone(audio.RegisteredVoices())
	for _, w := range audio.VoiceWaves {
		voices = append(voices, &audio.VoiceDef{Name: w, Wave: w, Attack: 0.002, Sustain: 1, Release: 0.1, Level: 1})
	}
	voices = append(voices, focusVoices(t, s)...)
	vels := []float64{softest, 0.0625, 0.125, 0.25, 0.5}
	for _, v := range voices {
		bent := v.HammerHz > 0 || v.Knock > 0 || v.EnvAmount > 0
		lo, hi, off := math.Inf(1), math.Inf(-1), 0.0
		for _, key := range []int{36, 48, 60, 72, 84} {
			db := func(vel float64) float64 {
				return 20 * math.Log10(loudest(audio.RenderNote(v, key, vel, 0.5, 0), sampledWindow))
			}
			full := db(1)
			for _, vel := range vels {
				d, by := db(vel)-full, 20*math.Log10(vel)
				lo, hi, off = min(lo, d/by), max(hi, d/by), max(off, math.Abs(d-by))
			}
		}
		t.Logf("%-12s %-6s %.3f to %.3f dB a dB, at most %.2f dB off straight", v.Name, cmp.Or(v.Wave, "sine"), lo, hi, off)
		if !bent && off > 0.05 {
			t.Errorf("%s, whose tone nothing moves with the velocity, is %.2f dB off straight", v.Name, off)
		}
	}
}

// TestAMeasuredVelocityPlaysAsLoudAsItWasPlayed: a line of the piano's at drawn
// keys and velocities, each note alone, measured and given velocities by its
// unit; recorded notes the sampler lays at those stand as far over or under
// the line's others as the piano's did, within 0.5 dB
func TestAMeasuredVelocityPlaysAsLoudAsItWasPlayed(t *testing.T) {
	var out bytes.Buffer
	s := newTestSession(t, &out)
	defer teardown(t, s)
	rate := audio.AudioSampleRate
	piano, rng := audio.VoiceDefByName("piano"), rand.New(rand.NewPCG(8, 1))
	var keys []int
	var labels, secs, x []float64
	var notes []heardNote
	at := rate / 10
	for range 24 {
		k, vel, sec := 48+rng.IntN(25), 0.1+0.9*rng.Float64(), 0.2+0.6*rng.Float64()
		y, n := audio.RenderNote(piano, k, vel, sec, 0.5), int(sec*float64(rate))
		x = append(x, make([]float64, max(at+len(y)-len(x), 0))...)
		for i, v := range y {
			x[at+i] += v
		}
		keys, labels, secs = append(keys, k), append(labels, vel), append(secs, sec)
		notes = append(notes, heardNote{at, at + n, at + n + int(measuredRing*float64(rate)), func(int) float64 { return float64(k) }})
		at += n + rate
	}
	levels := hearLevels(x, rate, notes)
	unit := heardUnit(levels, lineTop())
	vels := make([]float64, len(levels))
	for i, h := range levels {
		vels[i] = heardVelocity(h, unit)
		if vels[i] < 0 {
			t.Fatalf("key %d at %.2f: no velocity, its level known within %.2f dB", keys[i], labels[i], h.unsure())
		}
	}
	y, laid := lay(t, harmonicSet(t), keys, vels, secs)
	level := func(x []float64, n heardNote) float64 {
		seg := x[n.start : n.start+heldSpan(n, rate, len(x))]
		return 20 * math.Log10(slices.Max(windows(len(seg), rate, sampledWindow, meanSquare(seg))))
	}
	var a, b []float64
	for i := range notes {
		a, b = append(a, level(x, notes[i])), append(b, level(y, laid[i]))
	}
	var ma, mb float64
	for i := range a {
		ma, mb = ma+a[i]/float64(len(a)), mb+b[i]/float64(len(b))
	}
	worst := 0.0
	for i := range a {
		d := (b[i] - mb) - (a[i] - ma)
		worst = max(worst, math.Abs(d))
		if math.Abs(d) > 0.5 {
			t.Errorf("key %d at %.2f, measured at %.2f: %+.2f dB off the line's others as the piano played it", keys[i], labels[i], vels[i], d)
		}
	}
	t.Logf("unit %.4f; the worst of %d notes %.2f dB off", unit, len(a), worst)
}
