package stapes

import (
	"bytes"
	"cmp"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/lixenwraith/audio"
)

// heardCheck is a drawn note's measure against its level alone: the error in
// dB, its bound's half-width, the most notes sounding with it at once and the
// most coinciding interval it sounds at with another
type heardCheck struct {
	voice       string
	err, unsure float64
	poly        int
	interval    string
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
		type laid struct {
			n    heardNote
			midi int
			name string
			x    []float64
		}
		var notes []laid
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
					at, end := int(nt.at), min(int(nt.at+nt.dur), m.to)
					key := float64(midi)
					n := heardNote{at, end, min(end+int(measuredRing*float64(rate)), m.to), func(int) float64 { return key }}
					notes = append(notes, laid{n, midi, v.Name, audio.RenderNote(v, midi, nt.vel, nt.dur/float64(rate), 1)})
				}
			}
		}
		mix := make([]float64, m.to)
		for _, l := range notes {
			for j, v := range l.x[:min(len(l.x), m.to-l.n.start)] {
				mix[l.n.start+j] += v
			}
		}
		carry := func(int, []float64) {}
		if degraded {
			mix, carry = room(mix, rand.New(rand.NewPCG(seed^0xd, uint64(i))))
		}
		heard := make([]heardNote, len(notes))
		for j, l := range notes {
			heard[j] = l.n
		}
		got := hearLevels(mix, rate, heard)
		for j, l := range notes {
			span := heldSpan(l.n, rate, m.to)
			alone := make([]float64, span)
			copy(alone, l.x)
			carry(l.n.start, alone)
			truth := slices.Max(windows(span, rate, sampledWindow, meanSquare(alone)))
			if truth <= 0 || got[j] == (heardLevel{}) {
				continue
			}
			c := heardCheck{voice: l.name, err: 20 * math.Log10(got[j].level()/truth), unsure: got[j].unsure(), poly: 1, interval: "alone"}
			over := func(o heardNote) bool { return o.start < l.n.start+span && o.end > l.n.start }
			for q, o := range notes {
				if q == j || !over(o.n) {
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

// fitBound is the tightest bound, in tenths of a dB, that nine in ten of the
// notes known within it are measured within, of twenty at least
func fitBound(checks []heardCheck) float64 {
	for d := 1; d <= 60; d++ {
		b := float64(d) / 10
		if n, p90 := within(checks, b); n >= 20 && p90 <= b {
			return b
		}
	}
	return math.Inf(1)
}

// heardTable is checks by how many notes sound at once and by interval: each
// row's notes, the share given a velocity at bound and those's median and
// ninetieth percentile of error in dB
func heardTable(checks []heardCheck, bound float64) string {
	var b strings.Builder
	row := func(name string, keep func(heardCheck) bool) {
		var of []heardCheck
		for _, c := range checks {
			if keep(c) {
				of = append(of, c)
			}
		}
		n, p90 := within(of, bound)
		var e []float64
		for _, c := range of {
			if c.unsure <= bound {
				e = append(e, math.Abs(c.err))
			}
		}
		if n == 0 {
			fmt.Fprintf(&b, "\n  %-8s %4d notes %4.0f%%", name, len(of), 0.0)
			return
		}
		fmt.Fprintf(&b, "\n  %-8s %4d notes %4.0f%% median %4.2f p90 %5.2f", name, len(of), 100*float64(n)/float64(max(len(of), 1)), median(e), p90)
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

// TestAMeasuredLevelIsKnownWithinItsFittedBound: the notes of V's overlaps on
// the engine's voices, with no room and through degrade, measured against each
// alone: measuredBound is the tightest bound nine in ten of the notes known
// within it are, and holds on another seed's and on voices left out of the fit
func TestAMeasuredLevelIsKnownWithinItsFittedBound(t *testing.T) {
	var out bytes.Buffer
	s := newTestSession(t, &out)
	defer teardown(t, s)
	const loops = 3
	draw := func(seed uint64, voices []*audio.VoiceDef) (clean, roomed []heardCheck) {
		return drawHeard(t, s, seed, loops, false, voices), drawHeard(t, s, seed, loops, true, voices)
	}
	clean, roomed := draw(1, nil)
	b := fitBound(append(slices.Clone(clean), roomed...))
	t.Logf("fitted on seed 1: %.1f dB\nno room:%s\ndegrade:%s", b, heardTable(clean, b), heardTable(roomed, b))
	if b != measuredBound {
		t.Errorf("the fit gives %.1f dB, measuredBound is %.1f", b, measuredBound)
	}
	clean, roomed = draw(2, nil)
	leftClean, leftRoomed := draw(1, focusVoices(t, s))
	for _, c := range []struct {
		name   string
		checks []heardCheck
	}{{"seed 2", append(clean, roomed...)}, {"left out", append(leftClean, leftRoomed...)}} {
		n, p90 := within(c.checks, measuredBound)
		t.Logf("%s: %d of %d notes known within %.1f dB, nine in ten of them within %.2f", c.name, n, len(c.checks), measuredBound, p90)
		if p90 > measuredBound {
			t.Errorf("%s: nine in ten of the notes known within %.1f dB are within %.2f", c.name, measuredBound, p90)
		}
	}
	t.Logf("left out, no room:%s\nleft out, degrade:%s", heardTable(leftClean, measuredBound), heardTable(leftRoomed, measuredBound))
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
