package stapes

import (
	"bytes"
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/lixenwraith/audio"
	"github.com/lixenwraith/genetic"
)

// fitOptions sizes a voice fit: the notes of a register it listens to, the
// generations run over a pool of candidate voices, the seed every draw follows
// and the candidates scored at once, the machine's cores at 0
type fitOptions struct {
	notes, generations, pool int
	seed                     uint64
	cores                    int
}

var defaultFit = fitOptions{notes: 6, generations: 8, pool: 16, seed: 1}

// fitWords are the words that size a fit on a verb's line, each before its
// number, with the numbers it takes
var fitWords = [...]struct {
	word   string
	lo, hi int
	set    func(*fitOptions, int)
}{
	{"notes", 1, 64, func(o *fitOptions, n int) { o.notes = n }},
	{"pool", 4, 1024, func(o *fitOptions, n int) { o.pool = n }},
	{"generations", 1, 500, func(o *fitOptions, n int) { o.generations = n }},
	{"seed", 0, math.MaxInt32, func(o *fitOptions, n int) { o.seed = uint64(n) }},
}

// size sets the size a word of fitWords names to its number
func (o *fitOptions) size(word, number string) error {
	for _, f := range fitWords {
		if f.word != word {
			continue
		}
		n, err := strconv.Atoi(number)
		if err != nil || n < f.lo || n > f.hi {
			return fmt.Errorf("%s %q: want %d to %d", word, number, f.lo, f.hi)
		}
		f.set(o, n)
		return nil
	}
	return fmt.Errorf("%q: want notes, pool, generations or seed, each before its number", word)
}

// fitTail is how long past a note's end its window runs, for the release
const fitTail = 0.25

// ripped is a recording ripped this session: what was heard, and the samples it
// was heard in, so fit and compress work on it without the file; a score read
// from a MIDI file has no samples
type ripped struct {
	rip   *audio.Rip
	x     []float64
	rate  int
	file  string
	sound *audio.RipSound // the recording as Fidelity hears it, heard on first use
}

// errScore refuses a verb that hears the recording a score lacks
func errScore(rp *ripped) error {
	return fmt.Errorf("%s is a score, with no recording to hear or fit to", rp.file)
}

func cmdFit(s *Session, a []string) error {
	rp, ok := s.rips[a[0]]
	if !ok {
		return fmt.Errorf("%q: nothing ripped under that name this session (rip first)", a[0])
	}
	o, w := s.fit, a[1:]
	if len(w)%2 == 1 { // a number alone is the generations
		if err := o.size("generations", w[0]); err != nil {
			return err
		}
		w = w[1:]
	}
	for ; len(w) > 0; w = w[2:] {
		if err := o.size(w[0], w[1]); err != nil {
			return err
		}
	}
	if rp.x == nil {
		return errScore(rp)
	}
	s.fitVoices(rp, o)
	for _, v := range rp.rip.Voices(a[0]) {
		if _, err := s.eng.DefineVoice(v.Clone()); err != nil {
			return fmt.Errorf("voice %q: %w", v.Name, err)
		}
		s.voices.put(v, false)
		fmt.Fprintf(s.out, "voice %s: %s, registered; the patterns naming it play it\n", v.Name, cmp.Or(v.Wave, "sine"))
	}
	return nil
}

// cmdNotes lists a rip's transcript as heard before the grid, so a listener checks
// it by eye: each note's bar and step, its pitch as the key's degree with any
// accidental, length, loudness and support, a melody-register note's brightness
// where measured (the header counts the rest), and the wave and sure family of a
// note told its voice; or writes the transcript to a MIDI file at the rip's tempo.
func cmdNotes(s *Session, a []string) error {
	rp, ok := s.rips[a[0]]
	if !ok {
		return fmt.Errorf("%q: nothing ripped under that name this session (rip first)", a[0])
	}
	r := rp.rip
	if len(a) > 1 {
		if !isMIDI(a[1]) {
			return fmt.Errorf("%s: want a .mid file", a[1])
		}
		var b bytes.Buffer
		hits := r.Hits()
		if err := audio.WriteMIDI(&b, r.Notes, hits, r.BPM, r.Meter); err != nil {
			return err
		}
		if err := os.WriteFile(s.path(a[1]), b.Bytes(), 0o644); err != nil {
			return err
		}
		fmt.Fprintf(s.out, "%s: %d note(s) and %d hit(s) at %d bpm in %s written to %s, a track each for melody, under, bass and drums\n", a[0], len(r.Notes), len(hits), r.BPM, r.Meter, a[1])
		return nil
	}
	// support is how much of the recording's salience over a note's frames is
	// its pitch's: one for a note alone in its frames, little for one heard
	// beside a louder, a note to doubt; a score's notes are given, not heard
	var snd *audio.RipSound
	if rp.x != nil {
		var err error
		if snd, err = s.soundOf(rp); err != nil {
			return err
		}
	}
	support := func(n audio.RipNote) string {
		if snd == nil {
			return ""
		}
		f, pitches := snd.Frames, snd.Frames.Pitches()
		var own, all float64
		for t := int(n.Start / f.Hop); t < int(n.End/f.Hop) && (t+1)*pitches.Len() <= len(f.Salience); t++ {
			row := f.Salience[t*pitches.Len() : (t+1)*pitches.Len()]
			if i := n.MIDI - pitches.Min; i >= 0 && i < len(row) {
				own += float64(row[i])
			}
			var peak float32
			for _, v := range row {
				peak = max(peak, v)
			}
			all += float64(peak)
		}
		if all > 0 {
			own /= all
		}
		return fmt.Sprintf(", support %.2f", own)
	}
	var upper, unmeasured int
	for _, n := range r.Notes {
		if !n.Bass {
			upper++
			if n.Bright == audio.RipUnmeasured {
				unmeasured++
			}
		}
	}
	fmt.Fprintf(s.out, "%s: %d note(s) at %d bpm in %s, %s %s, tuned %+.0f cents, the first bar from %.2fs, brightness unmeasured on %d of %d melody-register note(s)\n", a[0], len(r.Notes), r.BPM, r.Meter, noteName(r.Root), scaleName(r.Scale), r.Tuning, r.Downbeat, unmeasured, upper)
	marks := [...]string{"bb", "b", "", "#", "x"}
	beat, per := r.Meter.Beat(), r.Meter.Beats()
	for _, n := range r.Notes {
		line := "melody"
		switch {
		case n.Bass:
			line = "bass"
		case n.Second:
			line = "under"
		}
		where, span := "before the first bar", 60/float64(max(r.BPM, 1))/audio.StepsPerBeat
		if i := sort.SearchFloat64s(r.Beats, n.Start+1e-9) - 1; i >= 0 && i+1 < len(r.Beats) {
			span = (r.Beats[i+1] - r.Beats[i]) / float64(beat)
			where = fmt.Sprintf("bar %d step %2d", i/per+1, beat*(i%per)+int(math.Round((n.Start-r.Beats[i])/span)))
		}
		deg, oct, acc := audio.PitchStep(r.Root, r.Scale, n.MIDI)
		bright := ""
		if n.Bright != audio.RipUnmeasured {
			bright = fmt.Sprintf(", bright %.2f", n.Bright)
		}
		bend := ""
		if n.Bend != 0 {
			bend = fmt.Sprintf(", bend %+.2f from %.2f", n.Bend, n.BendAt)
		}
		voice := ""
		if v := n.Voice; v != nil {
			voice = ", wave " + v.Wave
			if v.Family != "" {
				voice += ", family " + v.Family
			}
		}
		fmt.Fprintf(s.out, "%-6s %7.3fs %-20s %-4s degree %d%s octave %d, %.1f step(s) at %.2f%s%s%s%s\n", line, n.Start, where, noteName(n.MIDI), deg, marks[acc+2], oct, (n.End-n.Start)/span, n.Vel, support(n), bright, bend, voice)
	}
	return nil
}

// ripLines are a rip's three lines in the order their voices are fitted: the
// melody, the line under it and the bass, each with the notes that are its
var ripLines = [3]struct {
	word string
	of   func(audio.RipNote) bool
}{
	{"melody", func(n audio.RipNote) bool { return !n.Bass && !n.Second }},
	{"under", func(n audio.RipNote) bool { return !n.Bass && n.Second }},
	{"bass", func(n audio.RipNote) bool { return n.Bass }},
}

// lineVoices are the rip's voices in ripLines' order
func lineVoices(r *audio.Rip) [3]**audio.VoiceDef {
	return [3]**audio.VoiceDef{&r.MelodyVoice, &r.UnderVoice, &r.BassVoice}
}

// fitVoices fits a voice to each line of a rip, a voice the front end heard
// for it held as a seed, and says how far the fit came
func (s *Session) fitVoices(rp *ripped, o fitOptions) {
	sc := s.ripScene(rp.x, rp.rate, rp.rip, o)
	for reg, at := range lineVoices(rp.rip) {
		notes := sc.lines[reg].notes
		if len(notes) == 0 {
			continue
		}
		word := ripLines[reg].word
		fmt.Fprintf(s.out, "fitting the %s voice over %d note(s), %d generation(s) of %d...\n", word, min(len(notes), o.notes), o.generations, o.pool)
		f := s.fitVoice(sc, reg, *at, o)
		if f == nil {
			continue
		}
		*at, sc.lines[reg].voice = f.voice, f.voice
		fmt.Fprintf(s.out, "%s voice: %s, distance %.3f from the best seed's %.3f\n", word, cmp.Or(f.voice.Wave, "sine"), f.to, f.from)
	}
}

// fitSamples picks the notes a register is fitted on: the longest and loudest
// first, no pitch twice until every pitch has a note, each at least 80 ms
func fitSamples(notes []audio.RipNote, n int) []audio.RipNote {
	var cands []audio.RipNote
	for _, note := range notes {
		if note.End-note.Start >= 0.08 {
			cands = append(cands, note)
		}
	}
	sort.SliceStable(cands, func(i, j int) bool {
		return (cands[i].End-cands[i].Start)*cands[i].Vel > (cands[j].End-cands[j].Start)*cands[j].Vel
	})
	var out []audio.RipNote
	seen := map[int]bool{}
	for pass := range 2 {
		for _, c := range cands {
			if len(out) >= n {
				return out
			}
			if (pass == 0 && !seen[c.MIDI]) || (pass == 1 && !slices.Contains(out, c)) {
				out = append(out, c)
				seen[c.MIDI] = true
			}
		}
	}
	return out
}

// voiceGenome is a voice as a search sees it: its genes and, last, its wave as a
// fraction of the wave table; voiceFromGenome reads one back
func voiceGenome(v *audio.VoiceDef) []float64 {
	w := max(slices.Index(audio.VoiceWaves[:], cmp.Or(v.Wave, "sine")), 0)
	return append(v.Genes(), (float64(w)+0.5)/float64(len(audio.VoiceWaves)))
}

func voiceFromGenome(g []float64) *audio.VoiceDef {
	genes := audio.VoiceGeneCount
	w := min(max(int(g[genes]*float64(len(audio.VoiceWaves))), 0), len(audio.VoiceWaves)-1)
	v := audio.VoiceFromGenes(audio.VoiceWaves[w], g[:genes])
	v.Name = "fitted"
	return v
}

// genomeKey names a genome to three decimals, for a cache of its score
func genomeKey(g []float64) string {
	var b strings.Builder
	for _, v := range g {
		fmt.Fprintf(&b, "%.3f,", v)
	}
	return b.String()
}

// lineNotes are a rip's notes of one of ripLines' registers
func lineNotes(r *audio.Rip, reg int) []audio.RipNote {
	var notes []audio.RipNote
	for _, n := range r.Notes {
		if ripLines[reg].of(n) {
			notes = append(notes, n)
		}
	}
	return notes
}

// fitPart is a line as the fit hears it: its notes and the voice it plays
// them with
type fitPart struct {
	notes []audio.RipNote
	voice *audio.VoiceDef
}

// fitScene is a recording as the fit hears it: its samples and tuning, the
// lines played in it, each with its voice, and the drums' hits with the sound
// each drum is heard on
type fitScene struct {
	x     []float64
	rate  int
	cents float64
	lines []fitPart
	hits  []audio.RipHit
	drums map[audio.InstrumentType][]float64 // each drum's take struck at full velocity
}

// ripScene is a rip's recording, its three lines and its hits, each line's voice
// the one the front end heard or, where it heard none, the one presentVoices
// gives it
func (s *Session) ripScene(x []float64, rate int, r *audio.Rip, o fitOptions) *fitScene {
	sc := s.scene(x, rate, r.Tuning, r.Hits())
	for reg, at := range lineVoices(r) {
		sc.lines = append(sc.lines, fitPart{notes: lineNotes(r, reg), voice: *at})
	}
	sc.presentVoices(o.notes)
	return sc
}

// scene is a recording at its tuning with its hits, each drum heard on the
// sound the transport's kit plays it with
func (s *Session) scene(x []float64, rate int, cents float64, hits []audio.RipHit) *fitScene {
	sc := &fitScene{x: x, rate: rate, cents: cents, hits: hits, drums: map[audio.InstrumentType][]float64{}}
	for _, d := range fitDrums {
		name := audio.InstrumentName(d)
		if sd := audio.SoundDefByName(cmp.Or(s.kit[name], name)); sd != nil {
			if take, err := audio.RenderHit(sd, audio.SoundShape{}, 1, 0); err == nil {
				sc.drums[d] = take
			}
		}
	}
	return sc
}

// fitDrums are the drums a rip hears hits of
var fitDrums = [...]audio.InstrumentType{audio.InstrKick, audio.InstrSnare, audio.InstrHihat}

// presentVoices gives each line that has no voice the registered voice nearest
// its own notes with the drums alone beside them, so each window hears every
// line that plays in it
func (sc *fitScene) presentVoices(n int) {
	alone := &fitScene{x: sc.x, rate: sc.rate, cents: sc.cents, hits: sc.hits, drums: sc.drums}
	for i, p := range sc.lines {
		if p.voice != nil || len(p.notes) == 0 {
			continue
		}
		alone.lines = []fitPart{{notes: p.notes}}
		targets := alone.targets(0, n)
		best := math.Inf(1)
		for _, v := range audio.RegisteredVoices() {
			if d := fitDistance(v, p.notes, targets, sc.cents); d < best {
				best, sc.lines[i].voice = d, v
			}
		}
	}
}

// fitTarget is one of a line's notes the fit listens to: the note, the window
// from its start to its end and fitTail, the recording there as mel frames and
// the mel energy of what else is heard playing there, each other line's notes
// with its voice and each drum's hits, all frames by RipMelBands
type fitTarget struct {
	start, end, to float64
	midi           int
	vel, dur       float64
	mel            []float64
	beside         [][]float64
}

// targets picks n of line's notes and hears the recording in each one's
// window, beside what every other line and drum plays there
func (sc *fitScene) targets(line, n int) []fitTarget {
	samples := fitSamples(sc.lines[line].notes, max(n, 1))
	targets := make([]fitTarget, len(samples))
	for i, note := range samples {
		lo := min(int(note.Start*float64(sc.rate)), len(sc.x))
		hi := min(int((note.End+fitTail)*float64(sc.rate)), len(sc.x))
		mel := audio.MelFrames(sc.x[lo:hi], sc.rate)
		t := fitTarget{start: note.Start, end: note.End, to: note.Start + float64(hi-lo)/float64(sc.rate), midi: note.MIDI, vel: note.Vel, dur: note.End - note.Start, mel: make([]float64, len(mel)*audio.RipMelBands)}
		for f, bands := range mel {
			for b, v := range bands {
				t.mel[f*audio.RipMelBands+b] = float64(v)
			}
		}
		heard := func(x []float64) {
			if slices.ContainsFunc(x, func(v float64) bool { return v != 0 }) {
				t.beside = append(t.beside, melEnergy(x, len(t.mel)))
			}
		}
		for j, p := range sc.lines {
			if j != line && p.voice != nil {
				heard(sounding(p.voice, p.notes, sc.cents, t.start, t.to))
			}
		}
		for _, d := range fitDrums {
			heard(sc.struck(d, t.start, t.to))
		}
		targets[i] = t
	}
	return targets
}

// struck is what drum d's hits sound from from to to seconds, each its take at
// its velocity
func (sc *fitScene) struck(d audio.InstrumentType, from, to float64) []float64 {
	const sr = audio.AudioSampleRate
	lo, take := int(from*sr), sc.drums[d]
	out := make([]float64, max(int(to*sr)-lo, 0))
	for _, h := range sc.hits {
		if h.Drum != d || h.Start >= to {
			continue
		}
		k := int(h.Start*sr) - lo
		for i := max(-k, 0); i < len(take) && k+i < len(out); i++ {
			out[k+i] += take[i] * h.Vel
		}
	}
	return out
}

// sounding is what notes sound with v at cents from from to to seconds, each
// rendered from its start; a note let go before from by more than v's release
// is silent there, and where a note ends past to is no part of it
func sounding(v *audio.VoiceDef, notes []audio.RipNote, cents, from, to float64) []float64 {
	const sr = audio.AudioSampleRate
	lo := int(from * sr)
	out := make([]float64, max(int(to*sr)-lo, 0))
	for _, n := range notes {
		if n.Start >= to || n.End+v.Release < from {
			continue
		}
		x := audio.RenderNoteAt(v, n.MIDI, cents, n.Vel, min(n.End, to)-n.Start, max(to-n.End, 0))
		k := int(n.Start*sr) - lo
		for i := max(-k, 0); i < len(x) && k+i < len(out); i++ {
			out[k+i] += x[i]
		}
	}
	return out
}

// melEnergy is the engine's render x as mel frames before MelFrames' logarithm,
// at x's own scale, laid out as cells values a frame's bands after another's:
// MelFrames(x) is log1p(100 e / p²) of these, p the loudest sample
func melEnergy(x []float64, cells int) []float64 {
	m := audio.NewMelStream()
	y := audio.Resample(x, audio.AudioSampleRate, audio.RipRate)
	out := make([]float64, cells)
	for f, bands := range append(m.Write(y), m.Close()...) {
		copy(out[min(f*audio.RipMelBands, cells):], bands)
	}
	return out
}

// fitDistance is how far the recording in the targets' windows is from v
// playing the line's notes there beside what else is heard: each window's
// mixDistance, the line and every other part at a gain of its own, the mean
// over the windows; the fit's measure, which a refinement holds a voice to
func fitDistance(v *audio.VoiceDef, notes []audio.RipNote, targets []fitTarget, cents float64) float64 {
	// a note sounds alike up to any window's end it is rendered to, so each is
	// rendered once, to the last window it sounds in
	until := make([]float64, len(notes))
	for _, t := range targets {
		for k, n := range notes {
			if n.Start < t.to && n.End+v.Release >= t.start {
				until[k] = max(until[k], t.to)
			}
		}
	}
	played := make([][]float64, len(notes))
	for k, n := range notes {
		if to := until[k]; to > 0 {
			played[k] = audio.RenderNoteAt(v, n.MIDI, cents, n.Vel, min(n.End, to)-n.Start, max(to-n.End, 0))
		}
	}
	const sr = audio.AudioSampleRate
	var sum float64
	for _, t := range targets {
		lo := int(t.start * sr)
		x := make([]float64, max(int(t.to*sr)-lo, 0))
		for k, n := range notes {
			if n.Start >= t.to || n.End+v.Release < t.start {
				continue
			}
			at := int(n.Start*sr) - lo
			for i := max(-at, 0); i < len(played[k]) && at+i < len(x); i++ {
				x[at+i] += played[k][i]
			}
		}
		sum += mixDistance(append([][]float64{melEnergy(x, len(t.mel))}, t.beside...), t.mel)
	}
	return sum / float64(max(len(targets), 1))
}

// mixDistance is how far y, a window's mel frames, stands from the nearest mix
// of the parts, each part's mel energy at a gain of its own: the mean square of
// log1p(100 Σ g c) less y over the window's frames and bands, as MelDistance
// counts, the gains found by Levenberg-Marquardt over their logs
func mixDistance(parts [][]float64, y []float64) float64 {
	k := len(parts)
	g := make([]float64, k)
	cost := func(theta []float64) float64 {
		for i := range k {
			g[i] = math.Exp(theta[i])
		}
		var sum float64
		for j, want := range y {
			var m float64
			for i, c := range parts {
				m += g[i] * c[j]
			}
			r := math.Log1p(100*m) - want
			sum += r * r
		}
		return sum / float64(max(len(y), 1))
	}
	var heard, have float64
	for j, want := range y {
		heard += math.Expm1(want)
		for _, c := range parts {
			have += c[j]
		}
	}
	theta := make([]float64, k)
	if heard <= 0 || have <= 0 {
		return cost(theta) // silence on either side: no gain brings them nearer
	}
	for i := range theta {
		theta[i] = math.Log(heard / 100 / have) // every part at the gain that matches the energy heard
	}
	now, lambda := cost(theta), 1e-3
	jtj, jtr, step, next := make([]float64, k*k), make([]float64, k), make([]float64, k), make([]float64, k)
	for range mixSteps {
		clear(jtj)
		clear(jtr)
		for i := range k {
			g[i] = math.Exp(theta[i])
		}
		for j, want := range y {
			var m float64
			for i, c := range parts {
				m += g[i] * c[j]
			}
			r, d := math.Log1p(100*m)-want, 100/(1+100*m)
			for i, c := range parts {
				ji := d * g[i] * c[j]
				jtr[i] += ji * r
				for l := i; l < k; l++ {
					jtj[i*k+l] += ji * d * g[l] * parts[l][j]
				}
			}
		}
		moved := false
		for range mixSteps {
			if dampedSolve(jtj, jtr, lambda, step) {
				for i := range k {
					next[i] = theta[i] - step[i]
				}
				if d := cost(next); d < now {
					moved = d < now*(1-1e-6)
					copy(theta, next)
					now, lambda = d, lambda/4
					break
				}
			}
			lambda *= 8
		}
		if !moved {
			break
		}
	}
	return now
}

// mixSteps bounds a gain fit's steps and each step's tries at its damping
const mixSteps = 30

// dampedSolve sets x to the solution of (A + lambda diag(A)) x = b by Cholesky,
// A symmetric and given by its upper triangle; false where that is not
// positive definite
func dampedSolve(a, b []float64, lambda float64, x []float64) bool {
	k := len(b)
	l := make([]float64, k*k)
	for i := range k {
		for j := 0; j <= i; j++ {
			sum := a[j*k+i]
			if i == j {
				sum += lambda*a[i*k+i] + 1e-12
			}
			for m := range j {
				sum -= l[i*k+m] * l[j*k+m]
			}
			switch {
			case i != j:
				l[i*k+j] = sum / l[j*k+j]
			case sum <= 0:
				return false
			default:
				l[i*k+i] = math.Sqrt(sum)
			}
		}
	}
	for i := range k {
		sum := b[i]
		for m := range i {
			sum -= l[i*k+m] * x[m]
		}
		x[i] = sum / l[i*k+i]
	}
	for i := k - 1; i >= 0; i-- {
		sum := x[i]
		for m := i + 1; m < k; m++ {
			sum -= l[m*k+i] * x[m]
		}
		x[i] = sum / l[i*k+i]
	}
	return true
}

// fitted is a fit's outcome: the voice, its level set, the seed the measure put
// nearest, and the distance of each on the notes listened to
type fitted struct {
	voice, seed *audio.VoiceDef
	to, from    float64
}

// fitVoice searches for the voice whose engine-rendered notes sound most like
// a line's notes in the recording: a genome of genes and wave, seeded with
// every registered voice and the one given, scored by fitDistance around each
// sample note, the other lines playing beside it with their voices; nil when
// the line has no note long enough.
func (s *Session) fitVoice(sc *fitScene, line int, prior *audio.VoiceDef, o fitOptions) *fitted {
	targets := sc.targets(line, o.notes)
	if len(targets) == 0 {
		return nil
	}
	x, rate, cents, notes := sc.x, sc.rate, sc.cents, sc.lines[line].notes
	genes := audio.VoiceGeneCount
	decode, encode := voiceFromGenome, voiceGenome
	eval := genetic.Memoize(genomeKey, func(g []float64) float64 {
		v := decode(g)
		if audio.ValidateVoice(v) != nil {
			return math.Inf(-1)
		}
		return -fitDistance(v, notes, targets, cents)
	})
	var seeds [][]float64
	if prior != nil {
		seeds = append(seeds, encode(prior))
	}
	for _, v := range audio.RegisteredVoices() {
		seeds = append(seeds, encode(v))
	}
	best := math.Inf(-1)
	var seed *audio.VoiceDef // decoded here, before the engine owns the seeds
	for _, g := range seeds {
		d := eval(g)
		if seed == nil || d > best {
			seed = decode(g)
		}
		best = max(best, d)
	}
	bounds := make([]genetic.ParameterBounds, genes+1)
	for i := range bounds {
		bounds[i] = genetic.ParameterBounds{Min: 0, Max: 1}
	}
	cfg := genetic.DefaultConfig()
	cfg.PoolSize, cfg.EliteCount, cfg.MaxIterations = max(o.pool, 4), max(o.pool/8, 1), o.generations
	cfg.Parallelism, cfg.Seed = cmp.Or(o.cores, runtime.NumCPU()), o.seed
	cfg.PerturbationRate, cfg.PerturbationStrength = 0.15, 0.1
	e := genetic.NewEngine[[]float64, float64](eval, func(rng *rand.Rand) []float64 {
		g := make([]float64, genes+1)
		for i := range g {
			g[i] = rng.Float64()
		}
		return g
	}, &genetic.TournamentSelector[[]float64, float64]{TournamentSize: 3},
		&genetic.UniformCombiner[[]float64, float64, float64]{MixProbability: 0.5},
		&genetic.BoundedPerturbator{Bounds: bounds, Boundary: genetic.BoundaryReflect}, cfg)
	e.SetSeeds(seeds)
	e.SetSchedule(genetic.AdaptiveStrength[float64](0.15, 0.1, 0.01, 0.4, 1.5))
	e.SetTerminator(func(p *genetic.Pool[[]float64, float64], gen int) bool {
		fmt.Fprintf(s.out, "generation %d: distance %.3f\n", gen, -p.Stats.BestScore)
		return false
	})
	pool, err := e.Run(context.Background())
	if err != nil || len(pool.Members) == 0 {
		return nil
	}
	// the measure fits each part's gain, so the level is the search's blind
	// spot: set it so the voice's notes carry the loudness the recording has
	// around them
	v := decode(pool.Members[0].Data)
	var want, have float64
	for _, t := range targets {
		lo, hi := min(int(t.start*float64(rate)), len(x)), min(int(t.end*float64(rate)), len(x))
		want += rms(x[lo:hi])
		have += rms(audio.RenderNoteAt(v, t.midi, cents, t.vel, t.dur, 0))
	}
	if have > 0 && want > 0 {
		v.Level = min(max(v.Level*want/have, 0.1), 4) // the level ValidateVoice allows
	}
	return &fitted{voice: v, seed: seed, to: -pool.Members[0].Score, from: -best}
}

func rms(x []float64) float64 {
	var e float64
	for _, v := range x {
		e += v * v
	}
	return math.Sqrt(e / float64(max(len(x), 1)))
}

// cmdGenes says each voice named as a label states one, a line of JSON a voice:
// its wave and its genes in VoiceDef.Genes' order
func cmdGenes(s *Session, a []string) error {
	for _, name := range a {
		v, ok := s.voices.get(name)
		if !ok {
			return fmt.Errorf("unknown voice %q (ls voice lists them)", name)
		}
		line, err := json.Marshal(struct {
			Voice string    `json:"voice"`
			Wave  string    `json:"wave"`
			Genes []float64 `json:"genes"`
		}{name, cmp.Or(v.Wave, "sine"), v.Genes()})
		if err != nil {
			return err
		}
		fmt.Fprintf(s.out, "%s\n", line)
	}
	return nil
}

// fitsManifest is a fits directory's fits.json: the sizes its fits ran at, the
// checksum of the weight files its rips heard through, or that it fitted the
// labels' own notes and ripped nothing; a directory holds one kind
type fitsManifest struct {
	Version     int    `json:"version"`
	Notes       int    `json:"notes"`
	Pool        int    `json:"pool"`
	Generations int    `json:"generations"`
	Seed        uint64 `json:"seed"`
	Weights     string `json:"weights"`
	Given       bool   `json:"given,omitempty"`
}

const fitsFile = "fits.json"

// fitRow is one line's fit read against its loop's label. A distance is the
// fit's measure of a voice on the notes the fit listened to, or held on the
// line's other notes; a voice's genes are its own less the label's, far their
// RMS. Null where the label, the front end or the line gives none.
type fitRow struct {
	Loop       string    `json:"loop"`
	Line       string    `json:"line"`
	Notes      int       `json:"notes"`
	Listened   int       `json:"listened"`
	Held       int       `json:"held"`
	Matched    int       `json:"matched"` // the line's notes a labelled note answers
	Voice      string    `json:"voice"`   // the label's voice that plays most of them
	Share      float64   `json:"share"`   // its share of the labelled notes they match
	Family     string    `json:"family"`
	Wave       string    `json:"wave"`
	Fitted     string    `json:"fitted"`
	Heard      *float64  `json:"heard"` // the voice the front end heard
	Seed       *float64  `json:"seed"`  // the best seed
	Fit        *float64  `json:"fit"`
	Own        *float64  `json:"own"` // the label's own voice
	HeardHeld  *float64  `json:"heard_held"`
	SeedHeld   *float64  `json:"seed_held"`
	FitHeld    *float64  `json:"fit_held"`
	OwnHeld    *float64  `json:"own_held"`
	HeardGenes []float64 `json:"heard_genes"`
	SeedGenes  []float64 `json:"seed_genes"`
	FitGenes   []float64 `json:"fit_genes"`
	HeardFar   *float64  `json:"heard_far"`
	SeedFar    *float64  `json:"seed_far"`
	FitFar     *float64  `json:"fit_far"`
}

func cmdFits(s *Session, a []string) error {
	usage := errors.New("want fits <dir> from <corpus dir> [given] [part k/n] [notes <n>] [pool <n>] [generations <n>] [seed <n>]")
	if a[1] != "from" {
		return usage
	}
	o, part, parts, given := s.fit, 0, 0, false
	for w := a[3:]; len(w) > 0; w = w[2:] {
		if w[0] == "given" {
			given, w = true, w[1:]
			if len(w) == 0 {
				break
			}
		}
		if len(w) < 2 {
			return usage
		}
		var err error
		if w[0] == "part" {
			part, parts, err = parsePart(w[1])
		} else {
			err = o.size(w[0], w[1])
		}
		if err != nil {
			return err
		}
	}
	return s.fits(s.path(a[0]), s.path(a[2]), o, part, parts, given)
}

// fits reads the voice fit over a corpus directory's loops that kept their
// audio: each is ripped and its lines fitted as rip does it, at o's sizes, or
// given, each voice its label states fitted on its own notes, and a row a line
// fitted (fitRows) goes to <dir>/<the loop's checksum>.json. A loop on disk is
// kept, parts taking the loops i%parts == part; the last lines count them all.
func (s *Session) fits(dir, from string, o fitOptions, part, parts int, given bool) error {
	if info, err := os.Stat(from); err != nil {
		return err
	} else if !info.IsDir() {
		return fmt.Errorf("%s: fits reads a corpus directory, whose labels state each note's voice", from)
	}
	var corpus corpusManifest
	if data, err := os.ReadFile(filepath.Join(from, "corpus.json")); err != nil {
		return err
	} else if err := json.Unmarshal(data, &corpus); err != nil {
		return fmt.Errorf("%s: %w", filepath.Join(from, "corpus.json"), err)
	}
	if corpus.VoiceGenes != audio.VoiceGeneCount {
		return fmt.Errorf("%s: its labels state %d genes a voice, and a voice has %d", from, corpus.VoiceGenes, audio.VoiceGeneCount)
	}
	loops, silent, err := truthSources(from)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	// written whole or not there, as parts started together each write a file
	whole := func(file string, v any) error {
		tmp := fmt.Sprintf("%s.%d.part", file, os.Getpid())
		if err := writeJSON(tmp, v); err != nil {
			return err
		}
		return os.Rename(tmp, file)
	}
	m := fitsManifest{Version: 2, Notes: o.notes, Pool: o.pool, Generations: o.generations, Seed: o.seed, Weights: s.earSum, Given: given}
	if given {
		m.Weights = "" // nothing is ripped
	}
	manifest := filepath.Join(dir, fitsFile)
	if data, err := os.ReadFile(manifest); err == nil {
		var was fitsManifest
		if json.Unmarshal(data, &was) != nil || was != m {
			return fmt.Errorf("%s holds %s, and these are %s: a directory holds one kind", dir, was, m)
		}
	} else if err := whole(manifest, m); err != nil {
		return err
	}
	said := s.out
	s.out = io.Discard // a search's generations are rip's and fit's to say
	defer func() { s.out = said }()
	var rows []fitRow
	var taken, kept int
	var failed []string
	for i, l := range loops {
		if parts > 0 && i%parts != part {
			continue
		}
		recording, err := os.ReadFile(l.at(l.wav))
		if err != nil {
			return err
		}
		sum := sha256.Sum256(recording)
		file := filepath.Join(dir, fmt.Sprintf("%x.json", sum[:8]))
		var loop []fitRow
		if data, err := os.ReadFile(file); err == nil {
			if err := json.Unmarshal(data, &loop); err != nil {
				return fmt.Errorf("%s: %w", file, err)
			}
			kept++
		} else {
			var lab corpusLoop
			if data, err := os.ReadFile(l.at(l.truth)); err != nil {
				return err
			} else if err := json.Unmarshal(data, &lab); err != nil {
				return fmt.Errorf("%s: %w", l.truth, err)
			}
			x, rate, err := audio.ReadWAV(bytes.NewReader(recording))
			if err != nil {
				return fmt.Errorf("%s: %w", l.wav, err)
			}
			var lines []fitLine
			var sc *fitScene
			if given {
				sc = s.scene(x, rate, lab.Tuning, lab.hits())
				for _, v := range labelVoices(lab, corpus) {
					lines, sc.lines = append(lines, fitLine{word: v.name, voice: v}), append(sc.lines, fitPart{notes: v.notes})
				}
				sc.presentVoices(o.notes)
			} else {
				r, err := s.ripper().Samples(x, rate)
				if err != nil {
					failed = append(failed, fmt.Sprintf("%s: %v", l.wav, err))
					continue
				}
				for reg, at := range lineVoices(r) {
					lines = append(lines, fitLine{word: ripLines[reg].word, heard: *at})
				}
				sc = s.ripScene(x, rate, r, o)
			}
			loop = s.fitRows(strings.TrimSuffix(l.wav, ".wav"), sc, lines, lab, corpus, o)
			if err := whole(file, loop); err != nil {
				return err
			}
			fmt.Fprintf(said, "%s: %d line(s) fitted\n", l.wav, len(loop))
		}
		taken, rows = taken+1, append(rows, loop...)
	}
	what := fmt.Sprintf("fits of %d loop(s) of %s in %s at %d note(s), %d generation(s) of %d, seed %d: %d line(s)", taken, from, dir, o.notes, o.generations, o.pool, o.seed, len(rows))
	if given {
		what += ", each a voice its label states, on its notes"
	}
	if kept > 0 {
		what += fmt.Sprintf("; %d loop(s) were there", kept)
	}
	if silent > 0 {
		what += fmt.Sprintf("; %d loop(s) kept no audio", silent)
	}
	if parts > 0 {
		what += fmt.Sprintf("; part %d of %d", part, parts)
	}
	fmt.Fprintln(said, what)
	if len(failed) > 0 {
		fmt.Fprintf(said, "no rip of %d, as %s\n", len(failed), failed[0])
	}
	fitCounts(said, rows, corpus.Waves)
	return nil
}

// String names the kind of fits m holds
func (m fitsManifest) String() string {
	of := fmt.Sprintf("through the weights %q", m.Weights)
	if m.Given {
		of = "on the labels' notes"
	}
	return fmt.Sprintf("fits of %d note(s), %d generation(s) of %d, seed %d, %s", m.Notes, m.Generations, m.Pool, m.Seed, of)
}

// labelVoice is a voice a label's notes state: its instrument's name, its wave
// and family by the manifest's names, its genes where stated, and its notes
type labelVoice struct {
	name, wave, family string
	genes              []float64
	notes              []audio.RipNote
}

// labelVoices are the voices a label's notes were played with, in the order
// the label first states each
func labelVoices(lab corpusLoop, m corpusManifest) []*labelVoice {
	named := func(names []string, i int) string {
		if i < 0 || i >= len(names) {
			return ""
		}
		return names[i]
	}
	var out []*labelVoice
	for _, e := range lab.Events {
		if e.Drum >= 0 || e.MIDI <= 0 {
			continue
		}
		wave, family := named(m.Waves, e.Wave), named(m.Families, e.Family)
		i := slices.IndexFunc(out, func(v *labelVoice) bool {
			return v.name == e.Instr && v.wave == wave && v.family == family && slices.Equal(v.genes, e.Voice)
		})
		if i < 0 {
			i, out = len(out), append(out, &labelVoice{name: e.Instr, wave: wave, family: family, genes: e.Voice})
		}
		out[i].notes = append(out[i].notes, audio.RipNote{Start: float64(e.Frame) * audio.RipFrameSec, End: float64(e.End) * audio.RipFrameSec, MIDI: e.MIDI, Vel: e.Vel})
	}
	return out
}

// lineOwner is whose line a rip's line is: the label's voice that plays most
// of its notes that match a labelled note on the loop's clock, that voice's
// share of the labelled notes they match, a note two voices double counted for
// each, and how many of the line's notes match any; nil where none does
func lineOwner(notes []audio.RipNote, voices []*labelVoice, given []audio.RipNote) (owner *labelVoice, share float64, matched int) {
	var most, all int
	for _, v := range voices {
		n := audio.MatchNotes(notes, v.notes, true).All.Matched
		if all += n; n > most {
			owner, most = v, n
		}
	}
	if owner == nil {
		return nil, 0, 0
	}
	return owner, float64(most) / float64(all), audio.MatchNotes(notes, given, true).All.Matched
}

// fitLine is a line of a scene fits fits: a rip's, with the voice its front
// end heard, or a voice its label states
type fitLine struct {
	word  string
	heard *audio.VoiceDef
	voice *labelVoice
}

// fitRows fits each of the scene's lines in turn as rip does, at o's sizes,
// and reads it against the loop's label: a row a line fitted
func (s *Session) fitRows(loop string, sc *fitScene, lines []fitLine, lab corpusLoop, m corpusManifest, o fitOptions) []fitRow {
	finite := func(d float64) *float64 {
		if math.IsNaN(d) || math.IsInf(d, 0) {
			return nil
		}
		return &d
	}
	// a voice's distance as the search scores a seed, through its genome
	near := func(v *audio.VoiceDef, notes []audio.RipNote, targets []fitTarget) *float64 {
		if v == nil || len(targets) == 0 {
			return nil
		}
		if v = voiceFromGenome(voiceGenome(v)); audio.ValidateVoice(v) != nil {
			return nil
		}
		return finite(fitDistance(v, notes, targets, sc.cents))
	}
	apart := func(v *audio.VoiceDef, label []float64) ([]float64, *float64) {
		if v == nil || label == nil {
			return nil, nil
		}
		genes, sum := v.Genes(), 0.0
		for i, g := range label {
			genes[i] -= g
			sum += genes[i] * genes[i]
		}
		return genes, finite(math.Sqrt(sum / float64(len(genes))))
	}
	voices, given := labelVoices(lab, m), lab.notes()
	rows := []fitRow{}
	for i, l := range lines {
		notes := sc.lines[i].notes
		f := s.fitVoice(sc, i, l.heard, o)
		if f == nil {
			continue
		}
		// the fit's own choice of notes, in its order: those it heard, then the rest
		targets := sc.targets(i, len(notes))
		sc.lines[i].voice = f.voice // the lines after hear this one as fitted
		listened, held := targets[:min(max(o.notes, 1), len(targets))], targets[min(max(o.notes, 1), len(targets)):]
		row := fitRow{Loop: loop, Line: l.word, Notes: len(notes), Listened: len(listened), Held: len(held), Fitted: cmp.Or(f.voice.Wave, "sine"),
			Heard: near(l.heard, notes, listened), Seed: finite(f.from), Fit: finite(f.to),
			HeardHeld: near(l.heard, notes, held), SeedHeld: near(f.seed, notes, held), FitHeld: near(f.voice, notes, held)}
		if owner, share, matched := lineOwner(notes, voices, given); owner != nil {
			if l.voice != nil { // a voice's own notes are its line, whatever doubles them
				owner = l.voice
			}
			row.Matched, row.Voice, row.Share, row.Family, row.Wave = matched, owner.name, share, owner.family, owner.wave
			if owner.genes != nil {
				own := audio.VoiceFromGenes(owner.wave, owner.genes)
				row.Own, row.OwnHeld = near(own, notes, listened), near(own, notes, held)
				row.HeardGenes, row.HeardFar = apart(l.heard, owner.genes)
				row.SeedGenes, row.SeedFar = apart(f.seed, owner.genes)
				row.FitGenes, row.FitFar = apart(f.voice, owner.genes)
			}
		}
		rows = append(rows, row)
	}
	return rows
}

// fitCounts prints the rows counted, the lines of one voice alone and the rest
// apart, each whole and by its label's wave: how often the fitted wave is the
// label's, how often the label's own voice and how often the fit is the nearer
// on the notes listened to and held out, each distance's median, and how often
// the fit ends nearer the label's genes than the voice heard and the best seed
func fitCounts(w io.Writer, rows []fitRow, waves []string) {
	of := func(k, n int) string {
		if n == 0 {
			return "-"
		}
		return fmt.Sprintf("%d/%d", k, n)
	}
	median := func(v []float64) string {
		if len(v) == 0 {
			return "-"
		}
		slices.Sort(v)
		return fmt.Sprintf("%.3f", (v[(len(v)-1)/2]+v[len(v)/2])/2)
	}
	unowned := 0
	for _, r := range rows {
		if r.Matched == 0 {
			unowned++
		}
	}
	fmt.Fprintf(w, "%d line(s) match no labelled note and are left out. right: the fitted wave is the label's; short: the label's own voice is nearer the recording than the fit, the search stopped short; prefers: the fit is nearer than the label's own voice, the measure prefers another; heard, seed, fit, own: the median distance of the voice the front end heard, the best seed, the fit and the label's own voice; genes nearer than heard, seed: the fit ends nearer the label's genes than the voice heard or the best seed began\n", unowned)
	fmt.Fprintf(w, "%-32s | %-47s | %-47s | %-19s\n", "", "on the notes the fit listened to", "on the line's notes held out", "genes nearer than")
	fmt.Fprintf(w, "%-9s %-6s %5s %9s | %9s %9s %6s %6s %6s %6s | %9s %9s %6s %6s %6s %6s | %9s %9s\n", "lines", "wave", "rows", "right", "short", "prefers", "heard", "seed", "fit", "own", "short", "prefers", "heard", "seed", "fit", "own", "heard", "seed")
	for _, alone := range [...]bool{true, false} {
		for _, wave := range append([]string{"all"}, append(slices.Clone(waves), "")...) {
			var n, right, waved int
			var short, prefers, owned, nearer, both [2]int
			var dist [2][4][]float64
			for _, r := range rows {
				if r.Matched == 0 || (r.Share == 1) != alone || (wave != "all" && r.Wave != wave) {
					continue
				}
				n++
				if r.Wave != "" {
					waved++
					if r.Fitted == r.Wave {
						right++
					}
				}
				for h, d := range [2][4]*float64{{r.Heard, r.Seed, r.Fit, r.Own}, {r.HeardHeld, r.SeedHeld, r.FitHeld, r.OwnHeld}} {
					for k, v := range d {
						if v != nil {
							dist[h][k] = append(dist[h][k], *v)
						}
					}
					if fit, own := d[2], d[3]; fit != nil && own != nil {
						owned[h]++
						switch {
						case *own < *fit:
							short[h]++
						case *fit < *own:
							prefers[h]++
						}
					}
				}
				for k, began := range [2]*float64{r.HeardFar, r.SeedFar} {
					if began != nil && r.FitFar != nil {
						if both[k]++; *r.FitFar < *began {
							nearer[k]++
						}
					}
				}
			}
			if n == 0 {
				continue
			}
			lines := "mixed"
			if alone {
				lines = "one voice"
			}
			fmt.Fprintf(w, "%-9s %-6s %5d %9s", lines, cmp.Or(wave, "none"), n, of(right, waved))
			for h := range dist {
				fmt.Fprintf(w, " | %9s %9s", of(short[h], owned[h]), of(prefers[h], owned[h]))
				for k := range dist[h] {
					fmt.Fprintf(w, " %6s", median(dist[h][k]))
				}
			}
			fmt.Fprintf(w, " | %9s %9s\n", of(nearer[0], both[0]), of(nearer[1], both[1]))
		}
	}
}
